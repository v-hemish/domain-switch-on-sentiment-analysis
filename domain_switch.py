# Import statements
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import pandas as pd
import torch.nn as nn

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parameters
config = {
    "num_labels": 2,
    "hidden_dropout_prob": 0.2,
    "hidden_size": 768,
    "max_length": 512,
}

training_params = {
    "batch_size": 2,
    "epochs": 50,
    "output_folder": "./models/",
    "output_file": "model.bin",
    "learning_rate": 0.5e-5,
    "print_after_steps": 5,
    "save_steps": 5000,
}

# Define Dataset and dataloaders
from transformers import BertTokenizer, BertModel
class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df
        # self.tokenizer = BertTokenizer.from_pretrained('bert-small')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-cased')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-small-cased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        review = self.df.iloc[index]["text"]
        sentiment = self.df.iloc[index]["sentiment"]
        sentiment_dict = {
            "pos": 0,
            "neg": 1,
        }
        label = sentiment_dict[sentiment]
        encoded_input = self.tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length= config["max_length"],
                pad_to_max_length=True,
                return_overflowing_tokens=True,
            )
        if "num_truncated_tokens" in encoded_input and encoded_input["num_truncated_tokens"] > 0:
            # print("Attention! you are cropping tokens")
            pass

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None
        token_type_ids = encoded_input["token_type_ids"] if "token_type_ids" in encoded_input else None

        data_input = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "label": torch.tensor(label),
        }

        return data_input["input_ids"], data_input["attention_mask"], data_input["token_type_ids"], data_input["label"]

    def __len__(self):
        return self.df.shape[0]

imdb_data = pd.read_csv("./data/imdb_train.tsv", sep='\t')
source_dataset = CustomDataset(imdb_data)
source_dataloader = DataLoader(dataset=source_dataset, batch_size=training_params["batch_size"], shuffle=True, num_workers=2)

amazon_data = pd.read_csv("./data/amazon_train.tsv", sep= "\t")
target_dataset = CustomDataset(amazon_data)
target_dataloader = DataLoader(dataset=target_dataset, batch_size=training_params["batch_size"], shuffle=True, num_workers=2)

# Gradient Reversal Function
from torch.autograd import Function

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class NoGradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Simply pass back the gradient during the backward pass without reversing it.
        return grad_output, None


class DynamicGradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, p, base=0.015, scale=1, sigma=0.1):
        # Compute lambda using a sigmoid function centered around 0.5 with additional base and scale adjustments
        ctx.lambda_ = base + scale * (2 / (1 + torch.exp(-sigma * (p - 0.5))) - 1)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Apply the computed lambda to the gradient, reversing it
        return grad_output.neg() * ctx.lambda_, None, None, None, None

# Defining Model
import torch
import torch.nn as nn
import torch.optim as optim

class AdaptationModel(nn.Module):
    def __init__(self):
        super(AdaptationModel, self).__init__()
        
        num_labels = config["num_labels"]
        # self.bert = BertModel.from_pretrained('bert-small')
        # self.bert = BertModel.from_pretrained('bert-cased')
        # self.bert = BertModel.from_pretrained('bert-small-cased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config["hidden_size"], num_labels),
            nn.LogSoftmax(dim=1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(config["hidden_size"], 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          labels=None,
          grl_lambda = 1.0, 
          ):

        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        reversed_pooled_output = GradientReversalFn.apply(pooled_output, grl_lambda)

        sentiment_pred = self.sentiment_classifier(pooled_output)
        domain_pred = self.domain_classifier(reversed_pooled_output)

        return sentiment_pred.to(device), domain_pred.to(device)

# Compute Accuracy
def calculate_accuracy(logits, labels):
    
    predicted_labels_dict = {
      0: 0,
      1: 0,
    }
    
    predicted_label = logits.max(dim = 1)[1]
    
    for pred in predicted_label:
        predicted_labels_dict[pred.item()] += 1
    acc = (predicted_label == labels).float().mean()
    
    return acc, predicted_labels_dict

# Evaluate the model
def evaluate(model, dataset = "imdb", percentage = 5):
    with torch.no_grad():
        predicted_labels_dict = {                                                   
          0: 0,                                                                     
          1: 0,                                                                     
        }
        
        dev_data = pd.read_csv("./data/" + dataset + "_dev.tsv", sep = "\t")
        data_size = dev_data.shape[0]
        selected_for_evaluation = int(data_size*percentage/100)
        dev_data = dev_data.head(selected_for_evaluation)
        dataset = CustomDataset(dev_data)

        dataloader = DataLoader(dataset = dataset, batch_size = training_params["batch_size"], shuffle = True, num_workers = 2)

        mean_accuracy = 0.0
        total_batches = len(dataloader)
        
        for input_ids, attention_mask, token_type_ids, labels in dataloader:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids" : token_type_ids,
                "labels": labels,
            }
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            sentiment_pred, _ = model(**inputs)
            accuracy, predicted_labels = calculate_accuracy(sentiment_pred, inputs["labels"])
            mean_accuracy += accuracy
            predicted_labels_dict[0] += predicted_labels[0]
            predicted_labels_dict[1] += predicted_labels[1]  
        print(predicted_labels_dict)
    return mean_accuracy/total_batches

# Training configuration
train_config = {
    "learning_rate": 0.5e-5,
    "epochs": 50,
    "output_folder": "./models/",
    "output_file": "model_epoch",
    "batch_size": 100,
    "print_interval": 5,
    "save_interval": 5000
}

# Load the model
model = DomainAdaptationModel()
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])

# Loss functions
loss_fn_sentiment = nn.NLLLoss()
loss_fn_domain = nn.NLLLoss()

# Calculate maximum batches
max_batches = min(len(source_dataloader), len(target_dataloader))

# Training loop
for epoch in range(train_config["epochs"]):
    source_iter = iter(source_dataloader)
    target_iter = iter(target_dataloader)

    for batch_idx in range(max_batches):
        p = float(batch_idx + epoch * max_batches) / (train_config["epochs"] * max_batches)
        grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
        grl_lambda = torch.tensor(grl_lambda, dtype=torch.float32, device=device)
        
        model.train()

        if batch_idx % train_config["print_interval"] == 0:
            print(f"Training Step: {batch_idx}")

        # Zero gradients
        optimizer.zero_grad()

        # Source dataset training update
        src_input_ids, src_attention_mask, src_token_type_ids, src_labels = next(source_iter)
        src_inputs = {
            "input_ids": src_input_ids.to(device),
            "attention_mask": src_attention_mask.to(device),
            "token_type_ids": src_token_type_ids.to(device),
            "labels": src_labels.to(device),
            "grl_lambda": grl_lambda
        }
        
        sentiment_pred, domain_pred = model(**src_inputs)
        loss_sentiment = loss_fn_sentiment(sentiment_pred, src_inputs["labels"])
        domain_labels_source = torch.zeros(train_config["batch_size"], dtype=torch.long, device=device)
        loss_domain_source = loss_fn_domain(domain_pred, domain_labels_source)

        # Target dataset training update 
        tgt_input_ids, tgt_attention_mask, tgt_token_type_ids, _ = next(target_iterator)  # No labels for domain
        tgt_inputs = {
            "input_ids": tgt_input_ids.to(device),
            "attention_mask": tgt_attention_mask.to(device),
            "token_type_ids": tgt_token_type_ids.to(device),
            "grl_lambda": grl_lambda
        }

        _, domain_pred_target = model(**tgt_inputs)
        domain_labels_target = torch.ones(train_config["batch_size"], dtype=torch.long, device=device)
        loss_domain_target = loss_fn_domain(domain_pred_target, domain_labels_target)

        # Combine losses and apply gradient update
        total_loss = loss_sentiment + loss_domain_source + loss_domain_target
        total_loss.backward()
        optimizer.step()


    # Save model checkpoint
    model_save_path = os.path.join(train_config["output_folder"], f"{train_config['output_file']}{epoch}.bin")
    torch.save(model.state_dict(), model_save_path)

    # Evaluate after each epoch
    accuracy_amazon = evaluate(model, dataset="amazon", percentage=5)
    accuracy_imdb = evaluate(model, dataset="imdb", percentage=5)
    print(f"Accuracy on Amazon after epoch {epoch}: {accuracy_amazon}")
    print(f"Accuracy on IMDb after epoch {epoch}: {accuracy_imdb}")

# Final evaluation on entire development set
accuracy_amazon_full = evaluate(model, dataset="amazon", percentage=100)
accuracy_imdb_full = evaluate(model, dataset="imdb", percentage=100)
print(f"Final Accuracy on full Amazon dataset: {accuracy_amazon_full}")
print(f"Final Accuracy on full IMDb dataset: {accuracy_imdb_full}")