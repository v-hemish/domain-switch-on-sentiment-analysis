# Import necessary libraries
import json
import numpy as np
import pandas as pd
import pprint
from transformers import pipeline
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize the feature extraction pipeline
feature_extractor = pipeline('feature-extraction')

# Load the first 1000 reviews from the Amazon dataset
amazon_reviews = pd.read_csv("./imdb_50k/amazon_dev.tsv", sep="\t").head(1000)

# Initialize list to store features
amazon_review_features = []

# Constants for processing
TOTAL_REVIEWS = 1000
PRINT_INTERVAL = 100

# Process each review in the dataset
for index, row in amazon_reviews.iterrows():
    review_text = row["Review"]
    formatted_input = feature_extractor.tokenizer.cls_token + review_text 
    if index >= TOTAL_REVIEWS:
        break
    if (index + 1) % PRINT_INTERVAL == 0:
        print(f"Processed count: {index + 1}")
    try:
        review_features = feature_extractor(formatted_input)
        review_features_array = np.array(review_features)
        features = list(review_features_array[0][0])
        amazon_review_features.append(features)
    except Exception as e:
        print(e)

# Load the first 1000 reviews from the IMDB dataset
imdb_reviews = pd.read_csv("./imdb_50k/imdb_dev.tsv", sep="\t").head(1000)

# Initialize list to store features
imdb_review_features = []

# Process each review in the dataset
for index, row in imdb_reviews.iterrows():
    review_text = row["text"]
    formatted_input = feature_extractor.tokenizer.cls_token + review_text 
    if index >= TOTAL_REVIEWS:
        break
    if (index + 1) % PRINT_INTERVAL == 0:
        print(f"Processed count: {index + 1}")
    try:
        review_features = feature_extractor(formatted_input)
        review_features_array = np.array(review_features)
        features = list(review_features_array[0][0])
        imdb_review_features.append(features)
    except Exception as e:
        print(e)

# Output the number of features extracted
print(len(amazon_review_features), len(imdb_review_features))

# Combine features and perform t-SNE dimensionality reduction for 3D visualization
all_features = amazon_review_features + imdb_review_features
tsne_results_3d = TSNE(n_components=3).fit_transform(all_features)

# Assign labels for the scatter plot
labels = [0] * len(amazon_review_features) + [1] * len(imdb_review_features)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(tsne_results_3d[:, 0], tsne_results_3d[:, 1], tsne_results_3d[:, 2], c=labels, cmap='viridis')

# Add legend and labels
legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)
ax.set_title('3D t-SNE Visualization')
plt.show()
