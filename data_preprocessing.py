# Import required libraries
import json
import pprint
import numpy as np
import pandas as pd

# Function to remove tabs and new lines from text
def clean_text(text):
    return text.replace("\t", " ").replace("\n", ". ")

# Function to preprocess IMDB dataset
def preprocess_imdb_data(dataset_type):
    imdb_data_path = "../Imdb/imdb-50k-movie-reviews-test-your-bert/"
    imdb_file = dataset_type + ".csv"
    imdb_dataframe = pd.read_csv(imdb_data_path + imdb_file)
    imdb_dataframe["text"] = imdb_dataframe["text"].apply(clean_text)
    return imdb_dataframe

# Preprocess train and test datasets
imdb_train = preprocess_imdb_data("train")
imdb_test = preprocess_imdb_data("test")

# Save processed data
imdb_train.to_csv("./data/imdb_train.tsv", sep="\t", index=False)
imdb_test.to_csv("./data/imdb_dev.tsv", sep="\t", index=False)

# Function to determine sentiment based on score
def assign_sentiment(score):
    return "pos" if score > 3 else "neg"

# Function to preprocess Amazon reviews
def preprocess_amazon_reviews(dataset_type):
    amazon_data_folder = "../amazon-reviews/electronics/"
    amazon_file = dataset_type + ".tsv"
    amazon_dataframe = pd.read_csv(amazon_data_folder + amazon_file, sep="\t")
    amazon_filtered = amazon_dataframe[amazon_dataframe["Score"] != 3]
    amazon_filtered = amazon_filtered[amazon_filtered['Review'].apply(lambda x: isinstance(x, str))]
    amazon_filtered["sentiment"] = amazon_filtered["Score"].apply(assign_sentiment)
    del amazon_filtered["Score"]
    amazon_filtered["Review"] = amazon_filtered["Review"].apply(clean_text)
    amazon_filtered.columns = ['text', 'sentiment']
    return amazon_filtered

# Preprocess Amazon train and dev datasets
amazon_train = preprocess_amazon_reviews("train")
amazon_dev = preprocess_amazon_reviews("dev")

# Save processed Amazon reviews
amazon_train.to_csv("./data/amazon_train.tsv", sep="\t", index=False)
amazon_dev.to_csv("./data/amazon_dev.tsv", sep="\t", index=False)
