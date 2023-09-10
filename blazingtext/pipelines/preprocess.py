import argparse
import logging
import os
import pathlib
import requests
import tempfile

import re
import boto3
import json
import numpy as np
import pandas as pd

"""
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

train_ratio = 0.8
eval_ratio = 0.1
test_ratio = 0.1


reg = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


def preprocess(text, stem=False):
    text = re.sub(reg, ' ', str(text).lower()).strip()
    return text

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/sentiment-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)
    
    data = pd.read_csv(fn)
    
    data['text'] = data['text'].apply(lambda x: preprocess(x))
    data['review'] = data['review'].astype(int)
    
    logger.info("Applying Transformers...")

    data = data.rename(columns={'text': 'source', 'review': 'label'})
    
    logger.info("Splitting into train, test, evaluate datasets... ")
    
    train_data, test_eval_data = train_test_split(data, test_size=1 - train_ratio)
    eval_data, test_data = train_test_split(test_eval_data, test_size=test_ratio/(test_ratio + eval_ratio))
    
    logger.info("Writing out datasets to %s.", base_dir)
    
    train_data.to_json(f"{base_dir}/train/train.jsonl", orient='records', lines=True)
    eval_data.to_json(f"{base_dir}/validation/eval.jsonl", orient='records', lines=True)
    test_data.to_json(f"{base_dir}/test/test.jsonl", orient='records', lines=True)