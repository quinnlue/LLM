import sys
import os
from tqdm import tqdm
import pandas as pd
from src.tokenizer.tokenizer import Tokenizer


def tokenize(text, tokenizer):
    seq = tokenizer.encode(text)
    return seq


if __name__ == "__main__":
    train = pd.read_parquet("../../data/clean/train.parquet")
    test = pd.read_parquet("../../data/clean/test.parquet")
    validation = pd.read_parquet("../../data/clean/validation.parquet")
    tokenizer = Tokenizer(token_to_id_path=r"..\tokenizer\token_to_id.json", merges_path=r"..\tokenizer\merges.json")

    tokenized_validation = []
    for x in tqdm(validation['text'], desc="Tokenizing validation"):
        tokenized_validation.append(tokenize(x, tokenizer))
    validation['text'] = tokenized_validation
    validation.to_parquet("../../data/tokenized/validation.parquet")

    tokenized_test = []
    for x in tqdm(test['text'], desc="Tokenizing test"):
        tokenized_test.append(tokenize(x, tokenizer))
    test['text'] = tokenized_test
    test.to_parquet("../../data/tokenized/test.parquet")

    tokenized_train = []
    for x in tqdm(train['text'], desc="Tokenizing train"):
        tokenized_train.append(tokenize(x, tokenizer))
    train['text'] = tokenized_train
    train.to_parquet("../../data/tokenized/train.parquet")
