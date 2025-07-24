import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import tqdm
import pandas as pd
from src.tokenizer.tokenizer import Tokenizer


def tokenize(text, tokenizer):
    seq = tokenizer.encode(text)
    return seq


if __name__ == "__main__":
    tqdm.pandas()

    train = pd.read_parquet("../../data/clean/train.parquet")
    test = pd.read_parquet("../../data/clean/test.parquet")
    validation = pd.read_parquet("../../data/clean/validation.parquet")
    tokenizer = Tokenizer(token_to_id_path=r"..\tokenizer\token_to_id.json", merges_path=r"..\tokenizer\merges.json")

    validation['text'] = validation['text'].progress_apply(lambda x: tokenize(x, tokenizer))
    validation.to_parquet("../../data/tokenized/validation.parquet")

    test['text'] = test['text'].progress_apply(lambda x: tokenize(x, tokenizer))
    test.to_parquet("../../data/tokenized/test.parquet")

    train['text'] = train['text'].progress_apply(lambda x: tokenize(x, tokenizer))
    train.to_parquet("../../data/tokenized/train.parquet")




