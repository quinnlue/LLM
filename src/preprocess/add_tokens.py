import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tqdm import tqdm
import pandas as pd
from src.tokenizer.tokenizer import Tokenizer
import swifter
tqdm.pandas()
import time

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def tokenize(text, tokenizer):
    seq = tokenizer.encode(text)
    return seq

if __name__ == "__main__":
    # text = "Your input text here"
    # tokenizer = Tokenizer()
    # print(tokenizer.encode(text))

    # Use absolute paths based on project root
    # train = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/clean/train.parquet"))
    # test = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/clean/test.parquet"))
    validation = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/clean/validation.parquet"))
    tokenizer = Tokenizer(
        token_to_id_path=os.path.join(PROJECT_ROOT, "src/tokenizer/token_to_id.json"), 
        merges_path=os.path.join(PROJECT_ROOT, "src/tokenizer/merges.json")
    )

    start_time = time.time()
    validation.loc[:99, 'seq'] = validation.loc[:99, 'text'].apply(lambda x: tokenize(x, tokenizer))
    end_time = time.time()

    total_tokens = validation['seq'].dropna().apply(len).sum()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per second: {total_tokens / (end_time - start_time)}")

    # tokenized_test = []
    # for x in tqdm(test['text'], desc="Tokenizing test"):
    #     tokenized_test.append(tokenize(x, tokenizer))
    # test['text'] = tokenized_test
    # test.to_parquet(os.path.join(PROJECT_ROOT, "data/tokenized/test.parquet"))

    # tokenized_train = []
    # for x in tqdm(train['text'], desc="Tokenizing train"):
    #     tokenized_train.append(tokenize(x, tokenizer))
    # train['text'] = tokenized_train
    # train.to_parquet(os.path.join(PROJECT_ROOT, "data/tokenized/train.parquet"))