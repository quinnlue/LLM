import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tqdm import tqdm
import pandas as pd
from src.tokenizer.tokenizer import Tokenizer

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def tokenize(text, tokenizer):
    seq = tokenizer.encode(text)
    return seq

if __name__ == "__main__":
    # Use absolute paths based on project root
    train = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/clean/train.parquet"))
    test = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/clean/test.parquet"))
    validation = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/clean/validation.parquet"))
    tokenizer = Tokenizer(
        token_to_id_path=os.path.join(PROJECT_ROOT, "src/tokenizer/token_to_id.json"), 
        merges_path=os.path.join(PROJECT_ROOT, "src/tokenizer/merges.json")
    )

    # ... rest of the code remains the same ...
    tokenized_validation = []
    for x in tqdm(validation['text'], desc="Tokenizing validation"):
        tokenized_validation.append(tokenize(x, tokenizer))
    validation['text'] = tokenized_validation
    validation.to_parquet(os.path.join(PROJECT_ROOT, "data/tokenized/validation.parquet"))

    tokenized_test = []
    for x in tqdm(test['text'], desc="Tokenizing test"):
        tokenized_test.append(tokenize(x, tokenizer))
    test['text'] = tokenized_test
    test.to_parquet(os.path.join(PROJECT_ROOT, "data/tokenized/test.parquet"))

    tokenized_train = []
    for x in tqdm(train['text'], desc="Tokenizing train"):
        tokenized_train.append(tokenize(x, tokenizer))
    train['text'] = tokenized_train
    train.to_parquet(os.path.join(PROJECT_ROOT, "data/tokenized/train.parquet"))