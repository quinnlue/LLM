import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tqdm import tqdm
import pandas as pd
from src.tokenizer.tokenizer import Tokenizer
import time
from multiprocessing import Pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Global tokenizer variable for workers
tokenizer = None

def init_worker():
    global tokenizer
    tokenizer = Tokenizer(
        token_to_id_path=os.path.join(PROJECT_ROOT, "src/tokenizer/token_to_id.json"), 
        merges_path=os.path.join(PROJECT_ROOT, "src/tokenizer/merges.json")
    )

def tokenize(text):
    global tokenizer
    return tokenizer.encode(text)

if __name__ == "__main__":
    validation = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/clean/validation.parquet"))

    start_time = time.time()

    texts = validation['text'].loc[:99].tolist()
    processes = os.cpu_count()
    with Pool(processes=os.cpu_count(), initializer=init_worker) as pool:
        out = list(tqdm(pool.imap(tokenize, texts), total=len(texts)))

    end_time = time.time()

    total_tokens = sum(len(x) for x in out)
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per second: {total_tokens / (end_time - start_time)}")

    # text = "Your input text here"
    # tokenizer = Tokenizer()
    # print(tokenizer.encode(text))

    # Use absolute paths based on project root
    # train = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/clean/train.parquet"))
    # test = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/clean/test.parquet"))


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