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
    data_folder = os.path.join(PROJECT_ROOT, "data/owt")

    # Process each parquet file
    for path in os.listdir(data_folder):
        if path.endswith(".parquet"):
            file_path = os.path.join(data_folder, path)
            df = pd.read_parquet(file_path)

            texts = df['text'].tolist()

            start_time = time.time()

            with Pool(processes=os.cpu_count(), initializer=init_worker) as pool:
                # tokenize texts in parallel
                tokenized_texts = list(tqdm(pool.imap(tokenize, texts), total=len(texts)))

            end_time = time.time()

            # Replace 'text' column with tokenized output
            df['text'] = tokenized_texts
            df.to_parquet(file_path)

            total_tokens = sum(len(x) for x in tokenized_texts)
            print(f"File: {path}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            print(f"Total tokens: {total_tokens}")
            print(f"Tokens per second: {total_tokens / (end_time - start_time):.2f}")


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