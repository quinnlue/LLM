import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tqdm import tqdm
import pandas as pd
# from src.tokenizer.tokenizer import Tokenizer
import time
from multiprocessing import Pool
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

tokenizer = None

def init_worker():
    global tokenizer
    tokenizer = Tokenizer.from_file("src/tokenizer/tokenizer.json")
    # tokenizer = Tokenizer(
    #     token_to_id_path=os.path.join(PROJECT_ROOT, "src/tokenizer/token_to_id.json"), 
    #     merges_path=os.path.join(PROJECT_ROOT, "src/tokenizer/merges.json")
    # )

def tokenize(text):
    global tokenizer
    return tokenizer.encode(text).ids

if __name__ == "__main__":
    data_folder = os.path.join(PROJECT_ROOT, "data/raw")
    tgt_folder = os.path.join(PROJECT_ROOT, "data/tokenized")

    # Process each parquet file
    for path in os.listdir(data_folder):
        if path.endswith(".parquet"):
            src_path = os.path.join(data_folder, path)
            tgt_path = os.path.join(tgt_folder, path)
            df = pd.read_parquet(src_path)

            texts = df['text'].tolist()

            start_time = time.time()

            with Pool(processes=os.cpu_count(), initializer=init_worker) as pool:
                # tokenize texts in parallel
                tokenized_texts = list(tqdm(pool.imap(tokenize, texts), total=len(texts)))

            end_time = time.time()

            # Replace 'text' column with tokenized output
            df['seq'] = tokenized_texts
            df[['seq']].to_parquet(tgt_path)
            df[['text']].to_parquet(src_path)

            total_tokens = sum(len(x) for x in tokenized_texts)
            print(f"File: {path}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            print(f"Total tokens: {total_tokens}")
            print(f"Tokens per second: {total_tokens / (end_time - start_time):.2f}")

