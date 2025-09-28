import os

from tqdm import tqdm
import pandas as pd
# from src.tokenizer.tokenizer import Tokenizer
import time
from multiprocessing import Pool
from tokenizers import Tokenizer

tokenizer = None

def init_worker():
    global tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")

def tokenize(text):
    global tokenizer
    return tokenizer.encode(text).ids

if __name__ == "__main__":
    data_folder = os.path.join("data/extracted")
    tgt_folder = os.path.join("data/tokenized")
    os.makedirs(tgt_folder, exist_ok=True)

    residual_tokens = []

    for path in tqdm(sorted(os.listdir(data_folder)), desc="Processing files"):
        if path.endswith(".parquet"):
            src_path = os.path.join(data_folder, path) 
            tgt_path = os.path.join(tgt_folder, path)
            base_name = os.path.splitext(path)[0]
            df = pd.read_parquet(src_path, columns=['text'])

            texts = df['text'].tolist()

            start_time = time.time()

            CHUNK_SIZE = 513
            BATCH_ROWS = 256000
            total_chunks = 0
            part_idx = 0

            if os.path.exists(tgt_path):
                os.remove(tgt_path)

            chunk_batch = []
            with Pool(processes=12, initializer=init_worker) as pool:
                for seq in tqdm(pool.imap(tokenize, texts), total=len(texts)):
                    combined_tokens = residual_tokens + seq

                    num_full_chunks = len(combined_tokens) // CHUNK_SIZE
                    if num_full_chunks > 0:
                        for i in range(num_full_chunks):
                            start = i * CHUNK_SIZE
                            end = start + CHUNK_SIZE
                            chunk_batch.append(combined_tokens[start:end])
                        total_chunks += num_full_chunks

                    residual_tokens = combined_tokens[num_full_chunks * CHUNK_SIZE:]

                    if len(chunk_batch) >= BATCH_ROWS:
                        output_path = os.path.join(
                            tgt_folder,
                            f"{base_name}_part{part_idx}.parquet"
                        )
                        pd.DataFrame({'seq': chunk_batch}).to_parquet(
                            output_path,
                            engine='pyarrow',
                            compression='snappy',
                            index=False
                        )
                        part_idx += 1
                        chunk_batch = []

            if chunk_batch:
                output_path = os.path.join(
                    tgt_folder,
                    f"{base_name}_part{part_idx}.parquet"
                )
                pd.DataFrame({'seq': chunk_batch}).to_parquet(
                    output_path,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )

            end_time = time.time()

            total_tokens = total_chunks * CHUNK_SIZE
            print(f"File: {path}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            print(f"Total {CHUNK_SIZE}-token chunks: {total_chunks}")
            print(f"Tokens per second: {total_tokens / (end_time - start_time):.2f}")
            if residual_tokens:
                print(f"Carrying over {len(residual_tokens)} residual tokens to next file")

