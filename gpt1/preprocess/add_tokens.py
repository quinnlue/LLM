import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tqdm import tqdm
import pandas as pd
# from src.tokenizer.tokenizer import Tokenizer
import time
from multiprocessing import Pool
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

tokenizer = None

def init_worker():
    global tokenizer
    tokenizer = Tokenizer.from_file("src/tokenizer/tokenizer.json")



def tokenize(text):
    global tokenizer
    return tokenizer.encode(text).ids

if __name__ == "__main__":
    data_folder = os.path.join(PROJECT_ROOT, "data/extracted")
    tgt_folder = os.path.join(PROJECT_ROOT, "data/tokenized")

    # Process each parquet file
    for path in tqdm(os.listdir(data_folder), desc="Processing files"):
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

            # Ensure we start writing to a fresh file each run
            if os.path.exists(tgt_path):
                os.remove(tgt_path)

            chunk_batch = []
            with Pool(processes=12, initializer=init_worker) as pool:
                for seq in tqdm(pool.imap(tokenize, texts), total=len(texts)):
                    full_chunks = len(seq) // CHUNK_SIZE
                    total_chunks += full_chunks
                    for i in range(full_chunks):
                        chunk_batch.append(seq[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE])

                    # Write and reset batch when it gets large
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

            # Write any remaining chunks
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

