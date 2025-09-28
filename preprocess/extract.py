import os
import tarfile
import lzma
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

SOURCE_DIR = "data/raw"
OUTPUT_DIR = "data/extracted"
N_WORKERS = 12
MAX_CHUNK_SIZE_BYTES = 12 * 1024**3  # 12GB

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_xz(member_info):
    tar_path, member_name = member_info
    texts = []
    total_bytes = 0

    try:
        with tarfile.open(tar_path, 'r') as tar:
            member = tar.getmember(member_name)
            xz_f = tar.extractfile(member)
            if xz_f is None:
                return []

            with lzma.open(xz_f) as xz_stream:
                inner_tar = tarfile.open(fileobj=xz_stream)
                for inner_member in inner_tar.getmembers():
                    if inner_member.isfile() and inner_member.name.endswith(".txt"):
                        txt_f = inner_tar.extractfile(inner_member)
                        if txt_f is None:
                            continue
                        content = txt_f.read().decode('utf-8', errors='ignore').strip()
                        texts.append(content)
                        total_bytes += len(content.encode('utf-8'))

    except Exception as e:
        print(f"Error processing {member_name} in {tar_path}: {e}")

    return texts

def process_tar(tar_path):
    tar_name = Path(tar_path).stem
    texts = []
    total_bytes = 0
    parquet_files = []

    with tarfile.open(tar_path, 'r') as tar:
        xz_members = [m.name for m in tar.getmembers() if m.name.endswith(".xz")]

    # Parallel process the xz files inside this tar
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_xz, (tar_path, xz_name)): xz_name for xz_name in xz_members}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {tar_name}", position=0):
            chunk_texts = future.result()
            for text in chunk_texts:
                texts.append(text)
                total_bytes += len(text.encode('utf-8'))
                if total_bytes >= MAX_CHUNK_SIZE_BYTES:
                    df = pd.DataFrame({'text': texts})
                    chunk_file = os.path.join(OUTPUT_DIR, f"{tar_name}_chunk{len(parquet_files)}.parquet")
                    df.to_parquet(chunk_file, index=False)
                    parquet_files.append(chunk_file)
                    texts = []
                    total_bytes = 0

    # Save leftover
    if texts:
        df = pd.DataFrame({'text': texts})
        chunk_file = os.path.join(OUTPUT_DIR, f"{tar_name}_chunk{len(parquet_files)}.parquet")
        df.to_parquet(chunk_file, index=False)
        parquet_files.append(chunk_file)

    return f"{tar_name}: saved {len(parquet_files)} parquet chunks"

if __name__ == "__main__":
    tar_files = sorted([
        os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR)
        if f.endswith(".tar")
    ])

    # Process tar files sequentially
    for tar_path in tqdm(tar_files):
        print(process_tar(tar_path))