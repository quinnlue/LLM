import os, pandas as pd, numpy as np, multiprocessing as mp
from tqdm import tqdm

EOS_IDX, PAD_IDX = 1, 0
TOKS_PER_BIN     = 16             # keep all “16” constants in one place
N_WORKERS        = mp.cpu_count() # 24 on your machine
CHUNK_SIZE       = 512            # rows handed to a worker at once

# --- pure-Python work each worker still has to do -----------------------------

def _pad_chunk(chunk: list[tuple[list[int], int]]) -> list[list[int]]:
    """Pad a *chunk* of rows – much cheaper than one row at a time."""
    out = []
    for seq, bin_idx in chunk:
        max_len = TOKS_PER_BIN * bin_idx
        new_len = len(seq) + 1                       # + EOS
        padded  = np.full(max_len, PAD_IDX, dtype=np.int16)
        padded[:new_len] = (*seq, EOS_IDX)
        out.append(padded.tolist())
    return out

# -----------------------------------------------------------------------------


def _iter_chunks(rows, size=CHUNK_SIZE):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def process_file(path: str):
    df = pd.read_parquet(path)

    # vectorised length & bin calculation – no Python loop
    lens_no_eos = df.seq.str.len()
    df          = df[lens_no_eos <= 528]

    # +1 to account for the EOS token we will append later
    seq_lens_with_eos = lens_no_eos.add(1)

    df["bin"] = (
        seq_lens_with_eos.add(TOKS_PER_BIN - 1) // TOKS_PER_BIN
    ).astype(np.int16)

    rows = list(zip(df.seq, df["bin"]))

    with mp.Pool(N_WORKERS) as pool:
        padded_chunks = tqdm(
            pool.imap_unordered(_pad_chunk, _iter_chunks(rows)),
            total = (len(rows) + CHUNK_SIZE - 1) // CHUNK_SIZE,
            desc  = f"Padding {os.path.basename(path)}"
        )
        # flatten the list-of-lists
        df["seq"] = [seq for chunk in padded_chunks for seq in chunk]

    df.to_parquet(path)


if __name__ == "__main__":                     # REQUIRED on Windows
    parquet_files = [
        os.path.join("data", d, f)
        for d in os.listdir("data")
        for f in os.listdir(os.path.join("data", d))
        if f.endswith(".parquet")
    ]
    for fp in parquet_files:
        process_file(fp)