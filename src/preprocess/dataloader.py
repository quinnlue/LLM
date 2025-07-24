import sys
import os
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from src.core.tensor import Tensor
import numpy as np

class Dataset:
    def __init__(self, path, src_column, tgt_column):
        self.path = path
        self.src_column = src_column
        self.tgt_column = tgt_column
        self.df = self.load_data()

        self.data = self.df[self.src_column].values
        self.labels = self.df[self.tgt_column].values

    def load_data(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")
        
        if self.path.endswith(".parquet"):
            return pd.read_parquet(self.path)
        elif self.path.endswith(".csv"):
            return pd.read_csv(self.path)
        else:
            raise ValueError("Invalid file extension (accepted: .parquet, .csv)")

class Batch:
    def __init__(self, src: Tensor, tgt: Tensor):
        self.src = src
        self.tgt = tgt

    @property
    def size(self):
        return self.src.shape[0]

    @property
    def seq_len(self):
        return self.src.shape[1] if len(self.src.shape) > 1 else 1

    def __repr__(self):
        return f"Batch(size={self.size}, seq_len={self.seq_len})"

class DataLoader:
    def __init__(self, dataset: Dataset, is_binned=False, bin_column=None):
        self.dataset = dataset
        self.is_binned = is_binned
        self.bin_column = bin_column

        if self.is_binned and self.bin_column is None:
            raise ValueError("bin_column must be provided if is_binned is True")
            
        if self.bin_column and not self.is_binned:
             raise ValueError("is bin_column is provided, is_binned must be True")
        
    def __iter__(self):
        for batch in self.get_batches():
            yield batch

    def get_batches(self, shuffle: bool = True, max_tokens: int = 2000):
        if not self.is_binned:
            raise NotImplementedError("Only is_binned=True is supported")

        if max_tokens is None:
            raise ValueError("max_tokens must be provided")

        df = self.dataset.df.sample(frac=1).reset_index(drop=True) if shuffle else self.dataset.df

        batches = []
        for token_len, group in df.groupby(self.bin_column):
            examples_per_batch = max(1, max_tokens // int(token_len))

            for start in range(0, len(group), examples_per_batch):
                end = start + examples_per_batch
                batch_df = group.iloc[start:end]

                # Parse string representations of lists using ast.literal_eval
                src_seqs = batch_df[self.dataset.src_column].apply(ast.literal_eval).tolist()
                tgt_seqs = batch_df[self.dataset.tgt_column].apply(ast.literal_eval).tolist()

                src_tensor = Tensor(np.array(src_seqs, dtype=np.uint16), requires_grad=False)
                tgt_tensor = Tensor(np.array(tgt_seqs, dtype=np.uint16), requires_grad=False)

                batches.append(Batch(src_tensor, tgt_tensor))
        return batches

if __name__ == "__main__":
    dataset = Dataset(r"src/preprocess/dummy.csv", src_column="src", tgt_column="tgt")
    loader = DataLoader(dataset, is_binned=True, bin_column="bin_len")

    for idx, batch in enumerate(loader):
        print(f"Batch {idx}: {batch}, total tokens={batch.size * batch.seq_len}")

