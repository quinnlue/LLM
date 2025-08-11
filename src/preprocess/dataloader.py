import pandas as pd
import os
from src.utils.backend import xp
from src.core.tensor import Tensor
import numpy as np
import random

is_cuda = xp.__name__ == "cupy"
pin_memory = is_cuda

class Dataset:
    def __init__(self, path, src, batch_size, shuffle_rows: bool = True):
        df = pd.read_parquet(path)

        if shuffle_rows:
            df = df.sample(frac=1).reset_index(drop=True)

        self.src = np.stack(df[src].values)
        self.x = self.src[:, :-1]
        self.y = self.src[:, 1:]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.src) // self.batch_size
    
    def __iter__(self):
        for i in range(0, len(self.src), self.batch_size):
            yield Tensor(self.x[i:i+self.batch_size]), Tensor(self.y[i:i+self.batch_size])


class DataLoader:
    def __init__(self, src_dir, src_column, batch_size, shuffle_rows: bool = True, shuffle_files: bool = True):
        self.batch_size = batch_size
        self.shuffle_rows = shuffle_rows
        self.files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".parquet")]
        if shuffle_files:
            random.shuffle(self.files)

        self.src_column = src_column

    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        for f in self.files:
            dataset = Dataset(f, self.src_column, self.batch_size, self.shuffle_rows)
            for batch in dataset:
                yield batch
