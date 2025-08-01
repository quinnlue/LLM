import sys
import os
import ast
import random
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from src.core.tensor import Tensor
import numpy as np

class Dataset:
    def __init__(self, path, x_column, shuffle_rows: bool = True):
        self.x_column = x_column
        self.path = path
        self.df = self.load_data()
        if shuffle_rows:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.x = self.df[self.x_column].values

    def load_data(self):
        return pd.read_parquet(self.path)


class DataLoader:
    def __init__(self, path, x_column, is_binned=False, bin_column=None, max_tokens=1024):
        self.path = path
        self.is_binned = is_binned
        self.bin_column = bin_column
        self.x_column = x_column
        self.max_tokens = max_tokens
        if self.is_binned and self.bin_column is None:
            raise ValueError("bin_column must be provided if is_binned is True")
            
        if self.bin_column and not self.is_binned:
             raise ValueError("is bin_column is provided, is_binned must be True")
        
    def __iter__(self):
        for batch in self.get_batches():
            yield batch

    def _get_data(self, shuffle_files: bool = True, shuffle_rows: bool = True):
        if os.path.isdir(self.path):
            print(f"Loading .parquet files from dataset directory: {self.path}")
            files = os.listdir(self.path)
            if shuffle_files:
                random.shuffle(files)
            else:
                files = sorted(files)
            
            for f in files:
                file_path = os.path.join(self.path, f)
                if file_path.endswith(".parquet"):
                    file_path = os.path.join(self.path, f)
                    yield Dataset(file_path, self.x_column, shuffle_rows=shuffle_rows)
                else:
                    warnings.warn(f"Skipping {file_path} because it is not a .parquet file")

        else:
            print(f"Loading dataset from file: {self.path}")
            yield Dataset(self.path, self.x_column)
            
    def get_batches(self, shuffle: bool = True):
        if not self.is_binned:
            raise NotImplementedError("Only is_binned=True is supported")

        for dataset in self._get_data(shuffle_files=shuffle, shuffle_rows=shuffle):
            # These are stored in RAM
            batches = []
            for _, group in dataset.df.groupby(self.bin_column):
                examples_per_batch = max(1, self.max_tokens // len(group[self.x_column].values[0])) # we can index 0 because this will raise downstream if the lengths of the batches are different

                for start in range(0, len(group), examples_per_batch):
                    if len(group) - start < examples_per_batch:
                        break

                    batches.append(Tensor(
                        np.stack(group.iloc[start:start+examples_per_batch][dataset.x_column].values), 
                        requires_grad=False
                    ))

            if shuffle:
                random.shuffle(batches)
            
            for batch in batches:
                yield batch




