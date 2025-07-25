import pandas as pd
import string
import re
import time
from collections import Counter
from collections import defaultdict
import json
from tqdm import tqdm
import os
from tqdm import tqdm
import numpy as np
import heapq

class Tokenizer:
    def __init__(self, token_to_id_path="src/tokenizer/token_to_id.json", merges_path="src/tokenizer/merges.json", training_text=None, num_merges=21680):
        self.allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " \n\t")

        if token_to_id_path is None or merges_path is None:
            print("Training tokenizer...")
            self.tok2id, self.merges = self.train_bpe(training_text, num_merges=num_merges)
            self.save_token_to_id_map(self.tok2id, "src/tokenizer/token_to_id.json")
            self.save_merges(self.merges, "src/tokenizer/merges.json")

        else:
            self.tok2id, self.merges = self.load_token_to_id_map(token_to_id_path), self.load_merges(merges_path)
            print(f"Tokenizer loaded with {len(self.tok2id)} tokens")

        self.id2tok = {v: k for k, v in self.tok2id.items()}
        

    def __len__(self):
        return len(self.tok2id)
    
    def filter_text(self, text):
        return ''.join(ch for ch in text if ch in self.allowed_chars)

    def get_most_common_pair(self, seq):
        counts = defaultdict(int)
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            counts[pair] += 1
        if not counts:
            return None
        return max(counts, key=counts.get)

    def update_seq(self, seq, pair):
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:

                new_seq.append(''.join(pair))
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    def train_bpe(self, text, num_merges):
        merges: list[tuple[str, str]] = []
        seq = list(text)
        mapping = {ch: i for i, ch in enumerate(self.allowed_chars)}
        next_id = len(mapping)

        for _ in tqdm(range(num_merges)):
            pair = self.get_most_common_pair(seq)
            if pair is None:
                break
            seq  = self.update_seq(seq, pair)
            merged = ''.join(pair)
            if merged not in mapping:
                mapping[merged] = next_id
                merges.append(pair)
                next_id += 1

        return mapping, merges

    def encode(self, text: str) -> np.ndarray:
        tokens = list(text)

        # Map merges to ranks for quick lookup
        merge_ranks = {tuple(pair): i for i, pair in enumerate(self.merges)}

        pairs = self._get_pairs(tokens)
        heap = [(merge_ranks[pair], pair) for pair in pairs if pair in merge_ranks]
        heapq.heapify(heap)

        # Keep merging until no valid pairs left
        while heap:
            rank, pair = heapq.heappop(heap)
            if pair not in pairs:
                continue  # stale pair, skip

            # Merge the pair
            tokens = self._merge_pair(tokens, pair)

            # Recompute pairs after merge
            pairs = self._get_pairs(tokens)

            # Rebuild the heap with only valid pairs
            heap = [(merge_ranks[p], p) for p in pairs if p in merge_ranks]
            heapq.heapify(heap)

        return np.array([self.tok2id[tok] for tok in tokens], dtype=np.uint16)

    def _get_pairs(self, tokens):
        return set(zip(tokens, tokens[1:]))

    def _merge_pair(self, tokens, pair):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged


    def decode(self, token_ids):
        out = ""
        for i in token_ids:
            out += self.id2tok[i]
        return out

    def save_token_to_id_map(self, token_to_id, path):
        with open(path, 'w') as f:
            json.dump(token_to_id, f)

    def save_merges(self, merges, path):
        with open(path, 'w') as f:
            json.dump([list(pair) for pair in merges], f)

    def load_token_to_id_map(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def load_merges(self, path):
        with open(path, 'r') as f:
            return json.load(f)

if __name__ == "__main__":
    text = "Hey gang asdfasdfhows it going?"
    tokenizer = Tokenizer(token_to_id_path=r"src\tokenizer\token_to_id.json", merges_path=r"src\tokenizer\merges.json", training_text=text, num_merges=21680)
    print(tokenizer.decode(tokenizer.encode(text)))