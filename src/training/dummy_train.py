import pandas as pd
import sys
import os 
from tqdm import tqdm
import tracemalloc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.tensor import Tensor
from src.utils.backend import xp
from src.core.module import Module
from src.core.losses import CrossEntropy
from src.core.optim import AdamW


D_MODEL = 1024
N_HEADS = 4
VOCAB_SIZE = 21680
MAX_SEQ_LEN = 1024
PAD_IDX = 0
EOS_IDX = 1

class Test(Module):
    def __init__(self, d_model, n_heads, vocab_size, max_seq_len, pad_idx=0):
        super().__init__()
        self.e = self.embedding(vocab_size, d_model, max_seq_len, pad_idx)

        self.head1 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head2 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head3 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head4 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head5 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head6 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head7 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head8 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.project = self.linear(d_model, vocab_size, module_type="linear", layer_type="linear", name="project")
    
    def forward(self, idx):
        x, padding_mask = self.e.get_sentence_embedding(idx)
        x = self.head1(x, padding_mask)
        x = self.head2(x, padding_mask)
        x = self.head3(x, padding_mask)
        x = self.head4(x, padding_mask)
        x = self.head5(x, padding_mask)
        x = self.head6(x, padding_mask)
        x = self.head7(x, padding_mask)
        x = self.head8(x, padding_mask)
        x = self.project(x)
        return x

    def train(self, x, y, epochs, lr):
        optimizer = AdamW(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            # --- Start CPU memory tracking ---
            tracemalloc.start()
            # --- Start GPU memory tracking ---
            if hasattr(xp, "get_default_memory_pool"):
                mem_pool = xp.get_default_memory_pool()
                mem_pool.free_all_blocks()
                before_gpu = mem_pool.used_bytes()
            else:
                before_gpu = 0

            # Forward + Backward
            y_hat = self.forward(x)
            loss = CrossEntropy(y_hat, y, axis=-1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # --- Stop memory tracking ---
            current_cpu, peak_cpu = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if hasattr(xp, "get_default_memory_pool"):
                after_gpu = mem_pool.used_bytes()
                used_gpu = after_gpu - before_gpu
            else:
                used_gpu = 0

            # --- Print epoch info ---
            print(f"[Epoch {epoch}]")
            print(f"  Loss           : {loss.data}")
            print(f"  CPU Memory     : Current = {current_cpu / 1024**2:.2f} MB | Peak = {peak_cpu / 1024**2:.2f} MB")
            print(f"  GPU Memory Used: {used_gpu / 1024**2:.2f} MB\n")

             

def create_dummy_data(seq_len, batch_size):
    x = xp.random.randint(0, VOCAB_SIZE, (batch_size, seq_len)).astype(xp.int32)
    y = Tensor(x.copy())
    return x, y


model = Test(D_MODEL, N_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, PAD_IDX)

BATCH_SIZE = 8
x, y = create_dummy_data(MAX_SEQ_LEN, BATCH_SIZE)

model.train(x, y, epochs=100, lr=0.001)
