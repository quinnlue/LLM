import pandas as pd
import sys
import os 
from tqdm import tqdm
import tracemalloc
import gc

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
    def _log_gpu_mem(tag: str):
        if hasattr(xp, "get_default_memory_pool"):
            used = xp.get_default_memory_pool().used_bytes() / 1024**2
            print(f"    [GPU] {tag:<18}: {used:8.2f} MB")
    def train(self, x, y, epochs, lr):
        optimizer = AdamW(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            print(f"\n=== Epoch {epoch} ===")

            # ---- set up memory tracking --------------------------------
            tracemalloc.start()
            if hasattr(xp, "get_default_memory_pool"):
                xp.get_default_memory_pool().free_all_blocks()

            self._log_gpu_mem("start")

            # ---- forward ----------------------------------------------
            y_hat = self.forward(x)
            self._log_gpu_mem("after forward")

            # ---- loss --------------------------------------------------
            loss = CrossEntropy(y_hat, y, axis=-1)
            self._log_gpu_mem("after loss")

            # ---- backward ---------------------------------------------
            loss.backward()
            self._log_gpu_mem("after backward")

            # ---- optimiser --------------------------------------------
            optimizer.step()
            self._log_gpu_mem("after opt.step")

            optimizer.zero_grad()
            self._log_gpu_mem("after zero_grad")

            # ---- CPU mem & cleanup ------------------------------------
            current_cpu, peak_cpu = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            gc.collect()  # force Python GC
            if hasattr(xp, "get_default_memory_pool"):
                xp.get_default_memory_pool().free_all_blocks()
            self._log_gpu_mem("after gc")

            # ---- summary ----------------------------------------------
            print(f"    Loss                 : {loss.data}")
            print(f"    CPU  current / peak  : {current_cpu / 1024**2:.2f} MB | "
                  f"{peak_cpu / 1024**2:.2f} MB")

             

def create_dummy_data(seq_len, batch_size):
    x = xp.random.randint(0, VOCAB_SIZE, (batch_size, seq_len)).astype(xp.int32)
    y = Tensor(x.copy())
    return x, y


model = Test(D_MODEL, N_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, PAD_IDX)

BATCH_SIZE = 8
x, y = create_dummy_data(MAX_SEQ_LEN, BATCH_SIZE)

model.train(x, y, epochs=100, lr=0.001)
