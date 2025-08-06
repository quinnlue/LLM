import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.module import Module
from src.core.losses import CrossEntropy
from src.core.optim import AdamW
from src.core.tensor import Tensor
from src.preprocess.dataloader import DataLoader
from src.preprocess.dataloader import Dataset
from src.tokenizer.tokenizer import tokenizer
from src.utils.lr_scheduler import LRScheduler
from src.utils.backend import xp
from src.utils.logger import train_logger, val_logger

from tqdm import tqdm
import time
from datetime import datetime


# PATHS ------------------------------
TRAIN_DIR = r"data/train"
VAL_DIR = r"data/val"
TEST_DIR = r"data/test"

CHECKPOINT_DIR = r"checkpoints"

# MODEL HYPERPARAMETERS ------------------------------

VOCAB_SIZE = len(tokenizer.get_vocab())
D_MODEL = 768
N_HEADS = D_MODEL // 64
MAX_SEQ_LEN = 536 # CLIP TO 512 DURING INFERENCE
PAD_IDX = 0
EOS_IDX = 1
DEPTH = 12
EXPECTED_OPTIM_STEPS = 20000
MINI_BATCH_PER_STEP = 24
MAX_TOKENS_PER_MINI_BATCH = 48000



class Model(Module):
    def __init__(self, vocab_size, d_model, max_seq_len, pad_idx, n_heads, depth,checkpoint_interval_seconds: int = 3600):
        super().__init__()
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        self.best_val_loss = float("inf")
        self.e = self.embedding(vocab_size, d_model, max_seq_len, pad_idx)
        self.depth = depth
        self.heads = [self.transformer(d_model=d_model, n_heads=n_heads) for _ in range(depth)]


        self.project = self.linear(d_model, vocab_size, module_type="linear", layer_type="linear", name="project")
    
    def forward(self, idx):
        x, padding_mask = self.e.get_sentence_embedding(idx)
        for head in self.heads:
            x = head(x, padding_mask)
        x = self.project(x)
        return x
    
    def evaluate(self):
        dl = DataLoader(VAL_DIR, x_column="seq", is_binned=True, bin_column="bin", max_tokens=MAX_TOKENS_PER_MINI_BATCH)
        losses = []
        for batch in tqdm(dl, desc="Evaluating"):
            batch.requires_grad = False
            y_hat = self.forward(batch[:,:-1])
            loss = CrossEntropy(y_hat, batch[:,1:])
            losses.append(loss.item())
        return xp.mean(xp.array(losses))

    def checkpoint(self, optimizer):
        val_loss = self.evaluate()
        val_logger.info(f"Validation loss: {val_loss}")
        cp_path = os.path.join(CHECKPOINT_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.save_checkpoint(optimizer, cp_path)
        
    def train(self, epochs: int, dataloader: DataLoader, optimizer, mini_batch_per_step):
        last_cp_time = time.perf_counter()

        for epoch in range(epochs):
            for i, batch in enumerate(tqdm(dataloader, desc=f"Training epoch {epoch}")):
                y_hat = self.forward(batch[:,:-1])
                loss = CrossEntropy(y_hat, batch[:,1:])
                loss.backward()
                if i % mini_batch_per_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                train_logger.info(f"Training loss: {loss.item()}")

                # checkpointing & validation
                if time.perf_counter() - last_cp_time > self.checkpoint_interval_seconds:
                    self.checkpoint(optimizer, CHECKPOINT_DIR)
                    last_cp_time = time.perf_counter()

if __name__ == "__main__":
    scheduler = LRScheduler(
        warmup_steps=EXPECTED_OPTIM_STEPS // 100 * 3,
        total_steps=EXPECTED_OPTIM_STEPS,
        min_lr=1e-5,
        max_lr=3e-4,
        final_lr=1e-6,
        batch_per_step=MINI_BATCH_PER_STEP
    )
    model = Model(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, PAD_IDX, N_HEADS, DEPTH)
    optimizer = AdamW(model.parameters(), lr=scheduler, precision=(xp.float16, xp.float32))

    dl = DataLoader(TRAIN_DIR, x_column="seq", is_binned=True, bin_column="bin", max_tokens=MAX_TOKENS_PER_MINI_BATCH)

    # data set is about 7b tokens
    model.train(epochs=2, dataloader=dl, optimizer=optimizer, mini_batch_per_step=MINI_BATCH_PER_STEP)

