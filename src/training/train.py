import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.module import Module
from src.core.losses import CrossEntropy
from src.core.optim import AdamW
from src.core.tensor import Tensor
from src.preprocess.dataloader import DataLoader
from src.preprocess.dataloader import Dataset
from src.tokenizer.tokenizer import Tokenizer
from src.utils.lr_scheduler import LRScheduler

from tqdm import tqdm
import time
from datetime import datetime


# PATHS ------------------------------
TRAIN_DIR = r"data/train"
VAL_DIR = r"data/val"
TEST_DIR = r"data/test"

CHECKPOINT_DIR = r"checkpoints"

# MODEL HYPERPARAMETERS ------------------------------

VOCAB_SIZE = len(Tokenizer().tok2id)
D_MODEL = 1024
N_HEADS = D_MODEL // 64
MAX_SEQ_LEN = 2096 # CLIP TO 2048 DURING INFERENCE
PAD_IDX = 0
EOS_IDX = 1
EXPECTED_STEPS = 86400


class Model(Module):
    def __init__(self, vocab_size, d_model, max_seq_len, pad_idx, n_heads, checkpoint_interval_seconds: int = 3600):
        super().__init__()
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        self.best_val_loss = float("inf")
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
    
    def evaluate(self):
        pass

    def train(self, epochs: int, dataloader: DataLoader, optimizer):
        last_cp_time = time.perf_counter()

        for _ in range(epochs):
            for batch in tqdm(dataloader, desc="Training"):
                y_hat = self.forward(batch.src)
                loss = CrossEntropy(y_hat, batch.tgt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                # checkpointing & validation
                if time.perf_counter() - last_cp_time > self.checkpoint_interval_seconds:
                    val_loss = self.evaluate()
                    cp_path = os.path.join(CHECKPOINT_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                    self.save_checkpoint(optimizer, cp_path)
                    last_cp_time = time.perf_counter()



if __name__ == "__main__":
    optimizer = AdamW()
    lr_scheduler = LRScheduler(max_steps=EXPECTED_STEPS, warmup_steps=100, base_lr=3e-4, min_lr=1e-5, max_lr=1e-3)
    model = Model(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, PAD_IDX, N_HEADS)


    model.train(epochs=3, dataloader=DataLoader(TRAIN_DIR, x_column="x", is_binned=True, bin_column="bin"), optimizer=optimizer)

