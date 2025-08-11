import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.module import Module, Linear, LayerNorm
from src.core.losses import CrossEntropyWithLogits
from src.core.optim import SGD, AdamW
from src.utils.lr_scheduler import LRScheduler
from src.core.tensor import Tensor
from src.tokenizer.tokenizer import tokenizer
from src.utils.backend import xp
import time
from typing import List
from src.tokenizer.tokenizer import Tokenizer
import pandas as pd
import numpy as np


NUM_HEADS = 4
# src = np.random.randint(low=0, high=16, size=(15, 15))
src = np.load("src/training/first_batch.npy")
x = src[:, :-1]
y = src[:, 1:]





class Net(Module):
    def __init__(self, d_model, n_heads, vocab_size, max_seq_len):
        super().__init__()

        self.e = self.embedding(vocab_size, d_model, max_seq_len, name="Embedding")

        self.head1 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head2 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head3 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head4 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head5 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head6 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head7 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head8 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.project = self.linear(d_model, vocab_size, name="project")
    
    def forward(self, idx):
        x = self.e.get_sentence_embedding(idx)
        x = self.head1(x)
        x = self.head2(x)
        x = self.head3(x)
        x = self.head4(x)
        x = self.head5(x)
        x = self.head6(x)
        x = self.head7(x)
        x = self.head8(x)
        x = self.project(x)
        return x

    def train(self, x, y, epochs, optimizer):
        for epoch in range(epochs):
            y_hat = self.forward(x)
            # print(y_hat.shape, y.shape)
            loss = CrossEntropyWithLogits(y_hat, y, axis=-1)
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {loss.data}")
                
if __name__ == "__main__":
    D_MODEL = 768
    VOCAB_SIZE = len(tokenizer.get_vocab())
    N_HEADS = 12
    MAX_SEQ_LEN = 512
    EXPECTED_OPTIM_STEPS = 20_000
    WARMUP_STEPS = 200
    MIN_LR = 1e-5
    MAX_LR = 5e-4
    FINAL_LR = 1e-6
    CHECKPOINT_INTERVAL_SECONDS = 3600

    scheduler = LRScheduler(
        warmup_steps=WARMUP_STEPS,
        total_steps=EXPECTED_OPTIM_STEPS,
        min_lr=MIN_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR
        )


    model = Net(d_model=D_MODEL, n_heads=N_HEADS, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN)
    model._build((15, 15))
    optimizer = AdamW(model.parameters(), lr=scheduler, precision=(xp.float16, xp.float32), clip_norm=1.0)


    model.train(x, y, epochs=1000, optimizer=optimizer)


    
        