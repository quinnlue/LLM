import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.module import Module, Linear, LayerNorm
from src.core.losses import CrossEntropy, BCE
from src.core.optim import Standard, AdamW
from src.core.tensor import Tensor
from src.utils.backend import xp
import time
from typing import List
from src.tokenizer.tokenizer import Tokenizer
import pandas as pd
import numpy as np


NUM_HEADS = 4
src = np.random.randint(low=1, high=16, size=(128, 16))
x = src[:, :-1]
y = src[:, 1:]

x_mine = Tensor(x, requires_grad=False)
y_mine = Tensor(y, requires_grad=False)


class Net(Module):
    def __init__(self, d_model, n_heads, vocab_size, max_seq_len, pad_idx=0):
        super().__init__()
        self.e = self.embedding(vocab_size, d_model, max_seq_len, pad_idx, name="Embedding")

        self.heads = [self.transformer(d_model=d_model, n_heads=n_heads) for _ in range(NUM_HEADS)]
        self.project = self.linear(d_model, vocab_size, name="project")
    
    def forward(self, idx):
        x, padding_mask = self.e.get_sentence_embedding(idx)
        x = Tensor(x.data, requires_grad=False)
        for head in self.heads:
            x = head(x, padding_mask)
        x = self.project(x)
        return x

    def train(self, x, y, epochs, optimizer):
        for epoch in range(epochs):
            y_hat = self.forward(x)
            # print(y_hat.shape, y.shape)
            loss = CrossEntropy(y_hat, y, axis=-1)
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {loss.data}")
                
if __name__ == "__main__":
    D_MODEL = 48
    VOCAB_SIZE = 20
    N_HEADS = 4
    MAX_SEQ_LEN = 32
    PAD_IDX = 0

    model = Net(d_model=D_MODEL, n_heads=N_HEADS, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, pad_idx=PAD_IDX)
    model._build((128, 15))
    optimizer = AdamW(model.parameters(), lr=0.001, precision=(xp.float32, xp.float32))


    model.train(x_mine, y_mine, epochs=1000, optimizer=optimizer)


    
        