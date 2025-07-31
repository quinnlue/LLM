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

    def train(self, x, y, optimizer, epochs):
        for epoch in range(epochs):
            y_hat = self.forward(x)
            loss = CrossEntropy(y_hat, y, axis=-1)
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.data}")
                
if __name__ == "__main__":
    tok = Tokenizer(token_to_id_path="src/tokenizer/token_to_id.json", merges_path="src/tokenizer/merges.json")

    D_MODEL = 2
    VOCAB_SIZE = len(tok)
    N_HEADS = 1
    MAX_SEQ_LEN = 1024
    PAD_IDX = 0
    model = Test(d_model=D_MODEL, n_heads=N_HEADS, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, pad_idx=PAD_IDX)
    text = "This asdf"
    x = xp.array([tok.encode(text)])[:,:-1]
    y = Tensor(xp.array([tok.encode(text)]), requires_grad=True)[:,1:]

    optimizer = AdamW(model.parameters(), lr=0.01)

    model.train(x, y, optimizer, epochs=5)

    print(optimizer.params)

    model.save_model(optimizer, "../../checkpoints/transformer_implementation")

    print("-" * 100)


    new_model = Test(d_model=D_MODEL, n_heads=N_HEADS, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, pad_idx=PAD_IDX)
    new_model.load_model(optimizer, "../../checkpoints/transformer_implementation")
    print(optimizer.params)
        