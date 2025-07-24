from module import Module, Linear, LayerNorm
from loss import CrossEntropy, BCE
from optim import Standard
from tensor import Tensor
import numpy as np
from transformer import Transformer, Embedding
import time
from typing import List
from tokenizer import Tokenizer
class Test(Module):
    def __init__(self, d_model, n_heads, vocab_size, max_seq_len, pad_idx=0):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, max_seq_len, pad_idx)
        self.add_module('embedding', self.embedding)

        self.head1 = Transformer(d_model=d_model, n_heads=n_heads)
        self.head2 = Transformer(d_model=d_model, n_heads=n_heads)
        self.head3 = Transformer(d_model=d_model, n_heads=n_heads)
        self.head4 = Transformer(d_model=d_model, n_heads=n_heads)
        self.head5 = Transformer(d_model=d_model, n_heads=n_heads)
        self.head6 = Transformer(d_model=d_model, n_heads=n_heads)
        self.head7 = Transformer(d_model=d_model, n_heads=n_heads)
        self.head8 = Transformer(d_model=d_model, n_heads=n_heads)
        self.project = Linear(d_model, vocab_size)

        self.add_module('head1', self.head1)
        self.add_module('head2', self.head2)
        self.add_module('head3', self.head3)
        self.add_module('head4', self.head4)
        self.add_module('head5', self.head5)
        self.add_module('head6', self.head6)
        self.add_module('head7', self.head7)
        self.add_module('head8', self.head8)
        self.add_module('project', self.project)

        print(self.num_parameters)
    
    def get_sentence_embedding(self, idx):
        return self.embedding.get_sentence_embedding(idx)

    def forward(self, idx):
        x, padding_mask = self.get_sentence_embedding(idx)
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
        optimizer = Standard(self.parameters(), lr=lr)
        for epoch in range(epochs):
            y_hat = self.forward(x)
            loss = CrossEntropy(y_hat, y, axis=-1)
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.data}")
                


if __name__ == "__main__":
    tok = Tokenizer()

    D_MODEL = 512
    VOCAB_SIZE = len(tok)
    N_HEADS = 8
    MAX_SEQ_LEN = 1024
    PAD_IDX = 0
    model = Test(d_model=D_MODEL, n_heads=N_HEADS, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, pad_idx=PAD_IDX)
    text = "This is especially noticeable if you're on CPU or don't have big enough matrix acceleration (like CuBLAS/GEMM on GPU)."
    x = np.array([tok.encode(text)])[:,:-1]
    y = Tensor(np.array([tok.encode(text)]), requires_grad=True)[:,1:]

    # model.train(x, y, epochs=10000, lr=0.01)
        