import pandas as pd
import sys
import os 
from tqdm import tqdm
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import dlx as dlx
import dlx.nn as nn
from dlx import Tensor, CrossEntropyWithLogits, AdamW, xp
from torch.optim.lr_scheduler import LRScheduler

D_MODEL = 64
N_HEADS = D_MODEL // 16
VOCAB_SIZE = 20
MAX_SEQ_LEN = 16
PAD_IDX = 0
EOS_IDX = 1

class Test(nn.Module):
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
    
    def set_pbar(self, pbar, epochs, mini_batch_idx, running_loss, grad_norm, loss_history):
        
        avg_10k = sum(loss_history[-10000:]) / 10000
        avg_1k = sum(loss_history[-1000:]) / 1000
        avg_100 = sum(loss_history[-100:]) / 100

    def train(self, x, y, epochs, optimizer):
        loss_history = []
        pbar = tqdm(range(epochs), desc=f"Training: {epochs} epochs | prev_loss: {prev_loss}")

        for epoch in pbar:
            y_hat = self.forward(x)
            loss = CrossEntropyWithLogits(y_hat, y, axis=-1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            prev_loss = loss.data
            pbar.set_description(f"Training: {epochs} epochs | prev_loss: {prev_loss:.4f}")

             

def create_dummy_data(seq_len, batch_size):
    src = Tensor(xp.random.randint(0, VOCAB_SIZE, (batch_size, seq_len)).astype(xp.int32))
    x = src[:,:-1]
    y = src[:,1:]
    
    return x, y

# EXPECTED_OPTIM_STEPS = 10000
# MINI_BATCH_PER_STEP = 1



# model = Test(D_MODEL, N_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, PAD_IDX)

BATCH_SIZE = 128
x, y = create_dummy_data(MAX_SEQ_LEN, BATCH_SIZE)



# optimizer = AdamW(model.parameters(), lr=0.001, precision=(xp.float16, xp.float32))
# print('asdfasdf')
# model.train(x, y, epochs=100, optimizer=optimizer)

if __name__ == "__main__":


    model = Test(d_model=D_MODEL, n_heads=N_HEADS, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, pad_idx=PAD_IDX)
    model._build((128, 15))
    optimizer = AdamW(model.parameters(), lr=0.001, precision=(xp.float32, xp.float32))


    model.train(x, y, epochs=1000, optimizer=optimizer)
