import pandas as pd
import sys
import os 
from tqdm import tqdm
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.tensor import Tensor
from src.utils.backend import xp
from src.core.module import Module
from src.core.losses import CrossEntropy
from src.core.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler


D_MODEL = 16
N_HEADS = D_MODEL // 16
VOCAB_SIZE = 20
MAX_SEQ_LEN = 16
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
    
    def train(self, x, y, epochs, optimizer):
        for epoch in tqdm(range(epochs)):
            y_hat = self.forward(x)

            loss = CrossEntropy(y_hat, y, axis=-1)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()


            gc.collect()  # force Python GC

            print(f"Loss: {loss.data}")

             

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
