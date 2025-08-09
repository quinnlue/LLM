import os
import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
# from flash_attn.flash_attention import FlashAttention  # Make sure flash_attn is installed
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.tokenizer.tokenizer import tokenizer
from src.preprocess.dataloader import DataLoader
from src.utils.logger import train_logger, val_logger

# --- Constants ---
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
CHECKPOINT_DIR = "checkpoints"

VOCAB_SIZE = len(tokenizer.get_vocab())
D_MODEL = 768
N_HEADS = D_MODEL // 64
MAX_SEQ_LEN = 548
PAD_IDX = 0
EOS_IDX = 1
WARMUP_STEPS = 1600
TOTAL_STEPS = 86400
DEPTH = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Model Components ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(device)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class VanillaAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        residual = x
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1).bool()
        attn_out, _ = self.mha(x, x, x,
                          attn_mask=causal_mask,
                          key_padding_mask=padding_mask)
        x = self.norm1(residual + attn_out)

        residual = x
        x = self.ff(x)
        x = self.norm2(residual + x)

        return x


class Model(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, pad_idx, n_heads):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([VanillaAttentionLayer(d_model, n_heads) for _ in range(DEPTH)])
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        # idx: (B, S)
        padding_mask = idx == self.pad_idx  # (B, S)
        x = self.embed(idx)  # (B, S, D)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        logits = self.project(x)  # (B, S, vocab_size)
        return logits

    def evaluate(self, val_loader, loss_fn):
        self.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = torch.tensor(batch.data).to_device(device).long()
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                logits = self.forward(inputs)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                losses.append(loss.item())
        return sum(losses) / len(losses)

    def checkpoint(self, optimizer, val_loader, loss_fn):
        val_loss = self.evaluate(val_loader, loss_fn)
        print(f"[{datetime.now()}] Validation loss: {val_loss:.4f}")
        cp_path = os.path.join(CHECKPOINT_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pt")
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, cp_path)

    def train_model(self, epochs, train_loader, val_loader, optimizer, scheduler=None, loss_fn=None):
        loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        last_cp_time = time.perf_counter()
        self.to(device)
        losses = []
        for epoch in range(epochs):

            self.train()
            for i, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
                batch = torch.tensor(batch.data).to(device).long()
                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                optimizer.zero_grad()
                logits = self.forward(inputs)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                losses.append(loss.item())
                if i % 1 == 0:

                    print(f"Training loss: {np.array(losses).mean():.4f}")
                    losses = []

                # checkpointing & validation
                if time.perf_counter() - last_cp_time > 3600:  # 1 hour
                    self.checkpoint(optimizer, val_loader, loss_fn)
                    self.train()            # <-- switch back
                    last_cp_time = time.perf_counter()


# --- DataLoader / Dataset ---
# Assume you have PyTorch Dataset and DataLoader implementations for your data already


if __name__ == "__main__":
    # You need to create train_dataset and val_dataset as torch.utils.data.Dataset instances
    train_dl = DataLoader(TRAIN_DIR, x_column="seq", is_binned=True, bin_column="bin", max_tokens=32000)
    val_dl = DataLoader(VAL_DIR, x_column="seq", is_binned=True, bin_column="bin", max_tokens=32000)

    model = Model(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, PAD_IDX, N_HEADS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        progress = float(current_step - WARMUP_STEPS) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay to 0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train_model(epochs=3, train_loader=train_dl, val_loader=val_dl, optimizer=optimizer, scheduler=scheduler)
