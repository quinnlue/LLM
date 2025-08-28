import sys
sys.path.append(r"gpt1")

import os
from pathlib import Path
import time
from datetime import datetime
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


from gpt1.tokenizer.tokenizer import tokenizer
from gpt1.torch_train.model import Model
from dlx.utils.logger import train_logger, val_logger
from gpt1.training.utils import RunningLossTracker
from gpt1.torch_train.utils import load_latest_checkpoint, resize_token_embeddings

from gpt1.torch_train.train import (
    VOCAB_SIZE,
    D_MODEL,
    MAX_SEQ_LEN,
    PAD_IDX,
    N_HEADS,
    DEPTH,
    CHECKPOINT_DIR,
)

class SFTDataset(Dataset):
    def __init__(self, data_path: str, max_seq_len: int, data_column: str, mask_column: str):
        self.data = pd.read_parquet(data_path)
        self.data_column = data_column
        self.mask_column = mask_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[self.data_column].iloc[idx]
        mask = self.data[self.mask_column].iloc[idx]

        x = src[:,:-1]
        y = src[:,1:]

        x = torch.tensor(x, dtype=torch.long).requires_grad_(False)
        y = torch.tensor(y, dtype=torch.long).requires_grad_(False)
        mask = torch.tensor(mask, dtype=torch.bool).requires_grad_(False)

        return x, y, mask
        

R = 8
ALPHA = R
BATCH_SIZE = 128

EPOCHS = 3
_BASE_DIR = Path(__file__).resolve().parents[1]

# PATHS ------------------------------
TRAIN_DIR = _BASE_DIR / "data" / "finetune" / "train.parquet"
VAL_DIR = _BASE_DIR / "data" / "finetune" / "validation.parquet"
CHECKPOINT_DIR = _BASE_DIR / "checkpoints"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise ValueError("CPU is not supported for fine-tuning")

    model = Model(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
        n_heads=N_HEADS,
        transformer_depth=DEPTH,
        checkpoint_dir=CHECKPOINT_DIR,
        lora=True,
        lora_r=R,
        lora_alpha=ALPHA,
    ).to(device)
    print("initialized model")


    train_dataset = SFTDataset(TRAIN_DIR, MAX_SEQ_LEN, "tokens", "mask")
    val_dataset = SFTDataset(VAL_DIR, MAX_SEQ_LEN, "tokens", "mask")
    print("initialized datasets")
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, pin_memory=True)
    print("initialized dataloaders")
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name:
            lora_params.append(param)
        else:
            param.requires_grad = False
    print("initialized lora params")
    optimizer = optim.AdamW(lora_params, lr=1e-4)
    print("initialized optimizer")
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    print("initialized scheduler")
    scaler = GradScaler()
    print("initialized scaler")

    load_latest_checkpoint(model, optimizer, scheduler, scaler, device, CHECKPOINT_DIR)
    print("loaded latest checkpoint")
    exit()

    for epoch in range(EPOCHS):
        train_logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        train_logger.info(f"Learning rate: {scheduler.get_last_lr()[0]}")
        train_logger.info(f"Training...")
        train(model, optimizer, scheduler, scaler, device, train_loader, val_loader, epoch)
        train_logger.info(f"Validation...")
        val(model, optimizer, scheduler, scaler, device, val_loader, epoch)
    