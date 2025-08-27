import sys
sys.path.append(r"gpt1")

import os
from pathlib import Path
import time
from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from gpt1.preprocess.dataloader import DataLoader
from gpt1.tokenizer.tokenizer import tokenizer
from gpt1.torch_train.model import Model
from dlx.utils.logger import train_logger, val_logger
from gpt1.training.utils import RunningLossTracker
from gpt1.torch_train.utils import load_latest_checkpoint

from gpt1.torch_train.train import (
    VOCAB_SIZE,
    D_MODEL,
    MAX_SEQ_LEN,
    PAD_IDX,
    N_HEADS,
    DEPTH,
    CHECKPOINT_DIR,
)

R = 8
ALPHA = R


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

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    scaler = GradScaler()

    load_latest_checkpoint(model, optimizer, scheduler, scaler, device, CHECKPOINT_DIR)
    