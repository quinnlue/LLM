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
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import torch.nn.functional as F

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
    LOG_INTERVAL_STEPS,
    CHECKPOINT_INTERVAL_SECONDS,
)

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SFTDataset(Dataset):
    def __init__(self, data_path: str, max_seq_len: int, data_column: str, mask_column: str):
        df = pd.read_parquet(data_path)
        
        # Convert lists to numpy arrays and create input/output pairs
        x_data_list = []
        y_data_list = []
        masks_list = []
        
        for _, row in df.iterrows():
            tokens = np.array(row[data_column])
            mask = np.array(row[mask_column])
            
            # Create input (tokens[:-1]) and output (tokens[1:]) pairs
            x_seq = tokens[:-1]  # all tokens except the last one
            y_seq = tokens[1:]   # all tokens except the first one
            mask_seq = mask[1:] # mask for input sequence
            
            x_data_list.append(x_seq)
            y_data_list.append(y_seq)
            masks_list.append(mask_seq)
        
        # Convert lists to single numpy arrays first, then to tensors
        self.x_data = torch.from_numpy(np.array(x_data_list)).long()
        self.y_data = torch.from_numpy(np.array(y_data_list)).long()
        self.masks = torch.from_numpy(np.array(masks_list)).bool()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.masks[idx]

        

R = 8
ALPHA = R
BATCH_SIZE = 64

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
    ).to(device)  # Move model to device FIRST
    print("initialized model")


    train_dataset = SFTDataset(TRAIN_DIR, MAX_SEQ_LEN, "tokens", "mask")
    val_dataset = SFTDataset(VAL_DIR, MAX_SEQ_LEN, "tokens", "mask")
    print("initialized datasets")
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
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

    # Skip optimizer state because we’re only training LoRA parameters.
    load_latest_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        device,
        CHECKPOINT_DIR,
        strict=False,
        load_optim_state=False,
    )
    print("loaded latest checkpoint")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    print("initialized criterion")

    start_time = time.perf_counter()
    last_cp_time = start_time
    last_log_step = 0

    # Progress bar
    total_steps = EPOCHS * len(train_loader)
    pbar = tqdm(total=total_steps,  desc="Training", unit="step",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    model.train()
    global_step = 0
    accum = 0
    iter_count = 0
    loss_tracker = RunningLossTracker()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            iter_count += 1
            # Unpack the batch tuple (x_data, y_data, masks) and move to device
            x_data, y_data, loss_mask = batch
            
            
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            loss_mask = loss_mask.to(device)
            

            with autocast(device_type='cuda'):
                logits = model(x_data).view(-1, VOCAB_SIZE)
                targets = y_data.view(-1)
                loss_mask = loss_mask.view(-1)
                loss = F.cross_entropy(
                    logits[loss_mask],
                    targets[loss_mask],
                    ignore_index=PAD_IDX,
                    reduction='mean'
                )

                scaled_loss_value = float(loss.item())
 

            # Scale loss and backward pass
            scaler.scale(loss).backward()
            accum += 1

            # Unscale gradients for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum = 0
            global_step += 1

            # Update progress bar
            # Update running loss trackers in O(1)
            loss_tracker.update(scaled_loss_value)
            ema_100, ema_1k, ema_10k = loss_tracker.get_running_losses()
            pbar.set_postfix(
                loss=f"{scaled_loss_value:.4f}",
                ema_100=f"{ema_100:.4f}",
                ema_1k=f"{ema_1k:.4f}",
                ema_10k=f"{ema_10k:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}"
            )
            pbar.update(1)

            # Periodic training loss logging
            if iter_count - last_log_step >= LOG_INTERVAL_STEPS:
                train_logger.info(
                    f"iter={iter_count} step={global_step} loss={scaled_loss_value:.6f} lr={optimizer.param_groups[0]['lr']:.8f}"
                )
                last_log_step = iter_count

            # Checkpointing
            if time.perf_counter() - last_cp_time > CHECKPOINT_INTERVAL_SECONDS:
                # Validation before checkpoint
                val_loss = model.evaluate(val_loader, device)
                val_logger.info(
                    f"iter={iter_count} step={global_step} val_loss={val_loss:.6f}"
                )
                ema_100, ema_1k, ema_10k = loss_tracker.get_running_losses()
                pbar.set_postfix(
                    loss=f"{scaled_loss_value:.4f}",
                    ema_100=f"{ema_100:.4f}",
                    ema_1k=f"{ema_1k:.4f}",
                    ema_10k=f"{ema_10k:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}"
                )
                model.checkpoint(optimizer, scheduler, scaler)
                last_cp_time = time.perf_counter()

    pbar.close()

    # Final checkpoint
    model.checkpoint(optimizer, scheduler, scaler)
    