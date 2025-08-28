import sys
sys.path.append(r"gpt1")

import os
from pathlib import Path
import time
from datetime import datetime
import math
import random
import pandas as pd
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# torch DataLoader replaces custom one
from torch.utils.data import DataLoader as TorchDataLoader
from gpt1.tokenizer.tokenizer import tokenizer
from .model import Model
from dlx.utils.logger import train_logger, val_logger
from gpt1.training.utils import RunningLossTracker
from gpt1.torch_train.utils import load_latest_checkpoint


# Resolve project root (directory that contains the *gpt1* package)
_BASE_DIR = Path(__file__).resolve().parents[1]

# PATHS ------------------------------
TRAIN_DIR = _BASE_DIR / "data" / "train"
VAL_DIR = _BASE_DIR / "data" / "validation"
CHECKPOINT_DIR = _BASE_DIR / "checkpoints"

TRAIN_DIR = str(TRAIN_DIR)
VAL_DIR = str(VAL_DIR)
CHECKPOINT_DIR = str(CHECKPOINT_DIR)


# MODEL HYPERPARAMETERS ------------------------------
VOCAB_SIZE = len(tokenizer.get_vocab())
D_MODEL = 1024
N_HEADS = 16                 # 1024 % 16 == 0
MAX_SEQ_LEN = 512
PAD_IDX = 0
DEPTH = 12

# DATASET HYPERPARAMETERS ------------------------------
MINI_BATCH_PER_STEP = 4
BATCH_SIZE = 48
DATA_COLUMN = "seq"

# OPTIMIZER HYPERPARAMETERS ------------------------------
TOTAL_TOKENS = 8_800_000_000
EPOCHS = 1
_TOKENS_PER_STEP = BATCH_SIZE * MAX_SEQ_LEN * MINI_BATCH_PER_STEP
EXPECTED_OPTIM_STEPS = TOTAL_TOKENS // _TOKENS_PER_STEP
WARMUP_STEPS = EXPECTED_OPTIM_STEPS // 100 * 3
MAX_LR = 3e-4
FINAL_LR = 1e-6
CHECKPOINT_INTERVAL_SECONDS = 3600
LOG_INTERVAL_STEPS = 100


def build_cosine_lr(total_steps: int, warmup_steps: int, max_lr: float, final_lr: float):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))
        scale = cosine * (1.0 - final_lr / max_lr) + (final_lr / max_lr)
        return float(scale)
    return lr_lambda


# ---------- dataset ----------------------------------------------------------
class ParquetDataset(torch.utils.data.Dataset):
    """
    Map-style dataset that pulls sequences from parquet shards.
    Assumes each parquet file has a column ``src_column`` whose rows
    are lists / np arrays of token IDs.
    """
    def __init__(self, src_dir: str, src_column: str):
        self.src_column = src_column
        # Collect valid parquet files and compute cumulative row counts.
        # We proactively validate each file so that corrupted or non-parquet
        # files (which may have a *.parquet extension) are skipped instead of
        # crashing at runtime when the first bad file is encountered.
        self.files = []
        self.cum_rows = []
        total = 0
        for fname in os.listdir(src_dir):
            if not fname.endswith(".parquet"):
                continue
            fp = os.path.join(src_dir, fname)
            try:
                parquet_file = pq.ParquetFile(fp)
                n_rows = parquet_file.metadata.num_rows
            except (pq.lib.ArrowInvalid, OSError) as e:
                # Skip files that are not valid parquet or are unreadable.
                print(f"[ParquetDataset] WARNING: Skipping invalid parquet file '{fp}': {e}", file=sys.stderr)
                continue
            if n_rows == 0:
                # Empty shard – skip; avoids divide-by-zero issues later.
                print(f"[ParquetDataset] WARNING: Skipping empty parquet file '{fp}'", file=sys.stderr)
                continue
            self.files.append(fp)
            total += n_rows
            self.cum_rows.append(total)

        if not self.files:
            raise ValueError(f"No valid parquet files found in directory '{src_dir}'.")

        self._len = total

    def __len__(self) -> int:
        return self._len

    def _locate(self, idx: int) -> tuple[int, int]:
        # binary search cumulative offsets → (file_idx, row_idx)
        lo, hi = 0, len(self.cum_rows) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if idx < self.cum_rows[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        file_idx = lo
        prior = self.cum_rows[file_idx - 1] if file_idx else 0
        return file_idx, idx - prior

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, row_idx = self._locate(idx)
        df = pd.read_parquet(self.files[file_idx], columns=[self.src_column])
        seq = df.iloc[row_idx][self.src_column]
        return torch.as_tensor(seq, dtype=torch.long)
# -----------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dataset = ParquetDataset(TRAIN_DIR, DATA_COLUMN)
    val_dataset   = ParquetDataset(VAL_DIR,   DATA_COLUMN)

    train_dl = TorchDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = TorchDataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = Model(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
        n_heads=N_HEADS,
        transformer_depth=DEPTH,
        checkpoint_dir=CHECKPOINT_DIR,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.0)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=build_cosine_lr(
            total_steps=EXPECTED_OPTIM_STEPS,
            warmup_steps=WARMUP_STEPS,
            max_lr=MAX_LR,
            final_lr=FINAL_LR,
        ),
    )

    # Loss and GradScaler must be initialized before loading checkpoints
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = GradScaler()

    # --------------------------------------------------
    # Resume from latest checkpoint if one exists
    # --------------------------------------------------
    load_latest_checkpoint(model, optimizer, scheduler, scaler, device, CHECKPOINT_DIR)

    start_time = time.perf_counter()
    last_cp_time = start_time
    last_log_step = 0

    # Progress bar
    total_steps = EPOCHS * len(train_dl)
    pbar = tqdm(total=total_steps, desc="Training", unit="step",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    model.train()
    global_step = 0
    accum = 0
    iter_count = 0
    loss_tracker = RunningLossTracker()
    for epoch in range(EPOCHS):
        for batch in train_dl:
            iter_count += 1
            batch = torch.as_tensor(batch, dtype=torch.long, device=device)
            
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(batch[:, :-1])
                target = batch[:, 1:]
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1)) / MINI_BATCH_PER_STEP
                scaled_loss_value = float(loss.item() * MINI_BATCH_PER_STEP)

            # Scale loss and backward pass
            scaler.scale(loss).backward()
            accum += 1

            # Only update the pbar once per optimizer step
            if accum % MINI_BATCH_PER_STEP == 0:
                # Unscale gradients for gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
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
                val_loss = model.evaluate(val_dl, device)
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


if __name__ == "__main__":
    main()



