import sys
sys.path.append(r"gpt1")

import os
from pathlib import Path
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from gpt1.preprocess.dataloader import DataLoader
from gpt1.tokenizer.tokenizer import tokenizer
from .model import Model
from dlx.utils.logger import train_logger, val_logger
from gpt1.training.utils import RunningLossTracker


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
D_MODEL = 768
N_HEADS = 12
MAX_SEQ_LEN = 512
PAD_IDX = 0
DEPTH = 12

# DATASET HYPERPARAMETERS ------------------------------
MINI_BATCH_PER_STEP = 8
BATCH_SIZE = 64
DATA_COLUMN = "seq"

# OPTIMIZER HYPERPARAMETERS ------------------------------
EPOCHS = 1
EXPECTED_OPTIM_STEPS = 32_000
WARMUP_STEPS = 320
MIN_LR = 1e-5
MAX_LR = 5e-4
FINAL_LR = 1e-6
CHECKPOINT_INTERVAL_SECONDS = 3600
LOG_INTERVAL_STEPS = 100


def build_cosine_lr(total_steps: int, warmup_steps: int, min_lr: float, max_lr: float, final_lr: float):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # cosine decay from 1.0 to final_lr/max_lr
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))
        scale = cosine * (1.0 - final_lr / max_lr) + (final_lr / max_lr)
        return float(scale)
    return lr_lambda


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dl = DataLoader(
        src_dir=TRAIN_DIR,
        src_column=DATA_COLUMN,
        batch_size=BATCH_SIZE,
        shuffle_rows=True,
        shuffle_files=True,
    )
    val_dl = DataLoader(
        src_dir=VAL_DIR,
        src_column=DATA_COLUMN,
        batch_size=BATCH_SIZE,
        shuffle_rows=True,
        shuffle_files=True,
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

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=build_cosine_lr(
            total_steps=EXPECTED_OPTIM_STEPS,
            warmup_steps=WARMUP_STEPS,
            min_lr=MIN_LR,
            max_lr=MAX_LR,
            final_lr=FINAL_LR,
        ),
    )

    # --------------------------------------------------
    # Resume from latest checkpoint if one exists
    # --------------------------------------------------
    def _load_latest_checkpoint() -> None:
        if not os.path.isdir(CHECKPOINT_DIR):
            return
        ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
        if not ckpts:
            return
        latest = max(ckpts, key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)))
        cp_path = os.path.join(CHECKPOINT_DIR, latest)
        state = torch.load(cp_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        print(f"[INFO] Resumed from checkpoint {latest}")

    _load_latest_checkpoint()

    criterion = nn.CrossEntropyLoss()
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

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
            
            with autocast():
                logits = model(batch[:, :-1])
                target = batch[:, 1:]
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1)) / MINI_BATCH_PER_STEP
                scaled_loss_value = float(loss.item() * MINI_BATCH_PER_STEP)

            # Scale loss and backward pass
            scaler.scale(loss).backward()
            accum += 1

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


