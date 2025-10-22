import sys
import os

sys.path.append(r"LLM")

import dlx as dlx
from dlx import AdamW, xp
from dlx.utils import LRScheduler
from LLM.preprocess.dataloader import DataLoader
from LLM.tokenizer.tokenizer import tokenizer
from LLM.training.model import Model
from dlx.nn.losses import CrossEntropyWithLogits
import time
from tqdm import tqdm
from LLM.training.utils import ProgressBarManager
from pathlib import Path
from dlx.utils.logger import train_logger, val_logger
from LLM.training.utils import RunningLossTracker

from LLM.training.train import (
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

R = 8
ALPHA = R
BATCH_SIZE = 64

EPOCHS = 1

_BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = os.path.join(_BASE_DIR, "data", "finetune", "train.parquet")
VAL_DIR = os.path.join(_BASE_DIR, "data", "finetune", "validation.parquet")
CHECKPOINT_DIR = os.path.join(_BASE_DIR, "checkpoints")

def build_cosine_lr(total_steps: int, warmup_steps: int, max_lr: float, final_lr: float):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + xp.cos(xp.tensor(progress * 3.1415926535)))
        scale = cosine * (1.0 - final_lr / max_lr) + (final_lr / max_lr)
        return float(scale)
    return lr_lambda


if __name__ == "__main__":
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
    )
    print("initialized model")

    special_token_ids = xp.tensor([2, 51680, 51681], dtype=xp.long)

    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name:
            lora_params.append(param)

        elif name == "token_emb.weight":
            param.requires_grad = True

            def mask_grad(grad, ids=special_token_ids):
                mask = xp.zeros_like(grad)
                mask[ids] = 1
                return grad * mask
            param.grad_fn = mask_grad

            lora_params.append(param)

        else:
            param.requires_grad = False
    print("initialized lora params")


    train_dl = DataLoader(TRAIN_DIR, BATCH_SIZE)
    val_dl = DataLoader(VAL_DIR, BATCH_SIZE)
    # Calculate scheduler parameters
    TOTAL_STEPS = EPOCHS * len(train_dl)
    WARMUP_STEPS = TOTAL_STEPS // 100 * 3
    MAX_LR = 1e-4
    FINAL_LR = 1e-6  # Final learning rate

    optimizer = AdamW(lora_params, lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.0)
    print("initialized optimizer")

    scheduler = LRScheduler(
        optimizer,
        lr_lambda=build_cosine_lr(
            total_steps=TOTAL_STEPS,
            warmup_steps=WARMUP_STEPS,
            max_lr=MAX_LR,
            final_lr=FINAL_LR,
        ),
    )


    criterion = CrossEntropyWithLogits(ignore_index=PAD_IDX)

    start_time = time.perf_counter()
    last_cp_time = start_time
    last_log_step = 0

    # Progress bar
    total_steps = EPOCHS * len(train_dl)
    pbar = tqdm(total=total_steps,  desc="Training", unit="step",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    model.train()
    global_step = 0
    accum = 0
    iter_count = 0
    loss_tracker = RunningLossTracker()
    for epoch in range(EPOCHS):
        for batch in train_dl:
            iter_count += 1
            # Unpack the batch tuple (x_data, y_data, masks) and move to device
            x_data, y_data, loss_mask = batch
        
            logits = model(x_data).view(-1, VOCAB_SIZE)
            targets = y_data.view(-1)
            loss_mask = loss_mask.view(-1)
            loss = CrossEntropyWithLogits(
                logits[loss_mask],
                targets[loss_mask],
                ignore_index=PAD_IDX
            )

 
            # Scale loss and backward pass
            loss.backward()
            accum += 1

            # Optimizer step with scaler
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum = 0
            global_step += 1

            # Update progress bar
            # Update running loss trackers in O(1)
            loss_tracker.update(loss.data)
            ema_100, ema_1k, ema_10k = loss_tracker.get_running_losses()
            pbar.set_postfix(
                loss=f"{loss.data:.4f}",
                ema_100=f"{ema_100:.4f}",
                ema_1k=f"{ema_1k:.4f}",
                ema_10k=f"{ema_10k:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}"
            )
            pbar.update(1)

            # Periodic training loss logging
            if iter_count - last_log_step >= LOG_INTERVAL_STEPS:
                train_logger.info(
                    f"iter={iter_count} step={global_step} loss={loss.data:.6f} lr={optimizer.param_groups[0]['lr']:.8f}"
                )
                last_log_step = iter_count

            # Checkpointing
            if time.perf_counter() - last_cp_time > CHECKPOINT_INTERVAL_SECONDS:
                # Validation before checkpoint
                val_loss = model.evaluate(val_dl)
                val_logger.info(
                    f"iter={iter_count} step={global_step} val_loss={val_loss:.6f}"
                )
                ema_100, ema_1k, ema_10k = loss_tracker.get_running_losses()
                pbar.set_postfix(
                    loss=f"{loss.data:.4f}",
                    ema_100=f"{ema_100:.4f}",
                    ema_1k=f"{ema_1k:.4f}",
                    ema_10k=f"{ema_10k:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}"
                )
                model.checkpoint(optimizer)
                last_cp_time = time.perf_counter()

    pbar.close()

    model.checkpoint(optimizer)