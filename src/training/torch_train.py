import sys, os, math, time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ────────────────────────── 3rd-party ──────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# ────────────────────────── project deps ──────────────────────────
from src.preprocess.dataloader import DataLoader            # yields xp-based batches
from src.tokenizer.tokenizer import tokenizer               # yields vocab
from src.utils.backend import xp                            # numpy | cupy
from src.utils.logger import train_logger, val_logger       # existing loggers

# ────────────────────────── constants (copied from train.py) ──────────────────────────
TRAIN_DIR  = r"data/train"
VAL_DIR    = r"data/validation"
CHECKPOINT_DIR = r"checkpoints"

VOCAB_SIZE = len(tokenizer.get_vocab())   # 21680 in your data
D_MODEL    = 768
N_HEADS    = 12
MAX_SEQ_LEN = 512
PAD_IDX    = 0
EOS_IDX    = 1
DEPTH      = 12            # transformer layers


BATCH_SIZE = 180
MINI_BATCH_PER_STEP = 1
DATA_COLUMN  = "seq"
BIN_COLUMN   = "bin"

EPOCHS                    = 1
EXPECTED_OPTIM_STEPS      = 20_000
WARMUP_STEPS              = 200
MIN_LR, MAX_LR, FINAL_LR  = 3e-5, 1e-3, 1e-6
CHECKPOINT_INTERVAL_SECONDS = 3_600

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────── Scheduler (ported from src/utils/lr_scheduler.py) ──────────────────────────
def lr_schedule_lambda(step:int):
    """Exact same warm-up / anneal / final-dip schedule used in the original code."""
    if step < WARMUP_STEPS:                                 # warm-up
        return MIN_LR + (MAX_LR - MIN_LR) * step / WARMUP_STEPS
    anneal_steps  = EXPECTED_OPTIM_STEPS - WARMUP_STEPS
    final_dip_steps = max(1, int(anneal_steps * 0.1))
    anneal_steps  -= final_dip_steps
    if step < WARMUP_STEPS + anneal_steps:                  # cosine/linear anneal
        s = step - WARMUP_STEPS
        return MAX_LR - (MAX_LR - MIN_LR) * s / anneal_steps
    if step < EXPECTED_OPTIM_STEPS:                         # final dip
        s = step - WARMUP_STEPS - anneal_steps
        return MIN_LR - (MIN_LR - FINAL_LR) * s / final_dip_steps
    return FINAL_LR

# ────────────────────────── Model definition ──────────────────────────
class TransformerLM(nn.Module):
    """
    1. Token  + learnable positional embeddings
    2. DEPTH stacked nn.TransformerEncoderLayers (standard attention)
    3. Projection to vocabulary
    Equivalent tensor dimensions & call-signature to src/training/model.Model.forward().
    """
    def __init__(self,
                 vocab_size:int,
                 d_model:int,
                 n_heads:int,
                 depth:int,
                 max_seq_len:int,
                 pad_idx:int):
        super().__init__()

        self.pad_idx = pad_idx
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.dropout   = nn.Dropout(0.1)

        encoder_layer  = nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=n_heads,
                            dim_feedforward=d_model*4,
                            dropout=0.1,
                            activation='gelu',
                            batch_first=True)   # (B, S, D)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.Linear(d_model, vocab_size)

        # init
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_emb.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   mean=0, std=0.02)
        nn.init.normal_(self.proj.weight,      mean=0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, idx:torch.Tensor) -> torch.Tensor:
        """
        idx : (batch, seq) LongTensor
        returns logits : (batch, seq, vocab)
        """
        B, S = idx.shape
        device = idx.device
        pos_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, S)

        x = self.token_emb(idx) + self.pos_emb(pos_ids)
        x = self.dropout(x)

        # bool mask : True where padding tokens
        key_padding_mask = (idx == self.pad_idx)
        causal_mask = torch.triu(torch.ones(S, S, device=idx.device), diagonal=1).bool()
        x = self.transformer(x, src_key_padding_mask=key_padding_mask, mask=causal_mask)

        logits = self.proj(x)
        return logits

# ────────────────────────── helpers ──────────────────────────
def xp_to_torch(batch_xp):
    """Convert a src.core.tensor.Tensor to a torch.LongTensor (on DEVICE)
       while restoring the integer token IDs stored in float16."""
    arr = batch_xp.data
    if xp.__name__ == "cupy":          # cupy → host numpy
        arr = xp.asnumpy(arr)

    # 1) upgrade precision → round() guarantees exact int, then cast
    arr_int = np.rint(arr.astype(np.float32)).astype(np.int64)

    # 2) sanity-check / clamp
    if (arr_int >= VOCAB_SIZE).any() or (arr_int < 0).any():
        bad = arr_int[(arr_int >= VOCAB_SIZE) | (arr_int < 0)][0]
        raise ValueError(f"Found out-of-range token id {bad} (vocab={VOCAB_SIZE})")
        # Alternatively: arr_int = np.clip(arr_int, 0, VOCAB_SIZE-1)

    return torch.tensor(arr_int, dtype=torch.long, device=DEVICE)

def save_checkpoint(model, optimizer, step:int):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f"_step{step}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "step": step,
    }, os.path.join(CHECKPOINT_DIR, ckpt_name))
    train_logger.info(f"Saved checkpoint → {ckpt_name}")

# ────────────────────────── data-loading & training  ──────────────────────────
# Put everything below inside a guard so imports are side-effect free.
if __name__ == "__main__":
    # ─── data loaders ───
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

    # ─── model / optim / sched ───
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        depth=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
    ).to(DEVICE)

    criterion  = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer  = optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8)
    scheduler  = LambdaLR(
        optimizer,
        lambda step: lr_schedule_lambda(step) / MAX_LR,
    )
    # ─── AMP setup ───
    use_amp = DEVICE.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # ─── training loop ───
    global_step   = 0
    last_ckpt_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_loss = []
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}", position=0, leave=True):
            batch_t = xp_to_torch(batch)               # (B, S)
            inp, tgt = batch_t[:, :-1], batch_t[:, 1:]

            with autocast(enabled=use_amp):
                logits = model(inp)                    # (B, S-1, V)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
                loss = loss / MINI_BATCH_PER_STEP      # gradient accumulation if >1

            scaler.scale(loss).backward()

            if (global_step + 1) % MINI_BATCH_PER_STEP == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss.append(loss.item() * MINI_BATCH_PER_STEP)
            global_step += 1

            # ─── logging ───
            if global_step % 25 == 0:
                print(f"step {global_step}  loss {sum(epoch_loss[-25:])/25:.4f}")
                # train_logger.info(f"step {global_step}  loss {sum(epoch_loss[-50:])/50:.4f}")

            # ─── checkpointing ───
            if time.perf_counter() - last_ckpt_time > CHECKPOINT_INTERVAL_SECONDS:
                save_checkpoint(model, optimizer, global_step)
                last_ckpt_time = time.perf_counter()

        train_logger.info(f"Epoch {epoch} avg loss: {sum(epoch_loss)/len(epoch_loss):.4f}")

        # ─── quick validation ───
        model.eval()
        with torch.no_grad():
            val_losses = []
            # Fix: Add position=1 and leave=False for validation progress bar
            for vbatch in tqdm(val_dl, desc="Validation", position=1, leave=False):
                vb = xp_to_torch(vbatch)
                with autocast(enabled=use_amp):
                    logits = model(vb[:, :-1])
                    vl = criterion(logits.reshape(-1, VOCAB_SIZE), vb[:, 1:].reshape(-1))
                val_losses.append(vl.item())
        val_logger.info(f"Epoch {epoch}  val_loss: {sum(val_losses)/len(val_losses):.4f}")
        model.train()
