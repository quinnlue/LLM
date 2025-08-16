import sys
sys.path.append(r"gpt1")

import dlx as dlx
from dlx import AdamW, xp
from dlx.utils import LRScheduler
from gpt1.preprocess.dataloader import DataLoader
from gpt1.tokenizer.tokenizer import tokenizer
from gpt1.training.model import Model
from dlx.nn.losses import CrossEntropyWithLogits
import time
from tqdm import tqdm
from gpt1.training.utils import ProgressBarManager
from pathlib import Path

# Resolve project root (directory that contains the *gpt1* package)
_BASE_DIR = Path(__file__).resolve().parents[1]

# PATHS ------------------------------
TRAIN_DIR = _BASE_DIR / "data" / "train"
VAL_DIR = _BASE_DIR / "data" / "validation"
TEST_DIR = _BASE_DIR / "data" / "test"
CHECKPOINT_DIR = _BASE_DIR / "checkpoints"

# Convert Path objects to str for downstream code that expects strings
TRAIN_DIR = str(TRAIN_DIR)
VAL_DIR = str(VAL_DIR)
TEST_DIR = str(TEST_DIR)
CHECKPOINT_DIR = str(CHECKPOINT_DIR)

# MODEL HYPERPARAMETERS ------------------------------
VOCAB_SIZE = len(tokenizer.get_vocab())
D_MODEL = 768
N_HEADS = 12
MAX_SEQ_LEN = 512
PAD_IDX = 0
EOS_IDX = 1
DEPTH = 12

# DATASET HYPERPARAMETERS ------------------------------
MINI_BATCH_PER_STEP = 1
BATCH_SIZE = 16
DATA_COLUMN = "seq"
BIN_COLUMN = "bin"

# OPTIMIZER HYPERPARAMETERS ------------------------------
EPOCHS = 1
EXPECTED_OPTIM_STEPS = 20_000
WARMUP_STEPS = 200
MIN_LR = 1e-5
MAX_LR = 5e-4
FINAL_LR = 1e-6
CHECKPOINT_INTERVAL_SECONDS = 3600

scheduler = LRScheduler(
    warmup_steps=WARMUP_STEPS,
    total_steps=EXPECTED_OPTIM_STEPS,
    min_lr=MIN_LR,
    max_lr=MAX_LR,
    final_lr=FINAL_LR
    )

print("Loading train data...")
train_dl = DataLoader(
    src_dir=TRAIN_DIR,
    src_column=DATA_COLUMN,
    batch_size=BATCH_SIZE,
    shuffle_rows=True,
    shuffle_files=True,
)

print("Loading validation data...")
val_dl = DataLoader(
    src_dir=VAL_DIR,
    src_column=DATA_COLUMN,
    batch_size=BATCH_SIZE,
    shuffle_rows=True,
    shuffle_files=True,
)

print("Building model...")
model = Model(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    max_seq_len=MAX_SEQ_LEN,
    pad_idx=PAD_IDX,
    n_heads=N_HEADS,
    transformer_depth=DEPTH,
    checkpoint_interval_seconds=CHECKPOINT_INTERVAL_SECONDS,
    train_dir=TRAIN_DIR,
    validation_dir=VAL_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    epochs=EPOCHS,
    mini_batch_per_step=MINI_BATCH_PER_STEP,
)

print("Building optimizer...")
optimizer = AdamW(
    params=model.parameters(),
    lr=scheduler,
    precision=(xp.float16, xp.float32),
    clip_norm=1.0
)

print("Building criterion...")
criterion = CrossEntropyWithLogits

# TRAINING INFO VARS ------------------------------
start_time = time.perf_counter()
last_cp_time = start_time

# INITIALIZING PROGRESS BAR -----------------------
print("Initializing progress bar...")
total_steps = EPOCHS * len(train_dl)
progress_manager = ProgressBarManager(total_steps, start_time)
pbar = tqdm(
    total=total_steps,
    desc="Training",
    unit="step",
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
)

# TRAINING LOOP ------------------------------
print("Training...")
for epoch in range(EPOCHS):
    for batch_idx, batch in enumerate(train_dl):
        # Training step
        y_hat = model.forward(batch[:,:-1])
        loss = criterion(y_hat, batch[:,1:])/MINI_BATCH_PER_STEP
        
        loss_value = float(loss.data * MINI_BATCH_PER_STEP)
        
        loss.backward()
        
        if (batch_idx + 1) % MINI_BATCH_PER_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Update progress bar efficiently
        progress_manager.update_progress(loss_value, optimizer, pbar)
        
        # Checkpointing
        if time.perf_counter() - last_cp_time > CHECKPOINT_INTERVAL_SECONDS:
            model.checkpoint(optimizer)
            last_cp_time = time.perf_counter()

pbar.close()

# EVALUATION LOOP ------------------------------