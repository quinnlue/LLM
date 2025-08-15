import dlx as dlx
from dlx import AdamW, xp
from dlx.utils import LRScheduler
from ..preprocess.dataloader import DataLoader
from ..tokenizer.tokenizer import tokenizer
from .model import Model
from dlx.nn.losses import CrossEntropyWithLogits
import time
from tqdm import tqdm
from .training_utils import ProgressBarManager

# PATHS ------------------------------
TRAIN_DIR = r"data/train"
VAL_DIR = r"data/validation"
TEST_DIR = r"data/test"

CHECKPOINT_DIR = r"checkpoints"
exit()
# MODEL HYPERPARAMETERS ------------------------------
VOCAB_SIZE = len(tokenizer.get_vocab())
D_MODEL = 256
N_HEADS = 4
MAX_SEQ_LEN = 512
PAD_IDX = 0
EOS_IDX = 1
DEPTH = 2

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

model._build((1, MAX_SEQ_LEN))

optimizer = AdamW(
    params=model.parameters(),
    lr=scheduler,
    precision=(xp.float16, xp.float32),
    clip_norm=1.0
)

criterion = CrossEntropyWithLogits

# TRAINING INFO VARS ------------------------------
start_time = time.perf_counter()
last_cp_time = start_time

# Calculate total steps for progress tracking
total_steps = EPOCHS * len(train_dl)

# Initialize progress bar manager
progress_manager = ProgressBarManager(total_steps, start_time)

# TRAINING LOOP ------------------------------
data = xp.load("first_batch.npy")

# Create progress bar with proper formatting
pbar = tqdm(
    total=total_steps,
    desc="Training",
    unit="step",
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
)

for epoch in range(EPOCHS):
    for batch_idx, batch in enumerate(train_dl):
        # Training step
        y_hat = model.forward(batch[:,:-1])
        loss = criterion(y_hat, batch[:,1:])
        
        loss_value = float(loss.data)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress bar efficiently
        progress_manager.update_progress(loss_value, optimizer.lr, pbar)
        
        # Checkpointing
        if time.perf_counter() - last_cp_time > CHECKPOINT_INTERVAL_SECONDS:
            model.checkpoint(optimizer)
            last_cp_time = time.perf_counter()

pbar.close()

# EVALUATION LOOP ------------------------------