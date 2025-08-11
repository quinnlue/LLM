import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.optim import AdamW
from src.preprocess.dataloader import DataLoader
from src.tokenizer.tokenizer import tokenizer
from src.utils.lr_scheduler import LRScheduler
from src.utils.backend import xp
from src.training.model import Model


# PATHS ------------------------------
TRAIN_DIR = r"data/train"
VAL_DIR = r"data/validation"
TEST_DIR = r"data/test"

CHECKPOINT_DIR = r"checkpoints"

# MODEL HYPERPARAMETERS ------------------------------
VOCAB_SIZE = len(tokenizer.get_vocab())
D_MODEL = 512
N_HEADS = 8
MAX_SEQ_LEN = 512
PAD_IDX = 0
EOS_IDX = 1
DEPTH = 4

# DATASET HYPERPARAMETERS ------------------------------
MINI_BATCH_PER_STEP = 1
BATCH_SIZE = 16
DATA_COLUMN = "seq"
BIN_COLUMN = "bin"

# OPTIMIZER HYPERPARAMETERS ------------------------------
EPOCHS = 1
EXPECTED_OPTIM_STEPS = 20_000
WARMUP_STEPS = 500
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

model.train(optimizer, train_dl)