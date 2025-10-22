import os

import dlx as dlx
from dlx import Module, CrossEntropyWithLogits, xp
from dlx.utils import train_logger, val_logger
from dlx.nn.optim import Optimizer
from LLM.preprocess.dataloader import DataLoader
from LLM.tokenizer.tokenizer import tokenizer
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
import gc

class Model(Module):
    def __init__(
            self, 
            vocab_size: int, 
            d_model: int, 
            max_seq_len: int, 
            pad_idx: int, 
            n_heads: int, 
            transformer_depth: int, 
            checkpoint_interval_seconds: int,
            train_dir: str,
            validation_dir: str,
            checkpoint_dir: str,
            epochs: int,
            mini_batch_per_step: int,
            mlp_ratio: int = 4,
            lora: bool = False,
            lora_r: int = 16,
            lora_alpha: int = 16
        ):
        super().__init__()

        # CONSTANTS ---------------------------------------
        self.CHECKPOINT_INTERVAL_SECONDS = checkpoint_interval_seconds
        self.TRAIN_DIR = train_dir
        self.VAL_DIR = validation_dir
        self.CHECKPOINT_DIR = checkpoint_dir
        self.PAD = pad_idx
        self.is_cuda = xp.__name__ == "cupy"

        self.epochs = epochs
        self.mini_batch_per_step = mini_batch_per_step

        self.transformer_depth = transformer_depth
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.mlp_ratio = mlp_ratio
        self.lora = lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.head_dim = d_model // n_heads
        self.vocab_size = vocab_size

        # VARIABLES ---------------------------------------
        self.best_val_loss = float("inf")

        # MODEL ARCHITECTURE ------------------------------
        self.e = self.embedding(
            vocab_size, 
            d_model, 
            max_seq_len, 
            pad_idx
        )

        self.heads = [
            self.transformer(
                d_model=d_model, 
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                lora=lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha

            ) for _ in range(transformer_depth)
            ]

        
        self.project = self.linear(
            d_model, 
            vocab_size, 
            module_type="linear", 
            layer_type="linear", 
            name="project"
        )
        # --------------------------------------------------
    
    def forward(self, idx, kv_cache: xp.ndarray | None = None, current_position: int = 0):
        x = self.e.get_sentence_embedding(idx)
        for i, head in enumerate(self.heads):
            if kv_cache is not None:
                x = head(x, kv_cache['k'][i], kv_cache['v'][i], current_position)
            else:
                x = head(x)
        logits = self.project(x)
        return logits
    
    def evaluate(self, dl):
        losses = []
        for batch in tqdm(dl, desc="Evaluating"):
            batch.requires_grad = False
            y_hat = self.forward(batch[:,:-1])
            loss = CrossEntropyWithLogits(y_hat, batch[:,1:])
            losses.append(loss.data)
        return xp.mean(xp.array(losses))
    
    def checkpoint(self, optimizer: Optimizer):
        # val_loss = self.evaluate()
        # val_logger.info(f"Validation loss: {val_loss}")
        cp_path = os.path.join(self.CHECKPOINT_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.save_checkpoint(optimizer, cp_path)
        




if __name__ == "__main__":
    VOCAB_SIZE = len(tokenizer.get_vocab())
    D_MODEL = 1024
    N_HEADS = 16
    MAX_SEQ_LEN = 512
    PAD_IDX = 0
    DEPTH = 12

    model = Model(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
        n_heads=N_HEADS,
        transformer_depth=DEPTH,
        checkpoint_interval_seconds=3600,
        train_dir="data/train",
        validation_dir="data/validation",
        checkpoint_dir="checkpoints",
        epochs=1,
        mini_batch_per_step=8
    )

    print(model)