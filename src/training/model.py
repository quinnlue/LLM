import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.module import Module
from src.core.losses import CrossEntropyWithLogits
from src.preprocess.dataloader import DataLoader
from src.utils.backend import xp
from src.utils.logger import train_logger, val_logger
from src.core.optim import Optimizer

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
                n_heads=n_heads
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
    
    def forward(self, idx):
        x = self.e.get_sentence_embedding(idx)
        for head in self.heads:
            x = head(x)
        x = self.project(x)
        return x
    
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
        
    def train(
        self, 
        optimizer: Optimizer,
        dl: DataLoader,
    ):
        last_cp_time = time.perf_counter()
        loss_history = []
        data = np.load("src/training/first_batch.npy")
        for epoch in range(100):
            y_hat = self.forward(data[:,:-1])
            loss = CrossEntropyWithLogits(y_hat, data[:,1:])
            loss_history.append(float(loss.data))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {loss.data}")

        # for epoch in range(self.epochs):
        #     for i, batch in enumerate(tqdm(dl, desc=f"Training epoch {epoch}")):

        #         y_hat = self.forward(batch[:,:-1])
        #         loss = CrossEntropyWithLogits(y_hat, batch[:,1:])
        #         loss_history.append(float(loss.data))
        #         loss.backward()
        #         optimizer.step()
        #         optimizer.zero_grad()

        #         if (i + 1) % 1 == 0:
        #             train_logger.info(f"Training loss: {np.array(loss_history[-25:]).mean()}")

        #         # checkpointing & validation
        #         if time.perf_counter() - last_cp_time > self.CHECKPOINT_INTERVAL_SECONDS:
        #             self.checkpoint(optimizer)
        #             last_cp_time = time.perf_counter()



