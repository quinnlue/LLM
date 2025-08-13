import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.module import Module, Linear, LayerNorm
from src.core.losses import CrossEntropyWithLogits
from src.core.optim import SGD, AdamW
from src.utils.lr_scheduler import LRScheduler
from src.core.tensor import Tensor
from src.tokenizer.tokenizer import tokenizer
from src.utils.backend import xp
import time
from typing import List
from src.tokenizer.tokenizer import Tokenizer
import pandas as pd
import numpy as np


src = np.load("first_batch.npy") # this is of shape (16, 512)
x = src[:, :-1]
y = src[:, 1:]





class Net(Module):
    def __init__(self, d_model, n_heads, vocab_size, max_seq_len):
        super().__init__()
        # Store for debugging expectations
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Shape-check utility that raises on mismatch
        def _check_shape(name, actual, expected):
            def _shape_of(obj):
                try:
                    return tuple(obj.shape)
                except Exception:
                    return None
            actual_shape = _shape_of(actual)
            if expected is not None and actual_shape != expected:
                raise ValueError(f"Shape mismatch for {name}: expected {expected}, got {actual_shape}")
        self._check_shape = _check_shape

        self.e = self.embedding(vocab_size, d_model, max_seq_len, name="Embedding")

        self.head1 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head2 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head3 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head4 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head5 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head6 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head7 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.head8 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.project = self.linear(d_model, vocab_size, name="project")
    
    def forward(self, idx):
        # Input token indices
        B, T = idx.shape if hasattr(idx, "shape") else (None, None)

        # Embedding + positional encoding
        x = self.e.get_sentence_embedding(idx)

        # Transformer blocks (residual-preserving shape)
        x = self.head1(x)
        x = self.head2(x)
        x = self.head3(x)
        x = self.head4(x)
        x = self.head5(x)
        x = self.head6(x)
        x = self.head7(x)
        x = self.head8(x)

        # Final projection to vocabulary logits
        x = self.project(x)
        return x

    def train(self, x, y, epochs, optimizer):
        for epoch in range(1):
            # Inputs to training step
            B, T = x.shape if hasattr(x, "shape") else (None, None)

            y_hat = self.forward(x)

            # Expect logits for each position and vocab

            # Loss expects axis=-1 over vocab
            loss = CrossEntropyWithLogits(y_hat, y, axis=-1)
    
            loss.backward()

            # Snapshot params pre-step to compute update tensors post-step
            _pre_step = {}

            # Backprop debug: print parameter and update (grad) shapes before optimizer applies updates
            try:
                for pname, p in self.parameters().items():
                    g = p.grad
                    if g is None:
                        continue
                    try:
                        _pre_step[pname] = p.data.copy()
                    except Exception:
                        _pre_step[pname] = None
                    p_shape = tuple(p.shape) if hasattr(p, "shape") else None
                    g_shape = tuple(g.shape) if hasattr(g, "shape") else None
                    g_has_nan = False
                    g_has_inf = False
                    try:
                        g_has_nan = bool((g.data != g.data).any())  # NaN check without xp.isnan for portability
                        # Inf check: compare against large finite bound
                        from src.utils.backend import xp as _xp
                        g_has_inf = bool((_xp.isinf(g.data)).any())
                    except Exception:
                        pass
                    mismatch_note = " [MISMATCH]" if (p_shape is not None and g_shape is not None and p_shape != g_shape) else ""
                    print(f"[BACKPROP] update -> {pname}: param{p_shape}, grad{g_shape}{mismatch_note}" + (" [NaN]" if g_has_nan else "") + (" [Inf]" if g_has_inf else ""))
            except Exception as _dbg_ex:
                print(f"[BACKPROP] debug print failed: {_dbg_ex}")

            optimizer.step()

            # Post-step: compute and report actual applied update shapes
            try:
                from src.utils.backend import xp as _xp
                for pname, p in self.parameters().items():
                    old = _pre_step.get(pname, None)
                    if old is None:
                        continue
                    try:
                        upd = p.data - old
                        u_shape = tuple(upd.shape)
                        p_shape = tuple(p.shape) if hasattr(p, "shape") else None
                        mismatch_note = " [MISMATCH]" if (p_shape is not None and u_shape is not None and p_shape != u_shape) else ""
                        u_has_nan = bool((_xp.isnan(upd)).any()) if hasattr(_xp, 'isnan') else bool((upd != upd).any())
                        u_has_inf = bool((_xp.isinf(upd)).any()) if hasattr(_xp, 'isinf') else False
                        print(f"[UPDATE ] applied -> {pname}: param{p_shape}, update{u_shape}{mismatch_note}" + (" [NaN]" if u_has_nan else "") + (" [Inf]" if u_has_inf else ""))
                    except Exception as _upd_ex:
                        print(f"[UPDATE ] failed to compute update for {pname}: {_upd_ex}")
            except Exception as _post_ex:
                print(f"[UPDATE ] post-step debug failed: {_post_ex}")
            optimizer.zero_grad()
            if epoch % 1 == 0:
                print(f"Loss: {loss.data}")
                
if __name__ == "__main__":
    D_MODEL = 768
    VOCAB_SIZE = len(tokenizer.get_vocab())
    N_HEADS = 12
    MAX_SEQ_LEN = 512
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


    model = Net(d_model=D_MODEL, n_heads=N_HEADS, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN)
    model._build((15, 15))
    optimizer = AdamW(model.parameters(), lr=scheduler, precision=(xp.float16, xp.float32), clip_norm=1.0)


    model.train(x, y, epochs=1000, optimizer=optimizer)


    
        