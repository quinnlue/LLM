import math
import os
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, lora: bool = False, lora_r: int = 16, lora_alpha: int = 16):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got {d_model=} and {num_heads=}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.lora = lora

        if self.lora:
            self.q_lora_A = nn.Linear(d_model, lora_r, bias=False)
            self.k_lora_A = nn.Linear(d_model, lora_r, bias=False)
            self.v_lora_A = nn.Linear(d_model, lora_r, bias=False)
            self.q_lora_B = nn.Linear(lora_r, d_model, bias=False)
            self.k_lora_B = nn.Linear(lora_r, d_model, bias=False)
            self.v_lora_B = nn.Linear(lora_r, d_model, bias=False)
            self.o_lora_A = nn.Linear(d_model, lora_r, bias=False)
            self.o_lora_B = nn.Linear(lora_r, d_model, bias=False)
            self.scaling = lora_alpha / lora_r

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3 * D]
        q, k, v = qkv.split(self.d_model, dim=-1)

        if self.lora:
            q_lora_delta = self.scaling * (x @ self.q_lora_A.weight.T @ self.q_lora_B.weight)
            k_lora_delta = self.scaling * (x @ self.k_lora_A.weight.T @ self.k_lora_B.weight)
            v_lora_delta = self.scaling * (x @ self.v_lora_A.weight.T @ self.v_lora_B.weight)
            q = q + q_lora_delta
            k = k + k_lora_delta
            v = v + v_lora_delta

        # Reshape to heads
        def shape_heads(t: torch.Tensor) -> torch.Tensor:
            return (
                t.view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )  # [B, H, T, Hd]

        q = shape_heads(q)
        k = shape_heads(k)
        v = shape_heads(v)



        # Use PyTorch SDPA which dispatches to FlashAttention on supported GPUs/dtypes
        if x.is_cuda:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask, dropout_p=0.0, is_causal=True)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask, dropout_p=0.0, is_causal=True)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        if self.lora:
            attn_out = attn_out + self.scaling * (attn_out @ self.o_lora_A.weight.T @ self.o_lora_B.weight)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, lora: bool = False, lora_r: int = 16, lora_alpha: int = 16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FlashMHA(d_model, num_heads, lora, lora_r, lora_alpha)
        self.ln2 = nn.LayerNorm(d_model)
        self.lora = lora
        if self.lora:
            self.proj_up_lora_A = nn.Linear(d_model, lora_r, bias=False)
            self.proj_up_lora_B = nn.Linear(lora_r, d_model * mlp_ratio, bias=False)
            self.proj_down_lora_A = nn.Linear(d_model * mlp_ratio, lora_r, bias=False)
            self.proj_down_lora_B = nn.Linear(lora_r, d_model, bias=False)
            self.scaling = lora_alpha / lora_r

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pad_mask)
        x = self.ln2(x)
        if self.lora:
            proj_up_lora_delta = self.scaling * (x @ self.proj_up_lora_A.weight.T @ self.proj_up_lora_B.weight)
            proj_up = self.mlp[0](x) + proj_up_lora_delta

            activated = self.mlp[1](proj_up)
        
            proj_down_lora_delta = self.scaling * (activated @ self.proj_down_lora_A.weight.T @ self.proj_down_lora_B.weight)
            mlp_out = self.mlp[2](activated) + proj_down_lora_delta
        else:
            mlp_out = self.mlp(x)

        x = x + mlp_out
        return x


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        pad_idx: int,
        n_heads: int,
        transformer_depth: int,
        checkpoint_dir: Optional[str] = None,
        mlp_ratio: int = 4,
        lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 16,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.checkpoint_dir = checkpoint_dir

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Parameter(torch.empty(max_seq_len, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=n_heads, mlp_ratio=mlp_ratio, lora=lora, lora_r=lora_r, lora_alpha=lora_alpha)
            for _ in range(transformer_depth)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T] of token ids
        bsz, seq_len = idx.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        x = self.token_emb(idx)  # [B, T, D]
        x = x + self.pos_emb[:seq_len].unsqueeze(0)
        pad_mask = (idx == self.pad_idx).unsqueeze(1)         # [B, 1, K_len]  <── ONLY 3-D
                                                              # broadcast → (B, Q_len, K_len)
        print(pad_mask)
        print(idx)
        print(self.pad_idx)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def evaluate(self, dataloader, device: torch.device) -> float:
        self.eval()
        losses = []
        criterion = nn.CrossEntropyLoss()
        for batch in dataloader:
            batch = torch.as_tensor(batch, dtype=torch.long, device=device)
            logits = self.forward(batch[:, :-1])
            target = batch[:, 1:]
            loss = criterion(logits.reshape(-1, self.vocab_size), target.reshape(-1))
            losses.append(loss.item())
        self.train()
        return float(sum(losses) / max(1, len(losses)))

    def checkpoint(self, optimizer: torch.optim.Optimizer, scheduler: "torch.optim.lr_scheduler._LRScheduler | None" = None, scaler: "torch.cuda.amp.GradScaler | None" = None, *, keep_last: int = 3, min_free_gb: float = 1.0) -> None:
        """Safely persist a training checkpoint.

        This implementation adds several safeguards over a raw ``torch.save``:
        1. The checkpoint is first written to a *temporary* file and then atomically
           moved into place.  This prevents partially-written files from being
           picked up later if the job is interrupted or the disk becomes full.
        2. If the target filesystem has less than ``min_free_gb`` gigabytes of
           free space, the checkpoint is *skipped* to avoid training crashes.
        3. Only the ``keep_last`` most recent checkpoint files are retained –
           older ones are removed to limit disk usage.
        """
        if self.checkpoint_dir is None:
            return  # No checkpoint directory specified – silently skip.

        import logging
        import shutil
        import tempfile

        # Ensure the checkpoint directory exists.
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Safety: verify we still have enough free space left on the target drive.
        free_bytes = shutil.disk_usage(self.checkpoint_dir).free
        if free_bytes < min_free_gb * 1024 ** 3:
            logging.warning(
                "Skipping checkpoint – only %.2f GB free (minimum %.2f GB required)",
                free_bytes / 1024 ** 3,
                min_free_gb,
            )
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_path = os.path.join(self.checkpoint_dir, f"{timestamp}.pt")

        # Write to a temporary file first, then atomically move into place.
        try:
            with tempfile.NamedTemporaryFile(dir=self.checkpoint_dir, suffix=".tmp", delete=False) as tmp_fp:
                torch.save(
                    {
                        "model": self.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        **({"scheduler": scheduler.state_dict()} if scheduler is not None else {}),
                        **({"scaler": scaler.state_dict()} if scaler is not None else {}),
                    },
                    tmp_fp.name,
                    _use_new_zipfile_serialization=True,
                )
                tmp_fp.flush()
                os.fsync(tmp_fp.fileno())

            # Atomic move – ``replace`` works across POSIX & Windows (Python ≥3.3).
            os.replace(tmp_fp.name, final_path)
        except Exception as exc:
            logging.exception("Failed to write checkpoint: %s", exc)
            # Clean up the temporary file if it still exists.
            try:
                if os.path.exists(tmp_fp.name):
                    os.remove(tmp_fp.name)
            except Exception:  # noqa: BLE001 – best-effort cleanup.
                pass
            return

        # House-keeping: delete older checkpoints, keep only the most recent ones.
        try:
            ckpts = sorted(
                (p for p in os.listdir(self.checkpoint_dir) if p.endswith(".pt")),
                key=lambda p: os.path.getmtime(os.path.join(self.checkpoint_dir, p)),
            )
            for old_ckpt in ckpts[:-keep_last]:
                try:
                    os.remove(os.path.join(self.checkpoint_dir, old_ckpt))
                except Exception:
                    logging.warning("Could not remove old checkpoint %s", old_ckpt)
        except Exception as exc:
            logging.warning("Checkpoint pruning failed: %s", exc)


