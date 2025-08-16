import math
import os
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got {d_model=} and {num_heads=}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3 * D]
        q, k, v = qkv.split(self.d_model, dim=-1)

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
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FlashMHA(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
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
            TransformerBlock(d_model=d_model, num_heads=n_heads, mlp_ratio=mlp_ratio)
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
        for blk in self.blocks:
            x = blk(x)
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

    def checkpoint(self, optimizer: torch.optim.Optimizer) -> None:
        if self.checkpoint_dir is None:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        cp_path = os.path.join(self.checkpoint_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pt")
        torch.save({
            "model": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, cp_path)


