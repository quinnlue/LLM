import os
import sys
import unittest

import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from src.core.tensor import Tensor
from core.nn import Module
from src.utils.backend import xp


class TestEmbedding(unittest.TestCase):
    """Compare custom Embedding implementation against a minimal PyTorch reference."""

    def assert_close(self, a, b, atol=5e-3):
        is_all_close = xp.allclose(a, b, atol=atol)
        max_diff = xp.max(xp.abs(a - b))
        print(f"MAX DIFF: {max_diff}")
        self.assertTrue(is_all_close)

    def setUp(self):
        # Hyper-parameters
        self.vocab_size = 50
        self.d_model = 8
        self.max_seq_len = 10
        self.batch_size = 2
        self.seq_len = 6

        # Shared parameters (weights + positional encodings)
        self.W = xp.random.randn(self.vocab_size, self.d_model).astype(xp.float32)
        self.PE = xp.random.randn(self.max_seq_len, self.d_model).astype(xp.float32)

        # Random indices
        self.idx = xp.random.randint(0, self.vocab_size, size=(self.batch_size, self.seq_len)).astype(xp.int32)

        # ------------------------------------------------------------------
        # Custom embedding (our implementation)
        # ------------------------------------------------------------------
        class MyEmbedding(Module):
            def __init__(self, vocab_size, d_model, max_seq_len):
                super().__init__()
                self.embed = self.embedding(vocab_size, d_model, max_seq_len)

            def forward(self, idx):
                
                return self.embed.get_sentence_embedding(idx)

        self.my_embedding = MyEmbedding(self.vocab_size, self.d_model, self.max_seq_len)
        # Override parameters with shared tensors so both frameworks start identical
        self.my_embedding.embed.W = Tensor(self.W.copy(), requires_grad=True)
        self.my_embedding.embed.pe = Tensor(self.PE.copy(), requires_grad=True)

        # ------------------------------------------------------------------
        # PyTorch reference implementation
        # ------------------------------------------------------------------
        class PtEmbedding(nn.Module):
            def __init__(self, vocab_size, d_model, max_seq_len):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                # Positional encodings learned for fairness
                self.pe = nn.Parameter(torch.zeros(max_seq_len, d_model))

            def forward(self, idx):
                # idx: (B, T)
                out = self.embed(idx) + self.pe[: idx.size(1)].unsqueeze(0)
                return out

        self.pt_embedding = PtEmbedding(self.vocab_size, self.d_model, self.max_seq_len)

        # Copy parameters
        with torch.no_grad():
            self.pt_embedding.embed.weight.copy_(torch.from_numpy(self.W))
            self.pt_embedding.pe.copy_(torch.from_numpy(self.PE))

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_embedding_forward_and_backward(self):
        # Forward pass
        my_idx = self.idx.copy()
        pt_idx = torch.tensor(self.idx.copy(), dtype=torch.long, requires_grad=False)

        my_out = self.my_embedding.forward(my_idx)
        pt_out = self.pt_embedding.forward(pt_idx)

        # Loss
        my_loss = my_out.mean()
        pt_loss = pt_out.mean()

        # Backward
        my_loss.backward()
        pt_loss.backward()

        # Assert closeness of loss and gradients
        self.assert_close(my_loss.data, pt_loss.detach().numpy())
        self.assert_close(self.my_embedding.embed.W.grad.data, self.pt_embedding.embed.weight.grad.detach().numpy())
        self.assert_close(self.my_embedding.embed.pe.grad.data, self.pt_embedding.pe.grad.detach().numpy())


if __name__ == "__main__":
    unittest.main()
