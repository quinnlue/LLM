import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.transformer import Transformer, Embedding
from src.models.torch_transformer import TransformerBlock
from src.core.tensor import Tensor
from src.core.optim import SGD
from src.core.losses import CrossEntropyWithLogits
from src.core.module import Module
from src.utils.backend import xp, set_seed


class TestLayernorm(unittest.TestCase):
    def assert_close(self, a, b, atol=5e-3):
        is_all_close = xp.allclose(a, b, atol=atol)
        max_diff = xp.max(xp.abs(a - b))
        print(f"MAX DIFF: {max_diff}")
        self.assertTrue(is_all_close)

    def setUp(self):
        self.d_model = 8
        self.n_heads = 2
        self.d_head = self.d_model // self.n_heads

        self.x = xp.random.randn(1, 5, self.d_model).astype(xp.float32)

        self.gamma = xp.random.randn(self.d_model).astype(xp.float32)
        self.beta = xp.random.randn(self.d_model).astype(xp.float32)

        class MyLayernorm(Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.layernorm = self.layer_norm(axis=-1, eps=1e-5)

            def forward(self, x):
                return self.layernorm(x)
        
        class PtLayernorm(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.layernorm = nn.LayerNorm(d_model)

            def forward(self, x):
                return self.layernorm(x)
        
        self.my_layernorm = MyLayernorm(self.d_model, self.n_heads)
        self.my_layernorm._build(self.x.shape)
        self.pt_layernorm = PtLayernorm(self.d_model)

        self.my_layernorm.layernorm.gamma = Tensor(self.gamma.copy(), requires_grad=True)
        self.my_layernorm.layernorm.beta = Tensor(self.beta.copy(), requires_grad=True)

        with torch.no_grad():
            self.pt_layernorm.layernorm.weight.copy_(torch.from_numpy(self.gamma))
            self.pt_layernorm.layernorm.bias.copy_(torch.from_numpy(self.beta))
            
    def test_layernorm(self):
        my_x = Tensor(self.x.copy(), requires_grad=True)
        pt_x = torch.tensor(self.x.copy(), requires_grad=True, dtype=torch.float32)

        my_output = self.my_layernorm.forward(my_x)
        pt_output = self.pt_layernorm.forward(pt_x)

        my_loss = my_output.mean()
        pt_loss = pt_output.mean()

        my_loss.backward()
        pt_loss.backward()

        print(xp.mean(xp.abs(my_output.data - pt_output.detach().numpy())))
        print(xp.mean(xp.abs(my_loss.data - pt_loss.detach().numpy())))


        self.assert_close(my_loss.data, pt_loss.detach().numpy())
        self.assert_close(self.my_layernorm.layernorm.gamma.grad.data, self.pt_layernorm.layernorm.weight.grad.detach().numpy())
        self.assert_close(self.my_layernorm.layernorm.beta.grad.data, self.pt_layernorm.layernorm.bias.grad.detach().numpy())


if __name__ == "__main__":
    unittest.main()