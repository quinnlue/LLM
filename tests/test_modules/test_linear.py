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


class TestLinear(unittest.TestCase):
    def assert_close(self, a, b, atol=5e-3):
        is_all_close = xp.allclose(a, b, atol=atol)
        max_diff = xp.max(xp.abs(a - b))
        print(f"MAX DIFF: {max_diff}")
        self.assertTrue(is_all_close)

    def setUp(self):
        self.in_features = 10
        self.out_features = 20
        self.weight = xp.random.randn(self.out_features, self.in_features).astype(xp.float32)
        self.bias = xp.random.randn(self.out_features).astype(xp.float32)

        self.x = xp.random.randn(1, self.in_features).astype(xp.float32)
        class MyLinear(Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.fc1 = self.linear(in_features, out_features)

            def forward(self, x):
                return self.fc1(x)
        
        class PtLinear(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.fc1 = nn.Linear(in_features, out_features)

            def forward(self, x):
                return self.fc1(x)
        
        self.my_linear = MyLinear(self.in_features, self.out_features)
        self.pt_linear = PtLinear(self.in_features, self.out_features)

        self.my_linear.fc1.weight = Tensor(self.weight.copy().T, requires_grad=True)
        self.my_linear.fc1.bias = Tensor(self.bias.copy(), requires_grad=True)

        with torch.no_grad():
            self.pt_linear.fc1.weight.copy_(torch.from_numpy(self.weight))
            self.pt_linear.fc1.bias.copy_(torch.from_numpy(self.bias))
            
    def test_linear(self):
        my_x = Tensor(self.x.copy(), requires_grad=True)
        pt_x = torch.tensor(self.x.copy(), requires_grad=True, dtype=torch.float32)

        my_output = self.my_linear.forward(my_x)
        pt_output = self.pt_linear.forward(pt_x)

        my_loss = my_output.mean()
        pt_loss = pt_output.mean()

        my_loss.backward()
        pt_loss.backward()

        print(xp.mean(xp.abs(my_output.data - pt_output.detach().numpy())))
        print(xp.mean(xp.abs(my_loss.data - pt_loss.detach().numpy())))


        self.assert_close(my_loss.data, pt_loss.detach().numpy())
        self.assert_close(self.my_linear.fc1.weight.grad.data, self.pt_linear.fc1.weight.grad.detach().numpy().T)
        self.assert_close(self.my_linear.fc1.bias.grad.data, self.pt_linear.fc1.bias.grad.detach().numpy())


if __name__ == "__main__":
    unittest.main()