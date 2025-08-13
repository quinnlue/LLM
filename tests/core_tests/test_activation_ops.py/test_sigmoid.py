import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

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


class TestSigmoid(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-5):
        self.assertTrue(xp.allclose(a, b, atol=atol))

    def test_sigmoid_tensor(self):
        shapes = [
            (1,),
            (2,3),
            (3,4,5),
            (4,5,6,7),
        ]

        for shape in shapes:
            a_my = Tensor(xp.random.randn(*shape).astype(xp.float32), requires_grad=True)
            a_pt = torch.tensor(a_my.data, requires_grad=True)

            b_my = a_my._sigmoid()
            b_pt = torch.sigmoid(a_pt)

            c_my = b_my.mean()
            c_pt = b_pt.mean()

            c_my.backward()
            c_pt.backward()

            self.assert_close(c_my.data, c_pt.detach().numpy())
            self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())


if __name__ == "__main__":
    unittest.main()