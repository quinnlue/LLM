import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.transformer import Transformer, Embedding
from src.core.tensor import Tensor
from src.core.module import Module
from src.utils.backend import xp, set_seed

import unittest

set_seed(42)

class TestTransformer(unittest.TestCase):
    def my_forward(self, x):
        B, T, _ = x.shape




    def assert_close(self, a, b, atol=1e-5):
        is_all_close = xp.allclose(a, b, atol=atol)
        if not is_all_close:
            print(f"===================== FAILED ======================")
            print(f"a shape: {a.shape}")
            print(f"b shape: {b.shape}")
            print(f"a: {a}")
            print(f"b: {b}")
            print(f"Max diff: {xp.max(xp.abs(a - b))}")
            print(f"===================================================")
        self.assertTrue(is_all_close)

    def test_parameters_same(self):
        pass

    def test_transformer_forward(self):
        pass

    def test_transformer_backward(self):
        pass

    def test_cross_entropy_with_logits(self):
        pass






