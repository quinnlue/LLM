import unittest
import sys, os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.core.module import Module
from src.core.tensor import Tensor

class TestLayerNorm(unittest.TestCase):
    M = Module()

    def test_layernorm(self):
        x = Tensor([[1, 2, 3], [4, 7, 6]])
        ln = self.M.layer_norm(axis=-1)
        x = ln(x)
        self.assertEqual(x.shape, (2, 3))
        print(x.data)
        print(x.shape)

if __name__ == "__main__":
    unittest.main()