import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.core.tensor import Tensor
from src.core.losses import BinaryCrossEntropyWithLogits

import numpy as np
import torch

class TestBCELoss(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-5):
        print(f"Max diff: {np.max(np.abs(a - b))}")
        self.assertTrue(np.allclose(a, b, atol=atol))

    def test_bce_loss_2d(self):
        shapes = [
            (500, 6),
            (500, 6, 7),
        ]

        for shape in shapes:
            _data = np.random.randn(*shape)
            _tgt = np.random.rand(*shape)

            a_my = Tensor(_data, requires_grad=True)
            b_my = Tensor(_tgt, requires_grad=False)

            a_pt = torch.tensor(_data, requires_grad=True)
            b_pt = torch.tensor(_tgt, requires_grad=False)

            loss_my = BinaryCrossEntropyWithLogits(a_my, b_my)
            loss_pt = torch.nn.BCEWithLogitsLoss()(a_pt, b_pt)

            loss_my.backward()
            loss_pt.backward()

            self.assert_close(loss_my.data, loss_pt.detach().numpy())
            self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())


if __name__ == "__main__":
    unittest.main()


