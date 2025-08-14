import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.core.tensor import Tensor
from src.core.losses import CrossEntropyWithLogits

import numpy as np
import torch

class TestCETest(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-5):
        print(f"Max diff: {np.max(np.abs(a - b))}")
        self.assertTrue(np.allclose(a, b, atol=atol))

    def test_ce_loss_2d(self):
        shapes = [
            (500, 6),
            (500, 6, 7),
        ]

        for shape in shapes:
            _data = np.random.randn(*shape)
            _tgt = np.random.randint(0, shape[-1], size=shape[:-1])

            a_my = Tensor(_data, requires_grad=True)
            b_my = Tensor(_tgt, requires_grad=False)

            a_pt = torch.tensor(_data, requires_grad=True)
            b_pt = torch.tensor(_tgt, requires_grad=False).long()

            loss_my = CrossEntropyWithLogits(a_my, b_my)
            
            # For PyTorch, we need to handle the shape conversion properly
            if len(shape) == 2:
                # For 2D case, PyTorch expects (batch, vocab) and (batch,)
                loss_pt = torch.nn.CrossEntropyLoss()(a_pt, b_pt)
            else:
                # For 3D case, we need to reshape to (batch*seq, vocab) and (batch*seq,)
                batch_size, seq_len, vocab_size = shape
                a_pt_reshaped = a_pt.view(-1, vocab_size)
                b_pt_reshaped = b_pt.view(-1)
                loss_pt = torch.nn.CrossEntropyLoss()(a_pt_reshaped, b_pt_reshaped)

            loss_my.backward()
            loss_pt.backward()

            self.assert_close(loss_my.data, loss_pt.detach().numpy())
            
            # For gradients, we need to handle the reshaping back
            if len(shape) == 2:
                self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())
            else:
                # Reshape PyTorch gradients back to original shape
                a_pt_grad_reshaped = a_pt.grad.view(shape)
                self.assert_close(a_my.grad.data, a_pt_grad_reshaped.detach().numpy())

if __name__ == "__main__":
    unittest.main()