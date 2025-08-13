import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.core.tensor import Tensor
from src.core.losses import MeanSquaredError

import numpy as np
import torch

class TestMSELoss(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-5):
        self.assertTrue(np.allclose(a, b, atol=atol))

    def test_mse_loss_2d(self):
        a_my = Tensor(np.array([[1., 2., 3.], [4., 5., 6.]]), requires_grad=True)
        b_my = Tensor(np.array([[7., 8., 9.], [10., 11., 12.]]), requires_grad=False)

        a_pt = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        b_pt = torch.tensor([[7., 8., 9.], [10., 11., 12.]], requires_grad=False)


        loss_my = MeanSquaredError(a_my, b_my)
        loss_pt = torch.nn.MSELoss()(a_pt, b_pt)

        loss_my.backward()
        loss_pt.backward()

        self.assert_close(loss_my.data, loss_pt.detach().numpy())
        self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())

    def test_mse_loss_1d(self):
        a_my = Tensor(np.array([1., 2., 3.]), requires_grad=True)
        b_my = Tensor(np.array([4., 5., 6.]), requires_grad=False)

        a_pt = torch.tensor([1., 2., 3.], requires_grad=True)
        b_pt = torch.tensor([4., 5., 6.], requires_grad=False)

        loss_my = MeanSquaredError(a_my, b_my)
        loss_pt = torch.nn.MSELoss()(a_pt, b_pt)

        loss_my.backward()
        loss_pt.backward()

        self.assert_close(loss_my.data, loss_pt.detach().numpy())
        self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())



if __name__ == "__main__":
    unittest.main()