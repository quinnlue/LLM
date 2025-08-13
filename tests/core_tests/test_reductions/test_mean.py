import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch

from src.core.tensor import Tensor
from src.utils.backend import xp

class TestMean(unittest.TestCase):
    def setUp(self):
        self.shapes = [
            (1,),
            (2,3),
            (3,4,5),
            (4,5,6,7),
        ]

        self.axes = [
            0,
            1,
            2,
            3,
        ]
    def assert_close(self, a, b, atol=1e-5):
        self.assertTrue(xp.allclose(a, b, atol=atol))

    def test_mean(self):
        for i, shape in enumerate(self.shapes):
            for axis in self.axes[:i+1]:
                a_my = Tensor(xp.random.randn(*shape).astype(xp.float32), requires_grad=True)
                a_pt = torch.tensor(a_my.data, requires_grad=True)

                b_my = a_my.mean(axis=axis)
                b_pt = torch.mean(a_pt, dim=axis)

                c_my = b_my.mean()
                c_pt = b_pt.mean()

                c_my.backward()
                c_pt.backward()

                self.assert_close(c_my.data, c_pt.detach().numpy())
                self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())

    def test_keepdims(self):
        for i, shape in enumerate(self.shapes):
            for axis in self.axes[:i+1]:
                a_my = Tensor(xp.random.randn(*shape).astype(xp.float32), requires_grad=True)
                a_pt = torch.tensor(a_my.data, requires_grad=True)

                b_my = a_my.mean(axis=axis, keepdims=True)
                b_pt = torch.mean(a_pt, dim=axis, keepdim=True)

                self.assertEqual(b_my.data.shape, b_pt.shape)

                c_my = b_my.mean()
                c_pt = b_pt.mean()

                c_my.backward()
                c_pt.backward()

                self.assert_close(c_my.data, c_pt.detach().numpy())
                self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())

if __name__ == "__main__":
    unittest.main()
