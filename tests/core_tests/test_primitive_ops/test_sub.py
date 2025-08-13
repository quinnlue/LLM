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


class Testsub(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-5):
        self.assertTrue(xp.allclose(a, b, atol=atol))

    def test_sub_tensor_and_scalar(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        b_my = 1
        b_pt = 1

        c_my = a_my - b_my
        c_pt = a_pt - b_pt

        d_my = c_my.mean()
        d_pt = c_pt.mean()

        d_my.backward()
        d_pt.backward()

        self.assertTrue(xp.allclose(d_my.data, d_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))

    def test_sub_tensor_and_tensor(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_my = Tensor(xp.array([4.0, 5.0, 6.0]), requires_grad=True)
        b_pt = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        
        c_my = a_my - b_my
        c_pt = a_pt - b_pt

        d_my = c_my.mean()
        d_pt = c_pt.mean()

        d_my.backward()
        d_pt.backward()

        self.assertTrue(xp.allclose(d_my.data, d_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))
        self.assertTrue(xp.allclose(b_my.grad.data, b_pt.grad.detach().numpy()))

    def test_sub_tensor_and_tensor_requires_grad_false(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_my = Tensor(xp.array([4.0, 5.0, 6.0]), requires_grad=False)
        b_pt = torch.tensor([4.0, 5.0, 6.0], requires_grad=False)
        
        c_my = a_my - b_my
        c_pt = a_pt - b_pt

        d_my = c_my.mean()
        d_pt = c_pt.mean()

        d_my.backward()
        d_pt.backward()

        self.assertTrue(xp.allclose(d_my.data, d_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))

    def test_isub(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        b_my = Tensor(xp.array([4.0, 5.0, 6.0]), requires_grad=False)
        
        with self.assertRaises(NotImplementedError):
            a_my -= b_my

    def test_isub_scalar(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        b_my = 1
        
        with self.assertRaises(NotImplementedError):
            a_my -= b_my
        

    def test_rsub_scalar_and_tensor(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_my = 1
        b_pt = 1
        
        c_my = b_my - a_my
        c_pt = b_pt - a_pt

        d_my = c_my.mean()
        d_pt = c_pt.mean()

        d_my.backward()
        d_pt.backward()

        self.assertTrue(xp.allclose(d_my.data, d_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))

    def test_broadcast_sub(self):
        shapes = [
            ((3,), (1,)),
            ((2,3), (3,)),
            ((4,1,5), (1,5)),
            ((5,4,3), (4,3)),
        ]

        for shape_a, shape_b in shapes:
            a_data = xp.random.randn(*shape_a).astype(xp.float32)
            b_data = xp.random.randn(*shape_b).astype(xp.float32)

            a_my = Tensor(a_data, requires_grad=True)
            b_my = Tensor(b_data, requires_grad=True)

            a_pt = torch.tensor(a_data, requires_grad=True)
            b_pt = torch.tensor(b_data, requires_grad=True)

            c_my = a_my - b_my
            c_pt = a_pt - b_pt

            loss_my = c_my.mean()
            loss_pt = c_pt.mean()

            loss_my.backward()
            loss_pt.backward()

            # compare forward output
            self.assertTrue(xp.allclose(c_my.data, c_pt.detach().numpy()))

            # compare gradients
            self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))
            self.assertTrue(xp.allclose(b_my.grad.data, b_pt.grad.detach().numpy()))

    def test_neg(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        b_my = -a_my
        b_pt = -a_pt
        
        c_my = b_my.mean()
        c_pt = b_pt.mean()
        
        c_my.backward()
        c_pt.backward()
        
        self.assertTrue(xp.allclose(c_my.data, c_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))
        

if __name__ == "__main__":
    unittest.main()



