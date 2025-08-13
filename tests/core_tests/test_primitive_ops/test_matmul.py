import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
from src.core.tensor import Tensor
from src.utils.backend import xp



class TestMatmul(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-5):
        self.assertTrue(xp.allclose(a, b, atol=atol))

    def test_matmul_tensor_and_scalar_mat(self):
        a_my = Tensor(xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)

        with self.assertRaises(ValueError):
            a_my @ 2

    def test_matmul_vector_and_tensor(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        b_my = Tensor(xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), requires_grad=True)


        with self.assertRaises(ValueError):
            a_my @ b_my

    def test_matmul_tensor_and_tensor(self):
        _data = xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        a_my = Tensor(_data, requires_grad=True)
        b_my = Tensor(_data, requires_grad=True)

        a_pt = torch.tensor(_data, requires_grad=True)
        b_pt = torch.tensor(_data, requires_grad=True)

        c_my = a_my @ b_my
        c_pt = a_pt @ b_pt

        d_my = c_my.mean()
        d_pt = c_pt.mean()

        d_my.backward()
        d_pt.backward()

        self.assertTrue(xp.allclose(d_my.data, d_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))
        self.assertTrue(xp.allclose(b_my.grad.data, b_pt.grad.detach().numpy()))

    def test_matmul_tensor_and_tensor_requires_grad_false(self):
        _data = xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        a_my = Tensor(_data, requires_grad=True)
        b_my = Tensor(_data, requires_grad=False)

        a_pt = torch.tensor(_data, requires_grad=True)
        b_pt = torch.tensor(_data, requires_grad=False)

        c_my = a_my @ b_my
        c_pt = a_pt @ b_pt

        d_my = c_my.mean()
        d_pt = c_pt.mean()

        d_my.backward()
        d_pt.backward()

        self.assertTrue(xp.allclose(d_my.data, d_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))
        self.assertIsNone(b_my.grad)
        self.assertIsNone(b_pt.grad)

    def test_matmul_broadcast(self):
        shapes = [
            ( (4, 3), (3, 5) ),         # Matrix @ Matrix -> Matrix
            ( (2, 3, 4), (4, 5) ),      # Batch @ Matrix -> Batch
            ( (4, 4 , 5, 6), (6, 5) ),  # Big batch @ Matrix
        ]


        for shape_a, shape_b in shapes:
            a_data = xp.random.randn(*shape_a).astype(xp.float32)
            b_data = xp.random.randn(*shape_b).astype(xp.float32)

            a_my = Tensor(a_data, requires_grad=True)
            b_my = Tensor(b_data, requires_grad=True)

            a_pt = torch.tensor(a_data, requires_grad=True)
            b_pt = torch.tensor(b_data, requires_grad=True)

            c_my = a_my @ b_my
            c_pt = a_pt @ b_pt

            loss_my = c_my.mean()
            loss_pt = c_pt.mean()

            loss_my.backward()
            loss_pt.backward()

            # compare forward output
            assert xp.allclose(c_my.data, c_pt.detach().numpy(), rtol=1e-5, atol=1e-6)

            # compare gradients
            assert xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy(), rtol=1e-5, atol=1e-6)
            assert xp.allclose(b_my.grad.data, b_pt.grad.detach().numpy(), rtol=1e-5, atol=1e-6)

    def test_rmatmul(self):
        a_my = Tensor(xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), requires_grad=True)
        b_my = Tensor(xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), requires_grad=True)

        a_pt = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)
        b_pt = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)

        c_my = a_my @ b_my
        c_pt = a_pt @ b_pt

        d_my = c_my.mean()
        d_pt = c_pt.mean()

        d_my.backward()
        d_pt.backward()

        self.assert_close(d_my.data, d_pt.detach().numpy())
        self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())
        self.assert_close(b_my.grad.data, b_pt.grad.detach().numpy())


if __name__ == "__main__":
    unittest.main()
