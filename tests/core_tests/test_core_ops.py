import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.core.tensor import Tensor
from src.utils.backend import xp


def to_numpy(array_like):
    if xp.__name__ == "cupy":
        return xp.asnumpy(array_like)
    return array_like


class TestCoreOps(unittest.TestCase):
    margin = 1e-3

    def test_add_value_and_grad_match_pytorch(self):
        import torch

        rng = np.random.default_rng(0)
        shape = (5, 7)
        a_np = rng.standard_normal(shape).astype(np.float64)
        b_np = rng.standard_normal(shape).astype(np.float64)

        # Our tensor implementation
        a = Tensor(a_np, requires_grad=True)
        b = Tensor(b_np, requires_grad=True)
        out = (a + b).mean()
        out.backward()

        # PyTorch reference
        a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
        out_pt = (a_pt + b_pt).mean()
        out_pt.backward()

        # Assertions
        self.assertTrue(np.allclose(out.data, out_pt.item(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(b.grad.data), b_pt.grad.detach().numpy(), atol=self.margin))

    def test_mul_value_and_grad_match_pytorch(self):
        import torch

        rng = np.random.default_rng(1)
        shape = (4, 3, 6)
        a_np = rng.standard_normal(shape).astype(np.float64)
        b_np = rng.standard_normal(shape).astype(np.float64)

        # Our tensor implementation
        a = Tensor(a_np, requires_grad=True)
        b = Tensor(b_np, requires_grad=True)
        out = (a * b).mean()
        out.backward()

        # PyTorch reference
        a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
        out_pt = (a_pt * b_pt).mean()
        out_pt.backward()

        # Assertions
        self.assertTrue(np.allclose(out.data, out_pt.item(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(b.grad.data), b_pt.grad.detach().numpy(), atol=self.margin))

    def test_sub_value_and_grad_match_pytorch(self):
        import torch

        rng = np.random.default_rng(2)
        shape = (8,)
        a_np = rng.standard_normal(shape).astype(np.float64)
        b_np = rng.standard_normal(shape).astype(np.float64)

        a = Tensor(a_np, requires_grad=True)
        b = Tensor(b_np, requires_grad=True)
        out = (a - b).mean()
        out.backward()

        a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
        out_pt = (a_pt - b_pt).mean()
        out_pt.backward()

        self.assertTrue(np.allclose(out.data, out_pt.item(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(b.grad.data), b_pt.grad.detach().numpy(), atol=self.margin))

    def test_matmul_value_and_grad_match_pytorch(self):
        import torch

        rng = np.random.default_rng(3)
        batch, m, k, n = 2, 3, 4, 5
        a_np = rng.standard_normal((batch, m, k)).astype(np.float64)
        w_np = rng.standard_normal((k, n)).astype(np.float64)

        a = Tensor(a_np, requires_grad=True)
        w = Tensor(w_np, requires_grad=True)
        out = (a @ w).mean()
        out.backward()

        a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        w_pt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
        out_pt = (a_pt @ w_pt).mean()
        out_pt.backward()

        self.assertTrue(np.allclose(out.data, out_pt.item(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(w.grad.data), w_pt.grad.detach().numpy(), atol=self.margin))

    def test_broadcast_add_grad_sums(self):
        rng = np.random.default_rng(4)
        a_np = rng.standard_normal((2, 3, 4)).astype(np.float64)
        b_np = rng.standard_normal((4,)).astype(np.float64)

        a = Tensor(a_np, requires_grad=True)
        b = Tensor(b_np, requires_grad=True)
        out = (a + b).mean()
        out.backward()

        # For broadcasting, gradient w.r.t b should sum across broadcasted axes
        grad_b_manual = np.ones_like(a_np) / a_np.size  # derivative of mean
        grad_b_manual = grad_b_manual.sum(axis=(0, 1))  # sum over broadcast dims
        self.assertTrue(np.allclose(to_numpy(b.grad.data), grad_b_manual, atol=self.margin))

    def test_div_value_and_grad_match_pytorch(self):
        import torch

        rng = np.random.default_rng(6)
        shape = (2, 3, 5)
        a_np = rng.standard_normal(shape).astype(np.float64)
        b_np = rng.standard_normal(shape).astype(np.float64) + 1.0  # avoid zeros

        a = Tensor(a_np, requires_grad=True)
        b = Tensor(b_np, requires_grad=True)
        out = (a / b).mean()
        out.backward()

        a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
        out_pt = (a_pt / b_pt).mean()
        out_pt.backward()

        self.assertTrue(np.allclose(out.data, out_pt.item(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(b.grad.data), b_pt.grad.detach().numpy(), atol=self.margin))

    def test_rtruediv_value_and_grad_match_pytorch(self):
        import torch

        rng = np.random.default_rng(5)
        shape = (3, 4)
        a_np = rng.standard_normal(shape).astype(np.float64)

        a = Tensor(a_np, requires_grad=True)
        scalar = 1.5
        out = (scalar / a).mean()
        out.backward()

        a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        out_pt = (scalar / a_pt).mean()
        out_pt.backward()

        self.assertTrue(np.allclose(out.data, out_pt.item(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), atol=self.margin))

    def test_getitem_single_index_value_and_grad_match_pytorch(self):
        import torch

        rng = np.random.default_rng(7)
        a_np = rng.standard_normal((4, 6)).astype(np.float64)
        idx = 2

        a = Tensor(a_np, requires_grad=True)
        out = a[idx].mean()
        out.backward()

        a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        out_pt = a_pt[idx].mean()
        out_pt.backward()

        self.assertTrue(np.allclose(out.data, out_pt.item(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), atol=self.margin))

    def test_getitem_slice_value_and_grad_match_pytorch(self):
        import torch

        rng = np.random.default_rng(8)
        a_np = rng.standard_normal((5, 7)).astype(np.float64)
        slice_obj = np.s_[1:4, ::2]  # select rows 1-3 and every 2nd column

        a = Tensor(a_np, requires_grad=True)
        out = a[slice_obj].mean()
        out.backward()

        a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
        out_pt = a_pt[slice_obj].mean()
        out_pt.backward()
        a_grad = to_numpy(a.grad.data)
        a_pt_grad = a_pt.grad.detach().cpu().numpy()  # safe CPU move
        print(f"out: {out.data}, out_pt: {out_pt.item()}")
        print(f"a.grad:\n{np.array2string(a_grad, precision=4, floatmode='fixed')}\n")
        print(f"a_pt.grad:\n{np.array2string(a_pt_grad, precision=4, floatmode='fixed')}\n")
        self.assertTrue(np.allclose(out.data, out_pt.item(), atol=self.margin))
        self.assertTrue(np.allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), atol=self.margin))


if __name__ == "__main__":
    unittest.main()
