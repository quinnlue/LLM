import sys, os, unittest
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.core.tensor import Tensor
from src.utils.backend import xp

def to_numpy(a):
    return xp.asnumpy(a) if xp.__name__ == "cupy" else a

class TestDivisionBroadcast(unittest.TestCase):
    margin = 1e-3

    def test_div_broadcast_denominator_value_and_grad_match_pytorch(self):
        import torch
        rng = np.random.default_rng(0)
        a_np = rng.standard_normal((2, 3, 5)).astype(np.float32)
        b_np = (rng.standard_normal((2, 3, 1)).astype(np.float32) + 2.0)  # avoid small denom

        a = Tensor(a_np, requires_grad=True, dtype=xp.float32)
        b = Tensor(b_np, requires_grad=True, dtype=xp.float32)
        out = (a / b).mean()
        out.backward()

        a_pt = torch.tensor(a_np, dtype=torch.float32, requires_grad=True)
        b_pt = torch.tensor(b_np, dtype=torch.float32, requires_grad=True)
        out_pt = (a_pt / b_pt).mean()
        out_pt.backward()

        np.testing.assert_allclose(out.data, out_pt.item(), rtol=0, atol=self.margin)
        np.testing.assert_allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), rtol=0, atol=self.margin)
        np.testing.assert_allclose(to_numpy(b.grad.data), b_pt.grad.detach().numpy(), rtol=0, atol=self.margin)

        self.assertEqual(tuple(b.grad.data.shape), b_np.shape)

    def test_div_broadcast_numerator_value_and_grad_match_pytorch(self):
        import torch
        rng = np.random.default_rng(1)
        a_np = rng.standard_normal((1, 5)).astype(np.float32)  # broadcast across batch/seq
        b_np = (rng.standard_normal((4, 3, 5)).astype(np.float32) + 2.0)

        a = Tensor(a_np, requires_grad=True, dtype=xp.float32)
        b = Tensor(b_np, requires_grad=True, dtype=xp.float32)
        out = (a / b).mean()
        out.backward()

        a_pt = torch.tensor(a_np, dtype=torch.float32, requires_grad=True)
        b_pt = torch.tensor(b_np, dtype=torch.float32, requires_grad=True)
        out_pt = (a_pt / b_pt).mean()
        out_pt.backward()

        np.testing.assert_allclose(out.data, out_pt.item(), rtol=0, atol=self.margin)
        np.testing.assert_allclose(to_numpy(a.grad.data), a_pt.grad.detach().numpy(), rtol=0, atol=self.margin)
        np.testing.assert_allclose(to_numpy(b.grad.data), b_pt.grad.detach().numpy(), rtol=0, atol=self.margin)

        self.assertEqual(tuple(a.grad.data.shape), a_np.shape)

if __name__ == "__main__":
    unittest.main()