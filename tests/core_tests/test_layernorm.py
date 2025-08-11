import unittest, sys, os, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.core.module import Module
from src.core.tensor import Tensor
from src.utils.backend import xp

def to_numpy(a):
    return xp.asnumpy(a) if xp.__name__ == "cupy" else a

class TestLayerNorm(unittest.TestCase):
    margin = 3e-3  # relaxed for float16 fallback, but we set float32

    def test_layernorm_forward_shapes_and_stats_eps0(self):
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 4, 6)).astype(np.float32)

        M = Module()
        ln = M.layer_norm(axis=-1, eps=0.0)
        x = Tensor(x_np, requires_grad=True, dtype=xp.float32)
        y = ln(x)

        self.assertEqual(y.shape, x_np.shape)
        mean = to_numpy(xp.mean(y.data, axis=-1))
        var  = to_numpy(xp.var(y.data, axis=-1))
        np.testing.assert_allclose(mean, 0.0, rtol=0, atol=1e-4)
        np.testing.assert_allclose(var, 1.0, rtol=0, atol=1e-3)

    def test_layernorm_grads_match_pytorch(self):
        import torch
        rng = np.random.default_rng(7)
        B, S, D = 3, 5, 8
        x_np = rng.standard_normal((B, S, D)).astype(np.float32)
        eps = 1e-5

        M = Module()
        ln = M.layer_norm(axis=-1, eps=eps)
        x = Tensor(x_np, requires_grad=True, dtype=xp.float32)
        y = ln(x)
        loss = y.mean()
        loss.backward()

        ln_pt = torch.nn.LayerNorm(D, eps=eps, elementwise_affine=True)
        with torch.no_grad():
            ln_pt.weight.copy_(torch.ones(D, dtype=torch.float32))
            ln_pt.bias.copy_(torch.zeros(D, dtype=torch.float32))
        x_pt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        y_pt = ln_pt(x_pt)
        loss_pt = y_pt.mean()
        loss_pt.backward()

        np.testing.assert_allclose(to_numpy(x.grad.data), x_pt.grad.detach().numpy(), rtol=0, atol=self.margin)
        np.testing.assert_allclose(to_numpy(ln.gamma.grad.data), ln_pt.weight.grad.detach().numpy(), rtol=0, atol=self.margin)
        np.testing.assert_allclose(to_numpy(ln.beta.grad.data), ln_pt.bias.grad.detach().numpy(), rtol=0, atol=self.margin)

if __name__ == "__main__":
    unittest.main()