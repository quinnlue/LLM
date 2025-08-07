import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from src.core.module import Module
from src.core.tensor import Tensor
import numpy as np


# ---------- helpers ----------

def assert_allclose(a, b, atol=1e-3, rtol=1e-3):
    try:
        assert np.allclose(a, b, atol=atol, rtol=rtol)
    except:
        print(a)
        print(b)
        raise AssertionError(f"\n{a}\n!=\n{b}")


# ---------- tests ----------

def test_layernorm_lazy_init_and_stats():
    """Ensure LayerNorm lazily initialises its parameters and normalises correctly."""

    # Create a standalone Module instance to get a LayerNorm layer
    parent = Module()
    ln = parent.layer_norm(axis=-1)

    # ----- lazy init checks (before forward) -----
    assert ln.initialized is False, "LayerNorm should not be initialised before first forward pass"
    assert ln.gamma is None and ln.beta is None, "Parameters should be None before first forward pass"

    # Forward pass with dummy data
    x_arr = np.random.randn(2, 3, 4)  # Arbitrary shape, last dim = 4
    x = Tensor(x_arr, requires_grad=True)

    out = ln(x)

    # ----- lazy init checks (after forward) -----
    assert ln.initialized is True, "LayerNorm should be initialised after first forward pass"
    assert ln.gamma is not None and ln.beta is not None, "Parameters should be created after first forward pass"
    assert ln.gamma.shape == (x_arr.shape[-1],)
    assert ln.beta.shape == (x_arr.shape[-1],)

    # ----- output statistics -----
    mean = out.data.mean(axis=-1)
    var = ((out.data - mean[..., None]) ** 2).mean(axis=-1)

    assert_allclose(mean, 0.0, atol=1e-5, rtol=1e-5)
    assert_allclose(var, 1.0, atol=1e-5, rtol=1e-5)

    # ----- gradient flow -----
    loss = out.sum()
    loss.backward()

    assert ln.gamma.grad is not None, "Gamma should receive gradient signal"
    assert ln.beta.grad is not None, "Beta should receive gradient signal"


# Execute when run as a script (similar to other tests in repo)
if __name__ == "__main__":
    test_layernorm_lazy_init_and_stats()
