import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))


from src.core.tensor import Tensor
from src.core.losses import MSE, BCE, CrossEntropy
from src.utils.backend import xp


# ---------- helpers ----------
def assert_allclose(a, b, atol=1e-3, rtol=1e-3):
    assert xp.allclose(a, b, atol=atol, rtol=rtol), f"\n{a}\n!=\n{b}"


# ---------- tests ----------
def test_mse_value_and_grad():
    # shape (4,) so .mean()’s gradient scaling is correct
    y_hat_arr = xp.array([0.2, 0.5, 0.3, 0.8])
    y_arr     = xp.array([0.0, 1.0, 0.0, 1.0])

    y_hat = Tensor(y_hat_arr, requires_grad=True)
    y     = Tensor(y_arr,     requires_grad=False)

    loss = MSE(y_hat, y)

    # ----- value -----
    expected_loss = xp.mean((y_hat_arr - y_arr) ** 2)
    assert_allclose(loss.data, expected_loss)

    # ----- gradient -----
    loss.backward()
    N = y_hat_arr.size
    expected_grad = 2 * (y_hat_arr - y_arr) / N
    assert_allclose(y_hat.grad.data, expected_grad)


def test_bce_value_and_grad():
    y_hat_arr = xp.array([0.9, 0.2, 0.7, 0.4])
    y_arr     = xp.array([1.0, 0.0, 1.0, 0.0])

    y_hat = Tensor(y_hat_arr, requires_grad=True)
    y     = Tensor(y_arr,     requires_grad=False)

    loss = BCE(y_hat, y)

    # ----- value -----
    expected_loss = -xp.mean(
        y_arr * xp.log(y_hat_arr) + (1 - y_arr) * xp.log(1 - y_hat_arr)
    )
    assert_allclose(loss.data, expected_loss)

    # ----- gradient -----
    loss.backward()
    N = y_hat_arr.size
    expected_grad = (-y_arr / y_hat_arr + (1 - y_arr) / (1 - y_hat_arr)) / N
    assert_allclose(y_hat.grad.data, expected_grad)


def test_cross_entropy_value_and_grad():
    # shape: (B, S, V) = (2, 2, 3)
    logits_arr = xp.array(
        [
            [[2.0, 1.0, 0.1], [0.5, 2.1, 1.3]],
            [[1.2, 0.7, 3.3], [0.3, 0.1, 0.9]],
        ]
    )
    targets_arr = xp.array([[0, 2], [2, 1]])  # shape (B, S)

    logits  = Tensor(logits_arr,  requires_grad=True)
    targets = Tensor(targets_arr, requires_grad=False)

    loss = CrossEntropy(logits, targets)

    # ----- value -----
    max_logits   = logits_arr.max(axis=-1, keepdims=True)
    shifted      = logits_arr - max_logits
    logsumexp    = xp.log(xp.exp(shifted).sum(axis=-1, keepdims=True))
    log_softmax  = shifted - logsumexp
    batch_idx, seq_idx = xp.indices(targets_arr.shape)
    expected_loss = -log_softmax[batch_idx, seq_idx, targets_arr].mean()
    assert_allclose(loss.data, expected_loss)

    # ----- gradient -----
    loss.backward()
    softmax = xp.exp(shifted) / xp.exp(shifted).sum(axis=-1, keepdims=True)
    grad_expected = softmax.copy()
    grad_expected[batch_idx, seq_idx, targets_arr] -= 1.0
    grad_expected /= (targets_arr.shape[0] * targets_arr.shape[1])  # B * S
    assert_allclose(logits.grad.data, grad_expected)



test_mse_value_and_grad()
# test_bce_value_and_grad()
test_cross_entropy_value_and_grad()