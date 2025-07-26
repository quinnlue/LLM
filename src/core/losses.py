from src.core.tensor import Tensor
from src.utils.backend import xp

is_cuda = xp.__name__ == "cupy"

def MSE(y_hat: Tensor, y: Tensor):
    return ((y_hat - y) ** 2).mean()

def BCE(y_hat: Tensor, y: Tensor):
    return -(y * y_hat.log() + (1 - y) * (1 - y_hat).log()).mean()

# ... existing code ...

def CrossEntropy(logits: Tensor, y: Tensor, axis=-1, mask=False, pad_idx=0):
    """
    Memory-efficient fused cross-entropy.
    • Avoids materialising separate soft-max, one-hot and grad buffers.
    • Works with optional padding-mask exactly like the previous version.
    logits : Tensor (B, S, V)
    y      : Tensor (B, S)   integer targets
    """
    eps = 1e-10

    # -------- log-softmax (numerically stable) --------
    max_logits  = logits.data.max(axis=axis, keepdims=True)
    exp_shifted = xp.exp(logits.data - max_logits)
    sum_exp     = exp_shifted.sum(axis=axis, keepdims=True)
    log_probs   = logits.data - max_logits - xp.log(sum_exp + eps)   # (B,S,V)

    B, S, V = logits.shape
    batch_idx = xp.arange(B, dtype=xp.int32)[:, None]   # (B,1)
    seq_idx   = xp.arange(S, dtype=xp.int32)[None, :]   # (1,S)
    tgt_idx   = y.data.astype(xp.int32)                 # (B,S)

    # -------- optional padding mask --------
    if mask:
        valid = (tgt_idx != pad_idx)
    else:
        valid = xp.ones_like(tgt_idx, dtype=bool)

    n_valid = valid.sum() + eps  # total #tokens contributing to the loss

    # selected log-probabilities and loss
    selected_logp = log_probs[batch_idx, seq_idx, tgt_idx]
    loss_data = -(selected_logp * valid).sum() / n_valid

    out = Tensor(loss_data, requires_grad=logits.requires_grad)

    # -------- backward (∂L/∂logits) --------
    if out.requires_grad:
        out.parents = (logits,)
        def grad_fn(grad):
            softmax = xp.exp(log_probs)                 # reuse log_probs ⇒ softmax
            softmax[batch_idx, seq_idx, tgt_idx] -= 1   # softmax - one_hot
            softmax *= valid[..., None]                 # apply mask in-place
            grad_input = (softmax / n_valid) * grad.data
            return (Tensor(grad_input, requires_grad=False),)
        out.grad_fn = grad_fn

    return out

# ... existing code ...
