from src.core.tensor import Tensor
from src.utils.backend import xp

is_cuda = xp.__name__ == "cupy"

def MSE(y_hat: Tensor, y: Tensor):
    return ((y_hat - y) ** 2).mean()

def BCE(y_hat: Tensor, y: Tensor):
    return -(y * y_hat.log() + (1 - y) * (1 - y_hat).log()).mean()

def CrossEntropy(logits: Tensor, y: Tensor, axis=-1, mask=False, pad_idx=0):
    """
    Memory-efficient cross entropy that avoids creating large one-hot tensors.
    Uses log_softmax + gather instead of softmax + one_hot.
    """
    eps = 1e-10
    
    # Compute log_softmax: logits - logsumexp(logits)
    max_logits = logits.data.max(axis=axis, keepdims=True)
    shifted_logits = logits.data - max_logits
    logsumexp = xp.log(xp.sum(xp.exp(shifted_logits), axis=axis, keepdims=True) + eps)
    log_softmax = shifted_logits - logsumexp
    
    # Gather the target log probabilities
    B, S, V = log_softmax.shape
    batch_idx = xp.arange(B, dtype=xp.int32)[:, None]
    seq_idx = xp.arange(S, dtype=xp.int32)[None, :]  
    tgt_idx = xp.array(y.data).astype(xp.int32)

    if mask:
        mask = y.data != pad_idx
        tgt_idx = tgt_idx * mask

    idx = (batch_idx, seq_idx, tgt_idx)
    target_log_probs = log_softmax[idx]
    loss_data = -target_log_probs.mean()

    out = Tensor(loss_data, requires_grad=logits.requires_grad)

    if out.requires_grad:
        out.parents = (logits,)
        def grad_fn(grad):
            # Compute softmax for gradient
            exp_shifted = xp.exp(shifted_logits)
            softmax_data = exp_shifted / (exp_shifted.sum(axis=axis, keepdims=True) + eps)
            
            # Create sparse gradient: softmax - one_hot
            grad_input = softmax_data.copy()
            grad_input[idx] -= 1.0  # Subtract 1 only at target positions
            
            factor = 1 / logits.data.shape[0]
            grad_out = grad_input * factor * grad.data
            return (Tensor(grad_out, requires_grad=False),)
        out.grad_fn = grad_fn

    return out
