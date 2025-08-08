from src.core.tensor import Tensor
from src.utils.backend import xp

is_cuda = xp.__name__ == "cupy"

def MeanSquaredError(y_hat: Tensor, y: Tensor):
    return ((y_hat - y) ** 2).mean().exp().log()

def BinaryCrossEntropyWithLogits(logits: Tensor, y: Tensor):
    max_logits = xp.maximum(logits.data, 0)
    log_term = xp.log1p(xp.exp(-xp.abs(logits.data)))
    loss_data = (max_logits - logits.data * y.data + log_term).mean()

    out = Tensor(loss_data, requires_grad=logits.requires_grad)
    if out.requires_grad:
        out.parents = (logits,)
        def grad_fn(grad):
            sigmoid = 1.0 / (1.0 + xp.exp(-logits.data))
            grad_logits  = (sigmoid - y.data) * grad.data / logits.data.size
            return (Tensor(grad_logits, requires_grad=False),)
        out.grad_fn = grad_fn
    return out

def CrossEntropyWithLogits(logits: Tensor, y: Tensor, axis=-1, use_mask=True, pad_idx=0):
    eps = 1e-10

    # Normalize shapes: allow (B,S,V) or (B,V)
    if logits.data.ndim == 3:
        pass
    elif logits.data.ndim == 2:
        logits = logits[:, None, :]
        y = y[:, None]
    else:
        raise ValueError(f"Unsupported logits shape: {logits.data.shape}")

    # log softmax
    max_logits = logits.data.max(axis=axis, keepdims=True)
    shifted_logits = logits.data - max_logits
    logsumexp = xp.log(xp.sum(xp.exp(shifted_logits), axis=axis, keepdims=True) + eps)
    log_softmax = shifted_logits - logsumexp

    # indices
    B, S, _ = log_softmax.shape
    batch_idx = xp.arange(B)[:, None]
    seq_idx = xp.arange(S)[None, :]
    tgt_idx = xp.array(y.data).astype(xp.int32)
    idx = (batch_idx, seq_idx, tgt_idx)

    # mask
    apply_mask = use_mask and logits.data.ndim == 3
    if apply_mask:
        mask = (tgt_idx != pad_idx).astype(logits.data.dtype)  # (B,S)
        mask_exp = mask[..., None]                              # (B,S,1)

        target_log_probs = log_softmax[idx] * mask             # (B,S)
        denom = xp.maximum(mask.sum(), 1.0).astype(logits.data.dtype)
        loss_data = -target_log_probs.sum() / denom
    else:
        raise NotImplementedError("No mask not implemented")

    out = Tensor(loss_data, requires_grad=logits.requires_grad)

    if out.requires_grad:
        out.parents = (logits,)
        def grad_fn(grad):
            exp_shifted = xp.exp(shifted_logits)
            softmax_data = exp_shifted / (exp_shifted.sum(axis=axis, keepdims=True) + eps)

            grad_input = softmax_data.copy()
            grad_input[idx] -= 1.0

            if apply_mask:
                denom = xp.maximum(mask.sum(), 1.0).astype(logits.data.dtype)
                grad_input *= mask_exp
                factor = (1.0 / denom) * grad.data
            else:
                raise NotImplementedError("No mask not implemented")

            grad_out = grad_input * factor
            return (Tensor(grad_out, requires_grad=False),)
        out.grad_fn = grad_fn

    return out
