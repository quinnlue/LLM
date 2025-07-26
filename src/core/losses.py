from src.core.tensor import Tensor
from src.utils.backend import xp

is_cuda = xp.__name__ == "cupy"

def MSE(y_hat: Tensor, y: Tensor):
    return ((y_hat - y) ** 2).mean()

def BCE(y_hat: Tensor, y: Tensor):
    return -(y * y_hat.log() + (1 - y) * (1 - y_hat).log()).mean()

def CrossEntropy(logits: Tensor, y: Tensor, axis=-1, mask=False, pad_idx=0):
    eps = 1e-10
    e_x = xp.exp(logits.data - logits.data.max(axis=axis, keepdims=True))
    softmax_data = e_x / e_x.sum(axis=axis, keepdims=True)

    B, S, V = softmax_data.shape

    batch_idx = xp.arange(B, dtype=xp.int32)[:, None]
    seq_idx = xp.arange(S, dtype=xp.int32)[None, :]  
    tgt_idx = xp.array(y.data).astype(xp.int32)

    if mask:
        mask = y.data != pad_idx
        tgt_idx.data = tgt_idx.data * mask


    idx = (batch_idx, seq_idx, tgt_idx)
    probs = softmax_data[idx]
    loss_data = -xp.log(probs + eps).mean()

    out = Tensor(loss_data, requires_grad=logits.requires_grad)

    if out.requires_grad:
        out.parents = (logits,)
        def grad_fn(grad):
            one_hot = xp.zeros_like(logits.data)
            one_hot[idx] = 1
            factor = 1 / logits.data.shape[0]
            grad_input = (softmax_data - one_hot) * factor
            grad_out = grad_input * grad.data
            return (Tensor(grad_out, requires_grad=False),)
        out.grad_fn = grad_fn

    return out
