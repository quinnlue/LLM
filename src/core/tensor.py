from src.utils.backend import xp

class Tensor:
    def __init__(self, data, requires_grad=True, requires_mask=False, name=None, eps=1e-5, dtype=xp.float16):
        self.data = xp.array(data).astype(dtype)
        self.requires_grad = requires_grad
        self.parents = ()
        self.grad_fn = None
        self.grad = None
        self.requires_mask = requires_mask
        self.mask = None
        self.name = name
        self.eps = eps

        self.is_cuda = xp.__name__ == "cupy"


    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)
    
    def mean(self, axis=None, keepdims=False):
        out = Tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                if axis is None:
                    denom = self.data.size
                    grad_self = grad.data * xp.ones_like(self.data) / denom
                else:
                    if isinstance(axis, tuple):
                        denom = 1
                        for ax in axis:
                            denom *= self.data.shape[ax]
                    else:
                        denom = self.data.shape[axis]
                    grad_self = grad.data * xp.ones_like(self.data) / denom
                return (Tensor(grad_self, requires_grad=False),)
            out.grad_fn = grad_fn
        return out
    
    def max(self, axis=None, keepdims=False):
        out = Tensor(xp.max(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad.data * (self.data == xp.max(self.data, axis=axis, keepdims=keepdims)),)
            out.grad_fn = grad_fn
        return out

    def sum(self, axis=None):
        out = Tensor(xp.sum(self.data, axis=axis), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad.sum(axis=axis),)
            out.grad_fn = grad_fn
        return out
    
    def pow(self, other):
        return self.__pow__(other)
    
    @property
    def n_params(self):
        if self.data.ndim == 1:
            return self.data.shape[0]
        elif self.data.ndim == 2:
            return self.data.shape[0] * self.data.shape[1]
        else:
            raise ValueError(f"Tensor has {self.data.ndim} dimensions, expected 1 or 2")
    
    @property
    def T(self):
        # TODO: fix this
        return Tensor(self.data.T, requires_grad=self.requires_grad)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __neg__(self):
        out = Tensor(xp.negative(self.data), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                return (-grad,)
            out.grad_fn = grad_fn
        return out
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        out = Tensor(xp.add(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad:
            out.parents = (self, other)

            def _reduce_broadcast(grad_arr, target_shape):
                if grad_arr.shape == target_shape:
                    return grad_arr
                ndim_diff = grad_arr.ndim - len(target_shape)
                target_ext = (1,) * ndim_diff + target_shape
                axes = tuple(i for i, (g_dim, t_dim) in enumerate(zip(grad_arr.shape, target_ext)) if t_dim == 1)
                if axes:
                    grad_arr = grad_arr.sum(axis=axes, keepdims=True)
                return grad_arr.reshape(target_shape)

            def grad_fn(grad):
                grad.requires_grad = False
                grad_self  = _reduce_broadcast(grad.data, self.data.shape)
                grad_other = _reduce_broadcast(grad.data, other.data.shape)
                return (Tensor(grad_self,  requires_grad=False),
                        Tensor(grad_other, requires_grad=False))

            out.grad_fn = grad_fn
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __iadd__(self, other):
        # raise NotImplementedError("Not implemented")
        self.data += other.data if isinstance(other, Tensor) else other
        return self
    
    def astype(self, dtype):
        # Edits type in place
        self.data = self.data.astype(dtype)
        return self

    def transpose(self, axes):
        out = Tensor(xp.transpose(self.data, axes), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            inv_axes = tuple(xp.argsort(xp.array(axes)).tolist())
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad.transpose(inv_axes),)
            out.grad_fn = grad_fn
        return out
    
    def reshape(self, shape):
        out = Tensor(xp.reshape(self.data, shape), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            in_shape = self.data.shape
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad.reshape(in_shape),)
            out.grad_fn = grad_fn
        return out
    
    def __matmul__(self, other):
        out = Tensor(xp.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad:
            out.parents = (self, other)
            def grad_fn(grad):
                grad.requires_grad = False
                other_T = xp.swapaxes(other.data, -1, -2)
                self_T  = xp.swapaxes(self.data,  -1, -2)

                # For grad_self: grad @ other_T
                grad_self = xp.matmul(grad.data, other_T)
                
                # For grad_other: self_T @ grad
                # Need to handle broadcasting by summing over batch dimensions
                if self.data.ndim > 2 and other.data.ndim == 2:
                    # Case: (batch, ..., in) @ (in, out) -> (batch, ..., out)
                    # grad_other should be (in, out), so sum over batch dimensions
                    grad_other = xp.matmul(self_T, grad.data)
                    # Sum over all dimensions except the last two
                    sum_axes = tuple(range(grad_other.ndim - 2))
                    if sum_axes:
                        grad_other = xp.sum(grad_other, axis=sum_axes)
                else:
                    grad_other = xp.matmul(self_T, grad.data)
                    
                return (Tensor(grad_self, requires_grad=False), Tensor(grad_other, requires_grad=False))
            out.grad_fn = grad_fn
        return out
    
    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other.__matmul__(self)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        out = Tensor(xp.subtract(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad:
            out.parents = (self, other)
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad, -grad)
            out.grad_fn = grad_fn
        return out
    
    def __isub__(self, other):
        raise NotImplementedError("Not implemented")
        self.data -= other.data if isinstance(other, Tensor) else other
        return self
    
    def __pow__(self, other):
        out = Tensor(xp.power(self.data, other), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_self = grad.data * other * xp.power(self.data, other - 1)
                return (Tensor(grad_self, requires_grad=False),)
            out.grad_fn = grad_fn
        return out
    
    def __rsub__(self, other):
        return -self + other

    
    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.data.shape}, dtype={self.data.dtype})"
    
    def __str__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype})"

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
            
        out = Tensor(xp.multiply(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad:
            out.parents = (self, other)
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad * other, grad * self)
            out.grad_fn = grad_fn
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __imul__(self, other):
        raise NotImplementedError("Not implemented")
    #     # TODO: idk if this is right
    #     return self * other

    
    def __div__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        if not isinstance(self, Tensor):
            self = Tensor(self, requires_grad=False)

        out = Tensor(xp.divide(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad:
            out.parents = (self, other)
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad / other, -grad * self / other ** 2)
            out.grad_fn = grad_fn
        return out
    
    def __truediv__(self, other):
        return self.__div__(other)
    
    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other.__truediv__(self)
    
    def exp(self):
        out = Tensor(xp.exp(self.data), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad * out,)
            out.grad_fn = grad_fn
        return out
    
    def log(self):
        safe_data = xp.maximum(self.data, self.eps)
        out = Tensor(xp.log(safe_data), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                safe_self = xp.maximum(self.data, self.eps)
                return (grad / Tensor(safe_self, requires_grad=False),)
            out.grad_fn = grad_fn
        return out

    def _sigmoid(self):
        sigmoid_data = 1.0/(1.0+xp.exp(-self.data))
        out = Tensor(sigmoid_data, requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_self = grad.data * sigmoid_data * (1 - sigmoid_data)
                return (Tensor(grad_self, requires_grad=False),)
            out.grad_fn = grad_fn
        return out
    
    def _softmax(self, axis):
        max_values = self.data.max(axis=axis, keepdims=True)
        shifted = self.data - max_values
        exp_shifted = xp.exp(shifted)
        sum_exp = exp_shifted.sum(axis=axis, keepdims=True)
        softmax_data = exp_shifted / (sum_exp + self.eps)

        out = Tensor(softmax_data, requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                dot = (grad.data * softmax_data).sum(axis=axis, keepdims=True)
                grad_input = softmax_data * (grad.data - dot)
                return (Tensor(grad_input, requires_grad=False),)
            out.grad_fn = grad_fn
        return out
    
    def _dropout(self, p: float=0.1, train: bool=True):
        if not train:
            return self
        mask = xp.random.rand(*self.data.shape) < (1 - p)
        out = Tensor(self.data * mask, requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad * mask,)
            out.grad_fn = grad_fn
        return out
    
    def _relu(self):
        relu_data = xp.maximum(0, self.data)
        out = Tensor(relu_data, requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_self = grad.data * (self.data > 0)
                return (Tensor(grad_self, requires_grad=False),)
            out.grad_fn = grad_fn
        return out
    
    def _gelu(self):
        gelu_data = 0.5 * self.data * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (self.data + 0.044715 * xp.power(self.data, 3))))
        out = Tensor(gelu_data, requires_grad=self.requires_grad)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                u = xp.sqrt(2 / xp.pi) * (self.data + 0.044715 * xp.power(self.data, 3))
                grad_self = grad.data * (0.5 * (1 + xp.tanh(u)) + self.data * (1 - xp.power(xp.tanh(u), 2)) * (xp.sqrt(2 / xp.pi) + 0.044715 * 3 * xp.power(self.data, 2)))
                return (Tensor(grad_self, requires_grad=False),)
            out.grad_fn = grad_fn
        return out
    
    def __getitem__(self, key):
        out_data = self.data[key]
        out = Tensor(out_data, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                full = xp.zeros_like(self.data)
                full[key] = grad.data
                return (Tensor(full, requires_grad=False),)
            out.grad_fn = grad_fn

        return out

    def update(self, lr: float):
        if self.name is not None:
            print(self.name)
        self.data -= lr * self.grad


    def backward(self, grad=None, _visited=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = Tensor(xp.ones_like(self.data), requires_grad=False)

        if _visited is None:
            _visited = {}

        if self not in _visited:
            _visited[self] = grad
        else:
            _visited[self] += grad
            return

        # store the total gradient
        self.grad = _visited[self]

        # propagate once to each parent
        if self.grad_fn is not None:
            # Save a local copy, then immediately sever the links so the
            # graph from this Tensor backward is eligible for GC.
            parents_local = self.parents
            grads         = self.grad_fn(self.grad)

            # *******  CRUCIAL: break reference cycles  *******
            self.grad_fn = None
            self.parents = ()

            for parent, g in zip(parents_local, grads):
                parent.backward(g, _visited)
        
    def zero_grad(self):
        self.grad = None
        self.grad_fn = None
        self.parents = ()

            
        





