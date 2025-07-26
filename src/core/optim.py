from src.core.module import Module, Tensor
from src.utils.backend import xp

class Optimizer:
    def __init__(self, params, lr):
        self.params = {}

        for param in params:
            self.params[param.name] = {
                "param": param,
                'm_t': xp.zeros_like(param.data),
                'v_t': xp.zeros_like(param.data),
                't': 0
            }
            print(param.name)

        self.lr = lr

    def reduce_like(self, grad: Tensor, target_shape: tuple) -> Tensor:
        gshape = grad.data.shape
        tshape = target_shape

        if gshape == tshape:
            return grad

        if len(gshape) > len(tshape):
            tshape = (1,) * (len(gshape) - len(tshape)) + tshape

        assert len(gshape) == len(tshape), f"Incompatible shapes: {gshape} vs {target_shape}"

        axes_to_sum = []
        for i, (gdim, tdim) in enumerate(zip(gshape, tshape)):
            if gdim != tdim:
                if tdim == 1:
                    axes_to_sum.append(i)
                else:
                    raise ValueError(f"Cannot broadcast grad shape {gshape} to target {target_shape}")

        for axis in reversed(axes_to_sum):
            grad = Tensor(grad.data.sum(axis=axis, keepdims=True))

        grad = Tensor(grad.data.reshape(target_shape))
        return grad

    def zero_grad(self):
        for param in self.params.values():
            if param['param'].grad is not None:
                param['param'].grad = None

class AdamW(Optimizer):
    def __init__(self, params, lr, clip_norm=1.0, weight_decay=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.clip_norm = clip_norm
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    def step(self):
        for param in self.params.values():
            if param['param'].grad is None:
                continue

            param_tensor = param['param']
            grad = param['param'].grad
            m_t = param['m_t']
            v_t = param['v_t']
            t = param['t']

            # print(type(param_tensor), type(grad), type(m_t), type(v_t), type(t))

            if grad.shape != param_tensor.data.shape:
                grad = self.reduce_like(grad, param_tensor.data.shape)
            
            if m_t.shape != param_tensor.data.shape:
                m_t = self.reduce_like(m_t, param_tensor.data.shape)

            if v_t.shape != param_tensor.data.shape:
                v_t = self.reduce_like(v_t, param_tensor.data.shape)

            m_t = m_t * self.beta_1 + (1 - self.beta_1) * grad.data
            v_t = v_t * self.beta_2 + (1 - self.beta_2) * (grad.data ** 2)
            
            m_hat = m_t / (1 - self.beta_1 ** (t + 1))
            v_hat = v_t / (1 - self.beta_2 ** (t + 1))

            # param_tensor.data = param_tensor.data * (1 - self.lr * self.weight_decay)

            param_tensor.data = param_tensor.data - self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)


            param['m_t'] = m_t
            param['v_t'] = v_t
            param['t'] = t + 1







class Standard(Optimizer):
    def __init__(self, params, lr, clip_norm=1.0):
        super().__init__(params, lr)
        self.clip_norm = clip_norm

    def step(self):
        for param in self.params.values():
            if param['param'].grad is None:
                continue

            param['param'].grad = self.reduce_like(param['param'].grad, param['param'].data.shape)
            param['param'].data -= self.lr * param['param'].grad.data

