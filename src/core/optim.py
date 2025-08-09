from src.core.module import Module, Tensor
from src.utils.backend import xp
import numpy as np
import os
from src.utils.lr_scheduler import LRScheduler

class Optimizer:
    def __init__(self, params, lr: LRScheduler | float = 1e-3, clip_norm=1.0, precision: tuple[xp.dtype, xp.dtype] | xp.dtype | None = None):
        # Set precision -------------------------------------------------------------
        if precision is not None:
            if isinstance(precision, tuple):
                if len(precision) != 2:
                    raise ValueError("precision must contain a tuple of two elements")
                self.model_dtype, self.master_dtype = precision
                self.mixed_precision = True
            elif isinstance(precision, xp.dtype):
                self.model_dtype = self.master_dtype = precision
                self.mixed_precision = False
            else:
                raise ValueError("precision must be a dtype or tuple of dtypes")
        else:
            self.model_dtype = self.master_dtype = xp.float32
            self.mixed_precision = False

        
        
        # Set params -------------------------------------------------------------
        self._raw_params = params
        self.params = {}
        for name, param in self._raw_params.items():
            self.params[name] = {
                "param": param.astype(self.model_dtype),
            }

            # Set master param if mixed precision
            if self.mixed_precision:
                self.params[name]["param_master"] = Tensor(param.data.astype(self.master_dtype), requires_grad=False)
                self.params[name]["param_master"].requires_grad = False

        # Set lr scheduler -------------------------------------------------------------
        self.lr_scheduler = lr if isinstance(lr, LRScheduler) else None
        self.lr = lr
        self.t = 0

        self.clip_norm = clip_norm
        self.is_cuda = xp.__name__ == "cupy"


    def _clip_norm(self, grad: Tensor):
        if self.clip_norm is None:
            raise ValueError("clip_norm is not set")
        
        norm = xp.linalg.norm(grad.data)
        if norm > self.clip_norm:
            grad.data = grad.data * (self.clip_norm / norm)
        return grad

    def get_lr(self, step: int):
        if self.lr_scheduler is not None:
            return self.lr_scheduler(step)
        else:
            return self.lr

    def set_lr_scheduler(self, lr_scheduler: LRScheduler):
        self.lr_scheduler = lr_scheduler

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

    def _save_params(self, path):
        # ONLY CALL WITHIN THE SPECIFIC OPTIMIZER (AdamW, Standard, etc.)
        os.makedirs(path, exist_ok=True)
        os.makedirs(f"{path}/model", exist_ok=True)

        for name, param in self.params.items():
            if self.mixed_precision:
                xp.save(f"{path}/model/{name}.npy", param['param_master'].data)
            else:
                xp.save(f"{path}/model/{name}.npy", param['param'].data)

        xp.save(f"{path}/optim/t.npy", self.t)

    def _load_params(self, path):
        # ONLY CALL WITHIN THE SPECIFIC OPTIMIZER (AdamW, Standard, etc.)
        for name, param in self.params.items():
            if self.mixed_precision:
                param['param_master'].data = xp.load(f"{path}/model/{name}.npy")
                param['param'].data = param['param_master'].data.astype(self.model_dtype)
            else:
                param['param'].data = xp.load(f"{path}/model/{name}.npy")

class AdamW(Optimizer):
    def __init__(self, params, lr: LRScheduler | float = 1e-3, clip_norm=1.0, weight_decay=0.01, beta_1=0.9, beta_2=0.999, eps=1e-4, precision: tuple[xp.dtype, xp.dtype] | xp.dtype | None = None):
        # Fix: Pass the actual precision parameter instead of hardcoding
        super().__init__(params, lr=lr, clip_norm=clip_norm, precision=precision)
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.clip_norm = clip_norm
        self.eps = eps

        for name, param in self.params.items():
            # Create momentum and variance tensors with the correct dtype and shape
            param_shape = param['param'].data.shape
            self.params[name].update({
                'm_t': xp.zeros(param_shape, dtype=self.master_dtype),
                'v_t': xp.zeros(param_shape, dtype=self.master_dtype),
            })

    def _save_state(self, path):
        os.makedirs(path, exist_ok=True)
        os.makedirs(f"{path}/optim", exist_ok=True)
        os.makedirs(f"{path}/optim/m_t", exist_ok=True)
        os.makedirs(f"{path}/optim/v_t", exist_ok=True)

        self._save_params(path)
        for name, param in self.params.items():
            np.save(f"{path}/optim/m_t/{name}.npy", param['m_t'])
            np.save(f"{path}/optim/v_t/{name}.npy", param['v_t'])




    def _load_state(self, path):
        self._load_params(path)
        for name, param in self.params.items():
            param['m_t'] = xp.array(np.load(f"{path}/optim/m_t/{name}.npy"))
            param['v_t'] = xp.array(np.load(f"{path}/optim/v_t/{name}.npy"))

        self.t = int(np.load(f"{path}/optim/t.npy"))



    def step(self):
        for param in self.params.values():
            if param['param'].grad is None:
                continue

            if self.mixed_precision:
                master_param_tensor = param['param_master']
                dtype = self.master_dtype
            else:
                master_param_tensor = param['param']
                dtype = self.model_dtype

            grad = param['param'].grad

            grad = self._clip_norm(grad)

            m_t = param['m_t']
            v_t = param['v_t']

            if grad.shape != master_param_tensor.data.shape:
                grad = self.reduce_like(grad, master_param_tensor.data.shape)
            
            if m_t.shape != master_param_tensor.data.shape:
                m_t = self.reduce_like(m_t, master_param_tensor.data.shape)

            if v_t.shape != master_param_tensor.data.shape:
                v_t = self.reduce_like(v_t, master_param_tensor.data.shape)

            grad_data = grad.data.astype(dtype)

            del grad

            # Add numerical stability checks for mixed precision
            if xp.isnan(grad_data).any() or xp.isinf(grad_data).any():
                print(f"Warning: NaN/Inf detected in gradients, skipping update")
                continue

            effective_eps = max(self.eps, 1e-5) if self.master_dtype == xp.float16 else self.eps

            # # Scale gradients to prevent underflow in float16
            # if self.mixed_precision and self.master_dtype == xp.float16:
            #     grad_data = grad_data * 2**8  # Scale up by 256

            m_t = m_t * self.beta_1 + (1 - self.beta_1) * grad_data
            v_t = v_t * self.beta_2 + (1 - self.beta_2) * (grad_data ** 2)
            
            m_hat = m_t / max(1 - self.beta_1 ** (self.t + 2) + effective_eps, 1e-4)
            v_hat = v_t / max(1 - self.beta_2 ** (self.t + 2) + effective_eps, 1e-4)

            master_param_tensor.data = master_param_tensor.data * (1 - self.get_lr(self.t + 1) * self.weight_decay)

            master_param_tensor.data = master_param_tensor.data - self.get_lr(self.t + 1) * m_hat / (xp.sqrt(v_hat) + effective_eps).clip(min=1e-4)


            param['m_t'] = m_t
            param['v_t'] = v_t

            # sync master param to model param
            if self.mixed_precision:
                param['param'].data = master_param_tensor.data.astype(self.model_dtype)

        self.t += 1

class Standard(Optimizer):
    def __init__(self, params, lr: LRScheduler | float = 1e-3, clip_norm=1.0):
        super().__init__(params, lr=lr, clip_norm=clip_norm)
        self.clip_norm = clip_norm

    def step(self):
        """SGD without momentum, equivalent to torch.optim.SGD (no momentum, no weight decay).
        Supports optional global gradient clipping via ``clip_norm``.
        """
        # Increase timestep
        self.t += 1
        lr_t = self.get_lr(self.t)
        lr_t = 0.0005

        for param in self.params.values():
            if param['param'].grad is None:
                continue

            grad = param['param'].grad
            grad = self._clip_norm(grad)
            grad = self.reduce_like(grad, param['param'].data.shape)

            param['param'].data = param['param'].data - lr_t * grad.data
            del grad


        # # Compute global grad-norm for clipping (L2)
        # total_norm = 0.0
        # for param in self.params.values():
        #     if param['param'].grad is None:
        #         continue
        #     g = param['param'].grad.data
        #     total_norm += (g ** 2).sum()
        # total_norm = xp.sqrt(total_norm)

        # clip_coef = 1.0
        # if total_norm > self.clip_norm:
        #     clip_coef = self.clip_norm / (total_norm + 1e-6)

        # # Parameter update
        # for param in self.params.values():
        #     p_tensor = param['param']
        #     grad = p_tensor.grad
        #     if grad is None:
        #         continue
        #     # Broadcast-reduce if shapes mismatch (should rarely happen here)
        #     if grad.data.shape != p_tensor.data.shape:
        #         grad = self.reduce_like(grad, p_tensor.data.shape)
        #     grad_data = grad.data * clip_coef
        #     p_tensor.data -= lr_t * grad_data

