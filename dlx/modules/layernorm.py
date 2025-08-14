from ..nn.tensor import Tensor
from ..nn.module import Module
from ..utils.backend import xp

class LayerNorm(Module):
    def __init__(self, axis, module_dict, layer_dict, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.axis = axis
        self.gamma = None
        self.beta = None
        self.initialized = False
        self.module_dict = module_dict
        self.layer_dict = layer_dict

    def forward(self, x: Tensor, axis=-1):
        if not self.initialized:
            dim = x.shape[self.axis]
            self.gamma = Tensor(xp.ones(dim), requires_grad=True, name='gamma')
            self.beta = Tensor(xp.zeros(dim), requires_grad=True, name='beta')
            self.register_parameter(param=self.gamma, module_dict=self.module_dict, layer_type=self.layer_dict["type"], layer_dict=self.layer_dict, name="gamma")
            self.register_parameter(param=self.beta, module_dict=self.module_dict, layer_type=self.layer_dict["type"], layer_dict=self.layer_dict, name="beta")
            self.initialized = True
            
        mean = x.mean(axis=axis, keepdims=True)
        var = ((x - mean).pow(2)).mean(axis=axis, keepdims=True)
        out = (x - mean) / (var + self.eps).pow(0.5)
        return out * self.gamma + self.beta
    
    def __call__(self, x):
        return self.forward(x)