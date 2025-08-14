from ..nn.tensor import Tensor
from ..nn.module import Module
from ..utils.backend import xp


class Linear(Module):
    def __init__(self, in_features, out_features, module_dict, layer_dict=None, use_bias=True, name=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.module_dict = module_dict
        self.layer_dict = layer_dict
        self.use_bias = use_bias
        self.name = f"{name}_" if name is not None else ""

        # Initialize weights and register them as parameters
        self.weight = Tensor(self.xavier_uniform((in_features, out_features)), requires_grad=True)

        self.register_parameter(param=self.weight, module_dict=self.module_dict, layer_type=self.layer_dict["type"], layer_dict=self.layer_dict, name=f"{self.name}weight")
        
        if self.use_bias:
            self.bias = Tensor(xp.zeros(out_features), requires_grad=True)
            self.register_parameter(param=self.bias, module_dict=self.module_dict, layer_type=self.layer_dict["type"], layer_dict=self.layer_dict, name=f"{self.name}bias")

    @property
    def shape(self):
        return self.weight.shape

    def forward(self, x):
        out = x @ self.weight
        if self.use_bias:
            out = out + self.bias
        return out
    
    def __call__(self, x):
        return self.forward(x)