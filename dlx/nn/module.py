import os
import numpy as np
from ..utils.backend import xp
from .tensor import Tensor

class Module:
    def __init__(self, architecture={}):
        self.architecture = architecture

        """
        architecture:
        {
            "module_type_1": {
                "type": "module_type_1",
                "layers": [{"type": "layer_type_1", "name": "layer_name_1", "params": [{"type": "param_type_1", "name": "param_name_1", "param": "param_value_1"}]}]
            },
            "module_type_2": {
                "type": "module_type_2",
                "layers": [{"type": "layer_type_2", "name": "layer_name_2", "params": [{"type": "param_type_2", "name": "param_name_2", "param": "param_value_2"}]}]
            }
        }
        """

        self.is_training = True
        self.is_cuda = xp.__name__ == "cupy"

    def __str__(self):
        lines = ["Architecture:"]
        for module_name, module in self.architecture.items():
            lines.append(f"  {module_name} ({module['type']}):")
            for layer in module["layers"]:
                lines.append(f"    {layer['name']} ({layer['type']}):")
                for p in layer["params"]:
                    t = p["param"]
                    lines.append(f"      {p['name']}: shape={tuple(t.shape)}, dtype={t.dtype}")
        return "\n".join(lines)
    
    @property
    def num_parameters(self):
        return sum(p.n_params for p in self.parameters().values())

    def _build(self, input_shape: tuple):
        dummy_input = Tensor(xp.zeros(input_shape), requires_grad=False)
        self.forward(dummy_input)

    def eval(self, X: Tensor, y: Tensor, loss_fn):
        self.is_training = False
        try:
            y_hat = self.forward(X)
            loss = loss_fn(y_hat, y)
            print(f"Loss: {loss.data}")
        except Exception as e:
            print(f"Error: {e}")
        finally:    
            self.is_training = True

    def forward(self, x):
        raise NotImplementedError("Child class must implement forward()")
    
    def parameters(self):
        params = {}

        for module in self.architecture.values():
            for layer in module["layers"]:
                for param in layer["params"]:
                    params[param["name"]] = param["param"]
        
        return params
    
    def get_param_name(self, module_dict, layer_type, name=""):
        return f"{module_dict['type']}_{self.get_module_type_index(module_dict['type'])}_{layer_type}_{self.get_layer_type_index(module_dict['layers'], layer_type)}_{name}"

    def get_layer_type_index(self, layers, layer_type):
        return sum(1 for l in layers if l["type"] == layer_type)
    
    def get_module_type_index(self, module_type):
        return sum(1 for m in self.architecture.values() if m["type"] == module_type)
    
    def register_module(self, module_type: str):
        module_name = f"{module_type}_{self.get_module_type_index(module_type)}"
        if module_name in self.architecture:
            raise ValueError(f"Module {module_name} already registered")
        
        self.architecture[module_name] = {"type": module_type, "layers": []}
        return self.architecture[module_name]

    def register_layer(self, layer_type, module_dict, name):
        module_dict["layers"].append({"type": layer_type, "name": name, "params": []})
        return module_dict["layers"][-1]
    
    def register_parameter(self, param, module_dict,layer_type, layer_dict, name):
        full_name = self.get_param_name(module_dict, layer_type, name)
        layer_dict["params"].append({"type": param, "name": full_name, "param": param})
        

    def layer_norm(self, axis: int = -1, module_type: str = "layer_norm", layer_type: str = "layernorm", name: str | None = None, module_dict=None, layer_dict=None, eps: float = 1e-5):
       from ..modules.layernorm import LayerNorm
       if module_dict is None:
           module_dict = self.register_module(module_type)
       layer_dict = self.register_layer(layer_type, module_dict, name)
       layer = LayerNorm(axis, module_dict, layer_dict, eps=eps)
       return layer

    def linear(self, in_features, out_features, use_bias=True, module_type="linear", layer_type="linear", name=None, module_dict=None):
        from ..modules.linear import Linear
        if module_dict is None:
            module_dict = self.register_module(module_type)
        layer_dict = self.register_layer(layer_type, module_dict, name)
        layer = Linear(in_features, out_features, module_dict, layer_dict, use_bias, name)
        return layer
        
    def transformer(self, d_model, n_heads, mlp_ratio=4, module_type="transformer"):
        from ..modules.transformer import Transformer
        module_dict = self.register_module(module_type)
        layer = Transformer(d_model, n_heads, mlp_ratio, module_dict)
        return layer
    
    def embedding(self, vocab_size, d_model, max_seq_len, module_type="embedding", layer_type="embedding", name="embedding"):
        from ..modules.transformer import Embedding
        module_dict = self.register_module(module_type)
        layer_dict = self.register_layer(layer_type, module_dict, name)
        layer = Embedding(vocab_size, d_model, max_seq_len, module_dict, layer_dict)
        return layer

    def xavier_uniform(self, shape):
        fan_in, fan_out = shape[-1], shape[-2] if len(shape) > 1 else (shape[0], shape[0])
        limit = xp.sqrt(6 / (fan_in + fan_out))
        return xp.random.uniform(-limit, limit, size=shape)
    
    # WRAPPER FUNCTIONS -----------------------------------------------
    def dropout(self, x: Tensor, p=0.1):
        return x._dropout(p, train=self.is_training)
    
    def sigmoid(self, x):
        return x._sigmoid()
    
    def softmax(self, x, axis=1):
        return x._softmax(axis=axis)


    def gelu(self, x: Tensor):
        return x._gelu()
    
    def relu(self, x: Tensor):
        return x._relu()
        
    def softmax(self, x, axis=1):
        return x._softmax(axis=axis)

    def zero_grad(self):
        for param in self.parameters().values():
            if param.grad is not None:
                param.grad = None




