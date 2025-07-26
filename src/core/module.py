from src.utils.backend import xp
from src.core.tensor import Tensor

class Module:
    def __init__(self):
        # pipeline is a list of high level dicts of semantic names of the "layers" in the model 
        # (e.g. {name: "embedding", type: "embedding"}, {name: "head1", type: "transformer"}, {name: "project", type: "linear"})
        self.pipeline = []

        self._modules = {}
        self._parameters = {}
        self.is_training = True

        self.is_cuda = xp.__name__ == "cupy"

    
    @property
    def num_parameters(self):
        return sum(p.n_params for p in self.parameters())

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
        params = []
        
        for param in self._parameters.values():
            if param is not None:
                params.append(param)
        
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
        
        return params
    
    def get_param_name(self, module_type, layer_type, name="", is_layer=True):
        index = self.get_layer_type_index(layer_type)
        if is_layer:
            layer_name = f"{module_type}_{index}_{layer_type}"
        else:
            layer_name = f"{module_type}_{index}"

        if name:
            layer_name += f"_{name}"

        return layer_name

    def get_layer_type_index(self, layer_type):
        return sum(1 for l in self.pipeline if l["type"] == layer_type)
    
    def add_module(self, name, module, module_type):
        self.pipeline.append({"name": name, "type": module_type})
        self._modules[name] = module
    
    def register_parameter(self, param, module_type, layer_type=None, name=None):
        layer_name = self.get_param_name(module_type, layer_type, name)

        if layer_name in self._parameters:
            raise ValueError(f"Parameter {layer_name} already registered")
        
        self._parameters[layer_name] = param
        if param is not None:
            param.name = layer_name

    def layer_norm(self, axis=-1, module_type="linear", layer_type="layernorm", name=None):
        layer = LayerNorm(axis)
        self.add_module(self.get_param_name(module_type, layer_type, name, is_layer=False), layer, module_type)
        return layer

    def linear(self, in_features, out_features, bias=True, module_type="linear", layer_type="linear", name=None):
        layer = Linear(in_features, out_features, bias)
        self.add_module(self.get_param_name(module_type, layer_type, name), layer, module_type)
        return layer
        
    def transformer(self, d_model, n_heads, pad_idx=0, mlp_ratio=4, module_type="transformer", layer_type="transformer", name=None):
        from src.models.transformer import Transformer
        layer = Transformer(d_model, n_heads, pad_idx, mlp_ratio)
        self.add_module(self.get_param_name(module_type, layer_type, name, is_layer=False), layer, module_type)
        return layer
    
    def embedding(self, vocab_size, d_model, max_seq_len, pad_idx=0, module_type="embedding", layer_type="embedding", name=None):
        from src.models.transformer import Embedding
        layer = Embedding(vocab_size, d_model, max_seq_len, pad_idx)
        self.add_module(self.get_param_name(module_type, layer_type, name, is_layer=False), layer, module_type)
        return layer

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
        for param in self.parameters():
            if param.grad is not None:
                param.grad = None


class LayerNorm(Module):
    def __init__(self, axis, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.axis = axis
        self.gamma = None
        self.beta = None
        self.initialized = False

    def forward(self, x: Tensor, axis=-1):
        if not self.initialized:
            dim = x.shape[self.axis]
            self.gamma = Tensor(xp.ones(dim), requires_grad=True, name='gamma')
            self.beta = Tensor(xp.zeros(dim), requires_grad=True, name='beta')
            self.register_parameter(param=self.gamma, module_type="linear", layer_type="layer_norm", name="gamma")
            self.register_parameter(param=self.beta, module_type="linear", layer_type="layer_norm", name="beta")
            self.initialized = True
            
        mean = x.mean(axis=axis, keepdims=True)
        var = ((x - mean).pow(2)).mean(axis=axis, keepdims=True)
        out = (x - mean) / (var + self.eps).pow(0.5)
        return out * self.gamma + self.beta
    
    def __call__(self, x):
        return self.forward(x)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and register them as parameters
        self.weight = Tensor(xp.random.randn(in_features, out_features) * 0.01)
        self.register_parameter(param=self.weight, module_type="linear", layer_type="linear", name="weight")
        
        if bias:
            self.bias = Tensor(xp.random.randn(out_features) * 0.01)
            self.register_parameter(param=self.bias, module_type="linear", layer_type="linear", name="bias")
        # else:
        #     self.register_parameter(param=None, module_type="linear", layer_type=None, name="bias")

    @property
    def shape(self):
        return self.weight.shape

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out
    
    def __call__(self, x):
        return self.forward(x)
