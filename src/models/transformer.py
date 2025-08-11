from src.utils.backend import xp
from src.core.module import Module
from src.core.tensor import Tensor

class Transformer(Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, module_dict=None):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.mlp_ratio = mlp_ratio

        # Shape-check utility that raises on mismatch
        def _check_shape(name, actual, expected):
            def _shape_of(obj):
                try:
                    return tuple(obj.shape)
                except Exception:
                    return None
            actual_shape = _shape_of(actual)
            if expected is not None and actual_shape != expected:
                raise ValueError(f"Shape mismatch for {name}: expected {expected}, got {actual_shape}")
        self._check_shape = _check_shape

        if self.d_head * n_heads != d_model:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        # (self, in_features, out_features, module_dict, layer_dict, bias=True)
        self.q = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="q")
        self.k = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="k")
        self.v = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="v")
        self.o = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="o")

        self.proj_up = self.linear(d_model, d_model * mlp_ratio, module_dict=module_dict, layer_type="linear", name="proj_up")
        self.proj_down = self.linear(d_model * mlp_ratio, d_model, module_dict=module_dict, layer_type="linear", name="proj_down")

        self.ln1 = self.layer_norm(axis=-1, module_dict=module_dict, name="ln1")
        self.ln2 = self.layer_norm(axis=-1, module_dict=module_dict, name="ln2")

    
    def attend(self, x: Tensor):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        self._check_shape("attend/x", x, expected=(B, T, self.d_model))

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        self._check_shape("q linear", q, expected=(B, T, self.d_model))
        self._check_shape("k linear", k, expected=(B, T, self.d_model))
        self._check_shape("v linear", v, expected=(B, T, self.d_model))

        q  = q.reshape((B, T, self.n_heads, self.d_head))
        k  = k.reshape((B, T, self.n_heads, self.d_head))
        v  = v.reshape((B, T, self.n_heads, self.d_head))
        self._check_shape("q reshape", q, expected=(B, T, self.n_heads, self.d_head))
        self._check_shape("k reshape", k, expected=(B, T, self.n_heads, self.d_head))
        self._check_shape("v reshape", v, expected=(B, T, self.n_heads, self.d_head))

        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))
        self._check_shape("q transpose (B,H,T,Dh)", q, expected=(B, self.n_heads, T, self.d_head))
        self._check_shape("k transpose (B,H,T,Dh)", k, expected=(B, self.n_heads, T, self.d_head))
        self._check_shape("v transpose (B,H,T,Dh)", v, expected=(B, self.n_heads, T, self.d_head))

        kt = k.transpose((0, 1, 3, 2))
        self._check_shape("k^T (B,H,Dh,T)", kt, expected=(B, self.n_heads, self.d_head, T))

        atten_scores = q @ kt * (1 / (self.d_head ** 0.5))
        self._check_shape("attention scores", atten_scores, expected=(B, self.n_heads, T, T))

        # Masking
        casual_mask = xp.triu(xp.ones((T, T)) * -xp.inf, k=1).astype(xp.float16)
        atten_scores.data += casual_mask
        self._check_shape("attention scores masked", atten_scores, expected=(B, self.n_heads, T, T))

        atten_probs = self.softmax(atten_scores, axis=3)
        self._check_shape("attention probs", atten_probs, expected=(B, self.n_heads, T, T))

        output = atten_probs @ v
        self._check_shape("attended values", output, expected=(B, self.n_heads, T, self.d_head))

        output = output.transpose((0, 2, 1, 3))
        self._check_shape("attended transpose (B,T,H,Dh)", output, expected=(B, T, self.n_heads, self.d_head))

        output = output.reshape((B, T, -1))
        self._check_shape("attended merge heads", output, expected=(B, T, self.n_heads * self.d_head))

        output = self.o(output)
        self._check_shape("output projection", output, expected=(B, T, self.d_model))

        return output
    
    def forward(self, x: Tensor):
        # Pre-attention LN
        B, T, _ = x.shape
        self._check_shape("block input", x, expected=(B, T, self.d_model))

        residual = x
        x = self.ln1(x)
        self._check_shape("ln1", x, expected=(B, T, self.d_model))

        atten_out = self.attend(x)
        self._check_shape("attention out", atten_out, expected=(B, T, self.d_model))

        x = residual + atten_out
        self._check_shape("residual add 1", x, expected=(B, T, self.d_model))

        # MLP
        residual = x
        x = self.ln2(x)
        self._check_shape("ln2", x, expected=(B, T, self.d_model))

        x_mlp = self.proj_up(x)
        self._check_shape("mlp up", x_mlp, expected=(B, T, self.d_model * self.mlp_ratio))

        x_mlp = self.gelu(x_mlp)
        self._check_shape("mlp gelu", x_mlp, expected=(B, T, self.d_model * self.mlp_ratio))

        x_mlp = self.proj_down(x_mlp)
        self._check_shape("mlp down", x_mlp, expected=(B, T, self.d_model))

        x = residual + x_mlp
        self._check_shape("residual add 2", x, expected=(B, T, self.d_model))

        return x
    
    def __call__(self, x: Tensor):
        return self.forward(x)

class Embedding(Module):
    def __init__(self, vocab_size, d_model, max_seq_len, module_dict=None, layer_dict=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W = Tensor(xp.random.randn(vocab_size, d_model).astype(xp.float16), requires_grad=True)
        self.pe = Tensor(xp.random.randn(max_seq_len, d_model).astype(xp.float16), requires_grad=True)

        self.register_parameter(param=self.W, module_dict=module_dict, layer_type="embedding", layer_dict=layer_dict, name="embed")
        self.register_parameter(param=self.pe, module_dict=module_dict, layer_type="embedding", layer_dict=layer_dict, name="pe")
    
    def get_sentence_embedding(self, idx):
        idx = idx.astype(xp.int32)
        B, T = idx.shape
        embed_vectors = self.W[idx]
        pe_slice = self.pe[:T][None, :, :]
        output = embed_vectors + pe_slice

        return output



