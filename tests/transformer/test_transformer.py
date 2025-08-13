import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.transformer import Transformer, Embedding
from src.models.torch_transformer import TransformerBlock
from src.core.tensor import Tensor
from src.core.optim import SGD
from src.core.losses import CrossEntropyWithLogits
from src.core.module import Module
from src.utils.backend import xp, set_seed

set_seed(42)


class MyTransformer(Module):
    def __init__(self, d_model, n_heads, vocab_size):
        super().__init__()
        self.head1 = self.transformer(d_model=d_model, n_heads=n_heads)
        self.project = self.linear(d_model, vocab_size, name="project")

    def forward(self, x):
        x = self.head1(x)
        x = self.project(x)
        return x


class TorchTransformer(nn.Module):
    def __init__(self, d_model, n_heads, mlp_dim, vocab_size):
        super().__init__()
        self.head1 = TransformerBlock(d_model=d_model, num_heads=n_heads, mlp_dim=mlp_dim)
        self.project = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.head1(x)
        x = self.project(x)
        return x


class TestTransformer(unittest.TestCase):
    # Model configuration
    d_model = 16
    n_heads = 2
    mlp_dim = d_model * 4
    vocab_size = 10
    # Test data
    _data = xp.random.randn(1, 10, d_model).astype(xp.float32)
    _tgt = xp.random.randint(0, 10, (1, 10)).astype(xp.int32)
    data = Tensor(_data, requires_grad=False)
    tgt = Tensor(_tgt, requires_grad=False)
    data_pt = torch.tensor(_data, requires_grad=False, dtype=torch.float32)
    tgt_pt = torch.tensor(_tgt, requires_grad=False, dtype=torch.int32).long()
    
    # Parameter mapping for weight transposition
    WEIGHT_PARAMS = ['q_weight', 'k_weight', 'v_weight', 'o_weight', 
                     'proj_up_weight', 'proj_down_weight', 'project_weight']
    BIAS_PARAMS = ['q_bias', 'k_bias', 'v_bias', 'o_bias', 
                   'ln1_weight', 'ln1_bias', 'ln2_weight', 'ln2_bias',
                   'proj_up_bias', 'proj_down_bias', 'project_bias']

    def assert_close(self, a, b, atol=5e-3):
        """Assert that two arrays are close within tolerance."""
        is_all_close = xp.allclose(a, b, atol=atol)
        if not is_all_close:
            print(f"===================== FAILED ======================")
            print(f"a shape: {a.shape}")
            print(f"b shape: {b.shape}")
            print(f"a: {a}")
            print(f"b: {b}")
            print(f"Max diff: {xp.max(xp.abs(a - b))}")
            print(f"===================================================")
        self.assertTrue(is_all_close)

    def assert_grads_close(self, a, b, atol=1e-5):
        me = xp.mean(xp.abs(a - b))
        if me > atol:
            print(f"===================== FAILED ======================")
            print(f"ME: {me}")
            # print(f"a: {a}")
            # print(f"b: {b}")
            print(f"===================================================")
        self.assertTrue(me < atol)

    def setUp(self):
        """Set up models and copy weights from PyTorch to custom model."""
        self.model, self.model_pt = self.create_models()
        self.set_states(self.model, self.model_pt)

    def get_pt_params(self, model_pt):
        """Extract parameters from PyTorch model as dictionary."""
        return {
            'q_weight': model_pt.head1.q.weight.detach().numpy(),
            'q_bias': model_pt.head1.q.bias.detach().numpy(),
            'k_weight': model_pt.head1.k.weight.detach().numpy(),
            'k_bias': model_pt.head1.k.bias.detach().numpy(),
            'v_weight': model_pt.head1.v.weight.detach().numpy(),
            'v_bias': model_pt.head1.v.bias.detach().numpy(),
            'o_weight': model_pt.head1.o.weight.detach().numpy(),
            'o_bias': model_pt.head1.o.bias.detach().numpy(),
            'ln1_weight': model_pt.head1.ln1.weight.detach().numpy(),
            'ln1_bias': model_pt.head1.ln1.bias.detach().numpy(),
            'ln2_weight': model_pt.head1.ln2.weight.detach().numpy(),
            'ln2_bias': model_pt.head1.ln2.bias.detach().numpy(),
            'proj_up_weight': model_pt.head1.proj_up.weight.detach().numpy(),
            'proj_up_bias': model_pt.head1.proj_up.bias.detach().numpy(),
            'proj_down_weight': model_pt.head1.proj_down.weight.detach().numpy(),
            'proj_down_bias': model_pt.head1.proj_down.bias.detach().numpy(),
            'project_weight': model_pt.project.weight.detach().numpy(),
            'project_bias': model_pt.project.bias.detach().numpy()
        }

    def get_my_params(self, model, is_copy=True):
        """Extract parameters from custom model as dictionary."""
        params = {
            'q_weight': model.head1.q.weight.data,
            'q_bias': model.head1.q.bias.data,
            'k_weight': model.head1.k.weight.data,
            'k_bias': model.head1.k.bias.data,
            'v_weight': model.head1.v.weight.data,
            'v_bias': model.head1.v.bias.data,
            'o_weight': model.head1.o.weight.data,
            'o_bias': model.head1.o.bias.data,
            'ln1_weight': model.head1.ln1.gamma.data,
            'ln1_bias': model.head1.ln1.beta.data,
            'ln2_weight': model.head1.ln2.gamma.data,
            'ln2_bias': model.head1.ln2.beta.data,
            'proj_up_weight': model.head1.proj_up.weight.data,
            'proj_up_bias': model.head1.proj_up.bias.data,
            'proj_down_weight': model.head1.proj_down.weight.data,
            'proj_down_bias': model.head1.proj_down.bias.data,
            'project_weight': model.project.weight.data,
            'project_bias': model.project.bias.data
        }
        
        if is_copy:
            return {k: v.copy() for k, v in params.items()}
        return params

    def get_pt_grads(self, model_pt):
        """Extract gradients from PyTorch model as dictionary."""
        return {
            'q_weight': model_pt.head1.q.weight.grad.detach().numpy() if model_pt.head1.q.weight.grad is not None else None,
            'q_bias': model_pt.head1.q.bias.grad.detach().numpy() if model_pt.head1.q.bias.grad is not None else None,
            'k_weight': model_pt.head1.k.weight.grad.detach().numpy() if model_pt.head1.k.weight.grad is not None else None,
            'k_bias': model_pt.head1.k.bias.grad.detach().numpy() if model_pt.head1.k.bias.grad is not None else None,
            'v_weight': model_pt.head1.v.weight.grad.detach().numpy() if model_pt.head1.v.weight.grad is not None else None,
            'v_bias': model_pt.head1.v.bias.grad.detach().numpy() if model_pt.head1.v.bias.grad is not None else None,
            'o_weight': model_pt.head1.o.weight.grad.detach().numpy() if model_pt.head1.o.weight.grad is not None else None,
            'o_bias': model_pt.head1.o.bias.grad.detach().numpy() if model_pt.head1.o.bias.grad is not None else None,
            'ln1_weight': model_pt.head1.ln1.weight.grad.detach().numpy() if model_pt.head1.ln1.weight.grad is not None else None,
            'ln1_bias': model_pt.head1.ln1.bias.grad.detach().numpy() if model_pt.head1.ln1.bias.grad is not None else None,
            'ln2_weight': model_pt.head1.ln2.weight.grad.detach().numpy() if model_pt.head1.ln2.weight.grad is not None else None,
            'ln2_bias': model_pt.head1.ln2.bias.grad.detach().numpy() if model_pt.head1.ln2.bias.grad is not None else None,
            'proj_up_weight': model_pt.head1.proj_up.weight.grad.detach().numpy() if model_pt.head1.proj_up.weight.grad is not None else None,
            'proj_up_bias': model_pt.head1.proj_up.bias.grad.detach().numpy() if model_pt.head1.proj_up.bias.grad is not None else None,
            'proj_down_weight': model_pt.head1.proj_down.weight.grad.detach().numpy() if model_pt.head1.proj_down.weight.grad is not None else None,
            'proj_down_bias': model_pt.head1.proj_down.bias.grad.detach().numpy() if model_pt.head1.proj_down.bias.grad is not None else None,
            'project_weight': model_pt.project.weight.grad.detach().numpy() if model_pt.project.weight.grad is not None else None,
            'project_bias': model_pt.project.bias.grad.detach().numpy() if model_pt.project.bias.grad is not None else None,
        }

    def get_my_grads(self, model, is_copy=True):
        """Extract gradients from custom model as dictionary."""
        grads = {
            'q_weight': None if model.head1.q.weight.grad is None else model.head1.q.weight.grad.data,
            'q_bias': None if model.head1.q.bias.grad is None else model.head1.q.bias.grad.data,
            'k_weight': None if model.head1.k.weight.grad is None else model.head1.k.weight.grad.data,
            'k_bias': None if model.head1.k.bias.grad is None else model.head1.k.bias.grad.data,
            'v_weight': None if model.head1.v.weight.grad is None else model.head1.v.weight.grad.data,
            'v_bias': None if model.head1.v.bias.grad is None else model.head1.v.bias.grad.data,
            'o_weight': None if model.head1.o.weight.grad is None else model.head1.o.weight.grad.data,
            'o_bias': None if model.head1.o.bias.grad is None else model.head1.o.bias.grad.data,
            'ln1_weight': None if model.head1.ln1.gamma.grad is None else model.head1.ln1.gamma.grad.data,
            'ln1_bias': None if model.head1.ln1.beta.grad is None else model.head1.ln1.beta.grad.data,
            'ln2_weight': None if model.head1.ln2.gamma.grad is None else model.head1.ln2.gamma.grad.data,
            'ln2_bias': None if model.head1.ln2.beta.grad is None else model.head1.ln2.beta.grad.data,
            'proj_up_weight': None if model.head1.proj_up.weight.grad is None else model.head1.proj_up.weight.grad.data,
            'proj_up_bias': None if model.head1.proj_up.bias.grad is None else model.head1.proj_up.bias.grad.data,
            'proj_down_weight': None if model.head1.proj_down.weight.grad is None else model.head1.proj_down.weight.grad.data,
            'proj_down_bias': None if model.head1.proj_down.bias.grad is None else model.head1.proj_down.bias.grad.data,
            'project_weight': None if model.project.weight.grad is None else model.project.weight.grad.data,
            'project_bias': None if model.project.bias.grad is None else model.project.bias.grad.data,
        }

        if is_copy:
            return {k: (None if v is None else v.copy()) for k, v in grads.items()}
        return grads

    def models_are_equal(self):
        """Check if custom and PyTorch models have identical parameters."""
        my_params = self.get_my_params(self.model)
        pt_params = self.get_pt_params(self.model_pt)
        
        all_equal = True
        for param_name in my_params.keys():
            try:
                pt_param = pt_params[param_name].T if param_name in self.WEIGHT_PARAMS else pt_params[param_name]
                self.assert_close(my_params[param_name], pt_param)
            except AssertionError:
                print(f"✗ {param_name} does not match")
                all_equal = False
        
        return all_equal

    def grads_are_equal(self):
        """Check if custom and PyTorch models have identical gradients."""
        my_grads = self.get_my_grads(self.model)
        pt_grads = self.get_pt_grads(self.model_pt)

        all_equal = True
        for param_name in my_grads.keys():
            try:
                my_grad = my_grads[param_name]
                pt_grad = pt_grads[param_name].T if param_name in self.WEIGHT_PARAMS else pt_grads[param_name]

                if my_grad is None and pt_grad is None:
                    continue
                if my_grad is None or pt_grad is None:
                    raise AssertionError("One of the gradients is None")

                self.assert_grads_close(my_grad, pt_grad)
            except AssertionError:
                print(f"✗ grad mismatch: {param_name}")
                all_equal = False
        return all_equal

    def set_states(self, model, model_pt):
        """Copy weights from PyTorch model to custom model."""
        def safe_update(tensor, new_data, name):
            if tensor.data.shape != new_data.shape:
                raise ValueError(f"Shape mismatch for {name}: expected {tensor.data.shape}, got {new_data.shape}")
            tensor.data = new_data

        # Get PyTorch parameters
        pt_params = self.get_pt_params(model_pt)
        
        # Update attention components
        safe_update(model.head1.q.weight, pt_params['q_weight'].T, "q.weight")
        safe_update(model.head1.q.bias, pt_params['q_bias'], "q.bias")
        safe_update(model.head1.k.weight, pt_params['k_weight'].T, "k.weight")
        safe_update(model.head1.k.bias, pt_params['k_bias'], "k.bias")
        safe_update(model.head1.v.weight, pt_params['v_weight'].T, "v.weight")
        safe_update(model.head1.v.bias, pt_params['v_bias'], "v.bias")
        safe_update(model.head1.o.weight, pt_params['o_weight'].T, "o.weight")
        safe_update(model.head1.o.bias, pt_params['o_bias'], "o.bias")

        # Update layer normalization components
        safe_update(model.head1.ln1.gamma, pt_params['ln1_weight'], "ln1.gamma")
        safe_update(model.head1.ln1.beta, pt_params['ln1_bias'], "ln1.beta")
        safe_update(model.head1.ln2.gamma, pt_params['ln2_weight'], "ln2.gamma")
        safe_update(model.head1.ln2.beta, pt_params['ln2_bias'], "ln2.beta")
        
        # Update MLP components
        safe_update(model.head1.proj_up.weight, pt_params['proj_up_weight'].T, "proj_up.weight")
        safe_update(model.head1.proj_up.bias, pt_params['proj_up_bias'], "proj_up.bias")
        safe_update(model.head1.proj_down.weight, pt_params['proj_down_weight'].T, "proj_down.weight")
        safe_update(model.head1.proj_down.bias, pt_params['proj_down_bias'], "proj_down.bias")

        # Update projection layer
        safe_update(model.project.weight, pt_params['project_weight'].T, "project.weight")
        safe_update(model.project.bias, pt_params['project_bias'], "project.bias")

    def create_models(self):
        """Create and return custom and PyTorch models."""
        model = MyTransformer(self.d_model, self.n_heads, self.vocab_size)
        model._build(self.data.shape)
        model.zero_grad()
        model_pt = TorchTransformer(self.d_model, self.n_heads, self.mlp_dim, self.vocab_size)
        return model, model_pt

    def test_parameters_same(self):
        """Test that model parameters are identical after weight copying."""
        self.assertTrue(self.models_are_equal())

    def test_transformer_forward(self):
        """Test that forward passes produce identical outputs."""
        my_forward = self.model.forward(self.data).data
        pt_forward = self.model_pt(self.data_pt).detach().numpy()
        self.assert_close(my_forward, pt_forward)

    def test_cross_entropy_with_logits(self):
        """Test that cross entropy with logits produces identical outputs."""
        my_loss = np.array([float(CrossEntropyWithLogits(self.model.forward(self.data), self.tgt).data)])
        pt_logits = self.model_pt(self.data_pt)          # (B, T, V)
        pt_loss = F.cross_entropy(
            pt_logits.reshape(-1, pt_logits.size(-1)),   # (B*T, V)
            self.tgt_pt.reshape(-1),                     # (B*T,)

        )
        pt_loss = np.array([float(pt_loss.detach().numpy())])
        self.assert_close(my_loss, pt_loss)


    def test_transformer_backward(self):
        my_loss = CrossEntropyWithLogits(self.model.forward(self.data), self.tgt)
        my_loss.backward()
        pt_loss = F.cross_entropy(self.model_pt(self.data_pt), self.tgt_pt)
        pt_loss.backward()
        self.assertTrue(self.grads_are_equal())


if __name__ == "__main__":
    unittest.main()


