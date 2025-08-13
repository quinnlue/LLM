


import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F

from src.core.tensor import Tensor
from src.core.module import Module as M
from src.utils.backend import xp

class TestAttend(unittest.TestCase):
    def assert_close(self, a, b, atol=5e-3):
        is_all_close = xp.allclose(a, b, atol=atol)
        max_diff = xp.max(xp.abs(a - b))
        print(f"MAX DIFF: {max_diff}")
        self.assertTrue(is_all_close)

    def setUp(self):
        self.d_model = 8
        self.n_heads = 2
        self.d_head = self.d_model // self.n_heads

        self.q_weight = xp.random.randn(self.d_model, self.d_model).astype(xp.float32)
        self.k_weight = xp.random.randn(self.d_model, self.d_model).astype(xp.float32)
        self.v_weight = xp.random.randn(self.d_model, self.d_model).astype(xp.float32)
        self.o_weight = xp.random.randn(self.d_model, self.d_model).astype(xp.float32)

        self.q_bias = xp.random.randn(self.d_model).astype(xp.float32)
        self.k_bias = xp.random.randn(self.d_model).astype(xp.float32)
        self.v_bias = xp.random.randn(self.d_model).astype(xp.float32)
        self.o_bias = xp.random.randn(self.d_model).astype(xp.float32)

        self.gamma_1 = xp.ones(self.d_model).astype(xp.float32)
        self.beta_1 = xp.zeros(self.d_model).astype(xp.float32)
        self.gamma_2 = xp.ones(self.d_model).astype(xp.float32)
        self.beta_2 = xp.zeros(self.d_model).astype(xp.float32)

        self.proj_up_weight = xp.random.randn(self.d_model, self.d_model).astype(xp.float32)
        self.proj_up_bias = xp.random.randn(self.d_model).astype(xp.float32)
        self.proj_down_weight = xp.random.randn(self.d_model, self.d_model).astype(xp.float32)
        self.proj_down_bias = xp.random.randn(self.d_model).astype(xp.float32)

        self.x = xp.random.randn(1, 5, self.d_model).astype(xp.float32)

        def my_attend(x):
            B, T, _ = x.shape

            q_weight = Tensor(self.q_weight.copy(), requires_grad=True)
            k_weight = Tensor(self.k_weight.copy(), requires_grad=True)
            v_weight = Tensor(self.v_weight.copy(), requires_grad=True)
            o_weight = Tensor(self.o_weight.copy(), requires_grad=True)

            q_bias = Tensor(self.q_bias.copy(), requires_grad=True)
            k_bias = Tensor(self.k_bias.copy(), requires_grad=True)
            v_bias = Tensor(self.v_bias.copy(), requires_grad=True)
            o_bias = Tensor(self.o_bias.copy(), requires_grad=True)

            q = x @ q_weight + q_bias
            k = x @ k_weight + k_bias
            v = x @ v_weight + v_bias

            q  = q.reshape((B, T, self.n_heads, self.d_head))
            k  = k.reshape((B, T, self.n_heads, self.d_head))
            v  = v.reshape((B, T, self.n_heads, self.d_head))

            q = q.transpose((0, 2, 1, 3))
            k = k.transpose((0, 2, 1, 3))
            v = v.transpose((0, 2, 1, 3))
            kt = k.transpose((0, 1, 3, 2))

            atten_scores = q @ kt * (1 / (self.d_head ** 0.5))
            # TODO: remove hard coded float32
            casual_mask = xp.triu(xp.ones((T, T)) * -xp.inf, k=1).astype(xp.float32)
            atten_scores = atten_scores + casual_mask

            atten_probs = atten_scores._softmax(axis=3)

            output = atten_probs @ v

            output = output.transpose((0, 2, 1, 3))

            output = output.reshape((B, T, -1))

            output = output @ o_weight + o_bias

            return output, [q_weight, k_weight, v_weight, o_weight, q_bias, k_bias, v_bias, o_bias]
        
        def pt_attend(x):
            B, T, _ = x.size()

            q_weight = torch.tensor(self.q_weight.copy(), requires_grad=True)
            k_weight = torch.tensor(self.k_weight.copy(), requires_grad=True)
            v_weight = torch.tensor(self.v_weight.copy(), requires_grad=True)
            o_weight = torch.tensor(self.o_weight.copy(), requires_grad=True)

            q_bias = torch.tensor(self.q_bias.copy(), requires_grad=True)
            k_bias = torch.tensor(self.k_bias.copy(), requires_grad=True)
            v_bias = torch.tensor(self.v_bias.copy(), requires_grad=True)
            o_bias = torch.tensor(self.o_bias.copy(), requires_grad=True)

            q = x @ q_weight + q_bias
            k = x @ k_weight + k_bias
            v = x @ v_weight + v_bias

            q = q.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            k = k.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            v = v.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

            kt = k.transpose(2, 3)

            attn_scores = q @ kt * (1 / (self.d_head ** 0.5))

            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)

            attn_output = attn_weights @ v

            attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)  # concat heads
            
            attn_output = attn_output @ o_weight + o_bias
            
            return attn_output, [q_weight, k_weight, v_weight, o_weight, q_bias, k_bias, v_bias, o_bias]
        
        self.pt_attend = pt_attend        
        self.my_attend = my_attend



        def my_layernorm(x: Tensor, layer_id, axis=-1, eps=1e-5):
            if layer_id == 1:
                gamma = Tensor(self.gamma_1.copy(), requires_grad=True)
                beta = Tensor(self.beta_1.copy(), requires_grad=True)
            elif layer_id == 2:
                gamma = Tensor(self.gamma_2.copy(), requires_grad=True)
                beta = Tensor(self.beta_2.copy(), requires_grad=True)
            else:
                raise ValueError(f"Invalid layer_id: {layer_id}")
            
            mean = x.mean(axis=axis, keepdims=True)
            var = ((x - mean).pow(2)).mean(axis=axis, keepdims=True)
            out = (x - mean) / (var + eps).pow(0.5)
            return out * gamma + beta, [gamma, beta]
        
        def pt_layernorm(x: torch.Tensor, layer_id, axis=-1, eps=1e-5):
            if layer_id == 1:
                gamma = torch.tensor(self.gamma_1.copy(), requires_grad=True)
                beta = torch.tensor(self.beta_1.copy(), requires_grad=True)
            elif layer_id == 2:
                gamma = torch.tensor(self.gamma_2.copy(), requires_grad=True)
                beta = torch.tensor(self.beta_2.copy(), requires_grad=True)
            else:
                raise ValueError(f"Invalid layer_id: {layer_id}")
            
            return F.layer_norm(x, (x.shape[-1],), gamma, beta, eps), [gamma, beta]

        def my_forward(x: Tensor):
            proj_up_weight = Tensor(self.proj_up_weight.copy(), requires_grad=True)
            proj_up_bias = Tensor(self.proj_up_bias.copy(), requires_grad=True)
            proj_down_weight = Tensor(self.proj_down_weight.copy(), requires_grad=True)
            proj_down_bias = Tensor(self.proj_down_bias.copy(), requires_grad=True)


            residual = x
            x, my_layernorm_params_1 = my_layernorm(x, 1)

            atten_out, atten_params = my_attend(x)

            x = atten_out + residual.detach()

            # MLP
            residual = x
            x, my_layernorm_params_2 = my_layernorm(x, 2)

            x_mlp = x @ proj_up_weight + proj_up_bias

            x_mlp = x_mlp._gelu()

            x_mlp = x_mlp @ proj_down_weight + proj_down_bias

            x = x_mlp + residual.detach()

            return x, [
                *my_layernorm_params_1, 
                *my_layernorm_params_2, 
                *atten_params, 
                proj_up_weight, 
                proj_up_bias, 
                proj_down_weight,
                proj_down_bias
                ]
        
        def pt_forward(x: torch.Tensor):
            proj_up_weight = torch.tensor(self.proj_up_weight.copy(), requires_grad=True)
            proj_up_bias = torch.tensor(self.proj_up_bias.copy(), requires_grad=True)
            proj_down_weight = torch.tensor(self.proj_down_weight.copy(), requires_grad=True)
            proj_down_bias = torch.tensor(self.proj_down_bias.copy(), requires_grad=True)
            
            residual = x
            x, pt_layernorm_params_1 = pt_layernorm(x, 1)

            atten_out, atten_params = pt_attend(x)

            x = atten_out + residual.detach()

            # MLP
            residual = x
            x, pt_layernorm_params_2 = pt_layernorm(x, 2)

            x_mlp = x @ proj_up_weight + proj_up_bias

            x_mlp = F.gelu(x_mlp)

            x_mlp = x_mlp @ proj_down_weight + proj_down_bias

            x = residual + x_mlp

            return x, [
                *pt_layernorm_params_1, 
                *pt_layernorm_params_2, 
                *atten_params, 
                proj_up_weight, 
                proj_up_bias, 
                proj_down_weight,
                proj_down_bias
                ]

        self.my_forward = my_forward
        self.pt_forward = pt_forward

        self.my_layernorm = my_layernorm
        self.pt_layernorm = pt_layernorm

    def test_layernorm(self):
        my_x = Tensor(self.x.copy(), requires_grad=True)
        pt_x = torch.tensor(self.x.copy(), requires_grad=True, dtype=torch.float32)

        my_output, my_params = self.my_layernorm(my_x, 1)
        pt_output, pt_params = self.pt_layernorm(pt_x, 1)

        my_loss = my_output.mean()
        pt_loss = pt_output.mean()

        my_loss.backward()
        pt_loss.backward()

        self.assert_close(my_loss.data, pt_loss.detach().numpy())

        for my_param, pt_param in zip(my_params, pt_params):
            self.assert_close(my_param.grad.data, pt_param.grad.detach().numpy())
            self.assertIsNotNone(my_param.grad)
            self.assertIsNotNone(pt_param.grad)


    def test_attend(self):
        my_x = Tensor(self.x.copy(), requires_grad=True)
        pt_x = torch.tensor(self.x.copy(), requires_grad=True, dtype=torch.float32)

        my_output, my_params = self.my_attend(my_x)
        pt_output, pt_params = self.pt_attend(pt_x)

        my_loss = my_output.mean()
        pt_loss = pt_output.mean()

        my_loss.backward()
        pt_loss.backward()

        self.assert_close(my_loss.data, pt_loss.detach().numpy())

        for my_param, pt_param in zip(my_params, pt_params):
            self.assert_close(my_param.grad.data, pt_param.grad.detach().numpy())
            self.assertIsNotNone(my_param.grad)
            self.assertIsNotNone(pt_param.grad)


    def test_layernorm_attend_composition(self):
        my_x = Tensor(self.x.copy(), requires_grad=True)
        pt_x = torch.tensor(self.x.copy(), requires_grad=True, dtype=torch.float32)

        my_layernorm_output, my_layernorm_params = self.my_layernorm(my_x, 1)
        my_attend_output, my_attend_params = self.my_attend(my_layernorm_output)
        pt_layernorm_output, pt_layernorm_params = self.pt_layernorm(pt_x, 1)
        pt_attend_output, pt_attend_params = self.pt_attend(pt_layernorm_output)

        my_loss = my_attend_output.mean()
        pt_loss = pt_attend_output.mean()

        my_loss.backward()
        pt_loss.backward()

        self.assert_close(my_loss.data, pt_loss.detach().numpy())

        for my_param, pt_param in zip(my_layernorm_params, pt_layernorm_params):
            self.assert_close(my_param.grad.data, pt_param.grad.detach().numpy())
            self.assertIsNotNone(my_param.grad)
            self.assertIsNotNone(pt_param.grad)
        
        for my_param, pt_param in zip(my_attend_params, pt_attend_params):
            self.assert_close(my_param.grad.data, pt_param.grad.detach().numpy())
            self.assertIsNotNone(my_param.grad)
            self.assertIsNotNone(pt_param.grad)


    def test_forward(self):
        my_x = Tensor(self.x.copy(), requires_grad=True)
        pt_x = torch.tensor(self.x.copy(), requires_grad=True, dtype=torch.float32)

        my_output, my_params = self.my_forward(my_x)
        pt_output, pt_params = self.pt_forward(pt_x)

        

        my_loss = my_output.mean()
        pt_loss = pt_output.mean()

        my_loss.backward()
        pt_loss.backward()
        
        self.assert_close(my_loss.data, pt_loss.detach().numpy())
        for my_param, pt_param in zip(my_params, pt_params):
            print("==========================================")
            print(f"MAX DIFF: {xp.max(xp.abs(my_param.data - pt_param.detach().numpy()))}")
            print("==========================================")
            self.assert_close(my_param.grad.data, pt_param.grad.detach().numpy())
            self.assertIsNotNone(my_param.grad)
            self.assertIsNotNone(pt_param.grad)
        



if __name__ == "__main__":
    unittest.main()
