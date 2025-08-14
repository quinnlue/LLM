import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Separate Q, K, V projections
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        
        self.o = nn.Linear(d_model, d_model)
        
        # Norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # MLP
        self.proj_up = nn.Linear(d_model, mlp_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.proj_down = nn.Linear(mlp_dim, d_model)

    def forward(self, x):
        B, T, C = x.size()
        
        x_norm = self.ln1(x)
        
        # Project Q, K, V separately
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)
        
        # Reshape for multi-head: (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)  # concat heads
        
        attn_out = self.o(attn_output)
        x = attn_out + x  # residual connection
        
        x_norm2 = self.ln2(x)
        x_mlp = self.proj_up(x_norm2)
        x_mlp = self.gelu(x_mlp)
        mlp_out = self.proj_down(x_mlp)
        x = mlp_out + x  # residual connection
        
        return x
