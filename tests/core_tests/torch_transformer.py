import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert dropout == 0.0, "dropout must be 0.0"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)  # query, key, value
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        # Project Q, K, V and split heads
        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        qkv = qkv.reshape(B, T, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (B, num_heads, T, 3*head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, num_heads, T, head_dim)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, T, T)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, T, T)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)  # (B, T, d_model)
        
        # Output projection
        out = self.out_proj(attn_output)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        assert dropout == 0.0, "dropout must be 0.0"
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        # Multi-head attention + residual + norm
        x = x + self.mha(self.norm1(x), mask)
        # MLP + residual + norm
        x = x + self.mlp(self.norm2(x))
        return x
