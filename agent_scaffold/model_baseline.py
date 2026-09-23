"""
Modular Baseline Decoder-only Transformer in PyTorch.
Designed to allow clean, pluggable modifications to:
- Self-Attention mechanisms (MHA, Head dimension, scaling)
- Feed-Forward / Activation layers (GELU, ReLU, SiLU)
- Normalization (LayerNorm, RMSNorm)
- Depth / Width scaling (n_layers, d_model, n_heads, d_ff)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.0, scale_factor=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = scale_factor if scale_factor is not None else (1.0 / math.sqrt(self.d_k))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

def get_activation(act_name):
    act = act_name.lower().strip()
    if act == "relu":
        return nn.ReLU()
    elif act in ["silu", "swish"]:
        return nn.SiLU()
    elif act == "gelu":
        return nn.GELU()
    else:
        return nn.GELU()

def get_norm(norm_name, dim):
    if norm_name.lower().strip() == "rmsnorm":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=1024, dropout=0.0, activation="gelu", norm_type="layernorm", scale_factor=None):
        super().__init__()
        self.ln1 = get_norm(norm_type, d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, scale_factor)
        self.ln2 = get_norm(norm_type, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation(activation),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SmallTransformerLM(nn.Module):
    def __init__(self, vocab_size=1024, d_model=256, n_layers=6, n_heads=4, d_ff=1024, 
                 max_seq_len=256, dropout=0.0, activation="gelu", norm_type="layernorm", scale_factor=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation, norm_type, scale_factor) 
            for _ in range(n_layers)
        ])
        self.ln_f = get_norm(norm_type, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss
