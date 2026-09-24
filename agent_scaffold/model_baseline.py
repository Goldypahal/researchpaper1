"""
Modular Baseline Decoder-only Structural Transformer in PyTorch.
Supports Level 1 (Hyperparameter Search) and Level 2 (Structural Architecture Search):
- Attention Mechanisms: Multi-Head Attention (MHA), Grouped-Query Attention (GQA)
- Positional Encodings: Learned 1D embeddings, Sinusoidal, Rotary Position Embedding (RoPE), NoPE (none)
- FFN Sub-networks: Standard FFN (Linear-Act-Linear), SwiGLU / Gated FFN (LLaMA/PaLM style)
- Residual Topologies: Pre-LN, Post-LN, Parallel Transformer (GPT-J style)
- Normalization: LayerNorm, RMSNorm
- Activations: GELU, ReLU, SiLU
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Normalization Layers
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def get_norm(norm_name, dim):
    if str(norm_name).lower().strip() == "rmsnorm":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)

def get_activation(act_name):
    act = str(act_name).lower().strip()
    if act == "relu":
        return nn.ReLU()
    elif act in ["silu", "swish"]:
        return nn.SiLU()
    elif act == "gelu":
        return nn.GELU()
    return nn.GELU()

# ---------------------------------------------------------------------------
# Positional Encodings (RoPE & Sinusoidal)
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, q, k):
        T = q.size(2)
        cos = self.cos[:T, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T, :].unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope(x):
            d_half = x.shape[-1] // 2
            return (x * torch.cat([cos, cos], dim=-1)) + (rotate_half(x) * torch.cat([sin, sin], dim=-1))

        return apply_rope(q), apply_rope(k)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# ---------------------------------------------------------------------------
# Attention Module (MHA & GQA)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_kv_heads=None,
                 pos_encoding="learned", dropout=0.0, scale_factor=None, max_seq_len=256):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if (n_kv_heads is not None and n_kv_heads > 0) else n_heads
        assert n_heads % self.n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        self.pos_encoding = str(pos_encoding).lower().strip()

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale_factor if scale_factor is not None else (1.0 / math.sqrt(self.d_k))

        if self.pos_encoding == "rotary":
            self.rope = RotaryEmbedding(self.d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# ---------------------------------------------------------------------------
# FFN Module (Standard vs SwiGLU)
# ---------------------------------------------------------------------------
class ModularFFN(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, ffn_type="standard", activation="gelu", dropout=0.0):
        super().__init__()
        self.ffn_type = str(ffn_type).lower().strip()
        if self.ffn_type in ["swiglu", "gated"]:
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_ff, d_model, bias=False)
            self.act = get_activation(activation)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                get_activation(activation),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        if self.ffn_type in ["swiglu", "gated"]:
            return self.w3(self.act(self.w1(x)) * self.w2(x))
        return self.net(x)

# ---------------------------------------------------------------------------
# Structural Transformer Block with Topologies
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_kv_heads=None, d_ff=1024, dropout=0.0,
                 activation="gelu", norm_type="layernorm", scale_factor=None,
                 pos_encoding="learned", ffn_type="standard", topology="pre_ln", max_seq_len=256):
        super().__init__()
        self.topology = str(topology).lower().strip()
        self.ln1 = get_norm(norm_type, d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, n_kv_heads, pos_encoding, dropout, scale_factor, max_seq_len)
        self.ln2 = get_norm(norm_type, d_model)
        self.ffn = ModularFFN(d_model, d_ff, ffn_type, activation, dropout)

    def forward(self, x):
        if self.topology == "pre_ln":
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        elif self.topology == "post_ln":
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        elif self.topology == "parallel":
            # GPT-J / PaLM parallel formulation
            x = x + self.attn(self.ln1(x)) + self.ffn(self.ln2(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        return x

# ---------------------------------------------------------------------------
# Complete SmallTransformerLM
# ---------------------------------------------------------------------------
class SmallTransformerLM(nn.Module):
    def __init__(self, vocab_size=1024, d_model=256, n_layers=6, n_heads=4, n_kv_heads=None,
                 d_ff=1024, max_seq_len=256, dropout=0.0, activation="gelu", norm_type="layernorm",
                 scale_factor=None, pos_encoding="learned", ffn_type="standard", topology="pre_ln"):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_encoding = str(pos_encoding).lower().strip()
        self.token_emb = nn.Embedding(vocab_size, d_model)

        if self.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        elif self.pos_encoding == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        else:
            # For rotary or none, token embedding alone enters blocks
            self.pos_emb = None

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type,
                scale_factor=scale_factor,
                pos_encoding=pos_encoding,
                ffn_type=ffn_type,
                topology=topology,
                max_seq_len=max_seq_len
            )
            for _ in range(n_layers)
        ])
        self.ln_f = get_norm(norm_type, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx)

        if self.pos_emb is not None:
            if self.pos_encoding == "learned":
                pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
                x = x + self.pos_emb(pos)
            elif self.pos_encoding == "sinusoidal":
                x = x + self.pos_emb(x)

        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss
