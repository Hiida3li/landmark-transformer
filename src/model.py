"""A small Transformer encoder for gesture classification from landmark sequences.

Attention is implemented from scratch.
Shapes:  B = batch, T = frames (tokens), D = d_model, H = heads, dk = D // H
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)          # W_Q: (D, D)
        self.W_k = nn.Linear(d_model, d_model)          # W_K
        self.W_v = nn.Linear(d_model, d_model)          # W_V
        self.W_o = nn.Linear(d_model, d_model)          # output projection
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, D = x.shape

        # project:  X W_Q, X W_K, X W_V           (B, T, D)
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)

        # split into heads:  (B, T, D) -> (B, H, T, dk)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)

        # scores = Q K^T / sqrt(dk):  (B, H, T, dk) @ (B, H, dk, T) -> (B, H, T, T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)

        # softmax over the last axis: each row becomes a distribution over tokens
        attn = F.softmax(scores, dim=-1)                # (B, H, T, T)
        attn = self.drop(attn)

        # weighted sum of values:  (B, H, T, T) @ (B, H, T, dk) -> (B, H, T, dk)
        out = attn @ v

        # merge heads back:  (B, H, T, dk) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)
        return (out, attn) if return_attn else out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(                        # position-wise feed-forward
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))    # residual + pre-norm attention
        x = x + self.drop(self.ff(self.norm2(x)))      # residual + pre-norm feed-forward
        return x


class LandmarkTransformer(nn.Module):
    def __init__(self, n_features: int = 63, seq_len: int = 30, n_classes: int = 5,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)            # (63) -> (D): token embedding
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))  # learned positional encoding
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)               # classifier
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 63)
        x = self.embed(x) + self.pos                            # (B, T, D)
        for block in self.blocks:
            x = block(x)                                        # (B, T, D)
        x = self.norm(x)
        x = x.mean(dim=1)                                       # (B, D)  average over tokens
        return self.head(x)                                     # (B, n_classes)  logits


if __name__ == "__main__":
    model = LandmarkTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    x = torch.randn(4, 30, 63)
    logits = model(x)
    print("input:", x.shape, "-> logits:", logits.shape)

    _, attn = model.blocks[0].attn(model.embed(x) + model.pos, return_attn=True)
    print("attention:", attn.shape, "  row sums:", attn[0, 0].sum(dim=-1)[:3])