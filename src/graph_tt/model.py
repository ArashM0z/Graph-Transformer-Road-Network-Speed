"""Spatio-Temporal Graph Transformer.

Architecture:
    input -> linear embed -> [SpatialAttention -> TemporalAttention -> FFN] * L
    -> linear output

Spatial attention uses a learnable mask derived from a normalised adjacency
matrix so that each node attends only to itself plus its k-hop neighbours.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        mask: torch.Tensor | None) -> torch.Tensor:
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError("hidden must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = hidden // n_heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # (..., L, H) -> (..., n_heads, L, d_head)
        new_shape = x.shape[:-1] + (self.n_heads, self.d_head)
        return x.reshape(new_shape).transpose(-3, -2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # (..., n_heads, L, d_head) -> (..., L, H)
        x = x.transpose(-3, -2).contiguous()
        return x.reshape(x.shape[:-2] + (self.n_heads * self.d_head,))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._split(self.q(q))
        k = self._split(self.k(k))
        v = self._split(self.v(v))
        out = _scaled_dot_product(q, k, v, mask)
        return self.out(self.dropout(self._merge(out)))


class SpatialAttention(nn.Module):
    """Per-time-step attention across nodes, masked by adjacency reachability."""

    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, n_heads, dropout)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, H), adj_mask: (N, N)
        B, T, N, H = x.shape
        x_ = x.reshape(B * T, N, H)
        out = self.attn(x_, x_, x_, mask=adj_mask)
        out = self.dropout(out)
        return self.norm(x + out.reshape(B, T, N, H))


class TemporalAttention(nn.Module):
    """Per-node attention across the time axis (causal)."""

    def __init__(self, hidden: int, n_heads: int, dropout: float, T: int):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, n_heads, dropout)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, H) -> per node attend across T
        B, T, N, H = x.shape
        x_ = x.permute(0, 2, 1, 3).reshape(B * N, T, H)
        out = self.attn(x_, x_, x_, mask=self.mask[:T, :T])
        out = self.dropout(out)
        out = out.reshape(B, N, T, H).permute(0, 2, 1, 3)
        return self.norm(x + out)


class FeedForward(nn.Module):
    def __init__(self, hidden: int, ff_mult: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * ff_mult)
        self.fc2 = nn.Linear(hidden * ff_mult, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.gelu(self.fc1(x)))
        return self.norm(x + self.dropout(self.fc2(h)))


class GraphTransformerTraffic(nn.Module):
    def __init__(self, n_nodes: int, input_dim: int, hidden: int = 64,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1,
                 ff_mult: int = 4, encoder_length: int = 12,
                 decoder_length: int = 12):
        super().__init__()
        self.n_nodes = n_nodes
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.embed = nn.Linear(input_dim, hidden)
        self.pos_emb = nn.Parameter(torch.randn(encoder_length + decoder_length,
                                                hidden) * 0.02)
        total_len = encoder_length + decoder_length
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "spatial": SpatialAttention(hidden, n_heads, dropout),
                "temporal": TemporalAttention(hidden, n_heads, dropout, total_len),
                "ff": FeedForward(hidden, ff_mult, dropout),
            }))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """Predict ``decoder_length`` future steps for each node.

        Args:
            x: (B, T_enc, N, input_dim) — input feature time-series.
            adj_mask: (N, N) boolean mask, True where i can attend to j.

        Returns:
            (B, decoder_length, N) predicted values.
        """
        B, T, N, _ = x.shape
        if T != self.encoder_length:
            raise ValueError(
                f"expected T={self.encoder_length}, got {T}"
            )
        # right-pad with zeros along time for the decoder horizon
        pad = torch.zeros(B, self.decoder_length, N, x.size(-1), device=x.device)
        x = torch.cat([x, pad], dim=1)
        h = self.embed(x)
        h = h + self.pos_emb.unsqueeze(0).unsqueeze(2)
        for layer in self.layers:
            h = layer["spatial"](h, adj_mask)
            h = layer["temporal"](h)
            h = layer["ff"](h)
        out = self.head(h[:, self.encoder_length:]).squeeze(-1)  # (B, dec, N)
        return out
