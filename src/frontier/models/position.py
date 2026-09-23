"""Learned absolute positions and rotary position transforms."""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, time, head_dim]; rotate first/second half pairs.
        angles = torch.outer(positions.to(dtype=torch.float32), self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1).to(device=x.device)
        cos = angles.cos()[None, None, :, :].to(dtype=x.dtype)
        sin = angles.sin()[None, None, :, :].to(dtype=x.dtype)
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return x * cos + rotated * sin


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, width)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embedding(positions)
