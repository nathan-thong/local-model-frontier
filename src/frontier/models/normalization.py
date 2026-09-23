"""Normalization layers with an explicit, configurable choice."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight


def make_norm(kind: str, width: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(width)
    if kind == "rmsnorm":
        return RMSNorm(width)
    raise ValueError(f"unsupported normalization: {kind}")
