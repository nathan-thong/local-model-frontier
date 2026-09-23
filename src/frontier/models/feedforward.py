"""Position-wise feed-forward modules."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(
        self, width: int, hidden_width: int, kind: str, dropout: float, bias: bool
    ) -> None:
        super().__init__()
        self.kind = kind
        self.dropout = nn.Dropout(dropout)
        if kind == "gelu":
            self.up = nn.Linear(width, hidden_width, bias=bias)
            self.down = nn.Linear(hidden_width, width, bias=bias)
        elif kind == "swiglu":
            self.gate = nn.Linear(width, hidden_width, bias=bias)
            self.up = nn.Linear(width, hidden_width, bias=bias)
            self.down = nn.Linear(hidden_width, width, bias=bias)
        else:
            raise ValueError(f"unsupported feed-forward module: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "gelu":
            x = F.gelu(self.up(x), approximate="tanh")
        else:
            x = F.silu(self.gate(x)) * self.up(x)
        return self.dropout(self.down(x))
