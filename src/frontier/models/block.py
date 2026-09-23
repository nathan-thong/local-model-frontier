"""Transformer block supporting selectable residual normalization topology."""

from __future__ import annotations

import torch
from torch import nn

from frontier.config import ModelConfig
from frontier.models.feedforward import FeedForward
from frontier.models.normalization import make_norm
from frontier.models.sequence import SequenceState, build_sequence_module


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, sequence_type: str) -> None:
        super().__init__()
        self.residual_topology = config.residual_topology
        self.sequence = build_sequence_module(sequence_type, config)
        self.norm1 = make_norm(config.normalization, config.width)
        self.norm2 = make_norm(config.normalization, config.width)
        self.ffn = FeedForward(
            config.width, config.ffn_width, config.ffn, config.dropout, config.bias
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        state: SequenceState | None = None,
        use_cache: bool = False,
        position_start: int | None = None,
    ) -> tuple[torch.Tensor, SequenceState | None]:
        if self.residual_topology == "pre_norm":
            mixed, present = self.sequence(
                self.norm1(x), positions, state, use_cache, position_start
            )
            x = x + self.residual_dropout(mixed)
            x = x + self.residual_dropout(self.ffn(self.norm2(x)))
        else:
            mixed, present = self.sequence(x, positions, state, use_cache, position_start)
            x = self.norm1(x + self.residual_dropout(mixed))
            x = self.norm2(x + self.residual_dropout(self.ffn(x)))
        return x, present
