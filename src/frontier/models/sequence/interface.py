"""Registry contract for replacing attention with future sequence modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from frontier.config import ModelConfig

KVCache = tuple[torch.Tensor, torch.Tensor]
SequenceState = Any
SequenceFactory = Callable[[ModelConfig], nn.Module]
_REGISTRY: dict[str, SequenceFactory] = {}


@dataclass
class DecoderCache:
    """Per-layer module state plus the absolute position for the next input token."""

    states: list[SequenceState | None]
    position: int


class SequenceModule(Protocol):
    """A causal sequence mixer that can optionally carry per-layer decode state."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        state: SequenceState | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, SequenceState | None]: ...


def register_sequence_module(name: str, factory: SequenceFactory) -> None:
    if not name or name in _REGISTRY:
        raise ValueError(f"sequence module name is empty or already registered: {name!r}")
    _REGISTRY[name] = factory


def build_sequence_module(name: str, config: ModelConfig) -> nn.Module:
    try:
        factory = _REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise ValueError(
            f"sequence module {name!r} is not registered (available: {available})"
        ) from error
    return factory(config)
