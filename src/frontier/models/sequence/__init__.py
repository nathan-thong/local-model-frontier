"""Sequence-mixing module interface and built-in attention implementation."""

from .attention import CausalSelfAttention
from .interface import (
    DecoderCache,
    KVCache,
    SequenceModule,
    SequenceState,
    build_sequence_module,
    register_sequence_module,
)

__all__ = [
    "CausalSelfAttention",
    "DecoderCache",
    "KVCache",
    "SequenceModule",
    "SequenceState",
    "build_sequence_module",
    "register_sequence_module",
]
