"""Sequence-mixing module interface and built-in attention implementation."""

from .attention import CausalSelfAttention
from .interface import (
    AttentionState,
    DecoderCache,
    KVCache,
    RecurrentState,
    SequenceCostDescriptor,
    SequenceModule,
    SequenceModuleDescriptor,
    SequenceState,
    SequenceStateDescriptor,
    build_sequence_module,
    describe_sequence_module,
    iter_state_tensors,
    register_sequence_module,
)
from .local_attention import LocalCausalSelfAttention

__all__ = [
    "AttentionState",
    "CausalSelfAttention",
    "DecoderCache",
    "KVCache",
    "LocalCausalSelfAttention",
    "RecurrentState",
    "SequenceCostDescriptor",
    "SequenceModule",
    "SequenceModuleDescriptor",
    "SequenceState",
    "SequenceStateDescriptor",
    "build_sequence_module",
    "describe_sequence_module",
    "iter_state_tensors",
    "register_sequence_module",
]
