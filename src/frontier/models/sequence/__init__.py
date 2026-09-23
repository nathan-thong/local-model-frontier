"""Sequence-mixing module interface and registered reference implementations."""

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
from .linear_attention import LinearAttentionState, NormalizedCausalLinearAttention
from .local_attention import LocalCausalSelfAttention

__all__ = [
    "AttentionState",
    "CausalSelfAttention",
    "DecoderCache",
    "KVCache",
    "LinearAttentionState",
    "LocalCausalSelfAttention",
    "NormalizedCausalLinearAttention",
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
