"""Causal self-attention with MHA/GQA and an incremental KV cache."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from frontier.config import ModelConfig
from frontier.models.position import RotaryEmbedding

from .interface import (
    AttentionState,
    KVCache,
    SequenceCostDescriptor,
    SequenceModuleDescriptor,
    SequenceStateDescriptor,
    register_sequence_module,
)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        head_dim = config.width // config.query_heads
        if config.position == "rope" and head_dim % 2:
            raise ValueError("RoPE requires an even head dimension; adjust width or query_heads")
        self.query_heads = config.query_heads
        self.kv_heads = config.kv_heads
        self.head_dim = head_dim
        self.groups = config.query_heads // config.kv_heads
        self.dropout_p = config.dropout
        self.q_proj = nn.Linear(config.width, config.query_heads * head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.width, config.kv_heads * head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.width, config.kv_heads * head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.width, config.width, bias=config.bias)
        self.rope = (
            RotaryEmbedding(head_dim, config.rope_theta) if config.position == "rope" else None
        )
        self.out_dropout = nn.Dropout(config.dropout)

    @staticmethod
    def state_tensors(state: KVCache) -> AttentionState:
        return state

    @staticmethod
    def active_state_tensors(state: KVCache) -> tuple[torch.Tensor, ...]:
        """Attention retains every key and value slot in its cache."""
        return (state.key, state.value)

    def sequence_descriptor(self) -> SequenceModuleDescriptor:
        """Describe dense attention work separately from persistent KV storage."""
        return SequenceModuleDescriptor(
            name="attention",
            cost=SequenceCostDescriptor(
                training_estimator="decoder-module-mac-v1",
                executed_work="dense_full_causal_attention",
                notes=(
                    "Grouped K/V heads are expanded for attention operands; temporary "
                    "expansion is not part of persistent cache bytes."
                ),
            ),
            state=SequenceStateDescriptor(
                kind="attention_kv",
                storage_model="key and value tensors with absolute key_start_position",
                growth="linear_with_retained_context_tokens",
                active_regions="all retained key and value tensor elements",
            ),
            parameters=(
                ("query_heads", self.query_heads),
                ("kv_heads", self.kv_heads),
                ("head_dim", self.head_dim),
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        state: KVCache | None = None,
        use_cache: bool = False,
        position_start: int | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, time, width]")
        batch, time, width = hidden_states.shape
        if time <= 0:
            raise ValueError("attention input time dimension must be positive")
        if width != self.q_proj.in_features:
            raise ValueError("hidden_states width does not match the configured attention width")
        if positions.ndim != 1 or positions.numel() != time:
            raise ValueError("positions must contain one absolute position per input token")
        if positions.device != hidden_states.device:
            raise ValueError("positions and hidden_states must be on the same device")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use an integer dtype")
        validate_positions = position_start is None
        if position_start is None:
            position_start = int(positions[0].item())
        elif type(position_start) is not int or position_start < 0:
            raise ValueError("position_start must be a non-negative integer")
        if position_start < 0:
            raise ValueError("positions must start at a non-negative absolute position")

        if validate_positions and positions.device.type == "cpu":
            expected_positions = torch.arange(
                position_start,
                position_start + time,
                dtype=positions.dtype,
                device=positions.device,
            )
            if not torch.equal(positions, expected_positions):
                raise ValueError("positions must be contiguous and start at position_start")

        q = (
            self.q_proj(hidden_states)
            .view(batch, time, self.query_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(batch, time, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch, time, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        if self.rope is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        past_length = 0
        key_start_position = position_start
        if state is not None:
            if isinstance(state, tuple) and len(state) == 2:
                # Accept pre-contract in-memory tuple caches for callers holding a
                # cache across an API upgrade. Their start is inferable from the
                # current absolute position and retained length.
                if not all(isinstance(tensor, torch.Tensor) for tensor in state):
                    raise TypeError("legacy attention cache entries must be tensors")
                if any(tensor.ndim != 4 for tensor in state):
                    raise ValueError(
                        "cached key and value tensors must have shape [batch, heads, time, dim]"
                    )
                state = AttentionState(state[0], state[1], position_start - int(state[0].shape[2]))
            if not isinstance(state, AttentionState):
                raise TypeError(
                    "attention state must be AttentionState or a legacy (key, value) tuple"
                )
            if type(state.key_start_position) is not int or state.key_start_position < 0:
                raise ValueError("attention key_start_position must be a non-negative integer")
            if state.key.ndim != 4 or state.value.ndim != 4:
                raise ValueError(
                    "cached key and value tensors must have shape [batch, heads, time, dim]"
                )
            if state.key.shape != state.value.shape:
                raise ValueError("cached key and value tensors must have identical shapes")
            expected_prefix = (batch, self.kv_heads)
            if state.key.shape[:2] != expected_prefix or state.key.shape[3] != self.head_dim:
                raise ValueError(
                    "cached attention state shape must match the input batch, KV heads, and head dimension"
                )
            if (
                state.key.device != hidden_states.device
                or state.value.device != hidden_states.device
            ):
                raise ValueError(
                    "cached attention state and hidden_states must be on the same device"
                )
            if state.key.dtype != hidden_states.dtype or state.value.dtype != hidden_states.dtype:
                raise ValueError(
                    "cached attention state and hidden_states must have the same dtype"
                )
            past_length = int(state.key.shape[2])
            if state.key_start_position + past_length != position_start:
                raise ValueError(
                    "cached attention state must end at position_start without gaps or overlap"
                )
            key_start_position = state.key_start_position
            k = torch.cat((state.key, k), dim=2)
            v = torch.cat((state.value, v), dim=2)
        present: KVCache | None = AttentionState(k, v, key_start_position) if use_cache else None

        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        key_positions = torch.arange(
            key_start_position,
            key_start_position + past_length + time,
            device=hidden_states.device,
        )
        allowed = key_positions[None, :] <= positions[:, None]
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed[None, None, :, :],
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
        return self.out_dropout(self.out_proj(attended)), present


register_sequence_module("attention", CausalSelfAttention)
