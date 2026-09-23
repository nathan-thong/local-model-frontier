"""Dense masked sliding-window attention used as a semantic reference."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from frontier.config import ModelConfig

from .attention import CausalSelfAttention
from .interface import (
    AttentionState,
    KVCache,
    SequenceCostDescriptor,
    SequenceModuleDescriptor,
    SequenceStateDescriptor,
    register_sequence_module,
)


class LocalCausalSelfAttention(CausalSelfAttention):
    """Causal attention restricted to the last ``window_size`` absolute positions.

    The boolean mask preserves local-window semantics but uses dense attention
    operands. This implementation is a correctness reference, not a sparse backend.
    """

    def __init__(self, config: ModelConfig) -> None:
        config.validate()
        super().__init__(config)
        if type(config.window_size) is not int or config.window_size <= 0:
            raise ValueError("window_size must be a positive integer for local attention")
        self.window_size = config.window_size

    def sequence_descriptor(self) -> SequenceModuleDescriptor:
        """Declare dense masked work and the bounded retained KV suffix."""
        return SequenceModuleDescriptor(
            name="local_attention_reference",
            cost=SequenceCostDescriptor(
                training_estimator="decoder-module-mac-v1",
                executed_work="dense_masked_local_attention",
                notes=(
                    "The boolean window mask changes allowed dependencies but the reference "
                    "still evaluates dense attention score/value operands."
                ),
            ),
            state=SequenceStateDescriptor(
                kind="local_attention_kv",
                storage_model="key and value tensors with absolute key_start_position",
                growth="bounded_by_window_size_minus_one",
                active_regions="all retained key and value tensor elements",
            ),
            parameters=(
                ("query_heads", self.query_heads),
                ("kv_heads", self.kv_heads),
                ("head_dim", self.head_dim),
                ("window_size", self.window_size),
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
        if position_start is None:
            position_start = int(positions[0].item())
        elif type(position_start) is not int or position_start < 0:
            raise ValueError("position_start must be a non-negative integer")
        expected_positions = torch.arange(
            position_start,
            position_start + time,
            dtype=positions.dtype,
            device=positions.device,
        )
        if position_start < 0 or not torch.equal(positions, expected_positions):
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
            if past_length > self.window_size - 1:
                raise ValueError("local attention cache cannot exceed window_size - 1 prior keys")
            if state.key_start_position + past_length != position_start:
                raise ValueError(
                    "cached attention state must end at position_start without gaps or overlap"
                )
            key_start_position = state.key_start_position
            k = torch.cat((state.key, k), dim=2)
            v = torch.cat((state.value, v), dim=2)

        cache_key = k
        cache_value = v
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        key_positions = torch.arange(
            key_start_position,
            key_start_position + past_length + time,
            device=hidden_states.device,
        )
        allowed = (key_positions[None, :] <= positions[:, None]) & (
            key_positions[None, :] >= positions[:, None] - self.window_size + 1
        )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed[None, None, :, :],
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
        output = self.out_dropout(self.out_proj(attended))

        present: KVCache | None = None
        if use_cache:
            keep_length = min(self.window_size - 1, past_length + time)
            end_position = position_start + time
            if keep_length == 0:
                retained_key = cache_key[:, :, :0, :].clone(memory_format=torch.contiguous_format)
                retained_value = cache_value[:, :, :0, :].clone(
                    memory_format=torch.contiguous_format
                )
            else:
                retained_key = cache_key[:, :, -keep_length:, :].clone(
                    memory_format=torch.contiguous_format
                )
                retained_value = cache_value[:, :, -keep_length:, :].clone(
                    memory_format=torch.contiguous_format
                )
            present = AttentionState(
                retained_key,
                retained_value,
                key_start_position=end_position - keep_length,
            )
        return output, present


register_sequence_module("local_attention_reference", LocalCausalSelfAttention)
