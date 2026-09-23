"""Normalized causal linear attention implemented as a recurrent reference."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from frontier.config import ModelConfig
from frontier.models.position import RotaryEmbedding

from .interface import (
    SequenceCostDescriptor,
    SequenceModuleDescriptor,
    SequenceStateDescriptor,
    register_sequence_module,
)


@dataclass(frozen=True)
class LinearAttentionState:
    """FP32 recurrent key/value memory and the absolute position of the next token."""

    key_value_sum: torch.Tensor
    key_feature_sum: torch.Tensor
    next_position: int


class NormalizedCausalLinearAttention(nn.Module):
    """Causal ``ELU(x) + 1`` attention with constant-size decode state.

    This token-loop implementation is a correctness reference. Its recurrent
    accumulators are FP32 even when projections use an autocast dtype. The loop is
    intentionally simple and makes no parallel-training or fast-kernel claim.
    """

    epsilon = 1e-6
    accumulation_dtype = torch.float32

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.dropout != 0.0:
            raise ValueError(
                "normalized linear attention reference requires dropout=0 because "
                "attention-probability dropout is not implemented"
            )

        self.query_heads = config.query_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.width // config.query_heads
        self.groups = config.query_heads // config.kv_heads
        self.position_mode = config.position
        self.position_treatment = (
            "rotary_on_projected_qk_before_elu_plus_one"
            if config.position == "rope"
            else "learned_absolute_embedding_added_to_hidden_states_by_decoder"
        )

        self.q_proj = nn.Linear(config.width, config.query_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.width, config.kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.width, config.kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.width, config.width, bias=config.bias)
        self.rope = (
            RotaryEmbedding(self.head_dim, config.rope_theta) if config.position == "rope" else None
        )
        self.out_dropout = nn.Dropout(config.dropout)

    @staticmethod
    def state_tensors(state: LinearAttentionState) -> LinearAttentionState:
        return state

    @staticmethod
    def active_state_tensors(state: LinearAttentionState) -> tuple[torch.Tensor, ...]:
        return (state.key_value_sum, state.key_feature_sum)

    def sequence_descriptor(self) -> SequenceModuleDescriptor:
        """Describe recurrent work and the context-independent FP32 state."""
        return SequenceModuleDescriptor(
            name="normalized_linear_attention_reference",
            cost=SequenceCostDescriptor(
                training_estimator="decoder-module-mac-v1",
                executed_work="normalized_causal_linear_attention_token_loop",
                notes=(
                    "FP32 recurrent ELU+1 key-value and feature-sum accumulation with a "
                    "Python token loop; feature-map, epsilon, dropout, and loop overhead "
                    "are excluded from the registered MAC estimate."
                ),
            ),
            state=SequenceStateDescriptor(
                kind="linear_attention_recurrent",
                storage_model="FP32 key-feature/value outer-product sum S and key-feature sum z",
                growth="constant_with_context_for_fixed_batch_and_head_dimensions",
                active_regions="all elements of the key-value and key-feature accumulators",
            ),
            parameters=(
                ("query_heads", self.query_heads),
                ("kv_heads", self.kv_heads),
                ("head_dim", self.head_dim),
                ("accumulation_dtype", str(self.accumulation_dtype)),
                ("epsilon", self.epsilon),
                ("position_treatment", self.position_treatment),
            ),
        )

    def _position_start(
        self,
        positions: torch.Tensor,
        time: int,
        position_start: int | None,
    ) -> int:
        if positions.ndim != 1 or positions.numel() != time:
            raise ValueError("positions must contain one absolute position per input token")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use an integer dtype")
        if position_start is None:
            position_start = int(positions[0].item())
        elif type(position_start) is not int or position_start < 0:
            raise ValueError("position_start must be a non-negative integer")
        if position_start < 0:
            raise ValueError("positions must start at a non-negative absolute position")
        expected = torch.arange(
            position_start,
            position_start + time,
            dtype=positions.dtype,
            device=positions.device,
        )
        if not torch.equal(positions, expected):
            raise ValueError("positions must be contiguous and start at position_start")
        return position_start

    def _validate_state(
        self,
        state: LinearAttentionState,
        batch: int,
        hidden_states: torch.Tensor,
        position_start: int,
    ) -> None:
        if not isinstance(state, LinearAttentionState):
            raise TypeError("linear attention state must be a LinearAttentionState")
        if type(state.next_position) is not int or state.next_position < 0:
            raise ValueError("linear attention next_position must be a non-negative integer")
        if state.next_position != position_start:
            raise ValueError("linear attention state must end at position_start without gaps")
        expected_s = (batch, self.kv_heads, self.head_dim, self.head_dim)
        expected_z = (batch, self.kv_heads, self.head_dim)
        if tuple(state.key_value_sum.shape) != expected_s:
            raise ValueError("linear attention key_value_sum shape does not match this module")
        if tuple(state.key_feature_sum.shape) != expected_z:
            raise ValueError("linear attention key_feature_sum shape does not match this module")
        if (
            state.key_value_sum.dtype != self.accumulation_dtype
            or state.key_feature_sum.dtype != self.accumulation_dtype
        ):
            raise ValueError("linear attention recurrent state must use FP32 accumulation")
        if (
            state.key_value_sum.device != hidden_states.device
            or state.key_feature_sum.device != hidden_states.device
        ):
            raise ValueError("linear attention state and hidden_states must be on the same device")

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        state: LinearAttentionState | None = None,
        use_cache: bool = False,
        position_start: int | None = None,
    ) -> tuple[torch.Tensor, LinearAttentionState | None]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, time, width]")
        batch, time, width = hidden_states.shape
        if batch <= 0 or time <= 0:
            raise ValueError("linear attention batch and time dimensions must be positive")
        if width != self.q_proj.in_features:
            raise ValueError("hidden_states width does not match configured linear attention width")
        if positions.device != hidden_states.device:
            raise ValueError("positions and hidden_states must be on the same device")
        start = self._position_start(positions, time, position_start)

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

        if state is None:
            key_value_sum = torch.zeros(
                batch,
                self.kv_heads,
                self.head_dim,
                self.head_dim,
                dtype=self.accumulation_dtype,
                device=hidden_states.device,
            )
            key_feature_sum = torch.zeros(
                batch,
                self.kv_heads,
                self.head_dim,
                dtype=self.accumulation_dtype,
                device=hidden_states.device,
            )
        else:
            self._validate_state(state, batch, hidden_states, start)
            key_value_sum = state.key_value_sum
            key_feature_sum = state.key_feature_sum

        autocast = (
            torch.autocast(device_type=hidden_states.device.type, enabled=False)
            if hidden_states.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        outputs = []
        with autocast:
            q_feature = F.elu(q.to(dtype=self.accumulation_dtype)) + 1.0
            k_feature = F.elu(k.to(dtype=self.accumulation_dtype)) + 1.0
            value = v.to(dtype=self.accumulation_dtype)
            grouped_queries = q_feature.view(batch, self.kv_heads, self.groups, time, self.head_dim)
            for index in range(time):
                key = k_feature[:, :, index, :]
                current_value = value[:, :, index, :]
                key_value_sum = key_value_sum + torch.einsum("bhd,bhe->bhde", key, current_value)
                key_feature_sum = key_feature_sum + key
                query = grouped_queries[:, :, :, index, :]
                numerator = torch.einsum("bkgd,bkdf->bkgf", query, key_value_sum)
                denominator = torch.einsum("bkgd,bkd->bkg", query, key_feature_sum)
                outputs.append(
                    (numerator / (denominator.unsqueeze(-1) + self.epsilon)).reshape(
                        batch, self.query_heads, self.head_dim
                    )
                )

        attended = torch.stack(outputs, dim=2)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
        output = self.out_dropout(self.out_proj(attended.to(dtype=hidden_states.dtype)))
        present = (
            LinearAttentionState(key_value_sum, key_feature_sum, start + time)
            if use_cache
            else None
        )
        return output, present


def _build_normalized_linear_attention(config: ModelConfig) -> nn.Module:
    return NormalizedCausalLinearAttention(config)


register_sequence_module(
    "normalized_linear_attention_reference", _build_normalized_linear_attention
)
