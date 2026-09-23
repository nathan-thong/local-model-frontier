"""Causal self-attention with MHA/GQA and an incremental KV cache."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from frontier.config import ModelConfig
from frontier.models.position import RotaryEmbedding

from .interface import KVCache, register_sequence_module


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
    def state_tensors(state: KVCache) -> tuple[torch.Tensor, ...]:
        return state

    @staticmethod
    def active_state_tensors(state: KVCache) -> tuple[torch.Tensor, ...]:
        """Attention retains every key and value slot in its cache."""
        return state

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        state: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch, time, width = hidden_states.shape
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
        if state is not None:
            past_length = int(state[0].shape[2])
            k = torch.cat((state[0], k), dim=2)
            v = torch.cat((state[1], v), dim=2)
        present: KVCache | None = (k, v) if use_cache else None

        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        key_positions = torch.arange(past_length + time, device=hidden_states.device)
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
