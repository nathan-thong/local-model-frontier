"""Configurable decoder-only language model with cache-aware inference."""

from __future__ import annotations

import torch
from torch import nn

from frontier.config import ModelConfig
from frontier.models.block import TransformerBlock
from frontier.models.normalization import make_norm
from frontier.models.position import LearnedPositionEmbedding
from frontier.models.sequence import DecoderCache, SequenceState


class DecoderLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = (
            LearnedPositionEmbedding(config.max_seq_len, config.width)
            if config.position == "learned"
            else None
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, block_type) for block_type in config.sequence_types]
        )
        self.final_norm = (
            make_norm(config.normalization, config.width)
            if config.residual_topology == "pre_norm"
            else nn.Identity()
        )
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters() if not trainable_only or p.requires_grad)
        return sum(p.numel() for p in {id(p): p for p in params}.values())

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: DecoderCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, DecoderCache | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        batch, time = input_ids.shape
        if batch <= 0 or time <= 0:
            raise ValueError("input_ids batch and time dimensions must be positive")
        if cache is not None:
            if type(cache.position) is not int or cache.position < 0:
                raise ValueError("cache.position must be a non-negative absolute token position")
            if len(cache.states) != len(self.blocks):
                raise ValueError("cache must contain one entry per sequence block")
        start = 0 if cache is None else cache.position
        if start + time > self.config.max_seq_len:
            raise ValueError("input plus cached context exceeds model.max_seq_len")
        positions = torch.arange(start, start + time, device=input_ids.device)
        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            x = x + self.position_embedding(positions)[None, :, :]
        x = self.input_dropout(x)
        new_cache: list[SequenceState | None] = []
        for index, block in enumerate(self.blocks):
            prior = None if cache is None else cache.states[index]
            x, state = block(x, positions, prior, use_cache, start)
            new_cache.append(state)
        logits = self.lm_head(self.final_norm(x))
        return logits, DecoderCache(new_cache, start + time) if use_cache else None
