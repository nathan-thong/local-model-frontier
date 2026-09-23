"""JSON-backed experiment configuration and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 259
    max_seq_len: int = 256
    width: int = 128
    layers: int = 4
    query_heads: int = 4
    kv_heads: int = 4
    ffn_width: int = 512
    normalization: str = "rmsnorm"
    ffn: str = "gelu"
    position: str = "rope"
    residual_topology: str = "pre_norm"
    sequence_types: list[str] = field(default_factory=list)
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    def validate(self) -> None:
        if self.vocab_size < 259:
            raise ValueError(
                "vocab_size must include 256 bytes and BOS/EOS/PAD tokens (minimum 259)"
            )
        if self.width <= 0 or self.layers <= 0 or self.max_seq_len <= 1 or self.ffn_width <= 0:
            raise ValueError("width, layers, ffn_width and max_seq_len must be positive")
        if self.query_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("query_heads and kv_heads must be positive")
        if self.width % self.query_heads:
            raise ValueError("width must be divisible by query_heads")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads for GQA")
        if self.normalization not in {"layernorm", "rmsnorm"}:
            raise ValueError(f"unsupported normalization: {self.normalization}")
        if self.ffn not in {"gelu", "swiglu"}:
            raise ValueError(f"unsupported FFN: {self.ffn}")
        if self.position not in {"rope", "learned"}:
            raise ValueError(f"unsupported position encoding: {self.position}")
        if self.residual_topology not in {"pre_norm", "post_norm"}:
            raise ValueError(f"unsupported residual topology: {self.residual_topology}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not self.sequence_types:
            self.sequence_types = ["attention"] * self.layers
        elif len(self.sequence_types) != self.layers:
            raise ValueError("sequence_types must be empty or contain one module name per layer")


@dataclass
class TrainConfig:
    batch_size: int = 8
    context_length: int = 256
    max_steps: int | None = None
    max_tokens: int | None = 2_097_152
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation: int = 1
    eval_interval: int = 100
    checkpoint_interval: int = 100
    mixed_precision: str = "auto"
    deterministic: bool = True
    resume: bool = False

    def validate(self, model: ModelConfig) -> None:
        if self.context_length <= 0 or self.context_length > model.max_seq_len:
            raise ValueError("context_length must be positive and no larger than model.max_seq_len")
        if self.batch_size <= 0 or self.gradient_accumulation <= 0:
            raise ValueError("batch_size and gradient_accumulation must be positive")
        if (self.max_steps is None) == (self.max_tokens is None):
            raise ValueError("set exactly one of max_steps and max_tokens")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError(
                "learning rates must be non-negative and learning_rate must be positive"
            )
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.mixed_precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("mixed_precision must be auto, fp32, bf16 or fp16")
        if self.eval_interval <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("eval_interval and checkpoint_interval must be positive")


@dataclass
class EvalConfig:
    stride: int = 128
    max_validation_documents: int | None = None
    tasks_path: str | None = None
    generation_max_tokens: int = 64


@dataclass
class ProfileConfig:
    batch_size: int = 1
    prompt_lengths: list[int] = field(default_factory=lambda: [32, 128, 224])
    decode_tokens: int = 32
    warmup_steps: int = 2
    repeats: int = 5


@dataclass
class RunConfig:
    seed: int = 17
    output_dir: str = "runs/default"
    data_dir: str | None = None
    tokenizer: str = "utf8-byte-v1"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    profiling: ProfileConfig = field(default_factory=ProfileConfig)
    baseline_run: str | None = None

    def validate(self) -> None:
        self.model.validate()
        self.train.validate(self.model)
        if self.evaluation.stride <= 0 or self.evaluation.generation_max_tokens <= 0:
            raise ValueError("evaluation stride and generation length must be positive")
        if self.evaluation.stride > self.train.context_length:
            raise ValueError("evaluation stride cannot exceed training context_length")
        if self.profiling.batch_size <= 0 or self.profiling.decode_tokens <= 0:
            raise ValueError("profile batch size and decode length must be positive")
        if self.profiling.repeats <= 0 or self.profiling.warmup_steps < 0:
            raise ValueError("profile repeats must be positive and warmup_steps non-negative")
        if self.profiling.decode_tokens >= self.model.max_seq_len:
            raise ValueError("profile decode_tokens must be shorter than model.max_seq_len")
        if any(
            length <= 0 or length + self.profiling.decode_tokens > self.model.max_seq_len
            for length in self.profiling.prompt_lengths
        ):
            raise ValueError(
                "each profile prompt length plus decode_tokens must fit model.max_seq_len"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        config = cls(
            seed=raw.get("seed", 17),
            output_dir=raw.get("output_dir", "runs/default"),
            data_dir=raw.get("data_dir"),
            tokenizer=raw.get("tokenizer", "utf8-byte-v1"),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            evaluation=EvalConfig(**raw.get("evaluation", {})),
            profiling=ProfileConfig(**raw.get("profiling", {})),
            baseline_run=raw.get("baseline_run"),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> RunConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
