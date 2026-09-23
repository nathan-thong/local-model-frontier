"""Versioned analytical training-compute estimates for configured model blocks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from frontier.config import ModelConfig

ESTIMATOR_ID = "decoder-module-mac-v1"
SequenceMacEstimator = Callable[[ModelConfig, int], dict[str, int]]
_SEQUENCE_ESTIMATORS: dict[str, SequenceMacEstimator] = {}


def register_sequence_mac_estimator(name: str, estimator: SequenceMacEstimator) -> None:
    if not name or name in _SEQUENCE_ESTIMATORS:
        raise ValueError(
            f"sequence compute estimator name is empty or already registered: {name!r}"
        )
    _SEQUENCE_ESTIMATORS[name] = estimator


def _attention_macs_per_token(config: ModelConfig, context_length: int) -> dict[str, int]:
    head_dim = config.width // config.query_heads
    kv_width = config.kv_heads * head_dim
    projections = config.width * (config.width + 2 * kv_width + config.width)
    score_and_value = 2 * config.query_heads * head_dim * context_length
    return {
        "sequence_projection_macs_per_token": projections,
        "attention_score_value_macs_per_token": score_and_value,
    }


register_sequence_mac_estimator("attention", _attention_macs_per_token)
register_sequence_mac_estimator("local_attention_reference", _attention_macs_per_token)


def estimate_training_compute(
    config: ModelConfig, input_tokens: int, context_length: int
) -> dict[str, Any]:
    """Estimate dense MAC work and its conventional 3x training forward multiplier.

    This counts matrix multiply-accumulates and converts each MAC to two FLOPs. Backward
    matrix work is approximated as twice forward matrix work. Embedding lookups, softmax,
    normalization, activations, loss, optimizer, dropout and other scalar work are excluded.
    """
    config.validate()
    if input_tokens <= 0 or context_length <= 0 or context_length > config.max_seq_len:
        raise ValueError("input_tokens and context_length must be positive and fit the model")
    if input_tokens % context_length:
        raise ValueError("input_tokens must be a whole number of full-context training sequences")

    per_layer = []
    for index, sequence_type in enumerate(config.sequence_types):
        estimator = _SEQUENCE_ESTIMATORS.get(sequence_type)
        if estimator is None:
            return {
                "estimator": ESTIMATOR_ID,
                "status": "unavailable",
                "estimated_macs": None,
                "estimated_flops": None,
                "components": None,
                "unsupported_sequence_type": sequence_type,
                "unsupported_layer_index": index,
                "assumptions": "no compute estimator is registered for this sequence module",
            }
        per_layer.append(estimator(config, context_length))

    sequence_projection_per_token = sum(
        layer["sequence_projection_macs_per_token"] for layer in per_layer
    )
    attention_per_token = sum(layer["attention_score_value_macs_per_token"] for layer in per_layer)
    if config.ffn == "gelu":
        ffn_macs_per_token_per_layer = 2 * config.width * config.ffn_width
    else:
        ffn_macs_per_token_per_layer = 3 * config.width * config.ffn_width
    ffn_per_token = config.layers * ffn_macs_per_token_per_layer
    vocabulary_head_per_token = config.width * config.vocab_size

    components = {
        "sequence_projection_macs": sequence_projection_per_token * input_tokens,
        "attention_score_value_macs": attention_per_token * input_tokens,
        "feed_forward_macs": ffn_per_token * input_tokens,
        "vocabulary_head_macs": vocabulary_head_per_token * input_tokens,
    }
    total_macs = sum(components.values())
    components["total_macs"] = total_macs
    return {
        "estimator": ESTIMATOR_ID,
        "status": "estimated",
        "estimated_macs": total_macs,
        "estimated_flops": 6 * total_macs,
        "components": components,
        "sequence_modules": list(config.sequence_types),
        "per_layer_sequence_macs_per_token": per_layer,
        "assumptions": (
            "two FLOPs per MAC and 3x forward MACs for forward plus backward; charges full "
            "dense causal attention matrices at configured context; embedding table lookups, "
            "softmax, normalization, activations, loss, optimizer, dropout and scalar work "
            "are excluded; tied embedding weights are counted as vocabulary-head compute once"
        ),
    }
