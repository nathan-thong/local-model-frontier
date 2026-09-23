from frontier.config import ModelConfig
from frontier.profiling.compute import ESTIMATOR_ID, estimate_training_compute


def make_config(**changes):
    values = {
        "vocab_size": 259,
        "max_seq_len": 8,
        "width": 8,
        "layers": 2,
        "query_heads": 4,
        "kv_heads": 2,
        "ffn_width": 16,
        "normalization": "rmsnorm",
        "ffn": "gelu",
        "position": "rope",
        "tie_embeddings": True,
    }
    values.update(changes)
    return ModelConfig(**values)


def test_attention_compute_matches_hand_derived_small_shape():
    estimate = estimate_training_compute(make_config(), input_tokens=8, context_length=4)

    assert estimate["estimator"] == ESTIMATOR_ID
    assert estimate["status"] == "estimated"
    assert estimate["components"] == {
        "sequence_projection_macs": 3_072,
        "attention_score_value_macs": 1_024,
        "feed_forward_macs": 4_096,
        "vocabulary_head_macs": 16_576,
        "total_macs": 24_768,
    }
    assert estimate["estimated_macs"] == 24_768
    assert estimate["estimated_flops"] == 148_608


def test_compute_scales_by_tokens_context_layers_and_ffn_type():
    base = estimate_training_compute(make_config(), input_tokens=8, context_length=4)
    more_tokens = estimate_training_compute(make_config(), input_tokens=16, context_length=4)
    longer_context = estimate_training_compute(make_config(), input_tokens=8, context_length=8)
    swiglu = estimate_training_compute(make_config(ffn="swiglu"), 8, 4)
    more_layers = estimate_training_compute(make_config(layers=4), 8, 4)

    assert (
        more_tokens["components"]["sequence_projection_macs"]
        == 2 * base["components"]["sequence_projection_macs"]
    )
    assert (
        longer_context["components"]["attention_score_value_macs"]
        == 2 * base["components"]["attention_score_value_macs"]
    )
    assert swiglu["components"]["feed_forward_macs"] == 6_144
    assert (
        more_layers["components"]["attention_score_value_macs"]
        == 2 * base["components"]["attention_score_value_macs"]
    )


def test_gqa_projection_work_is_lower_than_mha_and_unknown_blocks_are_unpriced():
    gqa = estimate_training_compute(make_config(kv_heads=2), 8, 4)
    mha = estimate_training_compute(make_config(kv_heads=4), 8, 4)
    unknown = estimate_training_compute(
        make_config(sequence_types=["experimental", "experimental"]), 8, 4
    )

    assert (
        gqa["components"]["sequence_projection_macs"]
        < mha["components"]["sequence_projection_macs"]
    )
    assert unknown["status"] == "unavailable"
    assert unknown["estimated_flops"] is None
    assert unknown["unsupported_sequence_type"] == "experimental"
