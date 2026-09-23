import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from frontier.config import ModelConfig, RunConfig
from frontier.models import DecoderLanguageModel
from frontier.models.feedforward import FeedForward
from frontier.models.normalization import RMSNorm
from frontier.models.position import RotaryEmbedding
from frontier.models.sequence import AttentionState, DecoderCache
from frontier.models.sequence.attention import CausalSelfAttention
from frontier.profiling.compute import estimate_training_compute


def small_config(**overrides):
    values = {
        "vocab_size": 259,
        "max_seq_len": 32,
        "width": 32,
        "layers": 2,
        "query_heads": 4,
        "kv_heads": 2,
        "ffn_width": 64,
        "normalization": "rmsnorm",
        "ffn": "gelu",
        "position": "rope",
        "residual_topology": "pre_norm",
        "dropout": 0.0,
        "tie_embeddings": True,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_rmsnorm_matches_definition_and_preserves_shape():
    x = torch.randn(2, 3, 8)
    layer = RMSNorm(8, eps=1e-6)
    actual = layer(x)
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)


def test_swiglu_and_gelu_preserve_residual_width():
    x = torch.randn(2, 5, 16)
    for kind in ("gelu", "swiglu"):
        layer = FeedForward(16, 48, kind, dropout=0.0, bias=False)
        assert layer(x).shape == x.shape


def test_gqa_is_causal_and_has_expected_shapes():
    model = DecoderLanguageModel(small_config()).eval()
    ids_a = torch.tensor([[1, 7, 8, 9, 10]])
    ids_b = torch.tensor([[1, 7, 8, 99, 101]])
    logits_a, _ = model(ids_a)
    logits_b, _ = model(ids_b)
    assert logits_a.shape == (1, 5, 259)
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], atol=1e-6, rtol=1e-6)
    attention = model.blocks[0].sequence
    assert attention.query_heads == 4
    assert attention.kv_heads == 2
    assert attention.groups == 2


def _explicit_gqa_reference(attention, hidden_states, positions):
    batch, time, width = hidden_states.shape
    q = (
        attention.q_proj(hidden_states)
        .view(batch, time, attention.query_heads, attention.head_dim)
        .transpose(1, 2)
    )
    k = (
        attention.k_proj(hidden_states)
        .view(batch, time, attention.kv_heads, attention.head_dim)
        .transpose(1, 2)
    )
    v = (
        attention.v_proj(hidden_states)
        .view(batch, time, attention.kv_heads, attention.head_dim)
        .transpose(1, 2)
    )
    if attention.rope is not None:
        q = attention.rope(q, positions)
        k = attention.rope(k, positions)
    k = k.repeat_interleave(attention.groups, dim=1)
    v = v.repeat_interleave(attention.groups, dim=1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(attention.head_dim)
    allowed = torch.arange(time, device=hidden_states.device)[None, :] <= positions[:, None]
    scores = scores.masked_fill(~allowed[None, None], -torch.inf)
    probabilities = F.softmax(scores, dim=-1)
    attended = probabilities @ v
    attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
    return attention.out_dropout(attention.out_proj(attended))


def test_gqa_matches_explicit_repeated_kv_reference_in_forward_and_backward():
    torch.manual_seed(21)
    attention = CausalSelfAttention(small_config()).eval()
    inputs = torch.randn(2, 7, 32, requires_grad=True)
    positions = torch.arange(7)
    weights = torch.randn_like(inputs)

    actual = attention(inputs, positions)[0]
    (actual * weights).sum().backward()
    actual_input_gradient = inputs.grad.detach().clone()
    actual_parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in attention.named_parameters()
        if parameter.grad is not None
    }

    attention.zero_grad(set_to_none=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference = _explicit_gqa_reference(attention, reference_inputs, positions)
    (reference * weights).sum().backward()

    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_input_gradient, reference_inputs.grad, atol=1e-6, rtol=1e-6)
    for name, parameter in attention.named_parameters():
        if name in actual_parameter_gradients:
            torch.testing.assert_close(
                actual_parameter_gradients[name], parameter.grad, atol=1e-6, rtol=1e-6
            )


def _cache_storage_bytes(cache):
    seen = set()
    total = 0
    for state in cache.states:
        for tensor in state:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return total


def test_gqa_cache_storage_is_half_mha_for_identical_shape_and_sequence():
    ids = torch.randint(0, 256, (2, 13))
    cache_bytes = {}
    for kv_heads in (4, 2):
        model = DecoderLanguageModel(small_config(kv_heads=kv_heads)).eval()
        _, cache = model(ids, use_cache=True)
        actual = _cache_storage_bytes(cache)
        expected = (
            2
            * model.config.layers
            * ids.shape[0]
            * ids.shape[1]
            * kv_heads
            * (model.config.width // model.config.query_heads)
            * 4
        )
        assert actual == expected
        cache_bytes[kv_heads] = actual

    assert cache_bytes[2] / cache_bytes[4] == 0.5


def test_prepared_gqa_config_changes_kv_heads_and_matches_mha_compute():
    config_root = Path(__file__).resolve().parents[1] / "configs"
    baseline = RunConfig.from_json(config_root / "baseline_a.json")
    candidate = RunConfig.from_json(config_root / "gqa_a.json")

    baseline_model = baseline.model.__dict__.copy()
    candidate_model = candidate.model.__dict__.copy()
    candidate_model["kv_heads"] = baseline_model["kv_heads"]
    assert candidate_model == baseline_model
    assert candidate.model.kv_heads == 2
    assert baseline.data_dir == candidate.data_dir
    assert baseline.evaluation.__dict__ == candidate.evaluation.__dict__
    baseline_train = baseline.train.__dict__.copy()
    candidate_train = candidate.train.__dict__.copy()
    candidate_train["max_tokens"] = baseline_train["max_tokens"]
    assert candidate_train == baseline_train

    baseline_flops = estimate_training_compute(
        baseline.model, baseline.train.max_tokens, baseline.train.context_length
    )["estimated_flops"]
    candidate_flops = estimate_training_compute(
        candidate.model, candidate.train.max_tokens, candidate.train.context_length
    )["estimated_flops"]
    assert abs(candidate_flops - baseline_flops) / baseline_flops < 0.01


def test_incremental_kv_logits_match_full_sequence():
    torch.manual_seed(4)
    for position in ("rope", "learned"):
        model = DecoderLanguageModel(small_config(position=position)).eval()
        ids = torch.randint(0, 256, (2, 11))
        full, full_cache = model(ids)
        assert full_cache is None
        first, cache = model(ids[:, :5], use_cache=True)
        second, cache = model(ids[:, 5:8], cache=cache, use_cache=True)
        third, _ = model(ids[:, 8:], cache=cache, use_cache=True)
        cached = torch.cat((first, second, third), dim=1)
        torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)


def test_absolute_attention_offsets_survive_uneven_chunk_and_token_decode():
    torch.manual_seed(23)
    model = DecoderLanguageModel(small_config()).eval()
    ids = torch.randint(0, 256, (1, 9))
    _, prefix_cache = model(ids[:, :5], use_cache=True)

    def trim_cache(cache):
        states = [
            AttentionState(
                state.key[:, :, 2:, :].contiguous(),
                state.value[:, :, 2:, :].contiguous(),
                key_start_position=2,
            )
            for state in cache.states
        ]
        return DecoderCache(states=states, position=5)

    chunk_logits, chunk_cache = model(ids[:, 5:7], cache=trim_cache(prefix_cache), use_cache=True)
    first_logits, token_cache = model(ids[:, 5:6], cache=trim_cache(prefix_cache), use_cache=True)
    second_logits, token_cache = model(ids[:, 6:7], cache=token_cache, use_cache=True)
    torch.testing.assert_close(
        chunk_logits,
        torch.cat((first_logits, second_logits), dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert chunk_cache.position == token_cache.position == 7
    for state in (*chunk_cache.states, *token_cache.states):
        assert isinstance(state, AttentionState)
        assert state.key_start_position == 2
        assert state.key.shape[2] == 5


def test_attention_cache_mismatches_fail_with_specific_errors():
    attention = CausalSelfAttention(small_config()).eval()
    hidden = torch.zeros(1, 2, 32)
    positions = torch.arange(4, 6)

    wrong_batch = AttentionState(torch.zeros(2, 2, 4, 8), torch.zeros(2, 2, 4, 8), 0)
    with pytest.raises(ValueError, match="shape must match the input batch"):
        attention(hidden, positions, wrong_batch, position_start=4)

    wrong_dtype = AttentionState(
        torch.zeros(1, 2, 4, 8, dtype=torch.float64),
        torch.zeros(1, 2, 4, 8, dtype=torch.float64),
        0,
    )
    with pytest.raises(ValueError, match="same dtype"):
        attention(hidden, positions, wrong_dtype, position_start=4)

    wrong_device = AttentionState(
        torch.zeros(1, 2, 4, 8, device="meta"),
        torch.zeros(1, 2, 4, 8, device="meta"),
        0,
    )
    with pytest.raises(ValueError, match="same device"):
        attention(hidden, positions, wrong_device, position_start=4)

    wrong_offset = AttentionState(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8), 0)
    with pytest.raises(ValueError, match="end at position_start"):
        attention(hidden, torch.arange(5, 7), wrong_offset, position_start=5)


def test_attention_emits_explicit_state_and_accepts_legacy_tuple_cache():
    attention = CausalSelfAttention(small_config()).eval()
    first_hidden = torch.zeros(1, 2, 32)
    first_positions = torch.arange(2)
    _, state = attention(first_hidden, first_positions, use_cache=True)
    assert isinstance(state, AttentionState)
    assert state.key_start_position == 0

    next_hidden = torch.zeros(1, 1, 32)
    _, next_state = attention(
        next_hidden,
        torch.tensor([2]),
        state=(state.key, state.value),
        use_cache=True,
        position_start=2,
    )
    assert isinstance(next_state, AttentionState)
    assert next_state.key_start_position == 0
    assert next_state.key.shape[2] == 3


def test_tied_embedding_is_counted_once():
    model = DecoderLanguageModel(small_config())
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.parameter_count() == sum(
        parameter.numel() for parameter in {id(p): p for p in model.parameters()}.values()
    )


def test_rope_preserves_pairwise_vector_norm():
    rope = RotaryEmbedding(8)
    x = torch.randn(2, 3, 5, 8)
    positions = torch.arange(5)
    rotated = rope(x, positions)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)
