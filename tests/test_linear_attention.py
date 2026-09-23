import pytest
import torch
from torch.nn import functional as F

from frontier.config import ModelConfig
from frontier.models import DecoderLanguageModel
from frontier.models.sequence import (
    LinearAttentionState,
    NormalizedCausalLinearAttention,
)
from frontier.profiling.benchmark import _state_memory_accounting


def make_config(**overrides):
    values = {
        "vocab_size": 259,
        "max_seq_len": 128,
        "width": 32,
        "layers": 1,
        "query_heads": 4,
        "kv_heads": 2,
        "ffn_width": 64,
        "normalization": "rmsnorm",
        "ffn": "gelu",
        "position": "rope",
        "residual_topology": "pre_norm",
        "sequence_types": ["normalized_linear_attention_reference"],
        "dropout": 0.0,
        "bias": False,
        "tie_embeddings": True,
    }
    values.update(overrides)
    return ModelConfig(**values)


def slow_prefix_reference(module, hidden_states, positions):
    """Recompute each prefix independently as an O(T^2) oracle."""
    batch, time, width = hidden_states.shape
    q = (
        module.q_proj(hidden_states)
        .view(batch, time, module.query_heads, module.head_dim)
        .transpose(1, 2)
    )
    k = (
        module.k_proj(hidden_states)
        .view(batch, time, module.kv_heads, module.head_dim)
        .transpose(1, 2)
    )
    v = (
        module.v_proj(hidden_states)
        .view(batch, time, module.kv_heads, module.head_dim)
        .transpose(1, 2)
    )
    if module.rope is not None:
        q = module.rope(q, positions)
        k = module.rope(k, positions)
    q_feature = F.elu(q.float()) + 1.0
    k_feature = F.elu(k.float()) + 1.0
    value = v.float()
    grouped_q = q_feature.view(batch, module.kv_heads, module.groups, time, module.head_dim)
    outputs = []
    for index in range(time):
        prefix_key = k_feature[:, :, : index + 1]
        prefix_value = value[:, :, : index + 1]
        key_value_sum = torch.einsum("bhtd,bhte->bhde", prefix_key, prefix_value)
        key_feature_sum = prefix_key.sum(dim=2)
        query = grouped_q[:, :, :, index]
        numerator = torch.einsum("bkgd,bkdf->bkgf", query, key_value_sum)
        denominator = torch.einsum("bkgd,bkd->bkg", query, key_feature_sum)
        outputs.append(
            (numerator / (denominator.unsqueeze(-1) + module.epsilon)).reshape(
                batch, module.query_heads, module.head_dim
            )
        )
    attended = torch.stack(outputs, dim=2)
    attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
    return module.out_dropout(module.out_proj(attended.to(hidden_states.dtype)))


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("position", ["rope", "learned"])
def test_linear_attention_matches_independent_prefix_oracle(kv_heads, position):
    torch.manual_seed(31 + kv_heads)
    module = NormalizedCausalLinearAttention(
        make_config(kv_heads=kv_heads, position=position)
    ).eval()
    positions = torch.arange(7)
    hidden = torch.randn(2, 7, 32)

    actual, present = module(hidden, positions)
    expected = slow_prefix_reference(module, hidden, positions)

    assert present is None
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_linear_attention_forward_and_parameter_gradients_match_prefix_oracle():
    torch.manual_seed(73)
    module = NormalizedCausalLinearAttention(make_config()).eval()
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    positions = torch.arange(8)
    probe = torch.randn_like(hidden)
    parameters = list(module.parameters())

    actual = module(hidden, positions)[0]
    actual_grads = torch.autograd.grad((actual * probe).sum(), [hidden, *parameters])

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    expected = slow_prefix_reference(module, reference_hidden, positions)
    expected_grads = torch.autograd.grad((expected * probe).sum(), [reference_hidden, *parameters])

    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(actual_grads[0], expected_grads[0], atol=3e-6, rtol=3e-6)
    for actual_grad, expected_grad in zip(actual_grads[1:], expected_grads[1:], strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("position", ["rope", "learned"])
def test_linear_attention_cpu_bfloat16_autocast_keeps_recurrent_state_fp32(position):
    torch.manual_seed(211)
    module = NormalizedCausalLinearAttention(make_config(position=position)).eval()
    hidden = torch.randn(1, 7, 32, requires_grad=True)
    positions = torch.arange(7)

    reference = module(hidden, positions)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, state = module(hidden, positions, use_cache=True)

    assert state is not None
    assert state.key_value_sum.dtype == torch.float32
    assert state.key_feature_sum.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)

    actual_grads = torch.autograd.grad(actual.float().square().mean(), hidden)[0]
    assert torch.isfinite(actual_grads).all()


@pytest.mark.parametrize("position", ["rope", "learned"])
def test_decoder_full_chunk_and_token_cache_logits_match(position):
    torch.manual_seed(103)
    config = make_config(
        position=position,
        layers=2,
        sequence_types=["normalized_linear_attention_reference"] * 2,
    )
    model = DecoderLanguageModel(config).eval()
    token_ids = torch.randint(0, config.vocab_size - 3, (2, 11))

    full_logits, no_cache = model(token_ids)
    pieces = []
    cache = None
    start = 0
    for length in (3, 4, 1, 3):
        logits, cache = model(token_ids[:, start : start + length], cache=cache, use_cache=True)
        pieces.append(logits)
        start += length

    cached_logits = torch.cat(pieces, dim=1)
    assert no_cache is None
    assert cache is not None and cache.position == token_ids.shape[1]
    assert all(state.next_position == token_ids.shape[1] for state in cache.states)
    torch.testing.assert_close(cached_logits, full_logits, atol=2e-5, rtol=2e-5)


def test_linear_attention_is_causal_and_resets_by_omitting_state():
    torch.manual_seed(17)
    module = NormalizedCausalLinearAttention(make_config()).eval()
    positions = torch.arange(12)
    hidden = torch.randn(1, 12, 32)
    changed_future = hidden.clone()
    changed_future[:, 7:] += torch.randn_like(changed_future[:, 7:]) * 3

    output, state = module(hidden, positions, use_cache=True)
    changed_output = module(changed_future, positions)[0]
    reset_output, reset_state = module(hidden, positions, state=None, use_cache=True)

    torch.testing.assert_close(output[:, :7], changed_output[:, :7], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output, reset_output, atol=0, rtol=0)
    assert state is not None and reset_state is not None
    assert state.next_position == 12


def test_linear_attention_validates_absolute_offsets_and_cache_continuity():
    module = NormalizedCausalLinearAttention(make_config()).eval()
    hidden = torch.randn(1, 7, 32)
    first_positions = torch.arange(9, 13)
    _, state = module(hidden[:, :4], first_positions, use_cache=True, position_start=9)
    assert state is not None and state.next_position == 13

    _, continued = module(
        hidden[:, 4:], torch.arange(13, 16), state=state, use_cache=True, position_start=13
    )
    assert continued is not None and continued.next_position == 16

    with pytest.raises(ValueError, match="end at position_start"):
        module(hidden[:, 4:], torch.arange(14, 17), state=state, use_cache=True, position_start=14)
    with pytest.raises(ValueError, match="contiguous"):
        module(hidden[:, :3], torch.tensor([9, 11, 10]), position_start=9)


def test_linear_attention_recurrent_state_allocation_is_fixed_with_context():
    module = NormalizedCausalLinearAttention(make_config()).eval()
    expected_bytes = (
        2 * module.kv_heads * module.head_dim * module.head_dim * 4
        + 2 * module.kv_heads * module.head_dim * 4
    )
    allocated = []
    active = []
    for length in (1, 17, 64, 96):
        hidden = torch.randn(2, length, 32)
        _, state = module(hidden, torch.arange(length), use_cache=True)
        assert state is not None
        accounting = _state_memory_accounting([module], [state])
        assert accounting["status"].startswith("exact union")
        allocated.append(accounting["allocated_bytes"])
        active.append(accounting["active_bytes"])

    assert allocated == [expected_bytes] * 4
    assert active == [expected_bytes] * 4


def test_linear_attention_state_rejects_gap_wrong_shape_and_non_fp32_state():
    module = NormalizedCausalLinearAttention(make_config()).eval()
    hidden = torch.randn(1, 5, 32)
    _, state = module(hidden, torch.arange(5), use_cache=True)
    assert state is not None

    with pytest.raises(ValueError, match="key_value_sum shape"):
        module(
            hidden[:, :1],
            torch.tensor([5]),
            state=LinearAttentionState(state.key_value_sum[:, :, :, :-1], state.key_feature_sum, 5),
            use_cache=True,
            position_start=5,
        )
    with pytest.raises(ValueError, match="must use FP32"):
        module(
            hidden[:, :1],
            torch.tensor([5]),
            state=LinearAttentionState(state.key_value_sum.half(), state.key_feature_sum, 5),
            use_cache=True,
            position_start=5,
        )
    with pytest.raises(ValueError, match="end at position_start"):
        module(hidden[:, :1], torch.tensor([6]), state=state, use_cache=True, position_start=6)


def test_linear_attention_epsilon_keeps_zero_feature_denominator_finite():
    module = NormalizedCausalLinearAttention(make_config(bias=True)).eval()
    with torch.no_grad():
        module.q_proj.weight.zero_()
        module.q_proj.bias.fill_(-1000)
        module.k_proj.weight.zero_()
        module.k_proj.bias.fill_(-1000)
    hidden = torch.zeros(1, 4, 32, requires_grad=True)

    output = module(hidden, torch.arange(4))[0]
    output.square().sum().backward()

    assert torch.isfinite(output).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_linear_attention_long_prefix_has_finite_outputs_and_gradients():
    torch.manual_seed(9)
    module = NormalizedCausalLinearAttention(make_config()).eval()
    hidden = torch.randn(1, 96, 32, requires_grad=True)

    output = module(hidden, torch.arange(96))[0]
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_linear_attention_descriptor_and_dropout_boundary_are_explicit():
    module = NormalizedCausalLinearAttention(make_config())
    descriptor = module.sequence_descriptor()

    assert descriptor.name == "normalized_linear_attention_reference"
    assert descriptor.cost.executed_work == "normalized_causal_linear_attention_token_loop"
    assert descriptor.state.growth == "constant_with_context_for_fixed_batch_and_head_dimensions"
    assert dict(descriptor.parameters)["accumulation_dtype"] == "torch.float32"
    assert dict(descriptor.parameters)["position_treatment"] == (
        "rotary_on_projected_qk_before_elu_plus_one"
    )
    with pytest.raises(ValueError, match="requires dropout=0"):
        NormalizedCausalLinearAttention(make_config(dropout=0.1))
