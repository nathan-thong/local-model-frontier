import torch

from frontier.config import ModelConfig
from frontier.models import DecoderLanguageModel
from frontier.models.feedforward import FeedForward
from frontier.models.normalization import RMSNorm
from frontier.models.position import RotaryEmbedding


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


def test_incremental_kv_logits_match_full_sequence():
    torch.manual_seed(4)
    for position in ("rope", "learned"):
        model = DecoderLanguageModel(small_config(position=position)).eval()
        ids = torch.randint(0, 256, (2, 11))
        full, _ = model(ids)
        first, cache = model(ids[:, :5], use_cache=True)
        second, cache = model(ids[:, 5:8], cache=cache, use_cache=True)
        third, _ = model(ids[:, 8:], cache=cache, use_cache=True)
        cached = torch.cat((first, second, third), dim=1)
        torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)


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
