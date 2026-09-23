import pytest

from frontier.config import ModelConfig, ProfileConfig
from frontier.models import DecoderLanguageModel
from frontier.profiling.benchmark import profile_model


def test_profile_records_actual_and_analytic_kv_bytes():
    model = DecoderLanguageModel(
        ModelConfig(
            vocab_size=259,
            max_seq_len=16,
            width=16,
            layers=1,
            query_heads=4,
            kv_heads=2,
            ffn_width=32,
        )
    ).eval()
    profile = profile_model(
        model,
        ProfileConfig(prompt_lengths=[4], decode_tokens=3, warmup_steps=0, repeats=1),
    )
    workload = profile["workloads"][0]
    assert (
        workload["actual_kv_bytes_after_decode"]
        == workload["theoretical_kv_bytes_after_decode_loop"]
    )
    assert workload["actual_kv_bytes_after_decode"] > 0
    assert (
        workload["actual_state_storage_bytes_after_decode"]
        == workload["actual_kv_bytes_after_decode"]
    )
    assert workload["persistent_state_bytes_after_decode"] is None
    assert workload["active_state_bytes_after_decode"] == workload["actual_kv_bytes_after_decode"]
    assert workload["prefill_accelerator_memory"]["peak_allocated_bytes"] is None
    assert workload["decode_accelerator_memory"]["peak_allocated_bytes"] is None
    parameter_bytes = sum(
        parameter.untyped_storage().nbytes()
        for parameter in {id(p): p for p in model.parameters()}.values()
    )
    assert profile["model_tensor_bytes"] >= parameter_bytes


class CountingDecoder(DecoderLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.incremental_forwards = 0

    def forward(self, input_ids, cache=None, use_cache=False):
        if cache is not None:
            self.incremental_forwards += 1
        return super().forward(input_ids, cache=cache, use_cache=use_cache)


@pytest.mark.parametrize("decode_tokens", [1, 2, 3])
def test_decode_timing_counts_exactly_the_reported_incremental_forwards(decode_tokens):
    model = CountingDecoder(
        ModelConfig(
            vocab_size=259,
            max_seq_len=16,
            width=16,
            layers=1,
            query_heads=4,
            kv_heads=2,
            ffn_width=32,
        )
    ).eval()
    profile = profile_model(
        model,
        ProfileConfig(prompt_lengths=[4], decode_tokens=decode_tokens, warmup_steps=0, repeats=2),
    )

    workload = profile["workloads"][0]
    assert model.incremental_forwards == 2 * decode_tokens
    assert workload["timed_decode_forwards_per_repeat"] == decode_tokens
    assert workload["decode_repeats"] == 2
    assert workload["generated_tokens_per_repeat"] == decode_tokens + 1
    assert workload["state_position_tokens_after_decode"] == 4 + decode_tokens
    assert workload["theoretical_kv_cache_tokens_after_decode_loop"] == 4 + decode_tokens
    assert (
        workload["actual_kv_bytes_after_decode"]
        == workload["theoretical_kv_bytes_after_decode_loop"]
    )
    assert len(workload["prefill_to_first_token_latency"]["repetitions_seconds"]) == 2
    assert len(workload["decode_latency"]["repetitions_seconds"]) == 2
    assert workload["prefill_to_first_token_latency"]["sample_stddev_seconds"] is not None
    assert workload["decode_latency"]["sample_stddev_seconds"] is not None
    assert profile["profile_schema_version"] == 3
    assert workload["prefill_accelerator_memory_scope"].startswith("prefill warmup")
    assert "cached incremental decode" in workload["decode_accelerator_memory_scope"]


def test_profile_rejects_prompt_lengths_that_would_be_clipped():
    model = DecoderLanguageModel(
        ModelConfig(
            vocab_size=259,
            max_seq_len=16,
            width=16,
            layers=1,
            query_heads=4,
            kv_heads=2,
            ffn_width=32,
        )
    )

    with pytest.raises(ValueError, match="exceeds model.max_seq_len"):
        profile_model(
            model,
            ProfileConfig(prompt_lengths=[14], decode_tokens=3, warmup_steps=0, repeats=1),
        )
