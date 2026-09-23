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
    parameter_bytes = sum(
        parameter.untyped_storage().nbytes()
        for parameter in {id(p): p for p in model.parameters()}.values()
    )
    assert profile["model_tensor_bytes"] >= parameter_bytes
