import torch

from frontier.config import EvalConfig, ModelConfig, ProfileConfig, RunConfig, TrainConfig
from frontier.training.checkpoint import load_checkpoint
from frontier.training.trainer import train


def make_config(output_dir):
    return RunConfig(
        seed=31,
        output_dir=str(output_dir),
        model=ModelConfig(
            vocab_size=259,
            max_seq_len=16,
            width=16,
            layers=1,
            query_heads=4,
            kv_heads=2,
            ffn_width=32,
            normalization="rmsnorm",
            ffn="gelu",
            position="rope",
            residual_topology="pre_norm",
            dropout=0.0,
            tie_embeddings=True,
        ),
        train=TrainConfig(
            batch_size=2,
            context_length=16,
            max_steps=4,
            max_tokens=None,
            learning_rate=0.001,
            min_learning_rate=0.0001,
            warmup_steps=0,
            gradient_accumulation=1,
            eval_interval=2,
            checkpoint_interval=2,
            mixed_precision="fp32",
            deterministic=True,
        ),
        evaluation=EvalConfig(stride=8, max_validation_documents=1),
        profiling=ProfileConfig(prompt_lengths=[4], decode_tokens=4, warmup_steps=0, repeats=1),
    )


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    full = make_config(tmp_path / "full")
    interrupted = make_config(tmp_path / "interrupted")
    train(full)
    train(interrupted, stop_after_steps=2)
    train(interrupted, resume=True)
    full_state = load_checkpoint(tmp_path / "full" / "checkpoints" / "last.pt")
    resumed_state = load_checkpoint(tmp_path / "interrupted" / "checkpoints" / "last.pt")
    assert full_state["step"] == resumed_state["step"] == 4
    assert full_state["tokens_seen"] == resumed_state["tokens_seen"]
    for name, tensor in full_state["model"].items():
        torch.testing.assert_close(tensor, resumed_state["model"][name], atol=0, rtol=0)
