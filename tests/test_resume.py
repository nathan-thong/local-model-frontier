import json
import math

import pytest
import torch
from torch import nn

import frontier.training.trainer as trainer_module
from frontier.cli import _evaluate
from frontier.config import EvalConfig, ModelConfig, ProfileConfig, RunConfig, TrainConfig
from frontier.data.corpus import sha256_file
from frontier.tokenization import load_tokenizer_artifact
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
    summary = json.loads((tmp_path / "full" / "summary.json").read_text(encoding="utf-8"))
    assert summary["data"]["is_test_fixture"] is True
    assert summary["data"]["synthetic_fixture"] is True
    assert summary["data"]["content_origin"] == "synthetic"
    assert (
        summary["tokenizer"]["artifact_sha256"]
        == summary["evaluation"]["protocol"]["tokenizer_sha256"]
    )
    assert load_tokenizer_artifact(tmp_path / "full" / "tokenizer.json").name == "utf8-byte-v1"
    assert summary["data"]["manifest_sha256"] == sha256_file(
        tmp_path / "full" / "data_manifest.json"
    )
    assert summary["data"]["train_tokens_including_special_tokens"] > 0
    training = summary["training"]
    assert training["compute_estimator"] == "decoder-module-mac-v1"
    assert training["compute_status"] == "estimated"
    assert training["estimated_flops"] == training["estimated_macs"] * 6
    assert training["token_budget_planned"] == training["tokens_seen"]
    assert training["token_budget_overshoot"] == 0
    assert training["memory"]["schema_version"] == 1
    assert "optimizer step" in training["memory"]["accelerator_peak_scope"]
    assert training["memory"]["accelerator_peak_over_optimizer_steps"] == {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    reevaluated = _evaluate(tmp_path / "full")
    assert reevaluated["protocol"]["tokenizer_sha256"] == summary["tokenizer"]["artifact_sha256"]
    updated_summary = json.loads((tmp_path / "full" / "summary.json").read_text(encoding="utf-8"))
    assert (
        updated_summary["evaluation"]["protocol"]["tokenizer_sha256"]
        == summary["tokenizer"]["artifact_sha256"]
    )
    train(interrupted, stop_after_steps=2)
    train(interrupted, resume=True)
    full_state = load_checkpoint(tmp_path / "full" / "checkpoints" / "last.pt")
    resumed_state = load_checkpoint(tmp_path / "interrupted" / "checkpoints" / "last.pt")
    assert full_state["step"] == resumed_state["step"] == 4
    assert full_state["tokens_seen"] == resumed_state["tokens_seen"]
    for name, tensor in full_state["model"].items():
        torch.testing.assert_close(tensor, resumed_state["model"][name], atol=0, rtol=0)


def test_nonfinite_gradient_stops_before_update_and_persists_failure(tmp_path, monkeypatch):
    class NonfiniteGradientModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(config.vocab_size))
            self.logits.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))

        def forward(self, input_ids):
            logits = self.logits.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
            return logits, None

    monkeypatch.setattr(trainer_module, "DecoderLanguageModel", NonfiniteGradientModel)
    config = make_config(tmp_path / "nonfinite")
    config.train.max_steps = 1
    config.train.max_tokens = None

    with pytest.raises(FloatingPointError, match="non-finite gradient norm"):
        train(config)

    summary = json.loads((tmp_path / "nonfinite" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    failure = summary["training"]["failure"]
    assert failure["kind"] == "non_finite_gradient_norm"
    assert failure["step"] == 1
    assert failure["tokens_seen_before_step"] == 0
    assert failure["optimizer_updates_before_step"] == 0
    assert failure["loss_finite"] is True
    assert math.isfinite(failure["loss"])
    assert failure["gradient_norm"] is None
    assert failure["gradient_norm_finite"] is False
    assert failure["optimizer_step_applied"] is False
    metrics = [
        json.loads(line)
        for line in (tmp_path / "nonfinite" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failure_rows = [row for row in metrics if row["event"] == "training_failure"]
    assert len(failure_rows) == 1
    assert failure_rows[0]["kind"] == "non_finite_gradient_norm"


def test_nonfinite_loss_stops_and_persists_failure(tmp_path, monkeypatch):
    class NonfiniteLossModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(config.vocab_size))

        def forward(self, input_ids):
            logits = self.logits.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
            return logits * float("nan"), None

    monkeypatch.setattr(trainer_module, "DecoderLanguageModel", NonfiniteLossModel)
    config = make_config(tmp_path / "nonfinite-loss")
    config.train.max_steps = 1
    config.train.max_tokens = None

    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        train(config)

    summary = json.loads((tmp_path / "nonfinite-loss" / "summary.json").read_text(encoding="utf-8"))
    failure = summary["training"]["failure"]
    assert summary["status"] == "failed"
    assert failure["kind"] == "non_finite_loss"
    assert failure["loss"] is None
    assert failure["loss_finite"] is False
    assert failure["optimizer_step_applied"] is False
