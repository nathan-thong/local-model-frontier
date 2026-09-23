import hashlib
import json
from pathlib import Path

import pytest

from frontier.cli import main
from frontier.experiments.compare import compare_seeded_runs
from frontier.experiments.evidence import export_evidence_bundle, verify_evidence_bundle
from frontier.experiments.results import write_json


def _write_run(root: Path, run_id: str, seed: int, *, checkpoint: bool = True) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    data_dir = root / f"data-{run_id}"
    data_dir.mkdir()
    train_bytes = b"training corpus stays outside the bundle\n"
    validation_bytes = b"validation corpus stays outside the bundle\n"
    (data_dir / "train.txt").write_bytes(train_bytes)
    (data_dir / "validation.txt").write_bytes(validation_bytes)
    manifest = {
        "schema_version": 3,
        "is_test_fixture": False,
        "content_origin": "synthetic",
        "source": "example/Stories",
        "source_path_name": r"C:\Users\Alice\Private\stories.txt",
        "source_sha256": hashlib.sha256(b"raw source").hexdigest(),
        "source_metadata_sha256": hashlib.sha256(b"source metadata").hexdigest(),
        "source_metadata": {
            "dataset": "example/Stories",
            "dataset_revision": "abc1234",
            "license": "CC-BY-4.0",
            "source_url": "https://example.org/file?token=private-secret",
            "raw_prefix_sha256": hashlib.sha256(b"raw source").hexdigest(),
        },
        "preprocessing": {
            "schema_version": 1,
            "input_format": "UTF-8 text with one document per non-empty line",
            "line_handling": "strip leading and trailing Unicode whitespace",
            "duplicate_identity": "Unicode NFC then collapse whitespace",
            "split_unit": "normalized document identity",
        },
        "preprocessing_sha256": hashlib.sha256(b"preprocessing").hexdigest(),
        "split_seed": seed,
        "validation_fraction_requested": 0.2,
        "document_count": 2,
        "unique_document_count": 2,
        "train_document_count": 1,
        "validation_document_count": 1,
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_bytes).hexdigest(),
        "train_utf8_bytes": len(train_bytes),
        "validation_utf8_bytes": len(validation_bytes),
        "split_unit": "normalized document identity",
    }
    # Same JSON values, different serialization: both file hashes must remain explicit.
    (data_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_json(run_dir / "data_manifest.json", manifest)
    run_manifest_sha = hashlib.sha256((run_dir / "data_manifest.json").read_bytes()).hexdigest()

    config = {
        "seed": seed,
        "output_dir": str(run_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "model": {
            "vocab_size": 259,
            "max_seq_len": 8,
            "width": 16,
            "layers": 1,
            "query_heads": 2,
            "kv_heads": 2,
            "ffn_width": 32,
        },
        "train": {
            "batch_size": 1,
            "context_length": 8,
            "max_steps": 1,
            "max_tokens": None,
            "learning_rate": 0.001,
            "min_learning_rate": 0.0001,
            "warmup_steps": 0,
            "gradient_accumulation": 1,
            "eval_interval": 1,
            "checkpoint_interval": 1,
            "mixed_precision": "fp32",
            "deterministic": True,
        },
        "evaluation": {
            "stride": 4,
            "max_validation_documents": 1,
            "tasks_path": r"C:\Users\Alice\Private\tasks.jsonl",
        },
        "profiling": {"prompt_lengths": [2], "decode_tokens": 2, "warmup_steps": 0, "repeats": 1},
        "baseline_run": r"C:\Users\Alice\Private\baseline",
    }
    write_json(run_dir / "resolved_config.json", config)
    tokenizer = {"name": "utf8-byte-v1", "vocab_size": 259, "artifact_sha256": "a" * 64}
    write_json(run_dir / "tokenizer.json", tokenizer)
    if checkpoint:
        (run_dir / "checkpoints").mkdir()
        (run_dir / "checkpoints" / "last.pt").write_bytes(b"checkpoint-not-included")

    summary = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "completed",
        "resolved_config": config,
        "data": {
            "root": str(data_dir.resolve()),
            "source": "example/Stories",
            "content_origin": "synthetic",
            "is_test_fixture": False,
            "manifest_sha256": run_manifest_sha,
            "source_metadata": manifest["source_metadata"],
        },
        "tokenizer": {"name": "utf8-byte-v1", "vocab_size": 259, "artifact_sha256": "a" * 64},
        "environment": {
            "frontier": "0.1.0",
            "platform": "Windows-11-10.0.26200-SP0",
            "python": "3.12.14 (build details)",
            "torch": "2.14.0+cpu",
            "numpy": "2.5.3",
            "cuda_runtime": None,
            "cuda_available": False,
            "device": "cpu",
            "device_name": "AMD64 Family CPU",
            "cpu_count": 8,
            "git_revision": "b" * 40,
            "git_worktree_clean": True,
        },
        "training": {
            "step": 1,
            "optimizer_updates": 1,
            "tokens_seen": 8,
            "estimated_flops": 1000,
            "compute_estimator": "decoder-module-mac-v1",
            "compute_status": "estimated",
            "tokens_per_second": 100.0,
            "run_local_path": r"C:\Users\Alice\Private\never-export.json",
        },
        "evaluation": {
            "perplexity": {
                "mean_nll": 1.25,
                "perplexity": 3.49,
                "scored_tokens": 20,
                "scored_utf8_bytes": 18,
                "per_document": [{"document_text": "must not export"}],
            },
            "protocol": {
                "metrics_schema_version": 2,
                "context_length": 8,
                "stride": 4,
                "tokenizer": "utf8-byte-v1",
                "tokenizer_sha256": "a" * 64,
                "dataset_sha256": manifest["validation_sha256"],
                "tasks_path": r"C:\Users\Alice\Private\tasks.jsonl",
            },
        },
        "profiling": {
            "profile_schema_version": 4,
            "parameter_count": 100,
            "model_tensor_bytes": 400,
            "device": "cpu",
            "checkpoint_artifact_bytes": 23,
            "workloads": [
                {
                    "prompt_tokens": 2,
                    "decode_tokens_per_second": 90.0,
                    "active_state_bytes_after_decode": 64,
                    "prefill_cpu_memory": {"status": "measured", "peak_rss_increase_bytes": 1024},
                }
            ],
        },
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "evaluation.json", summary["evaluation"])
    write_json(run_dir / "profile.json", summary["profiling"])
    return run_dir


def test_export_evidence_scrubs_paths_excludes_large_artifacts_and_verifies_hashes(tmp_path):
    run_dir = _write_run(tmp_path, "candidate-seed-17-abc123", 17)
    output = tmp_path / "portable-evidence"

    assert main(["export-evidence", "--run-dirs", str(run_dir), "--output-dir", str(output)]) == 0
    report = verify_evidence_bundle(output)
    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    run = bundle["runs"][0]
    exported_config = json.loads((output / run["configuration_file"]).read_text(encoding="utf-8"))
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*.json"))

    assert report["status"] == "verified"
    assert run["data"]["prepared_manifest_semantic_match"] is True
    assert run["data"]["run_manifest_sha256"] != run["data"]["prepared_manifest_sha256"]
    assert run["artifact_availability"]["checkpoint"]["available"] is True
    assert run["artifact_availability"]["checkpoint"]["included"] is False
    assert run["artifact_availability"]["task_examples"]["included"] is False
    assert exported_config["data_dir"] == "data/prepared-corpus"
    assert exported_config["evaluation"]["tasks_path"] == "data/tasks.jsonl"
    assert exported_config["output_dir"] == "runs/candidate-seed-17-abc123-reproduction"
    assert "C:\\Users\\Alice" not in serialized
    assert "private-secret" not in serialized
    assert "must not export" not in serialized
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob("train.txt"))


def test_export_evidence_sanitizes_comparison_paths_and_checks_run_membership(tmp_path):
    baseline_id = "baseline-seed-17-abc123"
    candidate_id = "candidate-seed-17-def456"
    baseline = _write_run(tmp_path, baseline_id, 17)
    candidate = _write_run(tmp_path, candidate_id, 17)
    comparison_path = tmp_path / "comparison.json"
    write_json(
        comparison_path,
        {
            "schema_version": 3,
            "comparison_type": "paired multi-seed",
            "claim_scope": "synthetic corpus metric",
            "evidence_scope": "synthetic corpus metric",
            "comparable_for_metric_comparison": True,
            "comparable_for_improvement_claim": False,
            "comparable_for_human_corpus_claim": False,
            "minimum_seeds_for_claim": 3,
            "source_revision": "b" * 40,
            "source_worktrees_clean": True,
            "seeds": [17],
            "metrics": {"paired_mean_nll_delta_candidate_minus_baseline": {"mean": -0.01}},
            "pairwise_results": [
                {
                    "baseline_run": f"C:\\Users\\Alice\\runs\\{baseline_id}",
                    "candidate_run": f"C:\\Users\\Alice\\runs\\{candidate_id}",
                    "perplexity_delta_candidate_minus_baseline": -0.1,
                    "comparable_for_metric_comparison": True,
                    "checks": {"same_seed": True, "same_training_data": True},
                    "eligibility_blockers": [],
                }
            ],
            "group_checks": {"all_run_artifacts_valid": True},
            "claim_checks": {"research_data_provenance_complete": True},
            "claim_blockers": [],
            "human_corpus_claim_blockers": ["synthetic corpus"],
        },
    )
    output = tmp_path / "paired-evidence"

    export = main(
        [
            "export-evidence",
            "--run-dirs",
            str(baseline),
            str(candidate),
            "--output-dir",
            str(output),
            "--comparison",
            str(comparison_path),
        ]
    )
    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))

    assert export == 0
    assert bundle["comparison"]["pairwise_results"][0]["baseline_run_id"] == baseline_id
    assert bundle["comparison"]["pairwise_results"][0]["candidate_run_id"] == candidate_id
    assert "C:\\Users\\Alice" not in (output / "bundle.json").read_text(encoding="utf-8")
    assert (
        bundle["comparison"]["metrics"]["paired_mean_nll_delta_candidate_minus_baseline"]["mean"]
        == -0.01
    )


def test_verify_evidence_detects_tampering_and_unlisted_artifacts(tmp_path):
    run_dir = _write_run(tmp_path, "candidate-seed-17-abc123", 17, checkpoint=False)
    output = tmp_path / "portable-evidence"
    from frontier.experiments.evidence import export_evidence_bundle

    export_evidence_bundle([run_dir], output)
    bundle_path = output / "bundle.json"
    original_bundle = bundle_path.read_bytes()
    bundle_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_evidence_bundle(output)

    bundle_path.write_bytes(original_bundle)
    (output / "checkpoints").mkdir()
    (output / "checkpoints" / "last.pt").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unlisted files"):
        verify_evidence_bundle(output)
    (output / "checkpoints" / "last.pt").unlink()
    (output / "nested").mkdir()
    (output / "nested" / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted files"):
        verify_evidence_bundle(output)


@pytest.mark.parametrize(
    ("comparison", "message"),
    [
        (
            {
                "seeds": [17],
                "minimum_seeds_for_claim": 1,
                "comparable_for_improvement_claim": True,
                "pairwise_results": [],
            },
            "required distinct seed floor",
        ),
        (
            {
                "seeds": [42],
                "minimum_seeds_for_claim": 3,
                "comparable_for_improvement_claim": False,
                "pairwise_results": [
                    {
                        "baseline_run": "baseline-seed-17-abc123",
                        "candidate_run": "candidate-seed-17-def456",
                    }
                ],
            },
            "absent from its declared seed list",
        ),
    ],
)
def test_export_rejects_ineligible_comparisons_without_partial_bundle(
    tmp_path, comparison, message
):
    baseline = _write_run(tmp_path, "baseline-seed-17-abc123", 17)
    candidate = _write_run(tmp_path, "candidate-seed-17-def456", 17)
    comparison_path = tmp_path / "comparison.json"
    write_json(comparison_path, comparison)
    output = tmp_path / "portable-evidence"

    with pytest.raises(ValueError, match=message):
        export_evidence_bundle([baseline, candidate], output, comparison_path)

    assert not output.exists()
    assert not list(tmp_path.glob(".portable-evidence.*"))
    assert export_evidence_bundle([baseline], output)["status"] == "verified"


def test_export_recomputes_and_downgrades_forged_three_seed_eligibility(tmp_path):
    seeds = (17, 42, 123)
    baseline = [_write_run(tmp_path, f"baseline-seed-{seed}-abc123", seed) for seed in seeds]
    candidate = [_write_run(tmp_path, f"candidate-seed-{seed}-def456", seed) for seed in seeds]
    comparison_path = tmp_path / "comparison.json"
    comparison = compare_seeded_runs(baseline, candidate, comparison_path)
    honest_output = tmp_path / "honest-evidence"
    export_evidence_bundle([*baseline, *candidate], honest_output, comparison_path)
    honest_export = json.loads((honest_output / "bundle.json").read_text(encoding="utf-8"))[
        "comparison"
    ]
    assert honest_export["eligibility_reconstruction"] == "verified"
    assert honest_export["comparable_for_improvement_claim"] is False
    assert honest_export["metrics"]["baseline_perplexity"]["mean"] == 3.49

    comparison["comparable_for_metric_comparison"] = True
    comparison["comparable_for_improvement_claim"] = True
    comparison["comparable_for_human_corpus_claim"] = True
    for pair in comparison["pairwise_results"]:
        pair["comparable_for_metric_comparison"] = True
        pair["comparable_for_improvement_claim"] = True
    write_json(comparison_path, comparison)
    output = tmp_path / "portable-evidence"

    export_evidence_bundle([*baseline, *candidate], output, comparison_path)
    exported = json.loads((output / "bundle.json").read_text(encoding="utf-8"))["comparison"]

    assert exported["eligibility_reconstruction"] == "mismatch_or_unavailable"
    assert exported["comparable_for_metric_comparison"] is False
    assert exported["comparable_for_improvement_claim"] is False
    assert exported["comparable_for_human_corpus_claim"] is False


def test_export_downgrades_unrecomputed_one_seed_metric_eligibility(tmp_path):
    baseline_id = "baseline-seed-17-abc123"
    candidate_id = "candidate-seed-17-def456"
    baseline = _write_run(tmp_path, baseline_id, 17)
    candidate = _write_run(tmp_path, candidate_id, 17)
    comparison_path = tmp_path / "comparison.json"
    write_json(
        comparison_path,
        {
            "schema_version": 3,
            "comparison_type": "paired multi-seed",
            "evidence_scope": "synthetic corpus metric",
            "claim_scope": "synthetic corpus metric",
            "comparable_for_metric_comparison": True,
            "comparable_for_improvement_claim": False,
            "comparable_for_human_corpus_claim": False,
            "minimum_seeds_for_claim": 3,
            "source_revision": "b" * 40,
            "source_worktrees_clean": True,
            "seeds": [17],
            "metrics": {},
            "group_checks": {},
            "claim_checks": {},
            "claim_blockers": [],
            "human_corpus_claim_blockers": [],
            "pairwise_results": [
                {
                    "baseline_run": str(baseline),
                    "candidate_run": str(candidate),
                    "comparable_for_metric_comparison": True,
                    "comparable_for_improvement_claim": False,
                    "evidence_scope": "synthetic corpus metric",
                    "improvement_claim_requires": "three matched seeds",
                    "checks": {"same_seed": True},
                    "eligibility_blockers": [],
                }
            ],
        },
    )

    output = tmp_path / "portable-evidence"
    export_evidence_bundle([baseline, candidate], output, comparison_path)
    comparison = json.loads((output / "bundle.json").read_text(encoding="utf-8"))["comparison"]

    assert comparison["eligibility_reconstruction"] == "not_recomputed_below_three_pairs"
    assert comparison["comparable_for_metric_comparison"] is False
    assert comparison["comparable_for_improvement_claim"] is False
    assert comparison["comparable_for_human_corpus_claim"] is False
    assert comparison["claim_reconstruction_blocked"] is True
    pair = comparison["pairwise_results"][0]
    assert pair["comparable_for_metric_comparison"] is False
    assert pair["eligibility_reconstruction"] == "not_recomputed_below_three_pairs"
