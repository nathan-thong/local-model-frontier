import hashlib
import json

import pytest

from frontier.config import ModelConfig
from frontier.data.corpus import PREPROCESSING_CONTRACT
from frontier.experiments.compare import compare_runs, compare_seeded_runs
from frontier.profiling.compute import estimate_training_compute
from frontier.tokenization import ByteTokenizer, tokenizer_artifact

TOKENIZER_ARTIFACT = tokenizer_artifact(ByteTokenizer())
TOKENIZER_HASH = TOKENIZER_ARTIFACT["artifact_sha256"]


def canonical_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_run(
    root,
    steps,
    synthetic=False,
    seed=17,
    perplexity=10.0,
    width=128,
    revision="test-revision",
    worktree_clean=True,
    content_origin=None,
):
    root.mkdir()
    origin = content_origin or ("synthetic" if synthetic else "human")
    model_config = ModelConfig(width=width, layers=4 if width == 128 else 6)
    model_config.validate()
    train_config = {
        "batch_size": 8,
        "context_length": 256,
        "max_steps": steps,
        "max_tokens": None,
        "learning_rate": 0.0003,
        "min_learning_rate": 0.00003,
        "warmup_steps": 0,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "gradient_accumulation": 1,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "mixed_precision": "fp32",
        "deterministic": True,
        "resume": False,
    }
    tokens_per_step = (
        train_config["batch_size"]
        * train_config["context_length"]
        * train_config["gradient_accumulation"]
    )
    tokens_seen = steps * tokens_per_step
    compute = estimate_training_compute(model_config, tokens_seen, train_config["context_length"])
    source_metadata = {
        "dataset": "test-corpus",
        "revision": "test-revision-1",
        "license": "CC0-test",
        "content_origin": origin,
    }
    tokenizer_record = TOKENIZER_ARTIFACT
    data_root = root / "corpus"
    data_root.mkdir()
    train_path = data_root / "train.txt"
    validation_path = data_root / "validation.txt"
    train_path.write_text("training corpus text\n", encoding="utf-8")
    validation_path.write_text("held out validation text\n", encoding="utf-8")
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    validation_hash = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    source_name = "test smoke fixture" if synthetic else "test corpus"
    manifest = {
        "schema_version": 3,
        "source": source_name,
        "is_test_fixture": synthetic,
        "content_origin": origin,
        "source_sha256": hashlib.sha256(b"source corpus").hexdigest(),
        "source_metadata": source_metadata,
        "source_metadata_sha256": canonical_hash(source_metadata),
        "preprocessing": PREPROCESSING_CONTRACT,
        "preprocessing_sha256": canonical_hash(PREPROCESSING_CONTRACT),
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
    }
    manifest_path = root / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    resolved_config = {
        "seed": seed,
        "output_dir": str(root),
        "data_dir": str(data_root),
        "tokenizer": "utf8-byte-v1",
        "model": model_config.__dict__.copy(),
        "train": train_config,
        "evaluation": {
            "stride": 128,
            "max_validation_documents": None,
            "tasks_path": None,
            "generation_max_tokens": 64,
        },
        "profiling": {
            "batch_size": 1,
            "prompt_lengths": [32, 128, 224],
            "decode_tokens": 32,
            "warmup_steps": 2,
            "repeats": 5,
        },
        "baseline_run": None,
    }
    (root / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
    )
    (root / "tokenizer.json").write_text(
        json.dumps(tokenizer_record, indent=2) + "\n", encoding="utf-8"
    )
    environment = {
        "platform": "test",
        "torch": "2.14.0",
        "numpy": "2.0",
        "frontier": "0.1.0",
        "cuda_runtime": None,
        "device": "cpu",
        "device_name": "test cpu",
        "git_revision": revision,
        "git_worktree_clean": worktree_clean,
    }
    (root / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    summary = {
        "schema_version": 2,
        "run_id": root.name,
        "status": "completed",
        "resolved_config": resolved_config,
        "environment": environment,
        "tokenizer": {
            "name": "utf8-byte-v1",
            "vocab_size": 259,
            "artifact_sha256": TOKENIZER_HASH,
        },
        "data": {
            "root": str(data_root),
            "source": source_name,
            "is_test_fixture": synthetic,
            "synthetic_fixture": synthetic,
            "content_origin": origin,
            "source_metadata": source_metadata,
            "source_metadata_sha256": canonical_hash(source_metadata),
            "source_sha256": manifest["source_sha256"],
            "preprocessing": PREPROCESSING_CONTRACT,
            "preprocessing_sha256": canonical_hash(PREPROCESSING_CONTRACT),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
        },
        "evaluation": {
            "protocol": {
                "dataset_sha256": validation_hash,
                "context_length": 256,
                "stride": 128,
                "tokenizer_sha256": TOKENIZER_HASH,
            },
            "perplexity": {"perplexity": perplexity, "mean_nll": perplexity / 4.0},
        },
        "training": {
            "step": steps,
            "optimizer_updates": steps,
            "tokens_seen": tokens_seen,
            "tokens_per_step": tokens_per_step,
            "token_budget_requested": None,
            "token_budget_planned": tokens_seen,
            "token_budget_overshoot": 0,
            "estimated_flops": compute["estimated_flops"],
            "estimated_macs": compute["estimated_macs"],
            "compute_components": compute["components"],
            "compute_status": compute["status"],
            "compute_estimator": compute["estimator"],
        },
        "profiling": {"parameter_count": 100, "model_tensor_bytes": 400},
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_comparison_accepts_matched_compute_control(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    report = compare_runs(baseline, [candidate], tmp_path / "comparison.json")
    result = report["results"][0]
    assert result["comparable_for_improvement_claim"] is False
    assert result["perplexity_delta_candidate_minus_baseline"] == 0.0


def test_comparison_rejects_compute_mismatch(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_100)
    report = compare_runs(baseline, [candidate], tmp_path / "comparison.json")
    result = report["results"][0]
    assert result["comparable_for_improvement_claim"] is False
    assert result["checks"]["training_compute_within_1_percent"] is False


def test_comparison_rejects_compute_fields_inconsistent_with_consumed_tokens(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["training"]["estimated_flops"] += 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = compare_runs(baseline, [candidate], tmp_path / "comparison.json")["results"][0]

    assert result["checks"]["candidate_artifact_valid"] is False
    assert "training_budget_valid" in result["candidate_validation_failures"]
    assert result["comparable_for_metric_comparison"] is False


def test_comparison_rejects_tampered_saved_tokenizer_artifact(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    artifact_path = candidate / "tokenizer.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["byte_to_token_id"] = "altered mapping"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = compare_runs(baseline, [candidate], tmp_path / "comparison.json")["results"][0]

    assert "tokenizer_file_valid" in result["candidate_validation_failures"]
    assert result["comparable_for_metric_comparison"] is False


def test_comparison_rejects_corpus_files_changed_after_training(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    train_path = candidate / "corpus" / "train.txt"
    train_path.write_text("changed after training\n", encoding="utf-8")

    result = compare_runs(baseline, [candidate], tmp_path / "comparison.json")["results"][0]

    assert "data_files_match_summary" in result["candidate_validation_failures"]
    assert result["comparable_for_metric_comparison"] is False


def test_synthetic_smoke_is_never_eligible_for_capability_claim(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000, synthetic=True)
    create_run(candidate, 1_000, synthetic=True)
    report = compare_runs(baseline, [candidate], tmp_path / "comparison.json")
    result = report["results"][0]
    assert result["comparable_for_metric_comparison"] is True
    assert result["comparable_for_improvement_claim"] is False
    assert result["evidence_scope"] == "synthetic infrastructure fixture"


def test_seeded_comparison_requires_and_aggregates_matched_repeats(tmp_path):
    baseline_runs = []
    candidate_runs = []
    for seed, baseline_ppl, candidate_ppl in ((17, 10.0, 9.0), (42, 11.0, 9.5), (123, 9.0, 8.5)):
        baseline = tmp_path / f"baseline-{seed}"
        candidate = tmp_path / f"candidate-{seed}"
        create_run(baseline, 1_000, seed=seed, perplexity=baseline_ppl, width=128)
        create_run(candidate, 271, seed=seed, perplexity=candidate_ppl, width=256)
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)

    report = compare_seeded_runs(baseline_runs, candidate_runs, tmp_path / "seeded-comparison.json")

    assert report["comparable_for_metric_comparison"] is True
    assert report["comparable_for_improvement_claim"] is True
    assert report["comparable_for_human_corpus_claim"] is True
    assert report["schema_version"] == 3
    assert report["seeds"] == [17, 42, 123]
    assert report["metrics"]["baseline_perplexity"]["mean"] == 10.0
    assert report["metrics"]["paired_perplexity_delta_candidate_minus_baseline"]["mean"] == -1.0


def test_seeded_synthetic_corpus_claim_is_scoped_and_not_mislabeled_as_fixture(tmp_path):
    baseline_runs = []
    candidate_runs = []
    for seed in (17, 42, 123):
        baseline = tmp_path / f"synthetic-baseline-{seed}"
        candidate = tmp_path / f"synthetic-candidate-{seed}"
        create_run(baseline, 1_000, seed=seed, content_origin="synthetic", synthetic=False)
        create_run(candidate, 1_005, seed=seed, content_origin="synthetic", synthetic=False)
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)

    report = compare_seeded_runs(baseline_runs, candidate_runs, tmp_path / "synthetic-report.json")

    assert report["comparable_for_metric_comparison"] is True
    assert report["comparable_for_improvement_claim"] is True
    assert report["comparable_for_human_corpus_claim"] is False
    assert report["claim_scope"] == "synthetic corpus metric"
    assert report["claim_scope_limitation"] == (
        "synthetic-domain result; does not establish human-authored or broad language capability"
    )
    assert report["claim_blockers"] == []
    assert "human-authored corpus origin required" in report["human_corpus_claim_blockers"][-1]


def test_seeded_comparison_rejects_too_few_seeds(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_000)
    with pytest.raises(ValueError, match="at least 3"):
        compare_seeded_runs([baseline], [candidate], tmp_path / "comparison.json")


def test_seeded_comparison_blocks_unpinned_source(tmp_path):
    baseline_runs = []
    candidate_runs = []
    for seed in (17, 42, 123):
        baseline = tmp_path / f"baseline-{seed}"
        candidate = tmp_path / f"candidate-{seed}"
        create_run(baseline, 1_000, seed=seed, revision=None)
        create_run(candidate, 1_005, seed=seed, revision=None)
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)

    report = compare_seeded_runs(baseline_runs, candidate_runs, tmp_path / "seeded-comparison.json")

    assert report["comparable_for_metric_comparison"] is True
    assert report["comparable_for_improvement_claim"] is False
    assert "all runs must record one pinned git revision" in report["claim_blockers"]


def test_seeded_comparison_blocks_dirty_source_worktrees(tmp_path):
    baseline_runs = []
    candidate_runs = []
    for seed in (17, 42, 123):
        baseline = tmp_path / f"baseline-{seed}"
        candidate = tmp_path / f"candidate-{seed}"
        create_run(baseline, 1_000, seed=seed, worktree_clean=False)
        create_run(candidate, 1_005, seed=seed, worktree_clean=False)
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)

    report = compare_seeded_runs(baseline_runs, candidate_runs, tmp_path / "dirty-comparison.json")

    assert report["comparable_for_metric_comparison"] is False
    assert report["comparable_for_improvement_claim"] is False
    assert report["group_checks"]["all_source_worktrees_clean"] is False
    assert "all runs must use a clean Git working tree" in report["claim_blockers"]


def test_caller_cannot_reduce_minimum_seed_claim_floor(tmp_path):
    with pytest.raises(ValueError, match="cannot be weakened"):
        compare_seeded_runs([], [], tmp_path / "comparison.json", minimum_seeds=2)


def test_missing_matching_hashes_do_not_pass_pairwise_comparison(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    for root in (baseline, candidate):
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["data"].pop("train_sha256")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = compare_runs(baseline, [candidate], tmp_path / "comparison.json")["results"][0]

    assert result["checks"]["same_training_data"] is False
    assert result["checks"]["baseline_artifact_valid"] is False
    assert result["checks"]["candidate_artifact_valid"] is False
    assert result["comparable_for_metric_comparison"] is False


def test_unknown_corpus_origin_is_not_inferred_from_fixture_flag(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000, content_origin="unknown")
    create_run(candidate, 1_005, content_origin="unknown")

    result = compare_runs(baseline, [candidate], tmp_path / "comparison.json")["results"][0]

    assert result["evidence_scope"] == "corpus origin unknown"
    assert "explicit_corpus_content_origin_missing" in result["eligibility_blockers"]
    assert result["comparable_for_metric_comparison"] is True
    assert result["comparable_for_improvement_claim"] is False


def test_missing_tokenizer_artifact_hash_preserves_metrics_but_blocks_claim(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    for root in (baseline, candidate):
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["tokenizer"].pop("artifact_sha256")
        summary["evaluation"]["protocol"].pop("tokenizer_sha256")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = compare_runs(baseline, [candidate], tmp_path / "comparison.json")["results"][0]

    assert result["comparable_for_metric_comparison"] is False
    assert result["perplexity_delta_candidate_minus_baseline"] == 0.0
    assert "baseline_tokenizer_artifact_hash_missing_or_invalid" in result["eligibility_blockers"]
    assert "candidate_tokenizer_artifact_hash_missing_or_invalid" in result["eligibility_blockers"]
    assert result["comparable_for_improvement_claim"] is False


def test_missing_dataset_license_preserves_metrics_but_blocks_research_eligibility(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    for root in (baseline, candidate):
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["data"]["source_metadata"].pop("license")
        summary["data"]["source_metadata_sha256"] = canonical_hash(
            summary["data"]["source_metadata"]
        )
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = compare_runs(baseline, [candidate], tmp_path / "comparison.json")["results"][0]

    assert result["comparable_for_metric_comparison"] is True
    assert "baseline_research_data_provenance_missing_or_invalid" in result["eligibility_blockers"]
    assert "candidate_research_data_provenance_missing_or_invalid" in result["eligibility_blockers"]


def test_missing_source_hash_preserves_metric_values_but_blocks_claim(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    for root in (baseline, candidate):
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["data"].pop("source_sha256")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = compare_runs(baseline, [candidate], tmp_path / "comparison.json")["results"][0]

    assert result["comparable_for_metric_comparison"] is True
    assert result["perplexity_delta_candidate_minus_baseline"] == 0.0
    assert "baseline_research_data_provenance_missing_or_invalid" in result["eligibility_blockers"]


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("data", "train_sha256"),
        ("data", "validation_sha256"),
        ("tokenizer", "name"),
        ("evaluation", "protocol"),
        ("training", "compute_estimator"),
    ),
)
def test_matching_missing_metadata_fails_closed(tmp_path, section, field):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    for root in (baseline, candidate):
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary[section].pop(field)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = compare_runs(baseline, [candidate], tmp_path / "comparison.json")["results"][0]

    assert result["comparable_for_metric_comparison"] is False


def test_missing_flops_and_overflowed_perplexity_return_structured_metrics(tmp_path):
    baseline_runs = []
    candidate_runs = []
    for seed in (17, 42, 123):
        baseline = tmp_path / f"baseline-{seed}"
        candidate = tmp_path / f"candidate-{seed}"
        create_run(baseline, 1_000, seed=seed)
        create_run(candidate, 271, seed=seed, width=256)
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)
    missing_flops_summary = json.loads(
        (candidate_runs[0] / "summary.json").read_text(encoding="utf-8")
    )
    missing_flops_summary["training"]["estimated_flops"] = None
    (candidate_runs[0] / "summary.json").write_text(
        json.dumps(missing_flops_summary), encoding="utf-8"
    )
    overflow_summary = json.loads((candidate_runs[1] / "summary.json").read_text(encoding="utf-8"))
    overflow_summary["evaluation"]["perplexity"]["perplexity"] = None
    (candidate_runs[1] / "summary.json").write_text(json.dumps(overflow_summary), encoding="utf-8")

    report = compare_seeded_runs(baseline_runs, candidate_runs, tmp_path / "seeded-comparison.json")

    assert report["comparable_for_metric_comparison"] is False
    assert report["comparable_for_improvement_claim"] is False
    assert report["metrics"]["estimated_training_flop_relative_delta"]["mean"] is None
    assert report["metrics"]["candidate_perplexity"]["mean"] is None
    assert report["metrics"]["candidate_perplexity"]["valid_n"] == 2
    assert report["metrics"]["candidate_mean_nll"]["mean"] is not None


def test_comparison_keeps_unmeasured_profile_deltas_null(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    create_run(baseline, 1_000)
    create_run(candidate, 1_005)
    for root in (baseline, candidate):
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.pop("profiling")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = compare_runs(baseline, [candidate], tmp_path / "comparison.json")

    assert report["results"][0]["parameter_count_delta"] is None
    assert report["results"][0]["model_tensor_bytes_delta"] is None
