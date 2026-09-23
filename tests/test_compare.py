import json

import pytest

from frontier.experiments.compare import compare_runs, compare_seeded_runs


def create_run(
    root,
    flops,
    synthetic=False,
    seed=17,
    perplexity=10.0,
    width=128,
    revision="test-revision",
    worktree_clean=True,
):
    root.mkdir()
    summary = {
        "schema_version": 1,
        "status": "completed",
        "resolved_config": {
            "seed": seed,
            "model": {"width": width, "layers": 4 if width == 128 else 6},
            "train": {
                "batch_size": 8,
                "context_length": 256,
                "max_steps": 10,
                "max_tokens": None,
                "gradient_accumulation": 1,
                "warmup_steps": 1,
                "learning_rate": 0.0003,
                "min_learning_rate": 0.00003,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "mixed_precision": "fp32",
                "deterministic": True,
            },
        },
        "tokenizer": {"name": "utf8-byte-v1"},
        "data": {
            "source": "test smoke fixture" if synthetic else "test corpus",
            "synthetic_fixture": synthetic,
            "train_sha256": "train",
            "validation_sha256": "valid",
        },
        "evaluation": {
            "protocol": {"dataset_sha256": "valid", "context_length": 256, "stride": 128},
            "perplexity": {"perplexity": perplexity, "mean_nll": perplexity / 4.0},
        },
        "training": {"estimated_flops": flops, "compute_estimator": "decoder-dense-v1"},
        "profiling": {"parameter_count": 100, "model_tensor_bytes": 400},
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "environment.json").write_text(
        json.dumps(
            {
                "platform": "test",
                "torch": "2.14.0",
                "cuda_runtime": None,
                "device": "cpu",
                "device_name": "test cpu",
                "git_revision": revision,
                "git_worktree_clean": worktree_clean,
            }
        ),
        encoding="utf-8",
    )


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
        create_run(candidate, 1_005, seed=seed, perplexity=candidate_ppl, width=256)
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)

    report = compare_seeded_runs(baseline_runs, candidate_runs, tmp_path / "seeded-comparison.json")

    assert report["comparable_for_metric_comparison"] is True
    assert report["comparable_for_improvement_claim"] is True
    assert report["seeds"] == [17, 42, 123]
    assert report["metrics"]["baseline_perplexity"]["mean"] == 10.0
    assert report["metrics"]["paired_perplexity_delta_candidate_minus_baseline"]["mean"] == -1.0


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
        create_run(candidate, 1_005, seed=seed, width=256, worktree_clean=False)
        baseline_runs.append(baseline)
        candidate_runs.append(candidate)

    report = compare_seeded_runs(baseline_runs, candidate_runs, tmp_path / "dirty-comparison.json")

    assert report["comparable_for_metric_comparison"] is False
    assert report["comparable_for_improvement_claim"] is False
    assert report["group_checks"]["all_source_worktrees_clean"] is False
    assert "all runs must use a clean Git working tree" in report["claim_blockers"]


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
