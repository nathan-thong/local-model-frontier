"""Automated run comparison with explicit comparability checks."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from frontier.experiments.results import write_json


def _load(path: str | Path) -> tuple[Path, dict, dict]:
    root = Path(path)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    environment_path = root / "environment.json"
    summary["environment"] = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.exists()
        else {}
    )
    config = summary.get("resolved_config", {})
    return root, summary, config


def compare_runs(
    baseline_path: str | Path, candidate_paths: list[str | Path], output_path: str | Path
) -> dict:
    base_root, baseline, base_config = _load(baseline_path)
    rows = []
    baseline_data = baseline.get("data", {})
    baseline_eval = baseline.get("evaluation", {})
    base_flops = baseline.get("training", {}).get("estimated_flops")
    tolerance = 0.01
    for candidate_path in candidate_paths:
        cand_root, candidate, cand_config = _load(candidate_path)
        base_train = base_config.get("train", {})
        candidate_train = cand_config.get("train", {})

        def warmup_fraction(train_config: dict) -> float | None:
            batch_tokens = (
                train_config.get("batch_size", 0)
                * train_config.get("context_length", 0)
                * train_config.get("gradient_accumulation", 0)
            )
            if train_config.get("max_steps") is not None:
                steps = train_config["max_steps"]
            elif batch_tokens and train_config.get("max_tokens"):
                steps = math.ceil(train_config["max_tokens"] / batch_tokens)
            else:
                return None
            return train_config.get("warmup_steps", 0) / max(1, steps)

        base_warmup = warmup_fraction(base_train)
        candidate_warmup = warmup_fraction(candidate_train)
        warmup_relative_delta = (
            abs(base_warmup - candidate_warmup) / max(base_warmup, candidate_warmup, 1e-12)
            if base_warmup is not None and candidate_warmup is not None
            else None
        )
        environment_keys = (
            "platform",
            "torch",
            "numpy",
            "frontier",
            "cuda_runtime",
            "device",
            "device_name",
            "git_revision",
        )
        same_environment = all(
            baseline.get("environment", {}).get(key) == candidate.get("environment", {}).get(key)
            for key in environment_keys
        )
        recipe_keys = (
            "batch_size",
            "context_length",
            "learning_rate",
            "min_learning_rate",
            "weight_decay",
            "grad_clip",
            "gradient_accumulation",
            "mixed_precision",
            "deterministic",
        )
        checks = {
            "same_training_data": baseline_data.get("train_sha256")
            == candidate.get("data", {}).get("train_sha256"),
            "same_validation_data": baseline_data.get("validation_sha256")
            == candidate.get("data", {}).get("validation_sha256"),
            "same_tokenizer": baseline.get("tokenizer") == candidate.get("tokenizer"),
            "same_data_kind": baseline_data.get("synthetic_fixture")
            == candidate.get("data", {}).get("synthetic_fixture"),
            "same_evaluation_protocol": baseline_eval.get("protocol")
            == candidate.get("evaluation", {}).get("protocol"),
            "same_training_recipe": all(
                base_train.get(key) == candidate_train.get(key) for key in recipe_keys
            )
            and warmup_relative_delta is not None
            and warmup_relative_delta <= 0.05,
            "same_seed": base_config.get("seed") == cand_config.get("seed"),
            "same_environment": same_environment,
            "both_runs_completed": baseline.get("status") == "completed"
            and candidate.get("status") == "completed",
            "same_compute_estimator": baseline.get("training", {}).get("compute_estimator")
            == candidate.get("training", {}).get("compute_estimator"),
        }
        cand_flops = candidate.get("training", {}).get("estimated_flops")
        if base_flops and cand_flops:
            relative_delta = abs(cand_flops - base_flops) / base_flops
            checks["training_compute_within_1_percent"] = relative_delta <= tolerance
        else:
            relative_delta = None
            checks["training_compute_within_1_percent"] = False
        base_ppl = baseline_eval.get("perplexity", {}).get("perplexity")
        cand_ppl = candidate.get("evaluation", {}).get("perplexity", {}).get("perplexity")
        rows.append(
            {
                "baseline_run": str(base_root),
                "candidate_run": str(cand_root),
                "comparable_for_metric_comparison": all(checks.values()),
                "comparable_for_improvement_claim": False,
                "improvement_claim_requires": (
                    "Use compare-seeds with at least 3 matched seeds on non-synthetic corpus data."
                ),
                "evidence_scope": "synthetic infrastructure fixture"
                if baseline_data.get("synthetic_fixture")
                or candidate.get("data", {}).get("synthetic_fixture")
                else "corpus metric",
                "checks": checks,
                "training_recipe_warmup_fraction_relative_delta": warmup_relative_delta,
                "estimated_training_flop_relative_delta": relative_delta,
                "perplexity_delta_candidate_minus_baseline": cand_ppl - base_ppl
                if base_ppl is not None and cand_ppl is not None
                else None,
                "parameter_count_delta": (
                    candidate.get("profiling", {}).get("parameter_count")
                    - baseline.get("profiling", {}).get("parameter_count")
                    if candidate.get("profiling", {}).get("parameter_count") is not None
                    and baseline.get("profiling", {}).get("parameter_count") is not None
                    else None
                ),
                "model_tensor_bytes_delta": (
                    candidate.get("profiling", {}).get("model_tensor_bytes")
                    - baseline.get("profiling", {}).get("model_tensor_bytes")
                    if candidate.get("profiling", {}).get("model_tensor_bytes") is not None
                    and baseline.get("profiling", {}).get("model_tensor_bytes") is not None
                    else None
                ),
            }
        )
    report = {
        "schema_version": 1,
        "baseline_run": str(base_root),
        "compute_estimator": baseline.get("training", {}).get("compute_estimator"),
        "required_compute_relative_tolerance": tolerance,
        "results": rows,
    }
    write_json(output_path, report)
    return report


def _mean_and_sample_std(values: list[float]) -> dict:
    mean = sum(values) / len(values)
    sample_std = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1
        else 0.0
    )
    return {"mean": mean, "sample_std": sample_std}


def compare_seeded_runs(
    baseline_paths: list[str | Path],
    candidate_paths: list[str | Path],
    output_path: str | Path,
    minimum_seeds: int = 3,
) -> dict:
    if minimum_seeds < 2:
        raise ValueError("minimum_seeds must be at least 2")
    if len(baseline_paths) < minimum_seeds or len(candidate_paths) < minimum_seeds:
        raise ValueError(f"at least {minimum_seeds} runs per group are required")
    if len(baseline_paths) != len(candidate_paths):
        raise ValueError("baseline and candidate groups must contain the same number of runs")

    def index_by_seed(paths: list[str | Path]) -> dict[int, tuple[Path, dict, dict]]:
        indexed = {}
        for path in paths:
            loaded = _load(path)
            seed = loaded[2].get("seed")
            if not isinstance(seed, int):
                raise TypeError(f"run {path} has no integer seed in its resolved config")
            if seed in indexed:
                raise ValueError(f"duplicate seed {seed} in run group")
            indexed[seed] = loaded
        return indexed

    baseline_by_seed = index_by_seed(baseline_paths)
    candidate_by_seed = index_by_seed(candidate_paths)
    seeds = sorted(baseline_by_seed)
    if set(seeds) != set(candidate_by_seed):
        raise ValueError("baseline and candidate groups must have identical seed sets")

    baseline_runs = [baseline_by_seed[seed] for seed in seeds]
    candidate_runs = [candidate_by_seed[seed] for seed in seeds]

    def all_same(values: list[object]) -> bool:
        return bool(values) and all(value == values[0] for value in values[1:])

    def model_signature(run: tuple[Path, dict, dict]) -> str:
        return json.dumps(run[2].get("model", {}), sort_keys=True)

    def training_signature(run: tuple[Path, dict, dict]) -> str:
        train = dict(run[2].get("train", {}))
        train.pop("resume", None)
        return json.dumps(train, sort_keys=True)

    all_runs = baseline_runs + candidate_runs
    train_hashes = [run[1].get("data", {}).get("train_sha256") for run in all_runs]
    validation_hashes = [run[1].get("data", {}).get("validation_sha256") for run in all_runs]
    tokenizers = [run[1].get("tokenizer") for run in all_runs]
    protocols = [run[1].get("evaluation", {}).get("protocol") for run in all_runs]
    environment_keys = (
        "platform",
        "torch",
        "numpy",
        "frontier",
        "cuda_runtime",
        "device",
        "device_name",
        "git_revision",
    )
    environments = [
        {key: run[1].get("environment", {}).get(key) for key in environment_keys}
        for run in all_runs
    ]
    group_checks = {
        "same_training_data_across_all_runs": all_same(train_hashes),
        "same_validation_data_across_all_runs": all_same(validation_hashes),
        "same_tokenizer_across_all_runs": all_same(tokenizers),
        "same_evaluation_protocol_across_all_runs": all_same(protocols),
        "same_environment_across_all_runs": all_same(environments),
        "baseline_architecture_fixed_across_seeds": all_same(
            [model_signature(run) for run in baseline_runs]
        ),
        "candidate_architecture_fixed_across_seeds": all_same(
            [model_signature(run) for run in candidate_runs]
        ),
        "baseline_recipe_fixed_across_seeds": all_same(
            [training_signature(run) for run in baseline_runs]
        ),
        "candidate_recipe_fixed_across_seeds": all_same(
            [training_signature(run) for run in candidate_runs]
        ),
    }

    pairwise_results = []
    with tempfile.TemporaryDirectory(prefix="frontier-seeded-comparison-") as temp_dir:
        for seed in seeds:
            pair_path = Path(temp_dir) / f"seed-{seed}.json"
            pair_report = compare_runs(
                baseline_by_seed[seed][0],
                [candidate_by_seed[seed][0]],
                pair_path,
            )
            pairwise_results.append(pair_report["results"][0])

    pair_checks_pass = all(row["comparable_for_metric_comparison"] for row in pairwise_results)
    same_real_corpus = all(
        run[1].get("data", {}).get("synthetic_fixture") is False for run in all_runs
    )
    git_revisions = [run[1].get("environment", {}).get("git_revision") for run in all_runs]
    source_revision_pinned = all_same(git_revisions) and all(
        isinstance(revision, str) and revision for revision in git_revisions
    )
    metric_comparable = all(group_checks.values()) and pair_checks_pass
    claim_eligible = (
        metric_comparable
        and len(seeds) >= minimum_seeds
        and same_real_corpus
        and source_revision_pinned
    )

    baseline_ppl = [
        float(run[1]["evaluation"]["perplexity"]["perplexity"]) for run in baseline_runs
    ]
    candidate_ppl = [
        float(run[1]["evaluation"]["perplexity"]["perplexity"]) for run in candidate_runs
    ]
    baseline_nll = [float(run[1]["evaluation"]["perplexity"]["mean_nll"]) for run in baseline_runs]
    candidate_nll = [
        float(run[1]["evaluation"]["perplexity"]["mean_nll"]) for run in candidate_runs
    ]
    paired_ppl_delta = [
        candidate - baseline for baseline, candidate in zip(baseline_ppl, candidate_ppl)
    ]
    paired_nll_delta = [
        candidate - baseline for baseline, candidate in zip(baseline_nll, candidate_nll)
    ]
    flops_deltas = [
        float(row["estimated_training_flop_relative_delta"]) for row in pairwise_results
    ]
    synthetic_evidence = any(
        run[1].get("data", {}).get("synthetic_fixture") is True for run in all_runs
    )

    report = {
        "schema_version": 1,
        "comparison_type": "paired multi-seed",
        "minimum_seeds_for_claim": minimum_seeds,
        "source_revision": git_revisions[0] if source_revision_pinned else None,
        "seeds": seeds,
        "baseline_runs": {str(seed): str(baseline_by_seed[seed][0]) for seed in seeds},
        "candidate_runs": {str(seed): str(candidate_by_seed[seed][0]) for seed in seeds},
        "comparable_for_metric_comparison": metric_comparable,
        "comparable_for_improvement_claim": claim_eligible,
        "evidence_scope": (
            "synthetic infrastructure fixture" if synthetic_evidence else "multi-seed corpus metric"
        ),
        "claim_blockers": (
            []
            if claim_eligible
            else [name for name, passed in group_checks.items() if not passed]
            + (["pairwise comparability checks failed"] if not pair_checks_pass else [])
            + (["non-synthetic corpus data required"] if not same_real_corpus else [])
            + (
                [f"at least {minimum_seeds} matched seeds required"]
                if len(seeds) < minimum_seeds
                else []
            )
            + (
                ["all runs must record one pinned git revision"]
                if not source_revision_pinned
                else []
            )
        ),
        "group_checks": group_checks,
        "pairwise_results": pairwise_results,
        "metrics": {
            "baseline_perplexity": _mean_and_sample_std(baseline_ppl),
            "candidate_perplexity": _mean_and_sample_std(candidate_ppl),
            "paired_perplexity_delta_candidate_minus_baseline": _mean_and_sample_std(
                paired_ppl_delta
            ),
            "baseline_mean_nll": _mean_and_sample_std(baseline_nll),
            "candidate_mean_nll": _mean_and_sample_std(candidate_nll),
            "paired_mean_nll_delta_candidate_minus_baseline": _mean_and_sample_std(
                paired_nll_delta
            ),
            "estimated_training_flop_relative_delta": _mean_and_sample_std(flops_deltas),
        },
    }
    write_json(output_path, report)
    return report
