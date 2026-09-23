"""Portable, path-scrubbed evidence bundles for completed or interrupted runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from frontier.config import RunConfig
from frontier.experiments.planning import verify_sweep_plan
from frontier.experiments.results import write_json

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}\Z")
_REVISION = re.compile(r"[0-9a-f]{7,64}\Z")
_SAFE_LABEL = re.compile(r"[A-Za-z0-9 .,:+_;/()-]{0,200}\Z")
_ABSOLUTE_PATH = re.compile(
    r"(?:\b[A-Za-z]:[\\/]|\\\\|(?:^|\s)/(?:Users|home|root|tmp|private|mnt|workspace|var|opt)/)",
    re.IGNORECASE,
)

_TRAINING_FIELDS = {
    "step",
    "optimizer_updates",
    "tokens_seen",
    "tokens_per_step",
    "token_budget_requested",
    "token_budget_planned",
    "token_budget_overshoot",
    "optimizer_active_time_seconds",
    "optimizer_active_time_seconds_this_invocation",
    "wall_time_seconds_this_invocation",
    "tokens_per_second",
    "estimated_flops",
    "estimated_macs",
    "compute_components",
    "compute_estimator",
    "compute_status",
    "precision_requested",
    "autocast_dtype",
    "deterministic_algorithms",
    "memory",
    "process_memory_after_training",
    "accelerator_memory_after_training",
}
_EVALUATION_FIELDS = {
    "schema_version",
    "nll_sum",
    "content_nll_sum",
    "mean_nll",
    "perplexity",
    "perplexity_overflow",
    "scored_tokens",
    "scored_utf8_bytes",
    "bits_per_byte",
    "byte_metric_status",
}
_EVALUATION_PROTOCOL_FIELDS = {
    "metrics_schema_version",
    "dataset_sha256",
    "tokenizer",
    "tokenizer_sha256",
    "context_length",
    "stride",
    "document_boundary",
    "byte_normalization",
}
_PROFILE_FIELDS = {
    "profile_schema_version",
    "parameter_count",
    "trainable_parameter_count",
    "model_tensor_bytes",
    "checkpoint_artifact_bytes",
    "model_weights_artifact_bytes",
    "inference_dtype",
    "device",
    "energy_per_token_joules",
    "energy_measurement_status",
    "process_memory_after_load",
    "accelerator_memory_after_load",
    "cache_parity_max_abs_logit_error",
    "cache_parity_tokens",
    "workloads",
}
_WORKLOAD_FIELDS = {
    "batch_size",
    "requested_prompt_tokens",
    "prompt_tokens",
    "first_token_from_prefill_logits",
    "generated_tokens_per_repeat",
    "timed_decode_tokens",
    "timed_decode_forwards_per_repeat",
    "decode_repeats",
    "state_position_tokens_after_decode",
    "prefill_to_first_token_latency",
    "prefill_latency_seconds_median",
    "prefill_tokens_per_second",
    "decode_latency",
    "decode_latency_seconds_median",
    "decode_tokens_per_second",
    "decode_seconds_per_token",
    "theoretical_kv_cache_tokens_after_decode_loop",
    "theoretical_kv_bytes_after_decode_loop",
    "theoretical_kv_bytes_status",
    "actual_state_storage_bytes_after_decode",
    "allocated_state_bytes_after_decode",
    "actual_kv_bytes_after_decode",
    "persistent_state_bytes_after_decode",
    "active_state_bytes_after_decode",
    "active_state_bytes_status",
    "state_memory_accounting_status",
    "kv_cache_accounting_dtype",
    "process_memory_snapshot_after_workload",
    "prefill_cpu_memory",
    "decode_cpu_memory",
    "prefill_accelerator_memory",
    "decode_accelerator_memory",
}
_NESTED_METRIC_FIELDS = {
    "repetitions_seconds",
    "median_seconds",
    "min_seconds",
    "max_seconds",
    "sample_stddev_seconds",
    "rss_bytes",
    "private_bytes",
    "peak_rss_bytes",
    "peak_rss_increase_bytes",
    "allocated_bytes",
    "reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "status",
    "method",
    "isolation",
}
_SAFE_STRINGS = {
    "compute_estimator",
    "compute_status",
    "precision_requested",
    "autocast_dtype",
    "device",
    "inference_dtype",
    "energy_measurement_status",
    "theoretical_kv_bytes_status",
    "active_state_bytes_status",
    "state_memory_accounting_status",
    "kv_cache_accounting_dtype",
    "status",
    "method",
    "isolation",
}
_COMPARISON_METRICS = {
    "baseline_perplexity",
    "candidate_perplexity",
    "paired_perplexity_delta_candidate_minus_baseline",
    "baseline_mean_nll",
    "candidate_mean_nll",
    "paired_mean_nll_delta_candidate_minus_baseline",
    "estimated_training_flop_relative_delta",
}
_METRIC_SUMMARY_FIELDS = {"mean", "sample_std", "valid_n", "total_n"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_run_id(value: Any) -> str:
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        raise ValueError("run ID must be a short path-safe identifier")
    return value


def _safe_revision(value: Any) -> str | None:
    return value if isinstance(value, str) and _REVISION.fullmatch(value) else None


def _safe_string(value: Any) -> str | None:
    if not isinstance(value, str) or not _SAFE_LABEL.fullmatch(value):
        return None
    return value


def _project(value: Any, allowed: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _project(item, allowed)
            for key, item in value.items()
            if key in allowed and _project(item, allowed) is not _OMIT
        }
    if isinstance(value, list):
        projected = [_project(item, allowed) for item in value]
        return [item for item in projected if item is not _OMIT]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        return _safe_string(value) if value is not None else _OMIT
    return _OMIT


class _Omit:
    pass


_OMIT = _Omit()


def _project_scalars(value: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    projected = _project(value, allowed)
    return projected if isinstance(projected, dict) else {}


def _project_run_config(raw: dict[str, Any], run_id: str) -> dict[str, Any]:
    config = RunConfig.from_dict(raw)
    config.output_dir = f"runs/{run_id}-reproduction"
    config.data_dir = "data/prepared-corpus" if config.data_dir is not None else None
    if config.evaluation.tasks_path is not None:
        config.evaluation.tasks_path = "data/tasks.jsonl"
    config.baseline_run = None
    safe = config.to_dict()
    if _has_absolute_path(safe):
        raise ValueError("sanitized config still contains an absolute path")
    return safe


def _has_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_ABSOLUTE_PATH.search(value))
    if isinstance(value, dict):
        return any(
            _has_absolute_path(key) or _has_absolute_path(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_absolute_path(item) for item in value)
    return False


def _artifact_record(
    path: Path | None,
    *,
    included: bool = False,
    expected_sha256: str | None = None,
    included_file: str | None = None,
) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "included": False,
            "status": "missing",
            "expected_sha256": expected_sha256,
        }
    actual_hash = _sha256_file(path)
    matches = expected_sha256 is None or actual_hash == expected_sha256
    return {
        "available": True,
        "included": included,
        "status": "available" if matches else "hash_mismatch",
        "sha256": actual_hash,
        "bytes": path.stat().st_size,
        "expected_sha256": expected_sha256,
        "bundle_file": included_file if included else None,
    }


def _sanitize_source_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "dataset",
        "dataset_revision",
        "license",
        "content_origin",
        "raw_prefix_sha256",
        "selected_range_bytes",
        "selected_range_start_byte",
        "selected_range_end_byte_inclusive",
        "complete_document_count",
        "discarded_trailing_incomplete_bytes",
        "upstream_file_size_bytes",
        "complete_document_delimiter",
    }
    return _project_scalars(value, allowed)


def _sanitize_data_manifest(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "is_test_fixture",
        "content_origin",
        "source",
        "source_sha256",
        "source_metadata_sha256",
        "preprocessing_sha256",
        "split_seed",
        "validation_fraction_requested",
        "document_count",
        "unique_document_count",
        "train_document_count",
        "validation_document_count",
        "train_sha256",
        "validation_sha256",
        "train_utf8_bytes",
        "validation_utf8_bytes",
        "split_unit",
    }
    result = _project_scalars(value, allowed)
    result["preprocessing"] = _project_scalars(
        value.get("preprocessing", {}),
        {"schema_version", "input_format", "line_handling", "duplicate_identity", "split_unit"},
    )
    result["source_metadata"] = _sanitize_source_metadata(value.get("source_metadata"))
    return result


def _sanitize_environment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in (
        "frontier",
        "platform",
        "python",
        "torch",
        "numpy",
        "cuda_runtime",
        "cuda_available",
        "device",
        "device_name",
        "processor",
        "cpu_count",
        "git_worktree_clean",
    ):
        item = value.get(key)
        if key == "python" and isinstance(item, str):
            item = item.split(maxsplit=1)[0]
        safe = _project(item, {key})
        if safe is not _OMIT:
            result[key] = safe
    result["git_revision"] = _safe_revision(value.get("git_revision"))
    return result


def _recompute_comparison(
    value: dict[str, Any], run_dirs_by_id: dict[str, Path], seeds: list[int]
) -> dict[str, Any]:
    baseline_refs = value.get("baseline_runs")
    candidate_refs = value.get("candidate_runs")
    if not isinstance(baseline_refs, dict) or not isinstance(candidate_refs, dict):
        raise TypeError("comparison lacks source run membership needed for revalidation")

    def resolve_group(references: dict[str, Any]) -> list[Path]:
        by_seed = {}
        for raw_seed, raw_path in references.items():
            if (
                not isinstance(raw_seed, str)
                or not raw_seed.isdigit()
                or not isinstance(raw_path, str)
            ):
                raise ValueError("comparison source run references are malformed")
            seed = int(raw_seed)
            run_id = _safe_run_id(raw_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
            if run_id not in run_dirs_by_id or seed in by_seed:
                raise ValueError("comparison source run is not included exactly once")
            by_seed[seed] = run_dirs_by_id[run_id]
        if set(by_seed) != set(seeds):
            raise ValueError("comparison source run seeds differ from its paired results")
        return [by_seed[seed] for seed in seeds]

    from frontier.experiments.compare import compare_seeded_runs

    with tempfile.TemporaryDirectory(prefix="frontier-evidence-comparison-") as temporary_dir:
        return compare_seeded_runs(
            resolve_group(baseline_refs),
            resolve_group(candidate_refs),
            Path(temporary_dir) / "recomputed-comparison.json",
            minimum_seeds=max(3, len(seeds)),
        )


def _same_json_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_value(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _comparison_recomputation_matches(supplied: dict[str, Any], recomputed: dict[str, Any]) -> bool:
    for key in (
        "seeds",
        "comparable_for_metric_comparison",
        "comparable_for_improvement_claim",
        "comparable_for_human_corpus_claim",
        "minimum_seeds_for_claim",
        "source_revision",
        "source_worktrees_clean",
        "claim_scope",
        "claim_scope_limitation",
    ):
        if supplied.get(key) != recomputed.get(key):
            return False
    for key in ("metrics", "group_checks", "claim_checks"):
        if not _same_json_value(supplied.get(key), recomputed.get(key)):
            return False
    if supplied.get("claim_blocker_count") != len(recomputed.get("claim_blockers", [])):
        return False
    if supplied.get("human_corpus_claim_blocker_count") != len(
        recomputed.get("human_corpus_claim_blockers", [])
    ):
        return False

    supplied_pairs = {pair["seed"]: pair for pair in supplied["pairwise_results"]}
    raw_recomputed_pairs = recomputed["pairwise_results"]
    recomputed_seeds = recomputed["seeds"]
    if len(raw_recomputed_pairs) != len(recomputed_seeds):
        return False
    recomputed_pairs = {}
    for seed, pair in zip(recomputed_seeds, raw_recomputed_pairs):
        normalized = {"seed": seed}
        for key in ("baseline_run", "candidate_run"):
            raw_path = pair.get(key)
            normalized[f"{key}_id"] = (
                _safe_run_id(raw_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
                if isinstance(raw_path, str)
                else None
            )
        for key in (
            "comparable_for_metric_comparison",
            "comparable_for_improvement_claim",
            "evidence_scope",
            "improvement_claim_requires",
            "perplexity_delta_candidate_minus_baseline",
            "estimated_training_flop_relative_delta",
            "training_recipe_warmup_fraction_relative_delta",
            "model_tensor_bytes_delta",
            "parameter_count_delta",
        ):
            normalized[key] = pair.get(key)
        normalized["checks"] = {
            key: item for key, item in pair.get("checks", {}).items() if isinstance(item, bool)
        }
        normalized["eligibility_blocker_count"] = len(pair.get("eligibility_blockers", []))
        recomputed_pairs[seed] = normalized
    if supplied_pairs.keys() != recomputed_pairs.keys():
        return False
    for seed, supplied_pair in supplied_pairs.items():
        recomputed_pair = recomputed_pairs[seed]
        for key in ("baseline_run_id", "candidate_run_id"):
            if supplied_pair.get(key) != recomputed_pair.get(key):
                return False
        for key in (
            "comparable_for_metric_comparison",
            "comparable_for_improvement_claim",
            "evidence_scope",
            "improvement_claim_requires",
            "checks",
            "eligibility_blocker_count",
        ):
            if not _same_json_value(supplied_pair.get(key), recomputed_pair.get(key)):
                return False
        for key in (
            "perplexity_delta_candidate_minus_baseline",
            "estimated_training_flop_relative_delta",
            "training_recipe_warmup_fraction_relative_delta",
            "model_tensor_bytes_delta",
            "parameter_count_delta",
        ):
            if not _same_json_value(supplied_pair.get(key), recomputed_pair.get(key)):
                return False
    return True


def _sanitize_comparison(
    value: Any, run_ids: set[str], run_dirs_by_id: dict[str, Path]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("comparison artifact must contain a JSON object")
    declared_seeds = value.get("seeds", [])
    if not isinstance(declared_seeds, list) or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in declared_seeds
    ):
        raise ValueError("comparison seed list is malformed")
    if len(set(declared_seeds)) != len(declared_seeds):
        raise ValueError("comparison seed list contains duplicates")
    minimum_seeds = value.get("minimum_seeds_for_claim")
    claim_eligible = value.get("comparable_for_improvement_claim") is True
    if claim_eligible and (
        not isinstance(minimum_seeds, int)
        or isinstance(minimum_seeds, bool)
        or minimum_seeds < 3
        or len(declared_seeds) < minimum_seeds
    ):
        raise ValueError("comparison claims eligibility without the required distinct seed floor")
    raw_pairs = value.get("pairwise_results", [])
    if not isinstance(raw_pairs, list):
        raise TypeError("comparison pairwise results must be a list")
    pairs = []
    pair_seeds = set()
    for pair in raw_pairs:
        if not isinstance(pair, dict):
            raise TypeError("comparison pairwise result must be an object")
        baseline = str(pair.get("baseline_run", "")).replace("\\", "/").rstrip("/").split("/")[-1]
        candidate = str(pair.get("candidate_run", "")).replace("\\", "/").rstrip("/").split("/")[-1]
        baseline_id = _safe_run_id(baseline)
        candidate_id = _safe_run_id(candidate)
        if baseline_id not in run_ids or candidate_id not in run_ids:
            raise ValueError("comparison references a run not included in the evidence bundle")
        seeds = {
            run_id.rsplit("-seed-", 1)[-1].split("-", 1)[0]
            for run_id in (baseline_id, candidate_id)
        }
        if len(seeds) != 1 or not next(iter(seeds)).isdigit():
            raise ValueError("comparison pair does not contain the same parseable seed")
        seed = int(next(iter(seeds)))
        if seed not in declared_seeds:
            raise ValueError("comparison pair seed is absent from its declared seed list")
        if seed in pair_seeds:
            raise ValueError("comparison contains duplicate seed pairs")
        pair_seeds.add(seed)
        pair_result = {
            "seed": seed,
            "baseline_run_id": baseline_id,
            "candidate_run_id": candidate_id,
            "evidence_scope": _safe_string(pair.get("evidence_scope")),
            "improvement_claim_requires": _safe_string(pair.get("improvement_claim_requires")),
        }
        for key in (
            "comparable_for_metric_comparison",
            "comparable_for_improvement_claim",
        ):
            item = pair.get(key)
            if isinstance(item, bool):
                pair_result[key] = item
        for key in (
            "perplexity_delta_candidate_minus_baseline",
            "estimated_training_flop_relative_delta",
            "training_recipe_warmup_fraction_relative_delta",
            "model_tensor_bytes_delta",
            "parameter_count_delta",
        ):
            item = pair.get(key)
            if (isinstance(item, (int, float)) and not isinstance(item, bool)) or item is None:
                pair_result[key] = (
                    None if isinstance(item, float) and not math.isfinite(item) else item
                )
        pair_result["checks"] = {
            key: item for key, item in pair.get("checks", {}).items() if isinstance(item, bool)
        }
        pair_result["eligibility_blocker_count"] = len(pair.get("eligibility_blockers", []))
        pairs.append(pair_result)

    result: dict[str, Any] = {}
    for key in (
        "schema_version",
        "comparison_type",
        "evidence_scope",
        "claim_scope",
        "comparable_for_metric_comparison",
        "comparable_for_improvement_claim",
        "comparable_for_human_corpus_claim",
        "minimum_seeds_for_claim",
        "source_worktrees_clean",
    ):
        item = value.get(key)
        if isinstance(item, (bool, int, float, str)):
            result[key] = _safe_string(item) if isinstance(item, str) else item
    result["source_revision"] = _safe_revision(value.get("source_revision"))
    result["claim_scope_limitation"] = _safe_string(value.get("claim_scope_limitation"))
    result["improvement_claim_requires"] = _safe_string(value.get("improvement_claim_requires"))
    result["seeds"] = sorted(declared_seeds)
    if claim_eligible and (len(pair_seeds) < minimum_seeds or pair_seeds != set(declared_seeds)):
        raise ValueError("comparison claim lacks one included pair for every declared seed")
    raw_metrics = value.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        raise TypeError("comparison metrics must be an object")
    result["metrics"] = {
        name: _project_scalars(metric, _METRIC_SUMMARY_FIELDS)
        for name, metric in raw_metrics.items()
        if name in _COMPARISON_METRICS and isinstance(metric, dict)
    }
    for key in ("group_checks", "claim_checks"):
        result[key] = {
            name: check for name, check in value.get(key, {}).items() if isinstance(check, bool)
        }
    result["claim_blocker_count"] = len(value.get("claim_blockers", []))
    result["human_corpus_claim_blocker_count"] = len(value.get("human_corpus_claim_blockers", []))
    result["pairwise_results"] = sorted(pairs, key=lambda pair: pair["seed"])
    if len(pair_seeds) >= 3:
        try:
            recomputed = _recompute_comparison(value, run_dirs_by_id, result["seeds"])
            matches = _comparison_recomputation_matches(result, recomputed)
        except (OSError, TypeError, ValueError, KeyError):
            matches = False
        result["eligibility_reconstruction"] = "verified" if matches else "mismatch_or_unavailable"
        if not matches:
            result["comparable_for_metric_comparison"] = False
            result["comparable_for_improvement_claim"] = False
            result["comparable_for_human_corpus_claim"] = False
            result["claim_blocker_count"] = result.get("claim_blocker_count", 0) + 1
            result["claim_reconstruction_blocked"] = True
    else:
        result["eligibility_reconstruction"] = "not_recomputed_below_three_pairs"
        if claim_eligible:
            raise ValueError("comparison claims eligibility without three included seed pairs")
        result["comparable_for_metric_comparison"] = False
        result["comparable_for_improvement_claim"] = False
        result["comparable_for_human_corpus_claim"] = False
        result["pairwise_results"] = [
            {
                **pair,
                "comparable_for_metric_comparison": False,
                "comparable_for_improvement_claim": False,
                "eligibility_reconstruction": "not_recomputed_below_three_pairs",
                "eligibility_blocker_count": pair["eligibility_blocker_count"] + 1,
            }
            for pair in result["pairwise_results"]
        ]
        result["claim_blocker_count"] = result.get("claim_blocker_count", 0) + 1
        result["claim_reconstruction_blocked"] = True
    return result


def _sanitize_plan(value: Any, context_path: Path | None) -> dict[str, Any]:
    if not isinstance(value, dict) or not verify_sweep_plan(value):
        raise ValueError("sweep plan is malformed or its content hash does not verify")
    result = {
        "plan_type": value.get("plan_type"),
        "schema_version": value.get("schema_version"),
        "name": value.get("name"),
        "plan_sha256": value.get("plan_sha256"),
        "source_spec_sha256": value.get("source_spec_sha256"),
        "seeds": value.get("seeds"),
        "total_compute_cap_flops": value.get("total_compute_cap_flops"),
        "total_estimated_flops": value.get("total_estimated_flops"),
        "total_runtime_cap_seconds": value.get("total_runtime_cap_seconds"),
        "candidate_compute_deltas_vs_baseline": value.get("candidate_compute_deltas_vs_baseline"),
        "hardware": _sanitize_environment(value.get("hardware", {})),
        "arms": [
            {
                key: arm.get(key)
                for key in (
                    "arm_id",
                    "role",
                    "resolved_config_sha256",
                    "planned_optimizer_steps_per_seed",
                    "planned_tokens_per_seed",
                    "compute_estimator",
                    "estimated_flops_per_seed",
                    "estimated_macs_per_seed",
                    "compute_assumptions",
                )
            }
            for arm in value.get("arms", [])
        ],
        "jobs": [
            {
                key: job.get(key)
                for key in (
                    "job_id",
                    "arm_id",
                    "role",
                    "seed",
                    "run_id",
                    "config_sha256",
                    "planned_optimizer_steps",
                    "planned_tokens",
                    "estimated_flops",
                )
            }
            for job in value.get("jobs", [])
        ],
    }
    if context_path is not None and context_path.is_file():
        context = json.loads(context_path.read_text(encoding="utf-8"))
        context_fields = {
            "record_type",
            "recorded_date",
            "source_revision",
            "git_worktree_clean",
            "mha_config_sha256",
            "gqa_config_sha256",
            "corpus_manifest_sha256",
            "tokenizer_artifact_sha256",
            "plan_sha256",
            "planned_jobs",
            "warmup_fraction_gate",
            "warmup_fraction_relative_gap",
            "estimated_flop_delta_candidate_percent",
            "total_compute_cap_flops",
            "total_estimated_flops",
        }
        result["context"] = _project_scalars(context, context_fields)
    return result


def _run_evidence(
    run_dir: Path,
    config_file: str,
    data_manifest_file: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "resolved_config.json"
    if not summary_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"run requires summary.json and resolved_config.json: {run_dir.name}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(raw_config, dict):
        raise TypeError("run summary and resolved config must be JSON objects")
    run_id = _safe_run_id(summary.get("run_id", run_dir.name))
    config_record = _project_run_config(raw_config, run_id)
    if summary.get("resolved_config") != raw_config:
        raise ValueError(f"saved config and run summary differ for {run_id}")

    data_record = summary.get("data", {})
    raw_manifest_path = run_dir / "data_manifest.json"
    raw_manifest = (
        json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        if raw_manifest_path.is_file()
        else None
    )
    manifest_artifact = _artifact_record(
        raw_manifest_path,
        expected_sha256=data_record.get("manifest_sha256"),
    )
    prepared_manifest_path = None
    if raw_config.get("data_dir"):
        configured_data = Path(raw_config["data_dir"])
        if not configured_data.is_absolute():
            configured_data = Path.cwd() / configured_data
        prepared_manifest_path = configured_data / "data_manifest.json"
    prepared_manifest = None
    if prepared_manifest_path is not None and prepared_manifest_path.is_file():
        prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    manifest_semantic_match = (
        prepared_manifest == raw_manifest
        if prepared_manifest is not None and raw_manifest
        else None
    )
    manifest_artifact["semantic_match_with_prepared_manifest"] = manifest_semantic_match
    if manifest_semantic_match is False:
        manifest_artifact["status"] = "semantic_mismatch"
    prepared_manifest_artifact = _artifact_record(prepared_manifest_path)

    manifest_key = _sha256_bytes(
        json.dumps(raw_manifest or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )[:16]
    sanitized_manifest = _sanitize_data_manifest(raw_manifest) if raw_manifest else None
    if sanitized_manifest is not None:
        sanitized_manifest["run_manifest_file_sha256"] = manifest_artifact.get("sha256")
        sanitized_manifest["prepared_manifest_file_sha256"] = prepared_manifest_artifact.get(
            "sha256"
        )
        sanitized_manifest["prepared_manifest_semantic_match"] = manifest_semantic_match

    data_root_value = data_record.get("root")
    data_root = Path(data_root_value) if isinstance(data_root_value, str) else None
    corpus_records = {}
    if data_root is not None:
        for label, filename, digest_key in (
            ("training_corpus", "train.txt", "train_sha256"),
            ("validation_corpus", "validation.txt", "validation_sha256"),
        ):
            corpus_records[label] = _artifact_record(
                data_root / filename,
                expected_sha256=(raw_manifest or {}).get(digest_key),
            )

    tokenizer_path = run_dir / "tokenizer.json"
    tokenizer_json = (
        json.loads(tokenizer_path.read_text(encoding="utf-8")) if tokenizer_path.is_file() else {}
    )
    tokenizer_artifact_hash = summary.get("tokenizer", {}).get("artifact_sha256")
    tokenizer_record = _artifact_record(tokenizer_path, expected_sha256=None)
    tokenizer_record["artifact_sha256"] = tokenizer_artifact_hash
    tokenizer_record["artifact_hash_matches_summary"] = (
        tokenizer_json.get("artifact_sha256") == tokenizer_artifact_hash
        if tokenizer_path.is_file() and tokenizer_artifact_hash
        else None
    )

    task_path = raw_config.get("evaluation", {}).get("tasks_path")
    task_artifact = None
    if isinstance(task_path, str) and task_path:
        task_file = Path(task_path)
        if not task_file.is_absolute():
            task_file = Path.cwd() / task_file
        task_artifact = _artifact_record(task_file)

    artifact_availability = {
        "summary": _artifact_record(summary_path),
        "original_resolved_config": _artifact_record(config_path),
        "exported_resolved_config": {
            "available": True,
            "included": True,
            "status": "available",
            "bundle_file": config_file,
        },
        "run_data_manifest": manifest_artifact,
        "prepared_data_manifest": prepared_manifest_artifact,
        "training_corpus": corpus_records.get(
            "training_corpus", {"available": False, "included": False, "status": "unknown"}
        ),
        "validation_corpus": corpus_records.get(
            "validation_corpus", {"available": False, "included": False, "status": "unknown"}
        ),
        "tokenizer_artifact": tokenizer_record,
        "checkpoint": _artifact_record(run_dir / "checkpoints" / "last.pt"),
        "inference_weights": _artifact_record(run_dir / "weights.pt"),
        "evaluation_sidecar": _artifact_record(run_dir / "evaluation.json"),
        "profile_sidecar": _artifact_record(run_dir / "profile.json"),
    }
    if task_artifact is not None:
        artifact_availability["task_examples"] = task_artifact

    evaluation = summary.get("evaluation", {})
    profiling = summary.get("profiling", {})
    evidence = {
        "run_id": run_id,
        "status": _safe_string(summary.get("status")),
        "seed": raw_config.get("seed"),
        "source_revision": _safe_revision(summary.get("environment", {}).get("git_revision")),
        "source_worktree_clean": summary.get("environment", {}).get("git_worktree_clean"),
        "configuration_file": config_file,
        "original_config_sha256": _sha256_file(config_path),
        "sanitized_config_sha256": _sha256_bytes(
            json.dumps(config_record, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
            + b"\n"
        ),
        "data": {
            "content_origin": _safe_string(data_record.get("content_origin")),
            "is_test_fixture": data_record.get("is_test_fixture"),
            "source": _safe_string(data_record.get("source")),
            "source_metadata": _sanitize_source_metadata(data_record.get("source_metadata")),
            "manifest_file": f"data-manifests/{manifest_key}.json" if sanitized_manifest else None,
            "run_manifest_sha256": manifest_artifact.get("sha256"),
            "prepared_manifest_sha256": prepared_manifest_artifact.get("sha256"),
            "train_sha256": (raw_manifest or {}).get("train_sha256"),
            "validation_sha256": (raw_manifest or {}).get("validation_sha256"),
            "prepared_manifest_semantic_match": manifest_semantic_match,
        },
        "tokenizer": {
            "name": _safe_string(summary.get("tokenizer", {}).get("name")),
            "vocab_size": summary.get("tokenizer", {}).get("vocab_size"),
            "artifact_sha256": tokenizer_artifact_hash,
            "artifact_file_sha256": tokenizer_record.get("sha256"),
        },
        "environment": _sanitize_environment(summary.get("environment", {})),
        "training": _project_scalars(summary.get("training", {}), _TRAINING_FIELDS),
        "evaluation": {
            "perplexity": _project_scalars(evaluation.get("perplexity", {}), _EVALUATION_FIELDS),
            "protocol": _project_scalars(
                evaluation.get("protocol", {}), _EVALUATION_PROTOCOL_FIELDS
            ),
        },
        "profiling": _project_scalars(
            profiling, _PROFILE_FIELDS | _WORKLOAD_FIELDS | _NESTED_METRIC_FIELDS | _SAFE_STRINGS
        ),
        "artifact_availability": artifact_availability,
    }
    return evidence, config_record, sanitized_manifest


def _sanitize_context_path(plan_path: Path | None) -> Path | None:
    if plan_path is None:
        return None
    return plan_path.with_name(f"{plan_path.stem}.context.json")


def _file_index(root: Path) -> dict[str, dict[str, Any]]:
    index = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.relative_to(root).as_posix() == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        index[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return index


def _write_evidence_bundle(
    runs: list[Path],
    destination: Path,
    comparison_path: str | Path | None,
    sweep_plan_path: str | Path | None,
) -> None:
    destination.mkdir(parents=True)
    (destination / "configs").mkdir()
    (destination / "data-manifests").mkdir()
    run_evidence = []
    run_dirs_by_id: dict[str, Path] = {}
    data_manifest_files: dict[str, dict[str, Any]] = {}
    for run_dir in runs:
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"run lacks summary.json: {run_dir.name}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = _safe_run_id(summary.get("run_id", run_dir.name))
        config_file = f"configs/{run_id}.json"
        evidence, config_record, sanitized_manifest = _run_evidence(run_dir, config_file, None)
        config_path = destination / config_file
        write_json(config_path, config_record)
        evidence["sanitized_config_sha256"] = _sha256_file(config_path)
        if sanitized_manifest is not None:
            manifest_path = evidence["data"]["manifest_file"]
            data_path = destination / manifest_path
            if manifest_path not in data_manifest_files:
                data_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(data_path, sanitized_manifest)
                data_manifest_files[manifest_path] = sanitized_manifest
            else:
                evidence["data"]["manifest_file"] = manifest_path
        if _has_absolute_path(evidence) or _has_absolute_path(config_record):
            raise ValueError(f"sanitizer left an absolute path in run {run_id}")
        run_evidence.append(evidence)

    ids = {run["run_id"] for run in run_evidence}
    if len(ids) != len(run_evidence):
        raise ValueError("run IDs must be unique in an evidence bundle")
    run_dirs_by_id = {run["run_id"]: run_dir for run, run_dir in zip(run_evidence, runs)}
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "bundle_type": "frontier-portable-evidence-v1",
        "runs": sorted(run_evidence, key=lambda run: (run["seed"], run["run_id"])),
        "reproduction": {
            "prerequisites": [
                "Install this repository and its declared dependencies.",
                "Place the matching prepared corpus at data/prepared-corpus and verify its manifest hash before training.",
            ],
            "commands": [
                {
                    "run_id": run["run_id"],
                    "train": f"frontier train --config {run['configuration_file']}",
                    "evaluate": f"frontier evaluate --run-dir runs/{run['run_id']}-reproduction",
                    "profile": f"frontier profile --run-dir runs/{run['run_id']}-reproduction",
                }
                for run in sorted(run_evidence, key=lambda run: (run["seed"], run["run_id"]))
            ],
            "excluded_inputs": [
                "corpus text",
                "tokenizer files",
                "checkpoints",
                "model weights",
            ],
        },
        "source_artifacts": {},
    }

    if comparison_path is not None:
        comparison_file = Path(comparison_path).resolve()
        if not comparison_file.is_file():
            raise FileNotFoundError(f"comparison artifact does not exist: {comparison_file}")
        comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
        bundle["comparison"] = _sanitize_comparison(comparison, ids, run_dirs_by_id)
        bundle["source_artifacts"]["comparison"] = _artifact_record(comparison_file)
    if sweep_plan_path is not None:
        plan_file = Path(sweep_plan_path).resolve()
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        bundle["sweep_plan"] = _sanitize_plan(plan, _sanitize_context_path(plan_file))
        bundle["source_artifacts"]["sweep_plan"] = _artifact_record(plan_file)
        context_path = _sanitize_context_path(plan_file)
        bundle["source_artifacts"]["sweep_plan_context"] = _artifact_record(context_path)

    if _has_absolute_path(bundle):
        raise ValueError("sanitized evidence still contains a machine-specific absolute path")
    bundle_path = destination / "bundle.json"
    write_json(bundle_path, bundle)
    file_index = _file_index(destination)
    manifest = {
        "schema_version": 1,
        "manifest_type": "frontier-evidence-manifest-v1",
        "files": file_index,
        "file_count": len(file_index),
    }
    write_json(destination / "manifest.json", manifest)
    return verify_evidence_bundle(destination)


def export_evidence_bundle(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    comparison_path: str | Path | None = None,
    sweep_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write sanitized per-run evidence and optional paired-comparison metadata."""
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    runs = [Path(run_dir).resolve() for run_dir in run_dirs]
    if len({str(path) for path in runs}) != len(runs):
        raise ValueError("run directories must be distinct")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evidence bundle: {destination}")
    if any(destination == run or run in destination.parents for run in runs):
        raise ValueError("evidence output must not be inside an input run directory")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary_dir:
        staged_bundle = Path(temporary_dir) / destination.name
        _write_evidence_bundle(runs, staged_bundle, comparison_path, sweep_plan_path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite evidence bundle: {destination}")
        staged_bundle.rename(destination)
    return verify_evidence_bundle(destination)


def verify_evidence_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify included file hashes, inventory and path sanitization in a bundle."""
    root = Path(bundle_dir).resolve()
    manifest_path = root / "manifest.json"
    bundle_path = root / "bundle.json"
    if not manifest_path.is_file() or not bundle_path.is_file():
        raise FileNotFoundError("evidence bundle requires bundle.json and manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_type") != "frontier-evidence-manifest-v1":
        raise ValueError("unsupported evidence manifest")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or manifest.get("file_count") != len(expected):
        raise ValueError("evidence manifest file inventory is malformed")
    actual_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "manifest.json"
    }
    if actual_names != set(expected):
        raise ValueError("evidence bundle contains missing or unlisted files")
    for name, record in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("evidence manifest contains an unsafe file path")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("evidence file escapes the bundle directory")
        if not path.is_file() or _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"evidence file hash mismatch: {name}")
        if path.stat().st_size != record.get("bytes"):
            raise ValueError(f"evidence file size mismatch: {name}")
        if path.suffix.lower() == ".json":
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if _has_absolute_path(parsed):
                raise ValueError(f"evidence file contains a machine-specific path: {name}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("bundle_type") != "frontier-portable-evidence-v1":
        raise ValueError("unsupported evidence bundle")
    return {
        "status": "verified",
        "bundle_sha256": _sha256_file(bundle_path),
        "file_count": len(expected),
    }
