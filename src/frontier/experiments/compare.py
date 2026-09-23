"""Automated run comparison with fail-closed eligibility checks."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from frontier.config import RunConfig
from frontier.data.corpus import sha256_file
from frontier.experiments.results import write_json
from frontier.profiling.compute import estimate_training_compute

HASH_LENGTH = 64
RECIPE_KEYS = (
    "batch_size",
    "context_length",
    "learning_rate",
    "min_learning_rate",
    "weight_decay",
    "grad_clip",
    "gradient_accumulation",
    "mixed_precision",
    "deterministic",
    "warmup_steps",
)
ENVIRONMENT_KEYS = (
    "platform",
    "torch",
    "numpy",
    "frontier",
    "cuda_runtime",
    "device",
    "device_name",
    "git_revision",
    "git_worktree_clean",
)
CONTENT_ORIGINS = {"human", "synthetic", "mixed", "unknown"}


def _load(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(path)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TypeError(f"run summary must contain a JSON object: {root}")
    environment_path = root / "environment.json"
    saved_environment = summary.get("environment")
    environment_file_matches_summary = False
    try:
        disk_environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment_file_matches_summary = (
            isinstance(disk_environment, dict)
            and isinstance(saved_environment, dict)
            and disk_environment == saved_environment
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        disk_environment = saved_environment
    summary["environment"] = disk_environment
    if not isinstance(summary["environment"], dict):
        summary["environment"] = {}
    config = summary.get("resolved_config", {})
    config = config if isinstance(config, dict) else {}
    summary["_artifact_checks"] = _run_file_checks(
        root, summary, config, environment_file_matches_summary
    )
    return root, summary, config


def _finite(value: Any, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        is_finite = math.isfinite(value)
    except OverflowError:
        return False
    if not is_finite:
        return False
    return (not positive or value > 0) and (not nonnegative or value >= 0)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_seed(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == HASH_LENGTH
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _section(run: dict[str, Any], name: str) -> dict[str, Any]:
    value = run.get(name)
    return value if isinstance(value, dict) else {}


def _run_file_checks(
    root: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
    environment_file_matches_summary: bool,
) -> dict[str, bool]:
    data = _section(summary, "data")
    tokenizer = _section(summary, "tokenizer")
    evaluation = _section(summary, "evaluation")
    protocol = _section(evaluation, "protocol")
    checks = {
        "resolved_config_file_valid": False,
        "environment_file_matches_summary": environment_file_matches_summary,
        "tokenizer_file_valid": False,
        "data_manifest_file_matches_summary": False,
        "data_files_match_summary": False,
    }
    try:
        saved_config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
        checks["resolved_config_file_valid"] = (
            isinstance(saved_config, dict) and saved_config == config
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        artifact = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
        if isinstance(artifact, dict):
            payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
            digest = artifact.get("artifact_sha256")
            checks["tokenizer_file_valid"] = (
                _valid_hash(digest)
                and digest == _canonical_hash(payload)
                and digest == tokenizer.get("artifact_sha256")
                and payload.get("name") == tokenizer.get("name")
                and payload.get("vocab_size") == tokenizer.get("vocab_size")
                and digest == protocol.get("tokenizer_sha256")
            )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        manifest_path = root / "data_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_fields_match = isinstance(manifest, dict) and all(
            manifest.get(manifest_key) == data.get(summary_key)
            for manifest_key, summary_key in (
                ("content_origin", "content_origin"),
                ("is_test_fixture", "is_test_fixture"),
                ("source_sha256", "source_sha256"),
                ("train_sha256", "train_sha256"),
                ("validation_sha256", "validation_sha256"),
                ("source_metadata", "source_metadata"),
                ("source_metadata_sha256", "source_metadata_sha256"),
                ("preprocessing", "preprocessing"),
                ("preprocessing_sha256", "preprocessing_sha256"),
            )
        )
        checks["data_manifest_file_matches_summary"] = manifest_fields_match and sha256_file(
            manifest_path
        ) == data.get("manifest_sha256")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    data_root = data.get("root")
    if isinstance(data_root, str) and data_root:
        try:
            root_path = Path(data_root)
            checks["data_files_match_summary"] = sha256_file(root_path / "train.txt") == data.get(
                "train_sha256"
            ) and sha256_file(root_path / "validation.txt") == data.get("validation_sha256")
        except (OSError, UnicodeError, TypeError, ValueError):
            pass
    return checks


def _origin(run: dict[str, Any]) -> str:
    value = _section(run, "data").get("content_origin")
    return value if isinstance(value, str) and value in CONTENT_ORIGINS else "unknown"


def _training_budget_valid(run: dict[str, Any], config: dict[str, Any]) -> bool:
    """Verify the completed run's consumed tokens and compute against its config."""
    training = _section(run, "training")
    try:
        resolved = RunConfig.from_dict(config)
        model = resolved.model
        train = resolved.train
        step = training.get("step")
        tokens_seen = training.get("tokens_seen")
        optimizer_updates = training.get("optimizer_updates")
        tokens_per_step = train.batch_size * train.context_length * train.gradient_accumulation
        if not all(_positive_int(value) for value in (step, tokens_seen, optimizer_updates)):
            return False
        planned_steps = train.max_steps or math.ceil(train.max_tokens / tokens_per_step)
        planned_tokens = planned_steps * tokens_per_step
        if (
            step != planned_steps
            or tokens_seen != planned_tokens
            or tokens_seen != step * tokens_per_step
            or optimizer_updates != step
            or training.get("tokens_per_step") != tokens_per_step
            or training.get("token_budget_planned") != planned_tokens
            or training.get("token_budget_requested") != train.max_tokens
            or training.get("token_budget_overshoot")
            != (max(0, tokens_seen - train.max_tokens) if train.max_tokens is not None else 0)
        ):
            return False
        estimate = estimate_training_compute(model, tokens_seen, train.context_length)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False
    return (
        estimate["status"] == "estimated"
        and training.get("compute_status") == "estimated"
        and training.get("compute_estimator") == estimate["estimator"]
        and training.get("estimated_macs") == estimate["estimated_macs"]
        and training.get("estimated_flops") == estimate["estimated_flops"]
        and training.get("compute_components") == estimate["components"]
    )


def _run_validation(run: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    data = _section(run, "data")
    model = _section(config, "model")
    train = _section(config, "train")
    tokenizer = _section(run, "tokenizer")
    evaluation = _section(run, "evaluation")
    protocol = _section(evaluation, "protocol")
    metrics = _section(evaluation, "perplexity")
    training = _section(run, "training")
    environment = _section(run, "environment")
    step_budget = _positive_int(train.get("max_steps"))
    token_budget = _positive_int(train.get("max_tokens"))
    tokenizer_hash = tokenizer.get("artifact_sha256")
    protocol_tokenizer_hash = protocol.get("tokenizer_sha256")
    source_metadata = data.get("source_metadata")
    preprocessing = data.get("preprocessing")
    artifact_checks = _section(run, "_artifact_checks")
    source_revision = (
        source_metadata.get("dataset_revision", source_metadata.get("revision"))
        if isinstance(source_metadata, dict)
        else None
    )
    source_metadata_valid = (
        isinstance(source_metadata, dict)
        and isinstance(source_metadata.get("dataset"), str)
        and bool(source_metadata["dataset"])
        and isinstance(source_metadata.get("license"), str)
        and bool(source_metadata["license"])
        and isinstance(source_revision, str)
        and bool(source_revision)
        and source_metadata.get("content_origin") == _origin(run)
        and _valid_hash(data.get("source_metadata_sha256"))
        and data["source_metadata_sha256"] == _canonical_hash(source_metadata)
    )
    env_strings_valid = all(
        isinstance(environment.get(key), str) and bool(environment[key])
        for key in ("platform", "torch", "numpy", "frontier", "device", "device_name")
    )
    revision = environment.get("git_revision")
    recipe_valid = (
        all(key in train and train[key] is not None for key in RECIPE_KEYS)
        and all(
            _positive_int(train.get(key))
            for key in ("batch_size", "context_length", "gradient_accumulation")
        )
        and _valid_seed(train.get("warmup_steps"))
        and _finite(train.get("learning_rate"), positive=True)
        and _finite(train.get("min_learning_rate"), nonnegative=True)
        and _finite(train.get("weight_decay"), nonnegative=True)
        and _finite(train.get("grad_clip"), nonnegative=True)
        and isinstance(train.get("deterministic"), bool)
        and isinstance(train.get("mixed_precision"), str)
        and train["mixed_precision"] in {"auto", "fp32", "bf16", "fp16"}
        and (step_budget != token_budget)
    )
    return {
        "run_id_present": isinstance(run.get("run_id"), str) and bool(run["run_id"]),
        "run_completed": run.get("status") == "completed",
        "seed_valid": _valid_seed(config.get("seed")),
        "model_config_valid": _positive_int(model.get("layers"))
        and _positive_int(model.get("width")),
        "training_recipe_valid": recipe_valid,
        "tokenizer_valid": isinstance(tokenizer.get("name"), str)
        and bool(tokenizer["name"])
        and _positive_int(tokenizer.get("vocab_size")),
        "tokenizer_artifact_hash_valid": _valid_hash(tokenizer_hash)
        and _valid_hash(protocol_tokenizer_hash)
        and tokenizer_hash == protocol_tokenizer_hash,
        "research_data_provenance_valid": source_metadata_valid
        and _valid_hash(data.get("manifest_sha256"))
        and _valid_hash(data.get("source_sha256"))
        and artifact_checks.get("data_manifest_file_matches_summary") is True
        and isinstance(preprocessing, dict)
        and _valid_hash(data.get("preprocessing_sha256"))
        and data["preprocessing_sha256"] == _canonical_hash(preprocessing),
        "data_hashes_valid": _valid_hash(data.get("train_sha256"))
        and _valid_hash(data.get("validation_sha256")),
        "fixture_flag_valid": isinstance(data.get("synthetic_fixture"), bool),
        "evaluation_protocol_valid": bool(protocol) and _valid_hash(protocol.get("dataset_sha256")),
        "mean_nll_valid": _finite(metrics.get("mean_nll"), nonnegative=True),
        "perplexity_valid": _finite(metrics.get("perplexity"), nonnegative=True)
        and metrics["perplexity"] >= 1.0,
        "training_flops_valid": _finite(training.get("estimated_flops"), positive=True),
        "compute_estimator_valid": isinstance(training.get("compute_estimator"), str)
        and bool(training["compute_estimator"]),
        "training_budget_valid": _training_budget_valid(run, config),
        "resolved_config_file_valid": artifact_checks.get("resolved_config_file_valid") is True,
        "tokenizer_file_valid": artifact_checks.get("tokenizer_file_valid") is True,
        "data_files_match_summary": artifact_checks.get("data_files_match_summary") is True,
        "environment_file_matches_summary": artifact_checks.get("environment_file_matches_summary")
        is True,
        "environment_valid": all(key in environment for key in ENVIRONMENT_KEYS)
        and env_strings_valid
        and isinstance(environment.get("git_worktree_clean"), bool)
        and (revision is None or (isinstance(revision, str) and bool(revision))),
    }


def _environment_matches(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if not all(key in first and key in second for key in ENVIRONMENT_KEYS):
        return False
    for key in ("platform", "torch", "numpy", "frontier", "device", "device_name"):
        if not isinstance(first.get(key), str) or not first[key]:
            return False
        if not isinstance(second.get(key), str) or not second[key]:
            return False
    return (
        all(first.get(key) == second.get(key) for key in ENVIRONMENT_KEYS)
        and first.get("git_worktree_clean") is True
        and second.get("git_worktree_clean") is True
    )


def _float_or_none(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _evaluation_metric_or_none(value: Any, field: str) -> float | None:
    if field == "perplexity":
        return float(value) if _finite(value, nonnegative=True) and value >= 1 else None
    if field == "mean_nll":
        return float(value) if _finite(value, nonnegative=True) else None
    return _float_or_none(value)


def _delta_or_none(first: Any, second: Any) -> float | None:
    if _finite(first) and _finite(second):
        return second - first
    return None


def _warmup_fraction(train: dict[str, Any]) -> float | None:
    if not all(
        _positive_int(train.get(key))
        for key in ("batch_size", "context_length", "gradient_accumulation")
    ):
        return None
    batch_tokens = (
        train.get("batch_size", 0)
        * train.get("context_length", 0)
        * train.get("gradient_accumulation", 0)
    )
    if train.get("max_steps") is not None:
        steps = train["max_steps"]
    elif batch_tokens and train.get("max_tokens"):
        steps = math.ceil(train["max_tokens"] / batch_tokens)
    else:
        return None
    warmup = train.get("warmup_steps")
    if not _finite(steps, positive=True) or not _finite(warmup, nonnegative=True):
        return None
    return warmup / max(1, steps)


def _evidence_scope(runs: list[dict[str, Any]]) -> str:
    if any(_section(run, "data").get("synthetic_fixture") is True for run in runs):
        return "synthetic infrastructure fixture"
    origins = {_origin(run) for run in runs}
    if "unknown" in origins:
        return "corpus origin unknown"
    if origins == {"human"}:
        return "human-authored corpus metric"
    if origins == {"synthetic"}:
        return "synthetic corpus metric"
    return "mixed-origin corpus metric"


def compare_runs(
    baseline_path: str | Path, candidate_paths: list[str | Path], output_path: str | Path
) -> dict[str, Any]:
    base_root, baseline, base_config = _load(baseline_path)
    baseline_data = _section(baseline, "data")
    baseline_eval = _section(baseline, "evaluation")
    baseline_training = _section(baseline, "training")
    base_train = _section(base_config, "train")
    tolerance = 0.01
    rows = []

    for candidate_path in candidate_paths:
        cand_root, candidate, cand_config = _load(candidate_path)
        candidate_data = _section(candidate, "data")
        candidate_eval = _section(candidate, "evaluation")
        candidate_training = _section(candidate, "training")
        candidate_train = _section(cand_config, "train")
        baseline_validation = _run_validation(baseline, base_config)
        candidate_validation = _run_validation(candidate, cand_config)
        base_warmup = _warmup_fraction(base_train)
        cand_warmup = _warmup_fraction(candidate_train)
        warmup_delta = (
            abs(base_warmup - cand_warmup) / max(base_warmup, cand_warmup, 1e-12)
            if base_warmup is not None and cand_warmup is not None
            else None
        )
        recipe_matches = (
            all(
                key in base_train
                and key in candidate_train
                and base_train[key] == candidate_train[key]
                for key in RECIPE_KEYS
                if key != "warmup_steps"
            )
            and baseline_validation["training_recipe_valid"]
            and candidate_validation["training_recipe_valid"]
            and warmup_delta is not None
            and warmup_delta <= 0.05
        )
        base_ppl = _section(baseline_eval, "perplexity").get("perplexity")
        cand_ppl = _section(candidate_eval, "perplexity").get("perplexity")
        base_flops = baseline_training.get("estimated_flops")
        cand_flops = candidate_training.get("estimated_flops")
        if _finite(base_flops, positive=True) and _finite(cand_flops, positive=True):
            relative_delta = abs(cand_flops - base_flops) / base_flops
            compute_matches = relative_delta <= tolerance
        else:
            relative_delta = None
            compute_matches = False

        checks = {
            "baseline_artifact_valid": all(
                passed
                for key, passed in baseline_validation.items()
                if key not in {"tokenizer_artifact_hash_valid", "research_data_provenance_valid"}
            ),
            "candidate_artifact_valid": all(
                passed
                for key, passed in candidate_validation.items()
                if key not in {"tokenizer_artifact_hash_valid", "research_data_provenance_valid"}
            ),
            "same_training_data": _valid_hash(baseline_data.get("train_sha256"))
            and _valid_hash(candidate_data.get("train_sha256"))
            and baseline_data["train_sha256"] == candidate_data["train_sha256"],
            "same_validation_data": _valid_hash(baseline_data.get("validation_sha256"))
            and _valid_hash(candidate_data.get("validation_sha256"))
            and baseline_data["validation_sha256"] == candidate_data["validation_sha256"],
            "same_tokenizer": baseline_validation["tokenizer_valid"]
            and candidate_validation["tokenizer_valid"]
            and baseline_validation["tokenizer_artifact_hash_valid"]
            and candidate_validation["tokenizer_artifact_hash_valid"]
            and all(
                baseline["tokenizer"].get(key) == candidate["tokenizer"].get(key)
                for key in ("name", "vocab_size", "artifact_sha256")
            ),
            "same_data_kind": baseline_validation["fixture_flag_valid"]
            and candidate_validation["fixture_flag_valid"]
            and baseline_data["synthetic_fixture"] == candidate_data["synthetic_fixture"],
            "same_evaluation_protocol": baseline_validation["evaluation_protocol_valid"]
            and candidate_validation["evaluation_protocol_valid"]
            and baseline_eval.get("protocol") == candidate_eval.get("protocol"),
            "same_training_recipe": recipe_matches,
            "same_seed": _valid_seed(base_config.get("seed"))
            and _valid_seed(cand_config.get("seed"))
            and base_config["seed"] == cand_config["seed"],
            "same_environment": _environment_matches(
                _section(baseline, "environment"), _section(candidate, "environment")
            ),
            "both_runs_completed": baseline.get("status") == "completed"
            and candidate.get("status") == "completed",
            "same_compute_estimator": isinstance(baseline_training.get("compute_estimator"), str)
            and bool(baseline_training["compute_estimator"])
            and baseline_training.get("compute_estimator")
            == candidate_training.get("compute_estimator"),
            "training_compute_within_1_percent": compute_matches,
        }
        baseline_revision = _section(baseline, "environment").get("git_revision")
        candidate_revision = _section(candidate, "environment").get("git_revision")
        blockers = [name for name, passed in checks.items() if not passed]
        if not isinstance(baseline_revision, str) or not baseline_revision:
            blockers.append("baseline_pinned_revision_missing")
        if not isinstance(candidate_revision, str) or not candidate_revision:
            blockers.append("candidate_pinned_revision_missing")
        if _origin(baseline) == "unknown" or _origin(candidate) == "unknown":
            blockers.append("explicit_corpus_content_origin_missing")
        if not baseline_validation["tokenizer_artifact_hash_valid"]:
            blockers.append("baseline_tokenizer_artifact_hash_missing_or_invalid")
        if not candidate_validation["tokenizer_artifact_hash_valid"]:
            blockers.append("candidate_tokenizer_artifact_hash_missing_or_invalid")
        if not baseline_validation["research_data_provenance_valid"]:
            blockers.append("baseline_research_data_provenance_missing_or_invalid")
        if not candidate_validation["research_data_provenance_valid"]:
            blockers.append("candidate_research_data_provenance_missing_or_invalid")
        rows.append(
            {
                "baseline_run": str(base_root),
                "candidate_run": str(cand_root),
                "comparable_for_metric_comparison": all(checks.values()),
                "comparable_for_improvement_claim": False,
                "improvement_claim_requires": (
                    "At least three matched seeds, explicit corpus origin, a pinned clean revision, "
                    "and equivalent disclosed training compute."
                ),
                "evidence_scope": _evidence_scope([baseline, candidate]),
                "checks": checks,
                "baseline_validation_failures": [
                    key for key, passed in baseline_validation.items() if not passed
                ],
                "candidate_validation_failures": [
                    key for key, passed in candidate_validation.items() if not passed
                ],
                "eligibility_blockers": blockers,
                "training_recipe_warmup_fraction_relative_delta": warmup_delta,
                "estimated_training_flop_relative_delta": relative_delta,
                "perplexity_delta_candidate_minus_baseline": _delta_or_none(
                    _evaluation_metric_or_none(base_ppl, "perplexity"),
                    _evaluation_metric_or_none(cand_ppl, "perplexity"),
                ),
                "parameter_count_delta": _delta_or_none(
                    _section(baseline, "profiling").get("parameter_count"),
                    _section(candidate, "profiling").get("parameter_count"),
                ),
                "model_tensor_bytes_delta": _delta_or_none(
                    _section(baseline, "profiling").get("model_tensor_bytes"),
                    _section(candidate, "profiling").get("model_tensor_bytes"),
                ),
            }
        )

    report = {
        "schema_version": 2,
        "baseline_run": str(base_root),
        "compute_estimator": baseline_training.get("compute_estimator"),
        "required_compute_relative_tolerance": tolerance,
        "results": rows,
    }
    write_json(output_path, report)
    return report


def _mean_and_sample_std(values: list[float | None]) -> dict[str, float | int | None]:
    valid = [float(value) for value in values if _finite(value)]
    complete = len(valid) == len(values)
    mean = sum(valid) / len(valid) if complete and valid else None
    if complete and len(valid) == 1:
        sample_std = 0.0
    elif complete and mean is not None:
        sample_std = math.sqrt(sum((value - mean) ** 2 for value in valid) / (len(valid) - 1))
    else:
        sample_std = None
    return {"mean": mean, "sample_std": sample_std, "valid_n": len(valid), "total_n": len(values)}


def compare_seeded_runs(
    baseline_paths: list[str | Path],
    candidate_paths: list[str | Path],
    output_path: str | Path,
    minimum_seeds: int = 3,
) -> dict[str, Any]:
    if minimum_seeds < 3:
        raise ValueError("minimum_seeds must be at least 3; the claim floor cannot be weakened")
    if len(baseline_paths) < minimum_seeds or len(candidate_paths) < minimum_seeds:
        raise ValueError(f"at least {minimum_seeds} runs per group are required")
    if len(baseline_paths) != len(candidate_paths):
        raise ValueError("baseline and candidate groups must contain the same number of runs")

    def index_by_seed(
        paths: list[str | Path],
    ) -> dict[int, tuple[Path, dict[str, Any], dict[str, Any]]]:
        indexed = {}
        for path in paths:
            loaded = _load(path)
            seed = loaded[2].get("seed")
            if not _valid_seed(seed):
                raise TypeError(
                    f"run {path} has no valid non-negative integer seed in its resolved config"
                )
            if seed in indexed:
                raise ValueError(f"duplicate seed {seed} in run group")
            indexed[seed] = loaded
        return indexed

    baseline_by_seed = index_by_seed(baseline_paths)
    candidate_by_seed = index_by_seed(candidate_paths)
    seeds = sorted(baseline_by_seed)
    if set(seeds) != set(candidate_by_seed):
        raise ValueError("baseline and candidate groups must have identical seed sets")
    all_runs = list(baseline_by_seed.values()) + list(candidate_by_seed.values())
    run_ids = [run[1].get("run_id") for run in all_runs]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids) or len(
        set(run_ids)
    ) != len(run_ids):
        raise ValueError("every run must have a unique non-empty run_id across both groups")

    baseline_runs = [baseline_by_seed[seed] for seed in seeds]
    candidate_runs = [candidate_by_seed[seed] for seed in seeds]

    def all_same(values: list[Any]) -> bool:
        return bool(values) and all(value == values[0] for value in values[1:])

    def model_signature(run: tuple[Path, dict[str, Any], dict[str, Any]]) -> str | None:
        model = _section(run[2], "model")
        if not _positive_int(model.get("layers")) or not _positive_int(model.get("width")):
            return None
        return json.dumps(model, sort_keys=True)

    def training_signature(run: tuple[Path, dict[str, Any], dict[str, Any]]) -> str | None:
        train = _section(run[2], "train")
        if not all(key in train and train[key] is not None for key in RECIPE_KEYS):
            return None
        if _positive_int(train.get("max_steps")) == _positive_int(train.get("max_tokens")):
            return None
        normalized = dict(train)
        normalized.pop("resume", None)
        return json.dumps(normalized, sort_keys=True)

    train_hashes = [_section(run[1], "data").get("train_sha256") for run in all_runs]
    validation_hashes = [_section(run[1], "data").get("validation_sha256") for run in all_runs]
    tokenizers = [run[1].get("tokenizer") for run in all_runs]
    protocols = [_section(run[1], "evaluation").get("protocol") for run in all_runs]
    environments = [
        {key: _section(run[1], "environment").get(key) for key in ENVIRONMENT_KEYS}
        for run in all_runs
    ]
    origins = [_origin(run[1]) for run in all_runs]
    group_checks = {
        "all_run_artifacts_valid": all(
            all(
                passed
                for key, passed in _run_validation(run[1], run[2]).items()
                if key not in {"tokenizer_artifact_hash_valid", "research_data_provenance_valid"}
            )
            for run in all_runs
        ),
        "same_training_data_across_all_runs": all(_valid_hash(value) for value in train_hashes)
        and all_same(train_hashes),
        "same_validation_data_across_all_runs": all(
            _valid_hash(value) for value in validation_hashes
        )
        and all_same(validation_hashes),
        "same_tokenizer_across_all_runs": all(
            isinstance(value, dict)
            and isinstance(value.get("name"), str)
            and bool(value["name"])
            and _positive_int(value.get("vocab_size"))
            and _valid_hash(value.get("artifact_sha256"))
            for value in tokenizers
        )
        and all_same(
            [
                {
                    "name": value["name"],
                    "vocab_size": value.get("vocab_size"),
                    "artifact_sha256": value.get("artifact_sha256"),
                }
                for value in tokenizers
                if isinstance(value, dict)
            ]
        ),
        "same_evaluation_protocol_across_all_runs": all(
            isinstance(value, dict) and bool(value) and _valid_hash(value.get("dataset_sha256"))
            for value in protocols
        )
        and all_same(protocols),
        "same_environment_across_all_runs": all(
            _run_validation(run[1], run[2])["environment_valid"] for run in all_runs
        )
        and all_same(environments),
        "same_corpus_origin_across_all_runs": all_same(origins),
        "all_source_worktrees_clean": all(
            _section(run[1], "environment").get("git_worktree_clean") is True for run in all_runs
        ),
        "baseline_architecture_fixed_across_seeds": all(
            signature is not None for signature in [model_signature(run) for run in baseline_runs]
        )
        and all_same([model_signature(run) for run in baseline_runs]),
        "candidate_architecture_fixed_across_seeds": all(
            signature is not None for signature in [model_signature(run) for run in candidate_runs]
        )
        and all_same([model_signature(run) for run in candidate_runs]),
        "baseline_recipe_fixed_across_seeds": all(
            signature is not None
            for signature in [training_signature(run) for run in baseline_runs]
        )
        and all_same([training_signature(run) for run in baseline_runs]),
        "candidate_recipe_fixed_across_seeds": all(
            signature is not None
            for signature in [training_signature(run) for run in candidate_runs]
        )
        and all_same([training_signature(run) for run in candidate_runs]),
    }

    pairwise_results = []
    with tempfile.TemporaryDirectory(prefix="frontier-seeded-comparison-") as temp_dir:
        for seed in seeds:
            pair_path = Path(temp_dir) / f"seed-{seed}.json"
            pairwise_results.append(
                compare_runs(baseline_by_seed[seed][0], [candidate_by_seed[seed][0]], pair_path)[
                    "results"
                ][0]
            )
    pair_checks_pass = all(row["comparable_for_metric_comparison"] for row in pairwise_results)
    known_nonfixture_scope = all(
        _origin(run[1]) in {"human", "synthetic", "mixed"}
        and _section(run[1], "data").get("synthetic_fixture") is False
        for run in all_runs
    )
    human_corpus = all(_origin(run[1]) == "human" for run in all_runs)
    git_revisions = [_section(run[1], "environment").get("git_revision") for run in all_runs]
    source_revision_pinned = all_same(git_revisions) and all(
        isinstance(revision, str) and bool(revision) for revision in git_revisions
    )
    source_worktrees_clean = all(
        _section(run[1], "environment").get("git_worktree_clean") is True for run in all_runs
    )
    tokenizer_artifacts_pinned = all(
        _run_validation(run[1], run[2])["tokenizer_artifact_hash_valid"] for run in all_runs
    )
    research_data_provenance_complete = all(
        _run_validation(run[1], run[2])["research_data_provenance_valid"] for run in all_runs
    )
    metric_comparable = all(group_checks.values()) and pair_checks_pass
    scoped_claim_eligible = (
        metric_comparable
        and len(seeds) >= 3
        and len(seeds) >= minimum_seeds
        and known_nonfixture_scope
        and tokenizer_artifacts_pinned
        and research_data_provenance_complete
        and source_revision_pinned
        and source_worktrees_clean
    )
    human_corpus_claim_eligible = scoped_claim_eligible and human_corpus

    def run_metric_values(
        runs: list[tuple[Path, dict[str, Any], dict[str, Any]]], field: str
    ) -> list[float | None]:
        return [
            _evaluation_metric_or_none(
                _section(_section(run[1], "evaluation"), "perplexity").get(field), field
            )
            for run in runs
        ]

    baseline_ppl = run_metric_values(baseline_runs, "perplexity")
    candidate_ppl = run_metric_values(candidate_runs, "perplexity")
    baseline_nll = run_metric_values(baseline_runs, "mean_nll")
    candidate_nll = run_metric_values(candidate_runs, "mean_nll")
    paired_ppl = [_delta_or_none(a, b) for a, b in zip(baseline_ppl, candidate_ppl)]
    paired_nll = [_delta_or_none(a, b) for a, b in zip(baseline_nll, candidate_nll)]
    flop_deltas = [
        _float_or_none(row.get("estimated_training_flop_relative_delta"))
        for row in pairwise_results
    ]
    claim_scope = _evidence_scope([run[1] for run in all_runs])
    blockers = [name for name, passed in group_checks.items() if not passed]
    if not pair_checks_pass:
        blockers.append("pairwise comparability checks failed")
    if not known_nonfixture_scope:
        blockers.append("known non-fixture corpus origin required for a scoped claim")
    if not tokenizer_artifacts_pinned:
        blockers.append("all runs must pin and internally verify tokenizer artifacts")
    if not research_data_provenance_complete:
        blockers.append("all runs must pin corpus source, license, revision and preprocessing")
    if len(seeds) < max(3, minimum_seeds):
        blockers.append(f"at least {max(3, minimum_seeds)} matched seeds required")
    if not source_revision_pinned:
        blockers.append("all runs must record one pinned git revision")
    if not source_worktrees_clean:
        blockers.append("all runs must use a clean Git working tree")

    report = {
        "schema_version": 3,
        "comparison_type": "paired multi-seed",
        "minimum_seeds_for_claim": max(3, minimum_seeds),
        "source_revision": git_revisions[0] if source_revision_pinned else None,
        "source_worktrees_clean": source_worktrees_clean,
        "seeds": seeds,
        "baseline_runs": {str(seed): str(baseline_by_seed[seed][0]) for seed in seeds},
        "candidate_runs": {str(seed): str(candidate_by_seed[seed][0]) for seed in seeds},
        "comparable_for_metric_comparison": metric_comparable,
        "comparable_for_improvement_claim": scoped_claim_eligible,
        "comparable_for_human_corpus_claim": human_corpus_claim_eligible,
        "evidence_scope": claim_scope,
        "claim_scope": claim_scope,
        "claim_scope_limitation": (
            "synthetic-domain result; does not establish human-authored or broad language capability"
            if claim_scope == "synthetic corpus metric"
            else "mixed-origin result; conclusions are limited to this exact corpus and task"
            if claim_scope == "mixed-origin corpus metric"
            else "infrastructure fixture only; no research capability claim"
            if claim_scope == "synthetic infrastructure fixture"
            else "human-authored corpus metric; conclusions are limited to this exact corpus and task"
            if claim_scope == "human-authored corpus metric"
            else "corpus origin is unresolved; no improvement claim"
        ),
        "claim_blockers": [] if scoped_claim_eligible else blockers,
        "human_corpus_claim_blockers": (
            []
            if human_corpus_claim_eligible
            else [*blockers, "human-authored corpus origin required for this claim"]
        ),
        "group_checks": group_checks,
        "claim_checks": {
            "known_nonfixture_scope": known_nonfixture_scope,
            "tokenizer_artifacts_pinned": tokenizer_artifacts_pinned,
            "research_data_provenance_complete": research_data_provenance_complete,
        },
        "pairwise_results": pairwise_results,
        "metrics": {
            "baseline_perplexity": _mean_and_sample_std(baseline_ppl),
            "candidate_perplexity": _mean_and_sample_std(candidate_ppl),
            "paired_perplexity_delta_candidate_minus_baseline": _mean_and_sample_std(paired_ppl),
            "baseline_mean_nll": _mean_and_sample_std(baseline_nll),
            "candidate_mean_nll": _mean_and_sample_std(candidate_nll),
            "paired_mean_nll_delta_candidate_minus_baseline": _mean_and_sample_std(paired_nll),
            "estimated_training_flop_relative_delta": _mean_and_sample_std(flop_deltas),
        },
    }
    write_json(output_path, report)
    return report
