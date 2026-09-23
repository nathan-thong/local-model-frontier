"""Run the frozen S06-DP-001 CPU synthetic dependency screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from frontier.config import ModelConfig
from frontier.data.dependency import generate_s06_dependency_splits
from frontier.evaluation.dependency import evaluate_dependency_recall
from frontier.experiments.results import append_jsonl, write_json
from frontier.profiling.compute import ESTIMATOR_ID, estimate_training_compute
from frontier.profiling.memory import process_memory_bytes, unique_tensor_storage_bytes
from frontier.training.dependency_screen import state_dict_sha256, train_dependency_seed

EXPECTED_CANONICAL_CONFIG_SHA256 = (
    "4530e7367fb805031297d57ee99a95e87ad17e210d87827f68f09c934b933258"
)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state(repo_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, not status.strip()


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
    }


def _write_seed_result(seed_dir: Path, record: dict[str, Any]) -> None:
    write_json(seed_dir / "seed_result.json", record)


def _preserve_aborted_run(output_path: Path, error: BaseException, status: str) -> None:
    """Turn a started run's logs into an explicit failed/incomplete summary."""
    target = output_path.resolve()
    if not target.is_dir():
        return
    seed_records = []
    for seed_path in sorted(target.glob("seed-*/seed_result.json")):
        try:
            record = json.loads(seed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") == "starting":
            record["status"] = "incomplete"
            record["reason"] = "process stopped before the seed result was finalized"
            write_json(seed_path, record)
        seed_records.append(record)
    summary = {
        "schema_version": 1,
        "experiment_id": "S06-DP-001",
        "status": status,
        "decision_rule_passed": False,
        "reason": f"{type(error).__name__}: {error}",
        "seeds": seed_records,
        "preserved_seed_result_files": [
            path.relative_to(target).as_posix() for path in target.glob("seed-*/seed_result.json")
        ],
        "preserved_update_log_files": [
            path.relative_to(target).as_posix() for path in target.glob("seed-*/updates.jsonl")
        ],
        "scope": "Interrupted or failed S06-DP-001 run; consult retained per-seed logs.",
    }
    summary_path = target / "summary.json"
    write_json(summary_path, summary)
    manifest_path = target / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "experiment_id": "S06-DP-001"}
    manifest.update(
        {
            "status": status,
            "failure_reason": summary["reason"],
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        }
    )
    write_json(manifest_path, manifest)


def _training_compute(model_config: ModelConfig, consumed_tokens: int, context: int) -> int:
    if consumed_tokens == 0:
        return 0
    estimate = estimate_training_compute(model_config, consumed_tokens, context)
    if estimate["status"] != "estimated":
        raise RuntimeError("training compute estimate became unavailable")
    return int(estimate["estimated_flops"])


def _evaluation_forward_compute(model_config: ModelConfig, scored_tokens: int, context: int) -> int:
    if scored_tokens == 0:
        return 0
    estimate = estimate_training_compute(model_config, scored_tokens, context)
    if estimate["status"] != "estimated" or estimate["estimated_flops"] % 3:
        raise RuntimeError("cannot derive evaluation forward estimate from the frozen estimator")
    return int(estimate["estimated_flops"] // 3)


def _score_passes(config: dict[str, Any], seed_records: list[dict[str, Any]]) -> bool:
    rule = config["decision_rule"]
    for record in seed_records:
        if record.get("status") != "completed" or not record.get(
            "all_losses_and_gradient_norms_finite", False
        ):
            return False
        evaluation = record.get("evaluation")
        if not evaluation or not evaluation.get("complete"):
            return False
        metrics = evaluation.get("metrics", {})
        if not metrics.get("all_scored_values_finite", False):
            return False
        overall = metrics.get("full_vocabulary_exact_match")
        if overall is None or overall < rule["per_seed_full_vocabulary_exact_match_minimum"]:
            return False
        strata = evaluation.get("joint_strata", [])
        if len(strata) != len(config["task"]["joint_strata"]):
            return False
        for item in strata:
            if not item.get("all_scored_values_finite", False):
                return False
            value = item.get("full_vocabulary_exact_match")
            if (
                value is None
                or value < rule["per_seed_full_vocabulary_exact_match_minimum_each_joint_stratum"]
            ):
                return False
    return len(seed_records) == len(config["seeds"])


def run_screen(config_path: Path, output_path: Path) -> dict[str, Any]:
    run_started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    expected_config_path = (repo_root / "configs/s06/s06_dependency_screen.json").resolve()
    config_path = config_path.resolve()
    if config_path != expected_config_path:
        raise ValueError("S06-DP-001 requires its committed preregistration config")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    canonical_hash = _canonical_json_sha256(config)
    if canonical_hash != EXPECTED_CANONICAL_CONFIG_SHA256:
        raise ValueError("S06-DP-001 config differs from the preregistered canonical JSON")
    if config.get("experiment_id") != "S06-DP-001":
        raise ValueError("config experiment_id must be S06-DP-001")
    model_config = ModelConfig(**config["model"])
    model_config.validate()
    source_revision, source_clean = _git_state(repo_root)
    if not source_clean:
        raise RuntimeError("S06-DP-001 requires a clean Git worktree")
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            config["preregistration_revision"],
            source_revision,
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("run revision must descend from the frozen preregistration revision")

    output_path = output_path.resolve()
    runs_root = (repo_root / "runs").resolve()
    if output_path == runs_root or not output_path.is_relative_to(runs_root):
        raise ValueError("run output must be a child of the repository's ignored runs/ directory")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite dependency-screen output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir()
    shutil.copyfile(config_path, output_path / "resolved_config.json")

    training = config["training"]
    torch.set_num_threads(training["torch_num_threads"])
    torch.use_deterministic_algorithms(training["deterministic"])
    compute_budget = config["compute_budget"]
    task = config["task"]
    train_tokens = len(config["seeds"]) * task["training_examples_per_seed"] * task["input_length"]
    eval_tokens = task["evaluation_examples_total"] * task["input_length"] * len(config["seeds"])
    training_estimate = estimate_training_compute(model_config, train_tokens, task["input_length"])
    evaluation_estimate = estimate_training_compute(model_config, eval_tokens, task["input_length"])
    planned_train_flops = int(training_estimate["estimated_flops"])
    planned_eval_flops = int(evaluation_estimate["estimated_flops"] // 3)
    planned_total_flops = planned_train_flops + planned_eval_flops
    if (
        training_estimate["estimator"] != compute_budget["estimator"]
        or planned_train_flops != compute_budget["training_estimated_flops_all_seeds"]
        or planned_eval_flops != compute_budget["evaluation_forward_estimated_flops_all_seeds"]
        or planned_total_flops != compute_budget["total_estimated_flops"]
        or planned_total_flops > compute_budget["total_estimated_flops_cap"]
    ):
        raise RuntimeError("recomputed analytical budget differs from the preregistered budget")

    start_utc = datetime.now(UTC).isoformat()
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "running",
        "started_at_utc": start_utc,
        "source_revision": source_revision,
        "source_worktree_clean_before_run": source_clean,
        "preregistration_revision": config["preregistration_revision"],
        "config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_canonical_json_sha256": canonical_hash,
        "resolved_config_path": "resolved_config.json",
        "estimator": ESTIMATOR_ID,
        "planned_training_input_tokens": train_tokens,
        "planned_evaluation_input_tokens": eval_tokens,
        "planned_training_estimated_flops": planned_train_flops,
        "planned_evaluation_forward_estimated_flops": planned_eval_flops,
        "planned_total_estimated_flops": planned_total_flops,
        "compute_cap_estimated_flops": compute_budget["total_estimated_flops_cap"],
        "cumulative_wall_clock_cap_seconds": training["cumulative_wall_clock_cap_seconds"],
        "environment": _environment(),
        "scope": config["scope"],
    }
    write_json(output_path / "run_manifest.json", run_manifest)

    elapsed = lambda: time.perf_counter() - run_started
    cap_seconds = training["cumulative_wall_clock_cap_seconds"]
    stop_requested = lambda: elapsed() >= cap_seconds
    datasets, data_manifest = generate_s06_dependency_splits(config)
    write_json(output_path / "data_manifest.json", data_manifest)
    memory_readings = [{"phase": "after_data_generation", **process_memory_bytes()}]

    seed_records: list[dict[str, Any]] = []
    all_seeds_finite = True
    for seed in config["seeds"]:
        seed_name = f"training_seed_{seed}"
        seed_dir = output_path / f"seed-{seed}"
        seed_dir.mkdir()
        updates_path = seed_dir / "updates.jsonl"
        record: dict[str, Any] = {
            "seed": seed,
            "status": "starting",
            "updates_completed": 0,
            "tokens_seen": 0,
            "all_losses_and_gradient_norms_finite": True,
            "training_dataset_sha256": datasets[seed_name].dataset_sha256,
            "evaluation_dataset_sha256": datasets["evaluation"].dataset_sha256,
            "started_at_elapsed_seconds": elapsed(),
        }
        _write_seed_result(seed_dir, record)

        if stop_requested():
            record.update(
                {
                    "status": "incomplete",
                    "reason": "cumulative wall-clock cap reached before this seed started",
                    "completed_at_elapsed_seconds": elapsed(),
                }
            )
            seed_records.append(record)
            _write_seed_result(seed_dir, record)
            all_seeds_finite = False
            for later_seed in config["seeds"][config["seeds"].index(seed) + 1 :]:
                later_dir = output_path / f"seed-{later_seed}"
                later_dir.mkdir()
                skipped = {
                    "seed": later_seed,
                    "status": "not_started_incomplete",
                    "updates_completed": 0,
                    "tokens_seen": 0,
                    "all_losses_and_gradient_norms_finite": False,
                    "reason": "earlier cumulative runtime cap left no run budget",
                }
                _write_seed_result(later_dir, skipped)
                seed_records.append(skipped)
            break

        def record_event(event: dict[str, Any], path: Path = updates_path) -> None:
            append_jsonl(path, event)

        try:
            training_result = train_dependency_seed(
                config,
                datasets[seed_name],
                seed,
                stop_requested=stop_requested,
                record_event=record_event,
            )
            record.update(
                {
                    "status": training_result.status,
                    "updates_completed": training_result.updates_completed,
                    "updates_attempted": training_result.updates_attempted,
                    "tokens_seen": training_result.tokens_seen,
                    "all_losses_and_gradient_norms_finite": training_result.all_losses_and_gradient_norms_finite,
                    "initial_state_sha256": training_result.initial_state_sha256,
                    "training_elapsed_seconds": training_result.elapsed_seconds,
                    "reason": training_result.reason,
                    "parameter_count": training_result.model.parameter_count(),
                    "unique_parameter_and_buffer_tensor_bytes": unique_tensor_storage_bytes(
                        training_result.model
                    ),
                }
            )
            weights_path = seed_dir / "weights.pt"
            torch.save(
                {
                    name: tensor.detach().cpu()
                    for name, tensor in training_result.model.state_dict().items()
                },
                weights_path,
            )
            record["weights_file_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            record["weights_file_bytes"] = weights_path.stat().st_size
            record["final_state_sha256"] = state_dict_sha256(training_result.model)
            record["completed_at_elapsed_seconds"] = elapsed()
            if training_result.status == "completed" and not stop_requested():
                evaluation = evaluate_dependency_recall(
                    training_result.model,
                    datasets["evaluation"],
                    value_token_ids=task["value_token_ids"],
                    joint_strata=task["joint_strata"],
                    batch_size=training["batch_size"],
                    stop_requested=stop_requested,
                )
                for row in evaluation["per_example"]:
                    append_jsonl(seed_dir / "evaluation.jsonl", row)
                record["evaluation"] = {
                    key: value for key, value in evaluation.items() if key != "per_example"
                }
                if evaluation["complete"] and not stop_requested():
                    record["status"] = "completed"
                else:
                    record["status"] = "incomplete"
                    record["reason"] = "cumulative wall-clock cap interrupted final evaluation"
            elif training_result.status == "completed":
                record["status"] = "incomplete"
                record["reason"] = "cumulative wall-clock cap reached before final evaluation"
            if record["status"] != "completed" or not record.get(
                "all_losses_and_gradient_norms_finite", False
            ):
                all_seeds_finite = False
        except BaseException as error:  # Preserve each fixed seed's failure before continuing.
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            logged_events = []
            if updates_path.is_file():
                logged_events = [
                    json.loads(line)
                    for line in updates_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
            record["updates_attempted"] = sum(
                event.get("event") == "batch_start" for event in logged_events
            )
            record["updates_completed"] = sum(
                event.get("event") == "update" for event in logged_events
            )
            record["tokens_seen"] = max(
                (event.get("tokens_seen", 0) for event in logged_events), default=0
            )
            record.update(
                {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "all_losses_and_gradient_norms_finite": False,
                    "completed_at_elapsed_seconds": elapsed(),
                }
            )
            append_jsonl(
                updates_path,
                {
                    "event": "failure",
                    "seed": seed,
                    "reason": record["reason"],
                    "elapsed_seconds": elapsed(),
                },
            )
            all_seeds_finite = False
        seed_records.append(record)
        _write_seed_result(seed_dir, record)
        memory_readings.append(
            {"phase": f"after_seed_{seed}", "elapsed_seconds": elapsed(), **process_memory_bytes()}
        )
        if stop_requested() and len(seed_records) < len(config["seeds"]):
            # Remaining seeds receive explicit not-started records on the next loop iteration.
            continue

    # Seeds after a budget stop are recorded by the cap branch; a training failure alone
    # never removes one of the fixed seeds.
    planned_seed_ids = list(config["seeds"])
    recorded_seed_ids = [record["seed"] for record in seed_records]
    for seed in planned_seed_ids:
        if seed in recorded_seed_ids:
            continue
        seed_dir = output_path / f"seed-{seed}"
        seed_dir.mkdir(exist_ok=True)
        skipped = {
            "seed": seed,
            "status": "not_started_incomplete",
            "updates_completed": 0,
            "tokens_seen": 0,
            "all_losses_and_gradient_norms_finite": False,
            "reason": "cumulative runtime cap expired before this seed started",
        }
        _write_seed_result(seed_dir, skipped)
        seed_records.append(skipped)
    seed_records.sort(key=lambda record: planned_seed_ids.index(record["seed"]))

    actual_training_tokens = sum(record.get("tokens_seen", 0) for record in seed_records)
    actual_evaluation_tokens = sum(
        record.get("evaluation", {}).get("scored_examples", 0) * task["input_length"]
        for record in seed_records
    )
    actual_train_flops = _training_compute(
        model_config, actual_training_tokens, task["input_length"]
    )
    actual_eval_flops = _evaluation_forward_compute(
        model_config, actual_evaluation_tokens, task["input_length"]
    )
    source_revision_end, source_clean_end = _git_state(repo_root)
    elapsed_seconds = elapsed()
    runtime_exceeded = max(0.0, elapsed_seconds - cap_seconds)
    threshold_pass = _score_passes(config, seed_records)
    if runtime_exceeded > 0 or any(
        record["status"] in {"incomplete", "not_started_incomplete"} for record in seed_records
    ):
        overall_status = "incomplete"
    elif threshold_pass:
        overall_status = "passed"
    else:
        overall_status = "failed"
    source_state_unchanged = source_revision_end == source_revision and source_clean_end
    if not source_state_unchanged:
        overall_status = "failed"
        threshold_pass = False
    memory_readings.append(
        {"phase": "whole_run_end", "elapsed_seconds": elapsed_seconds, **process_memory_bytes()}
    )
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": overall_status,
        "decision_rule_passed": threshold_pass,
        "all_seeds_losses_and_gradient_norms_finite": all_seeds_finite,
        "source_revision": source_revision,
        "source_worktree_clean_before_run": source_clean,
        "source_revision_after_run": source_revision_end,
        "source_worktree_clean_after_run": source_clean_end,
        "source_state_unchanged_during_run": source_state_unchanged,
        "preregistration_revision": config["preregistration_revision"],
        "config_file_sha256": run_manifest["config_file_sha256"],
        "config_canonical_json_sha256": canonical_hash,
        "data_manifest_sha256": hashlib.sha256(
            (output_path / "data_manifest.json").read_bytes()
        ).hexdigest(),
        "environment": _environment(),
        "elapsed_seconds": elapsed_seconds,
        "cumulative_wall_clock_cap_seconds": cap_seconds,
        "runtime_cap_exceeded_seconds": runtime_exceeded,
        "planned_estimated_flops": planned_total_flops,
        "compute_estimator": ESTIMATOR_ID,
        "compute_estimator_assumptions": training_estimate["assumptions"],
        "actual_training_estimated_flops": actual_train_flops,
        "actual_evaluation_forward_estimated_flops": actual_eval_flops,
        "actual_total_estimated_flops": actual_train_flops + actual_eval_flops,
        "estimated_flop_cap": compute_budget["total_estimated_flops_cap"],
        "training_tokens_seen_all_seeds": actual_training_tokens,
        "evaluation_tokens_scored_all_seeds": actual_evaluation_tokens,
        "evaluation_split_sha256": datasets["evaluation"].dataset_sha256,
        "memory_readings": memory_readings,
        "memory_scope": "process-lifetime high-water readings, not isolated optimizer-step peaks",
        "seeds": seed_records,
        "scope": config["scope"],
        "limitations": [
            "Synthetic task only; no architecture control or corpus training.",
            "RoPE-before-ELU-plus-one position treatment is part of the tested implementation.",
            "Later distractor count and value-to-query distance are coupled joint strata.",
            "FLOPs are analytical estimates and exclude scalar, feature-map, Python, optimizer, and autograd-intermediate work.",
            "CPU process-lifetime peak memory does not isolate an optimizer-step peak.",
            "CUDA and energy were not used or validated by this CPU screen.",
        ],
    }
    if not source_state_unchanged:
        summary["eligibility_failure"] = (
            "Git revision changed or worktree became dirty while the screen was running"
        )
    write_json(output_path / "summary.json", summary)
    run_manifest.update(
        {
            "status": overall_status,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "data_manifest_sha256": summary["data_manifest_sha256"],
            "source_revision_after_run": source_revision_end,
            "source_worktree_clean_after_run": source_clean_end,
            "summary_sha256": hashlib.sha256(
                (output_path / "summary.json").read_bytes()
            ).hexdigest(),
        }
    )
    write_json(output_path / "run_manifest.json", run_manifest)
    return summary


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs/s06/s06_dependency_screen.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "runs/s06-dp-001",
    )
    args = parser.parse_args()
    output_preexisting = args.output.resolve().exists()
    try:
        summary = run_screen(args.config, args.output)
    except KeyboardInterrupt as error:
        if not output_preexisting:
            _preserve_aborted_run(args.output, error, "incomplete")
        print("S06-DP-001 interrupted; partial records were retained.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - preserve every top-level failed run record.
        if not output_preexisting:
            _preserve_aborted_run(args.output, error, "failed")
        print(
            f"S06-DP-001 failed before a complete run record could be written: {error}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "experiment_id": summary["experiment_id"],
                "status": summary["status"],
                "decision_rule_passed": summary["decision_rule_passed"],
                "source_revision": summary["source_revision"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "actual_total_estimated_flops": summary["actual_total_estimated_flops"],
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
