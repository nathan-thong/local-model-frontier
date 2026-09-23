"""Serial execution and resume checks for validated sweep plans."""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from frontier.config import RunConfig
from frontier.experiments.planning import (
    _canonical_json,
    _planned_budget,
    _sha256,
    verify_sweep_plan,
)
from frontier.training.trainer import train


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_job(plan: dict[str, Any], job: dict[str, Any]) -> RunConfig:
    if not isinstance(job, dict) or not isinstance(job.get("resolved_config"), dict):
        raise TypeError("sweep plan contains a malformed job")
    config_dict = job["resolved_config"]
    if _sha256(_canonical_json(config_dict)) != job.get("config_sha256"):
        raise ValueError(f"job config hash mismatch: {job.get('job_id')}")
    config = RunConfig.from_dict(config_dict)
    if config.seed != job.get("seed"):
        raise ValueError(f"job seed mismatch: {job.get('job_id')}")
    if str(Path(config.output_dir).resolve()) != str(Path(job.get("output_dir", "")).resolve()):
        raise ValueError(f"job output path mismatch: {job.get('job_id')}")
    steps, tokens, estimate = _planned_budget(config)
    if (
        steps != job.get("planned_optimizer_steps")
        or tokens != job.get("planned_tokens")
        or estimate["estimated_flops"] != job.get("estimated_flops")
    ):
        raise ValueError(f"job budget differs from its resolved config: {job.get('job_id')}")
    if plan["total_estimated_flops"] > plan["total_compute_cap_flops"]:
        raise ValueError("sweep plan exceeds its declared total-compute cap")
    return config


def _validate_completed_run(
    run_dir: Path, config: RunConfig, job: dict[str, Any], plan: dict[str, Any]
) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"completed sweep job lacks summary.json: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError(f"run is not completed: {run_dir}")
    if summary.get("resolved_config") != config.to_dict():
        raise ValueError(f"run config differs from its immutable sweep job: {run_dir}")
    training = summary.get("training", {})
    if (
        training.get("tokens_seen") != job["planned_tokens"]
        or training.get("estimated_flops") != job["estimated_flops"]
    ):
        raise ValueError(f"run consumed a different budget than planned: {run_dir}")
    environment = summary.get("environment", {})
    planned_hardware = plan.get("hardware", {})
    for key in ("platform", "torch", "cuda_runtime", "cuda_available"):
        if key in planned_hardware and environment.get(key) != planned_hardware[key]:
            raise ValueError(f"run hardware field {key} differs from plan: {run_dir}")
    return summary


def _initial_status(plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status_type": "frontier-sweep-status-v1",
        "plan_sha256": plan["plan_sha256"],
        "plan_path_name": plan_path.name,
        "status": "pending",
        "jobs": {
            job["job_id"]: {
                "status": "not_started",
                "run_id": job["run_id"],
                "output_dir": job["output_dir"],
            }
            for job in plan["jobs"]
        },
    }


def execute_sweep(
    plan_path: str | Path,
    status_path: str | Path | None = None,
    max_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Run plan jobs serially, resuming only runs with matching saved state."""
    if max_runtime_seconds is not None and (
        not math.isfinite(max_runtime_seconds) or max_runtime_seconds <= 0
    ):
        raise ValueError("max_runtime_seconds must be a finite positive duration")
    invocation_started = time.perf_counter()
    source = Path(plan_path).resolve()
    plan = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not verify_sweep_plan(plan):
        raise ValueError("sweep plan is malformed or its content hash does not verify")
    if plan.get("plan_type") != "frontier-sweep-plan-v1" or not isinstance(plan.get("jobs"), list):
        raise ValueError("unsupported or malformed sweep plan")
    if not plan["jobs"]:
        raise ValueError("sweep plan has no jobs")
    if any(
        not isinstance(job, dict) or not isinstance(job.get("job_id"), str) for job in plan["jobs"]
    ):
        raise ValueError("sweep plan contains a malformed job ID")
    if len({job["job_id"] for job in plan["jobs"]}) != len(plan["jobs"]):
        raise ValueError("sweep plan contains duplicate or malformed job IDs")
    if any(
        not isinstance(job.get("estimated_flops"), int)
        or isinstance(job["estimated_flops"], bool)
        or job["estimated_flops"] <= 0
        for job in plan["jobs"]
    ):
        raise ValueError("sweep plan contains an invalid job compute estimate")
    if sum(job.get("estimated_flops", 0) for job in plan["jobs"]) != plan.get(
        "total_estimated_flops"
    ):
        raise ValueError("sweep plan total does not equal the sum of its job estimates")
    if plan["total_estimated_flops"] > plan["total_compute_cap_flops"]:
        raise ValueError("sweep plan exceeds its declared total-compute cap")

    destination = Path(status_path).resolve() if status_path else source.with_suffix(".status.json")
    if destination == source:
        raise ValueError("status output must not overwrite the immutable sweep plan")
    lock_path = source.with_suffix(source.suffix + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError(
            f"sweep lock exists at {lock_path}; inspect the prior executor before removing it"
        ) from error

    try:
        os.write(
            lock_fd,
            json.dumps({"pid": os.getpid(), "plan_sha256": plan["plan_sha256"]}).encode("utf-8"),
        )
        os.close(lock_fd)
        lock_fd = -1

        status = _initial_status(plan, source)
        if destination.exists():
            saved = json.loads(destination.read_text(encoding="utf-8"))
            if (
                saved.get("status_type") != "frontier-sweep-status-v1"
                or saved.get("plan_sha256") != plan["plan_sha256"]
                or set(saved.get("jobs", {})) != set(status["jobs"])
            ):
                raise ValueError("existing sweep status does not match this immutable plan")
            status = saved
            if any(
                not isinstance(entry, dict)
                or entry.get("status") not in {"not_started", "running", "completed", "failed"}
                for entry in status["jobs"].values()
            ):
                raise ValueError("existing sweep status contains an unknown job state")
            status.pop("pause_reason", None)

        _atomic_json(destination, status)
        completed_compute = 0
        failure = None
        paused = False
        for job in plan["jobs"]:
            job_id = job["job_id"]
            job_status = status["jobs"][job_id]
            config = _validate_job(plan, job)
            run_dir = Path(job["output_dir"])

            if job_status["status"] == "completed":
                _validate_completed_run(run_dir, config, job, plan)
                completed_compute += job["estimated_flops"]
                continue
            if job_status["status"] == "failed":
                failure = job_status.get("error", "previous attempt failed")
                break
            elapsed_before_job = time.perf_counter() - invocation_started
            if max_runtime_seconds is not None and elapsed_before_job >= max_runtime_seconds:
                paused = True
                status["pause_reason"] = "runtime_budget"
                break
            if completed_compute + job["estimated_flops"] > plan["total_compute_cap_flops"]:
                failure = "next job would exceed the declared compute cap"
                job_status.update({"status": "failed", "error": failure})
                _atomic_json(destination, status)
                break

            run_summary = run_dir / "summary.json"
            checkpoint = run_dir / "checkpoints" / "last.pt"
            if run_summary.exists():
                try:
                    prior_summary = json.loads(run_summary.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    failure = f"cannot resume unreadable run summary: {error}"
                    job_status.update({"status": "failed", "error": failure})
                    _atomic_json(destination, status)
                    break
                if prior_summary.get("status") == "completed":
                    try:
                        completed_summary = _validate_completed_run(run_dir, config, job, plan)
                    except (OSError, ValueError) as error:
                        failure = str(error)
                        job_status.update({"status": "failed", "error": failure})
                        _atomic_json(destination, status)
                        break
                    job_status.update(
                        {
                            "status": "completed",
                            "actual_tokens": completed_summary["training"]["tokens_seen"],
                            "actual_estimated_flops": completed_summary["training"][
                                "estimated_flops"
                            ],
                            "recovered_completed_run": True,
                        }
                    )
                    completed_compute += job["estimated_flops"]
                    _atomic_json(destination, status)
                    continue
                if prior_summary.get("status") != "running" or not checkpoint.is_file():
                    failure = f"existing run is not a compatible resumable interruption: {run_dir}"
                    job_status.update({"status": "failed", "error": failure})
                    _atomic_json(destination, status)
                    break
                resume = True
            else:
                if run_dir.exists() and any(run_dir.iterdir()):
                    failure = (
                        f"refusing to use a non-empty run directory without a summary: {run_dir}"
                    )
                    job_status.update({"status": "failed", "error": failure})
                    _atomic_json(destination, status)
                    break
                resume = False

            job_status.update(
                {"status": "running", "attempt": int(job_status.get("attempt", 0)) + 1}
            )
            status["status"] = "running"
            _atomic_json(destination, status)
            try:
                remaining_seconds = (
                    max_runtime_seconds - (time.perf_counter() - invocation_started)
                    if max_runtime_seconds is not None
                    else None
                )
                if remaining_seconds is not None and remaining_seconds <= 0:
                    job_status["status"] = "running"
                    _atomic_json(destination, status)
                    paused = True
                    status["pause_reason"] = "runtime_budget"
                    break
                train(config, resume=resume, stop_after_seconds=remaining_seconds)
                partial_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                if (
                    partial_summary.get("status") == "running"
                    and partial_summary.get("training", {}).get("stop_reason") == "runtime_budget"
                ):
                    job_status.update(
                        {
                            "status": "running",
                            "partial_tokens": partial_summary["training"].get("tokens_seen"),
                            "pause_reason": "runtime_budget",
                        }
                    )
                    _atomic_json(destination, status)
                    paused = True
                    status["pause_reason"] = "runtime_budget"
                    break
                summary = _validate_completed_run(run_dir, config, job, plan)
            except Exception as error:  # noqa: BLE001 - persist any failed run before stopping.
                failure = f"{type(error).__name__}: {error}"
                job_status.update({"status": "failed", "error": failure})
                _atomic_json(destination, status)
                break
            job_status.update(
                {
                    "status": "completed",
                    "actual_tokens": summary["training"]["tokens_seen"],
                    "actual_estimated_flops": summary["training"]["estimated_flops"],
                }
            )
            completed_compute += job["estimated_flops"]
            _atomic_json(destination, status)

        status["status"] = (
            "failed"
            if failure
            else "paused"
            if paused
            else "completed"
            if all(entry["status"] == "completed" for entry in status["jobs"].values())
            else "running"
        )
        if failure:
            status["error"] = failure
        status["completed_estimated_flops"] = completed_compute
        status["elapsed_seconds_this_invocation"] = time.perf_counter() - invocation_started
        status["runtime_budget_seconds"] = max_runtime_seconds
        _atomic_json(destination, status)
        return status
    finally:
        if lock_fd >= 0:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        if lock_path.exists():
            lock_path.unlink()
