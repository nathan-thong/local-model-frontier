"""Immutable, model-free planning for small paired experiment sweeps."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

import torch

from frontier.config import RunConfig
from frontier.experiments.results import write_json
from frontier.profiling.compute import estimate_training_compute

_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _checked_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase path-safe ID")
    return value


def _read_arm(arm: Any, field: str, base_dir: Path) -> tuple[str, Path, bytes, RunConfig]:
    if not isinstance(arm, dict) or set(arm) != {"id", "config"}:
        raise ValueError(f"{field} must contain exactly id and config")
    arm_id = _checked_id(arm["id"], f"{field}.id")
    config_path = Path(arm["config"])
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    config_path = config_path.resolve()
    raw = config_path.read_bytes()
    config = RunConfig.from_dict(json.loads(raw))
    if config.train.resume:
        raise ValueError(f"{field} config must not request resume")
    if config.data_dir is not None:
        data_dir = Path(config.data_dir)
        if not data_dir.is_absolute():
            # Match `frontier train`: config data_dir paths resolve from the caller's CWD.
            config.data_dir = str((Path.cwd() / data_dir).resolve())
    return arm_id, config_path, raw, config


def _planned_budget(config: RunConfig) -> tuple[int, int, dict[str, Any]]:
    train = config.train
    tokens_per_step = train.batch_size * train.context_length * train.gradient_accumulation
    steps = train.max_steps or (train.max_tokens + tokens_per_step - 1) // tokens_per_step
    tokens = steps * tokens_per_step
    estimate = estimate_training_compute(config.model, tokens, train.context_length)
    if estimate["status"] != "estimated" or not isinstance(estimate["estimated_flops"], int):
        raise ValueError(
            "cannot enforce a compute cap for an arm without a registered compute estimator"
        )
    return steps, tokens, estimate


def _hardware_record() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_selection_policy": "cuda if available, otherwise cpu",
    }


def plan_sweep(spec_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Resolve a seed-paired sweep, hash it, and write it without model allocation."""
    source_path = Path(spec_path).resolve()
    raw_spec = source_path.read_bytes()
    spec = json.loads(raw_spec)
    required = {
        "schema_version",
        "name",
        "baseline",
        "candidates",
        "seeds",
        "output_root",
        "total_compute_cap_flops",
    }
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError(f"sweep spec must contain exactly: {', '.join(sorted(required))}")
    if spec["schema_version"] != 1:
        raise ValueError("unsupported sweep specification schema_version")
    name = _checked_id(spec["name"], "name")
    seeds = spec["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be a non-empty list of distinct integer seeds")
    candidates = spec["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("at least one candidate arm is required")
    cap = spec["total_compute_cap_flops"]
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise ValueError("total_compute_cap_flops must be a positive integer")

    base_dir = source_path.parent
    baseline_id, baseline_path, baseline_raw, baseline_config = _read_arm(
        spec["baseline"], "baseline", base_dir
    )
    candidate_arms = [
        _read_arm(candidate, f"candidates[{index}]", base_dir)
        for index, candidate in enumerate(candidates)
    ]
    arm_ids = [baseline_id, *(arm_id for arm_id, *_ in candidate_arms)]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("arm IDs must be unique, including the baseline")

    output_root = Path(spec["output_root"])
    if not output_root.is_absolute():
        output_root = base_dir / output_root
    output_root = output_root.resolve()
    arms = [(baseline_id, baseline_path, baseline_raw, baseline_config, "baseline")]
    arms.extend(
        (arm_id, config_path, raw, config, "candidate")
        for arm_id, config_path, raw, config in candidate_arms
    )

    resolved_arms = []
    jobs = []
    arm_compute: dict[str, int] = {}
    for arm_id, config_path, raw_config, config, role in arms:
        steps, tokens, compute = _planned_budget(config)
        arm_compute[arm_id] = compute["estimated_flops"]
        resolved_config = config.to_dict()
        config_hash = _sha256(_canonical_json(resolved_config))
        resolved_arms.append(
            {
                "arm_id": arm_id,
                "role": role,
                "source_config_name": config_path.name,
                "source_config_sha256": _sha256(raw_config),
                "resolved_config_sha256": config_hash,
                "planned_optimizer_steps_per_seed": steps,
                "planned_tokens_per_seed": tokens,
                "compute_estimator": compute["estimator"],
                "estimated_flops_per_seed": compute["estimated_flops"],
                "estimated_macs_per_seed": compute["estimated_macs"],
                "compute_assumptions": compute["assumptions"],
            }
        )
        for seed in seeds:
            job_config = json.loads(json.dumps(resolved_config))
            job_config["seed"] = seed
            run_id = f"{name}-{arm_id}-seed-{seed}-{config_hash[:10]}"
            job_config["output_dir"] = str(output_root / run_id)
            jobs.append(
                {
                    "job_id": run_id,
                    "arm_id": arm_id,
                    "role": role,
                    "seed": seed,
                    "run_id": run_id,
                    "output_dir": job_config["output_dir"],
                    "config_sha256": _sha256(_canonical_json(job_config)),
                    "resolved_config": job_config,
                    "planned_optimizer_steps": steps,
                    "planned_tokens": tokens,
                    "estimated_flops": compute["estimated_flops"],
                }
            )

    total_flops = sum(job["estimated_flops"] for job in jobs)
    if total_flops > cap:
        raise ValueError(
            f"planned compute {total_flops} FLOPs exceeds declared total cap {cap} FLOPs"
        )
    candidate_compute_deltas = {
        arm_id: {
            "relative_flop_delta_percent": 100.0
            * (arm_compute[arm_id] - arm_compute[baseline_id])
            / arm_compute[baseline_id],
            "within_one_percent": abs(arm_compute[arm_id] - arm_compute[baseline_id])
            / arm_compute[baseline_id]
            <= 0.01,
        }
        for arm_id in arm_ids[1:]
    }
    plan: dict[str, Any] = {
        "schema_version": 1,
        "plan_type": "frontier-sweep-plan-v1",
        "name": name,
        "source_spec_name": source_path.name,
        "source_spec_sha256": _sha256(raw_spec),
        "seeds": seeds,
        "seed_pairing": "same seed is paired across baseline and every candidate arm",
        "output_root": str(output_root),
        "total_compute_cap_flops": cap,
        "total_estimated_flops": total_flops,
        "candidate_compute_deltas_vs_baseline": candidate_compute_deltas,
        "hardware": _hardware_record(),
        "arms": resolved_arms,
        "jobs": jobs,
    }
    plan["plan_sha256"] = _sha256(_canonical_json(plan))
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable sweep plan: {destination}")
    write_json(destination, plan)
    return plan


def verify_sweep_plan(plan: dict[str, Any]) -> bool:
    """Check the content hash before an executor consumes a saved plan."""
    digest = plan.get("plan_sha256")
    if not isinstance(digest, str):
        return False
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return digest == _sha256(_canonical_json(payload))
