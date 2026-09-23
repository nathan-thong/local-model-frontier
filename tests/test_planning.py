import json
from pathlib import Path

import pytest

from frontier.cli import main
from frontier.config import RunConfig
from frontier.experiments.planning import plan_sweep, verify_sweep_plan
from frontier.experiments.runner import execute_sweep
from frontier.training.trainer import train


def _write_config(path, *, width=16, max_steps=2):
    config = {
        "seed": 17,
        "output_dir": "unused",
        "model": {
            "vocab_size": 259,
            "max_seq_len": 8,
            "width": width,
            "layers": 1,
            "query_heads": 2,
            "kv_heads": 2,
            "ffn_width": 32,
        },
        "train": {
            "batch_size": 1,
            "context_length": 8,
            "max_steps": max_steps,
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
        "evaluation": {"stride": 4, "max_validation_documents": 1},
        "profiling": {"prompt_lengths": [2], "decode_tokens": 2, "warmup_steps": 0, "repeats": 1},
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def _write_spec(tmp_path, *, cap=10**12):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write_config(tmp_path / "base.json")
    _write_config(tmp_path / "candidate.json", width=16, max_steps=2)
    spec = {
        "schema_version": 1,
        "name": "unit-sweep",
        "baseline": {"id": "control", "config": "base.json"},
        "candidates": [{"id": "candidate", "config": "candidate.json"}],
        "seeds": [17, 29],
        "output_root": "runs",
        "total_compute_cap_flops": cap,
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_plan_resolves_seed_pairs_budgets_and_hash_without_training(tmp_path):
    spec_path = _write_spec(tmp_path)
    output_path = tmp_path / "plan.json"

    plan = plan_sweep(spec_path, output_path)

    assert [job["seed"] for job in plan["jobs"]] == [17, 29, 17, 29]
    assert [job["arm_id"] for job in plan["jobs"]] == [
        "control",
        "control",
        "candidate",
        "candidate",
    ]
    assert all(job["planned_optimizer_steps"] == 2 for job in plan["jobs"])
    assert all(job["planned_tokens"] == 16 for job in plan["jobs"])
    assert plan["total_estimated_flops"] == sum(job["estimated_flops"] for job in plan["jobs"])
    assert plan["candidate_compute_deltas_vs_baseline"]["candidate"]["within_one_percent"]
    assert output_path.exists()
    assert verify_sweep_plan(json.loads(output_path.read_text(encoding="utf-8")))


def test_plan_rejects_compute_cap_before_writing(tmp_path):
    spec_path = _write_spec(tmp_path, cap=1)
    output_path = tmp_path / "plan.json"

    with pytest.raises(ValueError, match="exceeds declared total cap"):
        plan_sweep(spec_path, output_path)

    assert not output_path.exists()


def test_plan_does_not_overwrite_existing_plan(tmp_path):
    spec_path = _write_spec(tmp_path)
    output_path = tmp_path / "plan.json"
    plan_sweep(spec_path, output_path)

    with pytest.raises(FileExistsError, match="immutable sweep plan"):
        plan_sweep(spec_path, output_path)


def test_plan_hash_detects_edited_job_config(tmp_path):
    plan = plan_sweep(_write_spec(tmp_path), tmp_path / "plan.json")
    plan["jobs"][0]["resolved_config"]["seed"] = 99

    assert not verify_sweep_plan(plan)


def test_plan_rejects_duplicate_seeds_and_unsupported_compute(tmp_path):
    spec_path = _write_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["seeds"] = [17, 17]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct integer seeds"):
        plan_sweep(spec_path, tmp_path / "duplicate.json")

    spec["seeds"] = [17]
    spec["candidates"][0]["config"] = "unknown-sequence.json"
    _write_config(tmp_path / "unknown-sequence.json")
    config = json.loads((tmp_path / "unknown-sequence.json").read_text(encoding="utf-8"))
    config["model"]["sequence_types"] = ["unregistered"]
    (tmp_path / "unknown-sequence.json").write_text(json.dumps(config), encoding="utf-8")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="without a registered compute estimator"):
        plan_sweep(spec_path, tmp_path / "unsupported.json")


def test_plan_sweep_cli_writes_plan_and_prints_summary(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    output_path = tmp_path / "plan.json"

    assert main(["plan-sweep", "--spec", str(spec_path), "--output", str(output_path)]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert (
        printed["plan_sha256"] == json.loads(output_path.read_text(encoding="utf-8"))["plan_sha256"]
    )


def test_plan_resolves_relative_data_dir_like_the_training_cli(tmp_path):
    spec_path = _write_spec(tmp_path)
    config_path = tmp_path / "base.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data_dir"] = "data/tinystories"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan = plan_sweep(spec_path, tmp_path / "plan.json")

    assert plan["jobs"][0]["resolved_config"]["data_dir"] == str(
        (Path.cwd() / "data/tinystories").resolve()
    )


def _single_seed_plan(tmp_path, *, max_steps):
    spec_path = _write_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["seeds"] = [31]
    _write_config(tmp_path / "base.json", max_steps=max_steps)
    _write_config(tmp_path / "candidate.json", max_steps=max_steps)
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_sweep(spec_path, plan_path)
    return plan_path, json.loads(plan_path.read_text(encoding="utf-8"))


def test_execute_sweep_runs_jobs_and_validates_completed_rerun(tmp_path):
    plan_path, plan = _single_seed_plan(tmp_path, max_steps=1)

    status = execute_sweep(plan_path)
    repeated_status = execute_sweep(plan_path)

    assert status["status"] == "completed"
    assert all(job["status"] == "completed" for job in status["jobs"].values())
    assert repeated_status["status"] == "completed"
    assert repeated_status["completed_estimated_flops"] == plan["total_estimated_flops"]
    assert not plan_path.with_suffix(".json.lock").exists()


def test_execute_sweep_resumes_only_a_matching_running_checkpoint(tmp_path):
    plan_path, plan = _single_seed_plan(tmp_path, max_steps=2)

    paused = execute_sweep(plan_path, max_runtime_seconds=1e-12)
    first_job = plan["jobs"][0]
    assert paused["status"] == "paused"
    assert paused["jobs"][first_job["job_id"]]["status"] == "not_started"
    train(RunConfig.from_dict(first_job["resolved_config"]), stop_after_seconds=1e-12)
    partial_summary = json.loads(
        (Path(first_job["output_dir"]) / "summary.json").read_text(encoding="utf-8")
    )
    assert partial_summary["status"] == "running"
    assert partial_summary["training"]["stop_reason"] == "runtime_budget"

    status = execute_sweep(plan_path)

    assert status["status"] == "completed"
    assert status["jobs"][first_job["job_id"]]["status"] == "completed"
    assert status["jobs"][first_job["job_id"]]["attempt"] == 1


def test_execute_sweep_resumes_a_matching_external_interruption(tmp_path):
    plan_path, plan = _single_seed_plan(tmp_path, max_steps=2)
    first_job = plan["jobs"][0]
    train(RunConfig.from_dict(first_job["resolved_config"]), stop_after_steps=1)

    status = execute_sweep(plan_path)

    assert status["status"] == "completed"
    assert status["jobs"][first_job["job_id"]]["status"] == "completed"


def test_execute_sweep_refuses_tampered_plan_and_nonempty_unknown_run(tmp_path):
    plan_path, plan = _single_seed_plan(tmp_path, max_steps=1)
    plan["jobs"][0]["estimated_flops"] += 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash does not verify"):
        execute_sweep(plan_path)

    other_root = tmp_path / "other"
    plan_path, plan = _single_seed_plan(other_root, max_steps=1)
    run_dir = Path(plan["jobs"][0]["output_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "unexpected.txt").write_text("keep", encoding="utf-8")
    status = execute_sweep(plan_path)
    assert status["status"] == "failed"
    assert "non-empty run directory" in status["error"]
    assert (run_dir / "unexpected.txt").read_text(encoding="utf-8") == "keep"
