"""Profile a deterministic, random-initialized linear-attention reference model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from frontier.config import ModelConfig, ProfileConfig
from frontier.models import DecoderLanguageModel
from frontier.profiling.benchmark import profile_model


def _git_source_state(repo_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, not status.strip()


def profile_initialized_model(config_path: Path, output_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = config_path.resolve()
    if not config_path.is_relative_to(repo_root):
        raise ValueError("profile config must be within the repository")
    output_path = output_path.resolve()
    runs_root = (repo_root / "runs").resolve()
    if not output_path.is_relative_to(runs_root):
        raise ValueError("profile output must be within the ignored runs/ directory")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {output_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    seed = raw.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("profile seed must be an integer in [0, 2**63)")
    model_config = ModelConfig(**raw["model"])
    model_config.validate()
    settings = ProfileConfig(**raw["profiling"])
    revision, clean = _git_source_state(repo_root)
    if not clean:
        raise RuntimeError("initialized-model profile requires a clean Git worktree")

    torch.manual_seed(seed)
    model = DecoderLanguageModel(model_config).to(device="cpu").eval()
    measurements = profile_model(model, settings, input_seed=seed)
    record = {
        "schema_version": 1,
        "experiment_id": raw.get("experiment_id"),
        "source_revision": revision,
        "source_worktree_clean": clean,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "seed": seed,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_num_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu",
        },
        "profile_settings": raw["profiling"],
        "measurements": measurements,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = profile_initialized_model(args.config, args.output)
    print(
        json.dumps(
            {
                "status": "completed",
                "source_revision": record["source_revision"],
                "clean": record["source_worktree_clean"],
                "output": args.output.as_posix(),
                "workloads": len(record["measurements"]["workloads"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
