"""Single-device mixed-precision trainer for baseline experiments."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from frontier.config import RunConfig
from frontier.data.corpus import (
    encode_documents,
    fixture_documents,
    load_split,
    sample_batch,
    write_fixture,
)
from frontier.evaluation.perplexity import evaluate_perplexity
from frontier.experiments.results import append_jsonl, write_json, write_summary
from frontier.models import DecoderLanguageModel
from frontier.profiling.memory import accelerator_memory, process_memory_bytes
from frontier.tokenization import build_tokenizer
from frontier.training.checkpoint import load_checkpoint, save_checkpoint
from frontier.training.runtime import (
    capture_rng_state,
    resolve_device,
    resolve_precision,
    restore_rng_state,
    seed_everything,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _environment(device: torch.device) -> dict:
    git_revision = None
    git_worktree_clean = None
    try:
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=git_root,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=git_root,
        ).stdout
        git_worktree_clean = not git_status.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy_version,
        "frontier": "0.1.0",
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "git_revision": git_revision,
        "git_worktree_clean": git_worktree_clean,
    }


def estimate_training_flops_v1(
    parameter_count: int, tokens: int, layers: int, width: int, context_length: int
) -> int:
    """Approximate dense decoder work; assumptions and omissions are in the protocol doc."""
    per_token = 6 * parameter_count + 12 * layers * width * context_length
    return int(per_token * tokens)


def _make_optimizer(
    model: DecoderLanguageModel, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


def _write_jsonl_metrics(path: Path, row: dict) -> None:
    append_jsonl(path, {"schema_version": 1, **row})


def train(config: RunConfig, resume: bool = False, stop_after_steps: int | None = None) -> Path:
    config.validate()
    resume = resume or config.train.resume
    run_dir = Path(config.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    config_path = run_dir / "resolved_config.json"
    if resume:
        if not checkpoint_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                "--resume requires a prior run with resolved_config.json and checkpoints/last.pt"
            )
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != config.to_dict():
            raise ValueError(
                "resume config differs from resolved_config.json; use the original config unchanged"
            )
    elif any(run_dir.iterdir()):
        raise FileExistsError(
            f"run directory is not empty: {run_dir}; choose a new output_dir or use --resume"
        )

    if config.data_dir is None:
        data_root = run_dir / "data"
        if not (data_root / "train.txt").exists():
            manifest = write_fixture(data_root)
            train_docs, valid_docs = fixture_documents()
        else:
            train_docs, valid_docs, manifest = load_split(data_root)
    else:
        data_root = Path(config.data_dir).resolve()
        train_docs, valid_docs, manifest = load_split(data_root)
    tokenizer = build_tokenizer(config.tokenizer)
    train_tokens = encode_documents(train_docs, tokenizer)
    valid_tokens = encode_documents(valid_docs, tokenizer)
    if config.model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"configured vocab_size {config.model.vocab_size} differs from {tokenizer.name} size {tokenizer.vocab_size}"
        )

    device = resolve_device()
    amp_dtype, amp_enabled = resolve_precision(config.train.mixed_precision, device)
    environment = _environment(device)
    seed_everything(config.seed, config.train.deterministic)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    model = DecoderLanguageModel(config.model).to(device)
    optimizer = _make_optimizer(model, config.train.learning_rate, config.train.weight_decay)
    tokens_per_step = (
        config.train.batch_size * config.train.context_length * config.train.gradient_accumulation
    )
    planned_steps = config.train.max_steps or math.ceil(config.train.max_tokens / tokens_per_step)

    def lr_lambda(step: int) -> float:
        warmup = config.train.warmup_steps
        floor = config.train.min_learning_rate / config.train.learning_rate
        if warmup and step < warmup:
            return max(step, 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, planned_steps - warmup)))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    start_step = 0
    tokens_seen = 0
    optimizer_updates = 0
    prior_active_seconds = 0.0
    if resume:
        state = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        tokens_seen = state["tokens_seen"]
        optimizer_updates = state.get("optimizer_updates", start_step)
        prior_active_seconds = state.get("optimizer_active_time_seconds", 0.0)
        restore_rng_state(state["rng"], generator)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("data", {}).get("train_sha256") != manifest.get(
            "train_sha256"
        ) or summary.get("data", {}).get("validation_sha256") != manifest.get("validation_sha256"):
            raise ValueError("resume data hashes differ from the original run")
        saved_environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        environment_keys = (
            "python",
            "platform",
            "torch",
            "numpy",
            "cuda_runtime",
            "device",
            "device_name",
        )
        if any(saved_environment.get(key) != environment.get(key) for key in environment_keys):
            raise ValueError(
                "resume environment differs from the original run; exact resume requires the same environment"
            )
    else:
        write_json(config_path, config.to_dict())
        write_json(
            run_dir / "tokenizer.json",
            {
                "name": tokenizer.name,
                "vocab_size": tokenizer.vocab_size,
                "bos_id": tokenizer.bos_id,
                "eos_id": tokenizer.eos_id,
                "pad_id": tokenizer.pad_id,
                "implementation": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            },
        )
        write_json(run_dir / "environment.json", environment)
        write_json(run_dir / "data_manifest.json", manifest)
        summary = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "status": "running",
            "resolved_config": config.to_dict(),
            "tokenizer": {"name": tokenizer.name, "vocab_size": tokenizer.vocab_size},
            "environment": environment,
            "data": {
                "root": str(data_root),
                "source": manifest.get(
                    "source", manifest.get("source_path_name", "pre-split corpus")
                ),
                "synthetic_fixture": manifest.get("source")
                == "frontier deterministic smoke fixture v1",
                "train_sha256": manifest.get("train_sha256"),
                "validation_sha256": manifest.get("validation_sha256"),
                "train_documents": len(train_docs),
                "validation_documents": len(valid_docs),
            },
            "training": {},
            "evaluation": {},
            "profiling": {},
        }
        write_summary(run_dir, summary)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    train_started = time.perf_counter()
    active_seconds = 0.0
    target_step = planned_steps
    if config.train.max_steps is not None:
        target_step = config.train.max_steps
    if stop_after_steps is not None:
        if stop_after_steps <= 0:
            raise ValueError("stop_after_steps must be positive")
        invocation_target = min(target_step, start_step + stop_after_steps)
    else:
        invocation_target = target_step
    metrics_path = run_dir / "metrics.jsonl"
    for step in range(start_step + 1, invocation_target + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=device)
        step_start = time.perf_counter()
        for _ in range(config.train.gradient_accumulation):
            x, y = sample_batch(
                train_tokens,
                config.train.batch_size,
                config.train.context_length,
                generator,
                device,
            )
            autocast = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
                if amp_enabled
                else nullcontext()
            )
            with autocast:
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"non-finite training loss at optimizer step {step}")
            loss_total += loss.detach() / config.train.gradient_accumulation
            scaler.scale(loss / config.train.gradient_accumulation).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_applied = scaler.get_scale() >= scale_before_step
        if optimizer_step_applied:
            scheduler.step()
            optimizer_updates += 1
        tokens_seen += tokens_per_step
        _sync(device)
        elapsed = time.perf_counter() - step_start
        active_seconds += elapsed
        loss_value = float(loss_total.item())
        grad_norm_value = float(grad_norm.item())
        row = {
            "event": "train_step",
            "step": step,
            "tokens_seen": tokens_seen,
            "train_loss": loss_value,
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": grad_norm_value if math.isfinite(grad_norm_value) else None,
            "gradient_norm_finite": math.isfinite(grad_norm_value),
            "optimizer_step_applied": optimizer_step_applied,
            "step_seconds": elapsed,
            "tokens_per_second": tokens_per_step / elapsed,
        }
        _write_jsonl_metrics(metrics_path, row)
        if step % config.train.eval_interval == 0 or step == invocation_target:
            val = evaluate_perplexity(
                model,
                valid_tokens,
                context_length=config.train.context_length,
                stride=min(config.evaluation.stride, config.train.context_length),
                max_documents=config.evaluation.max_validation_documents,
            )
            _write_jsonl_metrics(metrics_path, {"event": "validation", "step": step, **val})
            summary["evaluation"] = {
                "perplexity": val,
                "protocol": {
                    "dataset_sha256": manifest.get("validation_sha256"),
                    "tokenizer": tokenizer.name,
                    "context_length": config.train.context_length,
                    "stride": min(config.evaluation.stride, config.train.context_length),
                    "document_boundary": "score within documents; no cross-document targets",
                },
            }
        if step % config.train.checkpoint_interval == 0 or step == invocation_target:
            state = {
                "schema_version": 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": capture_rng_state(generator),
                "step": step,
                "tokens_seen": tokens_seen,
                "optimizer_updates": optimizer_updates,
                "optimizer_active_time_seconds": prior_active_seconds + active_seconds,
                "config": config.to_dict(),
            }
            save_checkpoint(checkpoint_path, state)
        supported_estimator = all(name == "attention" for name in config.model.sequence_types)
        estimated_flops = (
            estimate_training_flops_v1(
                model.parameter_count(),
                tokens_seen,
                config.model.layers,
                config.model.width,
                config.train.context_length,
            )
            if supported_estimator
            else None
        )
        summary["training"] = {
            "step": step,
            "optimizer_updates": optimizer_updates,
            "tokens_seen": tokens_seen,
            "tokens_per_step": tokens_per_step,
            "optimizer_active_time_seconds": prior_active_seconds + active_seconds,
            "tokens_per_second": tokens_seen / (prior_active_seconds + active_seconds)
            if prior_active_seconds + active_seconds
            else None,
            "estimated_flops": estimated_flops,
            "compute_estimator": "decoder-dense-v1" if supported_estimator else None,
            "compute_estimator_assumptions": (
                "approximate; see docs/measurement_protocol.md" if supported_estimator else None
            ),
        }
        summary["status"] = "completed" if step == target_step else "running"
        write_summary(run_dir, summary)

    wall_time = time.perf_counter() - train_started
    summary["training"].update(
        {
            "wall_time_seconds_this_invocation": wall_time,
            "optimizer_active_time_seconds": prior_active_seconds + active_seconds,
            "optimizer_active_time_seconds_this_invocation": active_seconds,
            "tokens_per_second": tokens_seen / (prior_active_seconds + active_seconds)
            if prior_active_seconds + active_seconds
            else None,
            "precision_requested": config.train.mixed_precision,
            "autocast_dtype": str(amp_dtype),
            "deterministic_algorithms": config.train.deterministic,
            "process_memory_after_training": process_memory_bytes(),
            "accelerator_memory_after_training": accelerator_memory(device),
        }
    )
    summary["status"] = (
        "completed" if start_step >= target_step or invocation_target >= target_step else "running"
    )
    save_checkpoint(
        run_dir / "weights.pt",
        {"schema_version": 1, "model": model.state_dict(), "model_config": config.model.__dict__},
    )
    write_summary(run_dir, summary)
    return run_dir
