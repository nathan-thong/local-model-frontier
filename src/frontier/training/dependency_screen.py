"""Capped training loop for preregistered synthetic dependency screens."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from frontier.config import ModelConfig
from frontier.data.dependency import DependencyDataset
from frontier.models import DecoderLanguageModel
from frontier.training.runtime import seed_everything


@dataclass
class DependencyTrainingResult:
    model: DecoderLanguageModel
    seed: int
    status: str
    updates_completed: int
    updates_attempted: int
    tokens_seen: int
    all_losses_and_gradient_norms_finite: bool
    initial_state_sha256: str
    elapsed_seconds: float
    reason: str | None = None


def state_dict_sha256(model: torch.nn.Module) -> str:
    """Hash tensor names, shapes, dtypes, and canonical CPU tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _total_grad_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    squared_norms = [
        parameter.grad.detach().to(dtype=torch.float64).pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared_norms:
        return torch.tensor(0.0, dtype=torch.float64)
    return torch.stack(squared_norms).sum().sqrt()


def train_dependency_seed(
    config: dict[str, Any],
    dataset: DependencyDataset,
    seed: int,
    *,
    stop_requested: Callable[[], bool],
    record_event: Callable[[dict[str, Any]], None],
) -> DependencyTrainingResult:
    """Train one fixed pass and stop only at preregistered update boundaries."""
    training = config["training"]
    updates = training["updates_per_seed"]
    batch_size = training["batch_size"]
    input_length = config["task"]["input_length"]
    if len(dataset.targets) != updates * batch_size:
        raise ValueError("dataset size must equal the fixed one-pass update budget")
    if dataset.input_ids.shape != (len(dataset.targets), input_length):
        raise ValueError("dependency dataset shape differs from the frozen task contract")

    previous_num_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(training["torch_num_threads"])
    seed_everything(seed, training["deterministic"])
    model_config = ModelConfig(**config["model"])
    model = DecoderLanguageModel(model_config).to(device="cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        betas=tuple(training["adamw_betas"]),
        eps=training["adamw_epsilon"],
        weight_decay=training["weight_decay"],
        foreach=training["adamw_foreach"],
        fused=training["adamw_fused"],
    )
    initial_hash = state_dict_sha256(model)
    started = time.perf_counter()
    updates_completed = 0
    updates_attempted = 0
    all_finite = True
    status = "completed"
    reason = None

    for update_index in range(updates):
        if stop_requested():
            status = "incomplete"
            reason = "cumulative wall-clock cap reached before the next optimizer update"
            record_event(
                {
                    "event": "stop",
                    "seed": seed,
                    "next_update": update_index + 1,
                    "reason": reason,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            break

        start = update_index * batch_size
        end = start + batch_size
        input_ids = dataset.input_ids[start:end]
        targets = dataset.targets[start:end]
        updates_attempted += 1
        record_event(
            {
                "event": "batch_start",
                "seed": seed,
                "update": update_index + 1,
                "tokens_seen": updates_attempted * batch_size * input_length,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(input_ids, use_cache=False)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        loss_finite = bool(torch.isfinite(loss).item())
        if not loss_finite:
            all_finite = False
            status = "failed"
            reason = f"non-finite loss before optimizer update {update_index + 1}"
            record_event(
                {
                    "event": "failure",
                    "seed": seed,
                    "update": update_index + 1,
                    "loss": None,
                    "tokens_seen": updates_attempted * batch_size * input_length,
                    "loss_finite": False,
                    "reason": reason,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            break

        loss.backward()
        parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
        gradients_finite = all(
            torch.isfinite(parameter.grad).all().item() for parameter in parameters
        )
        pre_clip_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=training["gradient_clip_norm"],
            norm_type=training["gradient_norm_type"],
            error_if_nonfinite=False,
        )
        pre_clip_norm = float(pre_clip_tensor.item())
        post_clip_norm = float(_total_grad_norm(parameters).item())
        norm_finite = math.isfinite(pre_clip_norm) and math.isfinite(post_clip_norm)
        if not gradients_finite or not norm_finite:
            all_finite = False
            status = "failed"
            reason = f"non-finite gradient before optimizer update {update_index + 1}"
            record_event(
                {
                    "event": "failure",
                    "seed": seed,
                    "update": update_index + 1,
                    "loss": float(loss.detach().item()),
                    "pre_clip_gradient_norm": (
                        pre_clip_norm if math.isfinite(pre_clip_norm) else None
                    ),
                    "post_clip_gradient_norm": (
                        post_clip_norm if math.isfinite(post_clip_norm) else None
                    ),
                    "tokens_seen": updates_attempted * batch_size * input_length,
                    "loss_finite": loss_finite,
                    "gradients_finite": gradients_finite,
                    "gradient_norms_finite": norm_finite,
                    "reason": reason,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            break

        optimizer.step()
        updates_completed += 1
        parameters_finite = all(
            torch.isfinite(parameter).all().item() for parameter in model.parameters()
        )
        elapsed = time.perf_counter() - started
        event = {
            "event": "update",
            "seed": seed,
            "update": update_index + 1,
            "tokens_seen": updates_attempted * batch_size * input_length,
            "loss": float(loss.detach().item()),
            "pre_clip_gradient_norm": pre_clip_norm,
            "post_clip_gradient_norm": post_clip_norm,
            "loss_finite": loss_finite,
            "gradients_finite": gradients_finite,
            "gradient_norms_finite": norm_finite,
            "parameters_finite_after_update": parameters_finite,
            "learning_rate": training["learning_rate"],
            "elapsed_seconds": elapsed,
        }
        record_event(event)
        if not parameters_finite:
            all_finite = False
            status = "failed"
            reason = f"non-finite model parameter after optimizer update {update_index + 1}"
            record_event(
                {
                    "event": "failure",
                    "seed": seed,
                    "update": update_index + 1,
                    "reason": reason,
                    "elapsed_seconds": elapsed,
                }
            )
            break

    if status == "completed" and updates_completed != updates:
        status = "incomplete"
        reason = "fixed update budget was not completed"
    torch.set_num_threads(previous_num_threads)
    torch.use_deterministic_algorithms(previous_determinism)
    return DependencyTrainingResult(
        model=model,
        seed=seed,
        status=status,
        updates_completed=updates_completed,
        updates_attempted=updates_attempted,
        tokens_seen=updates_attempted * batch_size * input_length,
        all_losses_and_gradient_norms_finite=all_finite,
        initial_state_sha256=initial_hash,
        elapsed_seconds=time.perf_counter() - started,
        reason=reason,
    )
