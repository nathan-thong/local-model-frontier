"""Scorers for small synthetic key/value dependency tasks."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch.nn import functional as F

from frontier.data.dependency import DependencyDataset
from frontier.models.sequence.interface import DecoderCache


def _state_norms(
    cache: DecoderCache | None, batch_size: int
) -> list[list[dict[str, float | None]]]:
    if cache is None:
        return [[] for _ in range(batch_size)]
    per_example: list[list[dict[str, float | None]]] = [[] for _ in range(batch_size)]
    for layer_index, state in enumerate(cache.states):
        if state is None:
            raise ValueError(
                f"cached dependency evaluation returned no state for layer {layer_index}"
            )
        key_value_sum = getattr(state, "key_value_sum", None)
        key_feature_sum = getattr(state, "key_feature_sum", None)
        if not isinstance(key_value_sum, torch.Tensor) or not isinstance(
            key_feature_sum, torch.Tensor
        ):
            raise TypeError(f"dependency screen expected recurrent state at layer {layer_index}")
        kv_dims = tuple(range(1, key_value_sum.ndim))
        key_dims = tuple(range(1, key_feature_sum.ndim))
        kv_norm = torch.linalg.vector_norm(key_value_sum, dim=kv_dims)
        key_norm = torch.linalg.vector_norm(key_feature_sum, dim=key_dims)
        for index in range(batch_size):
            per_example[index].append(
                {
                    "layer": layer_index,
                    "key_value_sum_l2": _finite_or_none(float(kv_norm[index].item())),
                    "key_feature_sum_l2": _finite_or_none(float(key_norm[index].item())),
                }
            )
    return per_example


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def evaluate_dependency_recall(
    model: torch.nn.Module,
    dataset: DependencyDataset,
    *,
    value_token_ids: list[int],
    joint_strata: list[dict[str, int]],
    batch_size: int,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Score final-position predictions and retain exact per-example records."""
    if batch_size <= 0 or len(value_token_ids) < 2:
        raise ValueError("batch_size must be positive and at least two value IDs are required")
    stratum_by_slot = {item["target_slot"]: item for item in joint_strata}
    if set(stratum_by_slot) != set(range(len(joint_strata))):
        raise ValueError("joint strata must define one entry per target slot")
    prior_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    completed = True
    try:
        with torch.no_grad():
            for start in range(0, len(dataset.targets), batch_size):
                if stop_requested is not None and stop_requested():
                    completed = False
                    break
                end = min(start + batch_size, len(dataset.targets))
                input_ids = dataset.input_ids[start:end]
                targets = dataset.targets[start:end]
                logits, cache = model(input_ids, use_cache=True)
                final_logits = logits[:, -1, :]
                losses = F.cross_entropy(final_logits, targets, reduction="none")
                full_predictions = final_logits.argmax(dim=-1)
                value_logits = final_logits[:, value_token_ids]
                value_ids = torch.as_tensor(value_token_ids, device=final_logits.device)
                conditional_predictions = value_ids[value_logits.argmax(dim=-1)]
                state_norm_records = _state_norms(cache, end - start)
                for local_index, global_index in enumerate(range(start, end)):
                    slot = int(dataset.target_slots[global_index].item())
                    stratum = stratum_by_slot[slot]
                    target = int(targets[local_index].item())
                    nll = float(losses[local_index].item())
                    full_prediction = int(full_predictions[local_index].item())
                    conditional_prediction = int(conditional_predictions[local_index].item())
                    norms = state_norm_records[local_index]
                    logits_finite = bool(torch.isfinite(final_logits[local_index]).all().item())
                    nll_finite = bool(torch.isfinite(losses[local_index]).item())
                    norms_finite = all(
                        value is not None and math.isfinite(value)
                        for layer in norms
                        for value in layer.values()
                    )
                    finite = logits_finite and nll_finite and norms_finite
                    nll = nll if nll_finite else None
                    full_prediction = full_prediction if logits_finite else None
                    conditional_prediction = conditional_prediction if logits_finite else None
                    rows.append(
                        {
                            "example_index": global_index,
                            "input_identity_sha256": dataset.identity_hashes[global_index],
                            "input_token_ids": [
                                int(token) for token in input_ids[local_index].tolist()
                            ],
                            "target_token_id": target,
                            "target_slot": slot,
                            "later_distractor_pairs": stratum["later_distractor_pairs"],
                            "value_to_query_key_token_distance": stratum[
                                "value_to_query_key_token_distance"
                            ],
                            "target_nll_nats": nll,
                            "full_vocabulary_prediction": full_prediction,
                            "full_vocabulary_correct": (
                                full_prediction == target if logits_finite else None
                            ),
                            "conditional_value_prediction": conditional_prediction,
                            "conditional_value_correct": (
                                conditional_prediction == target if logits_finite else None
                            ),
                            "state_norms_by_layer": norms,
                            "all_scored_values_finite": bool(finite),
                        }
                    )
    finally:
        model.train(prior_training)

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        losses = [row["target_nll_nats"] for row in selected]
        full_correct = [row["full_vocabulary_correct"] for row in selected]
        conditional_correct = [row["conditional_value_correct"] for row in selected]
        return {
            "examples": count,
            "mean_target_nll_nats": (
                sum(losses) / count
                if count and all(value is not None for value in losses)
                else None
            ),
            "full_vocabulary_exact_match": (
                sum(full_correct) / count
                if count and all(value is not None for value in full_correct)
                else None
            ),
            "conditional_value_top1": (
                sum(conditional_correct) / count
                if count and all(value is not None for value in conditional_correct)
                else None
            ),
            "all_scored_values_finite": all(row["all_scored_values_finite"] for row in selected),
        }

    strata = []
    for slot in sorted(stratum_by_slot):
        stratum = stratum_by_slot[slot]
        selected = [row for row in rows if row["target_slot"] == slot]
        strata.append(
            {
                **stratum,
                **aggregate(selected),
            }
        )
    return {
        "complete": completed and len(rows) == len(dataset.targets),
        "scored_examples": len(rows),
        "metrics": aggregate(rows),
        "joint_strata": strata,
        "per_example": rows,
    }
