"""Document-aware, token-weighted causal language-model perplexity."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F


@torch.inference_mode()
def evaluate_perplexity(
    model: torch.nn.Module,
    documents: Sequence[torch.Tensor],
    context_length: int,
    stride: int,
    max_documents: int | None = None,
) -> dict[str, float | int]:
    if context_length <= 0 or stride <= 0 or stride > context_length:
        raise ValueError("require 0 < stride <= context_length")
    selected = list(documents[:max_documents] if max_documents is not None else documents)
    if not selected:
        raise ValueError("perplexity evaluation needs at least one document")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_nll = 0.0
    token_count = 0
    for document in selected:
        doc = document.to(device)
        if doc.numel() < 2:
            continue
        last_scored_target = 0
        length = int(doc.numel())
        while last_scored_target < length - 1:
            target_start = last_scored_target + 1
            target_end = min(length, target_start + stride)
            # Bound the model input itself to context_length while scoring this target stride.
            input_start = max(0, target_end - 1 - context_length)
            x = doc[input_start : target_end - 1].unsqueeze(0)
            y = doc[input_start + 1 : target_end]
            logits, _ = model(x)
            target_indices = torch.arange(input_start + 1, target_end, device=device)
            score = target_indices >= target_start
            pred_positions = target_indices[score] - 1 - input_start
            chosen_logits = logits[0, pred_positions].float()
            total_nll += float(F.cross_entropy(chosen_logits, y[score], reduction="sum").item())
            count = int(score.sum().item())
            token_count += count
            last_scored_target = target_end - 1
    model.train(was_training)
    if token_count == 0:
        raise ValueError("selected documents contain no next-token targets")
    mean_nll = total_nll / token_count
    return {
        "nll_sum": total_nll,
        "scored_tokens": token_count,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 709 else None,
        "perplexity_overflow": mean_nll >= 709,
    }
