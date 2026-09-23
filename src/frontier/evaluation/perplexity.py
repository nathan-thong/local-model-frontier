"""Document-aware, token-weighted causal language-model perplexity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F

BYTE_METRIC_PROTOCOL = {
    "name": "utf8-bits-per-byte-v1",
    "alignment": "tokenizer-provided byte counts aligned with encoded token IDs",
    "special_tokens": (
        "zero-byte targets are excluded from the byte-normalized numerator and denominator; "
        "all next-token targets remain in token perplexity"
    ),
}


def _perplexity(mean_nll: float | None) -> tuple[float | None, bool]:
    if mean_nll is None:
        return None, False
    overflow = mean_nll >= 709
    return (None if overflow else math.exp(mean_nll)), overflow


@torch.inference_mode()
def evaluate_perplexity(
    model: torch.nn.Module,
    documents: Sequence[torch.Tensor],
    context_length: int,
    stride: int,
    max_documents: int | None = None,
    token_byte_counts: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    if context_length <= 0 or stride <= 0 or stride > context_length:
        raise ValueError("require 0 < stride <= context_length")
    if max_documents is not None and (
        not isinstance(max_documents, int) or isinstance(max_documents, bool) or max_documents <= 0
    ):
        raise ValueError("max_documents must be a positive integer or None")
    selected = list(documents[:max_documents] if max_documents is not None else documents)
    if not selected:
        raise ValueError("perplexity evaluation needs at least one document")
    if token_byte_counts is not None and len(token_byte_counts) < len(selected):
        raise ValueError("token_byte_counts must contain one sequence for each selected document")

    device = next(model.parameters()).device
    was_training = model.training
    total_nll = 0.0
    content_nll_sum = 0.0
    token_count = 0
    byte_count = 0
    document_metrics = []
    model.eval()
    try:
        for index, document in enumerate(selected):
            if document.ndim != 1:
                raise ValueError(f"document {index} must be a one-dimensional token tensor")
            doc = document.to(device)
            aligned_bytes = None
            if token_byte_counts is not None:
                raw_byte_counts = token_byte_counts[index]
                if len(raw_byte_counts) != doc.numel() or any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in raw_byte_counts
                ):
                    raise ValueError(
                        f"token byte counts for document {index} must be non-negative integers "
                        "aligned to every token"
                    )
                aligned_bytes = torch.tensor(raw_byte_counts, dtype=torch.long, device=device)

            document_nll = 0.0
            document_content_nll = 0.0
            document_tokens = 0
            document_bytes = 0
            length = int(doc.numel())
            last_scored_target = 0
            while last_scored_target < length - 1:
                target_start = last_scored_target + 1
                target_end = min(length, target_start + stride)
                # Bound the model input itself while scoring each target stride once.
                input_start = max(0, target_end - 1 - context_length)
                x = doc[input_start : target_end - 1].unsqueeze(0)
                y = doc[input_start + 1 : target_end]
                logits, _ = model(x)
                target_indices = torch.arange(input_start + 1, target_end, device=device)
                score = target_indices >= target_start
                prediction_positions = target_indices[score] - 1 - input_start
                chosen_logits = logits[0, prediction_positions].float()
                token_nll = F.cross_entropy(chosen_logits, y[score], reduction="none")
                if not torch.isfinite(token_nll).all().item():
                    raise FloatingPointError(
                        f"non-finite next-token NLL in document {index} at target {target_start}"
                    )
                window_nll = float(token_nll.sum().item())
                if not math.isfinite(window_nll):
                    raise FloatingPointError(
                        f"non-finite NLL sum in document {index} at target {target_start}"
                    )
                document_nll += window_nll
                total_nll += window_nll
                window_tokens = int(token_nll.numel())
                document_tokens += window_tokens
                token_count += window_tokens
                if aligned_bytes is not None:
                    window_bytes = aligned_bytes[target_indices[score]]
                    byte_mask = window_bytes > 0
                    document_bytes += int(window_bytes[byte_mask].sum().item())
                    byte_count += int(window_bytes[byte_mask].sum().item())
                    if byte_mask.any().item():
                        byte_nll = float(token_nll[byte_mask].sum().item())
                        document_content_nll += byte_nll
                        content_nll_sum += byte_nll
                last_scored_target = target_end - 1

            document_mean_nll = document_nll / document_tokens if document_tokens else None
            document_perplexity, document_overflow = _perplexity(document_mean_nll)
            document_bpb = (
                document_content_nll / math.log(2) / document_bytes if document_bytes > 0 else None
            )
            document_metrics.append(
                {
                    "document_index": index,
                    "scored_tokens": document_tokens,
                    "nll_sum": document_nll,
                    "mean_nll": document_mean_nll,
                    "perplexity": document_perplexity,
                    "perplexity_overflow": document_overflow,
                    "scored_utf8_bytes": document_bytes if aligned_bytes is not None else None,
                    "content_nll_sum": (
                        document_content_nll if aligned_bytes is not None else None
                    ),
                    "bits_per_byte": document_bpb,
                    "byte_metric_status": (
                        "not_available_no_token_byte_alignment"
                        if aligned_bytes is None
                        else "available"
                        if document_bytes > 0
                        else "no_positive_byte_targets"
                    ),
                    "status": "scored" if document_tokens else "no_next_token_targets",
                }
            )
    finally:
        model.train(was_training)

    if token_count == 0:
        raise ValueError("selected documents contain no next-token targets")
    mean_nll = total_nll / token_count
    perplexity, overflow = _perplexity(mean_nll)
    return {
        "schema_version": 2,
        "nll_sum": total_nll,
        "scored_tokens": token_count,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "perplexity_overflow": overflow,
        "per_document": document_metrics,
        "scored_utf8_bytes": byte_count if token_byte_counts is not None else None,
        "content_nll_sum": content_nll_sum if token_byte_counts is not None else None,
        "bits_per_byte": (
            content_nll_sum / math.log(2) / byte_count
            if token_byte_counts is not None and byte_count > 0
            else None
        ),
        "byte_metric_status": (
            "not_available_no_token_byte_alignment"
            if token_byte_counts is None
            else "available"
            if byte_count > 0
            else "no_positive_byte_targets"
        ),
        "byte_metric_protocol": BYTE_METRIC_PROTOCOL if token_byte_counts is not None else None,
    }
