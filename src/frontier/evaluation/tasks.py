"""Generic downstream scoring for versioned JSONL prompt/target examples."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from frontier.inference import generate
from frontier.models import DecoderLanguageModel
from frontier.tokenization import Tokenizer


def read_tasks(path: str | Path) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    identifiers: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("prompt"), str)
                or not isinstance(row.get("target"), str)
            ):
                raise TypeError(
                    f"task line {line_number} needs string fields 'prompt' and 'target'"
                )
            example_id = row.get("id", f"line-{line_number}")
            if not isinstance(example_id, str) or not example_id:
                raise TypeError(f"task line {line_number} id must be a non-empty string")
            if example_id in identifiers:
                raise ValueError(f"duplicate task example id {example_id!r}")
            identifiers.add(example_id)
            tasks.append({"id": example_id, "prompt": row["prompt"], "target": row["target"]})
    if not tasks:
        raise ValueError(f"no task examples in {path}")
    return tasks


@torch.inference_mode()
def evaluate_tasks(
    model: DecoderLanguageModel,
    tokenizer: Tokenizer,
    path: str | Path,
    max_new_tokens: int,
) -> dict:
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    rows = read_tasks(path)
    task_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    device = next(model.parameters()).device
    details = []
    exact = 0
    nll_sum = 0.0
    scored_tokens = 0
    was_training = model.training
    model.eval()
    try:
        for row in rows:
            prompt_ids = tokenizer.encode(row["prompt"], add_bos=True, add_eos=False)
            target_ids = tokenizer.encode(row["target"], add_bos=False, add_eos=True)
            prompt = torch.tensor([prompt_ids], device=device)
            full = torch.tensor(prompt_ids + target_ids, device=device)
            target_position = len(prompt_ids)
            stride = max(1, model.config.max_seq_len // 2)
            example_nll = 0.0
            example_tokens = 0
            while target_position < full.numel():
                target_end = min(int(full.numel()), target_position + stride)
                input_start = max(0, target_end - 1 - model.config.max_seq_len)
                input_ids = full[input_start : target_end - 1].unsqueeze(0)
                labels = full[input_start + 1 : target_end]
                absolute_target_positions = torch.arange(input_start + 1, target_end, device=device)
                score_mask = absolute_target_positions >= target_position
                prediction_positions = absolute_target_positions[score_mask] - 1 - input_start
                selected_logits = model(input_ids)[0][0, prediction_positions].float()
                selected_labels = labels[score_mask]
                token_nll = F.cross_entropy(selected_logits, selected_labels, reduction="none")
                if not torch.isfinite(token_nll).all().item():
                    raise FloatingPointError(f"non-finite task NLL for example {row['id']!r}")
                window_nll = float(token_nll.sum().item())
                if not math.isfinite(window_nll):
                    raise FloatingPointError(f"non-finite task NLL sum for example {row['id']!r}")
                example_nll += window_nll
                example_tokens += int(token_nll.numel())
                target_position = target_end
            retained_prompt = min(len(prompt_ids), model.config.max_seq_len)
            if max_new_tokens > model.config.max_seq_len - retained_prompt:
                raise ValueError(
                    f"task prompt leaves fewer than {max_new_tokens} positions for generation: {row['prompt']!r}"
                )
            output = generate(model, prompt, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_id)
            prediction = tokenizer.decode(output[0, retained_prompt:].tolist()).strip()
            expected = row["target"].strip()
            matched = prediction == expected
            exact += int(matched)
            nll_sum += example_nll
            scored_tokens += example_tokens
            example_mean_nll = example_nll / example_tokens if example_tokens else None
            example_overflow = example_mean_nll is not None and example_mean_nll >= 709
            details.append(
                {
                    "example_id": row["id"],
                    "prompt": row["prompt"],
                    "target": row["target"],
                    "prediction": prediction,
                    "exact_match": matched,
                    "conditional_nll_sum": example_nll,
                    "conditional_scored_tokens": example_tokens,
                    "conditional_mean_nll": example_mean_nll,
                    "conditional_perplexity": (
                        math.exp(example_mean_nll)
                        if example_mean_nll is not None and not example_overflow
                        else None
                    ),
                    "conditional_perplexity_overflow": example_overflow,
                }
            )
    finally:
        model.train(was_training)
    return {
        "examples": len(rows),
        "exact_match": exact / len(rows),
        "conditional_nll_sum": nll_sum,
        "conditional_scored_tokens": scored_tokens,
        "conditional_mean_nll": nll_sum / scored_tokens if scored_tokens else None,
        "details": details,
        "protocol": {
            "adapter": "jsonl-prompt-target-v1",
            "artifact_sha256": task_hash,
            "scoring": "strip exact match; teacher-forced target token NLL; target includes EOS",
            "generation": "greedy; max_new_tokens as configured; stop at EOS",
        },
    }
