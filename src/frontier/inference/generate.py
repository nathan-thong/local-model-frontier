"""Greedy or sampled generation over the model's incremental cache interface."""

from __future__ import annotations

import torch

from frontier.models import DecoderLanguageModel


@torch.inference_mode()
def generate(
    model: DecoderLanguageModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [batch, non-empty time]")
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError("max_new_tokens and temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    retained_length = min(input_ids.shape[1], model.config.max_seq_len)
    if retained_length + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus requested generation exceeds model.max_seq_len")
    model_was_training = model.training
    model.eval()
    ids = input_ids[:, -model.config.max_seq_len :]
    logits, cache = model(ids, use_cache=True)
    generated = [ids]
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :].clone()
        next_logits[:, 256] = -torch.inf  # Do not generate BOS.
        next_logits[:, 258] = -torch.inf  # Do not generate PAD.
        if top_k is not None:
            keep = min(top_k, next_logits.shape[-1])
            cutoff = torch.topk(next_logits, keep, dim=-1).values[:, -1:]
            next_logits = next_logits.masked_fill(next_logits < cutoff, -torch.inf)
        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1, generator=generator)
        generated.append(next_token)
        if eos_id is not None and torch.all(next_token == eos_id):
            break
        logits, cache = model(next_token, cache=cache, use_cache=True)
    result = torch.cat(generated, dim=1)
    model.train(model_was_training)
    return result
