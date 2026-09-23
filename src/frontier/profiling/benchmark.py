"""Fixed-shape prefill and incremental decoding profile."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator
from pathlib import Path

import torch

from frontier.config import ProfileConfig
from frontier.models import DecoderLanguageModel
from frontier.profiling.memory import (
    accelerator_memory,
    process_memory_bytes,
    unique_tensor_storage_bytes,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_tensors(state: object) -> Iterator[torch.Tensor]:
    if isinstance(state, torch.Tensor):
        yield state
    elif isinstance(state, dict):
        for value in state.values():
            yield from _state_tensors(value)
    elif isinstance(state, (tuple, list)):
        for value in state:
            yield from _state_tensors(value)


def _state_storage_bytes(states: list[object]) -> int:
    seen: set[tuple[str, int]] = set()
    total = 0
    for state in states:
        for tensor in _state_tensors(state):
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return total


@torch.inference_mode()
def profile_model(
    model: DecoderLanguageModel,
    settings: ProfileConfig,
    checkpoint_path: str | Path | None = None,
    weights_path: str | Path | None = None,
) -> dict:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    precision_dtype = next(model.parameters()).dtype
    result: dict = {
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": model.parameter_count(trainable_only=True),
        "model_tensor_bytes": unique_tensor_storage_bytes(model),
        "checkpoint_artifact_bytes": Path(checkpoint_path).stat().st_size
        if checkpoint_path and Path(checkpoint_path).exists()
        else None,
        "model_weights_artifact_bytes": Path(weights_path).stat().st_size
        if weights_path and Path(weights_path).exists()
        else None,
        "process_memory_after_load": process_memory_bytes(),
        "accelerator_memory_after_load": accelerator_memory(device),
        "inference_dtype": str(precision_dtype),
        "device": str(device),
        "energy_per_token_joules": None,
        "energy_measurement_status": "unavailable; no supported energy sensor configured",
        "workloads": [],
    }
    for requested in settings.prompt_lengths:
        prompt_length = min(int(requested), model.config.max_seq_len - settings.decode_tokens)
        if prompt_length <= 0:
            continue
        ids = torch.randint(
            model.config.vocab_size - 3, (settings.batch_size, prompt_length), device=device
        )
        for _ in range(settings.warmup_steps):
            model(ids, use_cache=True)
        _sync(device)
        prefill_times = []
        for _ in range(settings.repeats):
            start = time.perf_counter()
            _, prefill_cache = model(ids, use_cache=True)
            _sync(device)
            prefill_times.append(time.perf_counter() - start)
            del prefill_cache

        def decode_cached(prompt_ids: torch.Tensor = ids) -> tuple[float, int]:
            logits, cache = model(prompt_ids, use_cache=True)
            _sync(device)
            decode_start = time.perf_counter()
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(1, settings.decode_tokens):
                logits, cache = model(token, cache=cache, use_cache=True)
                token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync(device)
            cache_bytes = _state_storage_bytes(cache.states)
            return time.perf_counter() - decode_start, cache_bytes

        for _ in range(settings.warmup_steps):
            decode_cached()
        decode_times = []
        actual_cache_bytes = None
        for _ in range(settings.repeats):
            elapsed, actual_cache_bytes = decode_cached()
            decode_times.append(elapsed)
        prefill_median = statistics.median(prefill_times)
        decode_median = statistics.median(decode_times)
        kv_element_size = 2 if precision_dtype in {torch.float16, torch.bfloat16} else 4
        cache_tokens = prompt_length + settings.decode_tokens - 1
        all_attention = all(name == "attention" for name in model.config.sequence_types)
        theoretical_kv_bytes = (
            2
            * model.config.layers
            * settings.batch_size
            * cache_tokens
            * model.config.kv_heads
            * (model.config.width // model.config.query_heads)
            * kv_element_size
            if all_attention
            else None
        )
        result["workloads"].append(
            {
                "batch_size": settings.batch_size,
                "prompt_tokens": prompt_length,
                "requested_decode_tokens": settings.decode_tokens,
                "prefill_latency_seconds_median": prefill_median,
                "prefill_tokens_per_second": settings.batch_size * prompt_length / prefill_median,
                "decode_latency_seconds_median": decode_median,
                "decode_tokens_per_second": settings.batch_size
                * settings.decode_tokens
                / decode_median,
                "decode_seconds_per_token": decode_median / settings.decode_tokens,
                "decode_measurement": "cached incremental loop; excludes prompt prefill",
                "theoretical_kv_bytes_after_decode_loop": theoretical_kv_bytes,
                "theoretical_kv_bytes_status": "all sequence blocks use standard attention"
                if all_attention
                else "not estimated for this sequence-module mix",
                "actual_kv_bytes_after_decode": actual_cache_bytes,
                "kv_cache_accounting_dtype": str(precision_dtype),
                "peak_process_memory": process_memory_bytes(),
                "peak_accelerator_memory": accelerator_memory(device),
            }
        )
    model.train(was_training)
    return result
