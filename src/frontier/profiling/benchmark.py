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


def _timing_summary(values: list[float]) -> dict[str, float | list[float] | None]:
    return {
        "repetitions_seconds": values,
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "sample_stddev_seconds": statistics.stdev(values) if len(values) > 1 else None,
    }


def _maximum_accelerator_memory(records: list[dict[str, int | None]]) -> dict[str, int | None]:
    return {
        key: max((record[key] for record in records if record[key] is not None), default=None)
        for key in (
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        )
    }


@torch.inference_mode()
def profile_model(
    model: DecoderLanguageModel,
    settings: ProfileConfig,
    checkpoint_path: str | Path | None = None,
    weights_path: str | Path | None = None,
) -> dict:
    device = next(model.parameters()).device
    if settings.batch_size <= 0 or settings.decode_tokens <= 0 or settings.repeats <= 0:
        raise ValueError("profile batch size, decode token count and repeats must be positive")
    if settings.warmup_steps < 0:
        raise ValueError("profile warmup_steps must be non-negative")
    for requested in settings.prompt_lengths:
        if requested <= 0 or requested + settings.decode_tokens > model.config.max_seq_len:
            raise ValueError(
                f"requested prompt length {requested} plus {settings.decode_tokens} decode "
                f"tokens exceeds model.max_seq_len={model.config.max_seq_len}"
            )
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    precision_dtype = next(model.parameters()).dtype
    result: dict = {
        "profile_schema_version": 3,
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
        prompt_length = int(requested)
        ids = torch.randint(
            model.config.vocab_size - 3, (settings.batch_size, prompt_length), device=device
        )
        if device.type == "cuda":
            _sync(device)
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(settings.warmup_steps):
            model(ids, use_cache=True)
        _sync(device)
        prefill_times = []
        for _ in range(settings.repeats):
            _sync(device)
            start = time.perf_counter()
            logits, prefill_cache = model(ids, use_cache=True)
            logits[:, -1, :].argmax(dim=-1)
            _sync(device)
            prefill_times.append(time.perf_counter() - start)
            del prefill_cache
        prefill_accelerator_memory = accelerator_memory(device)

        def decode_cached(
            prompt_ids: torch.Tensor = ids,
        ) -> tuple[float, int, int | None, dict[str, int | None]]:
            logits, cache = model(prompt_ids, use_cache=True)
            _sync(device)
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            decode_start = time.perf_counter()
            for _ in range(settings.decode_tokens):
                logits, cache = model(token, cache=cache, use_cache=True)
                token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync(device)
            cache_bytes = _state_storage_bytes(cache.states)
            decode_memory = accelerator_memory(device)
            state_position = getattr(cache, "position", None)
            if not isinstance(state_position, int):
                state_position = None
            return time.perf_counter() - decode_start, cache_bytes, state_position, decode_memory

        decode_memory_records = []
        for _ in range(settings.warmup_steps):
            _, _, _, phase_memory = decode_cached()
            decode_memory_records.append(phase_memory)
        decode_times = []
        actual_cache_bytes = None
        state_position_tokens = None
        for _ in range(settings.repeats):
            elapsed, actual_cache_bytes, state_position_tokens, phase_memory = decode_cached()
            decode_times.append(elapsed)
            decode_memory_records.append(phase_memory)
        decode_accelerator_memory = _maximum_accelerator_memory(decode_memory_records)
        prefill_summary = _timing_summary(prefill_times)
        decode_summary = _timing_summary(decode_times)
        prefill_median = float(prefill_summary["median_seconds"])
        decode_median = float(decode_summary["median_seconds"])
        kv_element_size = 2 if precision_dtype in {torch.float16, torch.bfloat16} else 4
        cache_tokens = prompt_length + settings.decode_tokens
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
                "requested_prompt_tokens": int(requested),
                "prompt_tokens": prompt_length,
                "first_token_from_prefill_logits": 1,
                "generated_tokens_per_repeat": 1 + settings.decode_tokens,
                "timed_decode_tokens": settings.decode_tokens,
                "timed_decode_forwards_per_repeat": settings.decode_tokens,
                "decode_repeats": settings.repeats,
                "state_position_tokens_after_decode": state_position_tokens,
                "prefill_to_first_token_latency": prefill_summary,
                "prefill_latency_seconds_median": prefill_median,
                "prefill_tokens_per_second": settings.batch_size * prompt_length / prefill_median,
                "decode_latency": decode_summary,
                "decode_latency_seconds_median": decode_median,
                "decode_tokens_per_second": settings.batch_size
                * settings.decode_tokens
                / decode_median,
                "decode_seconds_per_token": decode_median / settings.decode_tokens,
                "decode_measurement": (
                    "exactly N cached incremental forwards per repeat producing N tokens after "
                    "the first prefill-selected token; includes greedy token selection"
                ),
                "prefill_measurement": (
                    "prompt forward plus greedy selection of the first generated token"
                ),
                "theoretical_kv_cache_tokens_after_decode_loop": cache_tokens
                if all_attention
                else None,
                "theoretical_kv_bytes_after_decode_loop": theoretical_kv_bytes,
                "theoretical_kv_bytes_status": "all sequence blocks use standard attention"
                if all_attention
                else "not estimated for this sequence-module mix",
                "actual_state_storage_bytes_after_decode": actual_cache_bytes,
                "actual_kv_bytes_after_decode": actual_cache_bytes if all_attention else None,
                "persistent_state_bytes_after_decode": actual_cache_bytes
                if not all_attention
                else None,
                "active_state_bytes_after_decode": actual_cache_bytes if all_attention else None,
                "active_state_bytes_status": "exact for dense attention cache"
                if all_attention
                else "module does not expose active-versus-allocated state accounting",
                "kv_cache_accounting_dtype": str(precision_dtype),
                "process_memory_snapshot_after_workload": process_memory_bytes(),
                "prefill_accelerator_memory": prefill_accelerator_memory,
                "prefill_accelerator_memory_scope": (
                    "prefill warmup and measured repetitions; peak reset per prompt workload"
                ),
                "decode_accelerator_memory": decode_accelerator_memory,
                "decode_accelerator_memory_scope": (
                    "cached incremental decode after prompt prefill; maximum across warmup "
                    "and measured repetitions"
                ),
            }
        )
    model.train(was_training)
    return result
