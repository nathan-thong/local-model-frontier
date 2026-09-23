"""Fixed-shape prefill and incremental decoding profile."""

from __future__ import annotations

import hashlib
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from frontier.config import ProfileConfig
from frontier.models import DecoderLanguageModel
from frontier.models.sequence import describe_sequence_module, iter_state_tensors
from frontier.profiling.memory import (
    accelerator_memory,
    isolated_process_peak,
    process_memory_bytes,
    unique_tensor_storage_bytes,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_input_ids(
    vocab_size: int,
    batch_size: int,
    prompt_length: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Build and hash a repeatable random-token prompt before moving it to a device."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("profile input seed must be an integer in [0, 2**63)")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    ids_cpu = torch.randint(
        vocab_size - 3,
        (batch_size, prompt_length),
        generator=generator,
        device="cpu",
    )
    digest = hashlib.sha256(ids_cpu.contiguous().numpy().tobytes()).hexdigest()
    return ids_cpu.to(device), digest


def _state_memory_accounting(
    sequence_modules: list[torch.nn.Module], states: list[object]
) -> dict[str, int | str | None]:
    """Count state backing stores and module-declared active contiguous regions."""
    if len(sequence_modules) != len(states):
        raise ValueError("sequence module and state counts differ")
    allocated_tensors: list[torch.Tensor] = []
    module_state_tensors: list[list[torch.Tensor]] = []
    for module, state in zip(sequence_modules, states, strict=True):
        if state is None:
            module_state_tensors.append([])
            continue
        state_tensors = getattr(module, "state_tensors", None)
        try:
            state_tree = state_tensors(state) if callable(state_tensors) else state
            tensor_list = list(iter_state_tensors(state_tree))
        except TypeError as error:
            return {
                "allocated_bytes": None,
                "active_bytes": None,
                "status": f"{type(module).__name__} state traversal unavailable: {error}",
            }
        module_state_tensors.append(tensor_list)
        allocated_tensors.extend(tensor_list)

    seen_storages: set[tuple[str, int]] = set()
    allocated_bytes = 0
    for tensor in allocated_tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen_storages:
            seen_storages.add(key)
            allocated_bytes += storage.nbytes()

    storage_sizes: dict[tuple[str, int], int] = {}
    active_intervals: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for tensors in module_state_tensors:
        for tensor in tensors:
            storage = tensor.untyped_storage()
            if storage.nbytes():
                storage_sizes[(str(tensor.device), storage.data_ptr())] = storage.nbytes()

    for module, state in zip(sequence_modules, states, strict=True):
        if state is None:
            continue
        active_state_tensors = getattr(module, "active_state_tensors", None)
        if not callable(active_state_tensors):
            return {
                "allocated_bytes": allocated_bytes,
                "active_bytes": None,
                "status": f"{type(module).__name__} does not expose active state",
            }
        try:
            tensors = active_state_tensors(state)
            active_tensor_list = list(iter_state_tensors(tensors))
        except TypeError as error:
            return {
                "allocated_bytes": allocated_bytes,
                "active_bytes": None,
                "status": f"{type(module).__name__} active-state traversal unavailable: {error}",
            }
        for tensor in active_tensor_list:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key not in storage_sizes:
                raise ValueError("active state tensor is not backed by the returned sequence state")
            if not tensor.is_contiguous():
                return {
                    "allocated_bytes": allocated_bytes,
                    "active_bytes": None,
                    "status": "active state view is non-contiguous; exact bytes unavailable",
                }
            start = tensor.storage_offset() * tensor.element_size()
            end = start + tensor.numel() * tensor.element_size()
            if end > storage_sizes[key]:
                raise ValueError("active state view exceeds its backing storage")
            active_intervals[key].append((start, end))

    active_bytes = 0
    for intervals in active_intervals.values():
        intervals.sort()
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                active_bytes += end - start
                start, end = next_start, next_end
        active_bytes += end - start
    return {
        "allocated_bytes": allocated_bytes,
        "active_bytes": active_bytes,
        "status": "exact union of module-declared active contiguous tensor regions",
    }


@torch.inference_mode()
def _isolated_prefill(model: DecoderLanguageModel, input_ids: torch.Tensor) -> None:
    model.eval()
    logits, _ = model(input_ids, use_cache=True)
    logits[:, -1, :].argmax(dim=-1)


@torch.inference_mode()
def _isolated_decode(
    model: DecoderLanguageModel,
    token: torch.Tensor,
    cache: object,
    decode_tokens: int,
) -> None:
    model.eval()
    for _ in range(decode_tokens):
        logits, cache = model(token, cache=cache, use_cache=True)
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)


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
    input_seed: int = 0,
) -> dict:
    device = next(model.parameters()).device
    if settings.batch_size <= 0 or settings.decode_tokens <= 0 or settings.repeats <= 0:
        raise ValueError("profile batch size, decode token count and repeats must be positive")
    if settings.warmup_steps < 0:
        raise ValueError("profile warmup_steps must be non-negative")
    if type(input_seed) is not int or not 0 <= input_seed < 2**63:
        raise ValueError("profile input seed must be an integer in [0, 2**63)")
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
        "profile_schema_version": 5,
        "state_accounting_version": 2,
        "profile_input_seed": input_seed,
        "sequence_modules": [
            asdict(describe_sequence_module(block.sequence)) for block in model.blocks
        ],
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
        ids, prompt_input_sha256 = profile_input_ids(
            model.config.vocab_size,
            settings.batch_size,
            prompt_length,
            input_seed,
            device,
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

        prefill_cpu_memory = (
            isolated_process_peak(_isolated_prefill, (model, ids))
            if device.type == "cpu"
            else {
                "status": "unavailable",
                "baseline_rss_bytes": None,
                "peak_rss_bytes": None,
                "peak_rss_increase_bytes": None,
                "method": "CPU process high-water measurement applies only to CPU workloads",
                "isolation": None,
                "reason": f"model device is {device}",
            }
        )

        def decode_cached(
            prompt_ids: torch.Tensor = ids,
        ) -> tuple[float, dict[str, int | str | None], int | None, dict[str, int | None]]:
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
            state_memory = _state_memory_accounting(
                [block.sequence for block in model.blocks], cache.states
            )
            decode_memory = accelerator_memory(device)
            state_position = getattr(cache, "position", None)
            if not isinstance(state_position, int):
                state_position = None
            return time.perf_counter() - decode_start, state_memory, state_position, decode_memory

        decode_memory_records = []
        for _ in range(settings.warmup_steps):
            _, _, _, phase_memory = decode_cached()
            decode_memory_records.append(phase_memory)
        decode_times = []
        state_memory: dict[str, int | str | None] = {
            "allocated_bytes": None,
            "active_bytes": None,
            "status": "decode workload did not run",
        }
        state_position_tokens = None
        for _ in range(settings.repeats):
            elapsed, state_memory, state_position_tokens, phase_memory = decode_cached()
            decode_times.append(elapsed)
            decode_memory_records.append(phase_memory)
        decode_accelerator_memory = _maximum_accelerator_memory(decode_memory_records)

        if device.type == "cpu":
            decode_logits, decode_cache = model(ids, use_cache=True)
            decode_token = decode_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decode_cpu_memory = isolated_process_peak(
                _isolated_decode,
                (model, decode_token, decode_cache, settings.decode_tokens),
            )
        else:
            decode_cpu_memory = {
                "status": "unavailable",
                "baseline_rss_bytes": None,
                "peak_rss_bytes": None,
                "peak_rss_increase_bytes": None,
                "method": "CPU process high-water measurement applies only to CPU workloads",
                "isolation": None,
                "reason": f"model device is {device}",
            }
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
                "prompt_input_sha256": prompt_input_sha256,
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
                "actual_state_storage_bytes_after_decode": state_memory["allocated_bytes"],
                "allocated_state_bytes_after_decode": state_memory["allocated_bytes"],
                "actual_kv_bytes_after_decode": state_memory["allocated_bytes"]
                if all_attention
                else None,
                "persistent_state_bytes_after_decode": state_memory["allocated_bytes"],
                "active_state_bytes_after_decode": state_memory["active_bytes"],
                "state_memory_accounting_status": state_memory["status"],
                "kv_cache_accounting_dtype": str(precision_dtype),
                "process_memory_snapshot_after_workload": process_memory_bytes(),
                "prefill_cpu_memory": prefill_cpu_memory,
                "prefill_cpu_memory_scope": "isolated CPU process high-water delta for prefill",
                "decode_cpu_memory": decode_cpu_memory,
                "decode_cpu_memory_scope": (
                    "isolated CPU process high-water delta for cached decode after prefill"
                ),
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
