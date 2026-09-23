"""Sample process memory while one fresh CPU training step is running."""

from __future__ import annotations

import math
import time
from itertools import pairwise
from multiprocessing import get_context
from multiprocessing.connection import Connection
from typing import Any

from frontier.config import RunConfig
from frontier.profiling.memory import process_memory_bytes


def _cpu_step_worker(config_data: dict[str, Any], connection: Connection) -> None:
    try:
        config = RunConfig.from_dict(config_data)
        config.train.max_steps = 1
        config.train.max_tokens = None
        from frontier.training.trainer import train

        run_dir = train(config, cpu_step_memory_probe=connection)
        connection.send({"event": "run_complete", "run_dir": str(run_dir)})
    except BaseException as error:  # noqa: BLE001 - relay worker failures to the parent.
        try:
            connection.send(
                {
                    "event": "worker_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _unavailable_measurement(reason: str, sample_interval_ms: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "unavailable",
        "scope": (
            "one CPU optimizer step from batch sampling through forward/backward/update and sync; "
            "model, optimizer and data setup, validation and checkpointing excluded"
        ),
        "method": "parent samples current memory of a fresh spawned training worker",
        "sample_interval_ms_requested": sample_interval_ms,
        "sample_count": 0,
        "baseline": {"rss_bytes": None, "private_bytes": None},
        "sampled_peak": {"rss_bytes": None, "private_bytes": None},
        "sampled_increase": {"rss_bytes": None, "private_bytes": None},
        "step_seconds": None,
        "sample_gap_seconds": {"minimum": None, "mean": None, "maximum": None},
        "peak_is_sampled": True,
        "limitation": "Polling can miss transient peaks between samples.",
        "reason": reason,
    }


def _summarize_samples(
    baseline: dict[str, Any],
    samples: list[dict[str, Any]],
    sample_times: list[float],
    sample_interval_ms: float,
    step_seconds: float,
) -> dict[str, Any]:
    peaks: dict[str, int | None] = {}
    increases: dict[str, int | None] = {}
    baselines: dict[str, int | None] = {}
    for key in ("rss_bytes", "private_bytes"):
        baseline_value = baseline.get(key)
        values = [row.get(key) for row in samples if isinstance(row.get(key), int)]
        peak = max(values) if values else None
        baselines[key] = baseline_value if isinstance(baseline_value, int) else None
        peaks[key] = peak
        increases[key] = (
            max(0, peak - baseline_value)
            if isinstance(peak, int) and isinstance(baseline_value, int)
            else None
        )
    gaps = [right - left for left, right in pairwise(sample_times)]
    methods = sorted({str(row["method"]) for row in [baseline, *samples] if row.get("method")})
    rss_available = peaks["rss_bytes"] is not None
    private_available = peaks["private_bytes"] is not None
    return {
        "schema_version": 1,
        "status": "measured" if rss_available else "unavailable",
        "scope": (
            "one CPU optimizer step from batch sampling through forward/backward/update and sync; "
            "model, optimizer and data setup, validation and checkpointing excluded"
        ),
        "method": "; ".join(methods) or "process memory sensor unavailable",
        "sample_interval_ms_requested": sample_interval_ms,
        "sample_count": len(samples),
        "baseline": baselines,
        "sampled_peak": peaks,
        "sampled_increase": increases,
        "step_seconds": step_seconds,
        "sample_gap_seconds": {
            "minimum": min(gaps) if gaps else None,
            "mean": sum(gaps) / len(gaps) if gaps else None,
            "maximum": max(gaps) if gaps else None,
        },
        "peak_is_sampled": True,
        "limitation": (
            "Polling can miss transient peaks between samples; the completion-boundary sample "
            "is taken immediately after the worker reports the synchronized step."
        ),
        "reason": (
            None
            if rss_available
            else "current RSS sampling is unavailable; install psutil off Windows or use a supported Windows host"
        ),
        "private_memory_available": private_available,
    }


def profile_cpu_optimizer_step(
    config: RunConfig,
    sample_interval_ms: float = 10.0,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Run one normal optimizer step and sample its fresh worker's current memory."""
    if not math.isfinite(sample_interval_ms) or sample_interval_ms <= 0:
        raise ValueError("sample_interval_ms must be a finite positive duration")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a finite positive duration")
    config.validate()
    if config.train.resume:
        raise ValueError("CPU optimizer-step profiling requires a fresh non-resume config")

    context = get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_cpu_step_worker, args=(config.to_dict(), child))
    deadline = time.perf_counter() + timeout_seconds
    measurement: dict[str, Any] | None = None
    worker_status = "unknown"
    worker_error: str | None = None
    samples: list[dict[str, Any]] = []
    sample_times: list[float] = []

    try:
        process.start()
        child.close()
        interval_seconds = sample_interval_ms / 1000.0
        if not parent.poll(max(0.0, deadline - time.perf_counter())):
            worker_status = "timeout_before_step"
            measurement = _unavailable_measurement(
                f"worker did not finish setup within {timeout_seconds:g} seconds",
                sample_interval_ms,
            )
        else:
            message = parent.recv()
            if message.get("event") != "ready":
                worker_status = "failed_before_step"
                worker_error = message.get(
                    "error", f"unexpected worker event {message.get('event')}"
                )
                measurement = _unavailable_measurement(worker_error, sample_interval_ms)
            else:
                baseline = process_memory_bytes(process.pid)
                parent.send({"event": "begin"})
                samples.append(process_memory_bytes(process.pid))
                sample_times.append(time.perf_counter())
                step_seconds: float | None = None
                step_completed = False
                while time.perf_counter() < deadline:
                    remaining = max(0.0, deadline - time.perf_counter())
                    if parent.poll(min(interval_seconds, remaining)):
                        message = parent.recv()
                        if message.get("event") == "optimizer_step_complete":
                            step_seconds = float(message["step_seconds"])
                            samples.append(process_memory_bytes(process.pid))
                            sample_times.append(time.perf_counter())
                            step_completed = True
                            break
                        if message.get("event") == "worker_error":
                            worker_error = str(message.get("error", "training worker failed"))
                            worker_status = "failed_during_step"
                            break
                    else:
                        samples.append(process_memory_bytes(process.pid))
                        sample_times.append(time.perf_counter())
                if step_completed:
                    measurement = _summarize_samples(
                        baseline, samples, sample_times, sample_interval_ms, step_seconds or 0.0
                    )
                    parent.send(measurement)
                    remaining = max(0.0, deadline - time.perf_counter())
                    if parent.poll(remaining):
                        message = parent.recv()
                        if message.get("event") == "run_complete":
                            worker_status = "completed"
                        elif message.get("event") == "worker_error":
                            worker_status = "failed_after_step"
                            worker_error = str(message.get("error", "training worker failed"))
                    else:
                        worker_status = "timeout_after_step"
                        worker_error = f"worker did not finish its bounded run within {timeout_seconds:g} seconds"
                elif measurement is None:
                    reason = worker_error or f"optimizer step exceeded {timeout_seconds:g} seconds"
                    measurement = _unavailable_measurement(reason, sample_interval_ms)
                    worker_status = (
                        worker_status if worker_status != "unknown" else "timeout_during_step"
                    )

        if process.pid is not None:
            if worker_status == "completed":
                process.join(max(0.0, deadline - time.perf_counter()))
            if process.is_alive():
                process.terminate()
            process.join()
        if measurement is None:
            measurement = _unavailable_measurement(
                "worker did not return a measurement", sample_interval_ms
            )
        return {
            "status": measurement["status"],
            "measurement": measurement,
            "worker_status": worker_status,
            "worker_error": worker_error,
            "run_dir": config.output_dir
            if worker_status in {"completed", "timeout_after_step"}
            else None,
        }
    except Exception as error:  # noqa: BLE001 - optional probe failures return a structured status.
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join()
        failure = _unavailable_measurement(f"{type(error).__name__}: {error}", sample_interval_ms)
        return {
            "status": "unavailable",
            "measurement": failure,
            "worker_status": "probe_failed",
            "worker_error": f"{type(error).__name__}: {error}",
            "run_dir": None,
        }
    finally:
        parent.close()
        child.close()
