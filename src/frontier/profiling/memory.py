"""Portable best-effort process and accelerator memory readings."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.connection import Connection
from typing import Any

import torch


def _windows_process_memory_bytes() -> dict[str, int | str | None]:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    return {
        "rss_bytes": int(counters.WorkingSetSize) if ok else None,
        "private_bytes": int(counters.PrivateUsage) if ok else None,
        "peak_rss_bytes": int(counters.PeakWorkingSetSize) if ok else None,
        "method": "Win32 GetProcessMemoryInfo",
    }


def _native_peak_rss_bytes() -> tuple[int | None, str]:
    if os.name == "nt":
        memory = _windows_process_memory_bytes()
        peak = memory["peak_rss_bytes"]
        return (int(peak) if isinstance(peak, int) else None), str(memory["method"])

    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak *= 1024
        return int(peak), "resource.ru_maxrss"
    except ImportError:
        return None, "process high-water sensor unavailable"


def _isolated_worker(
    send: Connection,
    target: Callable[..., Any],
    args: tuple[Any, ...],
) -> None:
    """Measure one callable in a fresh process so its high-water mark is phase-local."""
    try:
        baseline = process_memory_bytes()
        peak_before, method = _native_peak_rss_bytes()
        target(*args)
        peak_after, after_method = _native_peak_rss_bytes()
        send.send(
            {
                "status": "measured" if peak_after is not None else "unavailable",
                "baseline_rss_bytes": baseline["rss_bytes"],
                "peak_rss_bytes": peak_after,
                "peak_rss_increase_bytes": (
                    max(0, peak_after - peak_before)
                    if peak_after is not None and peak_before is not None
                    else None
                ),
                "method": after_method if peak_after is not None else method,
                "isolation": "fresh spawned process per CPU phase",
                "reason": None if peak_after is not None else after_method,
            }
        )
    except BaseException as error:  # noqa: BLE001 - return worker failures as measurement status.
        try:
            send.send(
                {
                    "status": "unavailable",
                    "baseline_rss_bytes": None,
                    "peak_rss_bytes": None,
                    "peak_rss_increase_bytes": None,
                    "method": "isolated worker failed",
                    "isolation": "fresh spawned process per CPU phase",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        send.close()


def isolated_process_peak(
    target: Callable[..., Any],
    args: tuple[Any, ...] = (),
    timeout_seconds: float = 300.0,
) -> dict[str, int | str | None]:
    """Return a phase-local process high-water measurement from a spawned worker."""
    receive, send = get_context("spawn").Pipe(duplex=False)
    process = get_context("spawn").Process(target=_isolated_worker, args=(send, target, args))
    try:
        process.start()
        send.close()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "status": "unavailable",
                "baseline_rss_bytes": None,
                "peak_rss_bytes": None,
                "peak_rss_increase_bytes": None,
                "method": "isolated worker timed out",
                "isolation": "fresh spawned process per CPU phase",
                "reason": f"worker exceeded {timeout_seconds:g} seconds",
            }
        if receive.poll():
            return receive.recv()
        return {
            "status": "unavailable",
            "baseline_rss_bytes": None,
            "peak_rss_bytes": None,
            "peak_rss_increase_bytes": None,
            "method": "isolated worker exited without a result",
            "isolation": "fresh spawned process per CPU phase",
            "reason": f"worker exit code {process.exitcode}",
        }
    except Exception as error:  # noqa: BLE001 - return optional measurement failures as data.
        return {
            "status": "unavailable",
            "baseline_rss_bytes": None,
            "peak_rss_bytes": None,
            "peak_rss_increase_bytes": None,
            "method": "isolated worker could not start",
            "isolation": "fresh spawned process per CPU phase",
            "reason": f"{type(error).__name__}: {error}",
        }
    finally:
        send.close()
        receive.close()
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join()


def process_memory_bytes() -> dict[str, int | None]:
    if os.name == "nt":
        return _windows_process_memory_bytes()
    peak, peak_method = _native_peak_rss_bytes()
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process()
        info = process.memory_info()
        return {
            "rss_bytes": int(info.rss),
            "private_bytes": int(getattr(info, "private", info.rss)),
            "peak_rss_bytes": peak,
            "method": f"psutil current RSS; {peak_method}",
        }
    except ImportError:
        return {
            "rss_bytes": None,
            "private_bytes": None,
            "peak_rss_bytes": peak,
            "method": peak_method,
        }


def unique_tensor_storage_bytes(module: torch.nn.Module) -> int:
    seen: set[tuple[str, int]] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def accelerator_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def maximum_accelerator_peaks(
    records: list[dict[str, int | None]],
) -> dict[str, int | None]:
    """Return the largest phase peaks, preserving unavailable sensors as null."""
    return {
        key: max((row[key] for row in records if row.get(key) is not None), default=None)
        for key in ("peak_allocated_bytes", "peak_reserved_bytes")
    }
