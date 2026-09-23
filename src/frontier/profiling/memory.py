"""Portable best-effort process and accelerator memory readings."""

from __future__ import annotations

import os
import sys

import torch


def process_memory_bytes() -> dict[str, int | None]:
    if os.name == "nt":
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
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return {
            "rss_bytes": int(counters.WorkingSetSize) if ok else None,
            "private_bytes": int(counters.PrivateUsage) if ok else None,
            "peak_rss_bytes": int(counters.PeakWorkingSetSize) if ok else None,
            "method": "Win32 GetProcessMemoryInfo",
        }
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process()
        info = process.memory_info()
        return {
            "rss_bytes": int(info.rss),
            "private_bytes": int(getattr(info, "private", info.rss)),
            "peak_rss_bytes": None,
            "method": "psutil",
        }
    except ImportError:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform != "darwin":
                peak *= 1024
            return {
                "rss_bytes": None,
                "private_bytes": None,
                "peak_rss_bytes": int(peak),
                "method": "resource.ru_maxrss",
            }
        except ImportError:
            return {
                "rss_bytes": None,
                "private_bytes": None,
                "peak_rss_bytes": None,
                "method": "unavailable",
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
