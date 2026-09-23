"""Device, precision and deterministic RNG setup."""

from __future__ import annotations

import os
import random

import torch


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(requested: str, device: torch.device) -> tuple[torch.dtype, bool]:
    if requested == "fp32":
        return torch.float32, False
    if requested == "auto":
        if device.type == "cuda":
            return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), True
        return torch.float32, False
    if requested == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 was requested but the selected CUDA device does not support it")
        return torch.bfloat16, True
    if requested == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 autocast requires CUDA; use bf16 or fp32 on CPU")
        return torch.float16, True
    raise ValueError(f"unknown precision mode: {requested}")


def seed_everything(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


def capture_rng_state(sampler_generator: torch.Generator) -> dict:
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler": sampler_generator.get_state(),
    }
    try:
        import numpy as np  # type: ignore[import-not-found]

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def restore_rng_state(state: dict, sampler_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    sampler_generator.set_state(state["sampler"])
    if state.get("numpy") is not None:
        try:
            import numpy as np  # type: ignore[import-not-found]

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
