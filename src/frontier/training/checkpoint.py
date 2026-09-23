"""Atomic training checkpoints with optimizer, schedule and RNG state."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: str | Path, state: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, target)


def load_checkpoint(path: str | Path, map_location: torch.device | str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
