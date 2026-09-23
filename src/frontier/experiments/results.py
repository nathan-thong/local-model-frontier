"""Versioned JSON/CSV artifacts without third-party data dependencies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(flatten(item, name))
        elif isinstance(item, (list, tuple)) and all(isinstance(entry, dict) for entry in item):
            for index, entry in enumerate(item):
                flattened.update(flatten(entry, f"{name}[{index}]"))
        elif isinstance(item, (list, tuple)):
            flattened[name] = json.dumps(item, ensure_ascii=False, allow_nan=False)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            flattened[name] = item
    return flattened


def write_summary(run_dir: str | Path, summary: dict[str, Any]) -> None:
    root = Path(run_dir)
    write_json(root / "summary.json", summary)
    flat = flatten(summary)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(sorted(flat.items()))
