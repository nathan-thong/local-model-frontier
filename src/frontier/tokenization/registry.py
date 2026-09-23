"""Tokenizer registry so corpus encoding is decoupled from model code."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .byte import ByteTokenizer, Tokenizer

TokenizerFactory = Callable[[], Tokenizer]
TokenizerLoader = Callable[[dict[str, Any]], Tokenizer]
_REGISTRY: dict[str, TokenizerFactory] = {}
_LOADERS: dict[str, TokenizerLoader] = {}


def register_tokenizer(
    name: str,
    factory: TokenizerFactory | None,
    artifact_loader: TokenizerLoader | None = None,
) -> None:
    if not name or name in _REGISTRY or name in _LOADERS:
        raise ValueError(f"tokenizer name is empty or already registered: {name!r}")
    if factory is None and artifact_loader is None:
        raise ValueError("a tokenizer requires a factory or an artifact loader")
    if factory is not None:
        _REGISTRY[name] = factory
    if artifact_loader is not None:
        _LOADERS[name] = artifact_loader


def build_tokenizer(name: str) -> Tokenizer:
    try:
        return _REGISTRY[name]()
    except KeyError as error:
        if name in _LOADERS:
            raise ValueError(
                f"tokenizer {name!r} requires a fitted tokenizer artifact in the data directory"
            ) from error
        raise ValueError(
            f"tokenizer {name!r} is not registered (available: "
            f"{', '.join(sorted(set(_REGISTRY) | set(_LOADERS)))})"
        ) from error


def _canonical_bytes(artifact: dict[str, Any]) -> bytes:
    return json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def tokenizer_artifact(tokenizer: Tokenizer) -> dict[str, Any]:
    serializer = getattr(tokenizer, "to_dict", None)
    if not callable(serializer):
        raise TypeError(f"tokenizer {tokenizer.name!r} does not provide a serializable artifact")
    payload = serializer()
    if not isinstance(payload, dict) or payload.get("name") != tokenizer.name:
        raise ValueError("tokenizer artifact must be an object naming the active tokenizer")
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return {**payload, "artifact_sha256": digest}


def load_tokenizer_artifact(path: str | Path) -> Tokenizer:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError("tokenizer artifact file must contain a JSON object")
    digest = artifact.get("artifact_sha256")
    payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    expected_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if not isinstance(digest, str) or digest != expected_digest:
        raise ValueError("tokenizer artifact SHA-256 does not match its contents")
    name = payload.get("name")
    loader = _LOADERS.get(name) if isinstance(name, str) else None
    if loader is None:
        raise ValueError(f"no artifact loader registered for tokenizer {name!r}")
    tokenizer = loader(payload)
    if tokenizer_artifact(tokenizer) != artifact:
        raise ValueError("loaded tokenizer does not reproduce the serialized artifact")
    return tokenizer


register_tokenizer(ByteTokenizer.name, ByteTokenizer, ByteTokenizer.from_dict)
