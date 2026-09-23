"""Tokenizer registry so corpus encoding is decoupled from model code."""

from __future__ import annotations

from collections.abc import Callable

from .byte import ByteTokenizer, Tokenizer

TokenizerFactory = Callable[[], Tokenizer]
_REGISTRY: dict[str, TokenizerFactory] = {}


def register_tokenizer(name: str, factory: TokenizerFactory) -> None:
    if not name or name in _REGISTRY:
        raise ValueError(f"tokenizer name is empty or already registered: {name!r}")
    _REGISTRY[name] = factory


def build_tokenizer(name: str) -> Tokenizer:
    try:
        return _REGISTRY[name]()
    except KeyError as error:
        raise ValueError(
            f"tokenizer {name!r} is not registered (available: {', '.join(sorted(_REGISTRY))})"
        ) from error


register_tokenizer(ByteTokenizer.name, ByteTokenizer)
