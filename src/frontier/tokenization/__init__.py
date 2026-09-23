"""Tokenizer contracts and implementations."""

from .bpe import ByteBPETokenizer
from .byte import ByteTokenizer, Tokenizer
from .registry import (
    build_tokenizer,
    load_tokenizer_artifact,
    register_tokenizer,
    tokenizer_artifact,
)

register_tokenizer(ByteBPETokenizer.name, None, ByteBPETokenizer.from_dict)

__all__ = [
    "ByteBPETokenizer",
    "ByteTokenizer",
    "Tokenizer",
    "build_tokenizer",
    "load_tokenizer_artifact",
    "register_tokenizer",
    "tokenizer_artifact",
]
