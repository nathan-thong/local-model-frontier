"""A fixed-vocabulary UTF-8 byte tokenizer for offline, leakage-free baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]: ...

    def token_byte_counts(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]: ...

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str: ...


class ByteTokenizer:
    """Maps each UTF-8 byte to one token, plus fixed BOS, EOS and PAD IDs."""

    bos_id = 256
    eos_id = 257
    pad_id = 258
    vocab_size = 259
    name = "utf8-byte-v1"

    def to_dict(self) -> dict:
        """Return the complete, versioned artifact needed to recreate this tokenizer."""
        return {
            "schema_version": 1,
            "name": self.name,
            "vocab_size": self.vocab_size,
            "encoding": "UTF-8",
            "byte_to_token_id": "identity mapping for byte values 0 through 255",
            "special_tokens": {"bos": self.bos_id, "eos": self.eos_id, "pad": self.pad_id},
            "encode_defaults": {"add_bos": True, "add_eos": True},
            "decode_defaults": {"skip_special": True, "invalid_utf8": "replace"},
        }

    @classmethod
    def from_dict(cls, artifact: dict) -> ByteTokenizer:
        tokenizer = cls()
        if artifact != tokenizer.to_dict():
            raise ValueError("tokenizer artifact does not match the utf8-byte-v1 contract")
        return tokenizer

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def token_byte_counts(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Return UTF-8 byte coverage aligned with encode(); special tokens cover zero bytes."""
        byte_count = len(text.encode("utf-8"))
        return ([0] if add_bos else []) + [1] * byte_count + ([0] if add_eos else [])

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        raw = bytearray()
        for token in ids:
            value = int(token)
            if value < 256:
                raw.append(value)
            elif not skip_special:
                raw.extend(
                    {self.bos_id: b"<bos>", self.eos_id: b"<eos>", self.pad_id: b"<pad>"}.get(
                        value, b""
                    )
                )
        return raw.decode("utf-8", errors="replace")

    def encode_batch(
        self, texts: Sequence[str], add_bos: bool = True, add_eos: bool = True
    ) -> list[list[int]]:
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    @staticmethod
    def tensor(ids: Sequence[int], device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(ids, dtype=torch.long, device=device)
