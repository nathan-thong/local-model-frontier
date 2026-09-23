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

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str: ...


class ByteTokenizer:
    """Maps each UTF-8 byte to one token, plus fixed BOS, EOS and PAD IDs."""

    bos_id = 256
    eos_id = 257
    pad_id = 258
    vocab_size = 259
    name = "utf8-byte-v1"

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

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
