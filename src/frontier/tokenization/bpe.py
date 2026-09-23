"""Deterministic byte-level BPE fitted from an explicit training document set."""

from __future__ import annotations

import hashlib
import heapq
import unicodedata
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

BYTE_VOCAB_SIZE = 256
BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
FIRST_MERGE_ID = 259


def _identity(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ByteBPETokenizer:
    """UTF-8 byte fallback plus ordered adjacent-pair merges and fixed special IDs."""

    name = "byte-bpe-v1"
    bos_id = BOS_ID
    eos_id = EOS_ID
    pad_id = PAD_ID

    def __init__(
        self,
        merges: list[tuple[int, int]],
        *,
        target_vocab_size: int,
        min_pair_frequency: int,
        training_source_sha256: str,
        training_document_identity_sha256: list[str],
        training_document_count: int,
        training_utf8_bytes: int,
    ) -> None:
        if target_vocab_size < FIRST_MERGE_ID:
            raise ValueError("target vocabulary must include byte and special tokens")
        if min_pair_frequency < 1:
            raise ValueError("minimum pair frequency must be positive")
        if (
            not isinstance(training_source_sha256, str)
            or len(training_source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in training_source_sha256)
        ):
            raise ValueError("training source SHA-256 must be 64 lowercase hexadecimal characters")
        if training_document_count <= 0 or len(training_document_identity_sha256) != (
            training_document_count
        ):
            raise ValueError("training document count and identity list do not agree")
        if any(
            len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
            for identity in training_document_identity_sha256
        ):
            raise ValueError("training document identities must be SHA-256 hashes")
        if training_utf8_bytes < 0:
            raise ValueError("training UTF-8 byte count cannot be negative")

        self.merges = tuple((int(left), int(right)) for left, right in merges)
        self.target_vocab_size = int(target_vocab_size)
        self.min_pair_frequency = int(min_pair_frequency)
        self.training_source_sha256 = training_source_sha256
        self.training_document_identity_sha256 = list(training_document_identity_sha256)
        self.training_document_count = int(training_document_count)
        self.training_utf8_bytes = int(training_utf8_bytes)
        self.vocab_size = FIRST_MERGE_ID + len(self.merges)
        if self.vocab_size > self.target_vocab_size:
            raise ValueError("merge list exceeds target vocabulary size")

        token_bytes: list[bytes | None] = [bytes([value]) for value in range(BYTE_VOCAB_SIZE)]
        token_bytes.extend([None, None, None])
        seen_pairs: set[tuple[int, int]] = set()
        for rank, pair in enumerate(self.merges):
            left, right = pair
            new_id = FIRST_MERGE_ID + rank
            if pair in seen_pairs:
                raise ValueError("merge list contains a duplicate pair")
            if (
                left < 0
                or right < 0
                or left >= new_id
                or right >= new_id
                or token_bytes[left] is None
                or token_bytes[right] is None
            ):
                raise ValueError("merge pair references an invalid or special token")
            seen_pairs.add(pair)
            token_bytes.append(token_bytes[left] + token_bytes[right])
        self._token_bytes = token_bytes
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def fit(
        cls,
        training_documents: list[str],
        *,
        training_source_sha256: str,
        target_vocab_size: int = 512,
        min_pair_frequency: int = 2,
    ) -> ByteBPETokenizer:
        if not training_documents or any(not document for document in training_documents):
            raise ValueError("tokenizer fitting requires non-empty training documents")
        if target_vocab_size < FIRST_MERGE_ID:
            raise ValueError("target vocabulary must include byte and special tokens")

        sequences = [list(document.encode("utf-8")) for document in training_documents]
        merges: list[tuple[int, int]] = []
        while FIRST_MERGE_ID + len(merges) < target_vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(pairwise(sequence))
            if not pair_counts:
                break
            pair, frequency = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_pair_frequency:
                break

            new_id = FIRST_MERGE_ID + len(merges)
            merged_sequences: list[list[int]] = []
            for sequence in sequences:
                output: list[int] = []
                index = 0
                while index < len(sequence):
                    if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                        output.append(new_id)
                        index += 2
                    else:
                        output.append(sequence[index])
                        index += 1
                merged_sequences.append(output)
            merges.append(pair)
            sequences = merged_sequences

        identities = [
            _sha256(_identity(document).encode("utf-8")) for document in training_documents
        ]
        return cls(
            merges,
            target_vocab_size=target_vocab_size,
            min_pair_frequency=min_pair_frequency,
            training_source_sha256=training_source_sha256,
            training_document_identity_sha256=identities,
            training_document_count=len(training_documents),
            training_utf8_bytes=sum(
                len(document.encode("utf-8")) for document in training_documents
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "encoding": "UTF-8",
            "base_byte_vocab_size": BYTE_VOCAB_SIZE,
            "vocab_size": self.vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "min_pair_frequency": self.min_pair_frequency,
            "special_tokens": {"bos": self.bos_id, "eos": self.eos_id, "pad": self.pad_id},
            "merge_pairs": [list(pair) for pair in self.merges],
            "training_source_sha256": self.training_source_sha256,
            "training_document_identity_sha256": list(self.training_document_identity_sha256),
            "training_document_count": self.training_document_count,
            "training_utf8_bytes": self.training_utf8_bytes,
            "encoding_rule": "apply ordered pair merges; UTF-8 bytes remain the fallback",
            "decode_rule": "expand merges to bytes and decode UTF-8 with replacement",
        }

    @classmethod
    def from_dict(cls, artifact: dict[str, Any]) -> ByteBPETokenizer:
        if (
            artifact.get("schema_version") != 1
            or artifact.get("name") != cls.name
            or artifact.get("encoding") != "UTF-8"
            or artifact.get("base_byte_vocab_size") != BYTE_VOCAB_SIZE
            or artifact.get("special_tokens") != {"bos": BOS_ID, "eos": EOS_ID, "pad": PAD_ID}
            or artifact.get("encoding_rule")
            != "apply ordered pair merges; UTF-8 bytes remain the fallback"
            or artifact.get("decode_rule")
            != "expand merges to bytes and decode UTF-8 with replacement"
        ):
            raise ValueError("tokenizer artifact does not match the byte-bpe-v1 contract")
        merge_pairs = artifact.get("merge_pairs")
        if not isinstance(merge_pairs, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in pair)
            for pair in merge_pairs
        ):
            raise ValueError("tokenizer merge_pairs must be a list of integer pairs")
        identities = artifact.get("training_document_identity_sha256")
        if not isinstance(identities, list) or any(
            not isinstance(value, str) for value in identities
        ):
            raise ValueError("tokenizer training identities must be a list of hashes")
        for key in (
            "target_vocab_size",
            "min_pair_frequency",
            "training_document_count",
            "training_utf8_bytes",
            "vocab_size",
        ):
            value = artifact.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"tokenizer artifact {key} must be an integer")
        if not isinstance(artifact.get("training_source_sha256"), str):
            raise TypeError("tokenizer training source SHA-256 must be a string")
        tokenizer = cls(
            [(pair[0], pair[1]) for pair in merge_pairs],
            target_vocab_size=artifact.get("target_vocab_size"),
            min_pair_frequency=artifact.get("min_pair_frequency"),
            training_source_sha256=artifact.get("training_source_sha256"),
            training_document_identity_sha256=identities,
            training_document_count=artifact.get("training_document_count"),
            training_utf8_bytes=artifact.get("training_utf8_bytes"),
        )
        if tokenizer.vocab_size != artifact.get("vocab_size"):
            raise ValueError("tokenizer artifact vocabulary size does not match its merge list")
        if tokenizer.to_dict() != artifact:
            raise ValueError("tokenizer artifact contains unsupported or inconsistent fields")
        return tokenizer

    def _encode_content(self, content: list[int]) -> list[int]:
        if len(content) < 2 or not self.merges:
            return content

        count = len(content)
        tokens = list(content)
        previous = [index - 1 for index in range(count)]
        following = [index + 1 for index in range(count)]
        following[-1] = -1
        alive = [True] * count
        candidates: list[tuple[int, int, int, int, int]] = []

        def add_candidate(left_index: int, right_index: int) -> None:
            if left_index < 0 or right_index < 0:
                return
            pair = (tokens[left_index], tokens[right_index])
            rank = self._merge_ranks.get(pair)
            if rank is not None:
                heapq.heappush(
                    candidates,
                    (rank, left_index, right_index, pair[0], pair[1]),
                )

        for index in range(count - 1):
            add_candidate(index, index + 1)

        while candidates:
            rank, left_index, right_index, left_token, right_token = heapq.heappop(candidates)
            if (
                not alive[left_index]
                or not alive[right_index]
                or following[left_index] != right_index
                or tokens[left_index] != left_token
                or tokens[right_index] != right_token
            ):
                continue
            tokens[left_index] = FIRST_MERGE_ID + rank
            alive[right_index] = False
            next_index = following[right_index]
            following[left_index] = next_index
            if next_index >= 0:
                previous[next_index] = left_index
            add_candidate(previous[left_index], left_index)
            add_candidate(left_index, next_index)

        output: list[int] = []
        index = 0
        while index >= 0:
            if alive[index]:
                output.append(tokens[index])
            index = following[index]
        return output

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        content = self._encode_content(list(text.encode("utf-8")))
        return ([self.bos_id] if add_bos else []) + content + ([self.eos_id] if add_eos else [])

    def token_byte_counts(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        content = self._encode_content(list(text.encode("utf-8")))
        counts = [len(self._token_bytes[token_id]) for token_id in content]
        return ([0] if add_bos else []) + counts + ([0] if add_eos else [])

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        raw = bytearray()
        special = {self.bos_id: b"<bos>", self.eos_id: b"<eos>", self.pad_id: b"<pad>"}
        for token in ids:
            token_id = int(token)
            if 0 <= token_id < BYTE_VOCAB_SIZE:
                raw.append(token_id)
            elif token_id in special:
                if not skip_special:
                    raw.extend(special[token_id])
            elif FIRST_MERGE_ID <= token_id < self.vocab_size:
                raw.extend(self._token_bytes[token_id])
            else:
                raise ValueError(f"token ID {token_id} is outside the byte-bpe-v1 vocabulary")
        return raw.decode("utf-8", errors="replace")


def fit_byte_bpe_from_file(
    training_path: str | Path,
    *,
    target_vocab_size: int = 512,
    min_pair_frequency: int = 2,
) -> ByteBPETokenizer:
    path = Path(training_path)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    documents = [line.strip() for line in text.splitlines() if line.strip()]
    return ByteBPETokenizer.fit(
        documents,
        training_source_sha256=_sha256(raw),
        target_vocab_size=target_vocab_size,
        min_pair_frequency=min_pair_frequency,
    )
