"""Corpus loading, deterministic document splitting and seeded sampling."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import torch

from frontier.tokenization import Tokenizer

PREPROCESSING_CONTRACT = {
    "schema_version": 1,
    "input_format": "UTF-8 text with one document per non-empty line",
    "line_handling": "strip leading and trailing Unicode whitespace",
    "duplicate_identity": "Unicode NFC then collapse internal Unicode whitespace to ASCII spaces",
    "split_unit": "normalized document identity",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _document_identity(document: str) -> str:
    normalized = unicodedata.normalize("NFC", document)
    return " ".join(normalized.split())


def _ensure_new_output_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty data output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_documents(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8") as handle:
        docs = [line.strip() for line in handle if line.strip()]
    if not docs:
        raise ValueError(f"no non-empty documents found in {path}")
    return docs


def prepare_split(
    input_path: str | Path,
    output_dir: str | Path,
    validation_fraction: float,
    seed: int,
    source_metadata: dict | None = None,
    content_origin: str | None = None,
) -> dict:
    source = Path(input_path)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    docs = read_documents(source)
    unique_docs = list(dict.fromkeys(_document_identity(doc) for doc in docs))
    if len(unique_docs) < 2:
        raise ValueError("at least two non-empty documents are required for a split")
    if source_metadata is not None and not isinstance(source_metadata, dict):
        raise ValueError("source_metadata must be a JSON object")
    metadata_origin = source_metadata.get("content_origin") if source_metadata else None
    if content_origin is not None and metadata_origin not in (None, content_origin):
        raise ValueError("content_origin conflicts with source_metadata.content_origin")
    origin = content_origin if content_origin is not None else metadata_origin or "unknown"
    if not isinstance(origin, str) or origin not in {"human", "synthetic", "mixed", "unknown"}:
        raise ValueError("content_origin must be human, synthetic, mixed or unknown")
    provenance = dict(source_metadata) if source_metadata is not None else None
    if provenance is not None:
        provenance.setdefault("content_origin", origin)
    random.Random(seed).shuffle(unique_docs)
    valid_count = max(1, min(len(unique_docs) - 1, round(len(unique_docs) * validation_fraction)))
    valid_set = set(unique_docs[:valid_count])
    # Normalized duplicates stay together while retaining the source text verbatim.
    train_docs = [doc for doc in docs if _document_identity(doc) not in valid_set]
    valid_docs = [doc for doc in docs if _document_identity(doc) in valid_set]

    output = Path(output_dir)
    _ensure_new_output_directory(output)
    train_path = output / "train.txt"
    valid_path = output / "validation.txt"
    train_path.write_text("\n".join(train_docs) + "\n", encoding="utf-8")
    valid_path.write_text("\n".join(valid_docs) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 3,
        "is_test_fixture": False,
        "content_origin": origin,
        "preprocessing": PREPROCESSING_CONTRACT,
        "preprocessing_sha256": _canonical_json_sha256(PREPROCESSING_CONTRACT),
        "source_path_name": source.name,
        "source_sha256": sha256_file(source),
        "split_seed": seed,
        "validation_fraction_requested": validation_fraction,
        "document_count": len(docs),
        "unique_document_count": len(unique_docs),
        "train_document_count": len(train_docs),
        "validation_document_count": len(valid_docs),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(valid_path),
        "train_utf8_bytes": train_path.stat().st_size,
        "validation_utf8_bytes": valid_path.stat().st_size,
        "split_unit": PREPROCESSING_CONTRACT["split_unit"],
    }
    if provenance is not None:
        manifest["source"] = provenance.get("dataset", source.name)
        manifest["source_metadata"] = provenance
        manifest["source_metadata_sha256"] = _canonical_json_sha256(provenance)
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_split(data_dir: str | Path) -> tuple[list[str], list[str], dict]:
    root = Path(data_dir)
    train_path = root / "train.txt"
    valid_path = root / "validation.txt"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"expected train.txt and validation.txt in {root}")
    train_docs = read_documents(train_path)
    valid_docs = read_documents(valid_path)
    train_set = set(train_docs)
    overlap = train_set.intersection(valid_docs)
    if overlap:
        raise ValueError(f"train/validation contain {len(overlap)} identical documents")
    manifest_path = root / "data_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    if not isinstance(manifest, dict):
        raise TypeError(f"data manifest must contain a JSON object in {root}")
    manifest.setdefault("content_origin", "unknown")
    if not isinstance(manifest["content_origin"], str) or manifest["content_origin"] not in {
        "human",
        "synthetic",
        "mixed",
        "unknown",
    }:
        raise ValueError(f"invalid content_origin in data manifest in {root}")
    manifest.setdefault(
        "is_test_fixture", manifest.get("source") == "frontier deterministic smoke fixture v1"
    )
    if not isinstance(manifest["is_test_fixture"], bool):
        raise TypeError(f"is_test_fixture must be boolean in data manifest in {root}")
    train_hash = sha256_file(train_path)
    valid_hash = sha256_file(valid_path)
    if manifest.get("train_sha256") not in (None, train_hash):
        raise ValueError(f"train.txt hash does not match the data manifest in {root}")
    if manifest.get("validation_sha256") not in (None, valid_hash):
        raise ValueError(f"validation.txt hash does not match the data manifest in {root}")
    source_metadata = manifest.get("source_metadata")
    source_metadata_hash = manifest.get("source_metadata_sha256")
    if source_metadata is not None:
        if not isinstance(source_metadata, dict):
            raise TypeError(f"source_metadata must be a JSON object in data manifest in {root}")
        if source_metadata_hash is None and manifest.get("schema_version", 1) < 3:
            pass
        elif source_metadata_hash != _canonical_json_sha256(source_metadata):
            raise ValueError(f"source metadata hash does not match the data manifest in {root}")
    elif source_metadata_hash is not None:
        raise ValueError(f"source metadata hash is present without metadata in {root}")
    preprocessing = manifest.get("preprocessing")
    preprocessing_hash = manifest.get("preprocessing_sha256")
    if manifest.get("schema_version", 1) >= 3 and (
        preprocessing != PREPROCESSING_CONTRACT
        or preprocessing_hash != _canonical_json_sha256(preprocessing)
    ):
        raise ValueError(
            f"unsupported or corrupt preprocessing contract in data manifest in {root}"
        )
    normalized_overlap = {_document_identity(doc) for doc in train_docs}.intersection(
        _document_identity(doc) for doc in valid_docs
    )
    if normalized_overlap:
        raise ValueError(
            f"train/validation contain {len(normalized_overlap)} documents identical after normalization"
        )
    manifest["train_sha256"] = train_hash
    manifest["validation_sha256"] = valid_hash
    manifest.setdefault("preprocessing", PREPROCESSING_CONTRACT)
    return train_docs, valid_docs, manifest


def encode_documents(docs: Sequence[str], tokenizer: Tokenizer) -> list[torch.Tensor]:
    return [torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in docs]


def sample_batch(
    documents: Sequence[torch.Tensor],
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    usable = [doc for doc in documents if doc.numel() >= context_length + 1]
    if not usable:
        shortest = min(int(doc.numel()) for doc in documents)
        raise ValueError(
            f"each split needs at least one document of {context_length + 1} tokens; shortest is {shortest}"
        )
    choices = torch.randint(len(usable), (batch_size,), generator=generator)
    xs, ys = [], []
    for choice in choices.tolist():
        doc = usable[choice]
        offset = int(torch.randint(doc.numel() - context_length, (), generator=generator))
        window = doc[offset : offset + context_length + 1]
        xs.append(window[:-1])
        ys.append(window[1:])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def fixture_documents() -> tuple[list[str], list[str]]:
    """Small deterministic fixture; intentionally not a capability benchmark."""
    train = [
        (
            (
                f"Story {i}: a small fox found a blue stone near the quiet river. "
                f"The fox carried the stone home and told a kind friend. "
            )
            * 5
        ).strip()
        for i in range(24)
    ]
    valid = [
        (
            (
                f"Validation {i}: a curious bird saw a bright leaf beside the old tree. "
                f"It shared the leaf with a gentle rabbit. "
            )
            * 5
        ).strip()
        for i in range(4)
    ]
    return train, valid


def write_fixture(output_dir: str | Path) -> dict:
    root = Path(output_dir)
    _ensure_new_output_directory(root)
    train_docs, valid_docs = fixture_documents()
    train_path = root / "train.txt"
    valid_path = root / "validation.txt"
    train_path.write_text("\n".join(train_docs) + "\n", encoding="utf-8")
    valid_path.write_text("\n".join(valid_docs) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 3,
        "source": "frontier deterministic smoke fixture v1",
        "is_test_fixture": True,
        "content_origin": "synthetic",
        "preprocessing": PREPROCESSING_CONTRACT,
        "preprocessing_sha256": _canonical_json_sha256(PREPROCESSING_CONTRACT),
        "train_document_count": len(train_docs),
        "validation_document_count": len(valid_docs),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(valid_path),
        "train_utf8_bytes": train_path.stat().st_size,
        "validation_utf8_bytes": valid_path.stat().st_size,
        "split_unit": PREPROCESSING_CONTRACT["split_unit"],
    }
    (root / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
