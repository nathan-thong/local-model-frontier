"""Corpus loading, deterministic document splitting and seeded sampling."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

import torch

from frontier.tokenization import Tokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> dict:
    source = Path(input_path)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    docs = read_documents(source)
    unique_docs = list(dict.fromkeys(docs))
    if len(unique_docs) < 2:
        raise ValueError("at least two non-empty documents are required for a split")
    random.Random(seed).shuffle(unique_docs)
    valid_count = max(1, min(len(unique_docs) - 1, round(len(unique_docs) * validation_fraction)))
    valid_set = set(unique_docs[:valid_count])
    # Identical documents stay in one split so repeated lines cannot leak across the boundary.
    train_docs = [doc for doc in docs if doc not in valid_set]
    valid_docs = [doc for doc in docs if doc in valid_set]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.txt"
    valid_path = output / "validation.txt"
    train_path.write_text("\n".join(train_docs) + "\n", encoding="utf-8")
    valid_path.write_text("\n".join(valid_docs) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
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
        "split_unit": "document; lines are independent documents",
    }
    if source_metadata is not None:
        manifest["source"] = source_metadata.get("dataset", source.name)
        manifest["source_metadata"] = source_metadata
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
    train_hash = sha256_file(train_path)
    valid_hash = sha256_file(valid_path)
    if manifest.get("train_sha256") not in (None, train_hash):
        raise ValueError(f"train.txt hash does not match the data manifest in {root}")
    if manifest.get("validation_sha256") not in (None, valid_hash):
        raise ValueError(f"validation.txt hash does not match the data manifest in {root}")
    manifest["train_sha256"] = train_hash
    manifest["validation_sha256"] = valid_hash
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
    root.mkdir(parents=True, exist_ok=True)
    train_docs, valid_docs = fixture_documents()
    train_path = root / "train.txt"
    valid_path = root / "validation.txt"
    train_path.write_text("\n".join(train_docs) + "\n", encoding="utf-8")
    valid_path.write_text("\n".join(valid_docs) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source": "frontier deterministic smoke fixture v1",
        "train_document_count": len(train_docs),
        "validation_document_count": len(valid_docs),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(valid_path),
        "split_unit": "document",
    }
    (root / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
