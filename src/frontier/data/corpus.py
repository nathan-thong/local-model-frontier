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

PREPROCESSING_CONTRACT_V2 = {
    **PREPROCESSING_CONTRACT,
    "schema_version": 2,
    "duplicate_policy": "retain every source row; assign normalized duplicates to one split",
    "identity_record": "SHA-256 of normalized UTF-8 document identity",
    "split_allocation": "reserve one identity per split, then largest-remainder allocation",
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


def _identity_sha256(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _allocate_three_way_counts(
    unique_count: int, validation_fraction: float, test_fraction: float
) -> tuple[int, int, int]:
    if unique_count < 3:
        raise ValueError("at least three unique documents are required for train/validation/test")
    fractions = (1.0 - validation_fraction - test_fraction, validation_fraction, test_fraction)
    remaining = unique_count - 3
    exact = tuple(remaining * fraction for fraction in fractions)
    counts = [1 + int(value) for value in exact]
    unallocated = unique_count - sum(counts)
    order = sorted(range(3), key=lambda index: (-(exact[index] % 1), index))
    for index in order[:unallocated]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


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
    test_fraction: float = 0.0,
) -> dict:
    source = Path(input_path)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be zero or strictly between 0 and 1")
    if test_fraction and validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction plus test_fraction must be less than 1")
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
    if test_fraction:
        train_count, valid_count, test_count = _allocate_three_way_counts(
            len(unique_docs), validation_fraction, test_fraction
        )
        train_set = set(unique_docs[:train_count])
        valid_set = set(unique_docs[train_count : train_count + valid_count])
        test_set = set(
            unique_docs[train_count + valid_count : train_count + valid_count + test_count]
        )
        preprocessing = PREPROCESSING_CONTRACT_V2
    else:
        valid_count = max(
            1, min(len(unique_docs) - 1, round(len(unique_docs) * validation_fraction))
        )
        valid_set = set(unique_docs[:valid_count])
        train_set = set(unique_docs[valid_count:])
        test_set = set()
        preprocessing = PREPROCESSING_CONTRACT
    # Normalized duplicates stay together while retaining the source text verbatim.
    train_docs = [doc for doc in docs if _document_identity(doc) in train_set]
    valid_docs = [doc for doc in docs if _document_identity(doc) in valid_set]
    test_docs = [doc for doc in docs if _document_identity(doc) in test_set]

    output = Path(output_dir)
    _ensure_new_output_directory(output)
    train_path = output / "train.txt"
    valid_path = output / "validation.txt"
    train_path.write_bytes(("\n".join(train_docs) + "\n").encode("utf-8"))
    valid_path.write_bytes(("\n".join(valid_docs) + "\n").encode("utf-8"))
    if test_fraction:
        test_path = output / "test.txt"
        test_path.write_bytes(("\n".join(test_docs) + "\n").encode("utf-8"))
    identity_sets = {
        "train": train_set,
        "validation": valid_set,
        "test": test_set,
    }
    split_identity_sha256 = {
        name: sorted(_identity_sha256(identity) for identity in identities)
        for name, identities in identity_sets.items()
    }
    document_frequency: dict[str, int] = {}
    for document in docs:
        identity = _document_identity(document)
        document_frequency[identity] = document_frequency.get(identity, 0) + 1
    duplicate_groups = sorted(
        (
            {
                "identity_sha256": _identity_sha256(identity),
                "source_row_count": count,
            }
            for identity, count in document_frequency.items()
            if count > 1
        ),
        key=lambda row: row["identity_sha256"],
    )
    manifest = {
        "schema_version": 4 if test_fraction else 3,
        "is_test_fixture": False,
        "content_origin": origin,
        "preprocessing": preprocessing,
        "preprocessing_sha256": _canonical_json_sha256(preprocessing),
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
        "split_unit": preprocessing["split_unit"],
    }
    if test_fraction:
        manifest.update(
            {
                "test_fraction_requested": test_fraction,
                "test_document_count": len(test_docs),
                "test_unique_document_count": len(test_set),
                "test_sha256": sha256_file(test_path),
                "test_utf8_bytes": test_path.stat().st_size,
                "train_unique_document_count": len(train_set),
                "validation_unique_document_count": len(valid_set),
                "split_identity_sha256": split_identity_sha256,
                "duplicate_report": {
                    "policy": PREPROCESSING_CONTRACT_V2["duplicate_policy"],
                    "duplicate_group_count": len(duplicate_groups),
                    "duplicate_row_count": sum(
                        row["source_row_count"] - 1 for row in duplicate_groups
                    ),
                    "groups": duplicate_groups,
                },
            }
        )
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
    schema_version = manifest.get("schema_version", 1)
    expected_preprocessing = (
        PREPROCESSING_CONTRACT_V2 if schema_version >= 4 else PREPROCESSING_CONTRACT
    )
    if schema_version >= 3 and (
        preprocessing != expected_preprocessing
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
    if schema_version >= 4:
        test_path = root / "test.txt"
        if not test_path.is_file():
            raise FileNotFoundError(f"expected test.txt for schema-4 data manifest in {root}")
        test_hash = sha256_file(test_path)
        if manifest.get("test_sha256") != test_hash:
            raise ValueError(f"test.txt hash does not match the data manifest in {root}")
        if manifest.get("test_utf8_bytes") != test_path.stat().st_size:
            raise ValueError(f"test.txt byte count does not match the data manifest in {root}")
        split_identities = manifest.get("split_identity_sha256")
        if not isinstance(split_identities, dict) or set(split_identities) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"split identity hashes are missing or malformed in {root}")
        for split_name, identity_hashes in split_identities.items():
            if (
                not isinstance(identity_hashes, list)
                or any(
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    for digest in identity_hashes
                )
                or len(identity_hashes) != len(set(identity_hashes))
            ):
                raise ValueError(f"invalid {split_name} identity hashes in {root}")
        identity_sets = {
            split_name: set(identity_hashes)
            for split_name, identity_hashes in split_identities.items()
        }
        if (
            identity_sets["train"].intersection(identity_sets["validation"])
            or identity_sets["train"].intersection(identity_sets["test"])
            or identity_sets["validation"].intersection(identity_sets["test"])
        ):
            raise ValueError(f"normalized duplicate identities cross splits in {root}")
        for split_name, documents in (("train", train_docs), ("validation", valid_docs)):
            actual_hashes = sorted(
                {_identity_sha256(_document_identity(document)) for document in documents}
            )
            if actual_hashes != split_identities[split_name]:
                raise ValueError(
                    f"{split_name} identities do not match the data manifest in {root}"
                )
        for split_name, documents in (("train", train_docs), ("validation", valid_docs)):
            if manifest.get(f"{split_name}_document_count") != len(documents):
                raise ValueError(
                    f"{split_name} document count does not match the data manifest in {root}"
                )
            if manifest.get(f"{split_name}_unique_document_count") != len(
                identity_sets[split_name]
            ):
                raise ValueError(
                    f"{split_name} unique document count does not match the data manifest in {root}"
                )
        test_doc_count = manifest.get("test_document_count")
        test_unique_count = manifest.get("test_unique_document_count")
        if not isinstance(test_doc_count, int) or test_doc_count < 1:
            raise ValueError(f"invalid test document count in data manifest in {root}")
        if test_unique_count != len(identity_sets["test"]) or test_unique_count < 1:
            raise ValueError(f"invalid test unique document count in data manifest in {root}")
    manifest["train_sha256"] = train_hash
    manifest["validation_sha256"] = valid_hash
    manifest.setdefault("preprocessing", expected_preprocessing)
    return train_docs, valid_docs, manifest


def load_all_splits(
    data_dir: str | Path,
) -> tuple[list[str], list[str], list[str] | None, dict]:
    """Load all splits, including the sealed test set when the manifest declares one."""
    root = Path(data_dir)
    train_docs, valid_docs, manifest = load_split(root)
    if manifest.get("schema_version", 1) < 4:
        return train_docs, valid_docs, None, manifest

    test_docs = read_documents(root / "test.txt")
    expected_hashes = manifest["split_identity_sha256"]["test"]
    actual_hashes = sorted(
        {_identity_sha256(_document_identity(document)) for document in test_docs}
    )
    if actual_hashes != expected_hashes:
        raise ValueError(f"test identities do not match the data manifest in {root}")
    if len(test_docs) != manifest["test_document_count"]:
        raise ValueError(f"test document count does not match the data manifest in {root}")
    all_docs = train_docs + valid_docs + test_docs
    if len(all_docs) != manifest.get("document_count"):
        raise ValueError(f"total document count does not match the data manifest in {root}")
    document_frequency: dict[str, int] = {}
    for document in all_docs:
        identity = _document_identity(document)
        document_frequency[identity] = document_frequency.get(identity, 0) + 1
    duplicate_groups = sorted(
        (
            {
                "identity_sha256": _identity_sha256(identity),
                "source_row_count": count,
            }
            for identity, count in document_frequency.items()
            if count > 1
        ),
        key=lambda row: row["identity_sha256"],
    )
    if len(document_frequency) != manifest.get("unique_document_count"):
        raise ValueError(f"total unique document count does not match the data manifest in {root}")
    expected_duplicate_report = manifest.get("duplicate_report")
    if not isinstance(expected_duplicate_report, dict) or expected_duplicate_report != {
        "policy": PREPROCESSING_CONTRACT_V2["duplicate_policy"],
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_row_count": sum(row["source_row_count"] - 1 for row in duplicate_groups),
        "groups": duplicate_groups,
    }:
        raise ValueError(f"duplicate report does not match the data splits in {root}")
    return train_docs, valid_docs, test_docs, manifest


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
    train_path.write_bytes(("\n".join(train_docs) + "\n").encode("utf-8"))
    valid_path.write_bytes(("\n".join(valid_docs) + "\n").encode("utf-8"))
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
    (root / "data_manifest.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )
    return manifest
