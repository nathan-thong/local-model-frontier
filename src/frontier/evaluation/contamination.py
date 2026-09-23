"""Deterministic checks for exact task identity and long phrase reuse in corpus splits."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

from frontier.data.corpus import load_all_splits

from .tasks import read_tasks


def _normalize_identity(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_ngram_hashes(text: str, ngram_size: int) -> set[str]:
    tokens = _normalize_identity(text).casefold().split()
    return {
        _sha256(" ".join(tokens[start : start + ngram_size]))
        for start in range(max(0, len(tokens) - ngram_size + 1))
    }


def audit_task_contamination(
    task_path: str | Path,
    data_dir: str | Path,
    *,
    ngram_size: int = 13,
) -> dict:
    """Compare task strings to every split; return hashes and counts, never source text."""
    if not isinstance(ngram_size, int) or isinstance(ngram_size, bool) or ngram_size < 1:
        raise ValueError("ngram_size must be a positive integer")
    train, validation, test, manifest = load_all_splits(data_dir)
    if test is None:
        raise ValueError("task contamination audit requires a schema-4 train/validation/test split")
    documents_by_split = {"train": train, "validation": validation, "test": test}
    identities_by_split: dict[str, set[str]] = {}
    ngrams_by_split: dict[str, set[str]] = {}
    for split, documents in documents_by_split.items():
        identities_by_split[split] = {
            _sha256(_normalize_identity(document)) for document in documents
        }
        ngrams_by_split[split] = set()
        for document in documents:
            ngrams_by_split[split].update(_word_ngram_hashes(document, ngram_size))

    tasks = read_tasks(task_path)
    per_example = []
    total_exact_matches = 0
    total_phrase_matches = 0
    for task in tasks:
        task_strings = {
            "prompt": task["prompt"],
            "target": task["target"],
            "prompt_target": f"{task['prompt']} {task['target']}",
        }
        normalized_hashes = {
            name: _sha256(_normalize_identity(value)) for name, value in task_strings.items()
        }
        task_ngram_hashes: set[str] = set()
        for value in task_strings.values():
            task_ngram_hashes.update(_word_ngram_hashes(value, ngram_size))
        exact_matches = {
            split: sorted(
                name
                for name, identity_hash in normalized_hashes.items()
                if identity_hash in identities_by_split[split]
            )
            for split in documents_by_split
        }
        shared_ngram_counts = {
            split: len(task_ngram_hashes & ngrams_by_split[split]) for split in documents_by_split
        }
        exact_count = sum(len(names) for names in exact_matches.values())
        phrase_count = sum(shared_ngram_counts.values())
        total_exact_matches += exact_count
        total_phrase_matches += phrase_count
        per_example.append(
            {
                "example_id": task["id"],
                "task_string_sha256": normalized_hashes,
                "exact_identity_matches_by_split": exact_matches,
                "shared_ngram_counts_by_split": shared_ngram_counts,
                "unique_task_ngram_count": len(task_ngram_hashes),
            }
        )

    return {
        "schema_version": 1,
        "status": "passed"
        if total_exact_matches == 0 and total_phrase_matches == 0
        else "contamination_found",
        "task_file_sha256": hashlib.sha256(Path(task_path).read_bytes()).hexdigest(),
        "ngram_size_tokens": ngram_size,
        "normalization": {
            "exact_identity": "Unicode NFC followed by collapsing Unicode whitespace to ASCII spaces",
            "phrase_tokens": "normalized text, casefolded, then split on whitespace",
            "matching": "exact full identity or contiguous token n-gram; only SHA-256 values and counts are returned",
        },
        "source": {
            "data_manifest_sha256": hashlib.sha256(
                (Path(data_dir) / "data_manifest.json").read_bytes()
            ).hexdigest(),
            "split_content_sha256": {
                split: manifest[f"{split}_sha256"] for split in documents_by_split
            },
            "unique_document_counts": {
                split: len(identities_by_split[split]) for split in documents_by_split
            },
            "scanned_document_count": sum(
                len(identities) for identities in identities_by_split.values()
            ),
        },
        "task_count": len(tasks),
        "exact_identity_match_count": total_exact_matches,
        "shared_ngram_match_count": total_phrase_matches,
        "per_example": per_example,
    }
