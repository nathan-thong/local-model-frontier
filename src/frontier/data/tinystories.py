"""Prepare a pinned TinyStories prefix and retain source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

END_MARKER = "<|endoftext|>"
DATASET = "roneneldan/TinyStories"
UPSTREAM_FILE = "TinyStories-train.txt"
LICENSE = "cdla-sharing-1.0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_prefix(
    input_path: Path,
    output_path: Path,
    revision: str,
    upstream_size_bytes: int,
    source_start_byte: int = 0,
) -> dict:
    if source_start_byte != 0:
        raise ValueError(
            "TinyStories delimiter processing currently requires a file prefix from byte 0"
        )
    raw = input_path.read_bytes()
    if not raw:
        raise ValueError("source prefix is empty")
    if source_start_byte + len(raw) > upstream_size_bytes:
        raise ValueError("source byte range exceeds the pinned upstream file")
    marker = END_MARKER.encode("ascii")
    last_marker = raw.rfind(marker)
    if last_marker < 0:
        raise ValueError(f"no complete {END_MARKER} delimiter found in source prefix")
    complete_end = last_marker + len(marker)
    complete_text = raw[:complete_end].decode("utf-8")
    stories = complete_text.split(END_MARKER)[:-1]
    documents = [" ".join(story.split()) for story in stories]
    documents = [document for document in documents if document]
    if len(documents) < 2:
        raise ValueError("at least two complete TinyStories documents are required")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(documents) + "\n", encoding="utf-8")
    metadata = {
        "dataset": DATASET,
        "dataset_revision": revision,
        "source_file": UPSTREAM_FILE,
        "source_url": (
            f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{UPSTREAM_FILE}"
        ),
        "license": LICENSE,
        "selected_range_start_byte": source_start_byte,
        "selected_range_end_byte_inclusive": source_start_byte + len(raw) - 1,
        "selected_range_bytes": len(raw),
        "upstream_file_size_bytes": upstream_size_bytes,
        "raw_prefix_sha256": sha256_bytes(raw),
        "complete_document_delimiter": END_MARKER,
        "complete_document_count": len(documents),
        "discarded_trailing_incomplete_bytes": len(raw) - complete_end,
        "text_transform": (
            "Split at the source end-of-text marker, trim each story, and collapse "
            "all Unicode whitespace within each story to one ASCII space."
        ),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".source.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Downloaded source-file prefix")
    parser.add_argument("--output", type=Path, required=True, help="Line-oriented document file")
    parser.add_argument("--revision", required=True, help="Pinned Hugging Face dataset commit SHA")
    parser.add_argument("--upstream-size-bytes", type=int, required=True)
    parser.add_argument("--source-start-byte", type=int, default=0)
    args = parser.parse_args()
    metadata = prepare_prefix(
        args.input,
        args.output,
        args.revision,
        args.upstream_size_bytes,
        args.source_start_byte,
    )
    print(json.dumps(metadata, indent=2))
    return 0
