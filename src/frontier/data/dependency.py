"""Deterministic synthetic dependency tasks used by bounded architecture screens."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DependencyDataset:
    """Tokenized inputs and exact targets for one dependency-task split."""

    input_ids: torch.Tensor
    targets: torch.Tensor
    target_slots: torch.Tensor
    identity_hashes: tuple[str, ...]
    collision_nonces: tuple[int, ...]
    input_ids_sha256: str
    targets_sha256: str
    dataset_sha256: str

    def manifest(self, name: str, seed: int) -> dict[str, Any]:
        return {
            "name": name,
            "seed": seed,
            "examples": int(self.input_ids.shape[0]),
            "input_length": int(self.input_ids.shape[1]),
            "input_ids_sha256": self.input_ids_sha256,
            "targets_sha256": self.targets_sha256,
            "dataset_sha256": self.dataset_sha256,
            "identity_hashes_sha256": hashlib.sha256(
                "\n".join(self.identity_hashes).encode("ascii")
            ).hexdigest(),
            "identity_hashes": list(self.identity_hashes),
            "collision_nonce_max": max(self.collision_nonces, default=0),
            "target_slot_counts": {
                str(slot): int((self.target_slots == slot).sum().item()) for slot in range(4)
            },
        }


def _ranked_candidates(
    *, namespace: str, seed: int, example_index: int, nonce: int, pool: str, ids: list[int]
) -> list[int]:
    ranked = []
    for token_id in ids:
        message = (
            f"S06-DP-001|v1|{namespace}|{seed}|{example_index}|{nonce}|{pool}|{token_id}"
        ).encode()
        ranked.append((hashlib.sha256(message).digest(), token_id))
    return [token_id for _, token_id in sorted(ranked)]


def _input_identity(tokens: list[int]) -> str:
    packed = struct.pack(f"<{len(tokens)}H", *tokens)
    return hashlib.sha256(packed).hexdigest()


def _dataset_from_rows(
    rows: list[list[int]],
    targets: list[int],
    slots: list[int],
    identities: list[str],
    nonces: list[int],
) -> DependencyDataset:
    input_ids = torch.tensor(rows, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    slot_tensor = torch.tensor(slots, dtype=torch.long)
    packed_inputs = b"".join(struct.pack(f"<{len(row)}H", *row) for row in rows)
    packed_targets = b"".join(struct.pack("<H", target) for target in targets)
    packed_examples = b"".join(
        struct.pack(f"<{len(row) + 1}H", *row, target)
        for row, target in zip(rows, targets, strict=True)
    )
    return DependencyDataset(
        input_ids=input_ids,
        targets=target_tensor,
        target_slots=slot_tensor,
        identity_hashes=tuple(identities),
        collision_nonces=tuple(nonces),
        input_ids_sha256=hashlib.sha256(packed_inputs).hexdigest(),
        targets_sha256=hashlib.sha256(packed_targets).hexdigest(),
        dataset_sha256=hashlib.sha256(packed_examples).hexdigest(),
    )


def _generate_split(
    *,
    task: dict[str, Any],
    namespace: str,
    seed: int,
    count: int,
    used_identities: set[str],
) -> DependencyDataset:
    pair_count = task["pair_count"]
    rows: list[list[int]] = []
    targets: list[int] = []
    slots: list[int] = []
    identities: list[str] = []
    nonces: list[int] = []
    for example_index in range(count):
        target_slot = example_index % pair_count
        nonce = 0
        while True:
            keys = _ranked_candidates(
                namespace=namespace,
                seed=seed,
                example_index=example_index,
                nonce=nonce,
                pool="key",
                ids=task["key_token_ids"],
            )[:pair_count]
            values = _ranked_candidates(
                namespace=namespace,
                seed=seed,
                example_index=example_index,
                nonce=nonce,
                pool="value",
                ids=task["value_token_ids"],
            )[:pair_count]
            row = [task["bos_token_id"]]
            for key, value in zip(keys, values, strict=True):
                row.extend((key, value))
            row.extend((task["query_marker_token_id"], keys[target_slot]))
            identity = _input_identity(row)
            if identity not in used_identities:
                break
            nonce += 1
        used_identities.add(identity)
        rows.append(row)
        targets.append(values[target_slot])
        slots.append(target_slot)
        identities.append(identity)
        nonces.append(nonce)
    return _dataset_from_rows(rows, targets, slots, identities, nonces)


def generate_s06_dependency_splits(
    config: dict[str, Any],
) -> tuple[dict[str, DependencyDataset], dict[str, Any]]:
    """Generate the frozen train/evaluation identities in preregistered order."""
    task = config["task"]
    seeds = config["seeds"]
    train_count = task["training_examples_per_seed"]
    eval_count = task["evaluation_examples_total"]
    pair_count = task["pair_count"]
    if task["input_length"] != 2 * pair_count + 3:
        raise ValueError(
            "four-pair task input length must include BOS, pairs, query marker, and key"
        )
    if train_count != task["training_examples_per_target_slot_per_seed"] * pair_count:
        raise ValueError("training example count must match its balanced target-slot counts")
    if eval_count != task["evaluation_examples_per_target_slot"] * pair_count:
        raise ValueError("evaluation count must match its balanced target-slot counts")
    if train_count % pair_count or eval_count % pair_count:
        raise ValueError("training and evaluation example counts must balance target slots")
    used_identities: set[str] = set()
    datasets: dict[str, DependencyDataset] = {}
    manifests = []
    for seed in seeds:
        name = f"training_seed_{seed}"
        dataset = _generate_split(
            task=task,
            namespace="training",
            seed=seed,
            count=train_count,
            used_identities=used_identities,
        )
        datasets[name] = dataset
        manifests.append(dataset.manifest(name, seed))
    eval_seed = task["evaluation_seed"]
    evaluation = _generate_split(
        task=task,
        namespace="evaluation",
        seed=eval_seed,
        count=eval_count,
        used_identities=used_identities,
    )
    datasets["evaluation"] = evaluation
    manifests.append(evaluation.manifest("evaluation", eval_seed))
    return datasets, {
        "algorithm": task["example_generation"]["algorithm"],
        "sequence_identity_policy": task["sequence_identity_policy"],
        "generation_order": [*datasets.keys()],
        "unique_identities_total": len(used_identities),
        "splits": manifests,
    }
