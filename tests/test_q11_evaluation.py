import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from frontier.data.corpus import prepare_split
from frontier.evaluation.contamination import audit_task_contamination
from frontier.evaluation.tasks import _score_exact_match, evaluate_tasks, read_tasks
from frontier.tokenization import ByteTokenizer


class UniformTaskModel(nn.Module):
    def __init__(self, vocab_size: int = 259, max_seq_len: int = 32):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(max_seq_len=max_seq_len)

    def forward(self, input_ids, cache=None, use_cache=False):
        logits = self.anchor + torch.zeros(*input_ids.shape, self.vocab_size)
        return logits, None


def test_q11_task_artifact_matches_frozen_contract():
    contract_path = Path("docs/q11_task_contract_v1.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tasks_path = Path(contract["task_set"]["path"])
    raw = tasks_path.read_bytes()
    tasks = read_tasks(tasks_path)

    assert hashlib.sha256(raw).hexdigest() == contract["task_set"]["sha256"]
    assert len(tasks) == contract["task_set"]["example_count"] == 4
    assert [row["id"] for row in tasks] == [row["id"] for row in contract["hand_checked_examples"]]
    assert [row["target"] for row in tasks] == [
        row["expected_target"] for row in contract["hand_checked_examples"]
    ]


def test_exact_match_scorer_obeys_frozen_case_and_whitespace_rules():
    assert _score_exact_match("qom", "qom") is True
    assert _score_exact_match("  qom\n", "qom") is True
    assert _score_exact_match("QOM", "qom") is False
    assert _score_exact_match("qom.", "qom") is False


def test_task_nll_uses_utf8_target_bytes_and_excludes_eos_from_denominator(tmp_path):
    path = tmp_path / "task.jsonl"
    path.write_text('{"id":"byte-check","prompt":"A","target":"B"}\n', encoding="utf-8")
    model = UniformTaskModel()
    tokenizer = ByteTokenizer()

    result = evaluate_tasks(model, tokenizer, path, max_new_tokens=4)
    repeated = evaluate_tasks(model, tokenizer, path, max_new_tokens=4)
    row = result["details"][0]
    expected_nll = 2 * math.log(tokenizer.vocab_size)

    assert result == repeated
    assert result["conditional_scored_tokens"] == 2  # One target byte and EOS.
    assert result["conditional_scored_bytes"] == 1
    assert row["conditional_scored_bytes"] == len(b"B")
    assert row["conditional_nll_sum"] == pytest.approx(expected_nll, rel=1e-7, abs=1e-7)
    assert row["conditional_bits_per_utf8_byte"] == pytest.approx(
        expected_nll / math.log(2), rel=1e-7, abs=1e-7
    )
    assert result["conditional_bits_per_utf8_byte"] == pytest.approx(
        expected_nll / math.log(2), rel=1e-7, abs=1e-7
    )
    assert result["protocol"]["byte_normalization"].startswith("conditional target NLL")
    assert model.training


def test_task_contamination_audit_detects_full_identity_and_long_phrase_without_text_output(
    tmp_path,
):
    documents = [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron",
        "separate document with unrelated words and content",
        "third article holds another set of words for training",
        "fourth item contains distinct language for validation",
        "fifth entry uses additional tokens for a sealed test",
        "sixth final source document has unique content here",
    ]
    source = tmp_path / "corpus.txt"
    source.write_text("\n".join(documents) + "\n", encoding="utf-8")
    data_dir = tmp_path / "prepared"
    prepare_split(source, data_dir, validation_fraction=0.2, seed=17, test_fraction=0.2)
    tasks_path = tmp_path / "tasks.jsonl"
    tasks = [
        {
            "id": "exact-identity",
            "prompt": documents[0],
            "target": "answer",
        },
        {
            "id": "long-phrase",
            "prompt": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu novel",
            "target": "other",
        },
    ]
    tasks_path.write_text("".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8")

    report = audit_task_contamination(tasks_path, data_dir, ngram_size=13)
    repeated = audit_task_contamination(tasks_path, data_dir, ngram_size=13)
    serialized = json.dumps(report)

    assert report == repeated
    assert report["status"] == "contamination_found"
    assert report["source"]["scanned_document_count"] == 6
    assert report["per_example"][0]["exact_identity_matches_by_split"]
    assert sum(report["per_example"][1]["shared_ngram_counts_by_split"].values()) > 0
    assert documents[0] not in serialized
