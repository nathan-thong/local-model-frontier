import hashlib
import json
import unicodedata

import pytest
import torch
from torch import nn

from frontier.cli import main
from frontier.config import ModelConfig
from frontier.data.corpus import (
    encode_documents,
    load_all_splits,
    load_split,
    prepare_split,
    read_documents,
    sample_batch,
    write_fixture,
)
from frontier.data.tinystories import prepare_prefix
from frontier.evaluation.perplexity import evaluate_perplexity
from frontier.evaluation.tasks import evaluate_tasks
from frontier.models import DecoderLanguageModel
from frontier.tokenization import (
    ByteTokenizer,
    load_tokenizer_artifact,
    tokenizer_artifact,
)


def test_prepare_data_cli_records_verified_content_origin(tmp_path, capsys):
    source = tmp_path / "corpus.txt"
    source.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    output = tmp_path / "prepared"

    status = main(
        [
            "prepare-data",
            "--input",
            str(source),
            "--output",
            str(output),
            "--validation-fraction",
            "0.2",
            "--test-fraction",
            "0.2",
            "--content-origin",
            "human",
        ]
    )

    manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
    assert status == 0
    assert manifest["content_origin"] == "human"
    assert manifest["schema_version"] == 4
    assert (output / "test.txt").is_file()
    assert json.loads(capsys.readouterr().out)["content_origin"] == "human"


def test_byte_tokenizer_round_trips_utf8_and_special_ids():
    tokenizer = ByteTokenizer()
    text = "naïve 🐇"
    ids = tokenizer.encode(text)
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_size == 259


def test_tokenizer_artifact_round_trips_and_detects_tampering(tmp_path):
    tokenizer = ByteTokenizer()
    artifact = tokenizer_artifact(tokenizer)
    artifact_path = tmp_path / "tokenizer.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = load_tokenizer_artifact(artifact_path)

    assert type(loaded) is ByteTokenizer
    assert loaded.to_dict() == tokenizer.to_dict()
    artifact["vocab_size"] += 1
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_tokenizer_artifact(artifact_path)


def test_document_split_is_repeatable_and_disjoint(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("alpha\nbeta\nalpha\ngamma\ndelta\n", encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    source_metadata = {"dataset": "fixture-corpus", "license": "test-only"}
    one = prepare_split(source, first, 0.25, 9, source_metadata)
    two = prepare_split(source, second, 0.25, 9, source_metadata)
    assert one["train_sha256"] == two["train_sha256"]
    assert one["source"] == "fixture-corpus"
    assert one["source_metadata"] == {**source_metadata, "content_origin": "unknown"}
    assert one["content_origin"] == "unknown"
    assert one["is_test_fixture"] is False
    train = read_documents(first / "train.txt")
    valid = read_documents(first / "validation.txt")
    assert set(train).isdisjoint(valid)
    assert ("alpha" in train) == ("alpha" not in valid)
    assert one["source_metadata_sha256"] == two["source_metadata_sha256"]
    assert one["preprocessing"]["schema_version"] == 1
    assert one["train_utf8_bytes"] == (first / "train.txt").stat().st_size
    assert b"\r" not in (first / "train.txt").read_bytes()
    assert b"\r" not in (first / "validation.txt").read_bytes()
    assert (first / "train.txt").read_bytes() == (second / "train.txt").read_bytes()
    assert (first / "validation.txt").read_bytes() == (second / "validation.txt").read_bytes()


def test_three_way_split_freezes_identities_and_duplicate_report(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text(
        "café\ttext\ncafe\u0301 text\nother one\nthird row\nfourth row\nfifth row\nsixth row\nseventh row\n",
        encoding="utf-8",
    )
    first = tmp_path / "first-three-way"
    second = tmp_path / "second-three-way"
    metadata = {"dataset": "bounded-fixture", "license": "test-only"}

    one = prepare_split(source, first, 0.2, 29, metadata, test_fraction=0.2)
    two = prepare_split(source, second, 0.2, 29, metadata, test_fraction=0.2)
    train, valid, test, loaded = load_all_splits(first)

    assert one == two
    assert one["schema_version"] == 4
    assert one["train_unique_document_count"] == 3
    assert one["validation_unique_document_count"] == 2
    assert one["test_unique_document_count"] == 2
    assert one["duplicate_report"]["duplicate_group_count"] == 1
    assert one["duplicate_report"]["duplicate_row_count"] == 1
    assert one["duplicate_report"]["groups"][0]["source_row_count"] == 2
    assert train == read_documents(first / "train.txt")
    assert valid == read_documents(first / "validation.txt")
    assert test == read_documents(first / "test.txt")
    assert loaded["split_identity_sha256"] == one["split_identity_sha256"]
    identity_sets = [
        set(one["split_identity_sha256"][key]) for key in ("train", "validation", "test")
    ]
    assert not (identity_sets[0] & identity_sets[1])
    assert not (identity_sets[0] & identity_sets[2])
    assert not (identity_sets[1] & identity_sets[2])
    for split in ("train.txt", "validation.txt", "test.txt"):
        assert b"\r" not in (first / split).read_bytes()
        assert (first / split).read_bytes() == (second / split).read_bytes()


def test_three_way_loader_detects_test_content_not_matching_frozen_identity(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\n", encoding="utf-8")
    output = tmp_path / "prepared"
    prepare_split(source, output, 0.2, 7, test_fraction=0.2)
    test_path = output / "test.txt"
    test_path.write_text("altered held out content\n", encoding="utf-8")
    manifest_path = output / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["test_sha256"] = hashlib.sha256(test_path.read_bytes()).hexdigest()
    manifest["test_utf8_bytes"] = test_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="test identities do not match"):
        load_all_splits(output)


def test_three_way_split_requires_room_for_each_split(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("alpha\nbeta\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least three unique documents"):
        prepare_split(source, tmp_path / "too-small", 0.2, 1, test_fraction=0.2)

    source.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be less than 1"):
        prepare_split(source, tmp_path / "invalid-fractions", 0.5, 1, test_fraction=0.5)


def test_normalized_duplicate_documents_cannot_cross_splits(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("caf\u00e9\ttext\ncafe\u0301 text\nother one\nlast row\n", encoding="utf-8")

    manifest = prepare_split(source, tmp_path / "prepared", 0.5, 12)
    train, valid, _ = load_split(tmp_path / "prepared")

    assert manifest["unique_document_count"] == 3
    train_normalized = {unicodedata.normalize("NFC", " ".join(doc.split())) for doc in train}
    valid_normalized = {unicodedata.normalize("NFC", " ".join(doc.split())) for doc in valid}
    assert not train_normalized.intersection(valid_normalized)


def test_loader_rejects_normalized_duplicate_leakage(tmp_path):
    root = tmp_path / "leaky"
    root.mkdir()
    (root / "train.txt").write_text("caf\u00e9 text\ntrain row\n", encoding="utf-8")
    (root / "validation.txt").write_text("cafe\u0301\ttext\nvalid row\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identical after normalization"):
        load_split(root)


def test_content_origin_is_explicit_and_propagates_from_source_metadata(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    metadata = {
        "dataset": "tiny-stories-copy",
        "license": "test-only",
        "content_origin": "synthetic",
    }

    manifest = prepare_split(source, tmp_path / "prepared", 0.25, 3, metadata)
    train, valid, loaded = load_split(tmp_path / "prepared")

    assert manifest["content_origin"] == "synthetic"
    assert loaded["content_origin"] == "synthetic"
    assert loaded["is_test_fixture"] is False
    assert set(train).isdisjoint(valid)


def test_legacy_manifest_does_not_infer_content_origin_from_dataset_name(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "train.txt").write_text("train A\ntrain B\n", encoding="utf-8")
    (root / "validation.txt").write_text("held out\n", encoding="utf-8")
    (root / "data_manifest.json").write_text(
        json.dumps({"schema_version": 1, "source": "roneneldan/TinyStories"}), encoding="utf-8"
    )

    _, _, manifest = load_split(root)

    assert manifest["content_origin"] == "unknown"
    assert manifest["is_test_fixture"] is False


def test_builtin_smoke_data_is_both_a_test_fixture_and_synthetic(tmp_path):
    manifest = write_fixture(tmp_path / "fixture")

    assert manifest["is_test_fixture"] is True
    assert manifest["content_origin"] == "synthetic"


def test_data_preparation_refuses_to_overwrite_existing_corpus(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()
    manifest = output / "data_manifest.json"
    manifest.write_text('{"schema_version": 1}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_split(source, output, 0.25, 3)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_fixture(output)

    assert [path.name for path in output.iterdir()] == ["data_manifest.json"]
    assert manifest.read_text(encoding="utf-8") == '{"schema_version": 1}\n'


def test_content_origin_conflict_is_rejected_before_writing_split(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="conflicts"):
        prepare_split(
            source,
            output,
            0.25,
            3,
            {"dataset": "test", "content_origin": "human"},
            content_origin="synthetic",
        )

    assert not output.exists()


def test_tinystories_prefix_keeps_only_complete_documents_and_records_provenance(tmp_path):
    raw = b"One story\n<|endoftext|>\nTwo story<|endoftext|>\nCut off"
    source = tmp_path / "source.raw"
    output = tmp_path / "documents.txt"
    source.write_bytes(raw)

    metadata = prepare_prefix(source, output, "commit-sha", len(raw) + 100)

    assert read_documents(output) == ["One story", "Two story"]
    assert metadata["complete_document_count"] == 2
    assert metadata["content_origin"] == "synthetic"
    assert metadata["discarded_trailing_incomplete_bytes"] == len(b"\nCut off")
    assert metadata["selected_range_bytes"] == len(raw)
    assert json.loads(output.with_suffix(".txt.source.json").read_text()) == metadata


def test_tinystories_prefix_refuses_to_overwrite_input_or_prepared_output(tmp_path):
    raw = b"One story<|endoftext|>\nTwo story<|endoftext|>"
    source = tmp_path / "source.raw"
    output = tmp_path / "documents.txt"
    source.write_bytes(raw)

    with pytest.raises(ValueError, match="must not overwrite"):
        prepare_prefix(source, source, "commit-sha", len(raw))

    output.write_text("preserve this", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_prefix(source, output, "commit-sha", len(raw))

    assert source.read_bytes() == raw
    assert output.read_text(encoding="utf-8") == "preserve this"


def test_sampled_batch_targets_are_next_tokens_and_stay_in_one_document():
    tokenizer = ByteTokenizer()
    docs = encode_documents(["abcdefghijk", "mnopqrstuvw"], tokenizer)
    generator = torch.Generator().manual_seed(12)
    x, y = sample_batch(
        docs, batch_size=4, context_length=5, generator=generator, device=torch.device("cpu")
    )
    assert x.shape == y.shape == (4, 5)
    assert torch.equal(x[:, 1:], y[:, :-1])


class UniformModel(nn.Module):
    def __init__(self, vocab_size=259):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        logits = self.anchor + torch.zeros(*input_ids.shape, self.vocab_size)
        return logits, None


def test_perplexity_scores_each_document_target_once_and_token_weights():
    docs = [torch.tensor([256, 3, 4, 257]), torch.tensor([256, 5, 257])]
    model = UniformModel()
    metrics = evaluate_perplexity(
        model,
        docs,
        context_length=3,
        stride=2,
        token_byte_counts=[[0, 1, 1, 0], [0, 1, 0]],
    )
    assert metrics["scored_tokens"] == 5
    assert abs(metrics["mean_nll"] - torch.log(torch.tensor(259.0)).item()) < 1e-6
    assert abs(metrics["perplexity"] - 259.0) < 1e-3
    assert [row["scored_tokens"] for row in metrics["per_document"]] == [3, 2]
    assert metrics["scored_utf8_bytes"] == 3
    assert metrics["content_nll_sum"] == pytest.approx(3 * torch.log(torch.tensor(259.0)).item())
    assert metrics["bits_per_byte"] == pytest.approx(torch.log2(torch.tensor(259.0)).item())
    assert model.training


def test_perplexity_scores_long_document_targets_once_with_bounded_context():
    document = torch.tensor([256, 1, 2, 3, 4, 5, 6, 7, 257])
    metrics = evaluate_perplexity(
        UniformModel(),
        [document],
        context_length=3,
        stride=2,
        token_byte_counts=[[0, 1, 1, 1, 1, 1, 1, 1, 0]],
    )

    assert metrics["scored_tokens"] == 8
    assert metrics["per_document"][0]["scored_tokens"] == 8
    assert metrics["scored_utf8_bytes"] == 7
    assert metrics["per_document"][0]["status"] == "scored"


def test_perplexity_reports_documents_without_targets_and_rejects_all_empty_selection():
    short = torch.tensor([256])
    valid = torch.tensor([256, 1, 257])
    result = evaluate_perplexity(UniformModel(), [short, valid], context_length=2, stride=1)

    assert result["per_document"][0]["status"] == "no_next_token_targets"
    assert result["per_document"][0]["perplexity"] is None
    with pytest.raises(ValueError, match="no next-token targets"):
        evaluate_perplexity(UniformModel(), [short], context_length=2, stride=1)


def test_perplexity_restores_model_mode_after_failure_and_rejects_nonfinite_nll():
    class FailingModel(UniformModel):
        def forward(self, input_ids):
            raise RuntimeError("model failure")

    model = FailingModel()
    with pytest.raises(RuntimeError, match="model failure"):
        evaluate_perplexity(model, [torch.tensor([256, 1])], context_length=2, stride=1)
    assert model.training

    class NonFiniteModel(UniformModel):
        def forward(self, input_ids):
            logits = torch.full((*input_ids.shape, self.vocab_size), float("nan")) + self.anchor
            return logits, None

    nonfinite = NonFiniteModel()
    with pytest.raises(FloatingPointError, match="non-finite next-token NLL"):
        evaluate_perplexity(nonfinite, [torch.tensor([256, 1])], context_length=2, stride=1)
    assert nonfinite.training


def test_perplexity_validates_byte_alignment_and_document_limit():
    docs = [torch.tensor([256, 1, 257])]
    with pytest.raises(ValueError, match="aligned to every token"):
        evaluate_perplexity(
            UniformModel(), docs, context_length=2, stride=1, token_byte_counts=[[0, 1]]
        )
    with pytest.raises(ValueError, match="max_documents"):
        evaluate_perplexity(UniformModel(), docs, context_length=2, stride=1, max_documents=0)


def test_byte_tokenizer_reports_utf8_coverage_with_zero_byte_specials():
    tokenizer = ByteTokenizer()

    assert tokenizer.token_byte_counts("café") == [0, 1, 1, 1, 1, 1, 0]
    assert sum(tokenizer.token_byte_counts("café")) == len("café".encode())


def test_jsonl_task_adapter_scores_conditional_tokens_and_generation(tmp_path):
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text('{"id":"example-a","prompt":"A","target":"B"}\n', encoding="utf-8")
    model = DecoderLanguageModel(
        ModelConfig(
            vocab_size=259,
            max_seq_len=16,
            width=16,
            layers=1,
            query_heads=4,
            kv_heads=2,
            ffn_width=32,
        )
    )
    result = evaluate_tasks(model, ByteTokenizer(), task_path, max_new_tokens=4)
    assert result["examples"] == 1
    assert result["conditional_scored_tokens"] == 2  # target byte plus EOS
    assert len(result["details"]) == 1
    assert result["details"][0]["example_id"] == "example-a"
    assert result["details"][0]["conditional_scored_tokens"] == 2
    assert result["protocol"]["adapter"] == "jsonl-prompt-target-v1"
    assert len(result["protocol"]["artifact_sha256"]) == 64
    assert model.training


def test_task_adapter_restores_model_mode_on_failure_and_rejects_duplicate_ids(tmp_path):
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text('{"prompt":"A","target":"B"}\n', encoding="utf-8")
    model = DecoderLanguageModel(
        ModelConfig(
            vocab_size=259,
            max_seq_len=8,
            width=16,
            layers=1,
            query_heads=4,
            kv_heads=2,
            ffn_width=32,
        )
    )
    with pytest.raises(ValueError, match="leaves fewer"):
        evaluate_tasks(model, ByteTokenizer(), task_path, max_new_tokens=8)
    assert model.training

    task_path.write_text(
        '{"id":"same","prompt":"A","target":"B"}\n{"id":"same","prompt":"C","target":"D"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate task example id"):
        evaluate_tasks(model, ByteTokenizer(), task_path, max_new_tokens=2)
