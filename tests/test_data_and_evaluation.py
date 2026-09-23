import json

import torch
from torch import nn

from frontier.config import ModelConfig
from frontier.data.corpus import encode_documents, prepare_split, read_documents, sample_batch
from frontier.data.tinystories import prepare_prefix
from frontier.evaluation.perplexity import evaluate_perplexity
from frontier.evaluation.tasks import evaluate_tasks
from frontier.models import DecoderLanguageModel
from frontier.tokenization import ByteTokenizer


def test_byte_tokenizer_round_trips_utf8_and_special_ids():
    tokenizer = ByteTokenizer()
    text = "naïve 🐇"
    ids = tokenizer.encode(text)
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_size == 259


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
    assert one["source_metadata"] == source_metadata
    train = read_documents(first / "train.txt")
    valid = read_documents(first / "validation.txt")
    assert set(train).isdisjoint(valid)
    assert ("alpha" in train) == ("alpha" not in valid)


def test_tinystories_prefix_keeps_only_complete_documents_and_records_provenance(tmp_path):
    raw = b"One story\n<|endoftext|>\nTwo story<|endoftext|>\nCut off"
    source = tmp_path / "source.raw"
    output = tmp_path / "documents.txt"
    source.write_bytes(raw)

    metadata = prepare_prefix(source, output, "commit-sha", len(raw) + 100)

    assert read_documents(output) == ["One story", "Two story"]
    assert metadata["complete_document_count"] == 2
    assert metadata["discarded_trailing_incomplete_bytes"] == len(b"\nCut off")
    assert metadata["selected_range_bytes"] == len(raw)
    assert json.loads(output.with_suffix(".txt.source.json").read_text()) == metadata


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
    metrics = evaluate_perplexity(UniformModel(), docs, context_length=3, stride=2)
    assert metrics["scored_tokens"] == 5
    assert abs(metrics["mean_nll"] - torch.log(torch.tensor(259.0)).item()) < 1e-6
    assert abs(metrics["perplexity"] - 259.0) < 1e-3


def test_jsonl_task_adapter_scores_conditional_tokens_and_generation(tmp_path):
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text('{"prompt":"A","target":"B"}\n', encoding="utf-8")
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
