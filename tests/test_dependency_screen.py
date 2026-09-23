import copy
import hashlib
import json
from pathlib import Path

import torch

from frontier.config import ModelConfig
from frontier.data.dependency import generate_s06_dependency_splits
from frontier.evaluation.dependency import evaluate_dependency_recall
from frontier.models import DecoderLanguageModel
from frontier.profiling.compute import estimate_training_compute
from frontier.training.dependency_screen import state_dict_sha256, train_dependency_seed

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/s06/s06_dependency_screen.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_s06_dependency_preregistration_hash_and_generation_order_are_frozen():
    config = load_config()
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
        "4530e7367fb805031297d57ee99a95e87ad17e210d87827f68f09c934b933258"
    )
    assert config["task"]["example_generation"]["generation_order"] == (
        "training splits in the seeds array order, each in ascending example_index, "
        "followed by the shared evaluation split in ascending example_index"
    )


def tiny_model_config():
    return {
        "vocab_size": 259,
        "max_seq_len": 16,
        "width": 8,
        "layers": 2,
        "query_heads": 2,
        "kv_heads": 1,
        "ffn_width": 16,
        "normalization": "rmsnorm",
        "ffn": "gelu",
        "position": "rope",
        "residual_topology": "pre_norm",
        "sequence_types": ["normalized_linear_attention_reference"] * 2,
        "dropout": 0.0,
        "bias": False,
        "tie_embeddings": True,
        "rope_theta": 10000.0,
    }


def small_data_config():
    config = load_config()
    config["seeds"] = [17, 42, 123]
    config["task"]["training_examples_per_seed"] = 8
    config["task"]["training_examples_per_target_slot_per_seed"] = 2
    config["task"]["evaluation_examples_total"] = 8
    config["task"]["evaluation_examples_per_target_slot"] = 2
    return config


def test_s06_dependency_splits_are_deterministic_balanced_and_disjoint():
    config = small_data_config()
    first, first_manifest = generate_s06_dependency_splits(config)
    second, second_manifest = generate_s06_dependency_splits(config)

    assert first_manifest == second_manifest
    assert list(first) == [
        "training_seed_17",
        "training_seed_42",
        "training_seed_123",
        "evaluation",
    ]
    all_identities = []
    for name, dataset in first.items():
        assert torch.equal(dataset.input_ids, second[name].input_ids)
        assert torch.equal(dataset.targets, second[name].targets)
        assert len(dataset.identity_hashes) == len(set(dataset.identity_hashes))
        assert dataset.input_ids.shape[1] == config["task"]["input_length"]
        assert dataset.target_slots.tolist() == [index % 4 for index in range(len(dataset.targets))]
        for index, row in enumerate(dataset.input_ids.tolist()):
            slot = int(dataset.target_slots[index])
            assert row[1 + 2 * slot] == row[-1]
            assert dataset.targets[index].item() == row[2 + 2 * slot]
            assert row[0] == config["task"]["bos_token_id"]
            assert row[-2] == config["task"]["query_marker_token_id"]
        all_identities.extend(dataset.identity_hashes)
    assert len(all_identities) == len(set(all_identities))
    assert first["evaluation"].manifest("evaluation", 20260924)["target_slot_counts"] == {
        "0": 2,
        "1": 2,
        "2": 2,
        "3": 2,
    }
    assert first_manifest["unique_identities_total"] == 32


def test_dependency_evaluator_reports_full_and_conditional_accuracy_and_state_norms():
    config = small_data_config()
    datasets, _ = generate_s06_dependency_splits(config)
    model_config = ModelConfig(**tiny_model_config())
    model = DecoderLanguageModel(model_config).train()

    result = evaluate_dependency_recall(
        model,
        datasets["evaluation"],
        value_token_ids=config["task"]["value_token_ids"],
        joint_strata=config["task"]["joint_strata"],
        batch_size=4,
    )

    assert model.training
    assert result["complete"] is True
    assert result["metrics"]["examples"] == 8
    assert result["metrics"]["mean_target_nll_nats"] is not None
    assert result["metrics"]["full_vocabulary_exact_match"] is not None
    assert result["metrics"]["conditional_value_top1"] is not None
    assert [item["examples"] for item in result["joint_strata"]] == [2, 2, 2, 2]
    assert all(len(row["state_norms_by_layer"]) == 2 for row in result["per_example"])
    assert all(row["all_scored_values_finite"] for row in result["per_example"])


def test_dependency_evaluator_preserves_partial_scoring_when_cap_is_reached():
    config = small_data_config()
    datasets, _ = generate_s06_dependency_splits(config)
    model = DecoderLanguageModel(ModelConfig(**tiny_model_config()))

    result_rows = []

    def stop_after_one_batch():
        return bool(result_rows)

    # The wrapper marks its first forward so the scorer stops before the second batch.
    class StopAfterFirstBatch(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.calls = 0

        def forward(self, *args, **kwargs):
            output = self.wrapped(*args, **kwargs)
            self.calls += 1
            result_rows.append(True)
            return output

    wrapped = StopAfterFirstBatch(model)
    result = evaluate_dependency_recall(
        wrapped,
        datasets["evaluation"],
        value_token_ids=config["task"]["value_token_ids"],
        joint_strata=config["task"]["joint_strata"],
        batch_size=4,
        stop_requested=stop_after_one_batch,
    )
    assert wrapped.calls == 1
    assert result["complete"] is False
    assert result["scored_examples"] == 4
    assert len(result["per_example"]) == 4


def test_dependency_evaluator_records_nonfinite_outputs_without_invalid_json():
    config = small_data_config()
    datasets, _ = generate_s06_dependency_splits(config)
    wrapped_model = DecoderLanguageModel(ModelConfig(**tiny_model_config()))

    class NonfiniteLogits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            logits, cache = self.model(*args, **kwargs)
            return torch.full_like(logits, float("nan")), cache

    result = evaluate_dependency_recall(
        NonfiniteLogits(wrapped_model),
        datasets["evaluation"],
        value_token_ids=config["task"]["value_token_ids"],
        joint_strata=config["task"]["joint_strata"],
        batch_size=4,
    )
    assert result["complete"] is True
    assert result["metrics"]["all_scored_values_finite"] is False
    assert result["metrics"]["mean_target_nll_nats"] is None
    assert result["per_example"][0]["full_vocabulary_prediction"] is None
    assert json.dumps(result, allow_nan=False)


def test_dependency_training_executes_one_fixed_pass_and_logs_finite_gradients():
    config = small_data_config()
    config["model"] = tiny_model_config()
    config["training"].update(
        {
            "updates_per_seed": 2,
            "batch_size": 4,
            "torch_num_threads": 1,
        }
    )
    datasets, _ = generate_s06_dependency_splits(config)
    events = []
    previous_num_threads = torch.get_num_threads()
    result = train_dependency_seed(
        config,
        datasets["training_seed_17"],
        17,
        stop_requested=lambda: False,
        record_event=events.append,
    )
    assert result.status == "completed"
    assert torch.get_num_threads() == previous_num_threads
    assert result.updates_completed == 2
    assert result.updates_attempted == 2
    assert result.tokens_seen == 88
    assert result.all_losses_and_gradient_norms_finite
    updates = [event for event in events if event["event"] == "update"]
    assert [event["tokens_seen"] for event in updates] == [44, 88]
    assert all(
        event["pre_clip_gradient_norm"] >= event["post_clip_gradient_norm"] for event in updates
    )
    assert all(event["gradient_norms_finite"] for event in updates)
    assert state_dict_sha256(result.model) == state_dict_sha256(copy.deepcopy(result.model))


def test_dependency_training_stops_at_the_update_boundary_when_capped():
    config = small_data_config()
    config["model"] = tiny_model_config()
    config["training"].update(
        {
            "updates_per_seed": 2,
            "batch_size": 4,
            "torch_num_threads": 1,
        }
    )
    datasets, _ = generate_s06_dependency_splits(config)
    events = []
    result = train_dependency_seed(
        config,
        datasets["training_seed_17"],
        17,
        stop_requested=lambda: True,
        record_event=events.append,
    )
    assert result.status == "incomplete"
    assert result.updates_completed == 0
    assert result.updates_attempted == 0
    assert result.tokens_seen == 0
    assert events[0]["event"] == "stop"


def test_s06_dependency_compute_budget_matches_registered_estimator():
    config = load_config()
    model = ModelConfig(**config["model"])
    input_length = config["task"]["input_length"]
    training_tokens = (
        len(config["seeds"]) * config["task"]["training_examples_per_seed"] * input_length
    )
    evaluation_tokens = (
        len(config["seeds"]) * config["task"]["evaluation_examples_total"] * input_length
    )
    training = estimate_training_compute(model, training_tokens, input_length)
    evaluation_training_multiplier = estimate_training_compute(
        model, evaluation_tokens, input_length
    )
    evaluation_forward = evaluation_training_multiplier["estimated_flops"] // 3
    assert (
        training["estimated_flops"]
        == config["compute_budget"]["training_estimated_flops_all_seeds"]
    )
    assert (
        evaluation_forward
        == config["compute_budget"]["evaluation_forward_estimated_flops_all_seeds"]
    )
    assert (
        training["estimated_flops"] + evaluation_forward
        == config["compute_budget"]["total_estimated_flops"]
    )
    assert (
        config["compute_budget"]["total_estimated_flops"]
        <= config["compute_budget"]["total_estimated_flops_cap"]
    )
