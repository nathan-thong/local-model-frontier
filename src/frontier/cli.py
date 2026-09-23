"""Command line entry points for reproducible experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from frontier.config import RunConfig
from frontier.data.corpus import encode_documents, load_split, prepare_split
from frontier.evaluation.perplexity import evaluate_perplexity
from frontier.evaluation.tasks import evaluate_tasks
from frontier.experiments.compare import compare_runs, compare_seeded_runs
from frontier.experiments.results import write_json, write_summary
from frontier.models import DecoderLanguageModel
from frontier.profiling.benchmark import profile_model
from frontier.tokenization import build_tokenizer
from frontier.training.checkpoint import load_checkpoint
from frontier.training.runtime import resolve_device
from frontier.training.trainer import train


def _load_run(run_dir: Path) -> tuple[RunConfig, DecoderLanguageModel, dict, Path]:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "resolved_config.json"
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{run_dir} is not a trained run; expected resolved_config.json and checkpoints/last.pt"
        )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = RunConfig.from_dict(raw_config)
    device = resolve_device()
    model = DecoderLanguageModel(config.model).to(device)
    state = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {
            "schema_version": 1,
            "run_id": run_dir.name,
            "resolved_config": raw_config,
            "evaluation": {},
            "profiling": {},
        }
    )
    return config, model, summary, checkpoint_path


def _evaluate(run_dir: Path) -> dict:
    config, model, summary, _ = _load_run(run_dir)
    data_root = Path(summary["data"]["root"])
    _, valid_docs, manifest = load_split(data_root)
    tokenizer = build_tokenizer(config.tokenizer)
    valid_tokens = encode_documents(valid_docs, tokenizer)
    stride = config.evaluation.stride
    metrics = evaluate_perplexity(
        model,
        valid_tokens,
        context_length=config.train.context_length,
        stride=stride,
        max_documents=config.evaluation.max_validation_documents,
    )
    evaluation = {
        "perplexity": metrics,
        "protocol": {
            "dataset_sha256": manifest.get("validation_sha256"),
            "tokenizer": tokenizer.name,
            "context_length": config.train.context_length,
            "stride": stride,
            "document_boundary": "score within documents; no cross-document targets",
        },
    }
    if config.evaluation.tasks_path:
        task_path = Path(config.evaluation.tasks_path)
        if not task_path.is_absolute():
            task_path = Path.cwd() / task_path
        evaluation["tasks"] = evaluate_tasks(
            model,
            tokenizer,
            task_path,
            max_new_tokens=config.evaluation.generation_max_tokens,
        )
    summary["evaluation"] = evaluation
    write_json(run_dir / "evaluation.json", evaluation)
    write_summary(run_dir, summary)
    return evaluation


def _profile(run_dir: Path) -> dict:
    config, model, summary, checkpoint_path = _load_run(run_dir)
    profile = profile_model(
        model,
        config.profiling,
        checkpoint_path=checkpoint_path,
        weights_path=run_dir / "weights.pt",
    )
    # Full sequence versus chunked cached inference checks causal cache semantics on this checkpoint.
    device = next(model.parameters()).device
    length = min(8, config.model.max_seq_len)
    if length >= 2:
        ids = torch.randint(config.model.vocab_size - 3, (1, length), device=device)
        split = max(1, length // 2)
        full, _ = model(ids)
        first, cache = model(ids[:, :split], use_cache=True)
        second, _ = model(ids[:, split:], cache=cache, use_cache=True)
        cached = torch.cat((first, second), dim=1)
        profile["cache_parity_max_abs_logit_error"] = float((full - cached).abs().max().item())
        profile["cache_parity_tokens"] = length
    else:
        profile["cache_parity_max_abs_logit_error"] = None
        profile["cache_parity_tokens"] = length
    summary["profiling"] = profile
    write_json(run_dir / "profile.json", profile)
    write_summary(run_dir, summary)
    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frontier", description="Run reproducible small language-model experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train a configured baseline model")
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument(
        "--resume", action="store_true", help="resume exactly from checkpoints/last.pt"
    )
    train_parser.add_argument(
        "--seed", type=int, help="override the config seed for a planned seed repeat"
    )
    train_parser.add_argument(
        "--output-dir", type=Path, help="override the configured run directory"
    )

    eval_parser = commands.add_parser("evaluate", help="evaluate a trained run")
    eval_parser.add_argument("--run-dir", required=True, type=Path)

    profile_parser = commands.add_parser("profile", help="profile a trained run")
    profile_parser.add_argument("--run-dir", required=True, type=Path)

    data_parser = commands.add_parser(
        "prepare-data", help="split line-oriented UTF-8 documents deterministically"
    )
    data_parser.add_argument("--input", required=True, type=Path)
    data_parser.add_argument("--output", required=True, type=Path)
    data_parser.add_argument("--validation-fraction", type=float, default=0.02)
    data_parser.add_argument("--seed", type=int, default=17)
    data_parser.add_argument("--source-metadata", type=Path)

    compare_parser = commands.add_parser(
        "compare", help="compare runs and report whether controls are comparable"
    )
    compare_parser.add_argument("--baseline-run", required=True, type=Path)
    compare_parser.add_argument("--candidate-runs", nargs="+", required=True, type=Path)
    compare_parser.add_argument("--output", type=Path, default=Path("comparison.json"))

    seeded_parser = commands.add_parser(
        "compare-seeds", help="compare matched baseline/candidate groups across repeated seeds"
    )
    seeded_parser.add_argument("--baseline-runs", nargs="+", required=True, type=Path)
    seeded_parser.add_argument("--candidate-runs", nargs="+", required=True, type=Path)
    seeded_parser.add_argument("--minimum-seeds", type=int, default=3)
    seeded_parser.add_argument("--output", type=Path, default=Path("seeded-comparison.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            config = RunConfig.from_json(args.config)
            if args.seed is not None:
                config.seed = args.seed
            if args.output_dir is not None:
                config.output_dir = str(args.output_dir)
            run_dir = train(config, resume=args.resume)
            print(f"Training run saved to {run_dir}")
        elif args.command == "evaluate":
            metrics = _evaluate(args.run_dir.resolve())
            print(json.dumps(metrics, indent=2))
        elif args.command == "profile":
            metrics = _profile(args.run_dir.resolve())
            print(json.dumps(metrics, indent=2))
        elif args.command == "prepare-data":
            source_metadata = (
                json.loads(args.source_metadata.read_text(encoding="utf-8"))
                if args.source_metadata
                else None
            )
            manifest = prepare_split(
                args.input,
                args.output,
                args.validation_fraction,
                args.seed,
                source_metadata,
            )
            print(json.dumps(manifest, indent=2))
        elif args.command == "compare":
            report = compare_runs(args.baseline_run, args.candidate_runs, args.output)
            print(json.dumps(report, indent=2))
        elif args.command == "compare-seeds":
            report = compare_seeded_runs(
                args.baseline_runs,
                args.candidate_runs,
                args.output,
                args.minimum_seeds,
            )
            print(json.dumps(report, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
