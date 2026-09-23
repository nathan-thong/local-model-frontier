# Local Model Frontier

A research codebase for testing whether sequence architecture, low-bit weights, distillation and adaptive computation improve language-model capability at fixed resident-memory or compute budgets. The project begins with a small, conventional decoder-only Transformer and a reproducible train/evaluate/profile loop.

Repository code is licensed under [Apache-2.0](LICENSE). Dataset licenses and terms remain separate; no corpus text is included in the source repository.

## Status

Milestone 1 infrastructure is implemented and tested, and both conventional baselines have trained on a pinned, bounded TinyStories subset with three matched seeds each. Their estimated training-FLOP budgets differ by 0.044%; the larger model has worse held-out perplexity on this setup. The paired comparison passes its pinned-revision, clean-worktree and provenance checks, but no improvement is observed. The corpus is synthetic short fiction, so this is infrastructure and narrow baseline evidence rather than broad capability evidence. Run artifacts and results are recorded in [EXPERIMENTS.md](EXPERIMENTS.md). The baseline uses a fixed UTF-8 byte tokenizer, an infrastructure choice rather than a modeling recommendation.

## Quick start

Requires Python 3.10+, PyTorch 2.3+ and NumPy. Install the project and development tools, then run the offline smoke experiment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
frontier train --config configs/smoke.json
frontier evaluate --run-dir runs/smoke
frontier profile --run-dir runs/smoke
```

To continue an interrupted run, use the same resolved config and run directory with `frontier train --config configs/smoke.json --resume`. When the first invocation used `--output-dir`, pass the same override when resuming.

The smoke config creates a deterministic fixture under its run directory; it needs no downloaded dataset. For a real corpus, create UTF-8 text files with one document per line, or split a source file deterministically:

```powershell
frontier prepare-data --input path\to\corpus.txt --output data\corpus --validation-fraction 0.02 --seed 17
```

Point a training config's `data_dir` at the resulting directory. It contains `train.txt`, `validation.txt`, and a hash-bearing `data_manifest.json`. Keep data and checkpoints out of version control.

The baseline configs expect that prepared directory at `data/tinystories`. Export the selected, licensed corpus as one UTF-8 document per line before running. The preparation command splits on documents, groups identical text into one split, records hashes and rejects exact train/validation overlap. Use the same prepared directory for every control and candidate.

Baseline A and B configs freeze the planned training budgets. To make the three seed runs without editing the source config, use:

```powershell
frontier train --config configs/baseline_a.json --seed 17 --output-dir runs/baseline-a-seed17
frontier train --config configs/baseline_a.json --seed 42 --output-dir runs/baseline-a-seed42
frontier train --config configs/baseline_a.json --seed 123 --output-dir runs/baseline-a-seed123
```

Use the same seeds and corresponding output directories for `baseline_b.json`.

## Repository guide

- `RESEARCH.md` records assumptions, prior work and failure modes before experimental architecture work.
- `EXPERIMENTS.md` is the append-only research log. Planned experiments are marked not run.
- `src/frontier/models` contains the configurable Transformer and sequence-module interface.
- `src/frontier/training`, `evaluation`, `inference`, and `profiling` keep experiment stages separate.
- `configs` stores fully specified JSON run configurations.
- `tests` contains architecture, data, resume, profiling and comparison invariants.

See [docs/reproducibility.md](docs/reproducibility.md), [docs/measurement_protocol.md](docs/measurement_protocol.md), [docs/data_protocol.md](docs/data_protocol.md) and [docs/downstream_evaluation.md](docs/downstream_evaluation.md) before comparing results. The command line writes resolved configs, environment details, metrics, checkpoints, evaluation output, profiles and a machine-readable summary into each run directory.

## Current boundaries

The initial implementation provides full causal attention with MHA/GQA, LayerNorm/RMSNorm, GELU/SwiGLU, RoPE/learned positions, pre/post norm, mixed precision, resumable checkpoints, perplexity and generic JSONL task scoring. Local attention and recurrent/linear sequence modules are extension points for Milestone 2; no speed claim follows from an interface alone. Low-bit training, distillation, speculative decoding and adaptive compute remain future milestones.

## Research integrity

Every capability or efficiency claim needs a named control, a matched and disclosed training-compute budget, identical evaluation data and protocol, and repeated seeds. Report measured resident bytes and latency as well as parameter counts. Parameter count, nominal bit width, estimated FLOPs, and throughput are distinct quantities; none substitutes for the others.

## Reproducing the bounded TinyStories corpus

For the Milestone 1 corpus check, the repository uses a 20 MiB prefix of the pinned `roneneldan/TinyStories` training file. The [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md) declares `cdla-sharing-1.0` and describes the stories as synthetic generations. This small subset verifies the training and held-out perplexity path; it is not evidence of general language capability. Raw and prepared data stay under the git-ignored root `data/` directory.

From the repository root on Windows, fetch the pinned range, convert complete story records to one document per line, and split deterministically:

```powershell
curl.exe --fail --location --range 0-20971519 --max-filesize 20971520 --output data\tinystories-train-prefix.raw.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-train.txt
python scripts\prepare_tinystories_prefix.py --input data\tinystories-train-prefix.raw.txt --output data\tinystories-documents.txt --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 --upstream-size-bytes 1924281556
frontier prepare-data --input data\tinystories-documents.txt --output data\tinystories --validation-fraction 0.02 --seed 17 --source-metadata data\tinystories-documents.txt.source.json
```

The converter discards a partial trailing story and records the source revision, selected byte range, hashes, license and normalization. The M1 validation split is held out from this training-file prefix; it is not the dataset's separate validation file.

Use `frontier compare-seeds --baseline-runs ... --candidate-runs ... --output runs\seeded-comparison.json` for repeated-seed aggregation. An improvement claim requires at least three matched seeds, non-synthetic data, the same pinned Git revision, and clean Git worktrees recorded for every run; pairwise comparisons are metric-only.
