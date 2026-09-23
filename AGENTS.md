# Repository instructions

This is an ML systems and research repository for small language models. Keep work focused on falsifiable experiments and trustworthy measurements. It is not a chatbot application or a wrapper around an existing model.

## Before changing code

- Read `RESEARCH.md`, `EXPERIMENTS.md`, `docs/measurement_protocol.md`, and `docs/reproducibility.md` before changing architectures, training recipes, or evaluation protocols.
- Preserve the separation between `models`, `training`, `evaluation`, `inference`, `profiling`, and `experiments`.
- Change one research variable at a time unless an experiment explicitly tests an interaction.
- Keep future milestones as a roadmap until a concrete experiment calls for implementing them.

## Experiment records

- Record every completed experiment in `EXPERIMENTS.md` with a hypothesis, change, control, training budget, evaluation metrics, result, interpretation, and next experiment.
- Keep failed, negative, and non-comparable results. Do not replace an old result when the protocol or budget changes; add a new entry.
- State which quantities are measured and which are estimates. Do not treat FLOPs, parameter count, model-file size, resident memory, and throughput as interchangeable.
- Never present a capability or efficiency improvement without a named control, the same data/tokenizer/evaluation protocol, equivalent disclosed training compute, at least three paired seeds, a shared pinned revision, and clean Git worktrees. Use `frontier compare-seeds`; it blocks ineligible comparisons.
- Report the direction and size of a metric change. A comparison passing eligibility checks does not mean the candidate improved.

## Data and artifacts

- Keep corpora, checkpoints, weights, caches, and run outputs out of Git. The root `data/` and `runs/` directories are ignored for this reason.
- Record data source, license, revision, split procedure, and hashes. Use the same prepared split for controls and candidates.
- Do not commit credentials, machine-specific paths, or generated model artifacts.

## Validation

Run the test suite and style checks before committing changes:

```powershell
python -m pytest
ruff check src tests scripts
ruff format --check src tests scripts
```

Add tests for new architecture invariants, comparison rules, or measurement paths. For documentation-only changes, verify links and commands against the current CLI.
