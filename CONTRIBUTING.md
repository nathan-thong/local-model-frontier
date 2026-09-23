# Contributing

- Follow the research and repository rules in `AGENTS.md`.
- Read `RESEARCH.md` and `docs/measurement_protocol.md` before adding a model family or benchmark.
- Keep architecture, training, evaluation, inference and profiling code in their named modules.
- Record every run in `EXPERIMENTS.md`; do not remove failed or non-comparable results.
- Add a component invariant or integration test when adding a new architecture or measurement path.
- Do not claim efficiency from operation counts alone. Include measured workload, hardware, memory and latency.
- Keep datasets, checkpoints, credentials and machine-specific paths out of commits.
- State licenses and source revisions for any data or imported implementation.

Before submitting a change, run:

```powershell
python -m pytest
ruff check src tests scripts
ruff format --check src tests scripts
```

Keep pull requests small enough that the experiment variable and its control are clear. Include the changed configs, commands, compute budget, seed set, and comparison artifacts when a change affects a result.
