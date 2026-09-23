# Reproducibility

Every run stores its resolved config, tokenizer description, data hashes, environment, seed, metrics and checkpoint state under one run directory. Data preparation is deterministic by seed and splits documents before tokenization. The byte tokenizer has a fixed vocabulary and requires no corpus-fitted state. The run environment records the current Git revision when one exists; an uncommitted worktree is recorded as null. Treat null-revision runs as metric-only. Before an improvement claim, train the control and candidate from the same non-null revision and compare at least three matched seeds with frontier compare-seeds.

Training checkpoints include the model, optimizer, scheduler, mixed-precision scaler, Python/NumPy/PyTorch RNG states, CUDA RNG states, sampler generator state, step and consumed-token count. Resume uses those states and the original resolved config. The short resume integration test checks a controlled same-environment execution.

Use `deterministic: true` to request deterministic PyTorch algorithms. Some operations may not offer deterministic implementations, and bitwise equality across hardware, PyTorch versions or backends is not guaranteed. Performance benchmarks should record `deterministic`, device, kernel/backend selection and software versions. Do not mix timing results from different environments without labeling the difference.

Do not edit `resolved_config.json` or metrics after a run. To change an experiment, create a new run ID. Keep datasets and checkpoints outside source control and use their recorded hashes to identify exact inputs.
