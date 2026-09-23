# Measurement protocol

## Comparison contract

There is no single notion of a matched baseline. For each result, state whether the control matches training FLOPs, training tokens, wall time, inference resident memory, or inference latency. An architecture/capability improvement claim requires at minimum a matched training-compute control, the same data/tokenizer/evaluation protocol, at least three seed-matched runs, and one shared non-null Git revision recorded from clean worktrees. Use frontier compare-seeds to aggregate paired means and sample standard deviations; a pairwise comparison is metric-only. A Pareto point at a fixed memory budget also reports the training-compute budget used to obtain it. Report teacher cost separately and cumulatively for distillation.

## Stored and resident memory

- **Parameter count:** unique parameters; account for tied weights once.
- **Model tensor bytes:** bytes in unique parameter and buffer storages at the loaded dtype.
- **Artifact bytes:** serialized checkpoint file size, including metadata and optimizer state if present.
- **Resident memory:** process RSS and accelerator allocated/reserved bytes after load and during each phase. Report runtime baseline and measurement method.
- **Attention state:** for standard KV attention, theoretical tensor bytes are `2 * layers * batch * cached_tokens * kv_heads * head_dim * bytes_per_element`. Report actual allocated capacity and temporary/workspace memory separately.
- **Recurrent state:** report the module-defined persistent state tensors, batch capacity, dtype and bytes.

## Throughput and latency

Record hardware model, OS, Python/PyTorch/CUDA versions, device, precision, backend, thread counts, batch, prompt length and generation length. Warm up each workload, synchronize accelerator operations, measure repeated trials and report median plus spread. Report training end-to-end tokens/sec and steady-state step time. For inference, report prefill and cached decode separately; state whether tokenization, loading and sampling are included. Keep cold-start measurements separate from steady state. Use the same process and measurement procedure for controls.

## Compute accounting

Estimator v1 is an approximation for dense decoder-only Transformers, not a hardware FLOP counter. It estimates training work as `6 * unique_parameter_count * input_tokens + 12 * layers * width * context_length * input_tokens`, assuming a conventional forward/backward multiplier and causal-attention matrix work at configured context length. It does not model padding, optimizer work, embeddings specially, kernel fusion, recomputation or experimental blocks. Record its version and assumptions. Use profiler-derived counts only when the profiler, operations counted and coverage are documented. Do not compare estimators from different versions as exact compute.

For a matched-compute comparison using estimator v1, calculate token budgets before training and report actual estimated-FLOP delta. Target at most 1% difference where token granularity permits; explain larger deviations. Wall time is a separate hardware-dependent measurement.

## Evaluation

Perplexity is `exp(total scored negative log-likelihood / total scored tokens)`, with each target counted once, padding excluded, and document boundaries respected. Record tokenizer ID/hash, context, stride, EOS handling and dataset revision. Perplexity from different tokenizers is not directly comparable; use bits per byte or a byte-normalized score for cross-tokenizer comparisons. Downstream tasks must version examples, prompts, exact-match or scoring rules, generation settings and per-example outputs.

## Reproducibility limits

Store the resolved configuration, software/device versions, data manifest, seeds, run ID, checkpoint state and metrics. Deterministic kernels are opt-in because they can reduce throughput or reject unsupported operations. Promise exact repeatability only for a controlled same-software/same-hardware path and test it. Across platforms and releases, report numerical tolerances and seed variation rather than promising bitwise identity.
