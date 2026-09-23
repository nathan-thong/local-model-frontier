# Measurement protocol

## Comparison contract

There is no single notion of a matched baseline. For each result, state whether the control matches training FLOPs, training tokens, wall time, inference resident memory, or inference latency. An architecture/capability improvement claim requires at minimum a matched training-compute control, the same data/tokenizer/evaluation protocol, at least three seed-matched runs, and one shared non-null Git revision recorded from clean worktrees. Use frontier compare-seeds to aggregate paired means and sample standard deviations; a pairwise comparison is metric-only. A Pareto point at a fixed memory budget also reports the training-compute budget used to obtain it. Report teacher cost separately and cumulatively for distillation.

Seeded-comparison schema 3 distinguishes a test fixture from corpus origin. A three-seed improvement can be eligible only within its recorded evidence scope: for example, a synthetic-domain comparison must be labeled as such and cannot establish human-authored or broad language capability. A fixture, unknown origin, or missing provenance cannot support a scoped research claim. The report separately indicates human-corpus claim eligibility. Tokenizer artifact identity is required even for metric comparability; when it is missing, numeric deltas remain in the report as descriptive values but the runs are declared incomparable.

## Stored and resident memory

- **Parameter count:** unique parameters; account for tied weights once.
- **Model tensor bytes:** bytes in unique parameter and buffer storages at the loaded dtype.
- **Artifact bytes:** serialized checkpoint file size, including metadata and optimizer state if present.
- **Resident memory:** process RSS and accelerator allocated/reserved bytes after load and during each phase. Report runtime baseline and measurement method.
- **Attention state:** for standard KV attention, theoretical tensor bytes are `2 * layers * batch * cached_tokens * kv_heads * head_dim * bytes_per_element`. Report actual allocated capacity and temporary/workspace memory separately.
- **Recurrent state:** report the module-defined persistent state tensors, batch capacity, dtype and bytes.

## Throughput and latency

Record hardware model, OS, Python/PyTorch/CUDA versions, device, precision, backend, thread counts, batch, prompt length and generation length. Warm up each workload, synchronize accelerator operations, measure repeated trials and report every duration, median, range and sample standard deviation (null for one repetition). Report training end-to-end tokens/sec and steady-state step time. For inference, report time-to-first-token and steady-state cached decode separately. Time-to-first-token includes the prompt forward and greedy selection from its final logits. Cached decode runs exactly N incremental model forwards and reports N subsequent tokens; it excludes prompt prefill and the first token, and includes greedy selection after each forward. Report the generated token count and final cache position independently. State whether tokenization, loading and sampling are included. Keep cold-start measurements separate from steady state. Use the same process and measurement procedure for controls.

Profile schema 3 reports separate CUDA peaks for prefill and cached decode. Prefill peaks reset per prompt workload and include its warmups and measured repetitions. Decode peaks reset after the prompt forward and first-token selection, then report the maximum across warmups and measured repetitions. Allocated/reserved baselines remain visible alongside peaks. Full-attention KV bytes are kept separate from generic persistent state bytes; mixed or recurrent modules must expose active-versus-allocated state before that distinction can be claimed. CPU RSS and private-memory fields are point-in-time snapshots after a workload. A platform's `peak_rss_bytes` may be the process-lifetime high-water mark, so it is not a per-phase CPU peak and can depend on workload order. Use isolated worker processes before making per-phase CPU peak-memory claims. Do not compare these field meanings across profile schema versions.

Training summaries record a pre-step CPU/accelerator snapshot and a separate `training.memory` record. On CUDA, peak allocated and reserved memory are reset before every optimizer step, each step is synchronized, and the largest step peak in the current invocation is reported. Validation and checkpoint serialization are excluded from that phase peak. CPU readings remain absolute process snapshots; any platform-provided peak RSS is process-lifetime and is not attributed to training without an isolated worker. An interrupted/resumed run reports the current invocation's peak scope, not a synthetic maximum across invocations.

Training stops before applying an optimizer update when the loss or clipped gradient norm is non-finite. The run summary is marked failed and records the failing step, finite-value flags, prior completed tokens/updates and a JSONL failure event. Do not resume or omit a failed seed merely to obtain a successful comparison; any retry must be explicit and retain the failure record.

## Compute accounting

The historical `decoder-dense-v1` estimate is retained in old run records: `6 * unique_parameter_count * input_tokens + 12 * layers * width * context_length * input_tokens`. It is an approximation, not a hardware counter, and treats embedding parameters as dense work.

New supported runs use `decoder-module-mac-v1`. It separately counts attention-block Q/K/V/output projection MACs, dense attention score/value MACs, FFN MACs and vocabulary-head MACs, then reports two FLOPs per MAC and a 3x forward-MAC multiplier for forward plus backward. GQA uses the configured KV projection width. The token embedding lookup is excluded; a tied embedding is counted once as vocabulary projection compute. Causal masks are charged at dense context cost, even when masked cells are mathematically unused. Softmax, normalization, activations, loss, optimizer, dropout, scalar operations, kernel fusion and recomputation are not counted. These values are analytical estimates, not measured FLOPs. A sequence module without a registered estimator returns null compute and cannot enter an equivalent-compute claim. Record estimator version and assumptions, and never compare values from different estimator versions as exact compute.

For a matched-compute comparison, calculate token budgets before training and report actual estimated-FLOP delta. Record requested token budget, actual consumed tokens and full-step overshoot. Target at most 1% difference where token granularity permits; explain larger deviations. Wall time is a separate hardware-dependent measurement.

## Evaluation

Perplexity is `exp(total scored negative log-likelihood / total scored tokens)`, with each target counted once, padding excluded, and document boundaries respected. The evaluator reports token-weighted totals and per-document NLL, target count and perplexity; short documents with no next-token target are labeled. It restores the model's prior train/eval mode even after an evaluation error and raises on non-finite target losses. Record tokenizer ID/hash, context, stride, EOS handling and dataset revision. Perplexity from different tokenizers is not directly comparable.

When the tokenizer supplies a byte count for every encoded token, report `bits_per_byte` as the natural-log NLL of positive-byte-count targets divided by `ln(2)` and the sum of their UTF-8 byte counts. Special tokens with zero byte coverage stay in token perplexity but are excluded from both byte-normalized terms. The coverage list must align one-to-one with the encoded document tokens; tokenizer implementations are responsible for ensuring content-token coverage sums to the source text's UTF-8 length. This score is separately labeled and does not replace matched-tokenizer perplexity. Downstream tasks must version examples, prompts, exact-match or scoring rules, generation settings and per-example outputs.

## Reproducibility limits

Store the resolved configuration, software/device versions, data manifest, seeds, run ID, checkpoint state and metrics. Deterministic kernels are opt-in because they can reduce throughput or reject unsupported operations. Promise exact repeatability only for a controlled same-software/same-hardware path and test it. Across platforms and releases, report numerical tolerances and seed variation rather than promising bitwise identity.
