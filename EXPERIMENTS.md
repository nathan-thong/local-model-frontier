# Experiment log

This log records planned and completed experiments. Entries are never backfilled with expected results. Each completed result must link to its run artifact; preserve failures and incompatible comparisons. A change in budget or procedure creates a new run/experiment revision.

## M1-001 — Offline fixture smoke check

- **Status:** Completed; CPU FP32, PyTorch 2.14.0, seed 17.
- **Hypothesis:** The byte tokenizer, causal model, optimizer and loss path can overfit a tiny fixed corpus.
- **Change:** Train the smoke configuration on the checked-in deterministic fixture.
- **Control:** Same 2-layer conventional GQA Transformer at initialization; this fixture check is not a capability comparison.
- **Training budget:** 30 optimizer updates, 7,680 byte tokens, approximately 1.427 billion estimated dense-decoder FLOPs (estimator v1).
- **Evaluation metrics:** Final train loss 4.678; held-out NLL 4.824 and perplexity 124.504 over 2,180 fixture tokens; 22,784 parameters; FP32 tensor bytes 91,168; train throughput 41,174 tokens/sec measured over optimizer-active time. CPU RSS/VRAM and inference profiles are recorded in the artifact.
- **Result:** Training, evaluation, checkpoint load, and profiling completed. Full-sequence versus chunked-cache logits differed by at most `1.79e-7` over 8 tokens. Machine-readable artifacts are in [`runs/smoke-verified/summary.json`](runs/smoke-verified/summary.json), [`runs/smoke-verified/profile.json`](runs/smoke-verified/profile.json) and [`runs/smoke-verified/metrics.jsonl`](runs/smoke-verified/metrics.jsonl).
- **Interpretation:** The small model learns the synthetic fixture and validates the end-to-end path. These synthetic held-out results do not measure language capability or support an architecture claim.
- **Next experiment:** Resume equivalence, then train the prepared-corpus baselines A and B.

## M1-002 — Resume determinism

- **Status:** Completed; deterministic CPU FP32 test, PyTorch 2.14.0.
- **Hypothesis:** Checkpointing optimizer, scheduler, RNG and data-sampler state reproduces an uninterrupted short run in the same environment.
- **Change:** Compare 4 uninterrupted updates to 2 updates plus checkpoint/reload plus 2 updates.
- **Control:** Uninterrupted run, identical config and seed.
- **Training budget:** 4 optimizer updates per path; 128 byte tokens.
- **Evaluation metrics:** Model/optimizer tensor equality where deterministic kernels permit, final NLL, sampled batch/index stream, total tokens and estimated FLOPs.
- **Result:** Model tensors were bitwise equal after both paths in `tests/test_resume.py`. The full suite passes: 24 tests, including [tests/test_resume.py](tests/test_resume.py) and worktree-provenance coverage.
- **Interpretation:** Same-environment exact resume works for this deterministic CPU test. It does not establish equivalence across devices or software versions.
- **Next experiment:** Keep the same checks in the baseline integration test and begin corpus runs.

## M1-003 — Conventional baseline A

- **Status:** Completed for seed 17; Windows 11 CPU FP32, PyTorch 2.14.0.
- **Hypothesis:** The configured conventional decoder-only Transformer can learn from a prepared language corpus at small scale.
- **Change:** Train the 4-layer, width-128 decoder with UTF-8 byte tokenization, context 256, pre-norm RMSNorm, RoPE, GELU FFN, and tied embeddings.
- **Control:** Same architecture at initialization; this is a learning check, not a method comparison.
- **Training budget:** 2,097,152 input tokens, 1,024 optimizer updates, 13,625,783,746,560 estimated dense-decoder FLOPs (estimator v1), seed 17.
- **Evaluation metrics:** Validation NLL 1.50373 and perplexity 4.49843 over 411,608 scored tokens; 33,046 training tokens/sec; 820,736 parameters; 3,283,200 FP32 tensor bytes; 3,297,333-byte weights artifact; peak training RSS 663,212,032 bytes.
- **Result:** The run completed on the pinned TinyStories 20 MiB training-file prefix with the deterministic 2% document holdout. Artifacts: [summary.json](runs/tinystories-a-seed17/summary.json), [summary.csv](runs/tinystories-a-seed17/summary.csv), [profile.json](runs/tinystories-a-seed17/profile.json), and [data_manifest.json](runs/tinystories-a-seed17/data_manifest.json).
- **Interpretation:** End-to-end training and held-out perplexity evaluation work on a real prepared corpus. TinyStories is a synthetic short-story dataset and this byte-token setup does not measure broad language capability.
- **Next experiment:** Repeat the same recipe and token budget across the predeclared seeds in M1-004.

## M1-004 — Baseline A seed repeats

- **Status:** Completed; seeds 17, 42, and 123 on the same CPU FP32 environment.
- **Hypothesis:** Baseline A's held-out metric is not an artifact of one initialization seed.
- **Change:** Repeat baseline A without changing the data split, token budget, optimizer, tokenizer, or evaluation protocol.
- **Control:** The other baseline A seeds, paired by seed with baseline B in M1-005.
- **Training budget:** 2,097,152 input tokens and 13,625,783,746,560 estimated FLOPs per seed; 6,291,456 input tokens total.
- **Evaluation metrics:** Per-seed perplexities were 4.49843, 4.66806, and 4.54206. Mean perplexity was 4.56952 with sample standard deviation 0.08809; mean NLL was 1.51928 ± 0.01920. Training throughput was 33,046–34,355 tokens/sec.
- **Result:** All three runs completed. The aggregate and paired comparison are in [tinystories-ab-seeds.json](runs/tinystories-ab-seeds.json); individual summaries are [seed 17](runs/tinystories-a-seed17/summary.json), [seed 42](runs/tinystories-a-seed42/summary.json), and [seed 123](runs/tinystories-a-seed123/summary.json).
- **Interpretation:** The three-seed baseline is stable on this particular held-out split. All run environments report a null Git revision because the initial repository had not yet been committed; the comparison harness therefore blocks an improvement claim.
- **Next experiment:** Train baseline B with matched estimated compute and the same three seeds.

## M1-005 — Conventional baseline B, compute matched

- **Status:** Completed; seeds 17, 42, and 123 on the same CPU FP32 environment as baseline A.
- **Hypothesis:** A larger conventional model might improve held-out language modeling under baseline A's estimated training-FLOP budget.
- **Change:** Train the 6-layer, width-256 decoder with the same tokenizer, context length, optimizer family, data split, and evaluation protocol.
- **Control:** Baseline A paired by seed.
- **Training budget:** 407,552 input tokens and 13,631,773,212,672 estimated FLOPs per seed. This is a 0.04396% higher estimated budget than baseline A; the token-count difference is disclosed.
- **Evaluation metrics:** Baseline B mean perplexity was 7.77399 ± 0.12621 and mean NLL was 2.05070 ± 0.01627 across seeds. The paired perplexity delta, candidate minus baseline A, was +3.20447 ± 0.05706. Each pair passed the data, tokenizer, evaluation, recipe, seed, environment, and 1%-compute checks. The 3-seed metric comparison is marked comparable; improvement-claim eligibility is false because every run lacks a pinned Git revision.
- **Result:** Baseline B perplexity was higher on all three paired seeds. B has 4,788,224 parameters and 19,153,280 FP32 tensor bytes versus A's 820,736 parameters and 3,283,200 bytes. At batch 1, prompt 32 plus 32 cached decode tokens, A/B measured 15,755/9,002 prefill tokens/sec, 617/385 decode tokens/sec, and 258,048/774,144 KV-cache bytes. Peak training RSS was 663,212,032 bytes for A and 882,659,328 for B. Full paired results are in [tinystories-ab-seeds.json](runs/tinystories-ab-seeds.json); profiles are [baseline A](runs/tinystories-a-seed17/profile.json) and [baseline B](runs/tinystories-b-seed17/profile.json); all six per-run CSV summaries are alongside their JSON summaries.
- **Interpretation:** Under this narrow TinyStories, byte-token, and fixed-compute setup, the larger model used more parameters and memory, ran slower on CPU, and had worse held-out perplexity. This is a metric result for the disclosed setup, not a general architecture claim; no improvement is claimed. No CUDA or energy sensor was available.
- **Next experiment:** Pin the source revision before using this comparison as an improvement claim; then begin Milestone 2 with interchangeable sequence blocks and the same corpus, seed, and compute controls.

## M1-006 — Inference cache and profiling checks

- **Status:** Completed on offline fixture and TinyStories baseline checkpoints; Windows CPU FP32, PyTorch 2.14.0.
- **Hypothesis:** Incremental cached decoding agrees numerically with full-sequence evaluation, and the memory/throughput profiler uses consistent workloads.
- **Change:** Profile checkpoints with fixed prompt lengths and generation lengths.
- **Control:** Full-sequence logits and analytically calculated KV allocation for the same model.
- **Training budget:** None; report checkpoint provenance and inference device.
- **Evaluation metrics:** Logit difference, cache bytes, model bytes, process/device peak memory, prefill latency and decode latency/tokens per second.
- **Result:** On the smoke model, actual and analytic cache bytes matched at 3,840, 7,936 and 14,080 bytes; cached/full-sequence max logit error was 1.79e-7. On TinyStories A/B with a 32-token prompt and 32-token cached decode, cache bytes were 258,048/774,144, prefill throughput was 15,755/9,002 tokens/sec, and decode throughput was 617/385 tokens/sec. All profile and comparison outputs are linked from M1-003 and M1-005. Peak CPU process RSS during A/B inference profiling was 226,521,088/292,102,144 bytes; no CUDA device or energy sensor was available.
- **Interpretation:** Cache semantics and byte accounting agree for standard attention. The CPU timings are local measurements, not hardware-general performance claims.
- **Next experiment:** Profile interchangeable sequence blocks under the same prompts, token budget and evaluation protocol in Milestone 2.

## M1-007 — Matched-compute smoke baseline pair

- **Status:** Completed; CPU FP32, PyTorch 2.14.0, seed 17.
- **Hypothesis:** The comparison harness can verify a training-compute-matched control pair and refuse a scientific capability claim from synthetic data.
- **Change:** Train a 1-layer, width-16 baseline A and a 2-layer, width-16 baseline B with the same byte tokenizer, fixture, seed, context, batch, optimizer and evaluation protocol.
- **Control:** Baseline A, 120 updates and 1,920 input tokens.
- **Training budget:** Baseline B used 88 updates and 1,408 tokens. Estimator v1 measured 74,833,920 versus 74,612,736 FLOPs, a 0.296% difference.
- **Evaluation metrics:** Fixture NLL/perplexity, run-comparison checks, parameters, tensor bytes, peak process memory, KV bytes and CPU prefill/decode throughput.
- **Result:** A scored NLL 4.2921/perplexity 73.1221 with 5,984 parameters and 23,944 tensor bytes; B scored NLL 4.6785/perplexity 107.6103 with 7,808 parameters and 31,248 tensor bytes. The comparator verified matching data, tokenizer, evaluation protocol, training recipe, seed and environment; comparable_for_metric_comparison is true and comparable_for_improvement_claim is false because evidence scope is synthetic infrastructure fixture. For the 8-token prompt profile, A measured 12,674 prefill tokens/sec and 2,352 decode tokens/sec with 704 KV bytes; B measured 7,779 prefill tokens/sec and 1,325 decode tokens/sec with 1,408 KV bytes. These tiny CPU timing results are functional checks, not performance evidence. Artifacts: [runs/m1-smoke-a/summary.json](runs/m1-smoke-a/summary.json), [runs/m1-smoke-a/profile.json](runs/m1-smoke-a/profile.json), [runs/m1-smoke-b/summary.json](runs/m1-smoke-b/summary.json), [runs/m1-smoke-b/profile.json](runs/m1-smoke-b/profile.json), and [runs/m1-smoke-comparison.json](runs/m1-smoke-comparison.json).
- **Interpretation:** The matched-compute comparison and synthetic-evidence guard work. The two fixture scores do not establish language capability or an architecture improvement.
- **Next experiment:** Prepare a licensed corpus and run the published baseline A/B configs with the predeclared equal-compute protocol.

## M1-008 — Pinned multi-seed reproducibility and comparison

- **Status:** Completed; Windows 11 CPU FP32, PyTorch 2.14.0, source revision `7e518c4a24171444a5c0c613cc126346d19bb178`.
- **Hypothesis:** Once every run records a pinned source revision and distinct matched seeds, the comparison harness can verify the full experiment provenance and assess the larger conventional model under an equivalent estimated training-compute budget.
- **Change:** Re-run Baseline A and B with seeds 17, 42 and 123 against the same bounded TinyStories document split. Seed 17 reuses the correctly configured pinned runs; seeds 42 and 123 use seed-specific configs. Architecture, optimizer recipe, corpus, tokenizer and evaluation protocol remain fixed within each model group.
- **Control:** Baseline A, paired with Baseline B by seed. Both groups use the same source revision, train/validation hashes, UTF-8 byte tokenizer, CPU/software environment and evaluation protocol.
- **Training budget:** Per seed, A used 2,097,152 input tokens and 13,625,783,746,560 estimated FLOPs; B used 407,552 input tokens and 13,631,773,212,672 estimated FLOPs, 0.04396% more. Each group contains three seeds. The estimator is `decoder-dense-v1`; the FLOPs are estimates, not hardware counters.
- **Evaluation metrics:** A validation perplexity was 4.49843, 4.66806 and 4.54206 (mean 4.56952, sample SD 0.08809); B was 7.63929, 7.88951 and 7.79318 (mean 7.77399, sample SD 0.12621). The paired B-minus-A perplexity delta was +3.20447 (sample SD 0.05706); paired mean-NLL delta was +0.53141 (sample SD 0.00770). Training throughput ranged from 29,118–30,737 tokens/sec for A and 7,869–7,909 for B. At batch 1, 32 prompt tokens and 32 cached decode tokens, A/B measured 13,287/6,917 prefill tokens/sec, 591/361 decode tokens/sec, 258,048/774,144 actual KV-cache bytes, and 226,611,200/292,012,032 peak process RSS bytes. Parameter counts were 820,736/4,788,224; FP32 tensor sizes were 3,283,200/19,153,280 bytes. No CUDA device or energy sensor was available.
- **Result:** The three-seed comparator passed all corpus, tokenizer, evaluation, environment, clean-worktree, architecture, recipe, seed and 1%-compute checks. It records `comparable_for_metric_comparison: true`, `source_worktrees_clean: true`, and `comparable_for_improvement_claim: true`; this field means the evidence passes the reporting gates and does not mean the candidate improved. Baseline B had higher perplexity on every paired seed. The full comparison is [`runs/verified-tinystories-ab-seeds.json`](runs/verified-tinystories-ab-seeds.json). Run summaries: [A seed 17](runs/pinned-a-seed17/summary.json), [A seed 42](runs/verified-a-seed42/summary.json), [A seed 123](runs/verified-a-seed123/summary.json), [B seed 17](runs/pinned-b-seed17/summary.json), [B seed 42](runs/verified-b-seed42/summary.json), and [B seed 123](runs/verified-b-seed123/summary.json). Each run directory also contains its `profile.json`, `summary.csv`, environment, data manifest and checkpoint artifacts.
- **Interpretation:** Under this small synthetic-story corpus, byte-tokenizer and fixed-compute setup, the larger model used substantially more resident and KV memory, trained and decoded more slowly on CPU, and had worse held-out perplexity. This is a trustworthy result for the disclosed narrow setup; it does not establish general language capability or support a novel-architecture conclusion. The pinned comparator is now able to report the result, and the result is not an improvement.
- **Next experiment:** Begin Milestone 2 with one interchangeable sequence-block change at a time, using Baseline A as the control, the same pinned corpus/evaluation protocol, at least three matched seeds and an equivalent estimated-compute budget.
- **Protocol note:** The first pinned rerun attempt named folders for seeds 17/42/123 while leaving the configs at seed 17. `compare-seeds` rejected the duplicate seeds. Those duplicate-seed folders (`runs/pinned-a-seed42`, `runs/pinned-a-seed123`, `runs/pinned-b-seed42`, and `runs/pinned-b-seed123`) are excluded from this result and retained as audit artifacts. The corrected runs above use their actual configured seeds. The clean-worktree field was added after these runs; it was audited from the clean status recorded before the training set and the absence of tracked source changes through training. Their environment files mark this field as retroactively verified. Future runs capture it automatically; the CLI also supports a `--seed` override.

## Reporting erratum — corpus origin and legacy eligibility

The M1 TinyStories runs use synthetic short stories. In the version of the run schema used for M1, `synthetic_fixture: false` meant only that data came from a prepared corpus instead of the built-in smoke fixture. It did not mean the text was human-authored. The old comparator had no separate `content_origin` field and could mark the M1-008 report `comparable_for_improvement_claim: true` without checking this distinction.

The recorded perplexities, compute estimates, and negative direction remain the results of those runs. Interpret them only as a narrow TinyStories corpus comparison; do not cite the legacy eligibility flag as evidence of broad capability or natural-language generalization. New comparisons require an explicit origin value. Missing origin is recorded as unknown and blocks improvement eligibility. The original run summaries are preserved unchanged.

The linked `runs/` files are local ignored artifacts and are not present in a public clone. Until the portable evidence export is implemented, use the numerical results in this log as the public record and treat artifact links as author-local references.

## M2-GQA-001 - Grouped-KV attention pilot

- **Status:** Planned only; the MHA/GQA confirmation configs and three-seed, model-free sweep plan are frozen at clean commit `4a8dd97`. The schema-3 split is local at `data/tinystories-v3`; its raw prefix, synthetic origin, source metadata, split hashes and manifest hash are recorded in `docs/data_protocol.md`. The M1 schema-1 data remains unchanged. No candidate run has started.
- **Hypothesis:** Reducing KV heads from four to two will halve standard-attention cache storage at the same layer, batch, context and dtype, while preserving enough short-story language modeling quality to justify the configuration.
- **Change:** Create immutable `configs/m2/tinystories_v3_mha.json` and `configs/m2/tinystories_v3_gqa.json` copies of `configs/baseline_a.json` and `configs/gqa_a.json`, pointing both at `data/tinystories-v3`. The candidate changes KV heads from four to two. Match the `decoder-module-mac-v1` estimated FLOP budget by using 2,097,152 MHA tokens (1,024 optimizer updates) and 2,232,320 GQA tokens (1,090 updates); the planned totals are 13,611,288,231,936 and 13,610,794,352,640 estimated FLOPs, respectively (-0.00363%). Keep the byte tokenizer, model width/depth, FFN, normalization, positional encoding, optimizer and evaluation split fixed.
- **Control:** A fresh MHA-A run from the immutable `configs/m2/tinystories_v3_mha.json` copy, same seed, source revision, data split and evaluation protocol. The existing M1 runs use the older estimator and are not the control for this experiment.
- **Training budget:** First screen one paired seed (17) at approximately 5% of the MHA control's estimated compute: 51 MHA updates / 104,448 tokens / 677,905,956,864 FLOPs versus 54 GQA updates / 110,592 tokens / 674,296,233,984 FLOPs, a -0.53248% candidate delta. Use five warmup updates for both screen arms. Do not launch until the corpus manifest hashes verify. If stable, preregister a fresh three-seed confirmation at the full budgets above before training it.
- **Evaluation metrics:** Primary full-split validation mean NLL; report the paired candidate-minus-MHA mean and every seed delta. The predeclared practical loss budget is +0.02 nats/token, approximately the M1-A between-seed SD (0.01920) and about a 2% perplexity increase on this synthetic diagnostic. This is a decision threshold, not a formal significance test or broad capability claim. Also report perplexity, byte-normalized bits per UTF-8 byte, model tensor bytes, estimated training FLOPs, fixed-workload KV-cache bytes, prefill tokens/sec, cached-decode tokens/sec, and process peak RSS. The pilot is a stability/resource check only.
- **Result:** Pending; no training performed for this entry.
- **Interpretation:** Pending. The three-seed confirmation is practically acceptable on the selected synthetic diagnostic only if its mean paired NLL increase is at most +0.02 nats/token and the measured cache reduction is present under the fixed workload. Report every paired result and uncertainty; no significance or broad-capability claim follows from this threshold.
- **Next experiment:** Complete the separate 32-update-per-arm stability smoke, including an interrupted checkpoint/resume path, evaluation and profiling. It is a resource and plumbing check only. If that succeeds, run the preregistered 51/54-update single-seed screen; keep its score descriptive and move to the three paired confirmation seeds only if numerical stability and the cache-memory hypothesis survive.
