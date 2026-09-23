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
- **Result:** Model tensors were bitwise equal after both paths in `tests/test_resume.py`. The full suite passes: 21 tests, including [tests/test_resume.py](tests/test_resume.py).
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
