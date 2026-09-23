# Sequence-module experiment plan

Status: S01 correctness/config work, S02 explicit-state work, S03 dense local-attention correctness, and S06's normalized-linear C0 correctness/resource reference are implemented and tested. Q09-REPRO-004 independently reproduced all seven aggregate M2-GQA-002 comparator metrics at clean revision `dbab90d`; its result remains limited to synthetic TinyStories and a workload-specific CPU resource tradeoff. Q10 data/tokenizer and Q11 task/baseline gates are complete. S06-DP-001 completed at clean revision `1874c4d` but failed its fixed retrieval thresholds; the machine-readable metrics and artifact hashes are in [`s06_dependency_screen_result_v1.json`](s06_dependency_screen_result_v1.json). Do not retune or begin architecture-comparison training from this result. Any future paired architecture comparison still needs fresh same-revision controls. Each package remains a small change or bounded experiment; preserve negative results and use the experiment template.

## What the code currently supports

`ModelConfig.sequence_types` selects one registered sequence module per layer. `DecoderCache.position` is the absolute position for the next token; attention state explicitly stores its absolute key-start offset. The registered references are full causal attention with optional grouped KV heads and dense masked local attention. The local reference preserves local-window semantics but still executes dense score/value work. Baseline A has four query heads and four KV heads, so it is MHA; existing GQA support is not evidence that a trained GQA control exists. S01 checks GQA against an explicit repeated-K/V forward/backward reference, tests the two-to-four KV cache-storage ratio, and includes a compute-matched config. Attention expands grouped keys/values with `repeat_interleave`, which can create temporary memory beyond persistent KV storage. The model rejects positions beyond `max_seq_len`. Profile schema 5 writes per-module cost/state descriptors and state-accounting version 2. The compute estimator charges the local reference at dense context cost; it must not be applied unchanged to a sparse/windowed backend or recurrent blocks.

All performance claims below require the shared accounting and comparison work in the main roadmap. Parameters, resident memory, token count, training FLOPs, and wall time are separate controls. Preserve the same source revision for both arms by landing all implementation changes before training either arm. Three paired seeds are the minimum for reported method comparisons; one seed is only a screening run.

## S01 — Establish a trained GQA control

- **Prerequisites:** Frozen corpus/tokenizer/evaluation protocol and valid module-aware compute ledger; CPU fixture tests can proceed before that ledger exists.
- **Change and code:** Add a GQA config derived from `configs/baseline_a.json`, changing only `kv_heads` from 4 to 2 while retaining four query heads. Touch configs, `tests/test_architecture.py`, and accounting fixtures; avoid changing normalization or FFN at the same time.
- **Tests:** Compare GQA logits and gradients against an explicit reference with repeated K/V heads; verify causality, chunked cache parity, and KV byte counts. Verify the theoretical persistent-state ratio is one half at identical sequence length/dtype and distinguish it from peak process memory.
- **Experiment:** First one short stability pilot, then paired seeds 17/42/123 against fresh MHA A runs with equivalent disclosed training compute. Derive actual token budgets before running and preserve the same learning-rate schedule policy expressed as budget fractions. Record the changed token exposure. Report NLL, downstream scores, model bytes, persistent state, prefill/decode timing, and peak memory.
- **Gate:** Advance with a correct, stable GQA control even if capability decreases. Call it a control, not an improvement. If the change is dominated in measured workloads, retain MHA as the primary control and retain GQA as the relevant cache comparator.

## S02 — Make sequence state explicit and inspectable

- **Prerequisites:** S01 invariants; this is a contract change, not an architecture experiment.
- **Change and code:** Complete. `models/sequence/interface.py` defines `AttentionState`, `RecurrentState`, recursive tensor traversal, and cost/state descriptors. `DecoderCache.position` remains the absolute next position, and `AttentionState.key_start_position` supports a retained suffix. `models/language_model.py` and `profiling/benchmark.py` pass and report the contract; `inference/generate.py` needed no change because it already consumes the model's cache API. Invalid batch/device/dtype/shape/position state fails with specific errors.
- **Tests:** Complete. Nested and shared tensor storage is counted once; allocated capacity is separated from the union of active regions; no-cache and first-cache paths work; offsets survive uneven chunk/token decoding. Existing checkpoint parameter names and shapes load without migration.
- **Experiment:** Complete, with zero training. The same Q11 seed-17 checkpoint retained identical full/chunk/token logits digests and cache bytes before and after. Profile schema advanced from 4 to 5 and state accounting is identified as version 2. Exact hashes and validation are in [`docs/s02_state_result_v1.json`](s02_state_result_v1.json).
- **Gate:** Passed for the explicit-state contract. Full/chunk/token parity and CPU state-byte accounting pass. Bounded windows remain unimplemented; do not add them on top of implicit zero-based key indices.

## S03 — Implement a dense local-attention correctness reference

- **Prerequisites:** S02. Use `window_size=128`, context 256 only as a diagnostic starting point.
- **Change and code:** Implemented in commit `e6d77c65fd3571d02fc6844ad026ae5ba3d406c2`. Tests ran against a dirty working tree based on `fd3f2c462877d6a6c82d2207644a0ca007b7b06f`; the tested source changes were committed unchanged. `models/sequence/local_attention.py` registers `local_attention_reference`, reuses GQA projections/RoPE, validates a module-specific positive `window_size`, and constructs the window from absolute positions. The combined prefix and current chunk remain intact until every query is evaluated; only the final W-1 prior keys are copied into the next contiguous cache.
- **Tests:** Complete. Fixed-weight outputs and gradients match a slow explicit per-query-slice oracle; a covering window matches conventional attention. W=1, W larger than input, W-1/W/W+1/2W+1 lengths, exact W-sized prefill boundary, odd chunk partitions, full/cache parity, future/exclusion checks, nonzero absolute offsets, MQA/GQA/MHA, RoPE/learned positions, RMSNorm/LayerNorm and pre/post norm pass. The actual cache storage matches the bound, including a zero-byte cache at W=1. See [`s03_local_attention_result_v1.json`](s03_local_attention_result_v1.json).
- **Experiment:** Fixed-weight correctness and state accounting only; no learning fixture or training was run. The descriptor labels the implementation `dense_masked_local_attention`; the estimator charges full dense attention work. CUDA, latency and prefill savings were not measured.
- **Gate:** Passed for fixed-weight semantics and bounded retained state. No prefill speed or FLOP-saving claim follows. S06's C0 correctness and CPU resource reference also pass, but S06-DP-001 failed its preregistered learning thresholds. Defer S04 training; if the sequence campaign is reopened, first clear a separately justified dependency gate and freeze a fresh same-revision GQA control under Q10/Q11's frozen train/validation evaluation protocol.

## S04 — Test the local-attention capability tradeoff cheaply

- **Execution order:** Closed for now: S06-DP-001 failed its preregistered dependency-learning gate. Reconsider only after a separately justified, versioned screen is preregistered and passes. This ordering is a research-sequence decision; S04 still tests the dense local block independently.

- **Prerequisites:** S03; a separately justified, versioned dependency-learning gate that passes (S06-DP-001 failed); accounting that prices the actual dense implementation; frozen corpus and evaluation. For any comparison, train a fresh conventional GQA control and the local candidate at the same clean revision.
- **Change and code:** Add a single local-attention pilot config and planned `EXPERIMENTS.md` entry. Preserve heads, width, FFN, depth, tokenizer, and data.
- **Tests:** Run S03 invariants and deterministic resume on this module before training. Ensure evaluation and generation use the same window semantics as training.
- **Experiment:** One seed at a small predeclared fraction of baseline compute; record validation loss over consumed compute and simple position/distance-stratified retrieval diagnostics. This asks whether local attention is stable and whether obvious quality collapse occurs; it does not establish a frontier result. If stable, run three paired seeds against full GQA at equivalent compute under the actual implementation.
- **Gate:** Stop the candidate if it diverges or fails basic dependency tasks within its permitted span. If quality is worse but state memory falls, retain it as a measured tradeoff. Do not abandon all local attention from one short-context pilot; choose the next context/window test from the observed failure.

## S05 — Add one implementation that actually performs local prefill work

- **Prerequisites:** S03/S04; profile evidence that local attention is worth optimizing on available hardware.
- **Change and code:** Add an explicitly named backend to `local_attention.py`. Begin with query chunks and only their corresponding key bands, or integrate one maintained windowed kernel after checking current primary documentation and license. Keep the dense implementation as the reference. Record backend selection, fallback, and actual execution path in run artifacts.
- **Tests:** Compare outputs and gradients to S03 across head grouping, windows, nonmultiple lengths, boundaries, prefill chunks, and supported precision. Ensure backend fallback cannot silently retain a sparse-cost label. Test exceptions for unsupported shapes instead of quietly reporting invented savings.
- **Experiment:** Microbenchmark identical weights over context lengths 128/256/512/1024 and windows 32/128/256, limited by validated model context. Measure allocations and elapsed time; count the actual tiled work including overlap and padding. Backend-only comparisons use identical weights and need no retraining if numerically equivalent.
- **Gate:** Keep the optimized backend only when it wins a useful measured workload or enables a larger context within the same memory limit. A Python chunk loop may save memory while losing latency; report both. Do not build custom CUDA kernels before locating a measured bottleneck.

## S06 — Implement normalized causal linear attention as a reference

- **Current priority:** The C0 implementation, correctness and CPU resource reference are complete at `502014799903fc9500cebf109acc05eed8ba2e5c`; see [`s06_linear_reference_result_v1.json`](s06_linear_reference_result_v1.json). S06-DP-001 completed at clean revision `1874c4d` but failed its preregistered retrieval thresholds; see [`s06_dependency_screen_result_v1.json`](s06_dependency_screen_result_v1.json). Keep architecture-comparison training closed. Do not retune this result.

- **Prerequisites:** S02 and recurrent-state compute accounting. Read and cite the linear-attention paper already linked in `RESEARCH.md` before implementation; use its established formulation rather than claiming a new block.
- **Change and code:** Implemented `normalized_linear_attention_reference` in `models/sequence/linear_attention.py` with positive feature map `phi(x)=ELU(x)+1`, state `S=sum(phi(k) outer v)` and `z=sum(phi(k))`, and output `phi(q)^T S / (phi(q)^T z + epsilon)`. The token-loop reference stores FP32 accumulators, resets by omitting state, and validates contiguous absolute positions. RoPE configs apply RoPE to projected Q/K before the feature map; learned-position configs retain the decoder's learned embedding on hidden states. This position treatment is part of the reference and must remain frozen or be controlled in any comparison.
- **Tests:** Independent prefix-oracle outputs and gradients; MHA/GQA/MQA; full/chunk/token parity; RoPE/learned positions; CPU bfloat16-autocast output against FP32 with FP32 recurrent state and finite gradients; finite values for near-zero denominators; no future leakage; exact state reset; bounded state bytes independent of context; and long-prefix gradients. A separate mathematical gradcheck remains optional.
- **Experiment:** C0 correctness and CPU resource profiling used zero training. S06-DP-001 used the fixed three-seed, 256-update and 39B estimated-FLOP budget; training was finite and complete, but overall exact match was 15.63%–20.31% and every joint stratum remained below 23.44%, below the 75%/50% rules. The result and hashes are in [`s06_dependency_screen_result_v1.json`](s06_dependency_screen_result_v1.json). Do not start sequence-block comparisons from this failed gate.
- **Gate:** Stop if numerically unstable or unable to learn the fixture. Retain an honest slow reference if correct. A Python recurrence is not evidence for or against optimized recurrent latency.

## S07 — Decide whether a recurrent implementation deserves optimization

- **Prerequisites:** S06 stability and a visible quality/state-memory tradeoff.
- **Change and code:** Implement one mathematically equivalent scan/chunk formulation or integrate one maintained backend. Keep precision and epsilon behavior pinned; register the backend and its constraints. No gated recurrence, new feature maps, or new normalization in the same change.
- **Tests:** Cross-backend output/gradient/state parity, chunk-size independence within tolerance, resume equivalence, and measured memory scaling. Distinguish training autograd storage from persistent inference state.
- **Experiment:** Prefill and decode sweeps against S06 and GQA on the same device and input shapes. Profile actual hotspot coverage and allocations. Record scan overhead and crossover context length. If training kernels change numerical behavior materially, run fresh paired controls rather than reusing old results.
- **Gate:** Only optimize further when profiling identifies a tractable hotspot. If the block loses capability at every useful budget, stop kernel work and document the negative result.

## S08 — Test one hybrid schedule

- **Prerequisites:** Correct full attention and one cheap module, valid per-layer state accounting, and a frozen baseline on the target corpus. Train fresh all-attention, all-cheap, and hybrid controls/candidates on one clean revision before comparison.
- **Change and code:** Use existing `sequence_types` to define a four-layer schedule `[cheap, cheap, cheap, attention]`; compare with all-attention and all-cheap. Add per-module config only where the global config cannot express the intended block. Do not add a schedule searcher.
- **Tests:** Mixed state dataclasses in a single `DecoderCache`; independent layer offsets; full/chunk/token parity; attention layers grow their state while recurrent layers remain bounded; actual aggregate state bytes equal the sum of unique layer storages. Test checkpoint/resume and generation for the mixed schedule.
- **Experiment:** One seed pilot, then three paired seeds at equivalent training compute. Keep an additional equal-token diagnostic if useful, labeled separately. Include long-distance retrieval to test whether the occasional full-attention layer repairs the cheap module's observed failure. Report the attention fraction and per-layer cost breakdown.
- **Gate:** Advance if the hybrid has an undominated quality/resource tradeoff within measurement uncertainty. If not, test at most one predeclared alternative schedule, such as `[cheap, attention, cheap, attention]`, then stop rather than searching indefinitely on the held-out set.

## S09 — Separate window size from attention placement

- **Prerequisites:** S08 has a plausible signal and the test set has not been used for tuning.
- **Change and code:** Add at most four preregistered configs testing a small factorial: one/two full-attention layers and short/long local window (or pure recurrent/full attention counts if that is the selected cheap block). Keep tuning results on validation data.
- **Tests:** Generated schedule length equals model depth; each configuration's cost/state descriptors match instantiated modules; invalid module-specific settings fail early.
- **Experiment:** Screen under equal total tuning budgets, select one candidate on validation metrics, then train fresh three-seed candidate/control pairs for the fixed final evaluation. Include cumulative search compute in the ledger and separate it from per-run training compute.
- **Gate:** Stop the search when the predeclared budget is exhausted. Do not turn the best noisy single-seed score into a claimed gain. Preserve every screened config and rejection reason.

## S10 — Run context and memory sweeps with fresh controls

- **Prerequisites:** At least one candidate from S05/S08; robust memory profiling and pinned workload definitions.
- **Change and code:** Add configs for contexts 256/512/1024 where hardware and dataset support them, with fresh GQA controls trained at each context and equal per-comparison compute. Explicitly handle `DecoderLanguageModel.max_seq_len`: increasing this cap is a configuration/protocol change, not evidence of length generalization. Do not silently extrapolate learned position embeddings. Add workload manifests to `profiling/benchmark.py` and serialized results.
- **Tests:** Boundary behavior at maximum position, final generated token accounting, cache/state allocated capacity, storage retained by tensor views, and no state leakage between requests.
- **Experiment:** At each context profile batch sizes 1 and a small feasible larger batch; fixed prompt/decode shapes; median and spread across repeats; startup and steady state separately. Measure full process RSS/VRAM including workspaces. For fixed-resident-memory capability comparisons, select feasible architectures under a predeclared measured memory cap and retain equivalent training compute; parameter count is not a proxy for this constraint.
- **Gate:** Report a workload-specific Pareto frontier with uncertainty. Exclude out-of-memory points transparently and do not infer a hardware-general winner from CPU-only evidence. Advance to low-bit interactions only after the architecture tradeoffs are reproducible.

## S11 — Freeze the sequence research checkpoint

- **Prerequisites:** S01-S10 completed or explicitly stopped with reasons.
- **Change and code:** Write a short technical report section containing hypotheses, negative results, module equations, implementation/backend details, hardware, compute ledger, and links to immutable run summaries. Update `RESEARCH.md` assumptions using measured evidence without erasing the original claims under test. Keep corpora and checkpoints out of Git.
- **Tests:** A fresh environment can recreate the smallest control/candidate fixture and load exported checkpoints. Every report number resolves to a machine-readable artifact and its source/config/data hashes. Comparison eligibility and uncertainty are visible beside each result.
- **Experiment:** Independent rerun of one selected control/candidate pair using the documented command path; this is a reproducibility check rather than another tuning opportunity.
- **Gate:** If the candidate does not survive the rerun, mark the result unresolved and investigate before composing it with low-bit training or distillation. A negative architecture conclusion is an acceptable deliverable.

## Implementation handoff

For each item, the implementing agent should read the named source files, write the planned experiment entry, implement only that item's scope, run architecture/profiling/comparison tests plus the repository validation commands, and leave a concise change report. Avoid allocating a large training budget before invariants pass. Finish each code change before pinning and launching its experiments; never repair source midway through a reported run. No item authorizes claiming a gain from simulated storage, estimated FLOPs alone, or a favorable parameter count.
