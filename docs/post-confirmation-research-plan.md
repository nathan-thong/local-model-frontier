# Post-confirmation research plan

Status: conditional research execution plan, 23 September 2026. The first M2-GQA-001 full-budget attempt failed the comparator's warmup-fraction recipe gate and remains ineligible. The corrected M2-GQA-002 passed paired comparison eligibility on synthetic TinyStories: mean NLL delta -0.02307 nats/token, sample SD 0.02439, with exactly half the KV bytes and about 8.5% slower CPU decode. It is not a broad capability claim. Q08's scoped CPU probe now records sampled current memory, but only two boundary samples were captured for a 13 ms step and CUDA remains unvalidated. Q09's bundle was verified from a detached clean checkout; independent same-host fixture reproduction also passed from a separate clone, but M2-GQA-002 has not been independently rerun. Q10 corpus preparation and Q11 evaluation freeze are next, before any new learning comparison. This plan defines future work and does not authorize paid compute or publication.

## Objective and working rules

Find out, one falsifiable change at a time, whether small sequence models can retain useful task capability under the same actual resident-memory and training-compute budgets as a conventional Transformer. Optimize evidence quality and fast rejection. Novelty, parameter count, checkpoint file size, loss on synthetic stories, and estimated FLOPs by themselves are not wins.

Each package is one handoff to Sol or Luna. Begin with the assigned design section and current Git state. State the package ID and gate in the handoff. Change one cause at a time. Add tests for failure modes, run focused and full required checks, record commands and hashes, and stop if a dependency gate fails. Do not start the next scientific campaign until its data, controls, compute definition, and metric are frozen. Preserve every failed run and report every planned seed.

Use this budget ladder throughout: correctness tests; <=32-update smoke; one-seed screen at <=5% of confirmation compute; three or more paired confirmation seeds; independent rerun. A screen can kill an idea but cannot establish an improvement. Estimate the campaign cap before training, including controls, tuning, teacher inference, failed trials, and profiling where material. Do not silently repeat jobs or select only successful seeds. Do not change the active GQA source revision or data. Do not push.

## Gate 0 — close the active M2-GQA-001 confirmation

**G00. Finish the six planned jobs.** Resume the existing serial plan at its recorded checkpoint under its per-invocation time cap. Confirm all three MHA controls and three GQA candidates reach their planned tokens and estimated FLOPs. Stop for invalid provenance, non-finite state, budget overrun, or corrupted checkpoints; preserve the cause and do not replace failed seeds.

**G01. Evaluate and profile every completed run.** Use the frozen evaluation and workload. Capture complete validation NLL, PPL and BPB; parameter and tensor bytes; training tokens, updates, FLOPs, throughput and process peak; actual cache bytes; prefill/decode timing distributions; cached/full-logit parity. Profile runs serially on this CPU. Record unavailable CUDA and energy fields as unavailable with reason.

**G02. Run the paired comparator and audit the artifacts.** Require identical pinned source/data/tokenizer/evaluation contracts, three actual paired seeds, finite outcomes, clean provenance, and the frozen compute tolerance. Recalculate the pairwise deltas independently from summaries. Retain failures. The +0.02 nats/token practical screen is not a p-value or proof. Cache reduction is a separate resource result from quality.

**G03. Write the result and close the revision.** Append all eight experiment fields and all per-seed deltas to `EXPERIMENTS.md`; mark every artifact local or portable, and state the synthetic-only scope. Update queue status and the next handoff. Run tests/style only if source changes; check Markdown paths and diff; commit the record locally. Do not push.

**Branch rule:** If cache correctness or compute/provenance fails, repair only the harness and repeat the affected controls and candidates from a new clean revision. If GQA fails the quality screen or latency/resource tradeoff is dominated, record a negative result and stop this branch. If it passes, report the narrow finding and proceed to other independent sequence candidates; do not generalize beyond this corpus. If results are noisy around the margin, call inconclusive and preregister more seeds before spending them.

## Gate 1 — make claims and future comparisons auditable

**Q08. Runtime and memory accounting closure.** Implemented in profile schema 4: isolated CPU phase peaks; nested allocated/active state bytes by unique backing-store intervals; cumulative runtime across graceful resumes with interrupted-invocation recovery. Tests verify order isolation, aliases, opaque-state failure, and both resume paths. Remaining E08 items are CPU optimizer-step peak isolation and actual CUDA validation.

**Q09. Portable evidence bundle.** Export a sanitized per-run summary, resolved config, protocol version, data/tokenizer/source digests, command, software/hardware fields, metric schema and artifact-availability map. Never export corpus text, checkpoints, usernames, absolute machine paths, or secrets. A clean clone must validate every included hash and reproduce the comparison metadata.

**Q09b. Comparison failure matrix.** Add adversarial tests for missing versus null fields, non-finite/empty metrics, duplicated seeds, skipped failed seeds, differing tokenizers/splits, dirty revisions, invalid or unknown module compute, and corrupted/missing profiles. Every refusal must name its blocker. Keep metric comparability, claim eligibility and latency comparability as distinct outputs. This is a regression audit of completed Q03, not permission to weaken or re-open its gates casually.

**Q10. Corpus decision and benchmark freeze.** Audit candidate human-authored corpus terms at primary sources before acquiring it. If no suitable legal corpus is available, keep the TinyStories work explicitly diagnostic and do not use it for a general-capability claim. For an approved corpus: freeze revision, split by document before tokenizer fitting, normalize and audit duplicate/near-duplicate overlap, preserve a held-out set, hash all artifacts, and verify deterministic preparation.

**Q11. Downstream benchmark freeze.** Build a short list of known-answer, contamination-checked tasks with versioned adapters, documented scoring and stable samples. Keep training-visible, validation and sealed test examples separate. Validate every scorer against hand-checked fixtures before any model is tuned against it. If no suitable tasks are legally and operationally available, explicitly defer broad capability claims.

**Q12. New conventional control.** Train a modest conventional control from scratch on the frozen corpus with its tokenizer, context, optimizer, precision and evaluation protocol. Save all three paired seeds and complete profiles. Build a separate FLOP ledger for the new model/data/tokenizer; do not compare its token PPL to old byte-token PPL. Gate: reproducible data, no overlap, finite runs, useful task metrics, and full evidence bundle.

## Gate 2 — sequence-module campaign (Milestone 2)

**S02. Make sequence state explicit.** Specify per-layer state inputs/outputs, absolute token position, reset semantics, cache dtype/layout, and state-byte reporting. Test full versus token-by-token and chunked execution, mixed block types, batched variable lengths, reset, and checkpoint round trips. No training.

**S03. Dense local-attention oracle.** Implement causal window masking with a slow, obvious correctness oracle. Test left/right boundaries, window one, window >= context, padding, gradients, and cached decode against full attention. Explicitly call dense-mask prefill dense compute; do not claim sparse FLOPs or speed.

**S04. Local-attention smoke and screen.** First test a synthetic copy/retrieval dependency that requires context across the window boundary; then run one-seed <=5% language-model screen with identical baseline recipe. Gate on finite training, correct bounded state, useful quality signal and measured workload relevance. Stop if the task requires unavailable context or it is already dominated.

**S05. Decide whether optimized local prefill is warranted.** Profile the target prompts and hardware. Only if dense masking is the measured bottleneck, implement one backend/kernel and retain the oracle. Require numerical/gradient parity, exact cache bounds, no hidden materialization, and end-to-end latency and peak-memory improvement under identical workload. Otherwise close this task with the negative profile.

**S06. Normalized causal linear-attention reference.** Implement one well-specified recurrent/linear formulation, not a module zoo. Test causal no-future leakage, denominator stability, finite gradients, chunk/full/token parity, reset, variable lengths, and state size independent of context. Verify on copy, delayed retrieval, and simple associative-recall fixtures. Stop if it cannot learn the fixtures reliably.

**S07. Linear-module screen and optimization gate.** Compare the reference to the new conventional control under a declared equivalent compute budget. Measure real Python overhead and state. Run one-seed screen only after numerical gates. Build a fused/vectorized path only if the profiler says the Python recurrence is material; parity to oracle is mandatory.

**S08. One fixed hybrid schedule.** Choose exactly one schedule before seeing language-model results (initial proposal: three cheap blocks and one full-attention block in a four-layer stack). Compare pure attention, pure cheap block, and hybrid using the same width, FFN, optimizer, tokenizer/data and declared training FLOPs. Do not search schedules in this task. Screen one paired seed; promote only a plausible, non-dominated result.

**S09. Confirmation and selective schedule analysis.** Confirm only survivors with at least three paired seeds. If placement/window remains a live question, use at most four validation configurations, charge all search compute, and keep test data sealed. Separate quality eligibility from latency eligibility; compare latency only on matching hardware/backend/precision/thread and fixed warmup/workload.

**S10. Context-length generalization.** Train fresh controls and candidates with a declared context policy. Compare matched contexts, plus a preregistered longer held-out context if relevant. No extrapolating a short-context checkpoint and calling that equivalent training. Record retrieval/position-specific failures, not only aggregate PPL.

**S11. Sequence result freeze.** Publish an evidence bundle and concise technical report for all surviving and killed branches. An independent clean-clone rerun must reproduce hashes, metric calculation and broad effect direction. If not, report unresolved. Freeze a stable reference API before compression work depends on it.

## Gate 3 — low-bit weights and real memory (Milestone 3)

**C01. Define formats and ledger.** Specify exact signed values, group axes/sizes, scales, rounding, zero handling, packing order, padding, metadata and byte accounting for FP16/BF16, int8, 4-bit and ternary. Include master weights, scales, optimizer state, activations, temporary unpack buffers, runtime, and KV state in training/inference ledgers.

**C02. Fake-quant reference.** Add simulated quantization with explicit straight-through estimator and deterministic behavior. Test exact representable values, clipping, scale gradients, degenerate/all-zero groups, serialization, finite gradients, and equivalence to dequantized reference. Make clear fake quant does not reduce resident training storage.

**C03. Falsification screen.** Preregister a small format set (for example int8, int4, and one ternary definition), one quantization granularity, and the transition schedule. Screen <=5% compute; include unquantized matched control. Stop unstable or already dominated formats; do not cherry-pick bit width and scale after inspecting validation.

**C04. Quantization schedule ablation.** Compare start-from-step-one, warmup-then-quantize, and one gradual schedule only if C03 survives. Hold total optimizer FLOPs and data fixed; include all floating optimizer/master-state costs and report the actual peak. Promote at most one recipe.

**C05. Packed checkpoint contract.** Implement versioned packed tensors, scales, shape/layout/endianness, checksum, provenance and strict load validation. Tests must reject truncation, invalid scale, wrong dimensions, unknown versions and hash mismatch. File size is not resident memory.

**C06. Packed CPU reference inference.** Measure process RSS/peak and model allocations after load and during forward; count unpacked copies and scratch memory. Benchmark the same prompts and precision against the floating control. Establish a trustworthy floor even if slower; no speedup claim from smaller serialization.

**C07. Native kernel gate.** Build one native/backend-specific kernel only if C06 profiles matrix unpack or low-bit matmul as a material end-to-end bottleneck and a real target device exists. Keep portable fallback; verify output error, supported shapes, load cost, cache interactions, peak memory and full workload latency. Drop the kernel branch if it does not improve measured frontier.

**C08. Quantization robustness.** Repeat the surviving format over at least three paired seeds and one held-out task suite. Compare quality and actual resident memory with a fair conventional model allowed to spend the same memory budget. Include a control trained directly at smaller width if that is an available competing use of memory.

## Gate 4 — teacher use and distillation (Milestone 4)

**D01. Teacher/data/cost ledger.** Freeze teacher identity, version, inference precision, license/access, prompts, filtering, generated corpus hashes and all acquisition/generation compute. Never imply student-only FLOPs are total system cost.

**D02. Student SFT control.** Train student conventionally on exactly the examples available to the proposed distillation method. Freeze the downstream task suite and baseline before producing targets. Gate: reproducible three-seed student control and validator with no test leakage.

**D03. Logit-distillation math and cache.** Implement temperature-scaled teacher/student KL with correct masks, reductions and optional T-squared convention explicitly tested against hand calculations. Cache targets only with teacher/tokenizer/data/config hashes; support streaming to control storage. Compare on same input tokens and student compute; report teacher-inclusive and student-only costs.

**D04. Logit distillation screen.** One temperature and one target precision first; only preregister a tiny temperature ablation if the first screen survives. Include ordinary SFT on identical teacher-visible examples so extra data is not misattributed to soft labels. Kill if improvement is below task noise or teacher cost makes the system frontier worse.

**D05. Representation distillation.** Only after D04 survives. Choose one layer mapping/projection and one representation loss. Test shape mapping, normalization, gradients and teacher/student alignment on deterministic fixtures. Keep token targets and compute otherwise fixed; report projection cost and do not combine with trajectory targets.

**D06. Task/trajectory distillation.** Define task correctness validators before generation. Exclude held-out prompts and answers from training targets; audit contamination. Score reasoning/action traces by validated final task outcome and total teacher/student tokens, not stylistic similarity. Compare to SFT and logit distillation separately.

**D07. Quantization-distillation interaction.** Run only if one quant recipe and one distillation recipe each independently survive. Preregister all four cells (float+SFT, float+distill, quant+SFT, quant+distill), matched budgets, all seeds and interaction estimate. Do not report only the best corner.

## Gate 5 — conditional compute at inference

**I01. Freeze workloads and repeatability.** Define prompt lengths/content strata, output lengths, batch size, concurrency, warmup, threads, device, precision, synchronization, repetitions and P50/P95 reporting. Measure repeated baseline variance first; reject comparisons smaller than noise.

**I02. Static compute-allocation control.** Compare fixed shallow/deep or small/large paths at matched latency/compute budgets before adding a learned router. Include model(s), caches and routing overhead in resident memory. Report quality by difficulty slice and tail latency.

**I03. Greedy speculative decoding correctness.** Implement draft/target acceptance and fallback; for deterministic greedy output require exact token-for-token equality across toy exhaustive cases and real checkpoints. Count draft plus target work and bytes. If it cannot preserve output, stop before speed tests.

**I04. Sampling correctness.** Only after I03. Test acceptance correction against analytically enumerable toy distributions and statistical checks with preregistered tolerances. Label approximations plainly; do not claim exact acceleration for heuristic rejection.

**I05. Speculative workload screen.** Compare fixed generation distributions, short and long outputs, and multiple batch sizes. Include drafter training/acquisition cost, separate caches, verification latency and degraded draft cases. Stop if accepted tokens per verification do not beat measured overhead or total resident memory violates budget.

**I06. Adaptive-depth/router screen.** Start with a fixed allocation baseline, then test one small router. Freeze features, calibration split and decision threshold before held-out scoring. Include router FLOPs, route imbalance, worst-case memory, fallback and P95 latency. Keep only if it beats the static control at equivalent budget on held-out tasks.

**I07. Device-specific deployment pass.** Port only the validated surviving path to the actual target CPU/GPU/NPU. Report build/toolchain/backend and fallback. Measure energy/token only with declared sensor, boundary, integration method and uncertainty; otherwise leave unavailable.

## Gate 6 — combine, replicate, publish

**X01. Select no more than two independently surviving interventions.** Freeze a factorial/ablation matrix including the conventional control and each individual intervention. Match total student/training compute, data, memory target and evaluation. Charge tuning, teacher and search cost.

**X02. Full frontier table.** For each valid point record task score plus uncertainty, corpus NLL/BPB, parameter and checkpoint size, actual resident/peak memory, KV/state bytes, prefill/decode P50/P95, tokens/sec, training/teacher compute, wall time and energy or reason unavailable. Compare only compatible tasks and hardware. Use Pareto dominance, not an invented scalar blend.

**X03. Independent replication.** Provide a clean-checkout reproduction command and sanitized immutable artifacts. A second environment or operator reruns the decisive control and candidate with frozen seeds/protocol; report deviations and effect direction. Resolve or disclose discrepancies.

**X04. Technical report and release gate.** Write methods before conclusions; include preregistration, source/data/license, model configs, estimator assumptions, negative results, failure ledger, confidence intervals/seed plots, limitations and reproducibility instructions. No claim outruns the task/data/device scope. Re-run CI, package build, documentation links and secret/artifact scan. Publishing remains a separate explicit action; this plan does not authorize pushing.

## Immediate post-sweep sequence

1. Complete G00-G03 and make the tracked queue match the observed sweep status.
2. Run Q08-Q09 so runtime accounting is bounded and future evidence can survive beyond the workstation.
3. Resolve Q10-Q11 corpus and downstream-task eligibility before broad capability language; if that blocks, continue correctness and profiling work only.
4. Work S02-S04 (state contract/local reference) without training the full confirmation budget. Gate every architecture on correctness first.
5. Screen only the best bounded local candidate and one linear reference. If neither survives, stop the expensive architecture branch and report that result.
6. Try one hybrid only if its constituent cheap block survives. Confirm and freeze sequence results before C01.
7. Implement fake quantization and storage accounting before any native kernel. Establish SFT and teacher ledger before distillation. Establish fixed inference workloads before speculation or routing.
8. Combine at most two survivors, replicate independently, then write X02-X04.

## Ready-to-send Sol/Luna task prompt

> Read `AGENTS.md`, `RESEARCH.md`, `EXPERIMENTS.md`, `ROADMAP.md`, `docs/implementation_queue.md`, and this plan. Check the live Git status and experiment plan first. Implement only the first unfinished package whose dependencies are complete. Follow its exact acceptance gate; make one coherent change, add failure-oriented tests, run focused and full required checks, and update its status with evidence. Do not alter files or configs used by an active pinned run. Do not download data, provision compute, publish, or push. If the package is a scientific run, preregister it, pin a clean revision, account for every planned seed/failure and stay within the declared cap. End with files changed, validation, observed result, limitations and the next package.
