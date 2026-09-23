# Next execution plan

Status: planning document, 23 September 2026. The detailed module designs live in [`implementation_queue.md`](implementation_queue.md), [`roadmap-measurement.md`](roadmap-measurement.md), [`roadmap-sequence.md`](roadmap-sequence.md), and [`roadmap-compression.md`](roadmap-compression.md). This document orders that work into small handoffs and says when to stop. Only the first unfinished package is active; later packages are conditional plans, not authorization to spend compute, download data, or publish claims.

## Q09 bundle and bounded fixture reproductions: complete

The exporter, verifier, CLI, adversarial tests, sanitized six-run result bundle, and evidence documentation are committed in `cc921b37ddcaf4ef9a223678fd2984b88647afbb`. A detached clean checkout verified `results/m2-gqa-002/` with `frontier verify-evidence --bundle-dir results/m2-gqa-002`: status `verified`, bundle SHA-256 `df79b3578c70c6fa7c3a0d0a2d34e02f257cec16c14295287d73912d2b9bef3a`, eight inventoried files, and a clean worktree. The minimum deterministic fixture reproduction ran twice in the documented environment; Q09.f adds a separate clean-clone execution at the pinned current code revision. These runs verify fixture and checkout plumbing, not a full M2-GQA-002 training rerun.

| Step | Owner | Deliverable | Acceptance / stop rule |
|---|---|---|---|
| Q09.a Recover exporter | Sol | Repair `evidence.py` structure; make failed exports leave no partial destination; preserve distinct prepared-manifest and run-manifest hashes and label them accurately. | **Complete.** Compile, focused test, formatting, atomic-failure, and retry checks pass. |
| Q09.b Adversarial bundle tests | Sol | Cover malformed inputs, wrong seed membership, false eligibility claims, missing source artifacts, path/secret leaks, traversal, tampering, unlisted files, and semantic manifest equality with different serialization. | **Complete.** Forged or mutated bundles fail closed; under-three-seed eligibility is downgraded; valid bundles omit corpus text, tokenizer JSON, checkpoints, and weights. |
| Q09.c Export actual result | Luna | Export the six completed M2-GQA-002 run summaries with comparison and sweep-plan metadata into `results/m2-gqa-002/`; list omitted large artifacts and hashes. | **Complete.** The sanitized bundle verifies in place and after copying; no machine paths or credentials are present. Source run artifacts remain unchanged. |
| Q09.d Close the evidence protocol | Luna | Document exact export/verify commands, limits, artifact inventory, and machine-readable result links; commit one reviewed change locally. | **Complete.** Commit `cc921b37ddcaf4ef9a223678fd2984b88647afbb` passes the detached clean-checkout verifier with the recorded bundle hash and eight inventoried files. |
| Q09.e Clean-environment training reproduction | Luna | In a fresh documented repository environment, run the smallest deterministic CPU fixture twice from distinct output directories; compare model/checkpoint hashes and evaluation metrics. | **Complete, fixture scope only.** At clean revision `74bbe3d28f3f80a00fac3a03aac0108036cf4cdd`, both 30-update runs used 7,680 tokens and had identical model-state and `weights.pt` SHA-256 `43c47f2ef5785ce2e45ccff73bc9b8955a15a90373d11cf7955717798bc45fe7`, manifest SHA-256 `c70464681116bd8958c81710cb20768f06a82769d0654382a861960e8eaa9611`, train loss `4.678186893`, and validation NLL `4.824337328`. This verifies the minimal same-environment reproduction path, not an independent M2-GQA-002 training rerun. |
| Q09.f Independent clean-checkout fixture rerun | Luna | From a separate local clone at pinned revision `790a83ac5c60b26ed0bd983d267ae28374488bad`, run `configs/smoke.json` once with a fresh output directory; compare checkpoint hash, fixture manifest, update/token counts and validation metrics with Q09.e. Use the established Python environment and force imports from the cloned source tree. | **Complete, same-host fixture scope.** One 30-update / 7,680-token run from the separate clone reproduced `weights.pt` SHA-256 `43c47f2ef5785ce2e45ccff73bc9b8955a15a90373d11cf7955717798bc45fe7`, manifest SHA-256 `c70464681116bd8958c81710cb20768f06a82769d0654382a861960e8eaa9611`, and validation NLL `4.824337328`. The clone recorded the pinned revision and clean status. This is not an independent M2-GQA-002 comparison rerun. |

**Q09 stop condition:** If the exporter cannot safely sanitize a field, omit it and record that omission. Do not widen the allowlist just to retain a convenient field. If comparison eligibility cannot be reconstructed from source artifacts, export it as ineligible/unknown rather than trusting a supplied boolean.

## Q08 E08 CPU optimizer-step measurement: sampled CPU probe complete; CUDA open

At clean revision `53e9ed0`, the fresh CPU worker ran one production update with 16 accumulation microbatches and persisted its sampled RSS/private-commit record. Baseline was 282,738,688 RSS and 797,708,288 private bytes; sampled increases were 8,982,528 RSS and 3,522,560 private bytes. There were 11 readings during the 0.151018 s step, with observed gaps of 9.457–17.097 ms. This resolves boundary-only sampling for this workload, but the maximum remains sampled and can miss shorter transients. CUDA validation remains open because no suitable device was available. Do not turn this infrastructure probe into an architecture comparison.

## Q10.b source and deterministic preparation: next bounded package

The earlier Wikipedia acquisition plan is historical: Q10-SOURCE-001 cleared none of its 24 candidates, so no page body was requested. The Q10-SOURCE-002 PMC screen is also complete with a stop: 5/24 versions passed metadata checks (547,447 combined XML bytes), while 19 failed. Since the minimum is 20, no article XML is authorized. Q10-SOURCE-003 has preregistered a new PMC cohort that excludes author manuscripts and preprints; run its six ID-only queries, freeze the 48 candidate IDs and hashes, then screen metadata. Once a source passes its metadata and article-level rights gates, use deterministic extraction and document identities, keep corpus text out of Git, and require two clean preparations with matching source/content/split/tokenizer-input/manifest hashes before tokenizer fitting.

## Establish a useful evaluation domain before sequence training

Q10.a has been amended three times before text acquisition. OpenStax *Principles of Economics 2e* was withdrawn because its publisher page prohibits LLM training without permission. Q10-SOURCE-001's 24-page Wikipedia screen cleared zero pages under the full-history/attribution standard. Q10-SOURCE-002 froze and screened 24 PMCID candidates through six metadata-only searches: 5 passed (547,447 XML bytes), 19 failed, and the at-least-20 gate closed without any article body request. The exact-rights and date criteria were clarified before the screen completed. Q10-SOURCE-003 now preregisters a new filtered cohort; its queries have not run. No article text has been acquired. The source decisions and rejected alternatives are recorded in `docs/data_protocol.md`.

| Step | Owner | Deliverable | Acceptance / stop rule |
|---|---|---|---|
| Q10.a Dataset decision | Luna | Compare candidate corpora using primary dataset cards, licenses, source revisions, scale, language/domain fit, and local preparation cost. Write each decision and amendment in `docs/data_protocol.md`. | **Complete, amended before acquisition.** OpenStax was withdrawn because its publisher page prohibits LLM training without permission. WikiText lacks article revision mapping; full PG-19 is outside the first local-preparation bound. The Wikipedia screen cleared 0/24 pages; PMC cleared 5/24 metadata records and failed its minimum-20 gate. Q10-SOURCE-003 preregisters a separate 48-ID published-article cohort; its ID queries have not been run. |
| Q10.b Deterministic preparation | Sol | After a new source amendment passes its frozen metadata and rights gates, acquire only cleared source records. Apply article-level rights exclusions, then write deterministic train/validation/test identities and duplicate reports. | Two clean preparations yield identical source/content hashes, split hashes, tokenizer inputs and manifest hash. Exclude separately licensed text or media unless its terms and attribution are preserved. No normalized duplicate crosses splits. Keep corpus text out of Git and stop if rights cannot be verified. |
| Q10.c Tokenizer and held-out freeze | Sol | Fit the intended tokenizer on training documents only; freeze tokenizer artifact, validation set, and sealed test-set hashes. | No validation/test content affects tokenizer fitting; Unicode and special-token round trips pass. Hashes and access rules are recorded before tuning. |
| Q11.a Task-contract freeze | Luna | Select a tiny set of independently scored tasks with stable IDs, prompt/template version, scoring direction, generation limits, and provenance. Preserve byte-normalized likelihood where tokenizers differ. | Hand-calculated examples pass; evaluator is deterministic under the declared settings; no task overlaps training data by normalized content. |
| Q11.b Baseline evaluation record | Luna | Run existing conventional checkpoints through the frozen evaluator and add per-example, per-seed machine-readable summaries. | Missing task outputs and failed examples remain visible; test set is used once for the declared final evaluation, not model selection. |

The generic schema-4 preparation path now writes deterministic train/validation/test files, hashed normalized document identities and duplicate reports; legacy train/validation manifests remain supported. This closes only the splitter implementation gate. Q10-SOURCE-001 froze 24 exact English Wikipedia revisions with 3,272,433 combined bytes, but the review covered at most 500 recent history rows per page, so zero pages cleared full-history and attribution review. The body request gate stayed closed. Q10-SOURCE-002 froze 24 PMCID candidates across six queries. PMC metadata screening yielded 5 eligible versions (547,447 XML bytes) and 19 failures, below the 20-version gate; no article XML was requested. See both versioned candidate inventories and `EXPERIMENTS.md`.

Do not run an architecture comparison on a newly selected corpus until preparation, tokenizer, task, and baseline artifacts are frozen on one clean revision. TinyStories results remain useful for the named synthetic domain, not a substitute for broader evaluation.

## Compute discipline

- **C0, correctness:** unit tests, toy dependencies, and profiling smoke tests only; no long training jobs.
- **C1, screen:** one candidate seed, capped at 5% of the planned full paired-comparison FLOPs. Include setup, failed jobs, and every tuning attempt in the search ledger. Stop on divergence, invalid provenance, or a compute/profile mismatch.
- **C2, confirmation:** only after the screen passes, train at least three paired seeds on a clean pinned revision. No mid-run code fixes; any protocol/code fix invalidates both arms and requires a new revision and fresh pair.
- **C3, claim:** require eligible compute and data, a predeclared primary metric and practical-effect threshold, uncertainty, all failures, measured resource columns, and portable evidence. Passing eligibility is not evidence of a win.

At every stage, choose one primary metric and one primary workload before running. If its uncertainty cannot resolve the predeclared useful effect at the available budget, label the result inconclusive and stop or redesign; do not quietly increase budget or switch metrics after seeing outcomes.

## Milestone 2: test sequence ideas in the cheapest decisive order

Use the specs in [`roadmap-sequence.md`](roadmap-sequence.md). Each correctness package may run tiny deterministic fixtures; full training waits for Q10/Q11 and clean comparison eligibility.

| Step | Owner | Deliverable | Advance only if… |
|---|---|---|---|
| S02 State contract | Sol | Explicit attention/recurrent cache state, absolute positions, offsets, nested-storage accounting, and module descriptors. | Full-prefill, chunked-prefill, and token-decode parity holds; offsets and actual/active bytes are correct. No training. |
| S03 Local reference | Sol | Dense masked sliding-window attention, with W including the current token and bounded retained cache. | Boundary, gradient, causality, exclusion, odd-chunk, and slow-oracle tests pass. Label executed work dense. |
| S04 Local pilot | Luna | One fixed window candidate, one seed, predeclared small compute fraction, validation-only diagnostics. | Stable optimization and a real state/resource change. Stop immediately on divergence; do not call a pilot a gain. |
| S05 Local backend gate | Sol | First profile reference; only if it exposes a material bottleneck, add one tiled/chunked backend and keep the dense oracle. | Backend matches outputs/gradients and improves the declared device workload or enables a context under the same memory cap. Otherwise stop kernel work. |
| S06 Linear/recurrent oracle | Sol | One established normalized causal linear-attention formulation; explicit state/reset/position behavior. | No future leakage, finite long-prefix gradients, bounded inference state, chunk parity, and toy dependency learning. Otherwise stop before optimization. |
| S07 Recurrent optimization gate | Sol | Optimize one measured bottleneck with one scan/chunk backend, retaining the oracle. | Parity and real end-to-end benefit survive profiling. Slow Python recurrence is neither a positive nor negative optimized-kernel result. |
| S08 One hybrid | Luna | Fixed four-layer schedule using only a validated cheap block and full attention; compare all-attention and all-cheap. | State accounting and mixed-cache generation pass; pilot remains stable and justifies paired confirmation. Do not search schedules. |
| S09 Bounded schedule screen | Luna | At most four validation configs; include every screen in the compute ledger. | One candidate clears the preregistered practical threshold or the branch closes. No repeated peeking at held-out data. |
| S10 Context/resource profile | Sol | Fixed workloads at supported context lengths, fresh same-context GQA controls, allocated/active state and process/accelerator peaks. | Measurements include actual backend, shape, repetitions, resource applicability, and uncertainty; exclude OOM points transparently. |
| S11 Sequence decision | Luna | Paired three-seed confirmation of at most one survivor; portable evidence and report, including negative results. | Shared clean revision, same protocol, eligible compute/provenance, failures retained, independent reproduction. If no survivor, close Milestone 2 with that result. |

For any new claim, require at least three paired seeds, a same-revision control, same data/tokenizer/evaluation, equivalent disclosed training compute, and a declared practical effect. A test pass does not grant permission to expand the budget.

## Milestone 3: low-bit weights, first as an accounting problem

Use [`roadmap-compression.md`](roadmap-compression.md). Start only after conventional and sequence baselines are stable and evidence export is usable.

| Step | Owner | Deliverable | Kill gate |
|---|---|---|---|
| C01 Format ledger | Luna | Exact grouping, signed/ternary levels, scale/zero-point rules, rounding, padding, metadata and all training/inference storage categories. | If true packed resident bytes cannot be measured, do not claim memory benefit. |
| C02 Fake-quant oracle | Sol | Fake-quant linear with forward, STE/backward, saturation and serialization invariants; supported widths explicit. | Compare against a scalar reference and finite-difference sanity checks. Floating master/optimizer state remains charged in training. |
| C03 Cheap falsification | Luna | Small preregistered one-variable format screen with same-init/seed diagnostics as well as fresh-run controls. | Stop unstable or dominated formats. Do not silently promote the best noisy cell. |
| C04 Quantization schedule | Luna | Compare fixed schedules at equal student compute; disclose optimizer/master weights and all warmup/annealing cost. | Advance one schedule at most, only on a predeclared validation criterion. |
| C05 Packed format | Sol | Versioned packed checkpoint, scales/metadata, corruption checks, round-trip loader and exact size ledger. | Dequantized output tolerance and malformed-file rejection pass; no comparison based on file size alone. |
| C06 CPU reference | Sol | Execute a packed checkpoint end-to-end; compare identical weights and workload with float path. | Report resident memory and latency separately. A slower path may still be a memory tradeoff; no speed claim absent measured speedup. |
| C07 Native-kernel decision | Sol | Profile unpack/matmul hotspots and write a go/no-go before custom kernels. | Build only if a measured bottleneck and target hardware exist. Keep a tested fallback; stop if no likely practical win. |

## Milestone 4: teacher use and distillation, only after SFT control

| Step | Owner | Deliverable | Acceptance / stop rule |
|---|---|---|---|
| D01 Teacher ledger | Luna | Pin exact teacher revision/weights/tokenizer/license; record acquisition, generation, filtering, storage, and inference costs. | Unknown terms or unverifiable teacher artifacts stop the branch. Teacher cost is never hidden from total-cost reporting. |
| D02 SFT control | Sol | Deterministic student SFT batches, masks, packing, checkpoints, and equal-budget control. | Exact resume and fixed task evaluation pass before distillation exists. |
| D03 Logit reference | Sol | Temperature-scaled teacher/student logits, correct causal/padding masks, KL direction/reduction, and hard-label mixture. | Tiny tensors match analytical KL/gradient values; no padded token contributes. |
| D04 Target cache | Luna | Optional cached teacher targets, keyed by input/tokenizer/template/teacher hashes, with storage and generation ledger. | Cache reproduces online teacher targets exactly for sampled records; invalidated on any identity mismatch. |
| D05 Representation loss | Sol | One explicit layer mapping/projection and loss; isolate from logits objective. | Gradient and shape contracts pass; run only if D03 survives against SFT. |
| D06 Task/trajectory targets | Sol | Validated task target schema and contamination checks; isolate generated targets from representation effects. | No evaluation answer leakage; target provenance and filtering audit pass. Otherwise defer. |
| D07 Quantization × distillation | Luna | Full preregistered 2×2 factorial: SFT/quantized SFT × distill/quantized distill, matched student compute. | Estimate interaction with all cells; do not compare cherry-picked corners. Run only after both isolated interventions survive. |

## Inference and combined frontier: independent gates

| Step | Owner | Deliverable | Advance only if… |
|---|---|---|---|
| I01 Workload lock | Luna | Fixed prompt/output lengths, batch, warmup, repetitions, sampling, cache policy, device and latency statistic. | Repeated baseline noise is small enough to resolve the proposed effect; all cost categories are counted. |
| I02 Greedy speculation | Sol | Draft/target algorithm with exact token-stream equivalence; both cache update paths instrumented. | Every tiny deterministic case matches ordinary greedy decoding exactly; measure rejection rate, total latency, memory and draft work. |
| I03 Sampling speculation | Sol | Distribution-preserving acceptance/residual sampling with analytical tiny-vocabulary checks. | Empirical/analytical distributions match; do not approximate with greedy tests. |
| I04 Static allocation | Luna | Fixed small/large model routing or draft depth compared with fixed policies at equal average compute. | Quality/latency frontier improves after router, worst-case resident memory and tail cost are included. |
| I05 Adaptive depth | Sol | Learned/threshold layer exit only if I04 demonstrates useful predictable uncertainty signals. | Train/inference objective and auxiliary heads are costed; beat a smaller fixed model at equal average compute and memory. Otherwise close. |
| X01 Composition | Luna | Combine at most two individually surviving changes; fresh conventional controls and full interaction accounting. | No capability, memory, speed, or energy improvement claim without paired seeds, resource equivalence, uncertainty, and failure ledger. |
| X02 Pareto table | Sol | Workload-stratified nondominated records over capability, resident memory, state/KV bytes, latency, throughput and optional measured energy. | Unknown dimensions yield incomparable points, never free dominance. Show uncertainty and absolute measurements. |
| X03 Independent replication | Luna | Reproduce one decision-driving comparison from a clean checkout/environment. | Deviations are explicit; unexplained disagreement downgrades result to unresolved. |
| X04 Report/release review | Luna | Reproducibility report, evidence bundle, provenance/license audit and complete negative-result record. | Tests, links, builds, artifact hashes and claim scopes all pass. Publishing remains a separate explicit action. |

## Handoff contract for Sol or Luna

When work resumes, begin with Q10.b in `docs/implementation_queue.md`. Before editing: inspect `git status`, read `AGENTS.md`, `RESEARCH.md`, `EXPERIMENTS.md`, the measurement protocol, and the target specification. Implement one package only. Add tests that try to break the acceptance criteria, run focused tests then the repository-wide test/style/build/link checks, and update exact status evidence. Preserve historical runs. Do not train before the corpus and evaluation gates pass. End with the commit/status, files changed, exact commands/results, unresolved limitations, and the next row.
