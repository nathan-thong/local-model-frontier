# Research execution roadmap

Status: planning only, 23 September 2026. This document specifies future work; no result below is an observed improvement. Milestone 1 remains a record of completed infrastructure experiments. The next objective is one defensible comparison of sequence modules under a declared resource budget.

## Read and execute in this order

Start with the ordered [implementation queue](docs/implementation_queue.md) for the current blockers and handoff-sized packages. The [next execution plan](docs/next-execution-plan.md) records the Q09 portable-evidence and bounded fixture-reproduction closeout, then sequences later milestones into owner-sized handoffs with explicit gates. Read `AGENTS.md`, `RESEARCH.md`, `EXPERIMENTS.md`, and the measurement and reproducibility protocols before implementation. Use these work packages for the detailed designs and acceptance tests:

1. [Evidence and measurement](docs/roadmap-measurement.md): experiment eligibility, artifact access, data, evaluation, compute, profiling, and statistics.
2. [Sequence modules](docs/roadmap-sequence.md): conventional GQA, bounded local attention, one recurrent or linear reference, and hybrids.
3. [Compression and distillation](docs/roadmap-compression.md): simulated quantization, native representation, distillation, and conditional inference work.

Each numbered work package is a bounded implementation task, usually one reviewable change. These are submilestones within the original research milestones, not a reason to implement all features at once. Finish each acceptance gate before depending on it.

## What the existing repository establishes

- A conventional model trains, resumes, evaluates, and profiles on CPU. Tests and the first public CI run passed on Python 3.10 and 3.12.
- The completed language-model runs use a bounded synthetic TinyStories corpus, a byte tokenizer, and a context of 256 byte tokens. They establish infrastructure behavior and narrow corpus metrics.
- Baseline A uses four query heads and four KV heads: this configuration is MHA, even though the attention implementation supports GQA.
- Per-layer module selection and opaque per-layer decode state already exist. Only the conventional attention implementation is currently registered.
- The compute estimator explicitly describes dense Transformers. It cannot silently become the cost model for a different sequence module.
- No CUDA or energy measurement was available in the completed experiments. GPU performance, low-bit speedups, and broad capability remain untested.
- Training records now isolate CUDA peak counters to optimizer steps and distinguish CPU process snapshots from true phase peaks; no CUDA hardware was available to validate actual readings.

## Source audit findings and disposition

These were findings from reading earlier code and records, not new benchmark results. The first four are addressed in comparison schema 3: fixture status is distinct from known synthetic corpus origin and claim scope; the three-paired-seed floor cannot be weakened; missing tokenizer artifacts block metric comparability; invalid/missing compute and metrics yield explicit ineligibility. Add regression tests if a later change reopens any of these paths. E11's portable M2-GQA-002 bundle, same-environment CPU fixture reproduction, and separate clean-clone fixture rerun are verified; none independently reruns the M2-GQA-002 training comparison. Every retroactive provenance correction must remain labeled as an audit rather than original capture. Never rewrite historical run files.

Implementation has since completed E01-E03, E06 and E07. E05 now has document-level and byte-aligned evaluation accounting; E08 and E09 remain partial, with exact gates tracked in `docs/roadmap-measurement.md`. Q08 added isolated CPU inference-phase peaks, nested active/allocated sequence-state accounting, cumulative sweep runtime caps across resumes, and a sampled CPU optimizer-step record. Its one-step probe had only two samples over a gap as long as the step, so brief transient peaks remain unresolved; CUDA validation is unavailable on this host. Q09's portable evidence bundle verifies from a detached clean checkout; its deterministic CPU fixture reproduces in the documented environment and in a separate clean clone on this host. This does not independently rerun the M2-GQA-002 comparison. Q10.a's OpenStax choice was withdrawn after the publisher page's LLM-training restriction was found, before acquisition. The replacement decision is a bounded, versioned English Wikipedia article snapshot with page-level attribution and license checks; no source text has been downloaded. Q10.b now has a deterministic schema-4 train/validation/test splitter with hashed normalized document identities and duplicate reporting, plus a frozen 24-title candidate list and metadata-only screening rules. Exact article revision metadata, page-specific license/attribution audit, source acquisition, and repeat preparation remain open.

Milestone 2 S01 now has GQA reference, gradient, cache-size and matched-compute config tests. A separate schema-3 TinyStories split records explicit synthetic origin and verified source/train/validation hashes; the historical schema-1 data remains unchanged. The MHA/GQA smoke and screen passed their plumbing gates, including exact 50% KV-cache reduction in the profiled smoke. The seed-17 screen NLL delta was -0.103 at -0.532% estimated compute; it is not claim-eligible. The first fresh three-seed full-budget attempt was ineligible because the schedules used different warmup fractions. M2-GQA-002 corrected that prospectively and passed the comparator on clean revision `73bde6e`: mean paired GQA-minus-MHA NLL delta was -0.02307 nats/token (SD 0.02439), and cache bytes were halved. CPU cached decoding was about 8.5% slower; this is a synthetic-domain quality/cache tradeoff, not a broad or statistically conclusive capability claim. See [the experiment record](EXPERIMENTS.md#m2-gqa-002---corrected-grouped-kv-confirmation-preregistered) and [the detailed execution plan](docs/post-confirmation-research-plan.md).

Append an erratum when a reporting bug changes eligibility. Retain original measurements and records. A reporting correction does not invalidate every measurement or establish a new scientific result.

## First six implementation tasks

| Order | Task | Deliverable | Gate before moving on |
|---|---|---|---|
| 1 | Harden comparison eligibility | Required-field validation, finite metric checks, three-seed claim floor, structured blockers | Missing/null/NaN/duplicate/incomplete inputs fail closed; valid existing comparisons remain interpretable |
| 2 | Separate evidence scope and export evidence | Data-origin metadata, explicit claim scope, append-only errata, portable artifact manifest/export | A fresh checkout can identify exactly which public results are available and why a comparison is eligible |
| 3 | Add a bounded experiment runner | Predeclared seeds, config hashes, budget estimates, unique run IDs, resume policy, failure ledger | Folder names cannot substitute for configured seeds; no overwrite, duplicate jobs, or silent retry selection |
| 4 | Add a compact subword tokenizer | Serialized tokenizer artifact, digest, special-token contract, configurable vocabulary | Save/load and Unicode round trips pass; fit uses training split only; no hard-coded 259-token model requirement |
| 5 | Freeze a modest corpus and evaluation recipe | Verified source/license/revision, document splits, deterministic sample, held-out test, versioned task adapter | Exact overlap checks pass; evaluation works on known-answer fixtures and a conventional pilot |
| 6 | Establish the new conventional control | Frozen tokenizer/context/model/optimizer, three matched seeds, complete profiles | Every run comes from one clean revision and reports actual consumed budget; task scores and corpus loss are separately labeled |

Tasks 4–6 are a new baseline version. Do not compare its token perplexity directly with the old byte-token baseline. Keep old M1 artifacts. A tokenizer comparison needs its own compute-matched experiment and a byte-normalized metric; it is not part of a sequence-module ablation.

Implementation of module accounting and correctness tests can proceed before the new corpus control finishes. Claims using the new evaluation recipe must wait for that control.

## Budget ladder

Use explicit limits before launching any run. The following are planning defaults, not measured runtime promises:

| Level | Purpose | Default limit | Promotion rule |
|---|---|---|---|
| Unit | Mathematical and serialization invariants | Tiny tensors, deterministic CPU tests | Relevant invariants pass |
| Smoke | End-to-end wiring and finite gradients | One seed, at most 32 optimizer updates | Train/evaluate/resume/profile succeeds; no scientific claim |
| Pilot | Detect obvious instability or useless throughput | One paired seed; at most 5% of the predeclared full budget | Finite losses, valid accounting, plausible quality signal, useful workload coverage |
| Confirmation | Primary comparison | Three paired seeds, starting at the M1-A estimated budget scale when the estimator is applicable | All planned seeds finish or failures are reported; comparable source/data/evaluation/budgets |
| Extension | Context, memory, robustness, extra seeds | A separately approved experiment plan after the confirmation report | An unresolved scientific question justifies additional spend |

Do not transplant the M1 FLOP number to a new estimator as though the definitions were identical. Calculate a new budget ledger when tokenization, model shape, or estimator changes. Report optimizer updates, input and scored tokens, total estimated work, measured wall time, and hardware. Equivalent training FLOPs are required for an efficiency/capability claim; equal tokens are a useful additional diagnostic, not a replacement.

Prefer serial CPU experiments on the current machine. Parallel CPU training can contaminate timing measurements. Before a GPU or paid-compute run, measure a short pilot on the intended hardware and write the projected total runtime, peak memory, and cost into the experiment plan. No automatic cloud provisioning, large teacher download, or unbounded sweep is implied by this roadmap.

## Smallest useful Milestone 2 campaign

Keep the initial architecture campaign narrow. Use the same tokenizer, training corpus, evaluation context, FFN, normalization, optimizer family, and initialization policy across each paired experiment.

| Experiment | Hypothesis | Single principal change | Control | Primary failure condition |
|---|---|---|---|---|
| M2-GQA | Fewer KV heads reduce decode state with an acceptable quality cost | Four query heads, two KV heads | New conventional MHA control with four KV heads | Cache accounting is wrong, or the measured quality/resource tradeoff is dominated |
| M2-LOCAL | A bounded attention window reduces retained decode state | Local window; start with 128 at context 256 as a diagnostic | Full attention with the same GQA setting | Cached and reference execution disagree, state is unbounded, or measured benefits do not justify quality loss |
| M2-LINEAR | A simple recurrent linear-attention reference can trade retrieval quality for fixed state | One documented sequence-mixer family | Conventional attention under the same declared training compute | State grows with context, gradients are unstable, or throughput/quality fails the declared gate |
| M2-HYBRID | Occasional full attention recovers useful retrieval performance | One fixed schedule, initially three cheap blocks plus one attention block when using four layers | Pure attention and the corresponding pure cheap-block model | Hybrid adds cost without a useful measured tradeoff |

Before each run, record all eight required experiment fields in `EXPERIMENTS.md`; mark result and interpretation as pending. Choose a task-specific non-inferiority margin before seeing candidate results. Report effects and seed variation even when the margin is not met. Three seeds are a minimum for confirmation, not proof of statistical significance. An inconclusive result may require more seeds; it does not justify calling a win.

Use existing short contexts for correctness. Long-context performance is a separate campaign with new controls trained and evaluated under the same context policy. Do not extend a checkpoint beyond its configured context and describe the resulting score as equivalent evaluation.

A dense local mask is a valid reference implementation. It does not establish sparse prefill work. A Python recurrence is a valid numerical reference. It does not establish an efficient deployment kernel. Preserve both reference and optimized implementations and compare them numerically.

## What counts as a frontier result

Store the measured point and its conditions, not one opaque efficiency score. Each row should include:

- Corpus NLL, task-specific scores, tokenizer/context identity, and uncertainty.
- Unique parameter count, model tensor bytes, serialized weights, and total inference resident memory.
- Attention/recurrent state bytes, runtime/workspace memory, batch and context capacity.
- Prefill and decode latency distributions, throughput, cold load, and precision/backend.
- Training and teacher compute, data exposure, tuning budget, and total wall time.
- Energy per token only with a stated sensor, integration interval, attribution boundary, and measurement uncertainty; otherwise null with a reason.

Pareto dominance applies only to compatible workloads and declared metric directions. Do not average unrelated task scales or hardware into a score without a preregistered rule. A candidate with uncertain differences is not confidently dominant. A matched resident-memory comparison must give the conventional control a fair chance to use the same memory budget; equal parameter count alone does not satisfy that comparison.

Measured latency comparisons require the same hardware, backend, precision, thread configuration, warmup, prompt/decode workload, and measurement method. Quality comparisons and latency comparisons should have separate eligibility checks. Otherwise the harness can reject useful quality evidence merely because it was evaluated on different hardware, or accept invalid speed evidence because corpus hashes match.

## Compression and inference work: conditional order

After the sequence campaign is reportable, proceed through simulated quantization, quantized training stability, packed serialization, reference packed inference, and a native kernel only where the measured bottleneck justifies it. Keep floating master weights, scales, optimizer state, unpack buffers, activations, and cache in the memory ledger. Simulated ternary weights do not count as ternary resident storage.

For distillation, establish ordinary supervised training first, then logit, representation, and task/trajectory targets as separate ablations. Compare student-only cost and total teacher-plus-student cost. Match the data available to the control, including teacher-generated examples where relevant. If a teacher introduces extra data or knowledge, disclose it as part of the intervention.

Only after a stable model and inference profiler exist should adaptive depth or speculation enter the campaign. Adaptive computation needs routing overhead, quality, tail latency, and batch behavior. Speculation needs draft-plus-target resident memory and either exact distribution preservation tests or an explicit approximation label. Keep throughput acceleration distinct from increased task capability.

## Instructions for the implementing agent

1. Inspect the repository state and read the selected work package. Do not assume this planning snapshot remains current.
2. State the exact package being implemented and its acceptance gate. Implement one coherent change; do not scaffold downstream features speculatively.
3. Add tests for the relevant numerical, provenance, or measurement failure mode. Avoid tests that only restate the implementation.
4. Run the required tests and style checks from `AGENTS.md`. Documentation-only changes need link/command validation rather than model training.
5. Before a scientific run, commit the implementation/config/protocol so all control and candidate runs record the same clean revision. Record the experiment plan first. Config files outside the checkout must still be copied and hashed into the run record.
6. Execute the smallest budget level capable of resolving the next question. A pilot is exploratory; confirmation must use the frozen recipe and all planned seeds.
7. Append outcomes and failures to the experiment log. Preserve raw outputs, artifact hashes, and exact reproduction commands. Do not edit old result files to make comparison checks pass.
8. Report files changed, tests passed, observed results, limitations, and the next package. Publishing and compute provisioning follow the user's current instructions; the roadmap itself is not authorization for new external services.

Suggested handoff prompt:

> Read AGENTS.md, ROADMAP.md, and the linked work package. Implement the first unfinished evidence/measurement package only. Resolve ordinary implementation decisions independently. Add tests for the documented failure cases, run the required checks, and update the work-package status with evidence. Preserve all historical experiment records. Do not train new architecture candidates until the reporting gates and their compute accounting are valid. Finish with a precise handoff to the next package.

## Stop rules

- Stop a run on non-finite loss/gradients, corrupt state, invalid data manifest, exceeded budget, or numerical disagreement beyond a declared tolerance. Save the failure and diagnostic context.
- Stop an improvement claim on missing provenance, unequal undeclared compute, a mismatched tokenizer/evaluation protocol, selected-away failed seeds, or unmeasured resource assertions.
- Stop optimizing a kernel when profiling shows its component is not material to the declared workload.
- Stop combining interventions until individual effects are understood. A combined result needs ablations and a conventional control with equivalent total training compute.
- Stop expanding infrastructure when the next experiment can already answer the current question. The output is trustworthy evidence, including useful negative results.
