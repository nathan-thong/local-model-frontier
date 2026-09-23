# Compression, distillation and inference research roadmap

Status: proposed work. None of the stages below establishes an improvement until its experiment passes the comparison requirements in `AGENTS.md`. The architectural hypotheses and approach families are recorded in [RESEARCH.md](../RESEARCH.md).

The available environment is CPU-only. Implement CPU reference paths and tiny correctness experiments first. GPU kernels, GPU memory and energy measurements remain unverified until measured on suitable hardware. A missing measurement is `null` with a reason, never zero.

## Rules shared by every stage

- Freeze the dataset split, tokenizer, scoring policy, workload, configuration and revision before a comparison. Use at least three paired seeds for claims; single-seed runs are implementation pilots only.
- Declare training tokens, optimizer updates, estimated training FLOPs, wall time, hardware, numerical precision and teacher costs. Match the named conventional control's disclosed training compute; equal tokens or steps alone do not establish this. If the estimator cannot account for an intervention, label the comparison non-comparable until it can.
- Distillation needs two views: student-training compute matched to ordinary SFT, and total experiment compute including teacher generation/forward passes, target storage and teacher preparation. State whether a reused teacher is treated as an existing asset; publish marginal and amortized totals with the reuse count. An unconditional compute-efficiency claim needs the total-cost control.
- Before each experiment, add an `EXPERIMENTS.md` entry with hypothesis, change, control, training budget, evaluation metrics, result (`pending` initially), interpretation and next experiment. Record a numerical capability-loss tolerance and minimum worthwhile resource change before inspecting results; retain failed runs.
- Report stored checkpoint bytes, unique parameter bytes, live process resident memory, peak memory, activation/workspace memory, KV or recurrent state, and any teacher/draft residency separately. Fake quantization provides no native-memory or speed evidence.
- Give each implementation task one independently reviewable change. Add no integration until its prerequisite reference behavior passes. A stage is complete when its artifact and checks exist; advancement to expensive experiments is a separate gate.

## C01 — Quantization specification and accounting

**Prerequisites:** reproducible float control; existing profiling and comparison schemas understood.

**Scope and files:** add `src/frontier/quantization/spec.py` and typed configuration in `config.py`. Specify supported bit widths, signed ranges, per-tensor/per-output-channel scaling, grouping, clipping, rounding, zero handling, eligible tensors and exclusions. Start with symmetric int8 and int4 linear weights; define ternary separately. Add a manifest schema containing logical shape, packing layout, scale dtype, padding and actual payload bytes.

**Checks:** reject ambiguous or unsupported configurations; verify byte accounting with non-divisible groups and tied parameters; preserve config serialization.

**Control and budget:** no model training. Float payload is the storage reference. Estimated bit counts must not populate measured-memory fields.

**Advance/kill:** advance when every byte has an owner and invalid settings fail before training. Stop expanding formats if the specification cannot be implemented and tested unambiguously.

## C02 — Fake-quantized linear reference

**Prerequisites:** C01.

**Scope and files:** add `quantization/fake_quant.py` and `quantization/linear.py`; quantize/dequantize forward weights while retaining explicit floating master weights for gradients. Use one documented straight-through gradient estimator. Leave embeddings, normalization and output projection eligibility explicit; preserve tied weights. Do not add activation quantization yet.

**Checks:** hand-calculated quantizer outputs, all-zero weights, clipping boundaries, finite gradients, excluded tensor preservation, no duplicate tied master parameters, serialization/resume equality and a quantization-disabled equivalence test.

**Control and budget:** float linear layers with identical initialization and inputs. CPU arithmetic checks only, then a fixed tiny overfit pilot.

**Advance/kill:** advance on finite stable learning and correct resume. Treat a float-master implementation as optimization research; never label it a compressed runtime.

## C03 — Low-bit training falsification matrix

**Prerequisites:** C02 and an established small float language-model control.

**Scope and files:** add a small frozen config set under `configs/quantization/`: float, int8 weight simulation, int4 weight simulation, then ternary if the first two behave correctly. Change only the weight quantizer. Record clipping/saturation rates and scale summaries without logging full tensors.

**Checks:** pilot each config for loss/gradient stability; confirm the ledger includes quantizer operations and master-weight memory.

**Control and budget:** equal disclosed training compute, same data order and tokenizer, three paired seeds after pilots. Record actual tokens consumed under the compute cap; also publish equal-token diagnostics without presenting them as equivalent-compute evidence.

**Advance/kill:** advance only formats within the preregistered quality tolerance. If int8 already fails, inspect implementation or recipe before spending on ternary. A poor result remains a completed experiment.

## C04 — Quantization schedule ablation

**Prerequisites:** one C03 format survives.

**Scope and files:** support exactly two alternatives in `quantization/schedule.py`: quantization from step zero and a specified float warmup followed by quantization. Checkpoint schedule and observer state. Keep clipping/scaling rules fixed.

**Checks:** resumed training crosses the transition once; optimizer state and master parameters survive the transition; deterministic transition boundary.

**Control and budget:** same quantized format without warmup plus the float control, with total training compute including warmup matched. One short pilot, then three paired seeds only if stable.

**Advance/kill:** retain the simpler schedule unless the extra mechanism yields a repeatable quality advantage or stability improvement within the declared compute cap.

## C05 — Packed checkpoint format and round trip

**Prerequisites:** C01 plus a C03 checkpoint suitable for export.

**Scope and files:** add `quantization/packing.py` and `quantization/export.py`. Implement one format first, with explicit version, endianness, group layout, padding, scales and tensor names. Strip optimizer/master weights from inference exports; training checkpoints remain resumable separately. Inspect input metadata before allocating and reject inconsistent payload sizes.

**Checks:** exhaustive pack/unpack of small value sets, odd tensor sizes, signed extremes, corrupted/truncated payloads, tied storage, exact quantized integer round trip and numerical equivalence to fake-quantized inference.

**Control and budget:** no training; compare bytes and cold-load behavior with a float inference export of the same model. Full dequantization on load is allowed only as an explicitly labeled reference path.

**Advance/kill:** advance when actual bytes match the manifest and outputs match the reference. File-size reduction alone is not a resident-memory result.

## C06 — CPU packed-weight execution reference

**Prerequisites:** C05; evidence that weight bytes matter for the selected workload.

**Scope and files:** add `quantization/runtime.py` with a bounded-tile unpack-and-multiply CPU path. Keep weights packed between calls; expose maximum workspace and fallback behavior. Start with inference-only linear layers and one format. This reference is not advertised as a native integer kernel.

**Checks:** compare outputs against full dequantization across batch/sequence shapes; verify no persistent full float copy; test workspace bounds and deterministic fallback errors. Measure a fresh subprocess to avoid allocator-history confounds.

**Control and budget:** identical quantized values executed through full dequantization and ordinary float inference; no training. Include conversion, cold load, prefill, decode and process memory.

**Advance/kill:** proceed to kernel work only if measured resident-memory savings survive scales, workspaces and runtime overhead. Retain a slow correct reference; do not call it an acceleration.

## C07 — Native kernel feasibility gate

**Prerequisites:** C06 and a pinned target CPU ISA or available GPU; a profile showing where time is spent.

**Scope and files:** write a backend adapter in `quantization/backends/`; implement or integrate one licensed packed linear kernel for one format and device. Keep the CPU reference as oracle. Document build requirements, supported shapes, accumulation precision and fallback conditions. Do not implement several devices at once.

**Checks:** adversarial numeric inputs, tail dimensions, concurrent invocation, supported/unsupported device paths and output error bounds. Benchmark warm and cold execution across the actual model's shapes.

**Control and budget:** identical weights and workloads against C06 and the float runtime. Include packed-weight conversion and any retained caches. GPU results remain pending on CPU-only machines.

**Advance/kill:** advance only on an end-to-end memory/latency tradeoff that clears its declared threshold. Stop kernel investment when dispatch/unpacking dominates or model-level gains disappear despite microbenchmark wins.

## D01 — Teacher provenance and cost ledger

**Prerequisites:** reliable dataset manifests and budget/result schemas.

**Scope and files:** add `distillation/teacher.py` and `distillation/artifacts.py`. Begin with a locally trained conventional teacher using the same tokenizer. Record teacher checkpoint hash, configuration, license/provenance, precision, evaluation mode, dataset IDs, target-generation time/compute and cache bytes. Keep teacher parameters outside the student optimizer.

**Checks:** teacher frozen and deterministic in evaluation mode; stale/mismatched target caches rejected; split leakage checks; resumed generation neither duplicates nor drops examples.

**Control and budget:** no student comparison yet. Benchmark teacher target creation on a bounded dataset slice; do not silently download or run an unbounded teacher job.

**Advance/kill:** proceed when targets are auditable and affordable. If the teacher lacks measurable advantage on the chosen evaluation, switch teacher or stop that distillation experiment before a sweep.

## D02 — Ordinary SFT control and batch contract

**Prerequisites:** D01; labeled or completion data with an explicit loss mask.

**Scope and files:** add `distillation/batch.py` and a minimal training objective interface in `training/trainer.py`. Ordinary supervised fine-tuning means teacher-free cross-entropy on the same input/completion examples used by candidate objectives. Store prompt/completion boundaries, padding and truncation policy. A continued-pretraining control must be named separately if examples lack supervision structure.

**Checks:** prompt and padding tokens are excluded when specified; token weighting and gradient accumulation match a manually calculated loss; empty-target batches fail; existing pretraining path remains equivalent.

**Control and budget:** same student initialization, example order and training compute as later objectives. Establish the teacher-free reference across three seeds before claiming distillation gains.

**Advance/kill:** advance when masking and effective target counts are verified; stop if the task set cannot distinguish teacher and student behavior reliably.

## D03 — Full-logit distillation reference

**Prerequisites:** D01–D02 and matching teacher/student token vocabularies.

**Scope and files:** implement `distillation/losses.py` with stable temperature-scaled KL plus supervised cross-entropy and an explicit mixing coefficient. Start with full logits on tiny CPU examples. Accumulate the loss in float32; store temperature, reduction, masking and coefficients in run metadata. Do not introduce truncated targets yet.

**Checks:** KL zero for identical distributions, correct direction and temperature scaling, finite extreme-logit behavior, padded-token exclusion, teacher gradient absence, coefficient-zero equivalence to SFT and online/offline target agreement.

**Control and budget:** D02 at matched student compute, plus total-cost accounting from D01. Predeclare one temperature and coefficient before a short pilot; charge tuning to the experiment ledger.

**Advance/kill:** advance on stable training and an eligible three-seed advantage within the declared cost view. If only teacher-free extra training buys the same quality for less total compute, record that finding.

## D04 — Target caching without hidden approximations

**Prerequisites:** D03 correctness and a profile showing teacher work or target I/O limits progress.

**Scope and files:** add streamed full-logit cache shards to `distillation/artifacts.py`; retain precision, shape and checksums. Add top-k compression only as a separate experiment with an explicit treatment of remaining probability mass; never silently relabel renormalized top-k KL as full-distribution KL.

**Checks:** deterministic cache joins by example/token position, interrupted shard recovery, corrupt-cache rejection and documented approximation error against D03.

**Control and budget:** online/full-logit reference and compressed target variant at equal student compute; include target production, disk, I/O and teacher reuse assumptions.

**Advance/kill:** cache only if it removes measured bottlenecks. Reject compression if quality loss exceeds the preregistered tolerance or probability semantics are ambiguous.

## D05 — Representation distillation

**Prerequisites:** D03 and a stable objective interface.

**Scope and files:** add `distillation/representations.py` with explicit layer mapping, token mask and normalization. Train small projection adapters where widths differ; record whether they are discarded or retained at inference. Implement one matching loss first. Do not force matching across architectures without specifying which tensors have comparable meaning.

**Checks:** teacher states detached, mask respected, projection gradients present, adapters counted in training compute/memory and checkpoint resume exactness.

**Control and budget:** SFT, logit-only, representation-only and logit-plus-representation with the same student/data and equal disclosed training compute. Start with a pilot; only promote promising objectives to paired seeds.

**Advance/kill:** retain representation matching only if its incremental cost buys reproducible improvement over logit-only distillation. Drop fragile layer mappings rather than expanding an unbounded adapter search.

## D06 — Task and trajectory distillation

**Prerequisites:** D02 and task-specific validators with held-out evaluation examples.

**Scope and files:** add `distillation/trajectories.py` for teacher-generated task answers and, where externally checkable, action/state traces. Store prompt, output, validator outcome, filtering rules, teacher revision and generation budget. Treat written rationales as generated training targets, not verified reasoning or ground truth. Use outcome validators for arithmetic/code or other bounded tasks where available.

**Checks:** training/evaluation partition separation, validator fixtures, rejected-target accounting, prompt provenance, target-mask boundaries and duplicate detection.

**Control and budget:** ordinary SFT on original examples, answer-only distillation, then trajectory targets with equal total student compute and disclosed generation/validation costs. Match task mix; report additional target tokens.

**Advance/kill:** continue only if held-out task scores improve under eligible controls. Stop a trajectory method when validators cannot establish useful target correctness or gains depend on benchmark contamination.

## D07 — Quantization-aware distillation interaction

**Prerequisites:** one quantizer survives C03 and one distillation objective survives D03–D06 independently.

**Scope and files:** add exactly four configs: float/SFT, float/distillation, quantized/SFT, quantized/distillation. Reuse the same objective and quantizer modules. State whether master weights are retained during training and export all surviving quantized students using C05.

**Checks:** zero-distillation coefficient reproduces quantized SFT; disabled quantizer reproduces float distillation; resume includes quantizer, objective and teacher-cache identity.

**Control and budget:** a preregistered 2×2 factorial at equivalent disclosed training compute, three paired seeds, same teacher/data. Publish teacher-inclusive totals and the interaction effect rather than comparing only the best corners.

**Advance/kill:** native deployment proceeds only when quality and actual runtime-memory gates both pass. If fake-quant improvement disappears after export, treat export/runtime correctness as unresolved.

## I01 — Deterministic inference budget baselines

**Prerequisites:** trustworthy inference profiler and cached/full-forward equivalence.

**Scope and files:** extend `inference/generate.py` and `profiling/benchmark.py` with fixed prompt/output lengths, greedy and specified sampling modes, EOS policy, batch size, repeated measurements and per-request latency records. Add short/long context and batch-one/small-batch workloads.

**Checks:** generated token counts, seed handling, cache reset, warmup exclusion and synchronized device timing. Measure first-token latency and decode time separately.

**Control and budget:** conventional greedy/sampling inference on the same target checkpoint and prompts; no retraining. Include all resident model/state memory and publish latency dispersion.

**Advance/kill:** adaptive or speculative claims wait until baseline variance is smaller than the declared worthwhile latency change.

## I02 — Greedy speculative decoding correctness

**Prerequisites:** I01; a compatible small draft with the same tokenizer and special-token semantics.

**Scope and files:** implement `inference/speculative.py` for greedy draft-and-verify decoding with a fixed draft length. Track acceptance, rejected suffixes, target verification calls, draft work and cache rollback. Begin on tiny CPU models.

**Checks:** exact target-greedy token equality over adversarial acceptance/rejection patterns, EOS boundaries, context limits, chunk boundaries and cache rollback; include full rejection and full acceptance.

**Control and budget:** ordinary target greedy decoding on identical prompts. Include draft weights, draft cache, target cache, verification workspace and draft-training cost where a learned draft is introduced.

**Advance/kill:** benchmark only after exact output equivalence passes. Reject the deployment setting if combined resident memory violates the user's budget or draft overhead removes end-to-end latency savings.

## I03 — Sampling-preserving speculation

**Prerequisites:** I02; an independently specified rejection/correction algorithm.

**Scope and files:** extend `inference/speculative.py` with explicit proposal/target probabilities and correction sampling. Define the order of temperature/top-k/top-p transforms and guarantee compatible support. Keep a slow categorical reference.

**Checks:** exact analytical transition probabilities on tiny vocabularies plus seeded statistical distribution checks with predeclared tolerances; zero-probability cases, all accepted/rejected paths, EOS and numerical stability. Sequence equality under the same random seed is not the sampling correctness criterion.

**Control and budget:** ordinary target sampling with identical distribution transforms; measure draft+target time and memory, acceptance by workload and tail latency.

**Advance/kill:** preserve distribution correctness before optimizing. Stop if gains require changing the target distribution or dropping overhead from the ledger.

## I04 — Adaptive computation feasibility experiment

**Prerequisites:** I01; a fixed-depth student control and explicit cache semantics.

**Scope and files:** start with request-level selection between two independently profiled fixed-depth models in `inference/adaptive.py`. Evaluate a cheap predeclared routing signal. Specify whether both models stay resident or loading time is charged. Defer token-wise depth skipping until a design defines absent layer states, positions and cache updates without breaking causality.

**Checks:** deterministic routing, budget enforcement, no held-out labels at inference, complete time/memory accounting and fallback behavior. Evaluate routing separately from model quality.

**Control and budget:** always-small, always-large and a fixed/random routing policy at matched expected inference compute; disclose training compute of both models and router. Report capability by difficulty slice, worst-case memory and latency percentiles.

**Advance/kill:** advance only if the router beats simple fixed allocation at the stated budget. Stop if routing overhead or errors erase gains; no need to build dynamic-depth kernels first.

## I05 — Learned adaptive depth, only after the simple gate

**Prerequisites:** I04 supports useful conditional allocation; a written causal state/cache design and matched dense control exist.

**Scope and files:** add one request-level early-exit mechanism with one auxiliary exit head before token-level routing. Add `models/exits.py`, explicit exit policy and exit-head/teacher training costs. Sweep at most three preregistered thresholds using validation data, then freeze the threshold for held-out evaluation.

**Checks:** forced-final-exit equivalence, forced-early-exit consistency, cached/full sequence agreement under the defined policy, finite auxiliary gradients, threshold monotonicity of assigned budgets where the policy guarantees it and worst-case memory.

**Control and budget:** a shallower dense model trained for equal compute, the original full-depth model, and a fixed exit at matched average inference compute; count auxiliary head training and retained runtime parameters.

**Advance/kill:** retain adaptation only if measured task-quality/latency tradeoffs beat a static choice within declared tolerances. A FLOPs reduction with slower wall time is a negative latency result.

## X01 — Combined frontier replication

**Prerequisites:** independently surviving architecture, compression and inference interventions; no more than two new interactions per experiment.

**Scope and files:** add frozen experiment manifests under `configs/frontier/`, use existing `experiments/compare.py` and result serialization, and extend [technical_report.md](technical_report.md). Compare at fixed resident-memory and disclosed compute budgets; plot individual runs and nondominated points without hiding uncertainty. Keep capability metrics separate or preregister a justified aggregate.

**Checks:** missing resources disqualify a claimed frontier point; uncertainty/seed summaries include failures; budget-ineligible runs cannot be labeled improvements; fresh-process replication on a second machine when available.

**Control and budget:** conventional models spanning the same memory region, trained at equivalent disclosed compute; include teacher/draft/router costs and all retained models/state. Separate measured and estimated energy; use `null` when energy cannot be measured defensibly.

**Advance/kill:** publish the complete evidence and limitations, including dominance by conventional baselines. Scale training only after a repeatable frontier shift survives the held-out protocol and an independent reproduction.

## Smallest execution queue

1. Complete the stronger control and architecture work in the main roadmap.
2. Implement C01–C02 as one tightly reviewed reference change; run no large training sweep.
3. Run C03 pilots, promote at most one format, and implement C05 before any memory claim.
4. Implement D01–D03 with a tiny local teacher; compare against ordinary SFT before representation or trajectory additions.
5. Pursue C06–C07, D05–D07 or I02–I05 only when their individual prerequisite gates pass.

These stages are a decision tree, not a commitment to implement every idea. Stopping a branch after a clear negative result is successful research progress.
