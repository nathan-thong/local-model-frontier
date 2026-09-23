# Research assumptions and prior approaches

This project asks whether sequence architecture, weight precision, distillation and inference-time adaptive computation can jointly improve small language-model capability at fixed resident memory or compute. These are hypotheses to test, not expected outcomes. Milestone 1 establishes a conventional baseline and measurement discipline before changing the architecture.

## Assumptions under test

1. **Sequence mixing has a workload-dependent cost.** Full causal attention has useful content-based retrieval but its attention work grows quadratically with prompt length and its KV cache grows with sequence length. GQA reduces stored key/value heads. Sliding windows can cap attention span only when the implementation uses a genuinely sparse or windowed kernel; a dense mask alone does not prove lower compute or latency.
2. **Recurrence and linear attention may trade retrieval fidelity for bounded state.** Selective state-space and gated recurrent approaches can make state size independent of context length, while hybrids may preserve precise local retrieval through occasional attention. Training parallelism, kernel availability, stability, and sequential decode latency can dominate theoretical complexity.
3. **Ternary or ultra-low-bit weights may trade arithmetic complexity and storage.** Three-valued weights need about 1.585 bits of ideal information per weight, but real checkpoints also store scales, metadata, embeddings, activations, KV state and runtime buffers. A simulated quantizer is evidence about optimization behavior only; packed storage and native kernels are required to establish actual memory or speed gains.
4. **Distillation may recover capability after compression.** Logit targets, hidden-state matching, and task or trajectory targets expose different information and incur different teacher costs. Distillation gains must be compared against the same student trained on the same inputs and compute, with teacher generation/forward costs reported separately and in total.
5. **Adaptive computation may direct effort to difficult positions or requests.** Routing can reduce average FLOPs, but selection overhead, poor batch regularity and routing errors can increase wall-clock time or reduce quality. Report compute distributions and tail latency, not just average FLOPs.
6. **Inference-time acceleration and model capability are separate axes.** Speculative decoding can preserve a target model's output distribution while reducing serial target evaluations, but requires a draft and verifier. It does not by itself make the target model more capable; include the draft in resident-memory accounting.

## Likely bottlenecks and confounds

- Small models can be dominated by embeddings and the output projection; context length, vocabulary and weight tying materially affect parameter and compute budgets.
- At short contexts, a theoretically cheaper sequence block may be slower because it lacks fused kernels or increases launch overhead. At long contexts, memory bandwidth and KV-cache traffic may dominate.
- Different tokenizers change both sequence length and perplexity scale. Compare perplexity across models only with the same tokenizer and scoring policy; use bits per byte or another byte-normalized metric when tokenizers differ.
- Equal parameters, examples, tokens, steps, wall time and FLOPs are different controls. State which budget is matched for each claim. An equal-training-compute control is required for any claim that a method improves the capability-per-resource frontier.
- Dataset overlap, document-boundary handling, sequence packing, train/validation leakage, optimizer tuning and seed selection can each create apparent gains. Freeze the recipe for comparisons and report failed runs.
- Simulated quantization does not demonstrate deployable compression. Estimate versus measure FLOPs explicitly. Profile cold load, steady-state inference, KV/recurrent state and process/runtime memory separately.
- Tiny corpus or short-run results are useful for falsifying code and stability hypotheses, but do not support general capability conclusions or scaling claims.

## Prior approach families

- **Hybrid attention and recurrence:** Griffin combines gated linear recurrences with local attention; Mamba uses input-selective state-space dynamics and hardware-aware parallel scans. These motivate testing pure and hybrid sequence mixers after the baseline. [Griffin (De et al., 2024)](https://arxiv.org/abs/2402.19427), [Mamba (Gu & Dao, 2023/2024)](https://arxiv.org/abs/2312.00752).
- **Linear attention:** recurrent reformulations of attention offer constant-state decoding under particular kernel/feature-map assumptions. Their approximation and retrieval behavior need task-level checks. [Transformers are RNNs (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236).
- **Grouped-query attention:** shares key/value heads across query groups to reduce KV storage and bandwidth while retaining multiple query heads. [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245).
- **Native ternary weights:** BitNet b1.58 trains with weights in {-1, 0, +1}; its result motivates a later native-weight study, but packed representation and supported compute kernels must be measured directly. [The Era of 1-bit LLMs (Ma et al., 2024)](https://arxiv.org/abs/2402.17764).
- **Quantization-aware distillation:** low-bit-aware training can use teacher behavior to compensate for quantization error. Teacher logits and calibration/generation data carry compute and data costs that belong in the ledger. [LLM-QAT (Liu et al., 2023)](https://arxiv.org/abs/2305.17888).
- **Speculative decoding:** a small draft proposes multiple tokens for parallel verification by a target model; the original method preserves the target distribution under its sampling procedure. Measure acceptance, latency and combined resident memory. [Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192).
- **Adaptive depth/compute:** Mixture-of-Depths uses capacity-constrained token routing to allocate computation unevenly. Evaluate actual wall time and output quality against a compute-matched dense control. [Raposo et al. (2024)](https://arxiv.org/abs/2404.02258).

## Falsification order

1. Make the conventional baseline train, resume, evaluate, and profile correctly.
2. Replace one block family at a time and check numerical behavior before hybrid schedules.
3. Measure simulated quantization effects before building packing or native kernels.
4. Add distillation controls only after teacher and student compute are recordable.
5. Evaluate adaptive and speculative inference with task quality, latency and full memory cost.
6. Combine only interventions with independent evidence, then test interactions against a compute-matched baseline.
