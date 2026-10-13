# Paper-to-code map and unresolved choices

Source: the 32-page ACL Findings version of *FBS: Modeling Native Parallel
Reading inside a Transformer*, Sections 2–3 and Appendices B–E.

| Paper component | Current implementation | Scope |
|---|---|---|
| Prefix-predictive multi-horizon PAW, C.1 | Expected embedding over predicted top-K support; future tokens used only as loss labels | Native and Qwen3 models |
| Soft training / discrete inference span, C.2 | Soft sigmoid windows in training; floor span and hard horizon exclusion in evaluation; k=0 returns zero | Tested |
| BIOS chunking, D.1–D.2 | Online B/I closure, singleton/O handling, character-offset alignment, idiom-over-segmentation priority | Reference rules stated below |
| Sparse KV hard skip, E.1 | No Q/K/V, CH, attention or FFN call on skipped token-layer; no cache append; absolute positions preserved | Batch size 1; tested by forbidden-call mocks |
| Straight-through stochastic gate, E.2 | Bernoulli training, deterministic threshold inference | Dense surrogate computes branches; selected skipped keys are masked |
| Stage-1 objective, Section 3.1.3 | LM, preview, BIOS CE, cosine alignment and mean-skip regularizer | Exact auxiliary formulas/weights not recovered; defaults disclosed |
| Stage-1 continued pretraining | Actual Qwen3 weights, token budget, gradient accumulation, warmup/cosine optimizer | Large training unrun |
| Stage-2 PPO | Frozen reference; only SG/PAW trainable; clipped action ratios; compute-saving minus gold NLL degradation | Working local pipeline; original prompts unavailable |
| Parameter matching | Unique backbone/auxiliary/total counts and measured overhead | No unsupported <1% claim |
| Main evaluation | Local multiple-choice log-likelihood and continuation perplexity, raw prediction files, threshold sweep | Official benchmark preparation/grading not recovered |
| Timing, B.3 | Fixed lengths, warmups, repetitions, CUDA sync, median wall time, separate prefill/decode | CPU smoke verified; A100 speedups unverified |

## Architecture and data ambiguities

1. Section 3.1.1 names Qwen3-4B-Instruct while giving width 4096, 32 layers,
   32 heads and ~32K vocabulary. Public Qwen3-4B configurations have width 2560,
   36 layers and vocabulary 151936 ([official configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)).
   These are not interchangeable backbones. The original checkpoint identifier,
   immutable revision, tokenizer and trained weights are needed.
2. Independent full-vocabulary heads at every horizon/layer do not fit the
   paper's claimed few-million-parameter overhead. The native model retains that
   option for legacy compatibility. The Qwen adapter implements tied low-rank
   preview heads and bottleneck CH projections, explicitly a reconstruction.
   Rank 32 is a configurable choice, not a recovered paper hyperparameter.
3. Original 30B-token corpus sampling proportions, deduplication similarity
   definition, exact 12B-token training order, pkuseg model and idiom lexicon
   are not available. `fbs.data` performs exact normalized deduplication only;
   it does not claim to implement an unspecified similarity >0.7 filter.
4. Appendix D.2.4 uses q for both alignment coverage and classifier confidence,
   and does not specify a unique CTC target/blank/reduction protocol. This
   implementation uses explicit token-level CE, not an invented CTC objective.
5. The boundary classifier uses discrete BIOS decisions in the computation
   path; it receives supervised CE gradients where labels are present. There
   is no claim that argmax boundaries receive gradients from LM loss alone.
6. The paper's stage-1 threshold annealing and Bernoulli ST descriptions do not
   specify a single combined rule. This implementation follows E.2 Bernoulli ST;
   a configurable mean-skip penalty is the explicit conservative regularizer.

## Cache and calibration conventions

- CH reads only closed prefix chunks. A pending B/I span closes upon the next
  non-I label; the sequence endpoint does not flush it. A skipped token makes
  no CH or KV update. O and orphan I are singleton chunks. These choices are
  recorded because the paper does not determine every online boundary case.
- PAW uses hard spans during cached deployment and calibration; training uses
  soft spans. The floor span predictor has no gradient under hard calibration;
  other PAW parameters receive the gate-policy gradient. No end-to-end
  differentiability through floor is asserted.
- PPO records sampled gate actions and their probabilities along a greedy-token
  rollout, replays the same token/action trajectory for importance ratios, and
  uses a frozen full-compute reference for the gold-answer NLL. It applies the
  sampled leading gate schedule to gold teacher forcing. Cost counts include
  prompt and decode token-layers, and exclude auxiliary FLOPs. These are explicit
  choices for details missing from the published artifact, not claimed exact
  historical rollout semantics. The reward coefficients default to 0.1/0.1 and
  clip to 0.2 as stated in the paper; the value baseline is fixed at zero.
- Calibration rejects duplicate normalized development prompts and exact
  normalized overlap with the supplied held-out prompt set. The original 10k
  prompt list is needed to reproduce the reported training distribution.
- Native checkpoints record architecture, vocabulary when applicable, optimizer,
  RNG and token count. Continued-pretraining resume requires identical configuration
  and unchanged local data. It does not independently verify changed file content.

## What is still required for full numerical reproduction

- Original pretrained and FBS-S1/FBS-Full checkpoints, exact architecture and
  module dimensions, all objective weights and original dataset manifests.
- Complete original dev/evaluation prompt lists, benchmark templates and grader
  versions, including HumanEval-X/MBPP execution, BBH/GSM8K/math answer extraction,
  BIOS probes, idiom data and FactScore judging. The local likelihood evaluator
  does not claim those scores.
- Group A/B/C external acceleration baseline checkpoints and implementations,
  measured under the same harness, plus parameter counts including auxiliaries.
- A hardware-specific batched executor if reproducing batch 4/8 timings; the
  new cache executor intentionally supports only batch 1, the main timing case.
- Full-scale retraining and A100 measurements, five-shot task evaluation and
  paired uncertainty estimates. The code has been checked for mechanics only;
  no reported quality gain or ~30% latency gain has been validated here.

The reference reconstruction can be run now. These missing inputs prevent an
honest assertion that the published results have been completely reproduced.
