# Architecture and cost audit — 2026-10-14

**Conclusion:** the public paper contains a material architecture-description
conflict. The public evidence does **not** identify the historical experiment's
actual checkpoint, nor distinguish a transcription error from a different
experimental backbone. The low-rank implementation in PR #3 is a reconstruction,
not recovered original code. It meets a small *parameter* budget but its current
full-vocabulary PAW evaluation is computationally expensive.

## Sources and version check

- [ACL record](https://aclanthology.org/2026.findings-acl.200/) and
  [published PDF](https://aclanthology.org/2026.findings-acl.200.pdf), 32 pages,
  proceedings pp. 4106–4137. Inspected the actual PDF, including rendered pages
  4 and 19; this is not a conclusion based only on an HTML conversion.
- [arXiv v1](https://arxiv.org/abs/2601.21708v1), 2026-01-29, and
  [v2](https://arxiv.org/abs/2601.21708v2), 2026-04-08. Both downloaded PDFs retain
  the architecture conflict and parameter-budget statement.
- [Official Qwen3-4B config at the selected immutable revision](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json),
  copied without architectural changes to `configs/qwen3_4b_public.config.json`.
- [Official weight index](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/model.safetensors.index.json):
  398 tensor entries, BF16 total size 8,044,936,192 bytes. This independently
  agrees with 4,022,468,096 unique parameters; no weight shards were downloaded.
- [Original Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B) and
  [Qwen3-4B-Instruct-2507 config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json)
  (the latter inspected on the audit date).
- Repository before this change: merge commit
  `99e4c23cee810166c6f013ad22b7ed6a297fac16`, [PR #3](https://github.com/WtxwNs/FBS/pull/3).

| Evidence | ACL PDF page (printed page) | arXiv v1 / v2 PDF page |
|---|---|---|
| Backbone and weight-copy descriptions, §3.1.1 | 4 (4109) | 4 / 4 |
| Stage-1 token budget, §3.1.3 | 5 (4110) | 5 / 5 |
| Single A100 environment, B.1 | 16 (4121) | 13 / 14 |
| Compute proxy Eq. (4), B.3 | 17 (4122) | 15 / 16 |
| Parameter table and PAW size, B.7/C.1 | 19 (4124) | 17 / 18 |
| Default preview compressor, C.5 | 20 (4125) | 18 / 19 |

## Backbone: what is established

Section 3.1.1 first calls the model *comparable to* Qwen3-4B-Instruct; this
qualifier should not be silently omitted. However, its initialization paragraph
claims direct copying of public baseline attention/FFN weights, and its system
description identifies the original Qwen baseline. Those statements do not
resolve the incompatible shapes below.

| Field | Paper §3.1.1 | Public Qwen3-4B | Qwen3-4B-Instruct-2507 |
|---|---:|---:|---:|
| Hidden width | 4096 | 2560 | 2560 |
| Layers | 32 | 36 | 36 |
| Query heads | 32 | 32 | 32 |
| KV heads | unspecified | 8 (GQA) | 8 (GQA) |
| Head dimension | unspecified | 128 | 128 |
| Vocabulary tensor rows | approximately 32K | 151936 | 151936 |
| SwiGLU intermediate width | unspecified | 9728 | 9728 |
| Input/output embeddings tied | unspecified for backbone | yes | yes |
| RoPE base | unspecified | 1,000,000 | 5,000,000 |
| `max_position_embeddings` | unspecified here | 40960 | 262144 |

The model card's native-context statement and config's position limit are
separate fields; 40960 above is the literal config value.

**4096 is the public model's concatenated query width, not its hidden width:**
32 × 128 = 4096, while `hidden_size / num_attention_heads = 80` is NOT the head
dimension. Q/O weights have shape 4096×2560 / 2560×4096; K/V weights are
1024×2560. Q/K RMSNorm and RoPE also matter. A conventional 4096-wide,
32-layer MHA model cannot directly accept these weights. The old tiny parity
test accidentally used equal hidden and query widths; it now tests unequal
widths as well as GQA.

The original public `Qwen/Qwen3-4B` already has post-training and both thinking
modes. Thus an informal “Instruct” suffix could be a naming imprecision. The
separate `-Instruct-2507` release still has the same width/depth/vocabulary and
cannot explain the paper's tuple. Its different weights, RoPE and prompting
semantics also make silently substituting it inappropriate. The tuple alone is
insufficient to identify another model or infer its total parameters: FFN width,
KV geometry and embedding tying are missing.

## Exact independent parameter accounting

Let d=2560, V=151936, L=36, q=32×128=4096, kv=8×128=1024,
and m=9728. A bias-free Qwen layer has

`2*d*q + 2*d*kv + 3*d*m + 2*d + 2*128` parameters.

| Backbone tensor/component | Per layer | All layers / shared total |
|---|---:|---:|
| Q projection | 10,485,760 | 377,487,360 |
| K projection | 2,621,440 | 94,371,840 |
| V projection | 2,621,440 | 94,371,840 |
| O projection | 10,485,760 | 377,487,360 |
| MLP gate, up, down (each) | 24,903,680 | 896,532,480 each |
| Two block RMSNorms | 5,120 | 184,320 |
| Q/K RMSNorms | 256 | 9,216 |
| All decoder layers | 100,930,816 | 3,633,509,376 |
| Token embedding / tied LM head | — | 388,956,160 (count once) |
| Final RMSNorm | — | 2,560 |
| **Backbone** | — | **4,022,468,096** |

For the existing reconstruction, let PAW rank r=32, CH width c=32, SG width
g=32, and k_max=15. All auxiliary modules remain separate **across layers**;
only the vocabulary matrix is tied, and PAW transforms are shared across
horizons within a layer. The parameter formulas include biases:

- PAW per layer: window `(d+1)`, down/up `2dr`, horizon offsets `kr`,
  compressor `2dr+r+d`; total `4dr+2d+r+kr+1`.
- CH per layer: BIOS `4d+4`, Q/K/V `3(dc+c)`, output `cd+d`;
  total `4dc+5d+3c+4`.
- SG per layer: `(2dg+g)+(g+1)`.

| Component | Per layer | All 36 layers |
|---|---:|---:|
| PAW | 333,313 | 11,999,268 |
| CH | 340,580 | 12,260,880 |
| SG | 163,905 | 5,900,580 |
| **Auxiliary total** | **837,798** | **30,160,728** |
| **Backbone + auxiliaries** | — | **4,052,628,824** |

Overhead = **0.7498065%** of the backbone. This is a result about THIS code,
not validation of the paper's historical count. PR #3's k_max=8 produces
30,152,664 auxiliary parameters; increasing to 15 adds only 8,064 parameters
because horizons are rank-sized offsets, although it greatly increases work.
Disabling modules does not delete their allocated parameters; the inventory
reports allocated and trainable counts separately. Rounded “4.00B” table entries
are not exact tensor counts and do not substantiate a <1% difference.

### Dense-head interpretation versus unspecified sharing

C.1 defines horizon-specific vocabulary distributions and ties the embedding
used for expectation. It does **not** specify that prediction weights equal
that embedding, give a factorization rank, or unambiguously prescribe sharing
across layers. Therefore independent dense heads are a conditional literal
implementation, not a proven historical implementation.

Weights alone, excluding biases/compressors:

| Assumption | Public Qwen3-4B count |
|---|---:|
| One new d×V head | 388,956,160 |
| 15 horizon heads, shared across every layer | 5,834,342,400 |
| 15 independent heads at each of 36 layers | 210,036,326,400 |

Using the paper's own d=4096, L=32, k=15 and V=32000–32768 gives
62,914,560,000–64,424,509,440 head weights. These constructions cannot fit
“a few million”. Some unstated tying/compression is necessary for that claim;
its existence and exact form cannot be recovered from the equations alone.

Likewise CH's conceptual cross-attention does not specify all projection widths
or sharing. This repository's old full-width CH would add 944,455,824 parameters
on public Qwen shapes (2,148,532,352 on the stated paper shapes). The rank-32 CH
is a reconstruction choice, not a paper requirement.

## Compute is a separate budget

Count one multiply-add as two FLOPs. A public Qwen layer uses 201,850,880
FLOPs/token for Q/K/V/O and SwiGLU matrices. Attention adds `4*q*n` for n visible
keys, hence 8,388,608 at n=512. The final LM projection adds 777,912,320 once
per token. Full baseline total is approximately **8.3465 GFLOPs/token** at this
context length, excluding norms, activations and other non-matrix work.

The present PAW computes `E @ up(down(h) + horizon[r])` for every horizon.
The effective linear map is shared across horizons, with horizon-specific
additive offsets; this is not the same parameterization as arbitrary W_r.
Each horizon still costs `2*V*d + 4*d*r = 778,240,000` FLOPs per layer.
Top-K is selected **after** dense logits/softmax, so it saves expectation
aggregation, not the vocabulary projection. With eight active horizons across
36 layers, PAW alone is approximately **224.1451 GFLOPs/token**, **26.85×** the
entire baseline. Even skipping all backbone layers cannot remove this cost:
PAW runs before SG. This is a controlled cost scenario, not an observed span or
an A100 latency prediction. With one horizon the same fundamental issue remains.

CH projections with one newly closed chunk cost `8*d*c+8*d` per executed layer;
attention adds `4*c*M` for M cached chunks. Its cost grows with the chunk cache.
SG costs approximately `4*d*g+2*g` per layer, even when the layer is skipped.
Native Python cache concatenations, softmax/top-K and memory traffic add costs
not represented by these matrix formulas.

B.3 Eq. (4) counts executed decode layers; §3.1.4 describes end-to-end compute
including prefill. The equation contains neither a prefill term nor an explicit
always-paid PAW/SG term. This is a reporting-definition inconsistency unless an
additional accounting procedure is supplied. A weighted execution ratio can be
useful, but it must not be relabeled as complete FLOPs. `fbs.benchmark` already
labels it `backbone_execution_fraction`; that distinction remains.

## Training scale and feasibility

The documented 12B tokens / approximately 2M tokens per update imply about
6,000 updates. Gradient accumulation reduces microbatch storage, not total work
or optimizer-state requirements. The current straight-through training path
executes dense attention/FFN branches and all k_max preview horizons.

Using three times forward matrix FLOPs as a forward+backward approximation:

| 12B-token calculation | Approximate matrix FLOPs |
|---|---:|
| Public baseline, excluding training attention | 2.8960 × 10^20 |
| Baseline + present PAW, k_max=15 | 1.5419 × 10^22 |

At the [A100's dense BF16 theoretical peak of 312 TFLOPS](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf),
these correspond to about **10.7 and 572 days** at ideal utilization. These
are idealized arithmetic budgets, not measured runtimes or claims about what
hardware was actually used. Real utilization, CH, optimizer work and activation
recomputation worsen them; a materially different PAW implementation could
change them. The single-A100 description alone is not proof that baseline
training was impossible.

For the 4.0526B-parameter reconstruction, BF16 weights alone occupy about
8.11 GB. A conventional mixed-precision Adam layout with FP32 master weights
and moments needs roughly 16 bytes/parameter including gradients, about
64.84 GB before activations. Actual optimizer layouts differ: this runner's
plain AdamW follows parameter dtype for its moments and does not implement
FP32 master weights or mixed-precision loss scaling.

At microbatch 1 and sequence length 2048, one BF16 `[B,S,V]` tensor is
0.622 GB. One retained tensor of this size for every one of 15 horizons and
36 layers totals **336 GB**; this is an illustrative tensor-volume calculation,
not a measured peak. Autograd/CE can retain additional tensors or use FP32.
The runner has no activation checkpointing, sharding or fused vocabulary loss.
**The public configuration is a transparent reconstruction specification, not a
validated single-A100 12B-token training recipe.** Reduced tests cannot resolve
these full-scale feasibility issues.

## Public reconstruction decision and design provenance

Use `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`, with its own
pinned tokenizer and unmodified config. It is the least-assumptive match to the
paper's repeatedly named Qwen3-4B family. This is an explicit baseline choice,
not identification of the historical checkpoint. Do not silently substitute the
2507 release or manufacture a 4096-wide Qwen model. Continued pretraining uses
plain text; chat evaluation must record its template and thinking mode. For a
short non-thinking reconstruction, `enable_thinking=False` is an explicit choice.

| Design | Evidence status / current implementation |
|---|---|
| Prefix-only multi-horizon preview; future tokens only as labels | Paper-specified concept |
| k_max=15 | Paper default (C.3/C.6); corrected from PR #3's example value 8 |
| Soft training window / hard inference span | Paper-specified form; gamma=4 is a code choice |
| Top-K expectation approximation | Allowed by paper; K=5 chosen in code |
| PAW shared low-rank transform + horizon offsets | Reconstruction only; rank 32 unrecovered |
| PAW weighted mean + bottleneck MLP | Reconstruction only; C.5 default is grouped 1D conv, kernel 3, plus pooling |
| CH BIOS/pooling/cross-attention | Paper concept; rank 32 and online boundary cases are reconstruction choices |
| SG hidden width 32 and using normalized hidden state as its residual signal | Reconstruction choices |
| 12B tokens, ~2M batch tokens, 2e-5 LR, 4% warmup | Paper values; these have not been run at full scale |
| Sequence length 2048, loss weights, optimizer/precision details | Reconstruction choices |
| Public checkpoint/revision and non-thinking prompting policy | Explicit new public baseline, historical equivalence unknown |

## Recompute and inspect

```bash
OMP_NUM_THREADS=1 python -m fbs.audit --verify-meta \
  --output validation/public_architecture_audit.json
# Optional: include each unique parameter name, shape and count.
OMP_NUM_THREADS=1 python -m fbs.audit --verify-meta \
  --tensor-output /tmp/fbs_parameter_tensors.json
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Meta-device checking constructs the **full shapes**, without downloading weights,
allocating a 4B model in RAM, or executing a full-size forward pass. Separately,
a small actual CPU profiler test confirms that PAW's matrix FLOP formula includes
the full-vocabulary projection even with top_k=1. Neither check is full numerical
reproduction, a trained checkpoint check, or an A100 benchmark.

## Defensible correction and outstanding evidence

An immediate correction can accurately say:

> Section 3.1.1 gives an architecture tuple that does not match the public
> Qwen3-4B configuration. The public checkpoint uses hidden width 2560, 36
> layers, 32 query heads, 8 KV heads of dimension 128, intermediate width 9728,
> and 151936 vocabulary rows. The published text does not identify the exact
> experimental checkpoint or enough auxiliary dimensions/sharing rules to
> reconstruct the stated parameter and compute budgets. The current repository
> provides a separately identified public reconstruction.

It cannot yet accurately say “the experiments used that public configuration;
only the prose was wrong.” That requires original checkpoint/config/tokenizer
metadata, tensor shapes, the auxiliary state dictionary and original launch/log
records. If those prove the public architecture was used, correct §3.1.1 and
publish exact counts and cost accounting. If they prove a different architecture,
identify it and reassess weight copying, parameter matching and the baseline
comparison. If artifacts cannot be recovered, keep historical equivalence and
original speed/quality results explicitly unverified. No new low-rank module or
small-model test should be presented as evidence of historical equivalence.
