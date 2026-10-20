# FBS: Modeling Native Parallel Reading inside a Transformer

Reference implementation for [the FBS paper](https://arxiv.org/abs/2601.21708)
([ACL Findings](https://aclanthology.org/2026.findings-acl.200/)).

FBS combines predictive preview (PAW), online chunk context (CH), and a learned
skip gate (SG). The hard executor bypasses a skipped token-layer's Q/K/V,
attention, CH and FFN, leaves its layer cache unchanged, and advances the global
token position.

**Status:** this repository now includes training, Qwen3 weight transfer,
cached hard skipping, PPO gate calibration, local evaluation and fixed-length
timing. It is a tested reference reconstruction. **The paper's full numerical
results have not been reproduced.** Original checkpoints, data manifests and
some architecture details are missing or inconsistent. See
[REPRODUCTION.md](REPRODUCTION.md) before running large experiments.
The [architecture and cost audit](ARCHITECTURE_AUDIT.md) separates confirmed
publication conflicts from unrecovered experiment details. The public low-rank
reconstruction adds 30,160,728 parameters (0.750%), but still performs expensive
full-vocabulary PAW projections. Its small parameter budget is not evidence of
the paper's compute savings.

## Installation and checks

Validated with Python 3.12 and PyTorch 2.5.1 on CPU. Install a PyTorch build
compatible with your GPU separately if required; record that changed environment.
The original timing protocol uses A100 hardware, Python 3.10 and PyTorch 2.3.1.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-hf.txt
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

`requirements.txt` is enough for the small native model. `requirements-hf.txt`
adds the tested Qwen3 adapter version. The tests instantiate a tiny random Qwen3
locally, so no multi-gigabyte checkpoint download is required.

## End-to-end CPU smoke workflow

The supplied integer sequences are **synthetic plumbing fixtures**, not
substitutes for benchmark examples or training data.

```bash
python -m fbs.run --config configs/smoke.json
python -m fbs.calibrate \
  --checkpoint runs/smoke/latest.pt \
  --prompts data/smoke_calibration.jsonl \
  --held-out-prompts data/smoke_heldout.jsonl \
  --output runs/calibration/model.pt \
  --new-tokens 4 --epochs 1 --ppo-epochs 2
python -m fbs.evaluate \
  --checkpoint runs/calibration/model.pt \
  --data data/smoke_evaluation.jsonl --shots 0 \
  --output-dir runs/evaluation
python -m fbs.benchmark \
  --checkpoint runs/calibration/model.pt \
  --prompts data/smoke_heldout.jsonl --prompt-length 3 \
  --new-tokens 4 --warmup 1 --repetitions 2 \
  --output runs/timing.json
```

The smoke training run writes four optimizer steps, 128 target tokens, arguments,
loss records, parameter counts and a reloadable checkpoint. Existing training
outputs are protected; use a new output directory or explicitly `--resume` from
a checkpoint with the same configuration.

The original word-level example remains available:

```bash
python -m fbs.train --data_path data/sample.txt --epochs 1 \
  --d_model 64 --n_layers 2 --n_heads 4 --d_ff 128 --k_max 4 \
  --seed 42 --output_dir runs/word_toy
```

## Use pretrained Qwen3 weights

`fbs.hf.QwenFBSModel` copies the actual checkpoint's attention, MLP, normalization,
embedding and language-head weights. It preserves grouped-query attention,
Q/K normalization and rotary positions. Disabling PAW/CH/SG is tested to recover
the original Qwen3 logits in both cached and dense execution.

Prepare local JSONL documents using an explicitly selected tokenizer revision:

```bash
python -m fbs.data --input corpus.jsonl --output data/prepared_corpus.jsonl \
  --tokenizer Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c
```

Input documents have `text` and `source_id`. For Chinese BIOS supervision add
`--chinese-bios --idiom-lexicon path/to/original_lexicon.txt` and install `pkuseg`.
The exact original segmentation model/lexicon must be supplied separately.

`configs/qwen3_reconstruction.example.json` now specifies this public checkpoint,
its immutable revision, BF16 loading and expected tensor geometry. This is a
new public reconstruction baseline, not a recovered original checkpoint.
Run the weight-free audit first:

```bash
python -m fbs.audit --verify-meta --output validation/public_architecture_audit.json
```

Copy the example, replace data/output paths, and review module/loss settings and
the [training feasibility limits](ARCHITECTURE_AUDIT.md#training-scale-and-feasibility)
before launching a deliberately budgeted experiment:

```bash
python -m fbs.run --config configs/my_run.json --device cuda
```

The 12B-token budget, approximately 2M-token accumulated batch, 2e-5 learning rate
and 4% warmup come from the paper. Low-rank module dimensions, auxiliary-loss
coefficients and sequence length in the example are **reconstruction defaults**,
not recovered original values. `k_max=15` is the paper default; the former
example's value 8 was not that default. The mean-plus-MLP preview compressor is
also a reconstruction; the paper's default is grouped convolution plus pooling.
The code does not shrink the backbone to assert
parameter matching; `parameters.json` gives actual unique parameter counts.
Full pretraining needs substantial memory and compute. A small run or adapter-only
training is not a replication of the 12B-token experiment. The current dense
PAW training graph is not a validated single-A100 recipe; BF16 loading alone
does not resolve its activation memory or compute cost. Plain AdamW here does
not supply FP32 master weights or mixed-precision loss scaling.

## Evaluation and timing

`fbs.evaluate` scores prepared held-out multiple-choice records or continuation
likelihoods and writes per-example predictions. Explicit `shots` fields prevent
mislabeling zero-shot fixtures as five-shot results. It can sweep SG thresholds
on the same checkpoint and data. It does not replace official code-execution or
free-response benchmark graders.

`fbs.benchmark` defaults to 512 prompt tokens, 128 generated tokens, 20 warmups,
50 repetitions, greedy decoding and CUDA synchronization. Tokenization/I/O
are outside the timer. It records prefill/decode/total time and actual token-layer
execution counts. The first generated token comes from the final prefill logits,
so 128 generated tokens require 127 subsequent one-token forward calls.
The execution fraction excludes auxiliary-module cost and **is not TFLOPs**.
This Python executor is a correctness reference, not an optimized GPU kernel.

## Source map

- `fbs/model.py`: PAW, causal online CH, stochastic straight-through SG, dense training.
- `fbs/inference.py`: sparse per-layer KV cache and hard execution.
- `fbs/hf.py`: Qwen3 weight-preserving adapter.
- `fbs/audit.py`: full-shape parameter verification and scoped analytic FLOPs.
- `fbs/data.py`: character-offset BIOS labels and prompt-overlap checks.
- `fbs/run.py`, `fbs/checkpoint.py`: token-budget training and reproducible checkpoints.
- `fbs/calibrate.py`: SG/PAW PPO with a frozen full-compute reference.
- `fbs/evaluate.py`, `fbs/benchmark.py`: prediction artifacts and measured timing.
- [VALIDATION.md](VALIDATION.md): checks actually executed for this revision.

The old root `fbs_model_epoch1.pt` is a legacy toy checkpoint without a saved
architecture configuration. It is not a paper checkpoint; the new loader rejects
it rather than guessing its architecture.
