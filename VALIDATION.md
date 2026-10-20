# Executed validation

## Architecture audit update — 2026-10-14

On Python 3.12, PyTorch 2.5.1+cpu and Transformers 4.51.3:

- **20 tests passed**, with no skipped tests. The original 16 checks remain.
- Full public 4B shapes were constructed on the **meta device**. Exact unique
  tensor counts agree with the independent public BF16 weight-index size and
  the analytic counts for PAW/CH/SG. No pretrained weights or full-size forward
  execution were involved. Results: [public_architecture_audit.json](validation/public_architecture_audit.json).
- Qwen output-parity checking now uses `hidden_size != heads * head_dim` and GQA,
  matching the public model's non-square Q/O projection geometry.
- Incorrect expected backbone fields and a mutable revision are rejected;
  tied and frozen parameters are counted correctly.
- A small CPU operator-profile check matches the PAW matrix FLOP formula even
  with top_k=1. This verifies accounting, not performance or a trained model.
- The synthetic stage-1 runner was re-executed for 4 steps / 128 tokens to
  verify the expanded parameter report is written correctly. This remains a
  native toy workflow; no full-size training was attempted.

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 python -m fbs.audit --verify-meta \
  --output validation/public_architecture_audit.json
```

The numerical claims about parameter overhead now apply to the explicitly
specified **public reconstruction only**. Published experimental equivalence,
full training feasibility, scores and A100 timing remain unverified. The earlier
end-to-end synthetic runs below are retained as historical validation records.

## Previous pipeline validation — 2026-10-13

Checked on 2026-10-13 with Python 3.12, PyTorch 2.5.1+cpu,
Transformers 4.51.3 and Accelerate 1.6.0. No CUDA device was used and no
full-size Qwen checkpoint was downloaded.

## Automated checks

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

All **16 tests passed**, including:

- Prefix causality, shifted preview targets and weighted preview loss.
- Partial batches, padding and finite training gradients.
- Hard skipping executes no Q/K/V or FFN and preserves absolute positions.
- Dense and cached no-skip outputs agree; zero preview stays zero despite bias.
- Disabling SG avoids both gate execution and gate gradients.
- Checkpoint reload and fixed-length generation.
- A locally constructed tiny Qwen3 recovers its original dense/cached logits
  with FBS modules disabled; no remote model download is needed for this test.
- BIOS alignment and development/held-out overlap rejection.
- PPO calibration changes only PAW/SG parameters and leaves the backbone frozen.

## End-to-end execution

The measured training log and evaluation output are retained in
[`validation/cpu_smoke_results.json`](validation/cpu_smoke_results.json).

The following workflow completed on the supplied synthetic fixtures:

```bash
OMP_NUM_THREADS=1 python -m fbs.run --config configs/smoke.json
OMP_NUM_THREADS=1 python -m fbs.calibrate \
  --checkpoint runs/smoke/latest.pt \
  --prompts data/smoke_calibration.jsonl \
  --held-out-prompts data/smoke_heldout.jsonl \
  --output runs/calibration_final/model.pt \
  --new-tokens 4 --epochs 1 --ppo-epochs 2
OMP_NUM_THREADS=1 python -m fbs.evaluate \
  --checkpoint runs/calibration_final/model.pt \
  --data data/smoke_evaluation.jsonl --shots 0 \
  --output-dir runs/evaluation_final
```

Stage 1 completed **4 optimizer steps / 128 target tokens** and wrote a
reloadable checkpoint with configuration, optimizer and RNG state. Calibration
completed two prompt trajectories and two PPO passes per trajectory. Evaluation
reloaded that calibrated checkpoint and produced raw predictions and summaries
for two held-out fixtures at thresholds 1.0, 0.9, 0.8 and 0.7.

The timing entry point also completed a CPU smoke run with prompt length 3,
4 generated tokens, 1 warmup and 2 repetitions. This verified timing output and
execution counters, not a useful speed benchmark. The synthetic fixture scores
are not reported as model quality.

## Not established by these checks

The checks do not establish full-size pretrained training, resume equivalence
across machines, official benchmark scores, batch-4/8 deployment, A100 latency,
parameter-overhead claims or equivalence to the missing original checkpoints.
See [REPRODUCTION.md](REPRODUCTION.md) for the unresolved paper details and the
inputs required for complete numerical replication.
