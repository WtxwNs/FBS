# Executed validation

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
