"""Fixed-length prefill/decode timing with actual execution counters.

Use identical tokenized prompts and hardware for every compared checkpoint.
Numbers from the Python reference executor are never claims of paper speedups.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from .checkpoint import load_checkpoint
from .inference import prefill, decode_step


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure(model, prompt, new_tokens=128, threshold=.8):
    device = prompt.device
    synchronize(device)
    start = time.perf_counter()
    logits, state = prefill(model, prompt, threshold)
    synchronize(device)
    prefill_end = time.perf_counter()
    prefill_executed, prefill_skipped = state.executed, state.skipped
    generated = []
    for i in range(new_tokens):
        token = logits[:, -1].argmax(-1)
        generated.append(int(token.item()))
        if i + 1 < new_tokens:
            logits, _, _ = decode_step(model, token, state, threshold)
    synchronize(device)
    end = time.perf_counter()
    return dict(prefill_ms=(prefill_end-start)*1000, decode_ms=(end-prefill_end)*1000,
                total_ms=(end-start)*1000, tokens_per_second=new_tokens/(end-start),
                generated_tokens=len(generated), prefill_executed=prefill_executed,
                prefill_skipped=prefill_skipped, decode_executed=state.executed-prefill_executed,
                decode_skipped=state.skipped-prefill_skipped,
                backbone_execution_fraction=state.executed/(state.executed+state.skipped))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True, help="JSONL input_ids; each must have exactly prompt-length tokens")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()
    if args.new_tokens < 1 or args.repetitions < 1 or args.warmup < 0:
        raise ValueError("Invalid measurement counts")
    model, _ = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    prompts = [torch.tensor([json.loads(line)["input_ids"]], device=args.device) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if not prompts or any(p.shape[1] != args.prompt_length for p in prompts):
        raise ValueError("Every measured prompt must match the exact requested length")
    for i in range(args.warmup):
        measure(model, prompts[i % len(prompts)], args.new_tokens, args.threshold)
    rows = [measure(model, prompts[i % len(prompts)], args.new_tokens, args.threshold) for i in range(args.repetitions)]
    result = dict(protocol=vars(args), device=str(next(model.parameters()).device), dtype=str(next(model.parameters()).dtype),
                  torch=torch.__version__, parameters=model.parameter_report(), runs=rows,
                  median={key: statistics.median(r[key] for r in rows) for key in rows[0]},
                  compute_note="Execution fraction excludes PAW/CH/SG, embeddings and LM head; it is not TFLOPs or a latency estimate.")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["median"], indent=2))


if __name__ == "__main__":
    main()
