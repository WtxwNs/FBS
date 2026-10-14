"""Single-device continued pretraining from a recorded JSON configuration.

Data are supplied locally as tokenized JSONL. No hidden download, fabricated
corpus mixture, or claim of reproducing historical paper measurements.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch

from .model import FBSModel
from .checkpoint import save_checkpoint, load_checkpoint


def windows(path, seq_len, vocab_size):
    found = False
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            ids = row["input_ids"]
            labels = row.get("bios_labels", [-100] * len(ids))
            if len(labels) != len(ids) or any(i < 0 or i >= vocab_size for i in ids):
                raise ValueError("Invalid token IDs or BIOS alignment")
            for start in range(0, len(ids) - 1, seq_len):
                end = min(start + seq_len, len(ids) - 1)
                found = True
                yield (torch.tensor([ids[start:end]], dtype=torch.long),
                       torch.tensor([ids[start+1:end+1]], dtype=torch.long),
                       torch.tensor([labels[start:end]], dtype=torch.long))
    if not found:
        raise ValueError("Training data has no next-token targets")


def train(config, config_path, device, resume=None):
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    root = Path(config_path).resolve().parent
    data = (root / config["data"]).resolve()
    output = (root / config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "config.json").exists():
        raise FileExistsError("Output already contains a run; select a new output or --resume")
    if resume:
        model, checkpoint = load_checkpoint(resume, device)
        if checkpoint.get("training_config") != config:
            raise ValueError("Resume configuration differs from saved training configuration")
    elif "pretrained" in config:
        from .hf import QwenFBSModel
        spec = config["pretrained"]
        dtype_name = spec.get("dtype", "float32")
        if dtype_name not in ("float32", "bfloat16", "float16"):
            raise ValueError("pretrained.dtype must be float32, bfloat16 or float16")
        model = QwenFBSModel.from_pretrained(spec["model_id"], spec["revision"], device=device,
                                            dtype=getattr(torch, dtype_name),
                                            expected_config=spec.get("expected_config"),
                                            **config.get("fbs", {}))
        checkpoint = {}
    else:
        model = FBSModel(**config["model"]).to(device)
        checkpoint = {}
    model.loss_weights = config.get("loss_weights", {"preview": .1, "bios": .1, "align": .01, "gate": .01})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 2e-5),
                                 weight_decay=config.get("weight_decay", .01))
    completed = checkpoint.get("step", 0)
    tokens = checkpoint.get("tokens", 0)
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.startswith("cuda") and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if resume and (output / "train.jsonl").exists():
        records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines() if line.strip()]
        (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records if row["step"] <= completed))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "parameters.json").write_text(json.dumps(model.parameter_report(), indent=2) + "\n")
    seq_len = config.get("seq_len", 2048)
    if seq_len > model.max_seq_len:
        raise ValueError("Training context exceeds checkpoint context length")
    budget = config.get("token_budget", 12_000_000_000)
    batch_tokens = config.get("batch_tokens", 2_000_000)
    max_steps = config.get("max_steps", math.ceil(budget / batch_tokens))
    if min(budget, batch_tokens, max_steps, seq_len) < 1:
        raise ValueError("Training budgets must be positive")
    def repeat():
        while True:
            yield from windows(data, seq_len, model.vocab_size)
    stream = repeat()
    pending = next(stream)
    offset = 0
    skip_tokens = tokens
    while skip_tokens:
        available = pending[0].shape[1] - offset
        take = min(skip_tokens, available)
        offset += take
        skip_tokens -= take
        if offset == pending[0].shape[1]:
            pending, offset = next(stream), 0
    model.train()
    for step in range(completed, max_steps):
        if tokens >= budget:
            break
        optimizer.zero_grad()
        current_tokens, weighted_loss = 0, 0.
        # Normalize by actual token count after backward, including partial batches.
        while current_tokens < batch_tokens and tokens + current_tokens < budget:
            remaining = min(batch_tokens - current_tokens, budget - tokens - current_tokens,
                            pending[0].shape[1] - offset)
            inputs, targets, bios = (x[:, offset:offset+remaining].to(device) for x in pending)
            offset += remaining
            if offset == pending[0].shape[1]:
                pending, offset = next(stream), 0
            logits, loss = model(inputs, targets=targets, pseudo_labels=bios)
            count = targets.numel()
            (loss * count).backward()
            current_tokens += count
            weighted_loss += loss.item() * count
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(current_tokens)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.))
        fraction = step / max(max_steps, 1)
        warmup = config.get("warmup_fraction", .04)
        multiplier = (step + 1) / max(1, math.ceil(max_steps * warmup)) if fraction < warmup else .5 * (1 + math.cos(math.pi * (fraction - warmup) / max(1 - warmup, 1e-6)))
        for group in optimizer.param_groups:
            group["lr"] = config.get("learning_rate", 2e-5) * multiplier
        optimizer.step()
        tokens += current_tokens
        record = {"step": step+1, "tokens": tokens, "loss": weighted_loss/current_tokens, "learning_rate": optimizer.param_groups[0]["lr"]}
        with (output / "train.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        state = dict(training_config=config, optimizer=optimizer.state_dict(), step=step+1, tokens=tokens,
                     torch_rng=torch.get_rng_state())
        if device.startswith("cuda"):
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        if (step + 1) % config.get("save_every", 100) == 0 or step+1 == max_steps or tokens >= budget:
            save_checkpoint(output / "latest.pt", model, **state)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    train(config, args.config, args.device, args.resume)


if __name__ == "__main__":
    main()
