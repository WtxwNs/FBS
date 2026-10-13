"""PPO calibration of SG and PAW with a frozen full-compute reference.

Local records must contain prompt_ids, answer_ids, prompt_text and split='dev'.
This implements the paper reward and clipped gate-policy objective. The exact
original development prompt list was not released; supply it explicitly.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, save_checkpoint
from .inference import new_state, decode_step
from .data import assert_disjoint, normalize_prompt


def trace(model, tokens, actions=None, sample=False):
    state = new_state(model)
    logits, probs, decisions = [], [], []
    for index, token in enumerate(tokens):
        output, probability, choice = decode_step(model, token, state,
            gate_actions=None if actions is None else actions[index], sample=sample)
        logits.append(output)
        probs.append(probability)
        decisions.append(choice)
    return torch.cat(logits, dim=1), torch.stack(probs), decisions, state


@torch.no_grad()
def rollout(model, prompt, new_tokens):
    state = new_state(model)
    inputs = list(prompt.unbind())
    actions, probabilities = [], []
    for token in inputs:
        logits, prob, action = decode_step(model, token, state, sample=True)
        probabilities.append(prob)
        actions.append(action)
    for _ in range(new_tokens - 1):
        token = logits[0, -1].argmax()
        inputs.append(token)
        logits, prob, action = decode_step(model, token, state, sample=True)
        probabilities.append(prob)
        actions.append(action)
    return torch.stack(inputs), torch.stack(probabilities), actions, state


def answer_nll(model, prompt, answer, actions=None):
    inputs = torch.cat((prompt, answer[:-1]))
    logits, _, _, _ = trace(model, inputs, actions=actions)
    return F.cross_entropy(logits[0, len(prompt)-1:], answer)


def update(model, reference, optimizer, prompt, answer, new_tokens=128, clip=.2, alpha=.1, beta=.1, epochs=2):
    if len(answer) > new_tokens or len(answer) == 0:
        raise ValueError("Gold answer must fit the rollout horizon")
    model.eval()  # discrete, identical train/deployment PAW and cache semantics
    with torch.no_grad():
        inputs, old_probs, actions, state = rollout(model, prompt, new_tokens)
        gold_inputs = torch.cat((prompt, answer[:-1]))
        candidate = answer_nll(model, prompt, answer, actions[:len(gold_inputs)])
        # Fixed reference with all gate actions forced to compute.
        zeros = [[False] * len(reference.layers) for _ in gold_inputs]
        full = answer_nll(reference, prompt, answer, zeros)
        saved = state.skipped / (state.executed + state.skipped)
        reward = alpha * saved - beta * torch.relu(candidate - full).item()
        chosen = torch.tensor(actions, dtype=torch.bool, device=prompt.device)
        old_logp = torch.where(chosen, old_probs.clamp(1e-6,1-1e-6).log(), (-old_probs.clamp(1e-6,1-1e-6)).log1p())
    losses = []
    for _ in range(epochs):
        _, probabilities, _, _ = trace(model, inputs, actions=actions)
        probabilities = probabilities.clamp(1e-6, 1-1e-6)
        logp = torch.where(chosen, probabilities.log(), (-probabilities).log1p())
        ratio = (logp - old_logp).exp()
        advantage = torch.full_like(ratio, reward)  # zero state-independent baseline
        objective = torch.minimum(ratio * advantage, ratio.clamp(1-clip,1+clip) * advantage)
        loss = -objective.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        losses.append(loss.item())
    return {"reward": reward, "skip_fraction": saved, "candidate_gold_nll": candidate.item(),
            "reference_gold_nll": full.item(), "ppo_loss": losses[-1]}


def freeze_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.layers:
        for module in (layer.paw, layer.sg):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--held-out-prompts", required=True, help="JSONL with prompt_text for every held-out evaluation prompt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    held = [json.loads(line)["prompt_text"] for line in Path(args.held_out_prompts).read_text().splitlines() if line.strip()]
    if not rows or any(row.get("split") != "dev" for row in rows):
        raise ValueError("Calibration requires explicit development records")
    texts = [row["prompt_text"] for row in rows]
    if len(set(map(normalize_prompt, texts))) != len(texts):
        raise ValueError("Duplicate normalized calibration prompts")
    assert_disjoint(texts, held)
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    if not all(layer.use_sg for layer in model.layers):
        raise ValueError("Enable SG in the training checkpoint before calibration")
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    freeze_backbone(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    records = []
    for epoch in range(args.epochs):
        for index in torch.randperm(len(rows)).tolist():
            row = rows[index]
            result = update(model, reference, optimizer, torch.tensor(row["prompt_ids"], device=args.device),
                            torch.tensor(row["answer_ids"], device=args.device), args.new_tokens, epochs=args.ppo_epochs)
            result.update(epoch=epoch, prompt_index=index)
            records.append(result)
            print(json.dumps(result), flush=True)
    save_checkpoint(args.output, model, calibration=vars(args), records=records,
                    source_config=metadata["config"], seed=args.seed)


if __name__ == '__main__':
    main()
