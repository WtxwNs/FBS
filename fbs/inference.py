"""Reference hard-skip executor with sparse per-layer KV and online CH state.

Batch size one is explicit. Global positions advance even when individual
layers skip; their KV cache contains only actually computed positions.
This correctness-oriented Python executor is not an optimized CUDA kernel.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class LayerCache:
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    chunks: dict = field(default_factory=lambda: {"pending": [], "chunks": []})


@dataclass
class DecodeState:
    layers: list
    position: int = 0
    executed: int = 0
    skipped: int = 0


def new_state(model):
    return DecodeState([LayerCache() for _ in model.layers])


def decode_step(model, input_id, state, threshold=.8, gate_actions=None, sample=False):
    """One prefix-conditioned step. Optional forced actions replay an SG trajectory.

    The returned probabilities keep gradients for gate-policy calibration; cache
    values keep gradients only when the caller enables autograd.
    """
    if input_id.numel() != 1:
        raise ValueError("Reference sparse executor supports batch size one")
    if state.position >= model.max_seq_len:
        raise ValueError("Decode exceeds max_seq_len")
    token = input_id.reshape(1, 1)
    pos = torch.tensor([[state.position]], device=token.device)
    h = model.embed(token, pos)
    probabilities, decisions = [], []
    for index, (layer, cache) in enumerate(zip(model.layers, state.layers)):
        x = layer.norm1(h)
        z = layer.paw(x, model.token_emb.weight)[0] if layer.use_paw else torch.zeros_like(x)
        if layer.use_sg:
            _, prob = layer.sg(x, z, threshold=threshold)
            probability = prob.reshape(())
        else:
            probability = x.new_zeros(())
        if not layer.use_sg or threshold >= 1:
            skip = False
        elif gate_actions is not None:
            skip = bool(gate_actions[index])
        elif sample:
            skip = bool(torch.bernoulli(probability.detach()).item())
        else:
            skip = probability.detach().item() > threshold
        probabilities.append(probability)
        decisions.append(skip)
        if skip:
            state.skipped += 1
            continue  # no Q/K/V, CH, attention, FFN, or KV append
        attn = layer.attn
        if hasattr(attn, "project"):
            q, k, v = attn.project(x, pos)
            cache.keys.append(k)
            cache.values.append(v)
            cache.positions.append(state.position)
            sa = attn.attend(q, torch.cat(cache.keys, dim=2), torch.cat(cache.values, dim=2))
        else:
            q = attn.q_proj(x).view(1, attn.n_heads, attn.d_head)
            k = attn.k_proj(x).view(1, attn.n_heads, attn.d_head)
            v = attn.v_proj(x).view(1, attn.n_heads, attn.d_head)
            cache.keys.append(k)
            cache.values.append(v)
            cache.positions.append(state.position)
            keys = torch.cat(cache.keys, dim=0).transpose(0, 1)
            values = torch.cat(cache.values, dim=0).transpose(0, 1)
            weights = torch.softmax(torch.einsum("bhd,hkd->bhk", q, keys) / attn.d_head ** .5, dim=-1)
            context = torch.einsum("bhk,hkd->bhd", weights, values).reshape(1, 1, -1)
            sa = attn.o_proj(context)
        ch = layer.ch.step(x.reshape(-1), cache.chunks).reshape_as(h) if layer.use_ch else torch.zeros_like(h)
        fused = h + sa + z + ch
        h = fused + layer.ffn(layer.norm2(fused))
        state.executed += 1
    state.position += 1
    return model.lm_head(model.norm(h)), torch.stack(probabilities), decisions


def prefill(model, input_ids, threshold=.8, state=None):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Expected one nonempty prompt [1, length]")
    state = new_state(model) if state is None else state
    outputs = [decode_step(model, input_ids[:, i], state, threshold)[0] for i in range(input_ids.shape[1])]
    return torch.cat(outputs, dim=1), state


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=128, threshold=.8):
    if model.training:
        raise ValueError("Call model.eval() before generation")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    logits, state = prefill(model, input_ids, threshold)
    emitted = []
    for i in range(max_new_tokens):
        token = logits[:, -1].argmax(-1)
        emitted.append(token)
        if i + 1 < max_new_tokens:
            logits, _, _ = decode_step(model, token, state, threshold)
    return torch.cat((input_ids, torch.stack(emitted, dim=1)), dim=1), state
