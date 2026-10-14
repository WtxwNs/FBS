"""Exact parameter counts and explicitly scoped dense-matmul cost estimates.

No pretrained weights are needed. --verify-meta checks the formulas against
the full public architecture on PyTorch's metadata-only device. FLOPs count a
multiply-add as two operations; they are not profiler or latency measurements.
"""
import argparse
import json
from pathlib import Path


def parameter_inventory(model):
    groups = {name: 0 for name in ("backbone", "paw", "ch", "sg")}
    trainable = dict.fromkeys(groups, 0)
    tensors = []
    # named_parameters deduplicates tied Parameter objects by default.
    for name, parameter in model.named_parameters():
        group = next((key for key in ("paw", "ch", "sg") if f".{key}." in name), "backbone")
        count = parameter.numel()
        groups[group] += count
        trainable[group] += count if parameter.requires_grad else 0
        tensors.append({"name": name, "shape": list(parameter.shape), "parameters": count,
                        "group": group, "trainable": parameter.requires_grad})
    auxiliary = sum(groups[key] for key in ("paw", "ch", "sg"))
    return {"total": sum(groups.values()), "backbone": groups["backbone"],
            "auxiliary": auxiliary, "by_component": groups,
            "trainable_by_component": trainable,
            "overhead_fraction": auxiliary / groups["backbone"],
            "tensors": tensors}


def qwen_parameter_counts(c):
    """Dense Qwen3, SwiGLU, Q/K RMSNorm, global attention, tied or untied LM head."""
    if c["model_type"] != "qwen3" or c.get("attention_bias", False):
        raise ValueError("Audit formula supports bias-free dense Qwen3 only")
    d, v, layers = c["hidden_size"], c["vocab_size"], c["num_hidden_layers"]
    q, kv = c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"]
    per_layer = {"q_proj": d*q, "k_proj": d*kv, "v_proj": d*kv, "o_proj": q*d,
                 "gate_proj": d*c["intermediate_size"], "up_proj": d*c["intermediate_size"],
                 "down_proj": d*c["intermediate_size"], "input_layernorm": d,
                 "post_attention_layernorm": d, "q_norm": c["head_dim"], "k_norm": c["head_dim"]}
    embedding = v*d
    separate_lm_head = 0 if c.get("tie_word_embeddings", False) else v*d
    return {"per_layer": per_layer, "per_layer_total": sum(per_layer.values()),
            "all_layers": layers*sum(per_layer.values()), "embedding": embedding,
            "additional_lm_head": separate_lm_head, "final_norm": d,
            "total": layers*sum(per_layer.values())+embedding+separate_lm_head+d}


def auxiliary_parameter_counts(d, layers, horizons, preview_rank, chunk_rank, gate_hidden):
    r, c, g = preview_rank, chunk_rank, gate_hidden
    paw = {"window": d+1, "down": d*r, "up": r*d, "horizon_offsets": horizons*r,
           "compress_down": d*r+r, "compress_up": r*d+d}
    ch = {"bios": 4*d+4, "q_proj": d*c+c, "k_proj": d*c+c,
          "v_proj": d*c+c, "out_proj": c*d+d}
    sg = {"fc1": 2*d*g+g, "fc2": g+1}
    parts = {"paw": paw, "ch": ch, "sg": sg}
    totals = {name: layers*sum(values.values()) for name, values in parts.items()}
    return {"per_layer": parts, "all_layers": totals, "total": sum(totals.values())}


def cost_estimate(c, fbs, horizons, context, tokens):
    """Dominant dense matmuls for the existing, unfused PAW implementation.

    Decode: one query with `context` visible keys per executed layer. PAW is
    paid even on skipped layers. Training: all k_max horizons and dense backbone
    branches; the 3x forward factor is an approximation to forward+backward.
    """
    if horizons < 0 or context < 1 or tokens < 1:
        raise ValueError("Invalid cost scenario")
    d, v, layers = c["hidden_size"], c["vocab_size"], c["num_hidden_layers"]
    r, rank_ch, g = fbs["preview_rank"], fbs["chunk_rank"], fbs["gate_hidden"]
    counts = qwen_parameter_counts(c)
    linear = 2*sum(value for name, value in counts["per_layer"].items() if name.endswith("_proj"))
    attention = 4*c["num_attention_heads"]*c["head_dim"]*context
    lm_head = 2*d*v
    # down/up are currently recomputed inside every horizon loop.
    paw_per_horizon = 2*d*v + 4*d*r
    paw_layer = horizons*paw_per_horizon + 2*d + (4*d*r if horizons else 0)
    gate_layer = 4*d*g+2*g
    # One new closed chunk per token and one cache query. Actual count varies.
    ch_projection_layer = 8*d*rank_ch+8*d
    decode_baseline = layers*(linear+attention)+lm_head
    training_forward = layers*linear+lm_head
    training_paw_forward = layers*(fbs["k_max"]*paw_per_horizon+2*d+4*d*r)
    return {"convention": "2 FLOPs per multiply-add; analytical, not measured",
            "decode_context_keys": context, "decode_active_horizons": horizons,
            "backbone_linear_per_layer": linear, "backbone_attention_per_layer": attention,
            "lm_head_per_token": lm_head, "baseline_decode_per_token": decode_baseline,
            "paw_per_horizon_per_layer": paw_per_horizon,
            "paw_all_layers_per_token": layers*paw_layer,
            "sg_all_layers_per_token": layers*gate_layer,
            "ch_projection_per_executed_layer_one_new_chunk": ch_projection_layer,
            "ch_attention_per_cached_chunk": 4*rank_ch,
            "paw_to_baseline_decode_ratio": layers*paw_layer/decode_baseline,
            "minimum_with_all_backbone_layers_skipped": layers*(paw_layer+gate_layer)+lm_head,
            "training_tokens": tokens,
            "baseline_training_linear_flops_approx": 3*tokens*training_forward,
            "paw_training_linear_flops_approx": 3*tokens*training_paw_forward,
            "baseline_plus_paw_training_linear_flops_approx": 3*tokens*(training_forward+training_paw_forward),
            "exclusions": ["softmax/top-k/CE, norms, activations, memory traffic, Python overhead",
                           "training attention and CH; SG training matmuls",
                           "optimizer updates, rematerialization and communication"],
            "training_note": "Dense ST computes attention/FFN branches; skip fraction is not a training FLOPs discount."}


def build_report(c, fbs, horizons=8, context=512, tokens=12_000_000_000):
    base = qwen_parameter_counts(c)
    aux = auxiliary_parameter_counts(c["hidden_size"], c["num_hidden_layers"], fbs["k_max"],
                                     fbs["preview_rank"], fbs["chunk_rank"], fbs["gate_hidden"])
    d, v, layers, k = c["hidden_size"], c["vocab_size"], c["num_hidden_layers"], fbs["k_max"]
    return {"scope": "Public reconstruction; no historical weights or numerical results verified",
            "backbone": base, "reconstruction_auxiliary": aux,
            "total": base["total"]+aux["total"], "overhead_fraction": aux["total"]/base["total"],
            "hypothetical_dense_heads_weights_only": {
                "one_d_by_vocab_head": d*v, "shared_across_layers_horizon_specific": k*d*v,
                "independent_per_layer_and_horizon": layers*k*d*v,
                "paper_shape_V32000_k15": 32*15*4096*32000,
                "paper_shape_V32768_k15": 32*15*4096*32768},
            "dense_ch_with_biases_all_layers": layers*(4*d*d+8*d+4),
            "cost": cost_estimate(c, fbs, horizons, context, tokens)}


def verify_meta(c, fbs, report):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from .hf import QwenFBSModel
    with torch.device("meta"):
        model = QwenFBSModel(Qwen3ForCausalLM(Qwen3Config(**c)), **fbs)
    inventory = parameter_inventory(model)
    expected = {"backbone": report["backbone"]["total"], **report["reconstruction_auxiliary"]["all_layers"]}
    if inventory["by_component"] != expected or inventory["total"] != report["total"]:
        raise AssertionError({"actual": inventory["by_component"], "expected": expected})
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-config", default="configs/qwen3_4b_public.config.json")
    parser.add_argument("--config", default="configs/qwen3_reconstruction.example.json")
    parser.add_argument("--horizons", type=int, default=8, help="Illustrative realized decode span; not k_max")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--verify-meta", action="store_true")
    parser.add_argument("--tensor-output", help="Optional full shape/parameter inventory (requires --verify-meta)")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.tensor_output and not args.verify_meta:
        parser.error("--tensor-output requires --verify-meta")
    c = json.loads(Path(args.backbone_config).read_text())
    config = json.loads(Path(args.config).read_text())
    expected = config["pretrained"].get("expected_config", {})
    if any(c.get(key) != value for key, value in expected.items()):
        raise ValueError("Run configuration and backbone snapshot disagree")
    report = build_report(c, config["fbs"], args.horizons, args.context, config["token_budget"])
    report["source"] = config["pretrained"]
    if args.verify_meta:
        inventory = verify_meta(c, config["fbs"], report)
        report["verification"] = {"method": "Full-shape meta tensors; no weights downloaded or forward pass",
                                  "by_component": inventory["by_component"], "passed": True}
        if args.tensor_output:
            Path(args.tensor_output).write_text(json.dumps(inventory, indent=2)+"\n")
    text = json.dumps(report, indent=2)+"\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
