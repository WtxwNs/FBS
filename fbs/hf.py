"""Qwen3 weight-preserving FBS adapter, tested with transformers==4.51.3.

The paper's named backbone and its stated shape disagree. This adapter uses the
actual checkpoint configuration; it does not fabricate a 4096x32 'Qwen3-4B'.
A tied low-rank PAW is an explicit reconstruction choice, not a recovered
historical architecture. Report the measured parameter overhead.
"""
import re
import torch
from torch import nn
import torch.nn.functional as F

from .model import FBSModel, FBSBlock, PAW, ChunkHead, SkipGate


class QwenAttention(nn.Module):
    def __init__(self, source, rotary, config):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = source.q_proj, source.k_proj, source.v_proj, source.o_proj
        self.q_norm, self.k_norm = source.q_norm, source.k_norm
        self.rotary = rotary
        self.n_heads, self.n_kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.d_head = source.head_dim

    def project(self, x, positions):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        batch, seq, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(batch, seq, self.n_heads, self.d_head)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(batch, seq, self.n_kv_heads, self.d_head)).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.d_head).transpose(1, 2)
        cos, sin = self.rotary(x, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def attend(self, q, k, v, mask=None):
        repeats = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return self.o_proj(result.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1))

    def forward(self, x, active_mask=None):
        positions = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
        q, k, v = self.project(x, positions)
        visible = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).tril()[None, None]
        if active_mask is not None:
            visible = visible & active_mask[:, None, None, :]
        return self.attend(q, k, v, visible)


class QwenFBSModel(FBSModel):
    def __init__(self, backbone, k_max=15, preview_rank=32, chunk_rank=32, gate_hidden=32,
                 use_paw=True, use_ch=True, use_sg=True, source=None):
        nn.Module.__init__(self)
        cfg = backbone.config
        if cfg.model_type != "qwen3" or getattr(cfg, "use_sliding_window", False):
            raise ValueError("Only dense Qwen3 global-attention checkpoints are supported")
        if any(not isinstance(r, int) or isinstance(r, bool) or r < 1
               for r in (preview_rank, chunk_rank, gate_hidden)):
            raise ValueError("Qwen reconstruction requires positive integer auxiliary ranks/widths")
        self.vocab_size, self.d_model = cfg.vocab_size, cfg.hidden_size
        self.max_seq_len = cfg.max_position_embeddings
        self.token_emb, self.norm, self.lm_head = backbone.model.embed_tokens, backbone.model.norm, backbone.lm_head
        self.baseline_parameters = sum(p.numel() for p in backbone.parameters())
        self.layers = nn.ModuleList()
        for original in backbone.model.layers:
            # Avoid allocating a second full-size attention/FFN.
            layer = FBSBlock.__new__(FBSBlock)
            nn.Module.__init__(layer)
            layer.use_paw, layer.use_ch, layer.use_sg = use_paw, use_ch, use_sg
            layer.attn = QwenAttention(original.self_attn, backbone.model.rotary_emb, cfg)
            layer.ffn = original.mlp
            layer.norm1, layer.norm2 = original.input_layernorm, original.post_attention_layernorm
            layer.paw = PAW(cfg.hidden_size, cfg.vocab_size, k_max=k_max, rank=preview_rank)
            layer.ch = ChunkHead(cfg.hidden_size, d_chunk=chunk_rank)
            layer.sg = SkipGate(cfg.hidden_size, hidden=gate_hidden)
            layer.to(device=self.token_emb.weight.device, dtype=self.token_emb.weight.dtype)
            self.layers.append(layer)
        self.config = dict(kind="qwen3", backbone_config=cfg.to_dict(), k_max=k_max,
                           preview_rank=preview_rank, chunk_rank=chunk_rank, gate_hidden=gate_hidden,
                           use_paw=use_paw, use_ch=use_ch, use_sg=use_sg, source=source)

    def embed(self, input_ids, positions):
        return self.token_emb(input_ids)

    @classmethod
    def from_pretrained(cls, model_id, revision, expected_config=None, **kwargs):
        from transformers import AutoConfig, AutoModelForCausalLM
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("Specify an immutable 40-character checkpoint commit revision")
        cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        validate_backbone_config(cfg, expected_config)
        device = kwargs.pop("device", "cpu")
        dtype = kwargs.pop("dtype", torch.float32)
        backbone = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype,
                                                       trust_remote_code=False, attn_implementation="eager")
        return cls(backbone, source={"model_id": model_id, "revision": revision}, **kwargs).to(device)


def validate_backbone_config(config, expected=None):
    """Check the declared public architecture before downloading large weights."""
    actual = config.to_dict()
    mismatches = {key: {"expected": value, "actual": actual.get(key)}
                  for key, value in (expected or {}).items() if actual.get(key) != value}
    if mismatches:
        raise ValueError(f"Checkpoint architecture differs from the recorded configuration: {mismatches}")
