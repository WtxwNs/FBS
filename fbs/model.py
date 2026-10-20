"""
Implementation of the Fovea‑Block‑Skip (FBS) Transformer.

This module reconstructs core concepts from *“FBS: Modeling Native Parallel
Reading inside a Transformer”*. It is not recovered historical experiment
code; module widths, sharing and compression choices are audited in
ARCHITECTURE_AUDIT.md. Each FBS block augments a causal Transformer layer
with three modules (small parameter count does not imply small compute):

* **Parafovea‑Attention Window (PAW)** – predicts a variable lookahead
  window for each token, produces multi‑step next‑token distributions
  and compresses them into a preview vector that is added to the hidden
  state.
* **Chunk‑Head (CH)** – predicts BIOS‑style chunk boundaries and
  constructs chunk representations which are attended to by each token
  to capture phrase‑level semantics.
* **Skip‑Gate (SG)** – computes a skip probability based on the current
  hidden state and preview, allowing the model to bypass the costly
  attention/FFN computation for “easy” tokens.  A straight‑through
  estimator is used so that hard gates can be deployed at inference
  time.

The classes defined here are designed to be modular: PAW, CH and SG can
be used independently or combined within an `FBSBlock`.  A high‑level
`FBSModel` stacks multiple `FBSBlock`s together with an embedding
layer and an output projection head, implementing a causal language
model suitable for next‑token prediction.

While every effort has been made to follow the specification in the
original paper, this implementation intentionally simplifies some
aspects for clarity and ease of experimentation (e.g. fixed
compression via mean pooling in PAW, pseudo‑labels for CH).  It
nevertheless preserves the key algorithmic elements of FBS: dynamic
lookahead, chunk‑level integration and conditional layer skipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalSelfAttention(nn.Module):
    """Standard multi‑head causal self‑attention.

    This module implements scaled dot‑product attention with a causal mask
    so that each position can only attend to its own and previous
    positions.  It accepts a (batch, seq_len, d_model) tensor and
    returns an output tensor of the same shape.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Projection matrices
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, active_mask=None) -> torch.Tensor:
        batch, seq, dim = x.size()
        # Compute Q,K,V
        q = self.q_proj(x)  # (batch, seq, dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape to (batch, heads, seq, d_head)
        q = q.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)  # (batch, heads, seq, d_head)
        k = k.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        # Compute scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (batch, heads, seq, seq)
        # Causal mask: positions cannot attend to future tokens
        # Use an additive mask with -inf for invalid positions
        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        mask = mask[None, None].expand(batch, self.n_heads, -1, -1)
        if active_mask is not None:
            mask = mask | ~active_mask[:, None, None, :]
        visible = (~mask).any(dim=-1, keepdim=True)
        scores = scores.masked_fill(mask, float('-inf'))
        scores = torch.where(visible, scores, torch.zeros_like(scores))
        attn = F.softmax(scores, dim=-1) * ~mask
        # Weighted sum of values
        context = torch.matmul(attn, v)  # (batch, heads, seq, d_head)
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.o_proj(context)


class FeedForward(nn.Module):
    """Position‑wise feed‑forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PAW(nn.Module):
    """Parafovea‑Attention Window (PAW).

    Given token hidden states, PAW predicts a window size `k(i)` for each
    position `i`, produces multi‑step predictive distributions over the
    next tokens using a lightweight head, maps these distributions into
    preview embeddings via the embedding matrix, and compresses the
    resulting sequence of embeddings into a single preview vector.  The
    preview is added to the hidden state as an auxiliary channel.

    During training, the module can compute a multi‑step cross‑entropy
    loss given ground‑truth targets.  At inference the window is
    discretised and a hard cutoff is applied.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        k_max: int = 15,
        top_k: int = 5,
        gamma: float = 4.0,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        if k_max < 0:
            raise ValueError("k_max must be nonnegative")
        self.k_max = k_max
        self.rank = rank
        self.top_k = top_k
        self.gamma = gamma
        # Window predictor s_i -> in (−∞,+∞)
        self.window_pred = nn.Linear(d_model, 1)
        # Multi‑horizon predictive heads: one linear per horizon
        self.preview_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(k_max)
        ]) if rank is None else nn.ModuleList()
        if rank is not None:
            self.down = nn.Linear(d_model, rank, bias=False)
            self.up = nn.Linear(rank, d_model, bias=False)
            self.horizon = nn.Parameter(torch.zeros(k_max, rank))
        # A small layer to compress the concatenated preview embeddings
        # into a single vector (optional).  Here we use a simple MLP.
        self.compress = nn.Sequential(
            nn.Linear(d_model, d_model if rank is None else rank),
            nn.Tanh(),
            nn.Linear(d_model if rank is None else rank, d_model),
        )

    def forward(
        self,
        h: torch.Tensor,
        embedding_weight: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Hidden states `(batch, seq, d_model)`.
            embedding_weight: Embedding matrix `(vocab_size, d_model)` used
                to map token distributions into embeddings.  This should be
                tied to the model's input embedding.
            targets: Optional next-token indices `(batch, seq)`, already
                shifted one position relative to h, with -100 for padding.
                When provided, a
                multi‑horizon cross‑entropy loss is returned.

        Returns:
            z: Preview vectors `(batch, seq, d_model)`.
            loss: A scalar tensor containing the preview loss (0 if targets
                is None).
        """
        batch, seq, dim = h.size()
        device = h.device
        if self.k_max == 0:
            return torch.zeros_like(h), h.new_zeros(())
        # Predict continuous window length k_tilde in [0, k_max]
        s = self.window_pred(h).squeeze(-1)  # (batch, seq)
        k_tilde = self.k_max * torch.sigmoid(s)  # (batch, seq)
        # Prepare containers for preview embeddings and losses
        preview_embeds = h.new_zeros(batch, seq, self.k_max, dim)
        total_loss = h.new_zeros(1)
        # For each horizon r compute token distribution and preview embedding
        # We'll vectorise across batch and sequence for efficiency
        hard_k = torch.floor(k_tilde).long()
        # In eval, do not evaluate horizons outside every selected window.
        horizons = self.k_max if self.training else int(hard_k.max().item())
        for r in range(horizons):
            # Compute logits for r‑th next token; shape (batch, seq, vocab)
            logits = self.preview_heads[r](h) if self.rank is None else F.linear(
                self.up(self.down(h) + self.horizon[r]), embedding_weight)
            # Compute predictive probabilities
            probs = F.softmax(logits, dim=-1)
            # Optionally restrict to top‑k support to reduce overhead
            if self.top_k is not None and self.top_k < self.vocab_size:
                # Zero out probabilities except for the top‑k tokens
                topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
                topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                u = (F.embedding(topk_idx, embedding_weight) * topk_vals.unsqueeze(-1)).sum(-2)
            else:
                u = torch.matmul(probs, embedding_weight)
            preview_embeds[:, :, r, :] = u
            # If targets provided, compute cross‑entropy loss at i against target i+r
            if targets is not None:
                # targets already contains the next token: horizon r+1
                # therefore needs an additional shift of r, not r+1.
                target_shifted = torch.full_like(targets, fill_value=-100)  # ignore index
                if r == 0:
                    target_shifted = targets
                elif r < seq:
                    target_shifted[:, :-r] = targets[:, r:]
                # Cross‑entropy loss ignoring positions with ignore_index
                ce = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_shifted.view(-1),
                    ignore_index=-100,
                    reduction='none',
                ).view(batch, seq)
                # Soft window inclusion weight w_{i,r}
                # w_{i,r} = σ(γ (k_tilde[i] - (r+1) + 0.5))
                # Note: r is zero‑indexed here so horizon = r+1
                horizon = (r + 1)
                w = torch.sigmoid(self.gamma * (k_tilde - (horizon) + 0.5))  # (batch, seq)
                # C.1 weighted sum over horizons, averaged over valid input
                # positions. Do not divide by window weights: that cancels
                # the intended window-dependent objective.
                w = w * (target_shifted != -100)
                total_loss = total_loss + (w * ce).sum() / (targets != -100).sum().clamp_min(1)
        # Compress preview embeddings along the horizon dimension using weighted mean
        z_list = []
        for r in range(self.k_max):
            horizon = r + 1
            w = (torch.sigmoid(self.gamma * (k_tilde - horizon + 0.5)) if self.training
                 else (hard_k >= horizon).to(h.dtype)).unsqueeze(-1)  # (batch, seq, 1)
            z_list.append(preview_embeds[:, :, r, :] * w)
        z_sum = torch.stack(z_list, dim=2).sum(dim=2)  # (batch, seq, d_model)
        w_sum = torch.stack([
            (torch.sigmoid(self.gamma * (k_tilde - (r + 1) + 0.5)) if self.training
             else (hard_k >= r + 1).to(h.dtype))
            for r in range(self.k_max)
        ], dim=2).sum(dim=2).unsqueeze(-1)  # (batch, seq, 1)
        z = z_sum / (w_sum + 1e-6)
        # Optional compression via MLP to reduce overhead and introduce non‑linearity
        z = self.compress(z) * (w_sum > 0)
        return z, total_loss


class ChunkHead(nn.Module):
    """Chunk‑Head (CH).

    This module predicts BIOS‑style chunk boundaries and builds a
    parallel chunk cache to provide phrase‑level semantics.  Each token
    attends to the cached chunk representations via a single‑head
    attention mechanism.  During training a cross‑entropy loss can be
    computed against provided pseudo‑labels; otherwise the labels are
    inferred greedily from the model's own predictions.
    """

    def __init__(self, d_model: int, d_chunk: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_chunk = d_model if d_chunk is None else d_chunk
        # Classifier for BIOS labels (B, I, O, S)
        self.num_classes = 4
        self.label_head = nn.Linear(d_model, self.num_classes)
        # Projections for cross‑attention to chunk cache
        self.q_proj = nn.Linear(d_model, self.d_chunk)
        self.k_proj = nn.Linear(d_model, self.d_chunk)
        self.v_proj = nn.Linear(d_model, self.d_chunk)
        self.out_proj = nn.Linear(self.d_chunk, d_model)

    def forward(
        self,
        h: torch.Tensor,
        pseudo_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Hidden states `(batch, seq, d_model)`.
            pseudo_labels: Optional `(batch, seq)` tensor of integers in
                {0,1,2,3} representing the ground‑truth BIOS labels.  The
                ordering follows [B, I, O, S].  If provided, the module
                computes a cross‑entropy loss; otherwise it infers labels
                greedily from its own predictions.

        Returns:
            ch_out: Chunk‑enhanced representations `(batch, seq, d_model)`.
            loss: A scalar tensor containing the CH loss (0 if no
                pseudo_labels are provided).
        """
        batch, seq, dim = h.size()
        logits = self.label_head(h)
        loss = h.new_zeros(())
        if pseudo_labels is not None and (pseudo_labels != -100).any():
            loss = F.cross_entropy(logits.reshape(-1, 4), pseudo_labels.reshape(-1), ignore_index=-100)
        labels = logits.argmax(dim=-1)
        rows = []
        for b in range(batch):
            cache = {"pending": [], "chunks": []}
            rows.append(torch.stack([self.step(h[b, t], cache, int(labels[b, t])) for t in range(seq)]))
        return torch.stack(rows), loss

    def step(self, h, cache, label=None):
        """Consume one token; a B/I chunk closes when a non-I arrives.

        No end-of-sequence flush: that would make prefix outputs depend on
        whether more tokens will arrive. Only previously closed chunks are read.
        """
        if label is None:
            label = int(self.label_head(h).argmax().item())
        cache.setdefault("keys", [])
        cache.setdefault("values", [])
        def append_chunk(chunk):
            cache["chunks"].append(chunk)
            cache["keys"].append(self.k_proj(chunk))
            cache["values"].append(self.v_proj(chunk))
        if label != 1 and cache["pending"]:
            append_chunk(torch.stack(cache["pending"]).mean(0))
            cache["pending"] = []
        if cache["chunks"]:
            keys, values = torch.stack(cache["keys"]), torch.stack(cache["values"])
            weights = torch.softmax(self.q_proj(h) @ keys.T / self.d_chunk ** .5, dim=-1)
            result = self.out_proj(weights @ values)
        else:
            result = torch.zeros_like(h)
        if label == 0 or (label == 1 and cache["pending"]):
            cache["pending"].append(h)
        else:
            append_chunk(h)
        return result


class SkipGate(nn.Module):
    """Skip‑Gate (SG).

    Given the current token representation and its preview vector, SG
    computes a probability of skipping the expensive attention/FFN
    computation.  A straight‑through estimator is used so that the
    forward pass uses a discrete gate while the backward pass receives
    gradients through the underlying probability.
    """

    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        # Input size is concatenated hidden state and preview vector
        self.fc1 = nn.Linear(2 * d_model, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute skip gate decisions and probabilities.

        Args:
            h: Token hidden states `(batch, seq, d_model)`.
            z: Preview vectors from PAW `(batch, seq, d_model)`.
            threshold: Threshold used during inference to convert
                probabilities into hard gates.  During training a
                straight‑through estimator is used instead.

        Returns:
            g: Straight‑through gates `(batch, seq, 1)` where values in
                [0,1] are used for mixing during training and discrete
                {0,1} values are used in the forward pass.
            p: Skip probabilities `(batch, seq, 1)`.
        """
        # Concatenate hidden state and preview
        x = torch.cat([h, z], dim=-1)
        p = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))  # (batch, seq, 1)
        # Straight‑through estimator: sample hard gate from Bernoulli(p)
        # but use p in backward pass
        if threshold >= 1:
            hard = torch.zeros_like(p)
        elif self.training:
            hard = torch.bernoulli(p)
        else:
            hard = (p > threshold).to(p.dtype)
        g = hard if threshold >= 1 else hard + p - p.detach()
        return g, p


class FBSBlock(nn.Module):
    """Single FBS Transformer block.

    This block wraps a standard causal Transformer layer with PAW, CH
    and SG.  The forward pass returns both the transformed hidden
    states and the sum of auxiliary losses from PAW and CH.  Skip
    gating is implemented via a straight‑through mixture of the input
    hidden state and the output of the full computation path.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        vocab_size: int,
        k_max: int = 15,
        preview_rank: Optional[int] = None,
        use_paw: bool = True,
        use_ch: bool = True,
        use_sg: bool = True,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.use_paw, self.use_ch, self.use_sg = use_paw, use_ch, use_sg
        self.paw = PAW(d_model, vocab_size, k_max=k_max, rank=preview_rank)
        self.ch = ChunkHead(d_model)
        self.sg = SkipGate(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        embedding_weight: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        pseudo_labels: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Input hidden states `(batch, seq, d_model)`.
            embedding_weight: Embedding matrix used by PAW.
            targets: Optional ground truth token indices for PAW loss.
            pseudo_labels: Optional BIOS labels for CH loss.
            threshold: Threshold for converting skip probabilities into
                hard gates.  During training a straight‑through estimator
                is used.

        Returns:
            h_out: Output hidden states `(batch, seq, d_model)`.
            aux_loss: Sum of PAW and CH losses (0 if none provided).
        """
        # Layer normalisation on the input
        x = self.norm1(h)
        # Compute PAW preview and loss from current state
        z, paw_loss = self.paw(x, embedding_weight, targets) if self.use_paw else (torch.zeros_like(x), x.new_zeros(()))
        # Skip‑gate decisions from current state + preview
        if self.use_sg:
            g, probability = self.sg(x, z, threshold=threshold)
        else:
            g = probability = x.new_zeros(*x.shape[:2], 1)

        # Inference short-circuit: if every token is gated to skip,
        # bypass attention/CH/FFN and forward the previous hidden state.
        if targets is None and torch.all(g > 0.5):
            return h, paw_loss

        # Standard self‑attention
        sa_out = self.attn(x, active_mask=(g.detach().squeeze(-1) < .5))
        # Add residual
        h_sa = h + sa_out
        # Compute CH contribution and loss
        if self.use_ch:
            # Mirror deployed CH state: skipped tokens do not update its cache.
            ch_rows = []
            for b in range(h.shape[0]):
                cache = {"pending": [], "chunks": []}
                ch_rows.append(torch.stack([
                    self.ch.step(x[b, t], cache) if g[b, t].detach().item() < .5 else torch.zeros_like(x[b, t])
                    for t in range(h.shape[1])]))
            ch_out = torch.stack(ch_rows)
            ch_loss = x.new_zeros(())
            if pseudo_labels is not None and (pseudo_labels != -100).any():
                ch_loss = F.cross_entropy(self.ch.label_head(x).reshape(-1, 4), pseudo_labels.reshape(-1))
        else:
            ch_out, ch_loss = torch.zeros_like(x), x.new_zeros(())
        # Fuse: h + SA + PAW + CH
        fused = h_sa + z + ch_out
        fused_norm = self.norm2(fused)
        # Compute FFN
        ffn_out = self.ffn(fused_norm)
        # Apply gating: g=1 skip (copy input), g=0 compute (fused + ffn)
        h_out = g * h + (1 - g) * (fused + ffn_out)
        # Auxiliary losses
        valid = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device) if targets is None else targets != -100
        gate_loss = probability.squeeze(-1)[valid].mean() if valid.any() and self.use_sg else h.new_zeros(())
        alignment_mask = valid & (ch_out.detach().abs().sum(-1) > 0)
        align_loss = (1 - F.cosine_similarity(ch_out[alignment_mask], x[alignment_mask].detach(), dim=-1)).mean() if alignment_mask.any() else h.new_zeros(())
        self.last_losses = dict(preview=paw_loss, bios=ch_loss, align=align_loss, gate=gate_loss)
        aux_loss = paw_loss + ch_loss
        return h_out, aux_loss


class FBSModel(nn.Module):
    """Stacked FBS Transformer for causal language modelling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        k_max: int = 15,
        max_seq_len: int = 2048,
        preview_rank: Optional[int] = None,
        use_paw: bool = True,
        use_ch: bool = True,
        use_sg: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           d_ff=d_ff, k_max=k_max, max_seq_len=max_seq_len, preview_rank=preview_rank,
                           use_paw=use_paw, use_ch=use_ch, use_sg=use_sg)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            FBSBlock(d_model, n_heads, d_ff, vocab_size, k_max=k_max, preview_rank=preview_rank,
                     use_paw=use_paw, use_ch=use_ch, use_sg=use_sg)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Output head (tied weights)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights between input embedding and output projection
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        pseudo_labels: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Token indices `(batch, seq)`.
            targets: Optional targets for next‑token prediction and PAW
                loss.  If None, the model operates in inference mode and
                does not compute auxiliary losses.
            pseudo_labels: Optional BIOS labels for CH.
            threshold: Threshold for SG during inference.

        Returns:
            logits: Language model logits `(batch, seq, vocab_size)`.
            loss: Sum of language modelling loss and auxiliary losses
                (cross‑entropy + PAW + CH).  If `targets` is None, the
                loss is zero.
        """
        batch, seq = input_ids.size()
        if seq == 0 or seq > self.max_seq_len:
            raise ValueError("Sequence must be nonempty and fit max_seq_len")
        if not self.training and targets is None:
            from .inference import prefill
            outputs = [prefill(self, row.unsqueeze(0), threshold=threshold)[0] for row in input_ids]
            return torch.cat(outputs, dim=0), self.token_emb.weight.new_zeros(())
        device = input_ids.device
        # Compute token and position embeddings
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        h = self.embed(input_ids, positions)
        aux_loss_total = h.new_zeros(1)
        # Pass through stacked FBS blocks
        for layer in self.layers:
            h, aux_loss = layer(
                h,
                self.token_emb.weight,
                targets=targets,
                pseudo_labels=pseudo_labels,
                threshold=threshold,
            )
            weights = getattr(self, "loss_weights", None)
            if weights is not None:
                aux_loss = sum(weights.get(name, 0.) * value for name, value in layer.last_losses.items()) / len(self.layers)
            aux_loss_total = aux_loss_total + aux_loss
        # Final layer norm
        h = self.norm(h)
        # Output logits
        logits = self.lm_head(h)  # (batch, seq, vocab)
        # Compute language modelling loss
        lm_loss = h.new_zeros(1)
        if targets is not None and (targets != -100).any():
            lm_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                reduction='mean',
            )
        # Total loss
        total_loss = lm_loss + aux_loss_total
        return logits, total_loss

    def embed(self, input_ids, positions):
        return self.token_emb(input_ids) + self.pos_emb(positions)

    def parameter_report(self):
        # Deduplicate by Parameter identity, including tied embedding/LM weights.
        from .audit import parameter_inventory
        return parameter_inventory(self)
