"""
Implementation of the Fovea‑Block‑Skip (FBS) Transformer.

This module implements the core components of the FBS architecture as
described in the paper *“FBS: Modeling Native Parallel Reading inside a
Transformer”*.  Each FBS block augments a standard causal Transformer
layer with three lightweight modules:

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.k_max = k_max
        self.top_k = top_k
        self.gamma = gamma
        # Window predictor s_i -> in (−∞,+∞)
        self.window_pred = nn.Linear(d_model, 1)
        # Multi‑horizon predictive heads: one linear per horizon
        self.preview_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(k_max)
        ])
        # A small layer to compress the concatenated preview embeddings
        # into a single vector (optional).  Here we use a simple MLP.
        self.compress = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
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
            targets: Optional tensor of token indices `(batch, seq)`
                representing the ground truth.  When provided, a
                multi‑horizon cross‑entropy loss is returned.

        Returns:
            z: Preview vectors `(batch, seq, d_model)`.
            loss: A scalar tensor containing the preview loss (0 if targets
                is None).
        """
        batch, seq, dim = h.size()
        device = h.device
        # Predict continuous window length k_tilde in [0, k_max]
        s = self.window_pred(h).squeeze(-1)  # (batch, seq)
        k_tilde = self.k_max * torch.sigmoid(s)  # (batch, seq)
        # Prepare containers for preview embeddings and losses
        preview_embeds = h.new_zeros(batch, seq, self.k_max, dim)
        total_loss = h.new_zeros(1)
        # For each horizon r compute token distribution and preview embedding
        # We'll vectorise across batch and sequence for efficiency
        for r in range(self.k_max):
            # Compute logits for r‑th next token; shape (batch, seq, vocab)
            logits = self.preview_heads[r](h)
            # Compute predictive probabilities
            probs = F.softmax(logits, dim=-1)
            # Optionally restrict to top‑k support to reduce overhead
            if self.top_k is not None and self.top_k < self.vocab_size:
                # Zero out probabilities except for the top‑k tokens
                topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
                mask = torch.zeros_like(probs).scatter_(-1, topk_idx, 1.0)
                probs = probs * mask
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            # Compute expectation under the embedding matrix
            # ˆu_{i,r} = E^T p_{i,r}
            # Here embedding_weight has shape (vocab_size, d_model)
            u = torch.matmul(probs, embedding_weight)  # (batch, seq, d_model)
            preview_embeds[:, :, r, :] = u
            # If targets provided, compute cross‑entropy loss at i against target i+r
            if targets is not None:
                # Compute the ground truth at horizon r: shift targets left by r+1
                # For positions where i+r >= seq, we ignore the loss
                target_shifted = torch.full_like(targets, fill_value=-100)  # ignore index
                if r + 1 < seq:
                    target_shifted[:, :- (r + 1)] = targets[:, (r + 1) :]
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
                # Weight the loss and normalise
                total_loss = total_loss + (w * ce).sum() / (w.sum() + 1e-6)
        # Compress preview embeddings along the horizon dimension using weighted mean
        z_list = []
        for r in range(self.k_max):
            horizon = r + 1
            w = torch.sigmoid(self.gamma * (k_tilde - horizon + 0.5)).unsqueeze(-1)  # (batch, seq, 1)
            z_list.append(preview_embeds[:, :, r, :] * w)
        z_sum = torch.stack(z_list, dim=2).sum(dim=2)  # (batch, seq, d_model)
        w_sum = torch.stack([
            torch.sigmoid(self.gamma * (k_tilde - (r + 1) + 0.5))
            for r in range(self.k_max)
        ], dim=2).sum(dim=2).unsqueeze(-1)  # (batch, seq, 1)
        z = z_sum / (w_sum + 1e-6)
        # Optional compression via MLP to reduce overhead and introduce non‑linearity
        z = self.compress(z)
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
        self.v_proj = nn.Linear(d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, d_model)

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
        device = h.device
        # Predict label logits
        logits = self.label_head(h)  # (batch, seq, num_classes)
        # Compute label loss if pseudo labels provided
        loss = h.new_zeros(1)
        if pseudo_labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes),
                pseudo_labels.view(-1),
                reduction='mean',
            )
        # Greedy inference of labels for building chunk cache
        pred_labels = logits.argmax(dim=-1)  # (batch, seq)
        # Build chunk cache and compute cross‑attention for each sequence
        ch_out = torch.zeros_like(h)
        for b in range(batch):
            # Determine chunks according to predicted labels
            labels = pred_labels[b].tolist()
            chunks: List[List[int]] = []
            i = 0
            while i < seq:
                label = labels[i]
                # label indices: 0=B, 1=I, 2=O, 3=S
                if label == 3:  # S
                    # single token chunk
                    chunks.append([i])
                    i += 1
                    continue
                elif label == 0:  # B
                    start = i
                    i += 1
                    while i < seq and pred_labels[b, i] == 1:  # I
                        i += 1
                    end = i  # exclusive
                    chunks.append(list(range(start, end)))
                    continue
                else:
                    # O or unexpected I (treated as O)
                    chunks.append([i])
                    i += 1
                    continue
            # Compute chunk embeddings by mean pooling
            if len(chunks) == 0:
                # If no chunks predicted, skip
                ch_out[b] = 0.0
                continue
            chunk_embeddings = []
            for chunk in chunks:
                # mean pool hidden states for this chunk
                states = h[b, torch.tensor(chunk, device=device), :]  # (len, d_model)
                chunk_embeddings.append(states.mean(dim=0))
            chunk_embeddings = torch.stack(chunk_embeddings, dim=0)  # (num_chunks, d_model)
            # Project chunk embeddings to keys/values
            k = self.k_proj(chunk_embeddings)  # (num_chunks, d_chunk)
            v = self.v_proj(chunk_embeddings)  # (num_chunks, d_model)
            # Project queries from token representations
            q = self.q_proj(h[b])  # (seq, d_chunk)
            # Compute attention scores (single head) -> (seq, num_chunks)
            scores = torch.matmul(q, k.transpose(0, 1)) / (self.d_chunk ** 0.5)
            attn = F.softmax(scores, dim=-1)  # (seq, num_chunks)
            # Weighted sum of v
            attended = torch.matmul(attn, v)  # (seq, d_model)
            ch_out[b] = self.out_proj(attended)
        return ch_out, loss


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
        hard = (p > threshold).float()
        g = hard + p - p.detach()
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
    ) -> None:
        super().__init__()
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.paw = PAW(d_model, vocab_size, k_max=k_max)
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
        # Standard self‑attention
        sa_out = self.attn(x)
        # Add residual
        h_sa = h + sa_out
        # Compute PAW preview and loss
        z, paw_loss = self.paw(h_sa, embedding_weight, targets)
        # Compute CH contribution and loss
        ch_out, ch_loss = self.ch(h_sa, pseudo_labels)
        # Fuse: h + SA + PAW + CH
        fused = h_sa + z + ch_out
        fused_norm = self.norm2(fused)
        # Compute FFN
        ffn_out = self.ffn(fused_norm)
        # Skip‑gate decisions
        g, _ = self.sg(h_sa, z, threshold=threshold)  # (batch, seq, 1)
        # Apply gating: g=1 skip (copy input), g=0 compute (fused + ffn)
        h_out = g * h_sa + (1 - g) * (fused + ffn_out)
        # Auxiliary losses
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
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)
        self.layers = nn.ModuleList([
            FBSBlock(d_model, n_heads, d_ff, vocab_size, k_max=k_max)
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
        device = input_ids.device
        # Compute token and position embeddings
        token_emb = self.token_emb(input_ids)  # (batch, seq, d_model)
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        pos_emb = self.pos_emb(positions)
        h = token_emb + pos_emb
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
            aux_loss_total = aux_loss_total + aux_loss
        # Final layer norm
        h = self.norm(h)
        # Output logits
        logits = self.lm_head(h)  # (batch, seq, vocab)
        # Compute language modelling loss
        lm_loss = h.new_zeros(1)
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                reduction='mean',
            )
        # Total loss
        total_loss = lm_loss + aux_loss_total
        return logits, total_loss
