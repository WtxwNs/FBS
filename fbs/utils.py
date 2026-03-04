"""
Utility functions for data processing and batching.

The helpers in this module simplify the preparation of textual data
for language modelling: building a vocabulary, encoding text into
integer sequences, constructing mini‑batches with padding, and
generating pseudo BIOS labels for the chunk head.  These utilities
are deliberately simple and suitable for toy experiments.  For
large‑scale applications users may wish to replace them with a
sophisticated tokenizer and dataloader.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Iterable

import torch


def build_vocab(tokens: Iterable[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build a vocabulary mapping from tokens to indices.

    The vocabulary contains special tokens `<pad>` and `<unk>` at indices
    0 and 1 respectively.  Tokens occurring fewer than `min_freq` times
    are mapped to `<unk>`.

    Returns both the token‑to‑id and id‑to‑token dictionaries.
    """
    from collections import Counter

    counter = Counter(tokens)
    # Reserve 0 for <pad> and 1 for <unk>
    itos = ['<pad>', '<unk>']
    for token, freq in counter.items():
        if freq >= min_freq and token not in itos:
            itos.append(token)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, {i: tok for tok, i in stoi.items()}


def encode_text(tokens: List[str], stoi: Dict[str, int]) -> List[int]:
    """Convert a sequence of tokens into a list of vocabulary indices."""
    unk = stoi.get('<unk>', 1)
    return [stoi.get(tok, unk) for tok in tokens]


def load_corpus(path: str) -> List[str]:
    """Load a plain text file and return a list of whitespace‑separated tokens."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Simple whitespace tokenizer
    tokens = text.strip().split()
    return tokens


def batchify(
    sequences: List[List[int]],
    batch_size: int,
    seq_len: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Group sequences into batches of fixed length.

    Each batch contains `batch_size` sub‑sequences of length `seq_len`.
    Sequences shorter than the desired length are padded with `<pad>` (id=0).
    The function also returns a shifted copy of the input for next‑token
    prediction.

    Returns a list of `(input_ids, targets)` pairs, both of shape
    `(batch_size, seq_len)`.
    """
    # Flatten the list of sequences into a single long sequence
    flat = [tok for seq in sequences for tok in seq]
    num_tokens = len(flat)
    # Truncate so that it divides evenly into batch_size * seq_len
    total = (num_tokens // (batch_size * seq_len)) * (batch_size * seq_len)
    flat = flat[:total]
    # Reshape into (batch_size, -1)
    data = torch.tensor(flat, dtype=torch.long).view(batch_size, -1)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    # Generate sub‑sequences
    for i in range(0, data.size(1) - seq_len, seq_len):
        inp = data[:, i : i + seq_len].clone()
        tgt = data[:, i + 1 : i + seq_len + 1].clone()
        batches.append((inp, tgt))
    return batches


def generate_pseudo_labels(batch_seq: torch.Tensor) -> torch.Tensor:
    """Generate dummy BIOS labels for a batch of sequences.

    In the absence of a supervised chunking dataset, this function
    produces pseudo‑labels that mark every token as outside (`O`).
    The label mapping is: 0=B, 1=I, 2=O, 3=S.

    Returns a tensor of shape `(batch, seq)`.
    """
    batch, seq = batch_seq.size()
    # All tokens are labelled as 'O' (index 2)
    labels = torch.full((batch, seq), 2, dtype=torch.long, device=batch_seq.device)
    return labels
