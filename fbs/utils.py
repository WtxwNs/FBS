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
    Input padding uses `<pad>` (id=0); target padding uses -100 so that
    cross-entropy ignores it. Every adjacent token pair in the flattened
    corpus is retained, including a final partial batch.

    Returns a list of `(input_ids, targets)` pairs, both of shape
    `(batch_size, seq_len)`.
    """
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    # Flatten the list of sequences into a single long sequence
    flat = [tok for seq in sequences for tok in seq]
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    starts = range(0, len(flat) - 1, seq_len)
    for offset in range(0, len(starts), batch_size):
        inp = torch.zeros((batch_size, seq_len), dtype=torch.long)
        tgt = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        for row, start in enumerate(starts[offset : offset + batch_size]):
            window = flat[start : start + seq_len + 1]
            length = len(window) - 1
            inp[row, :length] = torch.tensor(window[:-1], dtype=torch.long)
            tgt[row, :length] = torch.tensor(window[1:], dtype=torch.long)
        batches.append((inp, tgt))
    return batches


def generate_pseudo_labels(batch_seq: torch.Tensor) -> torch.Tensor:
    """Generate simple BIOS pseudo labels from token boundaries.

    For this toy word-level tokenizer, each non-padding token is treated
    as a standalone chunk (`S`) because word boundaries are explicit in
    the whitespace tokenisation. Padding tokens remain `O`.

    The label mapping is: 0=B, 1=I, 2=O, 3=S.

    Returns a tensor of shape `(batch, seq)`.
    """
    # Start with all tokens as outside (`O`).
    labels = torch.full_like(batch_seq, 2, dtype=torch.long)
    # Mark non-padding tokens (id != 0) as singleton chunks (`S`).
    labels[batch_seq != 0] = 3
    return labels
