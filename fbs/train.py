"""
Training script for the FBS Transformer on a toy corpus.

This module provides a simple command‑line interface for training the
`FBSModel` defined in `fbs/model.py`.  It supports specifying
hyper‑parameters such as the number of layers, model dimension,
lookahead window size and learning rate.  The script reads a text
corpus, constructs a vocabulary, tokenises and batches the data, and
optimises the model using Adam.  Auxiliary losses from the PAW and
CH modules are included automatically when targets and pseudo labels
are provided.

Example usage:

```bash
python -m fbs.train \
    --data_path data/sample.txt \
    --epochs 5 \
    --batch_size 4 \
    --seq_len 32 \
    --d_model 64 \
    --n_layers 2 \
    --n_heads 4 \
    --k_max 8 \
    --learning_rate 1e-3
```

The trained model will be saved as `fbs_model.pt` in the current
working directory.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .model import FBSModel
from .utils import build_vocab, encode_text, load_corpus, batchify, generate_pseudo_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an FBS Transformer on a toy corpus")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training text file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of FBS layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed‑forward inner dimension")
    parser.add_argument("--k_max", type=int, default=8, help="Maximum lookahead window for PAW")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--min_freq", type=int, default=1, help="Minimum frequency for vocabulary inclusion")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    # Load and tokenise corpus
    tokens = load_corpus(args.data_path)
    print(f"Loaded {len(tokens)} tokens")
    # Build vocabulary
    stoi, itos = build_vocab(tokens, min_freq=args.min_freq)
    print(f"Vocabulary size: {len(stoi)}")
    # Encode corpus
    encoded = encode_text(tokens, stoi)
    # Group into sequences equal to seq_len * batch_size
    # For simplicity, create individual sequences of length args.seq_len + 1
    sequences: List[List[int]] = []
    for i in range(0, len(encoded) - args.seq_len - 1, args.seq_len + 1):
        sequences.append(encoded[i : i + args.seq_len + 1])
    # Batchify sequences
    batches = batchify(sequences, args.batch_size, args.seq_len)
    print(f"Number of batches: {len(batches)}")
    # Instantiate model
    model = FBSModel(
        vocab_size=len(stoi),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        k_max=args.k_max,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        with tqdm(batches, desc=f"Epoch {epoch}/{args.epochs}") as pbar:
            for inp, tgt in pbar:
                inp = inp.to(device)
                tgt = tgt.to(device)
                # Generate dummy BIOS labels (all O)
                pseudo = generate_pseudo_labels(inp).to(device)
                optimizer.zero_grad()
                logits, loss = model(inp, targets=tgt, pseudo_labels=pseudo)
                loss.backward()
                optimizer.step()
                # Accumulate loss per token for reporting
                batch_tokens = inp.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                avg_loss = total_loss / total_tokens
                perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "ppl": f"{perplexity:.2f}"})
        # Save checkpoint after each epoch
        ckpt_path = f"fbs_model_epoch{epoch}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
