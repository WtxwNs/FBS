# FBS: Modeling Native Parallel Reading inside a Transformer

<p align="center">
  <a href="https://arxiv.org/abs/2601.21708">
    <img src="https://img.shields.io/badge/arXiv-2601.21708-b31b1b.svg" alt="arXiv"/>
  </a>
  <a href="https://github.com/WtxwNs/BACH/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/WtxwNs/BACH" alt="License"/>
  </a>
  <img src="https://img.shields.io/github/repo-size/WtxwNs/BACH" alt="Repo Size"/>
  <img src="https://img.shields.io/github/stars/WtxwNs/BACH?style=social" alt="Stars"/>
</p>

This repository contains a minimal yet complete implementation of the **Fovea‑Block‑Skip (FBS) Transformer** as described in the paper *“FBS: Modeling Native Parallel Reading inside a Transformer”*.  The goal of this package is to faithfully reproduce the novel architectural components—Parafovea‑Attention Window (PAW), Chunk‑Head (CH) and Skip‑Gate (SG)—within a conventional causal Transformer and provide a runnable training script on a toy corpus.

The implementation focuses on clarity and modularity rather than chasing state‑of‑the‑art performance.  It is intended to serve as a reference implementation for researchers and engineers wishing to experiment with FBS‑style models.  All code is fully contained in this package and does not require any external proprietary dependencies.

## Contents

* `fbs/model.py` – Core PyTorch modules implementing the FBS block: a standard causal self‑attention + feed‑forward network augmented with PAW, CH and SG.
* `fbs/train.py` – A simple training loop that trains the FBS model on a toy language modelling task using a small sample corpus.  The script tokenises text at the word level and optimises the model using next‑token prediction.
* `fbs/utils.py` – Helper functions for building the vocabulary, batching sequences and computing losses.
* `data/sample.txt` – A tiny English/Chinese mixed dataset used by default in `train.py`.  Replace this file with your own corpus to train on larger datasets.
* `requirements.txt` – List of Python dependencies needed to run the code.

## Installation

1. Ensure you have Python 3.8 or later installed.  It is recommended to create a fresh virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running the toy training script

The following command will train a small FBS model for a few epochs on the included sample corpus.  The default hyper‑parameters are intentionally small so that training finishes quickly on a CPU.

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

During training the script reports the cross‑entropy loss and perplexity.  Upon completion the model checkpoint is saved to disk (`./fbs_model.pt`).  Feel free to adjust hyper‑parameters, increase the model size, or swap in your own dataset to explore the effects of PAW/CH/SG.

## Notes on the implementation

This codebase follows the algorithmic description in the paper as closely as possible while keeping the implementation concise and understandable:

1. **Parafovea‑Attention Window (PAW)** – Given the hidden state at each position, PAW predicts a variable lookahead window length `k(i)` (bounded by `k_max`) using a small linear head.  For each horizon `r∈{1,…,k_max}` the module produces a distribution over the `r`‑th next token, maps the distribution into a preview embedding by taking its expectation under the embedding matrix, and then compresses these embeddings into a single preview vector `z(i)` via weighted mean pooling.  The window weights follow the soft assignment in Appendix C.2 of the paper.  The preview vector is added to the token state before the feed‑forward network.

2. **Chunk‑Head (CH)** – A lightweight classifier predicts BIOS‑style chunk labels (`B`, `I`, `O`, `S`) for each token.  Based on the predicted labels, the tokens in the current chunk are pooled to produce a chunk representation.  Each token attends over the cached chunk representations via a single‑head attention to incorporate phrase‑level semantics.  The weak‑supervision pipeline described in Appendix D is beyond the scope of this toy implementation; instead we train the chunk classifier jointly with the language model using pseudo‑labels inferred from token boundaries.

3. **Skip‑Gate (SG)** – A small multi‑layer perceptron computes a skip probability from the current hidden state and PAW preview.  At inference time the gate can short‑circuit the expensive attention and feed‑forward computations, forwarding the previous hidden state instead.  In this demonstration the gate is trained using a straight‑through estimator, as described in Appendix E.2.

The supplied code is deliberately self‑contained and does not depend on any external large language model checkpoints.  To reproduce the full experiments reported in the FBS paper one would need to perform large‑scale continual pre‑training on billions of tokens and tune numerous hyper‑parameters; such a pipeline is far beyond the scope of a compact example.  Nevertheless, this repository provides a concrete and extensible starting point for researchers who wish to explore FBS on smaller problems or integrate its components into existing architectures.
