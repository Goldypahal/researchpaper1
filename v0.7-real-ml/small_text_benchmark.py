"""
Phase 19: Real ML Cross-Domain Benchmark (Small Language Model).
Evaluates whether architecture search insights transfer beyond synthetic grammars
to real natural language sequence modeling.

Task:
  Character/Byte-level next-token prediction on open literary text corpus.
  Vocabulary: ASCII characters (V = 128)
  Sequence length: 64 tokens

Splits:
  - D_train:      Training corpus slice (e.g. standard prose/drama)
  - D_val:        In-distribution holdout validation split
  - D_test:       In-distribution holdout test split
  - D_ood:        Cross-genre out-of-distribution text (e.g. scientific / technical prose)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import math
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List

REAL_TEXT_ID = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action.
"""

REAL_TEXT_OOD = """
The algorithm optimizes hyperparameter configurations using sequential
Bayesian optimization over surrogate Gaussian processes. The covariance
matrix captures spatial correlations across architectural dimensions,
including attention head count, feed-forward dimension, and normalization
layers. Empirical convergence rates depend critically on gradient variance,
matrix condition numbers, and numerical stability in low-precision arithmetic.
Theoretical lower bounds dictate that stochastic optimization in non-convex
landscapes scales inversely with spectral radius and curvature conditioning.
"""


def encode_text_to_dataset(text: str, seq_len: int = 64) -> torch.Tensor:
    ascii_vals = [min(127, max(0, ord(c))) for c in text]
    chunks = []
    for i in range(0, len(ascii_vals) - seq_len, seq_len // 2):
        chunk = ascii_vals[i:i + seq_len]
        if len(chunk) == seq_len:
            chunks.append(chunk)
    if not chunks:
        # Replicate if text is short
        rep = (ascii_vals * (seq_len // len(ascii_vals) + 2))[:seq_len * 10]
        for i in range(0, len(rep) - seq_len, seq_len // 2):
            chunks.append(rep[i:i + seq_len])
    return torch.tensor(chunks, dtype=torch.long)


def get_real_ml_splits(seq_len: int = 64) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    id_data = encode_text_to_dataset(REAL_TEXT_ID, seq_len=seq_len)
    ood_data = encode_text_to_dataset(REAL_TEXT_OOD, seq_len=seq_len)

    n_total = len(id_data)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_data = id_data[:n_train]
    val_data = id_data[n_train:n_train + n_val]
    test_data = id_data[n_train + n_val:]

    train_masks = torch.ones_like(train_data, dtype=torch.bool)
    val_masks = torch.ones_like(val_data, dtype=torch.bool)
    test_masks = torch.ones_like(test_data, dtype=torch.bool)
    ood_masks = torch.ones_like(ood_data, dtype=torch.bool)

    return {
        "train": (train_data, train_masks),
        "val": (val_data, val_masks),
        "test": (test_data, test_masks),
        "ood": (ood_data, ood_masks)
    }


if __name__ == "__main__":
    splits = get_real_ml_splits()
    print("Real ML splits created successfully:")
    for k, (d, m) in splits.items():
        print(f"  Split '{k}': shape {d.shape}")
