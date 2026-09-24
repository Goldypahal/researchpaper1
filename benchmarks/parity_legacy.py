"""
Legacy Parity Benchmark (for backwards compatibility with v0.1).
Synthesizes binary/modulo parity dependencies across the sequence.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from typing import Tuple, Dict, List, Optional, Set


def generate_parity_legacy_dataset(
    num_samples: int = 1200,
    seq_len: int = 128,
    vocab_size: int = 1024,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    data = []
    for _ in range(num_samples):
        seq = torch.randint(10, vocab_size, (seq_len,), dtype=torch.long)
        parity = 0
        for i in range(1, seq_len):
            if seq[i - 1] % 2 == 0:
                parity = (parity + 1) % 4
            if parity == 2:
                seq[i] = (seq[i] + 17) % (vocab_size - 10) + 10
        data.append(seq)
    masks = [torch.ones(seq_len, dtype=torch.bool) for _ in range(num_samples)]
    return torch.stack(data), torch.stack(masks)


def get_parity_benchmark_splits(
    num_train: int = 1000,
    num_val: int = 200,
    num_test: int = 200,
    num_ood: int = 300,
    seq_len: int = 128,
    vocab_size: int = 1024,
    seed: int = 42
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    data, masks = generate_parity_legacy_dataset(
        num_samples=num_train + num_val + num_test + num_ood,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed
    )
    
    i1 = num_train
    i2 = i1 + num_val
    i3 = i2 + num_test
    i4 = i3 + num_ood
    
    return {
        "train": (data[:i1], masks[:i1]),
        "val": (data[i1:i2], masks[i1:i2]),
        "test": (data[i2:i3], masks[i2:i3]),
        "ood": (data[i3:i4], masks[i3:i4])
    }
