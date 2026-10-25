"""
benchmarks/abs_position.py

Absolute Position Benchmark (Wrong-Prior Task).
Designed specifically to test whether LLMs suffer from dogmatic semantic priors:
In this synthetic sequence task, target prediction strictly depends on the ABSOLUTE position
index in the sequence (e.g. target_token = (abs_pos * 7 + 13) % vocab_size).

Because Rotary Position Encodings (RoPE) encode only relative pairwise distances (q^T R_{i-j} k),
they struggle to isolate absolute slot coordinates without additional markers, whereas standard
Learned and Sinusoidal Absolute Position Encodings learn this mapping with ease.

If an LLM search agent blindly proposes RoPE on this task due to its general Transformer prior,
it empirically reveals inductive bias rigidity resisting experimental feedback.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from typing import Tuple, Dict, Set, Optional

def generate_abs_position_samples(
    num_samples: int,
    seq_len: int = 128,
    vocab_size: int = 1024,
    seed: int = 42,
    existing_hashes: Optional[Set[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor, Set[int]]:
    """
    Generates sequence tensors where 50% of tokens are random distractors,
    and 50% (at masked positions) are strictly deterministic functions of absolute position index.
    """
    g = torch.Generator().manual_seed(seed)
    hashes = set(existing_hashes) if existing_hashes is not None else set()

    # Generate base sequences
    seqs = torch.randint(10, vocab_size, (num_samples, seq_len), generator=g, dtype=torch.long)
    masks = torch.zeros((num_samples, seq_len), dtype=torch.bool)

    # Every even position is an absolute coordinate target
    for pos in range(1, seq_len, 2):
        # Target token is a deterministic mathematical hash of the absolute position
        target_token = ((pos * 37 + 101) % (vocab_size - 10)) + 10
        seqs[:, pos] = target_token
        masks[:, pos] = True

    # Hash sequences for duplicate tracking
    for i in range(num_samples):
        h = hash(tuple(seqs[i].tolist()[:16]))
        hashes.add(h)

    return seqs, masks, hashes


def get_abs_position_benchmark_splits(
    num_train: int = 2000,
    num_val: int = 500,
    num_test: int = 500,
    num_ood: int = 500,
    seq_len: int = 128,
    vocab_size: int = 1024,
    seed: int = 42,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Generates train, val, test, and ood splits for the absolute position task."""
    all_hashes: Set[int] = set()

    train_data, train_masks, all_hashes = generate_abs_position_samples(
        num_samples=num_train, seq_len=seq_len, vocab_size=vocab_size, seed=seed, existing_hashes=all_hashes
    )
    val_data, val_masks, all_hashes = generate_abs_position_samples(
        num_samples=num_val, seq_len=seq_len, vocab_size=vocab_size, seed=seed + 10000, existing_hashes=all_hashes
    )
    test_data, test_masks, all_hashes = generate_abs_position_samples(
        num_samples=num_test, seq_len=seq_len, vocab_size=vocab_size, seed=seed + 20000, existing_hashes=all_hashes
    )
    # OOD: longer sequence length if specified, or perturbed offset
    ood_seq_len = kwargs.get("ood_seq_len", seq_len)
    ood_data, ood_masks, _ = generate_abs_position_samples(
        num_samples=num_ood, seq_len=ood_seq_len, vocab_size=vocab_size, seed=seed + 30000, existing_hashes=set()
    )

    return {
        "train": (train_data, train_masks),
        "val": (val_data, val_masks),
        "test": (test_data, test_masks),
        "ood": (ood_data, ood_masks)
    }
