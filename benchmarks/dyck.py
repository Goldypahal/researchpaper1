"""
Dyck-k Formal Grammar Benchmark with Stack Memory.
Belongs to Context-Free Languages (Chomsky Hierarchy Level 2).

Alphabet:
  k distinct bracket pairs:
    0: '(' = 10, ')' = 11
    1: '[' = 12, ']' = 13
    2: '{' = 14, '}' = 15
    3: '<' = 16, '>' = 17
  Distractor tokens: uniform random in [20, vocab_size - 1]

Splits:
  - D_train:      nesting depth d in [min_depth_id, max_depth_id]  (e.g., [1, 6])
  - D_val:        nesting depth d in [min_depth_id, max_depth_id], disjoint seed
  - D_test:       nesting depth d in [min_depth_id, max_depth_id], disjoint seed & verified disjoint samples
  - D_ood:        nesting depth d in [min_depth_ood, max_depth_ood] (e.g., [7, 12])
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from typing import Tuple, Dict, List, Optional, Set

OPEN_BRACKETS = {0: 10, 1: 12, 2: 14, 3: 16}
CLOSE_BRACKETS = {0: 11, 1: 13, 2: 15, 3: 17}
BRACKET_MATCH = {11: 10, 13: 12, 15: 14, 17: 16}
OPEN_TO_CLOSE = {10: 11, 12: 13, 14: 15, 16: 17}


def is_open_bracket(token: int) -> bool:
    return token in OPEN_TO_CLOSE


def is_close_bracket(token: int) -> bool:
    return token in BRACKET_MATCH


def verify_sequence_validity(seq: List[int], mask: Optional[List[bool]] = None) -> Tuple[bool, int, str]:
    """
    Verifies that seq obeys Dyck-k stack discipline.
    Returns:
      (is_valid, max_depth, error_msg)
    """
    stack = []
    max_depth = 0
    for idx, tok in enumerate(seq):
        if is_open_bracket(tok):
            stack.append(tok)
            if len(stack) > max_depth:
                max_depth = len(stack)
            if mask is not None and mask[idx]:
                return False, max_depth, f"Position {idx}: open bracket marked as structural close target"
        elif is_close_bracket(tok):
            if not stack:
                return False, max_depth, f"Position {idx}: closing bracket {tok} on empty stack"
            expected_open = BRACKET_MATCH[tok]
            actual_open = stack.pop()
            if actual_open != expected_open:
                return False, max_depth, f"Position {idx}: mismatched closing bracket {tok} for open bracket {actual_open}"
            if mask is not None and not mask[idx]:
                return False, max_depth, f"Position {idx}: closing bracket {tok} not flagged in structural mask"
        else:
            # Distractor token
            if mask is not None and mask[idx]:
                return False, max_depth, f"Position {idx}: distractor token {tok} marked in structural mask"
                
    if stack:
        return False, max_depth, f"Unclosed brackets remaining at end of sequence: {len(stack)}"
    return True, max_depth, ""


def generate_dyck_k_samples(
    num_samples: int = 1000,
    seq_len: int = 128,
    vocab_size: int = 1024,
    k: int = 4,
    min_depth: int = 1,
    max_depth: int = 6,
    p_bracket: float = 0.65,
    seed: int = 42,
    existing_hashes: Optional[Set[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor, Set[int]]:
    """
    Generates Dyck-k sequences with verified grammatical validity and depth bounds.
    Deduplicates against existing_hashes to guarantee zero dataset leakage.
    """
    gen = torch.Generator().manual_seed(seed)
    data: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    hashes = set(existing_hashes) if existing_hashes is not None else set()
    
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        stack = []
        seq = []
        mask = []
        
        for i in range(seq_len):
            remaining = seq_len - 1 - i
            
            # Must close stack if remaining steps are just enough to close open brackets
            if remaining <= len(stack) and len(stack) > 0:
                b = stack.pop()
                token = CLOSE_BRACKETS[b]
                is_structural = True
            elif len(stack) == 0:
                if remaining > 0 and torch.rand(1, generator=gen).item() < p_bracket:
                    b = torch.randint(0, k, (1,), generator=gen).item()
                    stack.append(b)
                    token = OPEN_BRACKETS[b]
                    is_structural = False
                else:
                    token = torch.randint(20, vocab_size, (1,), generator=gen).item()
                    is_structural = False
            else:
                if len(stack) < max_depth and remaining > len(stack):
                    r = torch.rand(1, generator=gen).item()
                    if r < (p_bracket / 2.0):
                        b = torch.randint(0, k, (1,), generator=gen).item()
                        stack.append(b)
                        token = OPEN_BRACKETS[b]
                        is_structural = False
                    elif r < p_bracket:
                        b = stack.pop()
                        token = CLOSE_BRACKETS[b]
                        is_structural = True
                    else:
                        token = torch.randint(20, vocab_size, (1,), generator=gen).item()
                        is_structural = False
                else:
                    # At max depth, can only close or emit distractor
                    if torch.rand(1, generator=gen).item() < p_bracket:
                        b = stack.pop()
                        token = CLOSE_BRACKETS[b]
                        is_structural = True
                    else:
                        token = torch.randint(20, vocab_size, (1,), generator=gen).item()
                        is_structural = False
            seq.append(token)
            mask.append(is_structural)
            
        # Verify sequence validity
        is_valid, depth, _ = verify_sequence_validity(seq, mask)
        if not is_valid:
            continue
            
        # If min_depth > 1, ensure the sequence reached at least min_depth
        if depth < min_depth:
            continue
            
        seq_tuple = tuple(seq)
        seq_hash = hash(seq_tuple)
        if seq_hash in hashes:
            continue  # Avoid duplicate sequences
            
        hashes.add(seq_hash)
        data.append(torch.tensor(seq, dtype=torch.long))
        masks.append(torch.tensor(mask, dtype=torch.bool))

    if len(data) < num_samples:
        raise RuntimeError(f"Failed to generate {num_samples} unique Dyck sequences within {max_attempts} attempts.")
        
    return torch.stack(data), torch.stack(masks), hashes


def get_dyck_benchmark_splits(
    num_train: int = 1000,
    num_val: int = 200,
    num_test: int = 200,
    num_ood: int = 300,
    seq_len: int = 128,
    vocab_size: int = 1024,
    k: int = 4,
    min_depth_id: int = 1,
    max_depth_id: int = 6,
    min_depth_ood: int = 7,
    max_depth_ood: int = 12,
    seed: int = 42,
    p_bracket: float = 0.65,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Constructs formal mathematically disjoint splits:
      - D_train
      - D_val
      - D_test
      - D_ood
    Guarantees:
      - D_train intersect D_val = empty
      - D_train intersect D_test = empty
      - D_val intersect D_test = empty
      - D_ood strictly has depth in [7, 12]
    """
    all_hashes: Set[int] = set()
    
    # Train
    train_data, train_masks, all_hashes = generate_dyck_k_samples(
        num_samples=num_train, seq_len=seq_len, vocab_size=vocab_size, k=k,
        min_depth=min_depth_id, max_depth=max_depth_id, p_bracket=p_bracket,
        seed=seed, existing_hashes=all_hashes
    )
    
    # Val
    val_data, val_masks, all_hashes = generate_dyck_k_samples(
        num_samples=num_val, seq_len=seq_len, vocab_size=vocab_size, k=k,
        min_depth=min_depth_id, max_depth=max_depth_id, p_bracket=p_bracket,
        seed=seed + 10000, existing_hashes=all_hashes
    )
    
    # Test
    test_data, test_masks, all_hashes = generate_dyck_k_samples(
        num_samples=num_test, seq_len=seq_len, vocab_size=vocab_size, k=k,
        min_depth=min_depth_id, max_depth=max_depth_id, p_bracket=p_bracket,
        seed=seed + 20000, existing_hashes=all_hashes
    )
    
    # OOD
    ood_data, ood_masks, _ = generate_dyck_k_samples(
        num_samples=num_ood, seq_len=seq_len, vocab_size=vocab_size, k=k,
        min_depth=min_depth_ood, max_depth=max_depth_ood, p_bracket=p_bracket,
        seed=seed + 30000, existing_hashes=set()  # depth difference naturally partitions grammar
    )
    
    return {
        "train": (train_data, train_masks),
        "val": (val_data, val_masks),
        "test": (test_data, test_masks),
        "ood": (ood_data, ood_masks)
    }
