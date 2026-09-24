"""
Hidden Finite State Machine (Hidden FSM) Benchmark.
Belongs to Regular Languages (Chomsky Hierarchy Level 3).

Formal Definition:
  M = (S, Sigma, T, E, s_0)
  S: Set of latent states, |S| = num_states (8 for ID, 16 for OOD)
  Sigma: Alphabet of symbols, |Sigma| = alphabet_size (16)
  T: State transition function, T: S x Sigma -> S
  E: Token emission function, E: S x Sigma -> Sigma
  s_0: Initial state sampled uniformly from S

The sequence generated is the sequence of emitted tokens e_t in [token_offset, token_offset + alphabet_size - 1].
The latent state s_t is not directly visible in the token stream, requiring the model to maintain
an internal algebraic state representation.

Splits:
  - D_train:      |S| = 8,  seed = seed
  - D_val:        |S| = 8,  seed = seed + 10000, disjoint sequences
  - D_test:       |S| = 8,  seed = seed + 20000, disjoint sequences
  - D_ood:        |S| = 16, seed = seed + 30000 (increased state-space capacity)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from typing import Tuple, Dict, List, Optional, Set


class HiddenFSM:
    def __init__(self, num_states: int = 8, alphabet_size: int = 16, token_offset: int = 10, seed: int = 42):
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.token_offset = token_offset
        self.seed = seed
        
        gen = torch.Generator().manual_seed(seed)
        # Deterministic transition table: T[s, sym] -> next_s
        self.T = torch.randint(0, num_states, (num_states, alphabet_size), generator=gen)
        # Deterministic emission table: E[s, sym] -> emitted_token
        self.E = torch.randint(token_offset, token_offset + alphabet_size, (num_states, alphabet_size), generator=gen)
        
    def step(self, state: int, symbol: int) -> Tuple[int, int]:
        """Returns (emitted_token, next_state)"""
        token = self.E[state, symbol].item()
        next_state = self.T[state, symbol].item()
        return token, next_state

    def generate_sequence(self, seq_len: int, gen: torch.Generator) -> Tuple[List[int], List[int]]:
        """Generates (tokens, latent_states)"""
        state = torch.randint(0, self.num_states, (1,), generator=gen).item()
        tokens = []
        states = [state]
        for _ in range(seq_len):
            sym = torch.randint(0, self.alphabet_size, (1,), generator=gen).item()
            tok, state = self.step(state, sym)
            tokens.append(tok)
            states.append(state)
        return tokens, states


def generate_fsm_samples(
    fsm: HiddenFSM,
    num_samples: int = 1000,
    seq_len: int = 128,
    seed: int = 42,
    existing_hashes: Optional[Set[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor, Set[int]]:
    gen = torch.Generator().manual_seed(seed)
    data: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    hashes = set(existing_hashes) if existing_hashes is not None else set()
    
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        tokens, _ = fsm.generate_sequence(seq_len, gen)
        seq_tuple = tuple(tokens)
        seq_hash = hash(seq_tuple)
        
        if seq_hash in hashes:
            continue
            
        hashes.add(seq_hash)
        data.append(torch.tensor(tokens, dtype=torch.long))
        # For FSM, all tokens predict the next step under regular state transition
        masks.append(torch.ones(seq_len, dtype=torch.bool))
        
    if len(data) < num_samples:
        raise RuntimeError(f"Failed to generate {num_samples} unique FSM sequences within {max_attempts} attempts.")
        
    return torch.stack(data), torch.stack(masks), hashes


def get_fsm_benchmark_splits(
    num_train: int = 1000,
    num_val: int = 200,
    num_test: int = 200,
    num_ood: int = 300,
    seq_len: int = 128,
    num_states_id: int = 8,
    num_states_ood: int = 16,
    alphabet_size: int = 16,
    vocab_size: int = 1024,
    seed: int = 42,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Constructs formal mathematically disjoint splits for Hidden FSM:
      - D_train
      - D_val
      - D_test
      - D_ood
    """
    fsm_id = HiddenFSM(num_states=num_states_id, alphabet_size=alphabet_size, seed=seed)
    fsm_ood = HiddenFSM(num_states=num_states_ood, alphabet_size=alphabet_size, seed=seed + 999)
    
    all_hashes: Set[int] = set()
    
    train_data, train_masks, all_hashes = generate_fsm_samples(
        fsm=fsm_id, num_samples=num_train, seq_len=seq_len,
        seed=seed, existing_hashes=all_hashes
    )
    
    val_data, val_masks, all_hashes = generate_fsm_samples(
        fsm=fsm_id, num_samples=num_val, seq_len=seq_len,
        seed=seed + 10000, existing_hashes=all_hashes
    )
    
    test_data, test_masks, all_hashes = generate_fsm_samples(
        fsm=fsm_id, num_samples=num_test, seq_len=seq_len,
        seed=seed + 20000, existing_hashes=all_hashes
    )
    
    ood_data, ood_masks, _ = generate_fsm_samples(
        fsm=fsm_ood, num_samples=num_ood, seq_len=seq_len,
        seed=seed + 30000, existing_hashes=set()
    )
    
    return {
        "train": (train_data, train_masks),
        "val": (val_data, val_masks),
        "test": (test_data, test_masks),
        "ood": (ood_data, ood_masks)
    }
