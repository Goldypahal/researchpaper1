"""
Task Complexity Ladder Framework (Phase 17).
Systematically varies formal language complexity to test the core theoretical hypothesis:

  Reasoning Value R(C, B) = Performance_LLM(C, B) - Performance_Classical(C, B)
  as a function of task complexity C and search budget B.

Complexity Dimensions:
  1. Dyck Hierarchical Depth Ladder:
     C_Dyck in {2, 4, 6, 8, 12, 16} maximum stack depth
  2. Hidden FSM Latent State Ladder:
     C_FSM in {4, 8, 16, 32, 64} latent states
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import math
import json
import torch
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.dyck import generate_dyck_k_samples
from benchmarks.fsm import HiddenFSM, generate_fsm_samples
from evaluator.immutable_evaluator import ImmutableEvaluator
from search_space import get_default_baseline, sanitize_configuration


def build_dyck_complexity_splits(depth: int, seq_len: int = 128, seed: int = 42) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Builds Dyck splits parametrized by exact maximum stack nesting depth."""
    min_d = max(1, depth // 2)
    max_d = depth
    
    train_data, train_masks, hashes = generate_dyck_k_samples(
        num_samples=600, seq_len=seq_len, k=4, min_depth=min_d, max_depth=max_d,
        p_bracket=0.65, seed=seed
    )
    val_data, val_masks, _ = generate_dyck_k_samples(
        num_samples=150, seq_len=seq_len, k=4, min_depth=min_d, max_depth=max_d,
        p_bracket=0.65, seed=seed + 10000, existing_hashes=hashes
    )
    # Extrapolation OOD: deeper than current ladder level
    ood_data, ood_masks, _ = generate_dyck_k_samples(
        num_samples=150, seq_len=seq_len, k=4, min_depth=max_d + 1, max_depth=max_d + 4,
        p_bracket=0.65, seed=seed + 30000
    )
    
    return {
        "train": (train_data, train_masks),
        "val": (val_data, val_masks),
        "test": (val_data, val_masks),
        "ood": (ood_data, ood_masks)
    }


def build_fsm_complexity_splits(num_states: int, seq_len: int = 128, seed: int = 42) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Builds FSM splits parametrized by exact latent state count."""
    fsm_id = HiddenFSM(num_states=num_states, alphabet_size=16, seed=seed)
    fsm_ood = HiddenFSM(num_states=num_states * 2, alphabet_size=16, seed=seed + 999)
    
    train_data, train_masks, hashes = generate_fsm_samples(fsm=fsm_id, num_samples=600, seq_len=seq_len, seed=seed)
    val_data, val_masks, _ = generate_fsm_samples(fsm=fsm_id, num_samples=150, seq_len=seq_len, seed=seed + 10000, existing_hashes=hashes)
    ood_data, ood_masks, _ = generate_fsm_samples(fsm=fsm_ood, num_samples=150, seq_len=seq_len, seed=seed + 30000)
    
    return {
        "train": (train_data, train_masks),
        "val": (val_data, val_masks),
        "test": (val_data, val_masks),
        "ood": (ood_data, ood_masks)
    }


class ComplexityLadderEvaluator(ImmutableEvaluator):
    """Custom ImmutableEvaluator instance taking explicit complexity ladder splits."""
    def __init__(self, task_name: str, splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]], seed: int = 42):
        self.task = task_name
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        self.evaluator_id = f"COMPLEXITY_EVAL_{task_name.upper()}_SEED_{seed}"
        self._splits = splits
        self._val_hash = self._hash_tensor(self._splits["val"][0])
        self._test_hash = self._hash_tensor(self._splits["test"][0])
        self._ood_hash = self._hash_tensor(self._splits["ood"][0])


def evaluate_complexity_point(
    task_family: str,
    complexity_val: int,
    seed: int = 42,
    steps: int = 30
) -> Dict[str, Any]:
    """Evaluates baseline reference and generalizability at a single complexity ladder rung."""
    if task_family.lower() == "dyck":
        splits = build_dyck_complexity_splits(depth=complexity_val, seed=seed)
        task_label = f"dyck_depth_{complexity_val}"
    else:
        splits = build_fsm_complexity_splits(num_states=complexity_val, seed=seed)
        task_label = f"fsm_states_{complexity_val}"

    evaluator = ComplexityLadderEvaluator(task_name=task_label, splits=splits, seed=seed)
    base_cfg = get_default_baseline()
    res = evaluator.train_and_evaluate_candidate(base_cfg, max_steps=steps)

    return {
        "task_family": task_family,
        "complexity_level": complexity_val,
        "seed": seed,
        "val_loss": res["val_loss"],
        "ood_loss": res["ood_loss"],
        "val_struct_acc": res["val_struct_acc"],
        "ood_struct_acc": res["ood_struct_acc"],
        "generalization_gap_rel": res["generalization_gap_rel"]
    }


if __name__ == "__main__":
    print("=== Testing Task Complexity Ladder Generation ===")
    dyck_point = evaluate_complexity_point("dyck", complexity_val=4, seed=42, steps=10)
    print(f"Dyck Depth 4: Val Loss = {dyck_point['val_loss']:.4f} | Struct Acc = {dyck_point['val_struct_acc']:.2f}%")
    fsm_point = evaluate_complexity_point("fsm", complexity_val=8, seed=42, steps=10)
    print(f"FSM States 8: Val Loss = {fsm_point['val_loss']:.4f} | Struct Acc = {fsm_point['val_struct_acc']:.2f}%")
