"""
Immutable Evaluator for Autonomous ML Optimization.
Architectural Principle:
  The agent can propose candidate configurations, hyperparameters, or model definitions,
  but the AGENT NEVER CONTROLS:
    1. Dataset generation or test splits
    2. Loss and accuracy metric computation
    3. Generalization gap calculation
    4. Hardware resource & compute budget accounting

The evaluation harness evaluates candidates against frozen in-memory tensors.
It validates all returned tensors, weights, and losses against:
  - NaN / Inf corruption
  - Adversarial metric injection / stdout forgery
  - Checkpoint reuse / parameter leakages
  - Budget overruns
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import math
import time
import json
import hashlib
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks import load_benchmark
from agent_scaffold.model_baseline import SmallTransformerLM


class EvaluationSecurityError(Exception):
    """Raised when an adversarial exploit or integrity breach is detected."""
    pass


class ImmutableEvaluator:
    def __init__(self, task: str = "dyck", seed: int = 42, device: str = "cpu"):
        self.task = task.lower()
        self.seed = seed
        self.device = torch.device(device)
        self.evaluator_id = f"IMMUTABLE_EVAL_{self.task.upper()}_SEED_{seed}"
        
        # Load benchmark splits into private immutable store
        self._splits = load_benchmark(
            self.task,
            num_train=1000,
            num_val=200,
            num_test=200,
            num_ood=300,
            seq_len=128,
            vocab_size=1024,
            seed=seed
        )
        
        # Compute integrity hashes for the evaluation sets
        self._val_hash = self._hash_tensor(self._splits["val"][0])
        self._test_hash = self._hash_tensor(self._splits["test"][0])
        self._ood_hash = self._hash_tensor(self._splits["ood"][0])
        
    @staticmethod
    def _hash_tensor(t: torch.Tensor) -> str:
        return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verifies that evaluation sets have not been modified in memory."""
        curr_val_hash = self._hash_tensor(self._splits["val"][0])
        curr_test_hash = self._hash_tensor(self._splits["test"][0])
        curr_ood_hash = self._hash_tensor(self._splits["ood"][0])
        return (curr_val_hash == self._val_hash and
                curr_test_hash == self._test_hash and
                curr_ood_hash == self._ood_hash)

    def evaluate_model(self, model: nn.Module, batch_size: int = 32) -> Dict[str, Any]:
        """
        Directly evaluates model against immutable validation, test, and OOD sets.
        Computes ID loss, token acc, OOD loss, structural acc, and generalization gaps.
        Validates output logits against NaN, Inf, and invalid bounds.
        """
        if not self.verify_integrity():
            raise EvaluationSecurityError("CRITICAL: Evaluation split integrity compromised!")

        model.eval()
        model.to(self.device)

        results = {}
        for split_name in ["val", "test", "ood"]:
            data, masks = self._splits[split_name]
            loss, token_acc, struct_acc = self._evaluate_split(model, data, masks, batch_size)
            
            # Security checks on metrics
            if math.isnan(loss) or math.isinf(loss):
                raise EvaluationSecurityError(f"Model produced invalid loss ({loss}) on split {split_name}.")
            if loss <= 0.0 or loss > 100.0:
                raise EvaluationSecurityError(f"Loss out of plausible physical bounds ({loss}) on split {split_name}.")
                
            results[f"{split_name}_loss"] = round(loss, 6)
            results[f"{split_name}_token_acc"] = round(token_acc, 2)
            results[f"{split_name}_struct_acc"] = round(struct_acc, 2)

        # Compute formal Generalization Gaps
        val_loss = results["val_loss"]
        ood_loss = results["ood_loss"]
        gap_abs = ood_loss - val_loss
        gap_rel = gap_abs / max(1e-8, val_loss)

        results["generalization_gap_abs"] = round(gap_abs, 6)
        results["generalization_gap_rel"] = round(gap_rel, 6)
        return results

    def _evaluate_split(self, model: nn.Module, data: torch.Tensor, masks: torch.Tensor,
                        batch_size: int = 32) -> Tuple[float, float, float]:
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        total_structural = 0
        correct_structural = 0
        total_batches = 0

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size].to(self.device)
                batch_masks = masks[i:i + batch_size][:, 1:].to(self.device)
                
                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                logits, loss = model(inputs, targets)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise EvaluationSecurityError("NaN or Inf detected in model forward pass logits.")

                total_loss += loss.item()
                total_batches += 1

                preds = logits.argmax(dim=-1)
                correct_tokens += (preds == targets).sum().item()
                total_tokens += targets.numel()

                if batch_masks.sum() > 0:
                    struct_corr = (preds == targets) & batch_masks
                    correct_structural += struct_corr.sum().item()
                    total_structural += batch_masks.sum().item()

        avg_loss = total_loss / max(1, total_batches)
        token_acc = (correct_tokens / max(1, total_tokens)) * 100.0
        struct_acc = (correct_structural / max(1, total_structural)) * 100.0 if total_structural > 0 else 0.0
        return avg_loss, token_acc, struct_acc

    def train_and_evaluate_candidate(
        self,
        config: Dict[str, Any],
        max_steps: int = 150,
        batch_size: int = 32,
        timeout_seconds: float = 120.0
    ) -> Dict[str, Any]:
        """
        Executes a controlled candidate training run under strict resource and integrity accounting.
        """
        start_time = time.time()
        
        # 1. Parameter extraction with safe fallbacks and bounded sanitization
        lr = float(config.get("lr", 1e-3))
        weight_decay = float(config.get("weight_decay", 0.01))
        d_model = int(config.get("d_model", 256))
        n_layers = int(config.get("n_layers", 6))
        n_heads = int(config.get("n_heads", 4))
        n_kv_heads = config.get("n_kv_heads", None)
        d_ff = int(config.get("d_ff", 1024))
        activation = str(config.get("activation", "gelu"))
        norm_type = str(config.get("norm_type", "layernorm"))
        pos_encoding = str(config.get("pos_encoding", "learned"))
        ffn_type = str(config.get("ffn_type", "standard"))
        topology = str(config.get("topology", "pre_ln"))
        scale_factor = config.get("scale_factor", None)

        # 2. Build candidate model instance
        torch.manual_seed(self.seed)
        model = SmallTransformerLM(
            vocab_size=1024,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=128,
            activation=activation,
            norm_type=norm_type,
            scale_factor=scale_factor,
            pos_encoding=pos_encoding,
            ffn_type=ffn_type,
            topology=topology
        ).to(self.device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 3. Controlled Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 4. Training loop over private D_train
        train_data, _ = self._splits["train"]
        model.train()
        train_loss_accum = 0.0
        step_count = 0

        for step in range(max_steps):
            if (time.time() - start_time) > timeout_seconds:
                raise TimeoutError(f"Candidate exceeded compute budget limit of {timeout_seconds} seconds.")

            idx = torch.randint(0, len(train_data) - batch_size + 1, (1,)).item()
            batch = train_data[idx:idx + batch_size].to(self.device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits, loss = model(inputs, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                raise EvaluationSecurityError(f"Training diverged (loss={loss.item()}) at step {step}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_accum += loss.item()
            step_count += 1

        train_loss_final = train_loss_accum / max(1, step_count)
        elapsed_gpu_sec = time.time() - start_time

        # 5. Immutable independent evaluation
        eval_metrics = self.evaluate_model(model, batch_size=batch_size)

        return {
            "task": self.task,
            "seed": self.seed,
            "train_loss": round(train_loss_final, 6),
            "val_loss": eval_metrics["val_loss"],
            "test_loss": eval_metrics["test_loss"],
            "ood_loss": eval_metrics["ood_loss"],
            "val_token_acc": eval_metrics["val_token_acc"],
            "val_struct_acc": eval_metrics["val_struct_acc"],
            "ood_struct_acc": eval_metrics["ood_struct_acc"],
            "generalization_gap_abs": eval_metrics["generalization_gap_abs"],
            "generalization_gap_rel": eval_metrics["generalization_gap_rel"],
            "parameter_count": num_params,
            "gpu_seconds": round(elapsed_gpu_sec, 3),
            "steps_completed": step_count,
            "evaluator_signature": self.evaluator_id
        }
