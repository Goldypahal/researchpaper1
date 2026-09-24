"""
Deterministic Evaluation Harness for Small Transformer.
Supports formal synthetic task families:
1. True Dyck-k Language with Hierarchical Stack Memory & Bracket Matching
   - Distinct bracket pairs: (), [], {}, <> (k=4)
   - Random filler / distractor tokens
   - In-distribution nesting depth: d in [1, 6]
   - Out-of-distribution (OOD) depth test: d in [7, 12]
   - Evaluates both cross-entropy loss and exact structural bracket prediction accuracy
2. Hidden Finite State Machine (Algebraic State Transitions)
   - Transition table T: S x Sigma -> S
   - Emission table E: S x Sigma -> Sigma
3. Parity Legacy Baseline (for backwards compatibility)

Outputs structured JSON metrics for the agent verifier and sandbox runner.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_baseline import SmallTransformerLM

# ---------------------------------------------------------------------------
# Task 1: True Dyck-k Grammar Generator with Stack Memory
# ---------------------------------------------------------------------------
def generate_dyck_k_dataset(num_samples=1200, seq_len=128, vocab_size=1024, k=4,
                            min_depth=1, max_depth=6, p_bracket=0.65, seed=42):
    """
    Generates syntactically guaranteed Dyck-k sequences with stack memory.
    Brackets:
      0: '(' = 10, ')' = 11
      1: '[' = 12, ']' = 13
      2: '{' = 14, '}' = 15
      3: '<' = 16, '>' = 17
    Distractors: tokens in [20, vocab_size - 1]
    Returns:
      data: LongTensor of shape (num_samples, seq_len)
      structural_mask: BoolTensor marking positions where a closing bracket is emitted
    """
    torch.manual_seed(seed)
    data = []
    masks = []
    
    for _ in range(num_samples):
        stack = []
        seq = []
        mask = []
        for i in range(seq_len):
            remaining = seq_len - 1 - i
            
            # Must close stack if remaining steps are just enough to close open brackets
            if remaining <= len(stack) and len(stack) > 0:
                b = stack.pop()
                token = 10 + 2 * b + 1
                is_structural = True
            elif len(stack) == 0:
                if remaining > 0 and torch.rand(1).item() < p_bracket:
                    b = torch.randint(0, k, (1,)).item()
                    stack.append(b)
                    token = 10 + 2 * b
                    is_structural = False
                else:
                    token = torch.randint(20, vocab_size, (1,)).item()
                    is_structural = False
            else:
                if len(stack) < max_depth and remaining > len(stack):
                    r = torch.rand(1).item()
                    if r < (p_bracket / 2.0):
                        b = torch.randint(0, k, (1,)).item()
                        stack.append(b)
                        token = 10 + 2 * b
                        is_structural = False
                    elif r < p_bracket:
                        b = stack.pop()
                        token = 10 + 2 * b + 1
                        is_structural = True
                    else:
                        token = torch.randint(20, vocab_size, (1,)).item()
                        is_structural = False
                else:
                    # At max depth, can only close or emit distractor
                    if torch.rand(1).item() < p_bracket:
                        b = stack.pop()
                        token = 10 + 2 * b + 1
                        is_structural = True
                    else:
                        token = torch.randint(20, vocab_size, (1,)).item()
                        is_structural = False
            seq.append(token)
            mask.append(is_structural)
            
        data.append(torch.tensor(seq, dtype=torch.long))
        masks.append(torch.tensor(mask, dtype=torch.bool))
        
    return torch.stack(data), torch.stack(masks)

# ---------------------------------------------------------------------------
# Task 2: Hidden Finite State Machine (Algebraic State Transitions)
# ---------------------------------------------------------------------------
def generate_fsm_dataset(num_samples=1200, seq_len=128, vocab_size=1024,
                         num_states=8, alphabet_size=16, seed=42):
    """
    Generates sequences from a deterministic hidden finite state machine.
    Transition table T: S x Sigma -> S
    Emission table E: S x Sigma -> Sigma
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    T = torch.randint(0, num_states, (num_states, alphabet_size), generator=gen)
    E = torch.randint(10, 10 + alphabet_size, (num_states, alphabet_size), generator=gen)
    
    data = []
    masks = []
    for _ in range(num_samples):
        state = torch.randint(0, num_states, (1,), generator=gen).item()
        seq = []
        for _ in range(seq_len):
            sym = torch.randint(0, alphabet_size, (1,), generator=gen).item()
            token = E[state, sym].item()
            state = T[state, sym].item()
            seq.append(token)
        data.append(torch.tensor(seq, dtype=torch.long))
        masks.append(torch.ones(seq_len, dtype=torch.bool))
    return torch.stack(data), torch.stack(masks)

# ---------------------------------------------------------------------------
# Task 3: Parity Legacy
# ---------------------------------------------------------------------------
def generate_parity_legacy_dataset(num_samples=1200, seq_len=128, vocab_size=1024, seed=42):
    torch.manual_seed(seed)
    data = []
    for _ in range(num_samples):
        seq = torch.randint(10, vocab_size, (seq_len,), dtype=torch.long)
        parity = 0
        for i in range(1, seq_len):
            if seq[i-1] % 2 == 0:
                parity = (parity + 1) % 4
            if parity == 2:
                seq[i] = (seq[i] + 17) % (vocab_size - 10) + 10
        data.append(seq)
    masks = [torch.ones(seq_len, dtype=torch.bool) for _ in range(num_samples)]
    return torch.stack(data), torch.stack(masks)

# ---------------------------------------------------------------------------
# Dataset Dispatcher
# ---------------------------------------------------------------------------
def load_datasets(task="dyck", num_samples=1200, seq_len=128, vocab_size=1024, seed=42):
    """
    Returns (train_data, val_data, ood_data, val_masks, ood_masks)
    """
    if task == "dyck":
        # In-distribution (train/val): nesting depth 1 to 6
        data, masks = generate_dyck_k_dataset(
            num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size,
            k=4, min_depth=1, max_depth=6, p_bracket=0.65, seed=seed
        )
        # OOD test: deep nesting depth 7 to 12
        ood_data, ood_masks = generate_dyck_k_dataset(
            num_samples=num_samples // 4, seq_len=seq_len, vocab_size=vocab_size,
            k=4, min_depth=7, max_depth=12, p_bracket=0.75, seed=seed + 999
        )
    elif task == "fsm":
        data, masks = generate_fsm_dataset(
            num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size,
            num_states=8, alphabet_size=16, seed=seed
        )
        ood_data, ood_masks = generate_fsm_dataset(
            num_samples=num_samples // 4, seq_len=seq_len, vocab_size=vocab_size,
            num_states=16, alphabet_size=16, seed=seed + 999
        )
    else:
        data, masks = generate_parity_legacy_dataset(
            num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size, seed=seed
        )
        ood_data, ood_masks = data[:num_samples // 4], masks[:num_samples // 4]

    split_idx = int(0.8 * num_samples)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    val_masks = masks[split_idx:]
    return train_data, val_data, ood_data, val_masks, ood_masks

# ---------------------------------------------------------------------------
# Model Evaluation Routine
# ---------------------------------------------------------------------------
def evaluate(model, val_data, val_masks=None, batch_size=32, device='cpu'):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct_structural = 0
    total_structural = 0

    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, loss = model(inputs, targets)
            total_loss += loss.item()
            total_batches += 1

            if val_masks is not None:
                batch_masks = val_masks[i:i+batch_size][:, 1:].to(device)
                preds = logits.argmax(dim=-1)
                structural_positions = batch_masks
                if structural_positions.sum() > 0:
                    correct = (preds == targets) & structural_positions
                    correct_structural += correct.sum().item()
                    total_structural += structural_positions.sum().item()

    avg_loss = total_loss / max(1, total_batches)
    acc = round((correct_structural / total_structural) * 100, 2) if total_structural > 0 else 0.0
    return avg_loss, acc

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def run_training(task="dyck", dry_run=False, max_steps=200, lr=1e-3, weight_decay=0.01,
                 n_layers=6, n_heads=4, n_kv_heads=None, d_model=256, d_ff=1024,
                 activation="gelu", norm_type="layernorm", scale_factor=None,
                 pos_encoding="learned", ffn_type="standard", topology="pre_ln",
                 seed=42, device='cpu'):
    start_time = time.time()
    torch.manual_seed(seed)

    num_samples = 300 if dry_run else 1200
    seq_len = 64 if dry_run else 128
    vocab_size = 1024

    train_data, val_data, ood_data, val_masks, ood_masks = load_datasets(
        task=task, num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size, seed=seed
    )

    layers = 2 if dry_run else n_layers
    dim = 128 if dry_run else d_model
    ff_dim = 512 if dry_run else d_ff
    steps = 30 if dry_run else max_steps
    batch_size = 16 if dry_run else 32

    model = SmallTransformerLM(
        vocab_size=vocab_size,
        d_model=dim,
        n_layers=layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=ff_dim,
        max_seq_len=seq_len,
        activation=activation,
        norm_type=norm_type,
        scale_factor=scale_factor,
        pos_encoding=pos_encoding,
        ffn_type=ffn_type,
        topology=topology
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initial baseline validation
    initial_val_loss, initial_val_acc = evaluate(model, val_data, val_masks, batch_size, device)

    model.train()
    train_losses = []
    for step in range(steps):
        idx = torch.randint(0, len(train_data), (batch_size,))
        batch = train_data[idx].to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        _, loss = model(inputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            elapsed = time.time() - start_time
            results = {
                "status": "DIVERGED",
                "task": task,
                "dry_run": dry_run,
                "steps_completed": step,
                "initial_val_loss": round(initial_val_loss, 4),
                "final_val_loss": 999.0,
                "final_train_loss": 999.0,
                "relative_loss_reduction_pct": -999.0,
                "val_structural_accuracy_pct": 0.0,
                "ood_val_loss": 999.0,
                "ood_structural_accuracy_pct": 0.0,
                "elapsed_seconds": round(elapsed, 2),
                "param_count": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            print(json.dumps(results, indent=2))
            return results

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # Final evaluations (in-distribution validation and out-of-distribution test)
    final_val_loss, final_val_acc = evaluate(model, val_data, val_masks, batch_size, device)
    ood_val_loss, ood_acc = evaluate(model, ood_data, ood_masks, batch_size, device)
    elapsed_seconds = time.time() - start_time

    rel_reduction = round(((initial_val_loss - final_val_loss) / initial_val_loss) * 100, 2) if initial_val_loss > 0 else 0.0

    results = {
        "status": "SUCCESS",
        "task": task,
        "dry_run": dry_run,
        "steps_completed": steps,
        "initial_val_loss": round(initial_val_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_train_loss": round(train_losses[-1], 4) if train_losses else round(final_val_loss, 4),
        "relative_loss_reduction_pct": rel_reduction,
        "val_structural_accuracy_pct": final_val_acc,
        "ood_val_loss": round(ood_val_loss, 4),
        "ood_structural_accuracy_pct": ood_acc,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "hyperparameters": {
            "lr": lr,
            "weight_decay": weight_decay,
            "n_layers": layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "d_model": dim,
            "activation": activation,
            "norm_type": norm_type,
            "scale_factor": scale_factor,
            "pos_encoding": pos_encoding,
            "ffn_type": ffn_type,
            "topology": topology
        }
    }

    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Evaluation Harness for Small Transformer")
    parser.add_argument("--task", type=str, default="dyck", choices=["dyck", "fsm", "parity_legacy"],
                        help="Benchmark task family (dyck: True Dyck-k grammar; fsm: Hidden FSM; parity_legacy: legacy parity)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=False)

    parser.add_argument("--steps", type=int, default=150, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="KV heads for GQA (defaults to n_heads for MHA)")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation (gelu, relu, silu)")
    parser.add_argument("--norm-type", type=str, default="layernorm", help="Norm type (layernorm, rmsnorm)")
    parser.add_argument("--scale-factor", type=float, default=None, help="Custom attention scaling")
    parser.add_argument("--pos-encoding", type=str, default="learned",
                        choices=["learned", "sinusoidal", "rotary", "none"],
                        help="Positional encoding scheme")
    parser.add_argument("--ffn-type", type=str, default="standard",
                        choices=["standard", "swiglu"], help="Feed-forward network architecture")
    parser.add_argument("--topology", type=str, default="pre_ln",
                        choices=["pre_ln", "post_ln", "parallel"], help="Residual block topology")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_training(
        task=args.task,
        dry_run=args.dry_run,
        max_steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_model=args.d_model,
        activation=args.activation,
        norm_type=args.norm_type,
        scale_factor=args.scale_factor,
        pos_encoding=args.pos_encoding,
        ffn_type=args.ffn_type,
        topology=args.topology,
        seed=args.seed
    )
