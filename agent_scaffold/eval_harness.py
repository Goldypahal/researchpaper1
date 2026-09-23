"""
Deterministic Evaluation Harness for Small Transformer.
Generates an unseen synthetic state-transition sequence dataset,
trains the model for a strict step budget, and evaluates holdout validation cross-entropy.
Outputs structured JSON metrics for the agent verifier.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import time
import math
import torch
import torch.nn as nn
from model_baseline import SmallTransformerLM

def generate_synthetic_dataset(num_samples=2000, seq_len=128, vocab_size=1024, seed=42):
    """
    Generates a deterministic algebraic Dyck / state-transition token stream.
    Requires the network to learn non-local bracket matching and parity state updates.
    """
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
    
    data = torch.stack(data)
    split_idx = int(0.8 * num_samples)
    return data[:split_idx], data[split_idx:]

def evaluate(model, val_data, batch_size=32, device='cpu'):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            _, loss = model(inputs, targets)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(1, total_batches)

def run_training(dry_run=False, max_steps=200, lr=1e-3, weight_decay=0.01,
                 n_layers=6, n_heads=4, d_model=256, d_ff=1024,
                 activation="gelu", norm_type="layernorm", scale_factor=None, device='cpu'):
    start_time = time.time()
    
    # Dataset scaling
    train_data, val_data = generate_synthetic_dataset(
        num_samples=300 if dry_run else 1200,
        seq_len=64 if dry_run else 128
    )

    layers = 2 if dry_run else n_layers
    dim = 128 if dry_run else d_model
    ff_dim = 512 if dry_run else d_ff
    steps = 30 if dry_run else max_steps
    batch_size = 16 if dry_run else 32

    model = SmallTransformerLM(
        vocab_size=1024,
        d_model=dim,
        n_layers=layers,
        n_heads=n_heads,
        d_ff=ff_dim,
        max_seq_len=128,
        activation=activation,
        norm_type=norm_type,
        scale_factor=scale_factor
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    initial_val_loss = evaluate(model, val_data, batch_size, device)
    
    train_losses = []
    for step in range(steps):
        idx = torch.randint(0, len(train_data), (batch_size,))
        batch = train_data[idx].to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        _, loss = model(inputs, targets)
        if torch.isnan(loss) or torch.isinf(loss):
            # Model diverged
            final_val_loss = 999.0
            elapsed = time.time() - start_time
            results = {
                "status": "DIVERGED",
                "dry_run": dry_run,
                "steps_completed": step,
                "initial_val_loss": round(initial_val_loss, 4),
                "final_val_loss": 999.0,
                "final_train_loss": 999.0,
                "relative_loss_reduction_pct": -999.0,
                "elapsed_seconds": round(elapsed, 2),
                "param_count": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            print(json.dumps(results, indent=2))
            return results

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    final_val_loss = evaluate(model, val_data, batch_size, device)
    elapsed_seconds = time.time() - start_time

    results = {
        "status": "SUCCESS",
        "dry_run": dry_run,
        "steps_completed": steps,
        "initial_val_loss": round(initial_val_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_train_loss": round(train_losses[-1], 4),
        "relative_loss_reduction_pct": round(((initial_val_loss - final_val_loss) / initial_val_loss) * 100, 2),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "hyperparameters": {
            "lr": lr,
            "weight_decay": weight_decay,
            "n_layers": layers,
            "n_heads": n_heads,
            "d_model": dim,
            "activation": activation,
            "norm_type": norm_type
        }
    }

    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Evaluation Harness")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=False)

    parser.add_argument("--steps", type=int, default=150, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation (gelu, relu, silu)")
    parser.add_argument("--norm-type", type=str, default="layernorm", help="Norm type (layernorm, rmsnorm)")
    parser.add_argument("--scale-factor", type=float, default=None, help="Custom attention scaling")
    args = parser.parse_args()

    run_training(
        dry_run=args.dry_run,
        max_steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        activation=args.activation,
        norm_type=args.norm_type,
        scale_factor=args.scale_factor
    )
