"""
Arm 2: Random Search Hyperparameter Baseline.
Performs uniform / log-uniform sampling over the identical parameter space
specified in the LLM Agent's system prompt:
1. lr: float in range [1e-5, 5e-2] (sampled log-uniformly)
2. weight_decay: float in range [0.0, 0.2] (sampled uniformly)
3. n_layers: int in {2, 4, 6, 8}
4. n_heads: int in {2, 4, 8}
5. d_model: int in {128, 256, 512} (divisible by n_heads)
6. activation: "gelu", "relu", or "silu"
7. norm_type: "layernorm" or "rmsnorm"
8. scale_factor: None (50% prob) or float in [0.05, 0.5]

Shares the exact same:
- Sandbox execution environment (sandbox_runner)
- Deterministic Dyck evaluation dataset (eval_harness)
- Stopping conditions (10 GPU-hour budget, 3-strike divergence, iteration cap)
- Trace logging conforming to trace_schema.json
"""

import os
import sys
import time
import math
import random
import argparse
from trace_logger import TraceLogger
from sandbox_runner import execute_in_sandbox


# Exact Parameter Space Specification (Mirrored from llm_agent.py SYSTEM_PROMPT)
PARAM_SPACE = {
    "lr_min": 1e-5,
    "lr_max": 5e-2,
    "weight_decay_min": 0.0,
    "weight_decay_max": 0.2,
    "n_layers_choices": [2, 4, 6, 8],
    "n_heads_choices": [2, 4, 8],
    "d_model_choices": [128, 256, 512],
    "activation_choices": ["gelu", "relu", "silu"],
    "norm_type_choices": ["layernorm", "rmsnorm"],
    "scale_factor_choices": [None, 0.05, 0.1, 0.15, 0.25]
}


def sample_random_hyperparameters(seed=None):
    if seed is not None:
        random.seed(seed)

    # Log-uniform sampling for learning rate
    log_lr_min = math.log10(PARAM_SPACE["lr_min"])
    log_lr_max = math.log10(PARAM_SPACE["lr_max"])
    sampled_lr = round(10 ** random.uniform(log_lr_min, log_lr_max), 6)

    # Uniform sampling for weight decay
    sampled_wd = round(random.uniform(PARAM_SPACE["weight_decay_min"], PARAM_SPACE["weight_decay_max"]), 4)

    sampled_layers = random.choice(PARAM_SPACE["n_layers_choices"])
    sampled_heads = random.choice(PARAM_SPACE["n_heads_choices"])
    
    # Ensure d_model is divisible by n_heads
    valid_d_models = [d for d in PARAM_SPACE["d_model_choices"] if d % sampled_heads == 0]
    sampled_d_model = random.choice(valid_d_models)

    sampled_activation = random.choice(PARAM_SPACE["activation_choices"])
    sampled_norm = random.choice(PARAM_SPACE["norm_type_choices"])
    sampled_scale = random.choice(PARAM_SPACE["scale_factor_choices"])

    return {
        "lr": sampled_lr,
        "weight_decay": sampled_wd,
        "n_layers": sampled_layers,
        "n_heads": sampled_heads,
        "d_model": sampled_d_model,
        "activation": sampled_activation,
        "norm_type": sampled_norm,
        "scale_factor": sampled_scale
    }


def run_random_search(experiment_id="exp_002_random_baseline", arm="Arm2_RandomSearchBaseline",
                      dry_run=False, max_iterations=5, budget_hours=10.0, seed=42):
    print("=== INITIALIZING RANDOM SEARCH HYPERPARAMETER BASELINE (ARM 2) ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Arm: {arm}")
    print(f"Max Iterations: {max_iterations}")
    print(f"GPU Budget: {budget_hours} Hours")
    print(f"Dry Run Mode: {dry_run}")
    print(f"Random Seed: {seed}\n")

    random.seed(seed)
    logger = TraceLogger(
        experiment_id=experiment_id,
        agent_arm=arm,
        total_gpu_budget_hours=budget_hours
    )

    harness_path = os.path.join(os.path.dirname(__file__), "eval_harness.py")
    harness_base_args = ["--dry-run"] if dry_run else ["--no-dry-run", "--steps", "50"]

    # Run baseline evaluation (Iteration 0)
    print("[Iteration 0] Running Baseline Model Evaluation...")
    base_run = execute_in_sandbox(harness_path, harness_base_args, timeout_sec=60 if dry_run else 300)

    if base_run["status"] != "SUCCESS" or not base_run["metrics"]:
        print(f"Baseline run failed! Stderr: {base_run['stderr_tail']}")
        return

    base_loss = base_run["metrics"]["final_val_loss"]
    print(f"Baseline Validation Cross-Entropy Loss: {base_loss:.4f}\n")

    hypo_0 = logger.add_hypothesis(
        iteration_idx=0,
        text="Baseline SmallTransformerLM (6 layers, 256 dim, 4 heads, GELU, LayerNorm).",
        target_component="attention",
        predicted_delta_loss=0.0,
        provider="ground_truth_baseline",
    )
    exp_0 = logger.add_experiment_spec(0, hypo_0, "Original unmodified model_baseline.py")
    run_0, _ = logger.add_execution_run(
        0, exp_0, "SUCCESS", 0, base_run["elapsed_seconds"], base_loss, base_run["metrics"]["final_train_loss"]
    )
    logger.add_reflection(0, run_0, hypo_0, "Baseline established as reference anchor.", "ACCEPT_AND_EXTEND", False)

    total_gpu_sec = base_run["elapsed_seconds"]
    consecutive_divergences = 0
    active_parent_hypo = hypo_0

    for it in range(1, max_iterations + 1):
        # Check stopping conditions
        if total_gpu_sec / 3600.0 >= budget_hours:
            print(f"[STOP] GPU Budget Exhausted ({total_gpu_sec/3600.0:.2f}h >= {budget_hours}h).")
            break
        if consecutive_divergences >= 3:
            print(f"[STOP] Three-strike divergence condition reached ({consecutive_divergences} consecutive failures).")
            break

        print(f"\n--- [Iteration {it}/{max_iterations}] ---")
        sampled_params = sample_random_hyperparameters()
        print(f"[Arm 2 Random Sampler] Sampled Parameters: {sampled_params}")

        # In random search, there is no technical hypothesis; log parameters tested
        hypo_text = f"[RANDOM_UNIFORM] Sampled point in design space: lr={sampled_params['lr']}, wd={sampled_params['weight_decay']}, layers={sampled_params['n_layers']}, heads={sampled_params['n_heads']}, d_model={sampled_params['d_model']}, act={sampled_params['activation']}, norm={sampled_params['norm_type']}."
        hypo_id = logger.add_hypothesis(
            iteration_idx=it,
            text=hypo_text,
            target_component="random_sampling",
            predicted_delta_loss=0.0,
            parent_hypo_id=active_parent_hypo,
            provider="random_uniform_sampler"
        )

        exp_id = logger.add_experiment_spec(it, hypo_id, str(sampled_params))

        # Build sandbox arguments
        run_args = ["--dry-run"] if dry_run else ["--no-dry-run", "--steps", "50"]
        run_args.extend(["--lr", str(sampled_params["lr"])])
        run_args.extend(["--weight-decay", str(sampled_params["weight_decay"])])
        run_args.extend(["--n-layers", str(sampled_params["n_layers"])])
        run_args.extend(["--n-heads", str(sampled_params["n_heads"])])
        run_args.extend(["--d-model", str(sampled_params["d_model"])])
        run_args.extend(["--activation", str(sampled_params["activation"])])
        run_args.extend(["--norm-type", str(sampled_params["norm_type"])])
        if sampled_params.get("scale_factor"):
            run_args.extend(["--scale-factor", str(sampled_params["scale_factor"])])

        # Execute candidate in sandbox
        res = execute_in_sandbox(harness_path, run_args, timeout_sec=60 if dry_run else 300)
        total_gpu_sec += res["elapsed_seconds"]

        # Parse evaluation results
        if res["status"] == "SUCCESS" and res["metrics"]:
            val_loss = res["metrics"]["final_val_loss"]
            train_loss = res["metrics"]["final_train_loss"]
            rel_gain = round(((base_loss - val_loss) / base_loss) * 100, 2)
            print(f"Result: Validation Loss = {val_loss:.4f} (Train Loss = {train_loss:.4f}, Rel Gain = {rel_gain:+0.2f}%)")

            if val_loss > 3.0 * base_loss:
                consecutive_divergences += 1
                status = "DIVERGED"
                decision = "REJECT_AND_BACKTRACK"
                print(f"[WARNING] Divergence! Val Loss {val_loss:.4f} > 3x Base.")
            else:
                consecutive_divergences = 0
                status = "SUCCESS"
                if val_loss < logger.incumbent_best_loss:
                    decision = "ACCEPT_AND_EXTEND"
                    active_parent_hypo = hypo_id
                    print(f"[IMPROVEMENT] Discovered new incumbent best loss: {val_loss:.4f}!")
                else:
                    decision = "REJECT_AND_BACKTRACK"
                    print(f"[REJECT] Val Loss {val_loss:.4f} did not beat incumbent {logger.incumbent_best_loss:.4f}.")
        else:
            val_loss = None
            train_loss = None
            rel_gain = None
            status = "RUNTIME_CRASH"
            decision = "REJECT_AND_BACKTRACK"
            consecutive_divergences += 1
            print(f"[CRASH] Execution failed! Stderr: {res['stderr_tail']}")

        run_id, is_best = logger.add_execution_run(
            iteration_idx=it,
            exp_id=exp_id,
            status=status,
            exit_code=res["exit_code"],
            gpu_seconds=res["elapsed_seconds"],
            val_loss=val_loss,
            train_loss=train_loss
        )

        logger.add_reflection(
            iteration_idx=it,
            run_id=run_id,
            hypo_id=hypo_id,
            reflection_text=f"Iteration {it} outcome: {status} with decision {decision}. Sampled by random search.",
            decision=decision,
            self_detected_failure=(status != "SUCCESS")
        )

        # Progressively save trace
        trace_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "execution_traces",
            f"trace_{experiment_id}.json"
        )
        logger.export_trace(trace_path)

    summary = logger.export_trace(trace_path)
    print("\n=== RANDOM SEARCH BASELINE COMPLETE ===")
    print(f"Total Iterations: {summary['summary_statistics']['total_iterations']}")
    print(f"Dead-End Ratio: {summary['summary_statistics']['dead_end_ratio']}")
    print(f"Path to First Improvement: {summary['summary_statistics']['path_to_first_improvement_steps']}")
    print(f"Final Best Loss: {summary['summary_statistics']['final_best_loss']}")
    print(f"Final Relative Gain: {summary['summary_statistics']['final_relative_improvement_pct']}%")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arm 2 Random Search Hyperparameter Baseline")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-hours", type=float, default=10.0)
    args = parser.parse_args()

    run_random_search(
        dry_run=args.dry_run,
        max_iterations=args.iterations,
        budget_hours=args.budget_hours,
        seed=args.seed
    )
