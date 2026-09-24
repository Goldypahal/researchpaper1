"""
Bayesian Optimization (TPE) Baseline (Arm 3) for SmallTransformerLM.
Uses Optuna's Tree-Structured Parzen Estimator (TPE) to suggest hyperparameters
across iterations, logging full graph traces compatible with trace_schema.json.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import argparse
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from trace_logger import TraceLogger
from sandbox_runner import execute_in_sandbox

def run_bayesian_optimization(experiment_id="exp_arm3_tpe_baseline",
                              arm="Arm3_BayesianOptimization_TPE",
                              dry_run=False, max_iterations=10,
                              budget_hours=10.0, seed=42, task="dyck"):
    print("=== INITIALIZING BAYESIAN OPTIMIZATION (TPE) BASELINE (ARM 3) ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Arm: {arm}")
    print(f"Task Family: {task}")
    print(f"Max Iterations: {max_iterations}")
    print(f"GPU Budget: {budget_hours} Hours")
    print(f"Dry Run Mode: {dry_run}")
    print(f"Random Seed: {seed}\n")

    logger = TraceLogger(
        experiment_id=experiment_id,
        agent_arm=arm,
        total_gpu_budget_hours=budget_hours
    )

    harness_path = os.path.join(os.path.dirname(__file__), "eval_harness.py")
    harness_base_args = ["--task", task] + (["--dry-run"] if dry_run else ["--no-dry-run", "--steps", "50"])

    # Iteration 0: Baseline Evaluation
    print("[Iteration 0] Running Baseline Model Evaluation...")
    base_run = execute_in_sandbox(harness_path, harness_base_args, timeout_sec=60 if dry_run else 300)

    if base_run["status"] != "SUCCESS" or not base_run["metrics"]:
        print(f"Baseline run failed! Stderr: {base_run['stderr_tail']}")
        return

    base_loss = base_run["metrics"]["final_val_loss"]
    print(f"Baseline Validation Cross-Entropy Loss: {base_loss:.4f}\n")
    logger.set_baseline_loss(base_loss)

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

    # Initialize Optuna Study with TPE Sampler
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    total_gpu_sec = base_run["elapsed_seconds"]
    consecutive_divergences = 0
    active_parent_hypo = hypo_0

    for it in range(1, max_iterations + 1):
        if total_gpu_sec / 3600.0 >= budget_hours:
            print(f"[STOP] GPU Budget Exhausted ({total_gpu_sec/3600.0:.2f}h >= {budget_hours}h).")
            break
        if consecutive_divergences >= 3:
            print(f"[STOP] Three-strike divergence condition reached ({consecutive_divergences} consecutive failures).")
            break

        print(f"\n--- [Iteration {it}/{max_iterations}] ---")

        # Ask TPE sampler for candidate configuration
        trial = study.ask()

        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        n_layers = trial.suggest_categorical("n_layers", [2, 4, 6, 8])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        d_model = trial.suggest_categorical("d_model", [128, 256, 384])
        activation = trial.suggest_categorical("activation", ["gelu", "relu", "silu"])
        norm_type = trial.suggest_categorical("norm_type", ["layernorm", "rmsnorm"])
        pos_encoding = trial.suggest_categorical("pos_encoding", ["learned", "sinusoidal", "rotary", "none"])
        ffn_type = trial.suggest_categorical("ffn_type", ["standard", "swiglu"])
        topology = trial.suggest_categorical("topology", ["pre_ln", "post_ln", "parallel"])
        kv_choice = trial.suggest_categorical("n_kv_heads", ["same", "1", "2"])
        if kv_choice == "same":
            n_kv_heads = n_heads
        else:
            kv_val = int(kv_choice)
            n_kv_heads = kv_val if (n_heads % kv_val == 0) else n_heads

        scale_str = trial.suggest_categorical("scale_factor", ["none", "0.25", "0.5", "1.0", "2.0"])
        scale_factor = None if scale_str == "none" else float(scale_str)

        sampled_params = {
            "lr": round(lr, 6),
            "weight_decay": round(weight_decay, 4),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "d_model": d_model,
            "activation": activation,
            "norm_type": norm_type,
            "scale_factor": scale_factor,
            "pos_encoding": pos_encoding,
            "ffn_type": ffn_type,
            "topology": topology
        }
        print(f"[Arm 3 TPE Sampler] Suggested Parameters: {sampled_params}")

        hypo_text = (
            f"[TPE_BAYESIAN] Probabilistic proposal: "
            f"layers={sampled_params['n_layers']}, heads={sampled_params['n_heads']}(kv={sampled_params['n_kv_heads']}), "
            f"d_model={sampled_params['d_model']}, pos={sampled_params['pos_encoding']}, "
            f"ffn={sampled_params['ffn_type']}, top={sampled_params['topology']}."
        )
        hypo_id = logger.add_hypothesis(
            iteration_idx=it,
            text=hypo_text,
            target_component="tpe_kernel_density",
            predicted_delta_loss=0.0,
            parent_hypo_id=active_parent_hypo,
            provider="optuna_tpe_sampler"
        )
        exp_id = logger.add_experiment_spec(it, hypo_id, str(sampled_params))

        # Build sandbox arguments
        run_args = ["--task", task] + (["--dry-run"] if dry_run else ["--no-dry-run", "--steps", "50"])
        run_args.extend(["--lr", str(sampled_params["lr"])])
        run_args.extend(["--weight-decay", str(sampled_params["weight_decay"])])
        run_args.extend(["--n-layers", str(sampled_params["n_layers"])])
        run_args.extend(["--n-heads", str(sampled_params["n_heads"])])
        run_args.extend(["--d-model", str(sampled_params["d_model"])])
        run_args.extend(["--activation", str(sampled_params["activation"])])
        run_args.extend(["--norm-type", str(sampled_params["norm_type"])])
        if sampled_params.get("scale_factor"):
            run_args.extend(["--scale-factor", str(sampled_params["scale_factor"])])
        if sampled_params.get("pos_encoding"):
            run_args.extend(["--pos-encoding", str(sampled_params["pos_encoding"])])
        if sampled_params.get("ffn_type"):
            run_args.extend(["--ffn-type", str(sampled_params["ffn_type"])])
        if sampled_params.get("topology"):
            run_args.extend(["--topology", str(sampled_params["topology"])])
        if sampled_params.get("n_kv_heads"):
            run_args.extend(["--n-kv-heads", str(sampled_params["n_kv_heads"])])

        res = execute_in_sandbox(harness_path, run_args, timeout_sec=60 if dry_run else 300)
        total_gpu_sec += res["elapsed_seconds"]

        if res["status"] == "SUCCESS" and res["metrics"]:
            val_loss = res["metrics"]["final_val_loss"]
            train_loss = res["metrics"]["final_train_loss"]
            rel_gain = round(((base_loss - val_loss) / base_loss) * 100, 2)
            print(f"Result: Validation Loss = {val_loss:.4f} (Train Loss = {train_loss:.4f}, Rel Gain = {rel_gain:+0.2f}%)")

            # Report back to Optuna
            study.tell(trial, val_loss)

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
            study.tell(trial, 999.0)
            val_loss = None
            train_loss = None
            status = "FAILED"
            decision = "REJECT_AND_BACKTRACK"
            consecutive_divergences += 1

        run_id, is_best = logger.add_execution_run(
            iteration_idx=it,
            exp_id=exp_id,
            status=status,
            exit_code=res.get("exit_code", 1),
            gpu_seconds=res["elapsed_seconds"],
            val_loss=val_loss,
            train_loss=train_loss
        )

        refl_text = f"TPE Trial {it} outcome: {status} with decision {decision}."
        logger.add_reflection(
            iteration_idx=it,
            run_id=run_id,
            hypo_id=hypo_id,
            reflection_text=refl_text,
            decision=decision,
            self_detected_failure=(status == "DIVERGED" or status == "FAILED"),
            cycle_sec=res["elapsed_seconds"]
        )

        trace_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "execution_traces",
            f"trace_{experiment_id}.json"
        )
        logger.export_trace(trace_path)

    summary = logger.export_trace(trace_path)
    print("\n=== BAYESIAN OPTIMIZATION (TPE) BASELINE COMPLETE ===")
    print(f"Total Iterations: {summary['summary_statistics']['total_iterations']}")
    print(f"Dead-End Ratio: {summary['summary_statistics']['dead_end_ratio']}")
    print(f"Path to First Improvement: {summary['summary_statistics']['path_to_first_improvement_steps']}")
    print(f"Final Best Loss: {summary['summary_statistics']['final_best_loss']}")
    print(f"Final Relative Gain: {summary['summary_statistics']['final_relative_improvement_pct']}%")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arm 3 Bayesian Optimization (TPE) Baseline")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--task", type=str, default="dyck", choices=["dyck", "fsm", "parity_legacy"])
    args = parser.parse_args()

    run_bayesian_optimization(
        dry_run=args.dry_run,
        max_iterations=args.iterations,
        budget_hours=args.budget_hours,
        seed=args.seed,
        task=args.task
    )
