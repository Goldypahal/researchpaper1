"""
Human Engineering Baseline Interactive CLI (Phase 14).
Provides human researchers/engineers with an identical interface to the LLM agent:
  - Views benchmark task specs and incumbent best validation loss
  - Prompts for technical hypothesis rationale and predicted delta loss
  - Enforces the exact same Level 1 + Level 2 search space bounds
  - Evaluates candidate under the Immutable Evaluator
  - Records think time, trajectory trace, AUC efficiency, and calibration MAE
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluator.immutable_evaluator import ImmutableEvaluator
from search_space import (
    SEARCH_SPACE_SPEC,
    sanitize_configuration,
    get_default_baseline
)
from scientific_stats.statistical_analysis import calculate_auc_search_curve


def run_human_session(
    participant_id: str = "human_expert_01",
    task: str = "dyck",
    seed: int = 42,
    max_iterations: int = 5,
    steps_per_candidate: int = 50,
    interactive: bool = False
):
    print("=====================================================================")
    print(f"HUMAN EXPERT BASELINE PROTOCOL: Participant {participant_id}")
    print(f"Task: {task.upper()} | Seed: {seed} | Horizon: K={max_iterations}")
    print("=====================================================================\n")

    evaluator = ImmutableEvaluator(task=task, seed=seed)
    base_cfg = get_default_baseline()
    base_res = evaluator.train_and_evaluate_candidate(base_cfg, max_steps=steps_per_candidate)
    best_val = base_res["val_loss"]
    best_ood = base_res["ood_loss"]
    cum_gpu_sec = base_res["gpu_seconds"]

    history = [{
        "iteration": 0,
        "config": base_cfg,
        "val_loss": base_res["val_loss"],
        "ood_loss": base_res["ood_loss"],
        "best_val_loss": best_val,
        "gpu_seconds": base_res["gpu_seconds"],
        "cumulative_gpu_sec": cum_gpu_sec,
        "think_time_sec": 0.0,
        "hypothesis": "Step 0 Baseline Reference",
        "predicted_delta_loss": 0.0,
        "actual_delta_loss": 0.0
    }]

    print(f"[Iteration 0] Baseline established: L_val = {best_val:.4f} | L_ood = {best_ood:.4f}\n")

    # METHODOLOGICAL DISTINCTION:
    # If not interactive, this is strictly a 'human_protocol_simulation' (interface smoke test).
    # Real human data is logged only when interactive=True with live human inputs.
    is_simulation = not interactive
    method_label = "human_protocol_simulation" if is_simulation else "human_empirical_subject"
    
    if is_simulation:
        print("[METHODOLOGICAL NOTICE] Running in simulation/smoketest mode.")
        print("  Tag: 'human_protocol_simulation'. NOT empirical human subject data.\n")
    
    # Pre-recorded heuristic trajectory for interface smoke-testing only
    smoke_test_trials = [
        {
            "hypothesis": "[Simulation Smoke-test] Increase depth from 6 to 8 layers with RMSNorm.",
            "predicted_delta_loss": 0.05,
            "modifications": {"n_layers": 8, "norm_type": "rmsnorm", "lr": 5e-4},
            "think_time_sec": 180.0
        },
        {
            "hypothesis": "[Simulation Smoke-test] Switch to Rotary Position Embeddings (RoPE).",
            "predicted_delta_loss": 0.08,
            "modifications": {"pos_encoding": "rotary", "d_model": 256, "n_heads": 4},
            "think_time_sec": 240.0
        },
        {
            "hypothesis": "[Simulation Smoke-test] Adopt SwiGLU feed-forward network.",
            "predicted_delta_loss": 0.04,
            "modifications": {"ffn_type": "swiglu", "d_ff": 1024, "lr": 8e-4},
            "think_time_sec": 150.0
        },
        {
            "hypothesis": "[Simulation Smoke-test] Decrease lr to 2e-4 with weight decay 0.05.",
            "predicted_delta_loss": 0.03,
            "modifications": {"lr": 2e-4, "weight_decay": 0.05},
            "think_time_sec": 120.0
        },
        {
            "hypothesis": "[Simulation Smoke-test] Switch to Pre-LN with 8 attention heads.",
            "predicted_delta_loss": 0.02,
            "modifications": {"n_heads": 8, "d_model": 256, "topology": "pre_ln"},
            "think_time_sec": 90.0
        }
    ]

    for it in range(1, max_iterations + 1):
        if interactive:
            print(f"\n--- [Iteration {it}/{max_iterations}] Interactive Human Decision ---")
            print(f"Current Incumbent Best Validation Loss: {best_val:.4f}")
            print(f"Current Incumbent OOD Loss: {best_ood:.4f}")
            t_think_start = time.time()
            hypo = input("Enter your scientific hypothesis rationale: ").strip()
            pred_delta_str = input("Enter your predicted delta loss reduction (e.g. 0.04): ").strip()
            try:
                pred_delta = float(pred_delta_str)
            except ValueError:
                pred_delta = 0.0
            
            # Interactive parameter selection
            print("Specify architectural modifications (leave blank to keep current):")
            cand_cfg = dict(base_cfg)
            lr_in = input(f"Learning rate [{cand_cfg['lr']}]: ").strip()
            if lr_in: cand_cfg["lr"] = float(lr_in)
            layers_in = input(f"Number of layers [{cand_cfg['n_layers']}]: ").strip()
            if layers_in: cand_cfg["n_layers"] = int(layers_in)
            heads_in = input(f"Attention heads [{cand_cfg['n_heads']}]: ").strip()
            if heads_in: cand_cfg["n_heads"] = int(heads_in)
            pos_in = input(f"Positional encoding [{cand_cfg['pos_encoding']}]: ").strip()
            if pos_in: cand_cfg["pos_encoding"] = pos_in
            ffn_in = input(f"FFN type [{cand_cfg['ffn_type']}]: ").strip()
            if ffn_in: cand_cfg["ffn_type"] = ffn_in
            
            think_time = round(time.time() - t_think_start, 2)
        else:
            trial_info = smoke_test_trials[min(it - 1, len(smoke_test_trials) - 1)]
            hypo = trial_info["hypothesis"]
            pred_delta = trial_info["predicted_delta_loss"]
            think_time = trial_info["think_time_sec"]

            cand_cfg = dict(base_cfg)
            cand_cfg.update(trial_info["modifications"])

        cand_cfg = sanitize_configuration(cand_cfg)

        eval_res = evaluator.train_and_evaluate_candidate(cand_cfg, max_steps=steps_per_candidate)
        val_loss = eval_res["val_loss"]
        ood_loss = eval_res["ood_loss"]
        actual_delta = round(base_res["val_loss"] - val_loss, 6)

        if val_loss < best_val:
            best_val = val_loss
            best_ood = ood_loss

        cum_gpu_sec += eval_res["gpu_seconds"]

        history.append({
            "iteration": it,
            "config": cand_cfg,
            "val_loss": val_loss,
            "ood_loss": ood_loss,
            "best_val_loss": best_val,
            "gpu_seconds": eval_res["gpu_seconds"],
            "cumulative_gpu_sec": round(cum_gpu_sec, 3),
            "think_time_sec": think_time,
            "hypothesis": hypo,
            "predicted_delta_loss": pred_delta,
            "actual_delta_loss": actual_delta
        })

        print(f"  [Iteration {it}] Hypo: {hypo[:50]}... | L_val: {val_loss:.4f} (Best: {best_val:.4f}) | "
              f"OOD: {ood_loss:.4f} | Pred: {pred_delta:+.4f} vs Actual: {actual_delta:+.4f}")

    # Compute AUC & Calibration
    gpu_times = [h["cumulative_gpu_sec"] for h in history]
    best_losses = [h["best_val_loss"] for h in history]
    auc = calculate_auc_search_curve(gpu_times, best_losses)

    pred_actual_pairs = [(h["predicted_delta_loss"], h["actual_delta_loss"]) for h in history[1:]]
    mae = sum(abs(p - a) for p, a in pred_actual_pairs) / len(pred_actual_pairs)

    total_think_time = sum(h["think_time_sec"] for h in history)

    trajectory_result = {
        "participant_id": participant_id,
        "method": method_label,
        "is_simulation": is_simulation,
        "task": task,
        "seed": seed,
        "baseline_val_loss": base_res["val_loss"],
        "best_val_loss": best_val,
        "best_ood_loss": best_ood,
        "improvement_pct": round(((base_res["val_loss"] - best_val) / base_res["val_loss"]) * 100.0, 2),
        "total_gpu_seconds": round(cum_gpu_sec, 3),
        "total_think_time_sec": total_think_time,
        "auc_search_curve": auc,
        "calibration_mae": round(mae, 6),
        "trajectory": history
    }

    out_dir = os.path.dirname(__file__)
    traces_dir = os.path.join(out_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    prefix = "sim" if is_simulation else "empirical"
    trace_path = os.path.join(traces_dir, f"trace_{prefix}_{participant_id}_{task}_seed_{seed}.json")
    with open(trace_path, "w") as f:
        json.dump(trajectory_result, f, indent=2)
    print(f"\n[Saved Trajectory] -> {trace_path}")

    # Update Registry
    reg_path = os.path.abspath(os.path.join(out_dir, "..", "EXPERIMENT_REGISTRY.json"))
    if os.path.exists(reg_path):
        with open(reg_path, "r") as f:
            reg = json.load(f)
        reg["experiments"].append({
            "experiment_id": f"EXP-PHASE14-{prefix.upper()}-{participant_id.upper()}-{task.upper()}-SEED-{seed}",
            "git_commit": "9c475fa",
            "task": task,
            "method": method_label,
            "seed": seed,
            "model": "Human_Subject" if not is_simulation else "Mock_Simulation",
            "budget": f"K={max_iterations}, {cum_gpu_sec:.1f}s compute, {total_think_time/60.0:.1f}m think",
            "dataset_version": f"{task}_v1_0",
            "prompt_version": "N/A",
            "result": {
                "best_val_loss": best_val,
                "best_ood_loss": best_ood,
                "improvement_pct": trajectory_result["improvement_pct"],
                "auc_search_curve": auc,
                "calibration_mae": trajectory_result["calibration_mae"],
                "is_simulation": is_simulation
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "COMPLETED"
        })
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=2)
        print(f"[Updated Registry] -> Human run logged in {reg_path}")

    return trajectory_result


if __name__ == "__main__":
    run_human_session()
