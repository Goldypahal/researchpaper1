"""
Multi-Seed Statistical Distribution Benchmark Runner.
Executes multi-seed runs for Arm 2 (Random Search) and multi-trial runs for Arm 1 (Autonomous Agent)
to characterize the empirical distribution of outcomes, compute variance, and perform
formal statistical hypothesis testing (Welch's t-test and Mann-Whitney U test).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from scipy import stats

from random_search_baseline import run_random_search
from agent_loop import run_agent_loop


def compute_distribution_metrics(values):
    vals = np.array(values, dtype=float)
    return {
        "count": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "median": float(np.median(vals)),
        "iqr": float(stats.iqr(vals)) if len(vals) > 1 else 0.0,
        "min_best": float(np.min(vals)),
        "max_worst": float(np.max(vals))
    }


def run_benchmark(arm2_seeds=[42, 101, 777, 999, 2026],
                  arm1_trials=5,
                  iterations=3,
                  api_key=None,
                  provider="nvidia",
                  model="nvidia/nemotron-3-ultra-550b-a55b",
                  run_arm1=True,
                  run_arm2=True):
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    arm2_results = []
    arm1_results = []

    # 1. Execute Arm 2 (Random Search across Seeds)
    if run_arm2:
        print(f"\n=======================================================")
        print(f"RUNNING ARM 2 (RANDOM SEARCH) ACROSS {len(arm2_seeds)} SEEDS")
        print(f"Seeds: {arm2_seeds} | Iterations per seed: {iterations}")
        print(f"=======================================================\n")

        for idx, seed in enumerate(arm2_seeds, 1):
            exp_id = f"exp_arm2_seed_{seed}"
            print(f"\n>>> [Arm 2 Seed {idx}/{len(arm2_seeds)}: seed={seed}] Starting...")
            try:
                summary = run_random_search(
                    experiment_id=exp_id,
                    arm="Arm2_RandomSearchBaseline",
                    dry_run=False,
                    max_iterations=iterations,
                    budget_hours=1.0,
                    seed=seed
                )
                stats_data = summary["summary_statistics"]
                arm2_results.append({
                    "seed": seed,
                    "final_best_loss": stats_data["final_best_loss"],
                    "relative_gain_pct": stats_data["final_relative_improvement_pct"],
                    "dead_end_ratio": stats_data["dead_end_ratio"],
                    "path_to_first_improvement": stats_data["path_to_first_improvement_steps"],
                    "status": "SUCCESS"
                })
            except Exception as e:
                print(f"[ERROR] Arm 2 seed {seed} failed: {e}")
                arm2_results.append({
                    "seed": seed,
                    "final_best_loss": None,
                    "relative_gain_pct": None,
                    "status": f"FAILED: {e}"
                })

    # 2. Execute Arm 1 (Autonomous Agent across Trials)
    if run_arm1:
        print(f"\n=======================================================")
        print(f"RUNNING ARM 1 (AUTONOMOUS AGENT) ACROSS {arm1_trials} TRIALS")
        print(f"Provider: {provider} | Model: {model} | Iterations: {iterations}")
        print(f"=======================================================\n")

        for trial in range(1, arm1_trials + 1):
            exp_id = f"exp_arm1_trial_{trial}"
            print(f"\n>>> [Arm 1 Trial {trial}/{arm1_trials}] Starting...")
            try:
                summary = run_agent_loop(
                    experiment_id=exp_id,
                    arm="Arm1_AutonomousAgent",
                    dry_run=False,
                    max_iterations=iterations,
                    budget_hours=1.0,
                    llm_provider=provider,
                    llm_model=model,
                    api_key=api_key,
                    allow_simulation=False
                )
                stats_data = summary["summary_statistics"]
                arm1_results.append({
                    "trial": trial,
                    "final_best_loss": stats_data["final_best_loss"],
                    "relative_gain_pct": stats_data["final_relative_improvement_pct"],
                    "dead_end_ratio": stats_data["dead_end_ratio"],
                    "path_to_first_improvement": stats_data["path_to_first_improvement_steps"],
                    "status": "SUCCESS"
                })
            except Exception as e:
                print(f"[ERROR] Arm 1 trial {trial} failed: {e}")
                arm1_results.append({
                    "trial": trial,
                    "final_best_loss": None,
                    "relative_gain_pct": None,
                    "status": f"FAILED: {e}"
                })

    # 3. Statistical Analysis & Comparison
    arm2_valid_losses = [r["final_best_loss"] for r in arm2_results if r.get("final_best_loss") is not None]
    arm1_valid_losses = [r["final_best_loss"] for r in arm1_results if r.get("final_best_loss") is not None]

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "SyntheticStateTransitionDyck_SmallTransformer",
        "iterations_per_run": iterations,
        "arm2_random_search": {
            "trials_attempted": len(arm2_seeds),
            "trials_successful": len(arm2_valid_losses),
            "individual_runs": arm2_results,
            "distribution": compute_distribution_metrics(arm2_valid_losses) if arm2_valid_losses else None
        },
        "arm1_autonomous_agent": {
            "trials_attempted": arm1_trials,
            "trials_successful": len(arm1_valid_losses),
            "individual_runs": arm1_results,
            "distribution": compute_distribution_metrics(arm1_valid_losses) if arm1_valid_losses else None
        }
    }

    # Hypothesis Testing (if both arms have >= 2 valid runs)
    if len(arm1_valid_losses) >= 2 and len(arm2_valid_losses) >= 2:
        t_stat, p_val_t = stats.ttest_ind(arm1_valid_losses, arm2_valid_losses, equal_var=False)
        u_stat, p_val_u = stats.mannwhitneyu(arm1_valid_losses, arm2_valid_losses, alternative="two-sided")
        
        # Cohen's d: (mean1 - mean2) / pooled_std
        mean1, mean2 = np.mean(arm1_valid_losses), np.mean(arm2_valid_losses)
        std1, std2 = np.std(arm1_valid_losses, ddof=1), np.std(arm2_valid_losses, ddof=1)
        pooled_std = math.sqrt(((len(arm1_valid_losses)-1)*std1**2 + (len(arm2_valid_losses)-1)*std2**2) / (len(arm1_valid_losses) + len(arm2_valid_losses) - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

        report["statistical_comparison"] = {
            "welch_t_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_val_t)
            },
            "mann_whitney_u_test": {
                "u_statistic": float(u_stat),
                "p_value": float(p_val_u)
            },
            "cohens_d": float(cohens_d),
            "statistically_significant_p05": bool(p_val_t < 0.05)
        }

    # Save JSON Report
    report_json_path = os.path.join(results_dir, "benchmark_distribution_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save Markdown Report
    report_md_path = os.path.join(results_dir, "benchmark_distribution_report.md")
    with open(report_md_path, "w") as f:
        f.write("# Empirical Distribution Benchmark: Arm 1 (Agent) vs. Arm 2 (Random Search)\n\n")
        f.write(f"- **Task**: SyntheticStateTransitionDyck (SmallTransformerLM, 50 steps/eval)\n")
        f.write(f"- **Iterations Per Run**: {iterations}\n")
        f.write(f"- **Date**: {report['timestamp']}\n\n")
        
        f.write("## 1. Distribution Summary\n\n")
        f.write("| Arm | N | Mean Loss | Std Dev | Median Loss | IQR | Best Loss | Worst Loss |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        if report["arm1_autonomous_agent"]["distribution"]:
            d1 = report["arm1_autonomous_agent"]["distribution"]
            f.write(f"| **Arm 1 (Agent)** | {d1['count']} | {d1['mean']:.4f} | {d1['std']:.4f} | {d1['median']:.4f} | {d1['iqr']:.4f} | **{d1['min_best']:.4f}** | {d1['max_worst']:.4f} |\n")
        if report["arm2_random_search"]["distribution"]:
            d2 = report["arm2_random_search"]["distribution"]
            f.write(f"| **Arm 2 (Random)** | {d2['count']} | {d2['mean']:.4f} | {d2['std']:.4f} | {d2['median']:.4f} | {d2['iqr']:.4f} | **{d2['min_best']:.4f}** | {d2['max_worst']:.4f} |\n")

        if "statistical_comparison" in report:
            sc = report["statistical_comparison"]
            f.write("\n## 2. Formal Hypothesis Testing\n\n")
            f.write(f"- **Welch's t-test**: $t = {sc['welch_t_test']['t_statistic']:.4f}$, $p = {sc['welch_t_test']['p_value']:.4f}$\n")
            f.write(f"- **Mann-Whitney U**: $U = {sc['mann_whitney_u_test']['u_statistic']:.4f}$, $p = {sc['mann_whitney_u_test']['p_value']:.4f}$\n")
            f.write(f"- **Cohen's d**: {sc['cohens_d']:.4f}\n")
            f.write(f"- **Statistically Significant ($\alpha=0.05$)**: **{'YES' if sc['statistically_significant_p05'] else 'NO'}**\n")

    print("\n=======================================================")
    print("BENCHMARK EXECUTION COMPLETE")
    print(f"Saved: {report_json_path}")
    print(f"Saved: {report_md_path}")
    print("=======================================================\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Distribution Benchmark")
    parser.add_argument("--arm2-only", action="store_true", help="Run only Arm 2 seeds")
    parser.add_argument("--arm1-only", action="store_true", help="Run only Arm 1 trials")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 777, 999, 2026])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--provider", type=str, default="nvidia")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    run_arm1 = not args.arm2_only
    run_arm2 = not args.arm1_only

    run_benchmark(
        arm2_seeds=args.seeds,
        arm1_trials=args.trials,
        iterations=args.iterations,
        api_key=args.api_key,
        provider=args.provider,
        model=args.model,
        run_arm1=run_arm1,
        run_arm2=run_arm2
    )
