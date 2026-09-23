"""
Executes Trials 2, 3, 4, 5 of Arm 1 (Autonomous Agent) with Nemotron-3 Ultra,
compiles all 5 trials of Arm 1 alongside all 5 seeds of Arm 2, and performs
formal statistical significance testing (Welch's t-test, Mann-Whitney U, Cohen's d).
"""

import os
import json
import math
import numpy as np
from scipy import stats
from agent_loop import run_agent_loop


def compute_distribution(vals):
    arr = np.array(vals, dtype=float)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(stats.iqr(arr)) if len(arr) > 1 else 0.0,
        "min_best": float(np.min(arr)),
        "max_worst": float(np.max(arr))
    }


def main():
    api_key = "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"
    provider = "nvidia"
    model = "nvidia/nemotron-3-ultra-550b-a55b"
    
    # Arm 1 Trials
    # Trial 1 is already recorded from earlier run:
    arm1_trial_1_loss = 7.0294
    arm1_results = [
        {"trial": 1, "final_best_loss": arm1_trial_1_loss, "status": "SUCCESS"}
    ]

    for trial in [2, 3, 4, 5]:
        print(f"\n==========================================")
        print(f"RUNNING ARM 1 (AUTONOMOUS AGENT) TRIAL {trial}/5")
        print(f"==========================================\n")
        exp_id = f"exp_arm1_trial_{trial}"
        try:
            summary = run_agent_loop(
                experiment_id=exp_id,
                arm="Arm1_AutonomousAgent",
                dry_run=False,
                max_iterations=3,
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
            print(f"[ERROR] Trial {trial} failed: {e}")
            arm1_results.append({
                "trial": trial,
                "final_best_loss": None,
                "status": f"FAILED: {e}"
            })

    # Arm 2 Verified 5-Seed Results
    arm2_results = [
        {"seed": 42, "final_best_loss": 6.9918, "status": "SUCCESS"},
        {"seed": 101, "final_best_loss": 6.9818, "status": "SUCCESS"},
        {"seed": 777, "final_best_loss": 6.9297, "status": "SUCCESS"},
        {"seed": 999, "final_best_loss": 6.9496, "status": "SUCCESS"},
        {"seed": 2026, "final_best_loss": 7.0653, "status": "SUCCESS"}
    ]

    arm1_losses = [r["final_best_loss"] for r in arm1_results if r["final_best_loss"] is not None]
    arm2_losses = [r["final_best_loss"] for r in arm2_results if r["final_best_loss"] is not None]

    dist_arm1 = compute_distribution(arm1_losses)
    dist_arm2 = compute_distribution(arm2_losses)

    # Statistical significance testing
    t_stat, p_val_t = stats.ttest_ind(arm1_losses, arm2_losses, equal_var=False)
    u_stat, p_val_u = stats.mannwhitneyu(arm1_losses, arm2_losses, alternative="two-sided")
    
    std1, std2 = dist_arm1["std"], dist_arm2["std"]
    pooled_std = math.sqrt(((len(arm1_losses)-1)*std1**2 + (len(arm2_losses)-1)*std2**2) / (len(arm1_losses) + len(arm2_losses) - 2))
    cohens_d = (dist_arm1["mean"] - dist_arm2["mean"]) / pooled_std if pooled_std > 0 else 0.0

    report = {
        "task": "SyntheticStateTransitionDyck_SmallTransformer",
        "iterations_per_run": 3,
        "arm1_agent": {
            "individual_runs": arm1_results,
            "distribution": dist_arm1
        },
        "arm2_random": {
            "individual_runs": arm2_results,
            "distribution": dist_arm2
        },
        "statistical_tests": {
            "welch_t_stat": float(t_stat),
            "welch_p_val": float(p_val_t),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p_val": float(p_val_u),
            "cohens_d": float(cohens_d),
            "statistically_significant_05": bool(p_val_t < 0.05)
        }
    }

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results")
    os.makedirs(results_dir, exist_ok=True)
    report_json_path = os.path.join(results_dir, "benchmark_distribution_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)

    report_md_path = os.path.join(results_dir, "benchmark_distribution_report.md")
    with open(report_md_path, "w") as f:
        f.write("# Empirical Statistical Comparison: Arm 1 (LLM Agent) vs. Arm 2 (Random Search)\n\n")
        f.write("## 1. Distribution Summary (N=5 per arm)\n\n")
        f.write("| Arm | N | Mean Loss | Std Dev | Median Loss | IQR | Best Loss | Worst Loss |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Arm 1 (Agent)** | {dist_arm1['count']} | {dist_arm1['mean']:.4f} | {dist_arm1['std']:.4f} | {dist_arm1['median']:.4f} | {dist_arm1['iqr']:.4f} | **{dist_arm1['min_best']:.4f}** | {dist_arm1['max_worst']:.4f} |\n")
        f.write(f"| **Arm 2 (Random)** | {dist_arm2['count']} | {dist_arm2['mean']:.4f} | {dist_arm2['std']:.4f} | {dist_arm2['median']:.4f} | {dist_arm2['iqr']:.4f} | **{dist_arm2['min_best']:.4f}** | {dist_arm2['max_worst']:.4f} |\n\n")
        
        f.write("## 2. Statistical Significance Testing\n\n")
        f.write(f"- **Welch's Two-Sample t-test**: $t = {t_stat:.4f}$, $p = {p_val_t:.4f}$\n")
        f.write(f"- **Mann-Whitney U Test**: $U = {u_stat:.4f}$, $p = {p_val_u:.4f}$\n")
        f.write(f"- **Effect Size (Cohen's d)**: $d = {cohens_d:.4f}$\n")
        f.write(f"- **Statistically Significant at $\\alpha=0.05$**: **{'YES' if p_val_t < 0.05 else 'NO'}**\n")

    print("\n==========================================")
    print("ALL 5 TRIALS OF ARM 1 & ARM 2 COMPLETE")
    print(f"Saved: {report_json_path}")
    print(f"Saved: {report_md_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
