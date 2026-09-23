"""
run_scaleup_n10.py
==================
Scales both Arm 1 (LLM Agent) and Arm 2 (Random Search Baseline) to N=10 independent trials.
Features:
- Reuses clean, verified N=5 runs for both arms (including clean Trial 1).
- Runs 5 new independent trials for Arm 1 (Trials 6-10) using Nemotron-3 Ultra (STRICT mode).
- Runs 5 new independent seeds for Arm 2 (Seeds 1234, 4321, 5555, 8888, 9999).
- Saves incremental checkpoints after each run to handle interruptions.
- Computes Welch's t-test, Mann-Whitney U, Cohen's d across all N=10 per arm.
- Updates benchmark_distribution_report.json and benchmark_distribution_report.md.
"""

import os
import sys
import json
import math
import datetime
import numpy as np
from scipy import stats as scipy_stats

from agent_loop import run_agent_loop
from random_search_baseline import run_random_search

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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"
)
PROVIDER = "nvidia"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "scaleup_n10_checkpoint.json")

# Verified Clean Baseline Runs (N=5 for Arm 1)
INITIAL_ARM1_RESULTS = [
    {"trial": 1, "final_best_loss": 7.0653, "relative_gain_pct": 0.64, "dead_end_ratio": 0.75, "path_to_first_improvement": 0, "status": "SUCCESS"},
    {"trial": 2, "final_best_loss": 7.0519, "relative_gain_pct": 0.82, "dead_end_ratio": 0.5,  "path_to_first_improvement": 2, "status": "SUCCESS"},
    {"trial": 3, "final_best_loss": 7.0098, "relative_gain_pct": 1.42, "dead_end_ratio": 0.5,  "path_to_first_improvement": 2, "status": "SUCCESS"},
    {"trial": 4, "final_best_loss": 7.0653, "relative_gain_pct": 0.64, "dead_end_ratio": 0.75, "path_to_first_improvement": 0, "status": "SUCCESS"},
    {"trial": 5, "final_best_loss": 6.9578, "relative_gain_pct": 2.15, "dead_end_ratio": 0.5,  "path_to_first_improvement": 1, "status": "SUCCESS"},
]

# Verified Baseline Runs (N=5 for Arm 2)
INITIAL_ARM2_RESULTS = [
    {"seed": 42,   "final_best_loss": 6.9918, "status": "SUCCESS"},
    {"seed": 101,  "final_best_loss": 6.9818, "status": "SUCCESS"},
    {"seed": 777,  "final_best_loss": 6.9297, "status": "SUCCESS"},
    {"seed": 999,  "final_best_loss": 6.9496, "status": "SUCCESS"},
    {"seed": 2026, "final_best_loss": 7.0653, "status": "SUCCESS"},
]

TARGET_ARM1_TRIALS = [6, 7, 8, 9, 10]
TARGET_ARM2_SEEDS  = [1234, 4321, 5555, 8888, 9999]


def compute_distribution(vals):
    arr = np.array(vals, dtype=float)
    return {
        "count":     int(len(arr)),
        "mean":      float(np.mean(arr)),
        "std":       float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median":    float(np.median(arr)),
        "iqr":       float(scipy_stats.iqr(arr)) if len(arr) > 1 else 0.0,
        "min_best":  float(np.min(arr)),
        "max_worst": float(np.max(arr)),
    }


def load_or_init_checkpoint():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                data = json.load(f)
            print(f"[CHECKPOINT] Loaded existing checkpoint from {CHECKPOINT_PATH}")
            return data
        except Exception as e:
            print(f"[WARNING] Could not read checkpoint ({e}), initializing fresh.")

    data = {
        "arm1_results": list(INITIAL_ARM1_RESULTS),
        "arm2_results": list(INITIAL_ARM2_RESULTS)
    }
    save_checkpoint(data)
    return data


def save_checkpoint(data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def generate_reports(arm1_results, arm2_results):
    arm1_losses = [r["final_best_loss"] for r in arm1_results if r.get("final_best_loss") is not None]
    arm2_losses = [r["final_best_loss"] for r in arm2_results if r.get("final_best_loss") is not None]

    dist_arm1 = compute_distribution(arm1_losses)
    dist_arm2 = compute_distribution(arm2_losses)

    t_stat, p_val_t = scipy_stats.ttest_ind(arm1_losses, arm2_losses, equal_var=False)
    u_stat, p_val_u = scipy_stats.mannwhitneyu(arm1_losses, arm2_losses, alternative="two-sided")

    n1, n2 = len(arm1_losses), len(arm2_losses)
    std1, std2 = dist_arm1["std"], dist_arm2["std"]
    pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0.0
    cohens_d = (dist_arm1["mean"] - dist_arm2["mean"]) / pooled_std if pooled_std > 0 else 0.0

    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    report = {
        "task": "SyntheticStateTransitionDyck_SmallTransformer",
        "iterations_per_run": 3,
        "sample_size": f"N={len(arm1_losses)} (Arm 1) vs N={len(arm2_losses)} (Arm 2)",
        "provenance_note": (
            "All Arm 1 trials drawn from identical post-fix pipeline (Nemotron-3 Ultra, STRICT mode). "
            "Arm 2 runs drawn from identical uniform/log-uniform random search baseline."
        ),
        "timestamp_utc": now_utc,
        "arm1_agent": {
            "individual_runs": arm1_results,
            "distribution": dist_arm1,
        },
        "arm2_random": {
            "individual_runs": arm2_results,
            "distribution": dist_arm2,
        },
        "statistical_tests": {
            "welch_t_stat": float(t_stat),
            "welch_p_val": float(p_val_t),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p_val": float(p_val_u),
            "cohens_d": float(cohens_d),
            "statistically_significant_05": bool(p_val_t < 0.05),
        },
    }

    # Write JSON report
    json_path = os.path.join(RESULTS_DIR, "benchmark_distribution_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write Markdown report
    md_path = os.path.join(RESULTS_DIR, "benchmark_distribution_report.md")
    with open(md_path, "w") as f:
        f.write("# Empirical Statistical Comparison: Arm 1 (LLM Agent) vs. Arm 2 (Random Search)\n\n")
        f.write(f"> **Sample Size**: N={len(arm1_losses)} per arm | **Date**: {today_str}\n")
        f.write("> **Provenance**: All Arm 1 trials executed via `nvidia/nemotron-3-ultra-550b-a55b` in strict mode.\n")
        f.write("> All Arm 2 trials sampled from the identical 8-dimensional search space.\n\n")

        f.write(f"## 1. Distribution Summary (N={len(arm1_losses)} per arm)\n\n")
        f.write("| Arm | N | Mean Loss | Std Dev | Median | IQR | Min (Best) | Max (Worst) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(
            f"| **Arm 1 (LLM Agent)** | {dist_arm1['count']} | {dist_arm1['mean']:.4f} | "
            f"{dist_arm1['std']:.4f} | {dist_arm1['median']:.4f} | {dist_arm1['iqr']:.4f} | "
            f"**{dist_arm1['min_best']:.4f}** | {dist_arm1['max_worst']:.4f} |\n"
        )
        f.write(
            f"| **Arm 2 (Random Search)** | {dist_arm2['count']} | {dist_arm2['mean']:.4f} | "
            f"{dist_arm2['std']:.4f} | {dist_arm2['median']:.4f} | {dist_arm2['iqr']:.4f} | "
            f"**{dist_arm2['min_best']:.4f}** | {dist_arm2['max_worst']:.4f} |\n\n"
        )

        f.write("## 2. Statistical Significance Testing\n\n")
        f.write(f"- **Welch's Two-Sample t-test**: $t = {t_stat:.4f}$, $p = {p_val_t:.4f}$\n")
        f.write(f"- **Mann-Whitney U Test**: $U = {u_stat:.1f}$, $p = {p_val_u:.4f}$\n")
        f.write(f"- **Effect Size (Cohen's d)**: $d = {cohens_d:.4f}$\n")
        sig_str = "YES" if p_val_t < 0.05 else "NO"
        f.write(f"- **Statistically Significant at $\\alpha=0.05$**: **{sig_str}**\n\n")

        f.write("## 3. Individual Trial Logs\n\n")
        f.write("### Arm 1 (LLM Agent)\n\n")
        f.write("| Trial | Best Val Loss | Rel. Gain % | Dead-End Ratio | Status |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for r in arm1_results:
            loss_str = f"{r['final_best_loss']:.4f}" if r.get("final_best_loss") is not None else "CRASH"
            gain_str = f"{r['relative_gain_pct']:.2f}%" if r.get("relative_gain_pct") is not None else "-"
            der_str  = f"{r['dead_end_ratio']:.2f}" if r.get("dead_end_ratio") is not None else "-"
            f.write(f"| {r['trial']} | {loss_str} | {gain_str} | {der_str} | {r.get('status', 'SUCCESS')} |\n")
        f.write("\n")

        f.write("### Arm 2 (Random Search Baseline)\n\n")
        f.write("| Seed | Best Val Loss | Status |\n")
        f.write("| :---: | :---: | :---: |\n")
        for r in arm2_results:
            loss_str = f"{r['final_best_loss']:.4f}" if r.get("final_best_loss") is not None else "CRASH"
            f.write(f"| {r['seed']} | {loss_str} | {r.get('status', 'SUCCESS')} |\n")
        f.write("\n")

        f.write("## 4. Key Takeaways for Research Paper 1\n\n")
        delta = dist_arm1["mean"] - dist_arm2["mean"]
        direction = "higher (worse)" if delta > 0 else "lower (better)"
        f.write(f"1. **Difference in Means**: Arm 1 validation loss is {abs(delta):.4f} {direction} than Arm 2 on average.\n")
        f.write(f"2. **Statistical Power**: Across N={len(arm1_losses)} trials per arm, the p-value is ${p_val_t:.4f}$. ")
        if p_val_t >= 0.05:
            f.write("The null hypothesis cannot be rejected: LLM-driven inductive reasoning does not produce a statistically significant advantage over random search under compute-bounded (3-iteration / 50-step) regimes.\n")
        else:
            f.write("The difference between arms is statistically significant at $\\alpha = 0.05$.\n")
        f.write(f"3. **Variance Profile**: Random search exhibits a spread with standard deviation $\\sigma={dist_arm2['std']:.4f}$ vs. Agent $\\sigma={dist_arm1['std']:.4f}$.\n")

    return report, json_path, md_path


def main():
    print("=" * 65)
    print("      N=10 BENCHMARK SCALE-UP: ARM 1 vs ARM 2")
    print("=" * 65)

    data = load_or_init_checkpoint()
    arm1_results = data["arm1_results"]
    arm2_results = data["arm2_results"]

    existing_arm1_trials = {r["trial"] for r in arm1_results if r.get("final_best_loss") is not None}
    existing_arm2_seeds  = {r["seed"] for r in arm2_results if r.get("final_best_loss") is not None}

    # -------------------------------------------------------------
    # 1. RUN REMAINING ARM 1 TRIALS (6 through 10)
    # -------------------------------------------------------------
    for trial in TARGET_ARM1_TRIALS:
        if trial in existing_arm1_trials:
            print(f"[SKIP] Arm 1 Trial {trial} already completed.")
            continue

        print(f"\n>>> [ARM 1] RUNNING TRIAL {trial} / 10 (Nemotron-3 Ultra, STRICT)")
        exp_id = f"exp_arm1_trial_{trial}"
        try:
            summary = run_agent_loop(
                experiment_id=exp_id,
                arm="Arm1_AutonomousAgent",
                dry_run=False,
                max_iterations=3,
                budget_hours=1.0,
                llm_provider=PROVIDER,
                llm_model=MODEL,
                api_key=API_KEY,
                allow_simulation=False,
            )
            s = summary["summary_statistics"]
            result_item = {
                "trial": trial,
                "experiment_id": exp_id,
                "final_best_loss": s["final_best_loss"],
                "relative_gain_pct": s["final_relative_improvement_pct"],
                "dead_end_ratio": s["dead_end_ratio"],
                "path_to_first_improvement": s["path_to_first_improvement_steps"],
                "status": "SUCCESS",
            }
            print(f"[SUCCESS] Arm 1 Trial {trial}: Best Val Loss = {s['final_best_loss']:.4f}")
        except Exception as e:
            print(f"[ERROR] Arm 1 Trial {trial} failed: {e}")
            result_item = {
                "trial": trial,
                "experiment_id": exp_id,
                "final_best_loss": None,
                "status": f"FAILED: {e}"
            }

        arm1_results.append(result_item)
        data["arm1_results"] = arm1_results
        save_checkpoint(data)
        generate_reports(arm1_results, arm2_results)

    # -------------------------------------------------------------
    # 2. RUN REMAINING ARM 2 SEEDS
    # -------------------------------------------------------------
    for seed in TARGET_ARM2_SEEDS:
        if seed in existing_arm2_seeds:
            print(f"[SKIP] Arm 2 Seed {seed} already completed.")
            continue

        print(f"\n>>> [ARM 2] RUNNING RANDOM SEARCH SEED {seed} (5/5 of Scale-up)")
        exp_id = f"exp_arm2_seed_{seed}"
        try:
            summary = run_random_search(
                experiment_id=exp_id,
                arm="Arm2_RandomSearchBaseline",
                dry_run=False,
                max_iterations=3,
                budget_hours=1.0,
                seed=seed,
            )
            s = summary["summary_statistics"]
            result_item = {
                "seed": seed,
                "experiment_id": exp_id,
                "final_best_loss": s["final_best_loss"],
                "status": "SUCCESS",
            }
            print(f"[SUCCESS] Arm 2 Seed {seed}: Best Val Loss = {s['final_best_loss']:.4f}")
        except Exception as e:
            print(f"[ERROR] Arm 2 Seed {seed} failed: {e}")
            result_item = {
                "seed": seed,
                "experiment_id": exp_id,
                "final_best_loss": None,
                "status": f"FAILED: {e}"
            }

        arm2_results.append(result_item)
        data["arm2_results"] = arm2_results
        save_checkpoint(data)
        generate_reports(arm1_results, arm2_results)

    # -------------------------------------------------------------
    # FINAL STATS & DISPLAY
    # -------------------------------------------------------------
    report, json_path, md_path = generate_reports(arm1_results, arm2_results)
    d1 = report["arm1_agent"]["distribution"]
    d2 = report["arm2_random"]["distribution"]
    st = report["statistical_tests"]

    print("\n" + "=" * 65)
    print(f"       FINAL N=10 BENCHMARK SCALE-UP COMPLETED")
    print("=" * 65)
    print(f"  Arm 1 (LLM Agent, N={d1['count']}):")
    print(f"    Mean: {d1['mean']:.4f} | Std: {d1['std']:.4f} | Median: {d1['median']:.4f} | Best: {d1['min_best']:.4f}")
    print(f"  Arm 2 (Random Search, N={d2['count']}):")
    print(f"    Mean: {d2['mean']:.4f} | Std: {d2['std']:.4f} | Median: {d2['median']:.4f} | Best: {d2['min_best']:.4f}")
    print(f"  Welch's t-test: t = {st['welch_t_stat']:.4f}, p = {st['welch_p_val']:.4f}")
    print(f"  Mann-Whitney U: U = {st['mann_whitney_u']:.1f}, p = {st['mann_whitney_p_val']:.4f}")
    print(f"  Cohen's d:      d = {st['cohens_d']:.4f}")
    print(f"  Statistically Significant (alpha=0.05): {st['statistically_significant_05']}")
    print(f"\n  Reports saved to:\n    {json_path}\n    {md_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
