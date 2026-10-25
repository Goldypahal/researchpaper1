#!/usr/bin/env python3
"""
scripts/measure_noise_and_fidelity.py

Executes Phase 4: Rigorous Noise Floor & Training Fidelity Measurement.
  1. Retrains baseline config + 5 random configs across 10 random seeds at 50 steps and 200 steps.
  2. Measures within-config retraining variance (sigma_retrain).
  3. Measures Spearman rank correlation (rho) between candidate rankings at 50 vs 200 steps.
  4. Computes power analysis: minimum seed count N required for 80% statistical power at alpha=0.01.
"""

import os
import sys
import json
import time
import math
import random
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluator.immutable_evaluator import ImmutableEvaluator
from search_space import get_default_baseline, sample_random_candidate

def measure_noise_and_fidelity(
    tasks: List[str] = ["dyck", "fsm"],
    num_random_configs: int = 5,
    num_seeds: int = 10,
    steps_short: int = 50,
    steps_long: int = 200,
    output_report: str = "noise_and_fidelity_report.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE 4: NOISE FLOOR, RANK CORRELATION, AND POWER ANALYSIS")
    print("=" * 70)

    report = {}

    for task in tasks:
        print(f"\n>>> Benchmarking Task: {task.upper()} <<<")
        evaluator = ImmutableEvaluator(task=task, seed=42)
        rng = random.Random(42)

        # Build candidate pool: baseline + num_random_configs
        configs = [("baseline", get_default_baseline())]
        for i in range(num_random_configs):
            configs.append((f"rand_cfg_{i+1}", sample_random_candidate(rng)))

        results_short = {name: [] for name, _ in configs}
        results_long = {name: [] for name, _ in configs}

        print(f"Evaluating {len(configs)} configs across {num_seeds} seeds at {steps_short} & {steps_long} steps...")

        for name, cfg in configs:
            for s in range(num_seeds):
                t_seed = 1000 + s
                # Short training (50 steps)
                res_50 = evaluator.train_and_evaluate_candidate(
                    config=cfg,
                    max_steps=steps_short,
                    train_seed=t_seed
                )
                results_short[name].append(res_50["val_loss"])

                # Long training (200 steps)
                res_200 = evaluator.train_and_evaluate_candidate(
                    config=cfg,
                    max_steps=steps_long,
                    train_seed=t_seed
                )
                results_long[name].append(res_200["val_loss"])

        # Compute within-config standard deviation (retraining noise floor)
        std_short = [np.std(losses, ddof=1) for losses in results_short.values()]
        std_long = [np.std(losses, ddof=1) for losses in results_long.values()]
        sigma_retrain_50 = float(np.mean(std_short))
        sigma_retrain_200 = float(np.mean(std_long))

        # Compute mean losses per config for rank agreement
        mean_losses_50 = [np.mean(results_short[name]) for name, _ in configs]
        mean_losses_200 = [np.mean(results_long[name]) for name, _ in configs]

        spearman_rho, spearman_pval = stats.spearmanr(mean_losses_50, mean_losses_200)

        # Statistical Power Analysis for paired Wilcoxon / t-test:
        # Minimum effect size of interest delta = 0.02 nats
        # N = (z_{1-alpha/2} + z_{1-beta})^2 * (sigma^2 / delta^2)
        delta_mde = 0.02
        z_alpha = 2.576  # alpha = 0.01 (two-sided)
        z_beta = 0.842   # power = 80% (beta = 0.20)
        n_needed_50 = math.ceil(((z_alpha + z_beta) ** 2) * (2 * (sigma_retrain_50 ** 2)) / (delta_mde ** 2))
        n_needed_200 = math.ceil(((z_alpha + z_beta) ** 2) * (2 * (sigma_retrain_200 ** 2)) / (delta_mde ** 2))

        print(f"\nTask {task.upper()} Results:")
        print(f"  Retrain Noise Floor (50 steps):  sigma = {sigma_retrain_50:.4f} nats")
        print(f"  Retrain Noise Floor (200 steps): sigma = {sigma_retrain_200:.4f} nats")
        print(f"  Ranking Agreement (Spearman):    rho   = {spearman_rho:.4f} (p = {spearman_pval:.4f})")
        print(f"  Power Analysis (MDE = 0.02 nats, alpha=0.01, 80% power):")
        print(f"    Required Seeds at 50 steps:  N = {n_needed_50}")
        print(f"    Required Seeds at 200 steps: N = {n_needed_200}")

        report[task] = {
            "sigma_retrain_50": round(sigma_retrain_50, 6),
            "sigma_retrain_200": round(sigma_retrain_200, 6),
            "spearman_rho": round(float(spearman_rho), 4),
            "spearman_pval": round(float(spearman_pval), 4),
            "n_required_50": n_needed_50,
            "n_required_200": n_needed_200,
            "recommendation": "50 steps valid as early-training proxy" if spearman_rho >= 0.80 else "200 steps recommended"
        }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_report}")
    return report

if __name__ == "__main__":
    measure_noise_and_fidelity()
