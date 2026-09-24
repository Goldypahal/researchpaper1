"""
Comparative Search Benchmark Runner.
Executes systematic search comparisons across:
- Arm 2: Uniform Random Search
- Arm 3: Bayesian Optimization (Optuna TPE)
- Arm 4: Evolutionary Search (Genetic Algorithm)
- Arm 1: LLM Agent (optional, simulated or live)

Generates comparative trajectory metrics, search efficiency statistics,
and exports benchmark_comparative_report.json and .md.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import argparse
import numpy as np

from random_search_baseline import run_random_search
from bayesian_opt_baseline import run_bayesian_optimization
from evolutionary_baseline import run_evolutionary_search
from agent_loop import run_agent_loop

SEEDS_DEFAULT = [42, 101, 777, 1234, 2026]

def run_benchmark(methods=["random", "tpe", "evolutionary"],
                  iterations=5, seeds=SEEDS_DEFAULT,
                  task="dyck", dry_run=False, budget_hours=10.0,
                  allow_simulation=True):
    print("=====================================================================")
    print("     AUTONOMOUS MODEL SEARCH BENCHMARK: COMPARATIVE RUNNER           ")
    print("=====================================================================")
    print(f"Task Family: {task}")
    print(f"Iterations (K): {iterations}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Dry Run Mode: {dry_run}\n")

    results_by_method = {m: [] for m in methods}

    for method in methods:
        print(f"\n==================== RUNNING METHOD: {method.upper()} ====================")
        for seed_idx, seed in enumerate(seeds):
            exp_id = f"cmp_{task}_{method}_seed_{seed}"
            print(f"\n>>> [{method.upper()}] Run {seed_idx+1}/{len(seeds)} (Seed: {seed}) <<<")

            if method == "random":
                summary = run_random_search(
                    experiment_id=exp_id,
                    dry_run=dry_run,
                    max_iterations=iterations,
                    budget_hours=budget_hours,
                    seed=seed,
                    task=task
                )
            elif method == "tpe":
                summary = run_bayesian_optimization(
                    experiment_id=exp_id,
                    dry_run=dry_run,
                    max_iterations=iterations,
                    budget_hours=budget_hours,
                    seed=seed,
                    task=task
                )
            elif method == "evolutionary":
                summary = run_evolutionary_search(
                    experiment_id=exp_id,
                    dry_run=dry_run,
                    max_iterations=iterations,
                    population_size=max(2, min(4, iterations // 2)),
                    budget_hours=budget_hours,
                    seed=seed,
                    task=task
                )
            elif method == "agent":
                summary = run_agent_loop(
                    experiment_id=exp_id,
                    dry_run=dry_run,
                    max_iterations=iterations,
                    budget_hours=budget_hours,
                    allow_simulation=allow_simulation,
                    task=task
                )
            else:
                print(f"Unknown method: {method}")
                continue

            if summary and "summary_statistics" in summary:
                stats = summary["summary_statistics"]
                results_by_method[method].append({
                    "seed": seed,
                    "baseline_loss": stats.get("baseline_loss"),
                    "final_best_loss": stats.get("final_best_loss"),
                    "final_relative_gain_pct": stats.get("final_relative_improvement_pct"),
                    "dead_end_ratio": stats.get("dead_end_ratio"),
                    "path_to_first_improvement": stats.get("path_to_first_improvement_steps"),
                    "total_gpu_hours": stats.get("total_gpu_hours_used")
                })

    # Statistical Aggregation
    report = {
        "benchmark_metadata": {
            "task": task,
            "iterations_per_run": iterations,
            "seeds": seeds,
            "dry_run": dry_run,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "method_summaries": {}
    }

    for method, runs in results_by_method.items():
        if not runs:
            continue
        losses = [r["final_best_loss"] for r in runs if r["final_best_loss"] is not None]
        gains = [r["final_relative_gain_pct"] for r in runs if r["final_relative_gain_pct"] is not None]
        dead_ends = [r["dead_end_ratio"] for r in runs if r["dead_end_ratio"] is not None]
        first_steps = [r["path_to_first_improvement"] for r in runs if r["path_to_first_improvement"] is not None]

        report["method_summaries"][method] = {
            "num_runs": len(runs),
            "loss_mean": round(float(np.mean(losses)), 4) if losses else None,
            "loss_std": round(float(np.std(losses)), 4) if losses else None,
            "loss_min": round(float(np.min(losses)), 4) if losses else None,
            "loss_max": round(float(np.max(losses)), 4) if losses else None,
            "gain_mean_pct": round(float(np.mean(gains)), 2) if gains else None,
            "gain_std_pct": round(float(np.std(gains)), 2) if gains else None,
            "dead_end_ratio_mean": round(float(np.mean(dead_ends)), 3) if dead_ends else None,
            "mean_path_to_first_improvement": round(float(np.mean(first_steps)), 2) if first_steps else None,
            "individual_runs": runs
        }

    # Save JSON report
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"comparative_search_benchmark_{task}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_path = os.path.join(out_dir, f"comparative_search_benchmark_{task}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Comparative Search Benchmark Report: Task `{task}`\n\n")
        f.write(f"- **Horizon (K)**: {iterations} iterations\n")
        f.write(f"- **Seeds Tested**: {seeds}\n")
        f.write(f"- **Timestamp**: {report['benchmark_metadata']['timestamp']}\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Method | Mean Loss | Std Loss | Best Loss | Mean Gain (%) | Dead-End Ratio | Path to First Imp. |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for method, s in report["method_summaries"].items():
            f.write(
                f"| **{method.upper()}** | {s['loss_mean']} | ±{s['loss_std']} | {s['loss_min']} | "
                f"{s['gain_mean_pct']}% | {s['dead_end_ratio_mean']} | {s['mean_path_to_first_improvement']} |\n"
            )
        f.write("\n")

    print("\n==================== BENCHMARK EXECUTION COMPLETE ====================")
    print(f"JSON Report written to: {json_path}")
    print(f"Markdown Report written to: {md_path}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparative Search Benchmark Runner")
    parser.add_argument("--methods", nargs="+", default=["random", "tpe", "evolutionary"],
                        help="Search methods to benchmark (random, tpe, evolutionary, agent)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of search iterations per trajectory")
    parser.add_argument("--task", type=str, default="dyck", choices=["dyck", "fsm", "parity_legacy"])
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=False)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 777])
    args = parser.parse_args()

    run_benchmark(
        methods=args.methods,
        iterations=args.iterations,
        seeds=args.seeds,
        task=args.task,
        dry_run=args.dry_run
    )
