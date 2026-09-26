"""
Main Comparative Experiment Orchestrator (Phase 8).
Executes matched-compute comparative optimization sweeps across:
  Methods: Random, TPE, Evolutionary (GA), LLM Agent
  Tasks: Dyck-4, Hidden FSM
  Replications: Multiple independent seeds

Outputs:
  - Trajectory JSONs in v0.3-comparative-search/traces/
  - Aggregated comparative results in v0.3-comparative-search/comparative_search_results.json
  - Markdown report in v0.3-comparative-search/comparative_search_report.md
  - Full audit logging in EXPERIMENT_REGISTRY.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_comparative_search import ComparativeSearchBenchmark
from scientific_stats.statistical_analysis import (
    summarize_distribution,
    compare_two_search_arms,
    holm_bonferroni_correction
)


def run_comparative_experiment(
    tasks: List[str] = ["dyck", "fsm"],
    methods: List[str] = ["random", "tpe", "evolutionary", "llm"],
    seeds: List[int] = [42, 101, 202, 303, 404],
    max_iterations: int = 10,
    steps_per_candidate: int = 25,
    eval_timeout: float = 300.0,
    llm_provider: str = None,
    llm_model: str = None,
    fresh_run: bool = True
):
    print("=====================================================================", flush=True)
    print("PHASE 8: MATCHED-COMPUTE COMPARATIVE SEARCH BENCHMARK", flush=True)
    print(f"Tasks: {tasks}", flush=True)
    print(f"Methods: {methods}", flush=True)
    print(f"Seeds: {seeds} (N={len(seeds)} replications per arm)", flush=True)
    print(f"Horizon: K={max_iterations} iterations, {steps_per_candidate} steps per candidate", flush=True)
    print(f"Mode: {'FRESH RUN (clean un-cached trajectories)' if fresh_run else 'RESUME (caching enabled)'}", flush=True)
    if "llm" in methods:
        print(f"LLM Provider: {llm_provider}", flush=True)
        print(f"LLM Model: {llm_model}", flush=True)
    print("=====================================================================\n", flush=True)

    out_dir = os.path.dirname(__file__)
    traces_dir = os.path.join(out_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    registry_entries = []
    all_trajectories = {t: {m: [] for m in methods} for t in tasks}

    for task in tasks:
        print(f"\n################### TASK: {task.upper()} ###################", flush=True)
        for method in methods:
            print(f"\n--> Running Search Arm: [{method.upper()}] on {task.upper()}", flush=True)
            for s in seeds:
                trace_file = os.path.join(traces_dir, f"trace_{task}_{method}_seed_{s}.json")
                if not fresh_run and os.path.exists(trace_file) and os.path.getsize(trace_file) > 100:
                    try:
                        with open(trace_file, "r") as tf:
                            cached_res = json.load(tf)
                        if len(cached_res.get("trajectory", [])) >= max_iterations:
                            # Integrity verification: reject cached trace if it contains simulated LLM proposals
                            is_contaminated = False
                            if method == "llm":
                                for step in cached_res.get("trajectory", []):
                                    prov = step.get("provider") or step.get("proposal_metadata", {}).get("provider")
                                    if prov == "adaptive_simulation":
                                        is_contaminated = True
                                        break
                            if is_contaminated:
                                print(f"  [Seed {s:3d}] Cached trace contains scripted simulation data. Discarding cache to enforce REAL LLM reasoning.", flush=True)
                            else:
                                all_trajectories[task][method].append(cached_res)
                                auc_val = cached_res.get("auc_normalized_gain", cached_res.get("auc_search_curve", 0.0))
                                print(f"  [Seed {s:3d}] [RESUMED FROM DISK] Base: {cached_res['baseline_val_loss']:.4f} -> Best: {cached_res['best_val_loss']:.4f} "
                                      f"(Gain: {cached_res['improvement_pct']:+.2f}%) | OOD: {cached_res['best_ood_loss']:.4f} | AUC Gain: {auc_val:.4f}", flush=True)
                                continue
                    except Exception as e:
                        print(f"  [Seed {s:3d}] Cache read failed ({e}), re-evaluating...", flush=True)

                t0 = time.time()
                bench = ComparativeSearchBenchmark(
                    task=task,
                    seed=s,
                    max_iterations=max_iterations,
                    steps_per_candidate=steps_per_candidate,
                    eval_timeout=eval_timeout,
                    llm_provider=llm_provider,
                    llm_model=llm_model
                )
                res = bench.run_trajectory(method=method)
                elapsed = time.time() - t0

                all_trajectories[task][method].append(res)

                # Save individual trajectory
                with open(trace_file, "w") as f:
                    json.dump(res, f, indent=2, default=str)

                auc_val = res.get("auc_normalized_gain", res.get("auc_search_curve", 0.0))
                print(f"  [Seed {s:3d}] Base: {res['baseline_val_loss']:.4f} -> Best: {res['best_val_loss']:.4f} "
                      f"(Gain: {res['improvement_pct']:+.2f}%) | OOD: {res['best_ood_loss']:.4f} | "
                      f"AUC Gain: {auc_val:.4f} | Eval Time: {res.get('total_eval_wall_clock_sec', res.get('total_gpu_seconds', 0.0)):.1f}s", flush=True)

                # Registry entry
                exp_entry = {
                    "experiment_id": f"EXP-PHASE8-{task.upper()}-{method.upper()}-SEED-{s}",
                    "git_commit": "bebf551a3d02e4cf0c058c42a2d48c8b25cb48bb",
                    "task": task,
                    "method": method,
                    "seed": s,
                    "model": f"{method}_search_agent",
                    "budget": f"K={max_iterations}, {res['total_gpu_seconds']:.1f}s GPU",
                    "dataset_version": f"{task}_v1_0",
                    "prompt_version": "SYSTEM_PROMPT_v2_0" if method == "llm" else "N/A",
                    "result": {
                        "baseline_loss": res["baseline_val_loss"],
                        "best_val_loss": res["best_val_loss"],
                        "best_ood_loss": res["best_ood_loss"],
                        "improvement_pct": res["improvement_pct"],
                        "auc_normalized_gain": auc_val,
                        "auc_search_curve": auc_val,
                        "raw_loss_auc": res.get("raw_loss_auc", 0.0),
                        "gpu_seconds": res["total_gpu_seconds"]
                    },
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": "COMPLETED"
                }
                registry_entries.append(exp_entry)

    # Statistical Aggregation
    print("\n\n=== COMPUTING RIGOROUS STATISTICAL DISTRIBUTIONS & HYPOTHESIS TESTS ===")
    analysis_results = {}

    for task in tasks:
        task_analysis = {"arms": {}, "comparisons_vs_random": {}, "comparisons_vs_llm": {}}
        for method in methods:
            runs = all_trajectories[task][method]
            best_vals = [r["best_val_loss"] for r in runs]
            best_oods = [r["best_ood_loss"] for r in runs]
            gains = [r.get("improvement_pct", 0.0) for r in runs]
            aucs = [r.get("auc_normalized_gain", r.get("auc_search_curve", 0.0)) for r in runs]
            gpu_times = [r["total_gpu_seconds"] for r in runs]
            cand_gpu_times = [r.get("candidate_train_gpu_sec", r["total_gpu_seconds"]) for r in runs]
            llm_inf_times = [r.get("llm_inference_sec", 0.0) for r in runs]
            total_wall_times = [r.get("total_search_wall_clock_sec", r["total_gpu_seconds"]) for r in runs]

            task_analysis["arms"][method] = {
                "best_val_loss": summarize_distribution(best_vals),
                "best_ood_loss": summarize_distribution(best_oods),
                "gain_pct": summarize_distribution(gains),
                "auc_normalized_gain": summarize_distribution(aucs),
                "auc_search_curve": summarize_distribution(aucs),
                "gpu_seconds": summarize_distribution(gpu_times),
                "candidate_train_gpu_sec": summarize_distribution(cand_gpu_times),
                "llm_inference_sec": summarize_distribution(llm_inf_times),
                "total_search_wall_clock_sec": summarize_distribution(total_wall_times)
            }

        # Comparative tests vs Random and vs LLM
        for other in ["tpe", "evolutionary", "llm"]:
            if other in methods and "random" in methods:
                other_vals = [r["best_val_loss"] for r in all_trajectories[task][other]]
                rand_vals = [r["best_val_loss"] for r in all_trajectories[task]["random"]]
                task_analysis["comparisons_vs_random"][f"{other}_vs_random"] = compare_two_search_arms(
                    other, other_vals, "random", rand_vals
                )

        if "llm" in methods:
            llm_vals = [r["best_val_loss"] for r in all_trajectories[task]["llm"]]
            for other in ["tpe", "evolutionary"]:
                if other in methods:
                    other_vals = [r["best_val_loss"] for r in all_trajectories[task][other]]
                    task_analysis["comparisons_vs_llm"][f"llm_vs_{other}"] = compare_two_search_arms(
                        "llm", llm_vals, other, other_vals
                    )

        analysis_results[task] = task_analysis

    # Save summary JSON
    results_path = os.path.join(out_dir, "comparative_search_results.json")
    with open(results_path, "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"\n[Saved Statistical Results] -> {results_path}")

    # Generate Markdown Report
    report_md = generate_comparative_report(analysis_results, methods, tasks, seeds, max_iterations)
    report_path = os.path.join(out_dir, "comparative_search_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"[Saved Comparative Report]  -> {report_path}")

    # Update EXPERIMENT_REGISTRY.json
    reg_path = os.path.abspath(os.path.join(out_dir, "..", "EXPERIMENT_REGISTRY.json"))
    if os.path.exists(reg_path):
        with open(reg_path, "r") as f:
            reg = json.load(f)
        reg["experiments"].extend(registry_entries)
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=2, default=str)
        print(f"[Updated Registry] -> {len(registry_entries)} comparative runs logged in {reg_path}")

    return analysis_results


def generate_comparative_report(
    analysis: Dict[str, Any],
    methods: List[str],
    tasks: List[str],
    seeds: List[int],
    horizon: int
) -> str:
    md = [
        "# Phase 8: Matched-Compute Comparative Search Benchmark Report",
        "",
        "**Core Research Question**: *When does semantic LLM reasoning provide marginal value over established search algorithms (Random, TPE, GA) under equal compute budgets, and why?*",
        "",
        f"**Search Horizon**: $K={horizon}$ candidate evaluations per trajectory.",
        f"**Seeds Evaluated**: `{seeds}` ($N={len(seeds)}$ replications per arm).",
        "**Budget Constraint**: All candidates evaluated by the Immutable Evaluator under identical gradient step budgets.",
        "",
        "---",
        ""
    ]

    for task in tasks:
        md.append(f"## Benchmark: {task.upper()}")
        md.append("")
        md.append("### 1. Performance Summary Across Search Arms")
        md.append("| Search Arm | $L^*_{\\text{val}}$ Mean [95% CI] | $L^*_{\\text{OOD}}$ Mean [95% CI] | Gain (%) Mean | Normalized Gain AUC | Cand Train GPU (s) | LLM Infer (s) | Total Wall-Clock (s) |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        arms = analysis[task]["arms"]
        for m in methods:
            a = arms[m]
            vl = a["best_val_loss"]
            ol = a["best_ood_loss"]
            gn = a["gain_pct"]
            auc = a["auc_search_curve"]
            cand_t = a.get("candidate_train_gpu_sec", a["gpu_seconds"])
            llm_t = a.get("llm_inference_sec", {"mean": 0.0})
            tot_t = a.get("total_search_wall_clock_sec", a["gpu_seconds"])

            md.append(
                f"| **{m.upper()}** | {vl['mean']:.4f} [{vl['ci_95'][0]:.4f}, {vl['ci_95'][1]:.4f}] | "
                f"{ol['mean']:.4f} [{ol['ci_95'][0]:.4f}, {ol['ci_95'][1]:.4f}] | "
                f"{gn['mean']:+.2f}% | {auc['mean']:.4f} | {cand_t['mean']:.1f}s | {llm_t['mean']:.1f}s | {tot_t['mean']:.1f}s |"
            )

        md.append("")
        md.append("### 2. Hypothesis Testing & Effect Sizes")
        md.append("| Comparison | Mean Difference | Cohen's $d$ | Hedges' $g$ | Welch's $t$ $p$-value | Mann-Whitney $U$ $p$-value | Permutation $p$-value |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for comp_name, comp_data in analysis[task].get("comparisons_vs_random", {}).items():
            md.append(
                f"| {comp_name} | {comp_data['mean_diff']:+.4f} | {comp_data['cohens_d']:+.2f} | "
                f"{comp_data['hedges_g']:+.2f} | {comp_data['welch_t']['p_value']:.4f} | "
                f"{comp_data['mann_whitney_u']['p_value']:.4f} | {comp_data['permutation_test']['p_value']:.4f} |"
            )

        for comp_name, comp_data in analysis[task].get("comparisons_vs_llm", {}).items():
            md.append(
                f"| {comp_name} | {comp_data['mean_diff']:+.4f} | {comp_data['cohens_d']:+.2f} | "
                f"{comp_data['hedges_g']:+.2f} | {comp_data['welch_t']['p_value']:.4f} | "
                f"{comp_data['mann_whitney_u']['p_value']:.4f} | {comp_data['permutation_test']['p_value']:.4f} |"
            )

        md.append("")
        md.append("---")
        md.append("")

    md.extend([
        "## 3. Scientific Takeaways",
        "- **Search Paradigm Comparison**: Compares uniform stochastic exploration (Random), surrogate model optimization (TPE), population-based genetic selection (Evolutionary), and semantic hypothesis generation (LLM).",
        "- **Task Complexity Scaling**: Evaluates whether the inductive biases discoverable by each search paradigm differ across Chomsky regular state tracking (Hidden FSM) versus context-free hierarchical grammar (Dyck-4).",
        "- **Matched Evaluation Budget**: All candidate configurations are strictly bounded by identical gradient update horizons in the Immutable Evaluator, isolating search algorithm efficacy from training budget disparities."
    ])

    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8: Matched-Compute Comparative Search Benchmark")
    parser.add_argument("--tasks", nargs="+", default=["dyck", "fsm"], help="Tasks to evaluate")
    parser.add_argument("--methods", nargs="+", default=["random", "tpe", "evolutionary", "llm"], help="Search methods")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202, 303, 404], help="Random seeds")
    parser.add_argument("--max-iterations", type=int, default=10, help="Search horizon K")
    parser.add_argument("--steps-per-candidate", type=int, default=25, help="Evaluation steps per candidate")
    parser.add_argument("--eval-timeout", type=float, default=300.0, help="Timeout in seconds per candidate evaluation")
    parser.add_argument("--llm-provider", type=str, default=None, help="LLM provider name")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model name")
    parser.add_argument("--fresh", action="store_true", default=True, help="Clean run without cached traces (default: True)")
    parser.add_argument("--resume", dest="fresh", action="store_false", help="Resume from cached traces if present")
    args = parser.parse_args()

    run_comparative_experiment(
        tasks=args.tasks,
        methods=args.methods,
        seeds=args.seeds,
        max_iterations=args.max_iterations,
        steps_per_candidate=args.steps_per_candidate,
        eval_timeout=args.eval_timeout,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        fresh_run=args.fresh
    )
