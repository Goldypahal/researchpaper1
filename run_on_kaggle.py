"""
Kaggle GPU Automated Research Pipeline.
Designed to execute on Kaggle's NVIDIA T4 / P100 GPU accelerators:
  - Eliminates local CPU thermal load and laptop heating.
  - Leverages fast CUDA tensor operations for large-sample sweeps (N=10 to 50).
  - Automatically exports all traces, logs, reports, and figures to a zip archive.

Usage in Kaggle Notebook (with GPU accelerator enabled):
  !python run_on_kaggle.py --seeds 5 --iterations 10
"""

import os
import sys
import time
import json
import zipfile
import argparse
import subprocess
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "v0.3-comparative-search"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "v0.4-reasoning-ablation"))

from evaluator.immutable_evaluator import ImmutableEvaluator
from main_experiment import run_comparative_experiment
from run_reasoning_ablation import run_reasoning_ablation_suite
from benchmarks.complexity_ladder import evaluate_complexity_point


def check_gpu_environment():
    print("=" * 70)
    print("KAGGLE GPU ENVIRONMENT VERIFICATION")
    print("=" * 70)
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        capability = torch.cuda.get_device_capability(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device Name: {device_name}")
        print(f"Device Count: {device_count}")
        print(f"Compute Capability: {capability}")
        print(f"Total GPU Memory: {total_mem:.2f} GB")
        print("Status: FAST ACCELERATION ENABLED.")
    else:
        print("WARNING: CUDA is not available. Please verify GPU accelerator is enabled in Kaggle Settings.")
    print("=" * 70 + "\n")
    return cuda_avail


def run_kaggle_sweep(
    tasks=["dyck", "fsm"],
    methods=["random", "tpe", "evolutionary", "llm"],
    num_seeds=5,
    iterations=10,
    steps_per_cand=50,
    run_ablations=True,
    llm_provider=None,
    llm_model=None,
    api_key=None,
    allow_simulation=False
):
    start_all = time.time()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # Preflight Check: Real LLM Connection Verification
    if "llm" in methods:
        print("=" * 70, flush=True)
        print("PREFLIGHT HEALTH CHECK: REAL LLM REASONING ADAPTER", flush=True)
        print("=" * 70, flush=True)
        from agent_scaffold.llm_agent import LLMResearchAgent, SimulationDisallowedError
        try:
            test_agent = LLMResearchAgent(
                provider=llm_provider,
                model=llm_model,
                api_key=api_key,
                strict=True,
                allow_simulation=allow_simulation
            )
            ok, test_resp = test_agent.verify_connection()
            if not ok:
                raise RuntimeError(f"LLM live verification ping failed: {test_resp}")
            print("Status: REAL LLM CONNECTION VERIFIED.\n", flush=True)
        except SimulationDisallowedError as e:
            print(f"\n[FATAL EXPERIMENT HALT] {e}\n", flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"\n[FATAL EXPERIMENT HALT] LLM adapter failed to connect: {e}\n", flush=True)
            sys.exit(1)

    # Select standardized seeds
    standard_seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    seeds = standard_seeds[:num_seeds]

    print(f"Starting Kaggle Experimental Sweep on device: {device_type.upper()}")
    print(f"Tasks: {tasks} | Methods: {methods}")
    print(f"Seeds: {seeds} (N={len(seeds)}) | Horizon: K={iterations} | Steps: {steps_per_cand}\n")

    # 1. Main Comparative Search Sweep
    print(">>> STAGE 1: Matched-Compute Comparative Search Benchmark <<<")
    comp_results = run_comparative_experiment(
        tasks=tasks,
        methods=methods,
        seeds=seeds,
        max_iterations=iterations,
        steps_per_candidate=steps_per_cand,
        llm_provider=llm_provider,
        llm_model=llm_model
    )

    # 2. Reasoning Ablation Sweep
    if run_ablations:
        print("\n>>> STAGE 2: Cognitive Reasoning Ablation Suite <<<")
        ablation_results = run_reasoning_ablation_suite(
            task="dyck",
            modes=["no_history", "history_no_reflection", "full", "critic_refine"],
            seeds=seeds[:min(3, len(seeds))],
            max_iterations=min(6, iterations),
            steps_per_candidate=steps_per_cand
        )

    # 3. Packaging Results
    elapsed = time.time() - start_all
    print(f"\nAll experimental sweeps completed in {elapsed/60.0:.2f} minutes.")

    # Create zip archive of all results, traces, and registry
    archive_name = "kaggle_experiment_results.zip"
    archive_path = os.path.join(WORKSPACE_ROOT, archive_name)
    print(f"Packaging artifacts into: {archive_path}")

    files_to_pack = [
        "EXPERIMENT_REGISTRY.json",
        "research_questions_and_hypotheses.md",
        "README.md",
        os.path.join("v0.2-benchmark-validation", "benchmark_validation_report.md"),
        os.path.join("v0.2-benchmark-validation", "benchmark_validation_results.json"),
        os.path.join("v0.3-comparative-search", "comparative_search_report.md"),
        os.path.join("v0.3-comparative-search", "comparative_search_results.json"),
        os.path.join("v0.4-reasoning-ablation", "reasoning_ablation_report.md"),
        os.path.join("v0.4-reasoning-ablation", "reasoning_ablation_results.json"),
        os.path.join("evaluator", "adversarial_gaming_report.md"),
    ]

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for rel_path in files_to_pack:
            abs_path = os.path.join(WORKSPACE_ROOT, rel_path)
            if os.path.exists(abs_path):
                zipf.write(abs_path, arcname=rel_path)

        # Include all traces
        for sub in [os.path.join("v0.3-comparative-search", "traces"), os.path.join("v0.4-reasoning-ablation", "traces")]:
            sub_abs = os.path.join(WORKSPACE_ROOT, sub)
            if os.path.exists(sub_abs):
                for root, _, files in os.walk(sub_abs):
                    for file in files:
                        fp = os.path.join(root, file)
                        arc = os.path.relpath(fp, WORKSPACE_ROOT)
                        zipf.write(fp, arcname=arc)

    # Copy to /kaggle/working/ if running in Kaggle subdirectory
    kaggle_out = "/kaggle/working"
    if os.path.exists(kaggle_out):
        dest_path = os.path.join(kaggle_out, archive_name)
        if os.path.abspath(dest_path) != os.path.abspath(archive_path):
            import shutil
            shutil.copy2(archive_path, dest_path)
            print(f"[Exported] Copied zip to: {dest_path}")

    print(f"\n[DONE] Kaggle execution finished. Download '{archive_name}' from the Kaggle Output panel.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run researchpaper1 experimental sweep on Kaggle GPU.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seed replications (e.g. 5, 10, 30)")
    parser.add_argument("--iterations", type=int, default=10, help="Search horizon K (iterations per arm)")
    parser.add_argument("--steps", type=int, default=50, help="Training steps per candidate")
    parser.add_argument("--no_ablations", action="store_true", help="Skip reasoning ablation suite")
    parser.add_argument("--provider", type=str, default=None, choices=["anthropic", "openai", "gemini", "groq", "openrouter", "nvidia", "local"], help="LLM Provider")
    parser.add_argument("--model", type=str, default=None, help="LLM model identifier")
    parser.add_argument("--api-key", type=str, default=None, help="Explicit API key (otherwise auto-read from environment/Kaggle Secrets)")
    parser.add_argument("--local-llm", type=str, default=None, help="HuggingFace model ID for on-GPU local inference (e.g., Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--allow-simulation", action="store_true", help="Opt-in to scripted simulation (FOR TESTING ONLY, forbidden for research papers)")
    args = parser.parse_args()

    if args.local_llm:
        os.environ["LOCAL_LLM_MODEL"] = args.local_llm

    check_gpu_environment()
    run_kaggle_sweep(
        num_seeds=args.seeds,
        iterations=args.iterations,
        steps_per_cand=args.steps,
        run_ablations=not args.no_ablations,
        llm_provider=args.provider or ("local" if args.local_llm else None),
        llm_model=args.model or args.local_llm,
        api_key=args.api_key,
        allow_simulation=args.allow_simulation
    )
