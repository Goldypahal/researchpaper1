"""
run_on_kaggle.py

Kaggle GPU Automated Research Pipeline (Resilient & Audited).
Enforces:
  1. Default RESUME behavior (never silently wipes or overwrites completed runs)
  2. Safe archiving: --fresh moves existing traces to traces_archived_<timestamp>/
  3. Strict version validation: skips only traces that match code_version & COMPLETE_VALID
  4. Time-guard: cleanly stops before Kaggle 12-hour session limit
  5. Continuous periodic packaging: zips traces every batch so no results are lost on timeout
  6. Supports all arms: random, tpe, evolutionary, prior_sampler, tpe_warm, and llm decomposition
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import glob
import shutil
import zipfile
import argparse
import subprocess
import torch
from typing import List, Dict, Any, Optional

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.run_comparative_search import ComparativeSearchRunner, get_code_version


def check_gpu_environment():
    print("=" * 70)
    print("KAGGLE GPU ENVIRONMENT VERIFICATION")
    print("=" * 70)
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device Name: {device_name}")
        print(f"Device Count: {device_count}")
        print(f"Total GPU Memory: {total_mem:.2f} GB")
        print("Status: FAST ACCELERATION ENABLED.")
    else:
        print("WARNING: CUDA is not available. Please verify GPU accelerator is enabled in Kaggle Settings.")
    print("=" * 70 + "\n")
    return cuda_avail


def package_results_zip(traces_dir: str, output_zip_path: str):
    """Packages all valid traces, registry entries, and logs into a zip file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)
    temp_zip = output_zip_path + ".tmp"
    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(traces_dir):
            for f in glob.glob(os.path.join(traces_dir, "*")):
                zf.write(f, os.path.join("traces", os.path.basename(f)))
        registry_path = os.path.join(WORKSPACE_ROOT, "registry.jsonl")
        if os.path.exists(registry_path):
            zf.write(registry_path, "registry.jsonl")
    if os.path.exists(output_zip_path):
        os.remove(output_zip_path)
    os.replace(temp_zip, output_zip_path)
    print(f"[ZIP PACKAGER] Updated archive: {output_zip_path} ({os.path.getsize(output_zip_path)/(1024**2):.2f} MB)")


def run_kaggle_pipeline(
    tasks: List[str] = ["dyck", "fsm", "abs_position"],
    arms: List[str] = ["random", "tpe", "evolutionary", "llm"],
    llm_modes: List[str] = ["named_real", "anonymized_real", "named_shuffled"],
    num_seeds: int = 10,
    iterations: int = 10,
    steps_per_cand: int = 50,
    llm_provider: str = "local",
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
    fresh_run: bool = False,
    max_hours: float = 11.0,
    zip_frequency: int = 5
):
    start_time = time.time()
    traces_dir = os.path.join(WORKSPACE_ROOT, "traces")

    # Safe archiving if fresh_run requested
    if fresh_run and os.path.exists(traces_dir):
        archive_dir = os.path.join(WORKSPACE_ROOT, f"traces_archived_{int(time.time())}")
        print(f"[SAFETY] --fresh requested: moving existing traces to {archive_dir}")
        shutil.move(traces_dir, archive_dir)

    os.makedirs(traces_dir, exist_ok=True)

    # Determine Kaggle output zip destination
    kaggle_working = "/kaggle/working" if os.path.exists("/kaggle/working") else WORKSPACE_ROOT
    output_zip = os.path.join(kaggle_working, "kaggle_experiment_results.zip")

    # Load standard seeds
    standard_seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333, 444, 555][:num_seeds]

    # Pre-generate zero-shot proposal pool for prior_sampler / tpe_warm
    proposal_pool = None
    if "prior_sampler" in arms or "tpe_warm" in arms:
        print("\n>>> PRE-GENERATING ZERO-SHOT PROPOSAL POOL FOR HYBRID ARMS <<<")
        allow_sim = os.environ.get("ALLOW_LLM_SIMULATION", "0").lower() in ("1", "true", "yes")
        temp_agent = LLMSearchAgent(
            provider=llm_provider,
            model=llm_model,
            strict=True,
            allow_simulation=allow_sim,
            task_framing="named",
            feedback="none"
        )
        proposal_pool = []
        for p_idx in range(1, 21):
            p = temp_agent.propose_experiment(
                iteration_idx=p_idx,
                history=[],
                incumbent_best_loss=99.0,
                base_loss=4.2,
                task="dyck",
                seed=42 + p_idx
            )
            proposal_pool.append(p)
        print(f"[SUCCESS] Pre-generated {len(proposal_pool)} zero-shot proposals for prior sampling.")

    # Execute Planned Runs
    runs_completed = 0
    trajectories_run = []

    for task in tasks:
        runner = ComparativeSearchRunner(
            task=task,
            seed=42,  # overridden per run
            iterations=iterations,
            steps_per_candidate=steps_per_cand,
            llm_provider=llm_provider,
            llm_model=llm_model,
            traces_dir=traces_dir
        )

        for arm in arms:
            # Map arm conditions
            if arm == "llm":
                target_modes = llm_modes
            else:
                target_modes = ["standard"]

            for mode_desc in target_modes:
                framing = "anonymized" if "anonymized" in mode_desc else "named"
                feedback = "shuffled" if "shuffled" in mode_desc else ("none" if "none" in mode_desc else "real")
                obfuscated = ("obfuscated" in mode_desc)

                for s in standard_seeds:
                    # Time guard check
                    elapsed_hours = (time.time() - start_time) / 3600.0
                    if elapsed_hours >= max_hours:
                        print(f"\n[TIME-GUARD] Approaching max budget ({elapsed_hours:.2f}h >= {max_hours:.2f}h). Stopping gracefully.")
                        package_results_zip(traces_dir, output_zip)
                        return

                    runner.seed = s
                    runner.evaluator = runner.evaluator if runner.evaluator.seed == s else None
                    if runner.evaluator is None:
                        from evaluator.immutable_evaluator import ImmutableEvaluator
                        runner.evaluator = ImmutableEvaluator(task=task, seed=s)

                    res = runner.run_trajectory(
                        arm=arm,
                        task_framing=framing,
                        feedback=feedback,
                        obfuscated=obfuscated,
                        proposal_pool=proposal_pool,
                        skip_if_valid=True
                    )
                    trajectories_run.append(res)
                    runs_completed += 1

                    # Continuous packaging
                    if runs_completed % zip_frequency == 0:
                        package_results_zip(traces_dir, output_zip)

    # Final packaging
    package_results_zip(traces_dir, output_zip)
    print("\n" + "=" * 70)
    print(f"PIPELINE RUN COMPLETE: {runs_completed} trajectories executed/verified.")
    print(f"Results archived at: {output_zip}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Research Runner (Audited & Resumable)")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--iterations", type=int, default=10, help="Proposals per trajectory (default: 10)")
    parser.add_argument("--steps", type=int, default=50, help="Training steps per candidate (default: 50)")
    parser.add_argument("--tasks", nargs="+", default=["dyck", "fsm", "abs_position"])
    parser.add_argument("--arms", nargs="+", default=["random", "tpe", "evolutionary", "prior_sampler", "llm"])
    parser.add_argument("--llm_modes", nargs="+", default=["named_real", "anonymized_real", "named_shuffled"])
    parser.add_argument("--provider", type=str, default="local", choices=["local", "nvidia", "groq", "mistral", "gemini", "openai", "anthropic"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--allow-simulation", action="store_true", help="Opt-in to simulation for CPU testing only")
    parser.add_argument("--fresh", action="store_true", help="Archive old traces and start clean")
    parser.add_argument("--max-hours", type=float, default=11.0, help="Session time-guard threshold in hours")
    args = parser.parse_args()

    if args.allow_simulation:
        os.environ["ALLOW_LLM_SIMULATION"] = "1"

    if args.provider == "local":
        os.environ["LOCAL_LLM_MODEL"] = args.model
        os.environ["LOCAL_LLM_QUANT"] = args.quantization

    check_gpu_environment()

    run_kaggle_pipeline(
        tasks=args.tasks,
        arms=args.arms,
        llm_modes=args.llm_modes,
        num_seeds=args.seeds,
        iterations=args.iterations,
        steps_per_cand=args.steps,
        llm_provider=args.provider,
        llm_model=args.model,
        fresh_run=args.fresh,
        max_hours=args.max_hours
    )
