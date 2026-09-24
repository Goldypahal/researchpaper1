"""
Master Scientific Reproduction Pipeline (Phase 24).
One-command execution validating the entire empirical research program:

1. Verification of Benchmarks & Unit Invariants (Phase 1)
2. Adversarial Specification-Gaming Evaluation (Phase 5)
3. Multi-Seed Benchmark Validation (Phase 2)
4. Matched-Compute Comparative Search (Phase 8: Random, TPE, GA, LLM)
5. Cognitive Reasoning Ablations (Phase 11: Memory, Reflection, Critic)
6. Statistical Significance & Audit Registry Verification
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import subprocess

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)


def run_stage(title: str, cmd: list):
    print(f"\n{'='*70}")
    print(f"STAGE: {title}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    t0 = time.time()
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUNBUFFERED"] = "1"
    
    proc = subprocess.run(cmd, cwd=WORKSPACE_ROOT, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n[FAILED] Stage '{title}' exited with code {proc.returncode} ({elapsed:.1f}s)")
        sys.exit(proc.returncode)
    print(f"\n[PASSED] Stage '{title}' completed in {elapsed:.1f}s")


def main():
    print("*********************************************************************")
    print("RESEARCHPAPER1: COMPLETE SCIENTIFIC REPRODUCTION PIPELINE")
    print(f"Workspace: {WORKSPACE_ROOT}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("*********************************************************************")

    # 1. Benchmark Unit Tests
    run_stage("Phase 1 - Formal Benchmark Invariants & Split Isolation", [
        sys.executable, "-m", "unittest", "benchmarks/tests/test_benchmarks.py"
    ])

    # 2. Adversarial Gaming Suite
    run_stage("Phase 5 - Adversarial Specification-Gaming Security Suite", [
        sys.executable, "evaluator/adversarial_gaming.py"
    ])

    # 3. Phase 2 Benchmark Validation
    run_stage("Phase 2 - Multi-Seed Benchmark Validation & Generalization Gaps", [
        sys.executable, "v0.2-benchmark-validation/run_benchmark_validation.py"
    ])

    # 4. Phase 8 Comparative Search
    run_stage("Phase 8 - Matched-Compute Comparative Search (Random vs TPE vs GA vs LLM)", [
        sys.executable, "v0.3-comparative-search/main_experiment.py"
    ])

    # 5. Phase 11 Reasoning Ablation
    run_stage("Phase 11 - LLM Cognitive Reasoning Ablations", [
        sys.executable, "v0.4-reasoning-ablation/run_reasoning_ablation.py"
    ])

    print("\n*********************************************************************")
    print("REPRODUCTION COMPLETE: All empirical stages executed successfully.")
    print(f"Master Audit Registry: {os.path.join(WORKSPACE_ROOT, 'EXPERIMENT_REGISTRY.json')}")
    print("*********************************************************************")


if __name__ == "__main__":
    main()
