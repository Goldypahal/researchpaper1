"""
Runs Seed 999 and Seed 2026 to complete the 5-seed benchmark for Arm 2.
"""

from random_search_baseline import run_random_search
import json
import os

seeds = [999, 2026]
results = []

for s in seeds:
    print(f"\n==========================================")
    print(f"RUNNING ARM 2 SEED: {s}")
    print(f"==========================================\n")
    exp_id = f"exp_arm2_seed_{s}"
    summary = run_random_search(
        experiment_id=exp_id,
        arm="Arm2_RandomSearchBaseline",
        dry_run=False,
        max_iterations=3,
        budget_hours=1.0,
        seed=s
    )
    results.append({
        "seed": s,
        "summary": summary["summary_statistics"]
    })

print("\nALL REMAINING SEEDS FINISHED.")
print(json.dumps(results, indent=2))
