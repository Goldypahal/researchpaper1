#!/usr/bin/env python3
"""
tools/generate_experiment_plan.py

Generates the canonical experiment plan (experiment_plan.json).
Defines every planned run across:
  - Tasks: dyck, fsm, abs_position
  - Search arms:
      - classical: random, tpe, evolutionary
      - hybrid: prior_sampler, tpe_warm
      - llm decomposition:
          - llm_named_real (Total effect)
          - llm_anonymized_real (Removes semantic prior)
          - llm_named_shuffled (Feedback scrambled)
          - llm_named_none (Zero-shot prior only)
          - llm_obfuscated_real (Names stripped)
"""

import os
import json
from typing import List, Dict, Any

STANDARD_SEEDS = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    111, 222, 333, 444, 555, 666, 777, 888, 999, 123,
    234, 345, 456, 567, 678, 789, 890, 901, 102, 213
]

def generate_plan(
    output_path: str = "experiment_plan.json",
    num_seeds: int = 30,
    iterations: int = 10,
    steps_per_cand: int = 50
) -> List[Dict[str, Any]]:
    seeds = STANDARD_SEEDS[:num_seeds]
    tasks = ["dyck", "fsm", "abs_position"]
    plan = []

    # 1. Classical Baselines
    for task in tasks:
        for arm in ["random", "tpe", "evolutionary", "prior_sampler", "tpe_warm"]:
            for seed in seeds:
                run_id = f"{task}__{arm}__seed{seed}"
                plan.append({
                    "run_id": run_id,
                    "task": task,
                    "arm": arm,
                    "task_framing": "named",
                    "feedback": "real",
                    "obfuscated": False,
                    "seed": seed,
                    "iterations": iterations,
                    "steps": steps_per_cand,
                    "type": "baseline"
                })

    # 2. LLM Decomposition Arms
    llm_conditions = [
        {"name": "llm_named_real", "framing": "named", "feedback": "real", "obfuscated": False},
        {"name": "llm_anonymized_real", "framing": "anonymized", "feedback": "real", "obfuscated": False},
        {"name": "llm_named_shuffled", "framing": "named", "feedback": "shuffled", "obfuscated": False},
        {"name": "llm_named_none", "framing": "named", "feedback": "none", "obfuscated": False},
        {"name": "llm_obfuscated_real", "framing": "named", "feedback": "real", "obfuscated": True}
    ]

    for task in tasks:
        for cond in llm_conditions:
            for seed in seeds:
                run_id = f"{task}__{cond['name']}__seed{seed}"
                plan.append({
                    "run_id": run_id,
                    "task": task,
                    "arm": "llm",
                    "task_framing": cond["framing"],
                    "feedback": cond["feedback"],
                    "obfuscated": cond["obfuscated"],
                    "seed": seed,
                    "iterations": iterations,
                    "steps": steps_per_cand,
                    "type": "llm_decomposition"
                })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(f"Generated {len(plan)} planned runs across {len(tasks)} tasks and {num_seeds} seeds -> {output_path}")
    return plan

if __name__ == "__main__":
    generate_plan()
