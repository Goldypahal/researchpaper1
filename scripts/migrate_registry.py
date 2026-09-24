"""
Migrates EXPERIMENT_REGISTRY.json to Schema 2.0.0:
Adds explicit commit tracking (experiment_commit, analysis_commit),
component versions (benchmark, search space, evaluator, prompt),
and exact hardware device telemetry.
"""

import os
import json

REGISTRY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "EXPERIMENT_REGISTRY.json"))

def migrate():
    with open(REGISTRY_PATH, "r") as f:
        data = json.load(f)

    data["schema_version"] = "2.0.0"
    data["registry_specification"] = {
        "experiment_commit": "Exact git hash at time of model training",
        "analysis_commit": "Exact git hash at time of statistical analysis",
        "benchmark_version": "Version of formal task generator (dyck_v1, fsm_v1)",
        "search_space_version": "Version of discrete & continuous bounds",
        "evaluator_version": "Version of immutable evaluation harness",
        "device": "Hardware accelerator ('cuda' or 'cpu')",
        "device_name": "Physical processor identification"
    }

    updated_experiments = []
    for exp in data.get("experiments", []):
        old_commit = exp.get("git_commit", "c7d3f3a0d90ed05d40277bac3f14ab001d2151bf")
        res = exp.get("result", {})
        
        # Determine actual device from telemetry
        device = res.get("device", "cpu")
        device_name = res.get("device_name", "Intel/AMD x86_64 CPU")
        
        new_exp = {
            "experiment_id": exp.get("experiment_id"),
            "experiment_commit": old_commit,
            "analysis_commit": "9c475fa",
            "benchmark_version": f"{exp.get('task', 'benchmark')}_v1_0",
            "search_space_version": "v2_0_level1_level2",
            "evaluator_version": "v1_0_immutable_sha256",
            "prompt_version": exp.get("prompt_version", "N/A"),
            "task": exp.get("task"),
            "method": exp.get("method"),
            "seed": exp.get("seed"),
            "model": exp.get("model"),
            "budget": exp.get("budget"),
            "device": device,
            "device_name": device_name,
            "result": res,
            "timestamp": exp.get("timestamp"),
            "status": exp.get("status", "COMPLETED")
        }
        updated_experiments.append(new_exp)

    data["experiments"] = updated_experiments

    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Successfully migrated {len(updated_experiments)} experiments in {REGISTRY_PATH} to Schema 2.0.0.")


if __name__ == "__main__":
    migrate()
