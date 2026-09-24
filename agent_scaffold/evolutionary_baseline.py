"""
Evolutionary / Genetic Algorithm Baseline (Arm 4) for SmallTransformerLM.
Implements population-based genetic search with tournament selection,
uniform crossover, and point mutation over the architecture search space.
Logs full graph traces compatible with trace_schema.json.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import random
import copy
import argparse

from trace_logger import TraceLogger
from sandbox_runner import execute_in_sandbox

# Design space definition
GENE_SPACE = {
    "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 0.001, 0.01, 0.05, 0.1],
    "n_layers": [2, 4, 6, 8],
    "n_heads": [2, 4, 8],
    "d_model": [128, 256, 384],
    "activation": ["gelu", "relu", "silu"],
    "norm_type": ["layernorm", "rmsnorm"],
    "scale_factor": [None, 0.25, 0.5, 1.0, 2.0],
    "pos_encoding": ["learned", "sinusoidal", "rotary", "none"],
    "ffn_type": ["standard", "swiglu"],
    "topology": ["pre_ln", "post_ln", "parallel"],
    "n_kv_heads": ["same", 1, 2]
}

def sample_random_individual():
    heads = random.choice(GENE_SPACE["n_heads"])
    kv_choice = random.choice(GENE_SPACE["n_kv_heads"])
    if kv_choice == "same":
        kv_heads = heads
    elif heads % kv_choice == 0:
        kv_heads = kv_choice
    else:
        kv_heads = heads

    return {
        "lr": random.choice(GENE_SPACE["lr"]),
        "weight_decay": random.choice(GENE_SPACE["weight_decay"]),
        "n_layers": random.choice(GENE_SPACE["n_layers"]),
        "n_heads": heads,
        "n_kv_heads": kv_heads,
        "d_model": random.choice(GENE_SPACE["d_model"]),
        "activation": random.choice(GENE_SPACE["activation"]),
        "norm_type": random.choice(GENE_SPACE["norm_type"]),
        "scale_factor": random.choice(GENE_SPACE["scale_factor"]),
        "pos_encoding": random.choice(GENE_SPACE["pos_encoding"]),
        "ffn_type": random.choice(GENE_SPACE["ffn_type"]),
        "topology": random.choice(GENE_SPACE["topology"])
    }

def crossover(parent1, parent2):
    """Uniform crossover: randomly inherit each gene from parent 1 or parent 2."""
    child = {}
    for gene in GENE_SPACE:
        child[gene] = parent1[gene] if random.random() < 0.5 else parent2[gene]
    return child

def mutate(individual, mutation_rate=0.35):
    """Point mutation: with probability mutation_rate, re-sample gene from space."""
    mutated = copy.deepcopy(individual)
    for gene, values in GENE_SPACE.items():
        if random.random() < mutation_rate:
            mutated[gene] = random.choice(values)
    return mutated

def tournament_selection(population, k=2):
    """Tournament selection: select best of k randomly sampled individuals."""
    candidates = random.sample(population, min(k, len(population)))
    candidates.sort(key=lambda ind: ind["val_loss"])
    return candidates[0]

def run_evolutionary_search(experiment_id="exp_arm4_evolutionary_baseline",
                            arm="Arm4_EvolutionarySearch_GA",
                            dry_run=False, max_iterations=10, population_size=4,
                            budget_hours=10.0, seed=42, task="dyck"):
    print("=== INITIALIZING EVOLUTIONARY (GA) SEARCH BASELINE (ARM 4) ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Arm: {arm}")
    print(f"Task Family: {task}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Population Size: {population_size}")
    print(f"GPU Budget: {budget_hours} Hours")
    print(f"Dry Run Mode: {dry_run}")
    print(f"Random Seed: {seed}\n")

    random.seed(seed)
    logger = TraceLogger(
        experiment_id=experiment_id,
        agent_arm=arm,
        total_gpu_budget_hours=budget_hours
    )

    harness_path = os.path.join(os.path.dirname(__file__), "eval_harness.py")
    harness_base_args = ["--task", task] + (["--dry-run"] if dry_run else ["--no-dry-run", "--steps", "50"])

    # Iteration 0: Baseline Evaluation
    print("[Iteration 0] Running Baseline Model Evaluation...")
    base_run = execute_in_sandbox(harness_path, harness_base_args, timeout_sec=60 if dry_run else 300)

    if base_run["status"] != "SUCCESS" or not base_run["metrics"]:
        print(f"Baseline run failed! Stderr: {base_run['stderr_tail']}")
        return

    base_loss = base_run["metrics"]["final_val_loss"]
    print(f"Baseline Validation Cross-Entropy Loss: {base_loss:.4f}\n")
    logger.set_baseline_loss(base_loss)

    hypo_0 = logger.add_hypothesis(
        iteration_idx=0,
        text="Baseline SmallTransformerLM (6 layers, 256 dim, 4 heads, GELU, LayerNorm).",
        target_component="attention",
        predicted_delta_loss=0.0,
        provider="ground_truth_baseline",
    )
    exp_0 = logger.add_experiment_spec(0, hypo_0, "Original unmodified model_baseline.py")
    run_0, _ = logger.add_execution_run(
        0, exp_0, "SUCCESS", 0, base_run["elapsed_seconds"], base_loss, base_run["metrics"]["final_train_loss"]
    )
    logger.add_reflection(0, run_0, hypo_0, "Baseline established as reference anchor.", "ACCEPT_AND_EXTEND", False)

    total_gpu_sec = base_run["elapsed_seconds"]
    consecutive_divergences = 0
    active_parent_hypo = hypo_0
    population = []

    for it in range(1, max_iterations + 1):
        if total_gpu_sec / 3600.0 >= budget_hours:
            print(f"[STOP] GPU Budget Exhausted ({total_gpu_sec/3600.0:.2f}h >= {budget_hours}h).")
            break
        if consecutive_divergences >= 3:
            print(f"[STOP] Three-strike divergence condition reached ({consecutive_divergences} consecutive failures).")
            break

        print(f"\n--- [Iteration {it}/{max_iterations}] ---")

        # In phase 1 (it <= population_size), initialize random chromosomes
        if len(population) < population_size:
            candidate = sample_random_individual()
            strategy_tag = "INITIAL_POPULATION"
            hypo_text = f"[GA_INIT] Random seed chromosome {it}/{population_size}: {candidate}"
        else:
            # Phase 2: Selection + Crossover + Mutation
            parent1 = tournament_selection(population, k=2)
            parent2 = tournament_selection(population, k=2)
            child = crossover(parent1["genes"], parent2["genes"])
            candidate = mutate(child, mutation_rate=0.35)
            strategy_tag = "CROSSOVER_MUTATION"
            hypo_text = (
                f"[GA_EVOLVE] Crossover from ParentA(loss={parent1['val_loss']:.4f}) "
                f"x ParentB(loss={parent2['val_loss']:.4f}) with point mutation: {candidate}"
            )

        print(f"[Arm 4 GA Sampler] [{strategy_tag}] Candidate: {candidate}")

        hypo_id = logger.add_hypothesis(
            iteration_idx=it,
            text=hypo_text,
            target_component="genetic_search",
            predicted_delta_loss=0.0,
            parent_hypo_id=active_parent_hypo,
            provider="evolutionary_ga_sampler"
        )
        exp_id = logger.add_experiment_spec(it, hypo_id, str(candidate))

        # Build sandbox arguments
        run_args = ["--task", task] + (["--dry-run"] if dry_run else ["--no-dry-run", "--steps", "50"])
        run_args.extend(["--lr", str(candidate["lr"])])
        run_args.extend(["--weight-decay", str(candidate["weight_decay"])])
        run_args.extend(["--n-layers", str(candidate["n_layers"])])
        run_args.extend(["--n-heads", str(candidate["n_heads"])])
        run_args.extend(["--d-model", str(candidate["d_model"])])
        run_args.extend(["--activation", str(candidate["activation"])])
        run_args.extend(["--norm-type", str(candidate["norm_type"])])
        if candidate.get("scale_factor"):
            run_args.extend(["--scale-factor", str(candidate["scale_factor"])])
        if candidate.get("pos_encoding"):
            run_args.extend(["--pos-encoding", str(candidate["pos_encoding"])])
        if candidate.get("ffn_type"):
            run_args.extend(["--ffn-type", str(candidate["ffn_type"])])
        if candidate.get("topology"):
            run_args.extend(["--topology", str(candidate["topology"])])
        if candidate.get("n_kv_heads"):
            run_args.extend(["--n-kv-heads", str(candidate["n_kv_heads"])])

        res = execute_in_sandbox(harness_path, run_args, timeout_sec=60 if dry_run else 300)
        total_gpu_sec += res["elapsed_seconds"]

        if res["status"] == "SUCCESS" and res["metrics"]:
            val_loss = res["metrics"]["final_val_loss"]
            train_loss = res["metrics"]["final_train_loss"]
            rel_gain = round(((base_loss - val_loss) / base_loss) * 100, 2)
            print(f"Result: Validation Loss = {val_loss:.4f} (Train Loss = {train_loss:.4f}, Rel Gain = {rel_gain:+0.2f}%)")

            # Add to / update population
            ind_record = {"genes": candidate, "val_loss": val_loss, "hypo_id": hypo_id}
            if len(population) < population_size:
                population.append(ind_record)
            else:
                # Replace worst in population if child is better (steady-state GA)
                population.sort(key=lambda ind: ind["val_loss"])
                if val_loss < population[-1]["val_loss"]:
                    population[-1] = ind_record

            if val_loss > 3.0 * base_loss:
                consecutive_divergences += 1
                status = "DIVERGED"
                decision = "REJECT_AND_BACKTRACK"
                print(f"[WARNING] Divergence! Val Loss {val_loss:.4f} > 3x Base.")
            else:
                consecutive_divergences = 0
                status = "SUCCESS"
                if val_loss < logger.incumbent_best_loss:
                    decision = "ACCEPT_AND_EXTEND"
                    active_parent_hypo = hypo_id
                    print(f"[IMPROVEMENT] Discovered new incumbent best loss: {val_loss:.4f}!")
                else:
                    decision = "REJECT_AND_BACKTRACK"
                    print(f"[REJECT] Val Loss {val_loss:.4f} did not beat incumbent {logger.incumbent_best_loss:.4f}.")
        else:
            val_loss = None
            train_loss = None
            status = "FAILED"
            decision = "REJECT_AND_BACKTRACK"
            consecutive_divergences += 1

        run_id, is_best = logger.add_execution_run(
            iteration_idx=it,
            exp_id=exp_id,
            status=status,
            exit_code=res.get("exit_code", 1),
            gpu_seconds=res["elapsed_seconds"],
            val_loss=val_loss,
            train_loss=train_loss
        )

        refl_text = f"GA Individual {it} outcome: {status} with decision {decision}."
        logger.add_reflection(
            iteration_idx=it,
            run_id=run_id,
            hypo_id=hypo_id,
            reflection_text=refl_text,
            decision=decision,
            self_detected_failure=(status == "DIVERGED" or status == "FAILED"),
            cycle_sec=res["elapsed_seconds"]
        )

        trace_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "execution_traces",
            f"trace_{experiment_id}.json"
        )
        logger.export_trace(trace_path)

    summary = logger.export_trace(trace_path)
    print("\n=== EVOLUTIONARY (GA) SEARCH BASELINE COMPLETE ===")
    print(f"Total Iterations: {summary['summary_statistics']['total_iterations']}")
    print(f"Dead-End Ratio: {summary['summary_statistics']['dead_end_ratio']}")
    print(f"Path to First Improvement: {summary['summary_statistics']['path_to_first_improvement_steps']}")
    print(f"Final Best Loss: {summary['summary_statistics']['final_best_loss']}")
    print(f"Final Relative Gain: {summary['summary_statistics']['final_relative_improvement_pct']}%")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arm 4 Evolutionary (GA) Search Baseline")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--task", type=str, default="dyck", choices=["dyck", "fsm", "parity_legacy"])
    args = parser.parse_args()

    run_evolutionary_search(
        dry_run=args.dry_run,
        max_iterations=args.iterations,
        population_size=args.population_size,
        budget_hours=args.budget_hours,
        seed=args.seed,
        task=args.task
    )
