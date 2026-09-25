"""
Phase 8: Main Comparative Search Benchmark.
Evaluates four search paradigms under strict EQUAL-COMPUTE BUDGET constraints:
  1. Random Search (Uniform / Log-Uniform stochastic baseline)
  2. Bayesian Optimization (TPE via Optuna)
  3. Evolutionary Algorithm (Genetic Algorithm with selection, crossover, mutation)
  4. LLM Search Agent (Semantic hypothesis generation, predicted delta loss, reflection)

All methods are evaluated by the same IMMUTABLE EVALUATOR on:
  - Benchmark 1: True Dyck-4 (Context-Free Grammar with stack depth hierarchy)
  - Benchmark 2: Hidden FSM (Regular language with algebraic latent state space)

Tracks:
  - Best-found Validation Loss (L*_val)
  - Out-of-Distribution Generalization Loss (L*_OOD)
  - Search Efficiency / Area Under the Search Curve (AUC)
  - Hypothesis Calibration (MAE between predicted and actual improvement)
  - Full trajectory traces exported to EXPERIMENT_REGISTRY.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import math
import json
import random
import copy
from typing import Dict, Any, List, Optional
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluator.immutable_evaluator import ImmutableEvaluator
from search_space import (
    SEARCH_SPACE_SPEC,
    sanitize_configuration,
    sample_random_candidate,
    get_default_baseline
)
from scientific_stats.statistical_analysis import (
    summarize_distribution,
    compare_two_search_arms,
    calculate_normalized_gain_auc,
    calculate_auc_search_curve,
    evaluate_hypothesis_calibration
)
from agent_scaffold.llm_agent import LLMResearchAgent


class ComparativeSearchBenchmark:
    def __init__(
        self,
        task: str = "dyck",
        seed: int = 42,
        max_iterations: int = 10,
        steps_per_candidate: int = 60,
        eval_timeout: float = 300.0,
        llm_provider: str = None,
        llm_model: str = None
    ):
        self.task = task.lower()
        self.seed = seed
        self.max_iterations = max_iterations
        self.steps_per_candidate = steps_per_candidate
        self.eval_timeout = eval_timeout
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.evaluator = ImmutableEvaluator(task=self.task, seed=self.seed)

    def run_trajectory(self, method: str, llm_mode: str = "full") -> Dict[str, Any]:
        """Runs a single optimization trajectory for the specified method."""
        method = method.lower()
        rng = random.Random(self.seed)
        
        # Step 0: Baseline anchor with timeout resiliency
        baseline_cfg = get_default_baseline()
        try:
            base_res = self.evaluator.train_and_evaluate_candidate(
                config=baseline_cfg,
                max_steps=self.steps_per_candidate,
                timeout_seconds=self.eval_timeout
            )
        except TimeoutError:
            print(f"  [Seed {self.seed}] Step 0 baseline timed out (> {self.eval_timeout}s). Evaluating with reduced steps.")
            base_res = self.evaluator.train_and_evaluate_candidate(
                config=baseline_cfg,
                max_steps=max(5, self.steps_per_candidate // 2),
                timeout_seconds=self.eval_timeout
            )
        
        history: List[Dict[str, Any]] = [{
            "iteration": 0,
            "config": baseline_cfg,
            "train_loss": base_res["train_loss"],
            "val_loss": base_res["val_loss"],
            "ood_loss": base_res["ood_loss"],
            "val_struct_acc": base_res["val_struct_acc"],
            "ood_struct_acc": base_res["ood_struct_acc"],
            "gap_rel": base_res["generalization_gap_rel"],
            "gpu_seconds": base_res["gpu_seconds"],
            "cumulative_gpu_sec": base_res["gpu_seconds"],
            "best_val_loss": base_res["val_loss"],
            "hypothesis": "Default Baseline Transformer",
            "predicted_delta_loss": 0.0,
            "actual_delta_loss": 0.0
        }]
        
        cum_gpu_sec = base_res["gpu_seconds"]
        cum_llm_sec = 0.0
        best_val = base_res["val_loss"]
        best_ood = base_res["ood_loss"]

        # Setup method-specific state
        if method == "tpe":
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.seed)
            )
            # Seed Optuna with baseline point
            # (Optuna will suggest from trial 1)
        elif method == "evolutionary":
            # Population initialized with baseline + random candidates
            pop_size = 4
            population: List[Dict[str, Any]] = [copy.deepcopy(baseline_cfg)]
            for _ in range(pop_size - 1):
                population.append(sample_random_candidate(rng))
            pop_scores = [best_val] + [99.0] * (pop_size - 1)
        elif method == "llm":
            # STRICT MODE: Real LLM reasoning required for empirical research.
            # Simulation fallback is strictly forbidden unless explicitly enabled.
            allow_sim = os.environ.get("ALLOW_LLM_SIMULATION", "0").lower() in ("1", "true", "yes")
            agent = LLMResearchAgent(
                provider=self.llm_provider,
                model=self.llm_model,
                strict=True,
                allow_simulation=allow_sim,
                reasoning_mode=llm_mode
            )

        # Iterations 1 to K
        for it in range(1, self.max_iterations + 1):
            pred_delta = 0.0
            hypo_text = ""
            prompt_hash = None
            raw_response = None
            proposal_metadata = {}

            if method == "random":
                cand_cfg = sample_random_candidate(rng)
                hypo_text = f"Uniform random sample #{it}"

            elif method == "tpe":
                trial = study.ask()
                cand_cfg = self._sample_from_optuna(trial)
                hypo_text = f"TPE expected improvement proposal #{it}"

            elif method == "evolutionary":
                cand_cfg = self._sample_evolutionary_candidate(population, pop_scores, rng)
                hypo_text = f"Genetic mutation & crossover candidate #{it}"

            elif method == "llm":
                try:
                    proposal = agent.propose_experiment(
                        iteration_idx=it,
                        history=history,
                        incumbent_best_loss=best_val,
                        base_loss=base_res["val_loss"],
                        task=self.task
                    )
                    # Integrity assertion: reject scripted simulation in strict mode
                    if not allow_sim and proposal.get("provider") == "adaptive_simulation":
                        raise RuntimeError(
                            f"FATAL INTEGRITY VIOLATION: Iteration {it} returned provider='adaptive_simulation'. "
                            f"Real LLM reasoning is required for empirical research."
                        )

                    cand_cfg = sanitize_configuration(proposal.get("modifications", {}))
                    # Fill any missing keys from baseline
                    for k, v in baseline_cfg.items():
                        if k not in cand_cfg:
                            cand_cfg[k] = v
                    cand_cfg = sanitize_configuration(cand_cfg)
                    hypo_text = proposal.get("hypothesis", proposal.get("hypothesis_text", f"LLM hypothesis #{it}"))
                    pred_delta = float(proposal.get("predicted_delta_loss", 0.0))
                    prompt_hash = proposal.get("prompt_hash")
                    raw_response = proposal.get("raw_response")
                    proposal_metadata = {
                        "prompt_hash": prompt_hash,
                        "raw_response": raw_response,
                        "provider": str(proposal.get("provider", "unknown")),
                        "model": str(proposal.get("model", self.llm_model or "local")),
                        "reasoning_mode": str(proposal.get("reasoning_mode", "full")),
                        "reasoning_effort": proposal.get("reasoning_effort"),
                        "token_usage": proposal.get("token_usage", {}),
                        "latency_seconds": proposal.get("latency_seconds"),
                        "critic_prompt_hash": proposal.get("critic_prompt_hash"),
                        "parse_audit": proposal.get("parse_audit", {})
                    }
                except Exception as e:
                    print(f"    [Iteration {it}] FATAL LLM PROPOSAL ERROR: {type(e).__name__}: {e}")
                    raise RuntimeError(f"LLM Agent proposal failed at iteration {it}: {e}") from e

            # Immutable Evaluation with robust failure recovery
            try:
                eval_res = self.evaluator.train_and_evaluate_candidate(
                    config=cand_cfg,
                    max_steps=self.steps_per_candidate,
                    timeout_seconds=self.eval_timeout
                )
            except TimeoutError as te:
                print(f"    [Iteration {it}] TIMEOUT: candidate exceeded budget limit ({self.eval_timeout}s).")
                eval_res = {
                    "train_loss": 99.0,
                    "val_loss": 99.0,
                    "ood_loss": 99.0,
                    "val_struct_acc": 0.0,
                    "ood_struct_acc": 0.0,
                    "generalization_gap_rel": 0.0,
                    "gpu_seconds": round(self.eval_timeout, 3),
                    "status": "TIMEOUT"
                }
            except Exception as e:
                print(f"    [Iteration {it}] CRASH: {type(e).__name__}: {e}")
                eval_res = {
                    "train_loss": 99.0,
                    "val_loss": 99.0,
                    "ood_loss": 99.0,
                    "val_struct_acc": 0.0,
                    "ood_struct_acc": 0.0,
                    "generalization_gap_rel": 0.0,
                    "gpu_seconds": 1.0,
                    "status": f"FAILED_{type(e).__name__}"
                }

            # Update Optuna if TPE
            if method == "tpe":
                study.tell(trial, eval_res["val_loss"])

            # Update GA population if evolutionary
            if method == "evolutionary":
                worst_idx = int(np_argmax := max(range(len(pop_scores)), key=lambda i: pop_scores[i]))
                if eval_res["val_loss"] < pop_scores[worst_idx]:
                    population[worst_idx] = copy.deepcopy(cand_cfg)
                    pop_scores[worst_idx] = eval_res["val_loss"]

            val_loss = eval_res["val_loss"]
            ood_loss = eval_res["ood_loss"]
            actual_delta = round(base_res["val_loss"] - val_loss, 6)

            if val_loss < best_val:
                best_val = val_loss
                best_ood = ood_loss

            candidate_train_sec = eval_res["gpu_seconds"]
            iter_llm_sec = float(proposal_metadata.get("latency_seconds", 0.0) or 0.0) if proposal_metadata else 0.0
            cum_llm_sec += iter_llm_sec
            cum_gpu_sec += candidate_train_sec

            h_entry = {
                "iteration": it,
                "config": cand_cfg,
                "train_loss": eval_res["train_loss"],
                "val_loss": val_loss,
                "ood_loss": ood_loss,
                "val_struct_acc": eval_res["val_struct_acc"],
                "ood_struct_acc": eval_res["ood_struct_acc"],
                "gap_rel": eval_res["generalization_gap_rel"],
                "candidate_train_gpu_sec": candidate_train_sec,
                "llm_inference_sec": iter_llm_sec,
                "eval_wall_clock_sec": candidate_train_sec,
                "gpu_seconds": candidate_train_sec,
                "cumulative_wall_clock_sec": round(cum_gpu_sec + cum_llm_sec, 3),
                "cumulative_gpu_sec": round(cum_gpu_sec, 3),
                "cumulative_llm_sec": round(cum_llm_sec, 3),
                "best_val_loss": best_val,
                "hypothesis": hypo_text,
                "predicted_delta_loss": pred_delta,
                "actual_delta_loss": actual_delta,
                "prompt_hash": prompt_hash,
                "raw_response": raw_response
            }
            if proposal_metadata:
                h_entry["proposal_metadata"] = proposal_metadata
            history.append(h_entry)

        # Calculate Search Efficiency (Normalized Performance Gain AUC - Higher is Better)
        wall_times = [h["cumulative_gpu_sec"] for h in history]
        best_losses = [h["best_val_loss"] for h in history]
        auc_gain = calculate_normalized_gain_auc(wall_times, best_losses, base_res["val_loss"])
        raw_loss_auc = calculate_auc_search_curve(wall_times, best_losses)

        # Extended Hypothesis Calibration for LLM
        calibration_report = None
        if method == "llm":
            preds = [h["predicted_delta_loss"] for h in history[1:]]
            actuals = [h["actual_delta_loss"] for h in history[1:]]
            calibration_report = evaluate_hypothesis_calibration(preds, actuals)

        calib_mae = None
        if calibration_report and isinstance(calibration_report, dict) and "mae" in calibration_report:
            calib_mae = calibration_report["mae"]

        # Write trajectory.jsonl alongside trace
        traces_dir = os.path.join(os.path.dirname(__file__), "traces")
        os.makedirs(traces_dir, exist_ok=True)
        jsonl_filename = f"trajectory_{self.task}_{method}_seed_{self.seed}.jsonl"
        if method == "llm" and llm_mode != "full":
            jsonl_filename = f"trajectory_{self.task}_{method}_{llm_mode}_seed_{self.seed}.jsonl"
        jsonl_path = os.path.join(traces_dir, jsonl_filename)
        with open(jsonl_path, "w") as jf:
            for entry in history:
                row = {
                    "task": self.task,
                    "method": method,
                    "seed": self.seed,
                    "iteration": entry["iteration"],
                    "config": entry["config"],
                    "val_loss": entry["val_loss"],
                    "ood_loss": entry["ood_loss"],
                    "eval_wall_clock_sec": entry.get("eval_wall_clock_sec", entry.get("gpu_seconds")),
                    "best_val_loss": entry["best_val_loss"],
                    "hypothesis": entry["hypothesis"],
                    "predicted_delta_loss": entry["predicted_delta_loss"],
                    "actual_delta_loss": entry["actual_delta_loss"],
                    "prompt_hash": entry.get("prompt_hash")
                }
                if "proposal_metadata" in entry:
                    row["proposal_metadata"] = entry["proposal_metadata"]
                jf.write(json.dumps(row, default=str) + "\n")

        return {
            "method": method,
            "task": self.task,
            "seed": self.seed,
            "baseline_val_loss": base_res["val_loss"],
            "best_val_loss": best_val,
            "best_ood_loss": best_ood,
            "improvement_pct": round(((base_res["val_loss"] - best_val) / base_res["val_loss"]) * 100.0, 2),
            "candidate_train_gpu_sec": round(cum_gpu_sec, 3),
            "llm_inference_sec": round(cum_llm_sec, 3),
            "total_search_wall_clock_sec": round(cum_gpu_sec + cum_llm_sec, 3),
            "total_eval_wall_clock_sec": round(cum_gpu_sec, 3),
            "total_gpu_seconds": round(cum_gpu_sec, 3),
            "auc_normalized_gain": auc_gain,
            "raw_loss_auc": raw_loss_auc,
            "auc_search_curve": auc_gain,
            "calibration_report": calibration_report,
            "calibration_mae": calib_mae,
            "trajectory": history
        }

    def _sample_from_optuna(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Maps Optuna search spaces to candidate configuration."""
        lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2)
        d_model = trial.suggest_categorical("d_model", [128, 256, 512])
        n_layers = trial.suggest_categorical("n_layers", [2, 4, 6, 8])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024, 2048])
        activation = trial.suggest_categorical("activation", ["gelu", "relu", "silu"])
        norm_type = trial.suggest_categorical("norm_type", ["layernorm", "rmsnorm"])
        pos_encoding = trial.suggest_categorical("pos_encoding", ["learned", "sinusoidal", "rotary", "none"])
        ffn_type = trial.suggest_categorical("ffn_type", ["standard", "swiglu"])
        topology = trial.suggest_categorical("topology", ["pre_ln", "post_ln", "parallel"])
        scale_choice = trial.suggest_categorical("scale_factor", ["none", "0.05", "0.1", "0.15", "0.25"])
        scale_factor = None if scale_choice == "none" else float(scale_choice)

        cand = {
            "lr": round(lr, 6),
            "weight_decay": round(weight_decay, 4),
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "activation": activation,
            "norm_type": norm_type,
            "pos_encoding": pos_encoding,
            "ffn_type": ffn_type,
            "topology": topology,
            "scale_factor": scale_factor,
            "n_kv_heads": "same"
        }
        return sanitize_configuration(cand)

    def _sample_evolutionary_candidate(
        self,
        population: List[Dict[str, Any]],
        scores: List[float],
        rng: random.Random,
        mutation_rate: float = 0.3
    ) -> Dict[str, Any]:
        """Tournament selection followed by uniform crossover and point mutation."""
        # 2-way tournament
        p1 = min(rng.sample(range(len(population)), 2), key=lambda i: scores[i])
        p2 = min(rng.sample(range(len(population)), 2), key=lambda i: scores[i])
        parent1, parent2 = population[p1], population[p2]

        child = {}
        # Crossover
        for k in parent1.keys():
            child[k] = parent1[k] if rng.random() < 0.5 else parent2[k]

        # Mutation
        random_cand = sample_random_candidate(rng)
        for k in child.keys():
            if rng.random() < mutation_rate:
                child[k] = random_cand[k]

        return sanitize_configuration(child)
