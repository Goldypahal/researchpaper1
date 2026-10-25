#!/usr/bin/env python3
"""
scripts/run_comparative_search.py

The Unified, Rigorous, Resumable Comparative Search Runner.
Implements the full decomposition matrix and eliminates historical confounds:
  - Real Bayesian TPE (n_startup_trials=3, multivariate=True)
  - Real GA and Random Search over single-source-of-truth search space
  - Prior-Sampler and TPE-Warm-Start arms
  - Decomposition arms: Named vs Anonymized, Real vs Shuffled vs Zero-Shot, Obfuscated
  - Atomic trace writes with tempfiles and fsync
  - Zero-crash failure handling: failures yield status tagging and val_loss=None
  - Unbiased post-search evaluation: R=5 fresh training seeds on holdout test set
  - Append-only registry.jsonl logging
"""

import os
import sys
import copy
import time
import json
import math
import random
import hashlib
import subprocess
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import optuna
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

from search_space import (
    SEARCH_SPACE_SPEC,
    sample_random_candidate,
    sample_from_optuna,
    get_default_baseline,
    validate_proposal,
    sanitize_configuration
)
from evaluator.immutable_evaluator import ImmutableEvaluator
from agent_scaffold.llm_agent import LLMSearchAgent


def get_code_version() -> str:
    """Returns real git commit hash or SHA-256 fallback of Python codebase."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0
        ).decode("utf-8").strip()
        if commit:
            return commit
    except Exception:
        pass

    # Fallback SHA-256
    hasher = hashlib.sha256()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py_files = []
    for root, _, files in os.walk(root_dir):
        if ".git" in root or ".venv" in root or "__pycache__" in root:
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    for pf in sorted(py_files):
        try:
            with open(pf, "rb") as f:
                hasher.update(f.read())
        except Exception:
            pass
    return f"sha256_{hasher.hexdigest()[:16]}"


def write_trace_atomic(filepath: str, payload: Dict[str, Any]):
    """Writes JSON payload atomically to avoid corrupt or half-written traces on interruptions."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = filepath + f".tmp_{os.getpid()}_{int(time.time()*1000)}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, filepath)


def append_registry_entry(registry_path: str, record: Dict[str, Any]):
    """Appends finished trajectory summary to append-only registry.jsonl."""
    os.makedirs(os.path.dirname(os.path.abspath(registry_path)), exist_ok=True)
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


class ComparativeSearchRunner:
    def __init__(
        self,
        task: str = "dyck",
        seed: int = 42,
        iterations: int = 10,
        steps_per_candidate: int = 50,
        eval_timeout: float = 120.0,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        traces_dir: Optional[str] = None,
        registry_path: Optional[str] = None,
        retrain_seeds: int = 5
    ):
        self.task = task.lower()
        self.seed = seed
        self.iterations = iterations
        self.steps_per_candidate = steps_per_candidate
        self.eval_timeout = eval_timeout
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.retrain_seeds = retrain_seeds

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.traces_dir = traces_dir or os.path.join(base_dir, "traces")
        self.registry_path = registry_path or os.path.join(base_dir, "registry.jsonl")
        os.makedirs(self.traces_dir, exist_ok=True)

        self.evaluator = ImmutableEvaluator(task=self.task, seed=self.seed)
        self.code_version = get_code_version()

    def generate_run_id(
        self,
        arm: str,
        framing: str = "named",
        feedback: str = "real",
        obfuscated: bool = False
    ) -> str:
        """Constructs unique run_id distinguishing all arms and decomposition conditions."""
        parts = [self.task, arm]
        if arm == "llm":
            parts.extend([framing, feedback])
            if obfuscated:
                parts.append("obfuscated")
        return f"{'_'.join(parts)}__seed{self.seed}"

    def run_trajectory(
        self,
        arm: str = "random",
        task_framing: str = "named",
        feedback: str = "real",
        obfuscated: bool = False,
        proposal_pool: Optional[List[Dict[str, Any]]] = None,
        skip_if_valid: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a single search trajectory with full audit, atomic writes, and unbiased test reporting.
        """
        run_id = self.generate_run_id(arm, framing=task_framing, feedback=feedback, obfuscated=obfuscated)
        trace_path = os.path.join(self.traces_dir, f"trace_{run_id}.json")
        jsonl_path = os.path.join(self.traces_dir, f"trajectory_{run_id}.jsonl")

        # Skip logic
        if skip_if_valid and os.path.exists(trace_path):
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if (
                    existing.get("status") == "COMPLETE_VALID"
                    and existing.get("code_version") == self.code_version
                    and len(existing.get("trajectory", [])) == self.iterations + 1
                ):
                    print(f"[{run_id}] Trace already complete and verified. Skipping.")
                    return existing
            except Exception:
                pass

        print(f"\n{'='*70}\n[STARTING RUN] run_id: {run_id}\n{'='*70}", flush=True)
        started_at = time.time()
        rng = random.Random(self.seed)

        # 1. Step 0 Baseline
        baseline_cfg = get_default_baseline()
        base_res = self.evaluator.train_and_evaluate_candidate(
            config=baseline_cfg,
            max_steps=self.steps_per_candidate,
            timeout_seconds=self.eval_timeout,
            train_seed=self.seed
        )

        history: List[Dict[str, Any]] = [{
            "iteration": 0,
            "config": baseline_cfg,
            "status": "OK",
            "train_loss": base_res["train_loss"],
            "val_loss": base_res["val_loss"],
            "test_loss": base_res["test_loss"],
            "ood_loss": base_res["ood_loss"],
            "val_token_acc": base_res["val_token_acc"],
            "val_struct_acc": base_res["val_struct_acc"],
            "ood_struct_acc": base_res["ood_struct_acc"],
            "gpu_seconds": base_res["gpu_seconds"],
            "cumulative_gpu_sec": base_res["gpu_seconds"],
            "best_val_loss": base_res["val_loss"],
            "incumbent_config": copy.deepcopy(baseline_cfg),
            "hypothesis": "Default Baseline Architecture",
            "predicted_delta_loss": 0.0,
            "actual_delta_loss": 0.0
        }]

        cum_train_gpu_sec = base_res["gpu_seconds"]
        cum_llm_sec = 0.0
        incumbent_val_loss = base_res["val_loss"]
        incumbent_config = copy.deepcopy(baseline_cfg)

        # 2. Setup arm-specific state
        study = None
        population = None
        pop_scores = None
        llm_agent = None

        if arm in ("tpe", "tpe_warm"):
            # Confound 4 fix: real TPE with n_startup_trials=3, multivariate=True
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=3, multivariate=True)
            )
            # If warm-started, enqueue top zero-shot proposals
            if arm == "tpe_warm" and proposal_pool:
                for warm_p in proposal_pool[:3]:
                    mods = warm_p.get("modifications", warm_p)
                    try:
                        clean_warm = sanitize_configuration(mods)
                        study.enqueue_trial(clean_warm)
                    except Exception as e:
                        print(f"Warning: could not enqueue warm proposal: {e}")

        elif arm == "evolutionary":
            # Real GA population
            pop_size = 4
            population = [copy.deepcopy(baseline_cfg)]
            for _ in range(pop_size - 1):
                population.append(sample_random_candidate(rng))
            pop_scores = [incumbent_val_loss] + [99.0] * (pop_size - 1)

        elif arm == "llm":
            allow_sim = os.environ.get("ALLOW_LLM_SIMULATION", "0").lower() in ("1", "true", "yes")
            effective_provider = "adaptive_simulation" if (allow_sim and (not self.llm_provider or self.llm_provider in ("local", "simulation"))) else self.llm_provider
            llm_agent = LLMSearchAgent(
                provider=effective_provider,
                model=self.llm_model,
                strict=True,
                allow_simulation=allow_sim,
                task_framing=task_framing,
                feedback=feedback,
                obfuscated=obfuscated
            )

        # Clear/initialize live jsonl log
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            jf.write(json.dumps(history[0], default=str) + "\n")

        # 3. Main Search Trajectory Loop
        crashed_iterations = 0
        invalid_proposals = 0

        for it in range(1, self.iterations + 1):
            it_start = time.time()
            trial = None
            cand_cfg = None
            hypo_text = f"Candidate #{it}"
            pred_delta = 0.0
            prop_metadata = {}
            prop_status = "OK"

            incumbent_before = incumbent_val_loss

            try:
                if arm == "random":
                    cand_cfg = sample_random_candidate(rng)
                    hypo_text = f"Uniform random candidate #{it}"

                elif arm in ("tpe", "tpe_warm"):
                    trial = study.ask()
                    cand_cfg = sample_from_optuna(trial)
                    hypo_text = f"TPE proposal #{it}"

                elif arm == "evolutionary":
                    # Tournament selection and mutation
                    p1_idx = rng.randint(0, len(population) - 1)
                    p2_idx = rng.randint(0, len(population) - 1)
                    parent_idx = p1_idx if pop_scores[p1_idx] <= pop_scores[p2_idx] else p2_idx
                    parent = population[parent_idx]

                    cand_cfg = copy.deepcopy(parent)
                    # Mutate 1-2 parameters
                    keys_to_mutate = rng.sample(list(SEARCH_SPACE_SPEC.keys()), k=rng.randint(1, 2))
                    for k in keys_to_mutate:
                        spec_k = SEARCH_SPACE_SPEC[k]
                        if spec_k["type"] == "float":
                            scale = 0.2 * (spec_k["high"] - spec_k["low"])
                            cand_cfg[k] = max(spec_k["low"], min(spec_k["high"], cand_cfg[k] + rng.gauss(0, scale)))
                        else:
                            cand_cfg[k] = rng.choice(spec_k["choices"])
                    cand_cfg = sanitize_configuration(cand_cfg)
                    hypo_text = f"GA mutant #{it} from parent {parent_idx}"

                elif arm == "prior_sampler":
                    if not proposal_pool:
                        cand_cfg = sample_random_candidate(rng)
                        hypo_text = f"Prior pool empty fallback #{it}"
                    else:
                        pool_item = proposal_pool[(it - 1) % len(proposal_pool)]
                        cand_cfg = sanitize_configuration(pool_item.get("modifications", pool_item))
                        hypo_text = pool_item.get("hypothesis_text", f"Prior sample #{it}")

                elif arm == "llm":
                    t_llm = time.time()
                    proposal = llm_agent.propose_experiment(
                        iteration_idx=it,
                        history=history,
                        incumbent_best_loss=incumbent_val_loss,
                        base_loss=base_res["val_loss"],
                        task=self.task,
                        seed=self.seed
                    )
                    llm_dur = time.time() - t_llm
                    cum_llm_sec += llm_dur

                    hypo_text = proposal.get("hypothesis_text", f"LLM hypothesis #{it}")
                    pred_delta = proposal.get("predicted_delta_loss", 0.0)
                    prop_status = proposal.get("status", "OK")
                    prop_metadata = {
                        "provider": proposal.get("provider"),
                        "model": proposal.get("model"),
                        "parse_status": proposal.get("parse_status"),
                        "parse_attempts": proposal.get("parse_attempts"),
                        "latency_seconds": proposal.get("latency_seconds"),
                        "token_usage": proposal.get("token_usage", {}),
                        "violations": proposal.get("violations", [])
                    }

                    if not proposal.get("is_valid", False):
                        invalid_proposals += 1
                        prop_status = "INVALID_PROPOSAL"
                        # Invalid proposal consumes an iteration without spending GPU compute
                        cand_cfg = None
                    else:
                        cand_cfg = sanitize_configuration(proposal.get("modifications", {}))

            except Exception as e:
                print(f"  [Iteration {it}] Proposal generation error: {e}")
                prop_status = f"PROPOSAL_FAILED:{type(e).__name__}"
                crashed_iterations += 1
                cand_cfg = None

            # 4. Evaluation Stage
            eval_res = None
            if cand_cfg is not None:
                try:
                    eval_res = self.evaluator.train_and_evaluate_candidate(
                        config=cand_cfg,
                        max_steps=self.steps_per_candidate,
                        timeout_seconds=self.eval_timeout,
                        train_seed=self.seed
                    )
                    cum_train_gpu_sec += eval_res["gpu_seconds"]
                except TimeoutError:
                    prop_status = "TIMEOUT"
                    crashed_iterations += 1
                except Exception as e:
                    prop_status = f"CRASH:{type(e).__name__}"
                    crashed_iterations += 1

            # Update search arms
            val_loss = None
            if eval_res is not None and prop_status == "OK":
                val_loss = eval_res["val_loss"]
                if study is not None and trial is not None:
                    study.tell(trial, val_loss)
                if population is not None:
                    worst_idx = max(range(len(pop_scores)), key=lambda i: pop_scores[i])
                    if val_loss < pop_scores[worst_idx]:
                        population[worst_idx] = copy.deepcopy(cand_cfg)
                        pop_scores[worst_idx] = val_loss
                if val_loss < incumbent_val_loss:
                    incumbent_val_loss = val_loss
                    incumbent_config = copy.deepcopy(cand_cfg)
            else:
                if study is not None and trial is not None:
                    study.tell(trial, state=TrialState.FAIL)

            # Measure actual delta relative to incumbent before this trial
            actual_delta = round(incumbent_before - val_loss, 6) if val_loss is not None else 0.0

            step_entry = {
                "iteration": it,
                "config": cand_cfg,
                "status": prop_status,
                "train_loss": eval_res["train_loss"] if eval_res else None,
                "val_loss": val_loss,
                "test_loss": eval_res["test_loss"] if eval_res else None,
                "ood_loss": eval_res["ood_loss"] if eval_res else None,
                "val_token_acc": eval_res["val_token_acc"] if eval_res else 0.0,
                "val_struct_acc": eval_res["val_struct_acc"] if eval_res else 0.0,
                "ood_struct_acc": eval_res["ood_struct_acc"] if eval_res else 0.0,
                "gpu_seconds": eval_res["gpu_seconds"] if eval_res else 0.0,
                "cumulative_train_gpu_sec": round(cum_train_gpu_sec, 3),
                "cumulative_llm_sec": round(cum_llm_sec, 3),
                "cumulative_wall_sec": round(cum_train_gpu_sec + cum_llm_sec, 3),
                "best_val_loss": incumbent_val_loss,
                "incumbent_config": copy.deepcopy(incumbent_config),
                "hypothesis": hypo_text,
                "predicted_delta_loss": pred_delta,
                "actual_delta_loss": actual_delta,
                "proposal_metadata": prop_metadata
            }
            history.append(step_entry)

            # Append to live jsonl log
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(step_entry, default=str) + "\n")

            print(
                f"  Iter {it:02d}/{self.iterations}: status={prop_status:<16s} "
                f"val={str(val_loss):<8s} best={incumbent_val_loss:.4f} "
                f"(train_gpu={cum_train_gpu_sec:.1f}s, llm={cum_llm_sec:.1f}s)",
                flush=True
            )

        # 5. Unbiased Multi-Seed Holdout Evaluation (Phase 2.2)
        print(f"\n[{run_id}] Retraining incumbent and baseline across {self.retrain_seeds} fresh seeds...", flush=True)
        incumbent_retrain = self.evaluator.retrain_and_evaluate_incumbent(
            config=incumbent_config,
            r_seeds=self.retrain_seeds,
            seed_offset=10000,
            max_steps=self.steps_per_candidate
        )
        baseline_retrain = self.evaluator.retrain_and_evaluate_incumbent(
            config=baseline_cfg,
            r_seeds=self.retrain_seeds,
            seed_offset=10000,
            max_steps=self.steps_per_candidate
        )

        unbiased_gain_test = round(baseline_retrain["test_loss_mean"] - incumbent_retrain["test_loss_mean"], 6)
        unbiased_gain_pct = round((unbiased_gain_test / max(1e-6, baseline_retrain["test_loss_mean"])) * 100.0, 2)

        total_wall_sec = round(time.time() - started_at, 3)
        status_flag = "FAILED" if crashed_iterations > 2 else "COMPLETE_VALID"

        final_payload = {
            "run_id": run_id,
            "status": status_flag,
            "code_version": self.code_version,
            "task": self.task,
            "arm": arm,
            "task_framing": task_framing,
            "feedback": feedback,
            "obfuscated": obfuscated,
            "seed": self.seed,
            "iterations": self.iterations,
            "steps_per_candidate": self.steps_per_candidate,
            "crashed_iterations": crashed_iterations,
            "invalid_proposals": invalid_proposals,
            "baseline_val_loss": base_res["val_loss"],
            "best_val_loss": incumbent_val_loss,
            "incumbent_config": incumbent_config,
            "unbiased_baseline_test_loss": baseline_retrain["test_loss_mean"],
            "unbiased_incumbent_test_loss": incumbent_retrain["test_loss_mean"],
            "unbiased_incumbent_test_std": incumbent_retrain["test_loss_std"],
            "unbiased_gain_test": unbiased_gain_test,
            "unbiased_gain_pct": unbiased_gain_pct,
            "baseline_retrain": baseline_retrain,
            "incumbent_retrain": incumbent_retrain,
            "cum_train_gpu_sec": round(cum_train_gpu_sec, 3),
            "cum_llm_sec": round(cum_llm_sec, 3),
            "total_wall_sec": total_wall_sec,
            "trajectory": history
        }

        # Write atomic trace
        write_trace_atomic(trace_path, final_payload)

        # Append registry record
        append_registry_entry(self.registry_path, {
            "run_id": run_id,
            "status": status_flag,
            "task": self.task,
            "arm": arm,
            "task_framing": task_framing,
            "feedback": feedback,
            "seed": self.seed,
            "unbiased_incumbent_test_loss": incumbent_retrain["test_loss_mean"],
            "unbiased_gain_pct": unbiased_gain_pct,
            "cum_train_gpu_sec": round(cum_train_gpu_sec, 3),
            "cum_llm_sec": round(cum_llm_sec, 3),
            "code_version": self.code_version,
            "timestamp": time.time()
        })

        print(
            f"[{run_id}] COMPLETE. Unbiased Test Loss: {incumbent_retrain['test_loss_mean']:.4f} "
            f"(Gain: {unbiased_gain_pct:+.2f}%) | Trace: {trace_path}\n"
        )
        return final_payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run rigorous comparative architecture search.")
    parser.add_argument("--task", type=str, default="dyck", choices=["dyck", "fsm", "abs_position"])
    parser.add_argument("--arm", type=str, default="random", choices=["random", "tpe", "tpe_warm", "evolutionary", "prior_sampler", "llm"])
    parser.add_argument("--framing", type=str, default="named", choices=["named", "anonymized"])
    parser.add_argument("--feedback", type=str, default="real", choices=["real", "none", "shuffled"])
    parser.add_argument("--obfuscated", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--provider", type=str, default="local")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    runner = ComparativeSearchRunner(
        task=args.task,
        seed=args.seed,
        iterations=args.iterations,
        steps_per_candidate=args.steps,
        llm_provider=args.provider,
        llm_model=args.model
    )
    runner.run_trajectory(
        arm=args.arm,
        task_framing=args.framing,
        feedback=args.feedback,
        obfuscated=args.obfuscated
    )
