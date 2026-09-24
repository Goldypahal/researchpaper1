"""
Autonomous Research Agent Outer-Loop Meta-Optimizer.
Orchestrates:
1. LLM-driven hypothesis formulation (Claude / GPT-4o), STRICT by default
   -- no silent fallback to scripted simulation without explicit opt-in
2. Quantitative parameter mutation specification
3. Sandbox trial execution (via sandbox_runner)
4. Empirical evaluation (via eval_harness)
5. History accumulation and feedback loop
6. Trace graph logging conforming to trace_schema.json, including which
   provider (real model or simulation) produced each hypothesis
7. Four exact stopping conditions enforcement
"""

import os
import time
import argparse
from trace_logger import TraceLogger
from sandbox_runner import execute_in_sandbox
from llm_agent import LLMResearchAgent, SimulationDisallowedError


def run_agent_loop(experiment_id="exp_001_transformer_opt", arm="Arm1_AutonomousAgent",
                   dry_run=False, max_iterations=5, budget_hours=10.0,
                   llm_provider=None, llm_model=None, api_key=None, allow_simulation=False,
                   task="dyck", reasoning_mode="full"):
    print(f"=== INITIALIZING AUTONOMOUS RESEARCH AGENT SCAFFOLD ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Arm: {arm}")
    print(f"Task Family: {task}")
    print(f"Reasoning Mode: {reasoning_mode}")
    print(f"Max Iterations: {max_iterations}")
    print(f"GPU Budget: {budget_hours} Hours")
    print(f"Dry Run Mode: {dry_run}")
    print(f"Allow Simulation Fallback: {allow_simulation}\n")

    # Initialize LLM reasoning client. Raises SimulationDisallowedError if no
    # real provider is available and allow_simulation was not explicitly set.
    try:
        agent = LLMResearchAgent(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key,
            allow_simulation=allow_simulation,
            reasoning_mode=reasoning_mode,
        )
    except SimulationDisallowedError as e:
        print(f"\n[FATAL] {e}\n")
        raise

    logger = TraceLogger(
        experiment_id=experiment_id,
        agent_arm=arm,
        total_gpu_budget_hours=budget_hours
    )

    harness_path = os.path.join(os.path.dirname(__file__), "eval_harness.py")
    harness_base_args = ["--task", task] + (["--dry-run"] if dry_run else ["--no-dry-run", "--steps", "50"])

    # Run baseline evaluation (Iteration 0)
    print("\n[Iteration 0] Running Baseline Model Evaluation...")
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
    logger.add_reflection(0, run_0, hypo_0, "Baseline established as reference anchor.", "ACCEPT_AND_EXTEND", False, base_run["elapsed_seconds"])

    # Iteration history buffer fed to the LLM agent
    history = [{
        "iteration_index": 0,
        "hypothesis": "Baseline model",
        "target_component": "baseline",
        "hyperparameters": base_run["metrics"].get("hyperparameters", {}),
        "status": "SUCCESS",
        "val_loss": base_loss,
        "relative_gain_pct": 0.0,
        "traceback": None,
        "provider": "ground_truth_baseline",
    }]

    consecutive_divergences = 0
    total_gpu_sec = base_run["elapsed_seconds"]
    active_parent_hypo = hypo_0
    simulation_used_count = 0

    # Main Agentic Optimization Loop
    for it in range(1, max_iterations + 1):
        # 1. Evaluate stopping conditions
        if (total_gpu_sec / 3600.0) >= budget_hours:
            print(f"[STOPPING TRIGGER] GPU Budget Exceeded ({total_gpu_sec/3600.0:.2f} >= {budget_hours}h).")
            break
        if consecutive_divergences >= 3:
            print("[STOPPING TRIGGER] 3-Strike Divergence Rule triggered. Terminating run.")
            break

        step_start = time.time()
        print(f"--- [Iteration {it}/{max_iterations}] ---")

        # 2. LLM Formulates Next Hypothesis & Modifications
        print("Prompting LLM Agent to analyze history and formulate next hypothesis...")
        llm_response = agent.generate_next_hypothesis(
            iteration_idx=it,
            history=history,
            incumbent_best_loss=logger.incumbent_best_loss,
            base_loss=base_loss
        )

        response_provider = llm_response.get("provider", "unknown")
        if response_provider == "adaptive_simulation":
            simulation_used_count += 1

        hypo_text = llm_response.get("hypothesis_text", f"Iteration {it} modification")
        target_comp = llm_response.get("target_component", "optimizer")
        predicted_delta = float(llm_response.get("predicted_delta_loss", 0.0))
        mods = llm_response.get("modifications", {})

        print(f"[Provider: {response_provider}] Agent Hypothesis: {hypo_text}")
        print(f"Target Component: {target_comp} | Predicted Delta Loss: {predicted_delta:+.4f}")
        print(f"Proposed Hyperparameters: {mods}")

        hypo_id = logger.add_hypothesis(
            iteration_idx=it,
            text=hypo_text,
            target_component=target_comp,
            predicted_delta_loss=predicted_delta,
            parent_hypo_id=active_parent_hypo,
            provider=response_provider,
        )

        exp_id = logger.add_experiment_spec(it, hypo_id, str(mods))

        # 3. Translate LLM modifications to sandbox execution arguments
        run_args = list(harness_base_args)
        if "lr" in mods:
            run_args.extend(["--lr", str(mods["lr"])])
        if "weight_decay" in mods:
            run_args.extend(["--weight-decay", str(mods["weight_decay"])])
        if "n_layers" in mods:
            run_args.extend(["--n-layers", str(mods["n_layers"])])
        if "n_heads" in mods:
            run_args.extend(["--n-heads", str(mods["n_heads"])])
        if "d_model" in mods:
            run_args.extend(["--d-model", str(mods["d_model"])])
        if "activation" in mods:
            run_args.extend(["--activation", str(mods["activation"])])
        if "norm_type" in mods:
            run_args.extend(["--norm-type", str(mods["norm_type"])])
        if mods.get("scale_factor"):
            run_args.extend(["--scale-factor", str(mods["scale_factor"])])
        if "pos_encoding" in mods and mods["pos_encoding"]:
            run_args.extend(["--pos-encoding", str(mods["pos_encoding"])])
        if "ffn_type" in mods and mods["ffn_type"]:
            run_args.extend(["--ffn-type", str(mods["ffn_type"])])
        if "topology" in mods and mods["topology"]:
            run_args.extend(["--topology", str(mods["topology"])])
        if "n_kv_heads" in mods and mods["n_kv_heads"]:
            run_args.extend(["--n-kv-heads", str(mods["n_kv_heads"])])

        # 4. Execute candidate in sandbox
        res = execute_in_sandbox(harness_path, run_args, timeout_sec=60 if dry_run else 300)
        total_gpu_sec += res["elapsed_seconds"]

        # 5. Parse evaluation results and check divergence
        if res["status"] == "SUCCESS" and res["metrics"]:
            val_loss = res["metrics"]["final_val_loss"]
            train_loss = res["metrics"]["final_train_loss"]
            rel_gain = round(((base_loss - val_loss) / base_loss) * 100, 2)
            print(f"Result: Validation Loss = {val_loss:.4f} (Train Loss = {train_loss:.4f}, Rel Gain = {rel_gain:+0.2f}%)")

            if val_loss > 3.0 * base_loss:
                consecutive_divergences += 1
                status = "DIVERGED"
                decision = "REJECT_AND_BACKTRACK"
                self_fail = True
                print(f"[WARNING] Divergence! Val Loss {val_loss:.4f} > 3x Base. Backtracking.")
            else:
                consecutive_divergences = 0
                status = "SUCCESS"
                if val_loss < logger.incumbent_best_loss:
                    decision = "ACCEPT_AND_EXTEND"
                    self_fail = False
                    active_parent_hypo = hypo_id
                    print(f"[IMPROVEMENT] Discovered new incumbent best loss: {val_loss:.4f}!")
                else:
                    decision = "REJECT_AND_BACKTRACK"
                    self_fail = False
                    print(f"[REJECT] Val Loss {val_loss:.4f} did not beat incumbent {logger.incumbent_best_loss:.4f}. Backtracking.")
        else:
            val_loss = None
            train_loss = None
            rel_gain = None
            status = "RUNTIME_CRASH"
            decision = "REJECT_AND_BACKTRACK"
            self_fail = True
            consecutive_divergences += 1
            print(f"[CRASH] Sandbox execution failed! Stderr: {res['stderr_tail']}")

        # Record to trace logger
        run_id, is_best = logger.add_execution_run(
            iteration_idx=it,
            exp_id=exp_id,
            status=status,
            exit_code=res["exit_code"],
            gpu_seconds=res["elapsed_seconds"],
            val_loss=val_loss,
            train_loss=train_loss
        )

        cycle_duration = time.time() - step_start
        reflection_text = f"Iteration {it} outcome: {status} with decision {decision}. Agent prediction was {predicted_delta:+.4f}. Provider: {response_provider}."
        logger.add_reflection(
            iteration_idx=it,
            run_id=run_id,
            hypo_id=hypo_id,
            reflection_text=reflection_text,
            decision=decision,
            self_detected_failure=self_fail,
            cycle_sec=cycle_duration
        )

        # Append to LLM history buffer for next round reflection
        history.append({
            "iteration_index": it,
            "hypothesis": hypo_text,
            "target_component": target_comp,
            "hyperparameters": mods,
            "status": status,
            "val_loss": val_loss,
            "relative_gain_pct": rel_gain,
            "traceback": res["stderr_tail"] if status != "SUCCESS" else None,
            "provider": response_provider,
        })

        # Progressively save trace conforming to trace_schema.json
        trace_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "execution_traces",
            f"trace_{experiment_id}.json"
        )
        logger.export_trace(trace_path)

    summary = logger.export_trace(trace_path)
    print("\n=== EXPERIMENT RUN COMPLETE ===")
    print(f"Total Iterations: {summary['summary_statistics']['total_iterations']}")
    print(f"Dead-End Ratio: {summary['summary_statistics']['dead_end_ratio']}")
    print(f"Path to First Improvement: {summary['summary_statistics']['path_to_first_improvement_steps']}")
    print(f"Backtracking Count: {summary['summary_statistics']['backtracking_count']}")
    print(f"Final Best Loss: {summary['summary_statistics']['final_best_loss']}")
    print(f"Final Relative Gain: {summary['summary_statistics']['final_relative_improvement_pct']}%")
    print(f"Simulated (non-real) hypotheses used: {simulation_used_count} / {max_iterations}")
    if simulation_used_count > 0:
        print("*** WARNING: This trace contains simulated hypotheses. Do NOT report it as real agent output. ***")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Research Agent Loop")
    parser.add_argument("--arm", type=str, default="Arm1_AutonomousAgent")
    parser.add_argument("--iterations", type=int, default=3)

    # Corrected dry-run flags with explicit boolean destination
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Enable fast dry run mode (training only; does not affect LLM calls)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Run full training on GPU/CPU")
    parser.set_defaults(dry_run=False)

    parser.add_argument("--provider", type=str, default=None, help="LLM Provider (anthropic, openai, nvidia)")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. claude-3-5-sonnet-20241022, gpt-4o, nvidia/nemotron-3-ultra-550b-a55b)")
    parser.add_argument("--api-key", type=str, default=None, help="API Key for the provider")
    parser.add_argument("--allow-simulation", dest="allow_simulation", action="store_true",
                         help="Explicitly permit falling back to the scripted simulation when no API key/provider "
                              "is available. Only use this for plumbing/CI smoke tests. Output from these runs "
                              "must never be reported as real agent results.")
    parser.add_argument("--task", type=str, default="dyck", choices=["dyck", "fsm", "parity_legacy"])
    parser.add_argument("--reasoning-mode", type=str, default="full",
                        choices=["full", "no_history", "history_no_reflection", "critic_refine"],
                        help="Reasoning ablation condition (full, no_history, history_no_reflection, critic_refine)")
    args = parser.parse_args()

    run_agent_loop(
        arm=args.arm,
        dry_run=args.dry_run,
        max_iterations=args.iterations,
        llm_provider=args.provider,
        llm_model=args.model,
        api_key=args.api_key,
        allow_simulation=args.allow_simulation,
        task=args.task,
        reasoning_mode=args.reasoning_mode,
    )
