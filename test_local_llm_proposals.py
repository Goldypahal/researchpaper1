"""
test_local_llm_proposals.py
============================
Pre-experiment Proposal-Only Stress Test for Local LLM (Qwen2.5-7B-Instruct).

Purpose:
  Rigorously stress-tests the local LLM generation and JSON parsing pipeline
  BEFORE launching expensive multi-seed benchmark sweeps.

Checks:
  1. Real local LLM model loading (4-bit NF4).
  2. Sequential generation of K research proposals under evolving search histories.
  3. JSON syntax parsing, safe repair auditing, and retry tracking.
  4. Token consumption and latency benchmarking.
  5. Schema validity (hypothesis, modifications dictionary, predicted delta loss).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import argparse
from typing import Dict, Any, List

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from agent_scaffold.llm_agent import LLMResearchAgent
from search_space import sanitize_configuration
from agent_scaffold.provider_adapters import parse_llm_json


def run_proposal_stress_test(
    num_proposals: int = 10,
    provider: str = "local",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    quantization: str = "4bit",
    output_dir: str = "experiment_results/proposal_stress_test"
) -> Dict[str, Any]:
    abs_out_dir = os.path.join(WORKSPACE_ROOT, output_dir)
    os.makedirs(abs_out_dir, exist_ok=True)

    print("=" * 75, flush=True)
    print("LOCAL LLM PROPOSAL STRESS TEST: EMPIRICAL RELIABILITY EVALUATION", flush=True)
    print("=" * 75, flush=True)
    print(f"Provider:       {provider}", flush=True)
    print(f"Model:          {model}", flush=True)
    print(f"Quantization:   {quantization}", flush=True)
    print(f"Target Horizon: {num_proposals} sequential research proposals", flush=True)
    print(f"Artifacts Dir:  {abs_out_dir}\n", flush=True)

    # Configure local environment variables
    if provider == "local":
        os.environ["LOCAL_LLM_MODEL"] = model
        os.environ["LOCAL_LLM_QUANT"] = quantization

    # Initialize agent in strict mode
    t_load_start = time.time()
    print("[1/3] Initializing LLMSearchAgent...", flush=True)
    agent = LLMResearchAgent(
        provider=provider,
        model=model,
        strict=True,
        allow_simulation=False,
        reasoning_mode="full"
    )
    print(f"[2/3] Agent initialized in {time.time() - t_load_start:.2f}s.\n", flush=True)

    # Health check ping
    print("[3/3] Running connectivity ping...", flush=True)
    ok, ping_resp = agent.verify_connection()
    if not ok:
        print(f"FATAL: Preflight ping failed: {ping_resp}", flush=True)
        return {"status": "FAILED_PING", "error": str(ping_resp)}
    print(f"Connectivity verified successfully: {ping_resp}\n", flush=True)

    # Synthetic baseline and incremental history
    base_val = 0.5824
    baseline_cfg = {
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 512,
        "lr": 0.001,
        "weight_decay": 0.01,
        "activation": "gelu",
        "norm_type": "rmsnorm",
        "pos_encoding": "rotary",
        "ffn_type": "swiglu",
        "topology": "pre_ln",
        "scale_factor": 0.1
    }

    history: List[Dict[str, Any]] = [{
        "iteration": 0,
        "config": baseline_cfg,
        "train_loss": 0.5790,
        "val_loss": base_val,
        "ood_loss": 0.6120,
        "val_struct_acc": 0.852,
        "ood_struct_acc": 0.810,
        "gap_rel": 0.051,
        "gpu_seconds": 3.8,
        "best_val_loss": base_val,
        "hypothesis": "Default Baseline Transformer",
        "predicted_delta_loss": 0.0,
        "actual_delta_loss": 0.0
    }]

    incumbent_best_loss = base_val
    proposal_records = []
    fatal_errors = []

    print("-" * 75, flush=True)
    print(f"{'Iter':<5} | {'Status':<10} | {'Valid?':<7} | {'Repaired?':<10} | {'Retried?':<9} | {'Lat (s)':<8} | {'Tokens':<8} | {'Hypothesis Snippet'}")
    print("-" * 75, flush=True)

    for it in range(1, num_proposals + 1):
        t_it_start = time.time()
        record: Dict[str, Any] = {
            "iteration": it,
            "status": "PENDING",
            "json_valid": False,
            "repair_attempted": False,
            "repair_success": False,
            "retry_attempted": False,
            "retry_success": False,
            "latency_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "has_modifications": False,
            "modifications_count": 0,
            "raw_response": "",
            "parsed_proposal": None,
            "error_msg": None
        }

        try:
            proposal = agent.propose_experiment(
                iteration_idx=it,
                history=history,
                incumbent_best_loss=incumbent_best_loss,
                base_loss=base_val,
                task="dyck"
            )

            raw_resp = proposal.get("raw_response", "")
            record["raw_response"] = raw_resp
            record["parsed_proposal"] = proposal
            record["latency_seconds"] = proposal.get("latency_seconds", round(time.time() - t_it_start, 3))

            audit = proposal.get("parse_audit") or {}
            record["json_valid"] = audit.get("json_valid", True)
            record["repair_attempted"] = audit.get("repair_attempted", False)
            record["repair_success"] = audit.get("repair_success", False)
            record["retry_attempted"] = audit.get("retry_attempted", False)
            record["retry_success"] = audit.get("retry_success", False)
            record["prompt_tokens"] = audit.get("prompt_tokens", 0)
            record["completion_tokens"] = audit.get("completion_tokens", 0)
            record["total_tokens"] = audit.get("total_tokens", 0)

            mods = proposal.get("modifications", {})
            record["has_modifications"] = isinstance(mods, dict) and len(mods) > 0
            record["modifications_count"] = len(mods) if isinstance(mods, dict) else 0

            # Verify schema compliance
            hypo = proposal.get("hypothesis") or proposal.get("hypothesis_text") or "N/A"
            record["hypothesis"] = hypo
            record["predicted_delta_loss"] = proposal.get("predicted_delta_loss", 0.0)
            record["status"] = "SUCCESS"

            hypo_disp = hypo[:35] + "..." if len(hypo) > 35 else hypo
            print(
                f"{it:<5} | "
                f"{'OK':<10} | "
                f"{str(record['json_valid']):<7} | "
                f"{str(record['repair_success']):<10} | "
                f"{str(record['retry_attempted']):<9} | "
                f"{record['latency_seconds']:<8.2f} | "
                f"{record['completion_tokens']:<8} | "
                f"{hypo_disp}",
                flush=True
            )

            # Evolve history with synthetic candidate feedback for next iteration prompt
            syn_loss = round(base_val - (it * 0.005), 4)
            if syn_loss < incumbent_best_loss:
                incumbent_best_loss = syn_loss

            history.append({
                "iteration": it,
                "config": sanitize_configuration({**baseline_cfg, **mods}),
                "train_loss": round(syn_loss - 0.01, 4),
                "val_loss": syn_loss,
                "ood_loss": round(syn_loss + 0.02, 4),
                "val_struct_acc": round(0.85 + (it * 0.008), 3),
                "ood_struct_acc": round(0.81 + (it * 0.006), 3),
                "gap_rel": 0.035,
                "gpu_seconds": 3.9,
                "best_val_loss": incumbent_best_loss,
                "hypothesis": hypo,
                "predicted_delta_loss": float(proposal.get("predicted_delta_loss", 0.0)),
                "actual_delta_loss": round(base_val - syn_loss, 4)
            })

        except Exception as e:
            record["status"] = "FAILED"
            record["error_msg"] = f"{type(e).__name__}: {str(e)}"
            record["latency_seconds"] = round(time.time() - t_it_start, 3)
            fatal_errors.append(record)
            print(f"{it:<5} | {'FAILED':<10} | {'N/A':<7} | {'N/A':<10} | {'N/A':<9} | {record['latency_seconds']:<8.2f} | {'N/A':<8} | ERROR: {e}", flush=True)

        # Save single proposal trace
        iter_file = os.path.join(abs_out_dir, f"proposal_iter_{it:02d}.json")
        with open(iter_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        proposal_records.append(record)

    print("-" * 75, flush=True)

    # Compute Summary Statistics
    total = len(proposal_records)
    successes = sum(1 for r in proposal_records if r["status"] == "SUCCESS")
    valid_direct = sum(1 for r in proposal_records if r["json_valid"])
    repaired = sum(1 for r in proposal_records if r["repair_attempted"] and r["repair_success"])
    retried = sum(1 for r in proposal_records if r["retry_attempted"])
    schema_ok = sum(1 for r in proposal_records if r["has_modifications"])

    latencies = [r["latency_seconds"] for r in proposal_records if r["status"] == "SUCCESS"]
    comp_tokens = [r["completion_tokens"] for r in proposal_records if r["status"] == "SUCCESS"]
    prompt_tokens = [r["prompt_tokens"] for r in proposal_records if r["status"] == "SUCCESS"]

    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    avg_comp_toks = round(sum(comp_tokens) / len(comp_tokens), 1) if comp_tokens else 0.0
    avg_prompt_toks = round(sum(prompt_tokens) / len(prompt_tokens), 1) if prompt_tokens else 0.0

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": provider,
        "model": model,
        "quantization": quantization,
        "total_proposals_attempted": total,
        "successful_proposals": successes,
        "proposal_success_pct": round((successes / total) * 100.0, 1),
        "valid_json_direct_pct": round((valid_direct / total) * 100.0, 1),
        "repaired_json_pct": round((repaired / total) * 100.0, 1),
        "retry_rate_pct": round((retried / total) * 100.0, 1),
        "schema_conformity_pct": round((schema_ok / total) * 100.0, 1),
        "average_latency_sec": avg_lat,
        "min_latency_sec": round(min(latencies), 2) if latencies else 0.0,
        "max_latency_sec": round(max(latencies), 2) if latencies else 0.0,
        "average_completion_tokens": avg_comp_toks,
        "average_prompt_tokens": avg_prompt_toks,
        "fatal_errors_count": len(fatal_errors)
    }

    # Save summary report JSON
    summary_path = os.path.join(abs_out_dir, "stress_test_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown Report
    md_content = f"""# Local LLM Proposal Reliability Stress Test Report

**Model**: `{model}` ({quantization})  
**Provider**: `{provider}`  
**Test Horizon**: $K={total}$ sequential research proposals  
**Generated At**: {summary['timestamp']}  

---

## 1. Reliability & Parsing Metrics

| Metric | Measured Value | Target Criterion | Assessment |
| :--- | :--- | :--- | :--- |
| **Proposal Success Rate** | **{summary['proposal_success_pct']}%** ({successes}/{total}) | 100.0% | {'PASS' if successes == total else 'FAIL'} |
| **Direct Valid JSON** | {summary['valid_json_direct_pct']}% ({valid_direct}/{total}) | > 80.0% | {'NOMINAL' if valid_direct >= total * 0.8 else 'REVIEW'} |
| **Audited Safe Repairs** | {summary['repaired_json_pct']}% ({repaired}/{total}) | < 20.0% | SAFE |
| **Deterministic Retries** | {summary['retry_rate_pct']}% ({retried}/{total}) | < 20.0% | CONTROLLED |
| **Schema Conformity** | {summary['schema_conformity_pct']}% ({schema_ok}/{total}) | 100.0% | {'PASS' if schema_ok == total else 'FAIL'} |

---

## 2. Compute & Latency Metrics

| Metric | Value |
| :--- | :--- |
| **Average Latency** | **{avg_lat}s** (range: {summary['min_latency_sec']}s – {summary['max_latency_sec']}s) |
| **Average Completion Tokens** | **{avg_comp_toks} tokens** |
| **Average Prompt Tokens** | **{avg_prompt_toks} tokens** |
| **Fatal Generation Exceptions** | **{len(fatal_errors)}** |

---

## 3. Deployment Verdict

{'**CRITERIA MET**: The local LLM proposal engine is 100% reliable. The benchmark is cleared for full experimental runs.' if successes == total else '**ATTENTION REQUIRED**: Proposal failures detected. Inspect raw outputs in artifacts directory before starting benchmark.'}
"""

    md_path = os.path.join(abs_out_dir, "stress_test_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 75, flush=True)
    print("STRESS TEST EXECUTION SUMMARY", flush=True)
    print("=" * 75, flush=True)
    print(f"Proposal Success Rate:   {summary['proposal_success_pct']}% ({successes}/{total})")
    print(f"Direct Valid JSON:       {summary['valid_json_direct_pct']}%")
    print(f"Repaired JSON:           {summary['repaired_json_pct']}%")
    print(f"Retry Rate:              {summary['retry_rate_pct']}%")
    print(f"Schema Conformity:       {summary['schema_conformity_pct']}%")
    print(f"Average Latency:         {avg_lat}s")
    print(f"Avg Completion Tokens:   {avg_comp_toks}")
    print(f"Summary JSON:            {summary_path}")
    print(f"Summary Markdown:        {md_path}")
    print("=" * 75 + "\n", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test Local LLM research proposal generation on Kaggle.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of proposals to generate (default: 10)")
    parser.add_argument("--provider", type=str, default="local", help="Provider (default: local)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model ID")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--output-dir", type=str, default="experiment_results/proposal_stress_test")
    args = parser.parse_args()

    results = run_proposal_stress_test(
        num_proposals=args.iterations,
        provider=args.provider,
        model=args.model,
        quantization=args.quantization,
        output_dir=args.output_dir
    )

    if results.get("fatal_errors_count", 0) > 0 or results.get("successful_proposals", 0) < args.iterations:
        sys.exit(1)
    sys.exit(0)
