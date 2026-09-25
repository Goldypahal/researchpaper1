"""
Pre-Flight Provider Benchmark & Comparative Evaluation Suite.
Evaluates LLM reasoning providers under identical agent prompt conditions before
launching large multi-seed experimental sweeps on Kaggle GPU.

Evaluated Providers:
  1. Groq    (Default: llama-3.3-70b-versatile)
  2. Mistral (Default: mistral-small-latest)
  3. Gemini  (Default: gemini-3.6-flash)
  [Optional: OpenRouter with pinned underlying model/provider]

10 Core Scientific Dimensions Measured:
  1. Successful Calls / Total
  2. Failure Rate (%)
  3. Latency (s) (min, mean, max)
  4. Output Tokens (mean, total)
  5. Total Tokens (mean, total)
  6. JSON Validity (%)
  7. Proposal Validity (%) (search space bounds, head-dim divisibility, valid types)
  8. Rate-Limit Behavior (HTTP 429 encounters, retries, backoffs)
  9. Reasoning Quality (inductive bias hypothesis, target component, delta loss)
  10. Modification Usefulness (non-trivial architecture mutations, bounds safety)

Usage:
  python scripts/benchmark_providers.py --calls 3
  python scripts/benchmark_providers.py --providers groq mistral gemini
  python scripts/benchmark_providers.py --groq-model llama-3.3-70b-versatile --mistral-model mistral-small-latest
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)

from agent_scaffold.provider_adapters import create_provider_adapter, SimulationDisallowedError
from search_space import validate_proposal, SEARCH_SPACE_SPEC

# Standardized realistic multi-trial agent prompt for sequence modeling
STANDARDIZED_HISTORY = [
    {
        "iteration_index": 0,
        "hypothesis": "Baseline SmallTransformerLM (6 layers, 256 dim, 4 heads, GELU, LayerNorm).",
        "target_component": "attention",
        "hyperparameters": {
            "lr": 0.001, "weight_decay": 0.01, "n_layers": 6, "n_heads": 4,
            "d_model": 256, "activation": "gelu", "norm_type": "layernorm",
            "pos_encoding": "learned", "ffn_type": "standard", "topology": "pre_ln"
        },
        "status": "SUCCESS",
        "val_loss": 4.1800,
        "relative_gain_pct": 0.0,
        "traceback": None,
        "provider": "baseline"
    },
    {
        "iteration_index": 1,
        "hypothesis": "Test wider model dimension (d_model=512) with standard attention to increase memory capacity.",
        "target_component": "attention",
        "hyperparameters": {
            "lr": 0.001, "weight_decay": 0.01, "n_layers": 6, "n_heads": 4,
            "d_model": 512, "activation": "gelu", "norm_type": "layernorm",
            "pos_encoding": "learned", "ffn_type": "standard", "topology": "pre_ln"
        },
        "status": "SUCCESS",
        "val_loss": 4.1420,
        "relative_gain_pct": 0.91,
        "traceback": None,
        "provider": "arm1"
    }
]

STANDARDIZED_USER_PROMPT = f"""Task: DYCK sequence modeling
Current Iteration: 2
Baseline Loss (Iteration 0): 4.1800
Current Incumbent Best Validation Loss: 4.1420
Reasoning Mode: full

Execution History so far:
{json.dumps(STANDARDIZED_HISTORY, indent=2)}

Formulate Iteration 2's hypothesis and concrete parameter modifications. Output ONLY valid JSON matching the specified schema.
"""


def evaluate_candidate(
    provider_name: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    num_calls: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Runs standardized evaluation across N calls for a single (provider, model) candidate.
    """
    if verbose:
        print(f"\n" + "-" * 75)
        print(f"BENCHMARKING: Provider='{provider_name.upper()}' | Model='{model_name or 'default'}'")
        print(f"Target Calls: {num_calls} | Standardized Dyck-4 Sequence Modeling Prompt")
        print("-" * 75)

    from agent_scaffold.llm_agent import SYSTEM_PROMPT

    try:
        adapter = create_provider_adapter(
            provider=provider_name,
            model=model_name,
            api_key=api_key,
            strict=True,
            allow_simulation=False
        )
    except SimulationDisallowedError:
        print(f"  [SKIPPED] Missing credentials for provider '{provider_name}'.")
        return {
            "provider": provider_name,
            "model": model_name or "unconfigured",
            "status": "MISSING_CREDENTIALS",
            "successful_calls": 0,
            "total_calls": num_calls,
            "failure_rate_pct": 100.0,
            "latencies": [],
            "mean_latency": None,
            "output_tokens": 0,
            "total_tokens": 0,
            "json_valid_count": 0,
            "json_validity_pct": 0.0,
            "proposal_valid_count": 0,
            "proposal_validity_pct": 0.0,
            "rate_limit_hits": 0,
            "reasoning_scores": [],
            "mean_reasoning_score": 0.0,
            "usefulness_scores": [],
            "mean_usefulness_score": 0.0,
            "proposals": [],
            "errors": ["API key or environment variable not found."]
        }
    except Exception as e:
        print(f"  [ERROR] Adapter initialization failed: {e}")
        return {
            "provider": provider_name,
            "model": model_name or "error",
            "status": "INIT_FAILED",
            "successful_calls": 0,
            "total_calls": num_calls,
            "failure_rate_pct": 100.0,
            "latencies": [],
            "mean_latency": None,
            "output_tokens": 0,
            "total_tokens": 0,
            "json_valid_count": 0,
            "json_validity_pct": 0.0,
            "proposal_valid_count": 0,
            "proposal_validity_pct": 0.0,
            "rate_limit_hits": 0,
            "reasoning_scores": [],
            "mean_reasoning_score": 0.0,
            "usefulness_scores": [],
            "mean_usefulness_score": 0.0,
            "proposals": [],
            "errors": [str(e)]
        }

    actual_model = adapter.model
    if verbose:
        print(f"  Adapter initialized: {adapter.name} (model: {actual_model})")
        print(f"  Verifying 1-shot connectivity...")

    conn_ok, conn_resp = adapter.verify_connection()
    if not conn_ok:
        print(f"  [FAILED] Connection verification failed: {conn_resp}")
        return {
            "provider": provider_name,
            "model": actual_model,
            "status": "CONNECTION_FAILED",
            "successful_calls": 0,
            "total_calls": num_calls,
            "failure_rate_pct": 100.0,
            "latencies": [],
            "mean_latency": None,
            "output_tokens": 0,
            "total_tokens": 0,
            "json_valid_count": 0,
            "json_validity_pct": 0.0,
            "proposal_valid_count": 0,
            "proposal_validity_pct": 0.0,
            "rate_limit_hits": 0,
            "reasoning_scores": [],
            "mean_reasoning_score": 0.0,
            "usefulness_scores": [],
            "mean_usefulness_score": 0.0,
            "proposals": [],
            "errors": [f"Connection verification failed: {conn_resp}"]
        }

    if verbose:
        print(f"  [OK] Connection verified. Commencing {num_calls} benchmark trials...\n")

    latencies = []
    output_tokens_list = []
    total_tokens_list = []
    json_valid_count = 0
    proposal_valid_count = 0
    rate_limit_hits = 0
    reasoning_scores = []
    usefulness_scores = []
    proposals = []
    errors = []

    for call_idx in range(1, num_calls + 1):
        t0 = time.time()
        try:
            parsed, raw_text, usage, latency = adapter.complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=STANDARDIZED_USER_PROMPT,
                temperature=0.7,
                max_retries=3
            )
            latencies.append(latency)

            # Token tracking
            out_tok = usage.get("completion_tokens") or usage.get("output_tokens") or len(raw_text.split()) * 1.3
            tot_tok = usage.get("total_tokens") or (out_tok + len(STANDARDIZED_USER_PROMPT.split()) * 1.3)
            output_tokens_list.append(int(out_tok))
            total_tokens_list.append(int(tot_tok))

            # JSON Validity
            is_json_valid = isinstance(parsed, dict) and len(parsed) > 0
            if is_json_valid:
                json_valid_count += 1

            # Proposal Validity (Search Space & Schema)
            is_prop_valid, violations = validate_proposal(parsed)
            if is_prop_valid:
                proposal_valid_count += 1
            else:
                if verbose:
                    print(f"    [Call {call_idx}] Proposal Violations: {violations}")

            # Reasoning Quality Score (0 to 10 scale)
            # Checks for inductive bias explanation, target component clarity, and predicted delta
            hypo_text = parsed.get("hypothesis") or parsed.get("hypothesis_text") or ""
            r_score = 0.0
            if len(hypo_text.strip()) > 20:
                r_score += 4.0
            if any(term in hypo_text.lower() for term in ["attention", "stack", "bracket", "rope", "rotary", "norm", "swiglu", "capacity", "depth", "closure"]):
                r_score += 3.0
            if parsed.get("target_component") in ["attention", "ffn", "topology", "positional_encoding", "normalization", "optimizer"]:
                r_score += 1.5
            pred_delta = parsed.get("predicted_delta_loss")
            if isinstance(pred_delta, (int, float)) and 0.001 <= pred_delta <= 0.5:
                r_score += 1.5
            reasoning_scores.append(round(r_score, 1))

            # Modification Usefulness Score (0 to 10 scale)
            # Checks that non-trivial modifications were made and invariants preserved
            mods = parsed.get("modifications", {})
            u_score = 0.0
            if isinstance(mods, dict) and len(mods) >= 3:
                u_score += 4.0
            d_m = mods.get("d_model", 256)
            n_h = mods.get("n_heads", 4)
            if isinstance(d_m, int) and isinstance(n_h, int) and n_h > 0 and d_m % n_h == 0:
                u_score += 3.0
            if "lr" in mods and 1e-5 <= mods["lr"] <= 5e-2:
                u_score += 3.0
            usefulness_scores.append(round(u_score, 1))

            proposals.append({
                "call": call_idx,
                "hypothesis": hypo_text,
                "target_component": parsed.get("target_component"),
                "predicted_delta_loss": parsed.get("predicted_delta_loss"),
                "modifications": mods,
                "latency_sec": latency,
                "violations": violations
            })

            if verbose:
                print(f"    [Call {call_idx}/{num_calls}] Latency: {latency:.2f}s | JSON: {'OK' if is_json_valid else 'FAIL'} | Proposal: {'VALID' if is_prop_valid else 'INVALID'} | Reasoning: {r_score:.1f}/10 | Usefulness: {u_score:.1f}/10")

        except Exception as e:
            err_str = str(e)
            errors.append(f"Call {call_idx}: {err_str}")
            if "429" in err_str or "rate limit" in err_str.lower() or "quota" in err_str.lower():
                rate_limit_hits += 1
            if verbose:
                print(f"    [Call {call_idx}/{num_calls}] FAILED: {err_str}")

        # Short pause between calls to respect burst rate limits
        time.sleep(1.0)

    success_calls = len(latencies)
    fail_rate = round(((num_calls - success_calls) / num_calls) * 100.0, 1)
    mean_lat = round(sum(latencies) / len(latencies), 3) if latencies else None
    mean_reasoning = round(sum(reasoning_scores) / len(reasoning_scores), 2) if reasoning_scores else 0.0
    mean_usefulness = round(sum(usefulness_scores) / len(usefulness_scores), 2) if usefulness_scores else 0.0

    return {
        "provider": provider_name,
        "model": actual_model,
        "status": "COMPLETED" if success_calls == num_calls else ("PARTIAL" if success_calls > 0 else "FAILED"),
        "successful_calls": success_calls,
        "total_calls": num_calls,
        "failure_rate_pct": fail_rate,
        "latencies": latencies,
        "mean_latency": mean_lat,
        "min_latency": min(latencies) if latencies else None,
        "max_latency": max(latencies) if latencies else None,
        "output_tokens_mean": round(sum(output_tokens_list) / len(output_tokens_list), 1) if output_tokens_list else 0,
        "total_tokens_mean": round(sum(total_tokens_list) / len(total_tokens_list), 1) if total_tokens_list else 0,
        "json_valid_count": json_valid_count,
        "json_validity_pct": round((json_valid_count / max(1, success_calls)) * 100.0, 1),
        "proposal_valid_count": proposal_valid_count,
        "proposal_validity_pct": round((proposal_valid_count / max(1, success_calls)) * 100.0, 1),
        "rate_limit_hits": rate_limit_hits,
        "mean_reasoning_score": mean_reasoning,
        "mean_usefulness_score": mean_usefulness,
        "proposals": proposals,
        "errors": errors
    }


def run_provider_benchmark_suite(
    providers: Optional[List[str]] = None,
    models: Optional[Dict[str, str]] = None,
    calls_per_provider: int = 3,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the full comparative benchmark across recommended providers in sequence:
      1. Groq
      2. Mistral
      3. Gemini
      [Optional: OpenRouter]
    """
    # Recommended default test order for LLM Qualification:
    # 1. NVIDIA Nemotron (Super / Ultra / 70B)
    # 2. Groq GPT-OSS (120B / 20B) or Llama
    # 3. Mistral Candidate (mistral-small / codestral)
    # [Gemini kept as secondary; quota is currently 20 RPD]
    default_order = ["nvidia", "groq", "mistral"]
    target_providers = providers or default_order
    model_map = {
        "nvidia": "nvidia/llama-3.1-nemotron-70b-instruct",
        "groq": "openai/gpt-oss-120b",
        "mistral": "mistral-small-latest",
        "gemini": "gemini-3.6-flash",
        "openrouter": "nvidia/nemotron-3-ultra-550b-a55b"
    }
    if models:
        model_map.update(models)

    print("=" * 80)
    print("LLM QUALIFICATION & PRE-FLIGHT BENCHMARK SUITE")
    print("=" * 80)
    print("Candidate Pool Architecture:")
    print("                 LLM Qualification")
    print("                         |")
    print("         +---------------+----------------+")
    print("         v               v                v")
    print("      NVIDIA            Groq            Mistral")
    print("      Nemotron          GPT-OSS         candidate")
    print("         |                |")
    print("      Super/Ultra       20B/120B")
    print(f"\nEvaluating Providers in order: {target_providers}")
    print(f"Calls per candidate: {calls_per_provider}")
    print(f"Standardized Task: Synthetic Dyck-4 Sequence Modeling")
    print("Measuring 10 core dimensions: success rate, latency, tokens, JSON & proposal validity,")
    print("rate-limits, reasoning quality, and modification usefulness.\n")

    results = []
    for prov in target_providers:
        m = model_map.get(prov)
        res = evaluate_candidate(
            provider_name=prov,
            model_name=m,
            num_calls=calls_per_provider,
            verbose=True
        )
        results.append(res)

    # Compile Summary Table
    print("\n" + "=" * 80)
    print("PROVIDER COMPARISON SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Provider':<10} | {'Model':<24} | {'Success':<8} | {'Lat(s)':<7} | {'JSON%':<6} | {'Prop%':<6} | {'Reason':<7} | {'Usefulness':<10} | {'Status'}"
    print(header)
    print("-" * len(header))

    for r in results:
        prov = r["provider"]
        mod = (r["model"][:22] + "..") if len(r["model"]) > 24 else r["model"]
        succ = f"{r['successful_calls']}/{r['total_calls']}"
        lat = f"{r['mean_latency']:.2f}" if r["mean_latency"] is not None else "N/A"
        json_v = f"{r['json_validity_pct']:.0f}%" if r["successful_calls"] > 0 else "N/A"
        prop_v = f"{r['proposal_validity_pct']:.0f}%" if r["successful_calls"] > 0 else "N/A"
        rsn = f"{r['mean_reasoning_score']:.1f}/10" if r["successful_calls"] > 0 else "N/A"
        use = f"{r['mean_usefulness_score']:.1f}/10" if r["successful_calls"] > 0 else "N/A"
        stat = r["status"]
        print(f"{prov:<10} | {mod:<24} | {succ:<8} | {lat:<7} | {json_v:<6} | {prop_v:<6} | {rsn:<7} | {use:<10} | {stat}")

    print("=" * 80)

    # Scientific Guidance and Recommendation
    print("\n[SCIENTIFIC PROTOCOL RECOMMENDATIONS]:")
    print("1. FROZEN EXPERIMENTAL VARIABLE:")
    print("   Choose a SINGLE provider + model for your main experimental arm across all seeds:")
    print("   seeds = [42, 101, 202, 303, 404] must ALL use the exact same provider/model.")
    print("   Never do: seed 42 -> Gemini, seed 101 -> Groq (violates experimental controls).")
    print("2. MODEL DEPENDENCE AS SEPARATE ARMS:")
    print("   To study LLM provider differences, define distinct conditions: LLM-Groq, LLM-Mistral, LLM-Gemini.")
    print("3. QUOTA REPORTING FOR PAPERS:")
    print("   Do NOT cite generic 'free tier = 14.4k/day' or '1B tokens/month' as universal constants.")
    print("   Record the exact observed quota for the specific account, model, and date stamp.")
    print("4. OPENROUTER GATEWAY:")
    print("   Treat OpenRouter as a gateway, not a model. Pin the exact underlying model for reproducibility.")
    print("=" * 80 + "\n")

    # Save artifact
    out_dir = output_dir or os.path.join(WORKSPACE_ROOT, "experiment_results", "provider_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "provider_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_providers": target_providers,
            "calls_per_provider": calls_per_provider,
            "results": results
        }, f, indent=2)
    print(f"[Artifact Saved] JSON Results: {json_path}")

    # Generate Markdown Report
    md_path = os.path.join(out_dir, "provider_benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Provider Benchmark & Comparative Evaluation Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("This benchmark measures the live execution characteristics of candidate reasoning providers under standardized prompt conditions for autonomous sequence modeling.\n\n")
        f.write("| Provider | Model | Successful Calls | Mean Latency (s) | JSON Validity | Proposal Validity | Reasoning (0-10) | Usefulness (0-10) | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            lat = f"{r['mean_latency']:.2f}s" if r['mean_latency'] is not None else "N/A"
            json_v = f"{r['json_validity_pct']:.0f}%" if r["successful_calls"] > 0 else "N/A"
            prop_v = f"{r['proposal_validity_pct']:.0f}%" if r["successful_calls"] > 0 else "N/A"
            rsn = f"{r['mean_reasoning_score']:.1f}/10" if r["successful_calls"] > 0 else "N/A"
            use = f"{r['mean_usefulness_score']:.1f}/10" if r["successful_calls"] > 0 else "N/A"
            f.write(f"| **{r['provider'].capitalize()}** | `{r['model']}` | {r['successful_calls']}/{r['total_calls']} | {lat} | {json_v} | {prop_v} | {rsn} | {use} | {r['status']} |\n")

        f.write("\n## 2. Experimental Rigor & Variable Freezing Protocol\n\n")
        f.write("- **Frozen Experimental Variable:** The primary LLM search arm must lock `provider` and `model` across all seeds (`[42, 101, 202, 303, 404]`).\n")
        f.write("- **Prohibition of Seed-Provider Mixing:** Alternating providers across seeds within the same arm (e.g., seed 42 on Gemini, seed 101 on Groq) introduces severe confounding variance.\n")
        f.write("- **Model Dependence Study:** If investigating provider differences, instantiate distinct arms (`LLM-Groq`, `LLM-Mistral`, `LLM-Gemini`) with identical seed allocations.\n")
        f.write("- **Accurate Quota Documentation:** Cite exact observed account/model quotas on specific test dates rather than generic marketing figures.\n")
        f.write("- **Gateway Reproducibility:** When using OpenRouter, explicitly pin the underlying model identifier.\n")

    print(f"[Artifact Saved] Markdown Report: {md_path}")
    return {"results": results, "json_path": json_path, "md_path": md_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Qualification & Pre-flight Provider Benchmark Suite.")
    parser.add_argument("--calls", type=int, default=3, help="Calls per candidate (default: 3)")
    parser.add_argument("--providers", nargs="+", default=["nvidia", "groq", "mistral"], help="List of providers to test (e.g. nvidia groq mistral)")
    parser.add_argument("--nvidia-model", type=str, default="nvidia/llama-3.1-nemotron-70b-instruct", help="NVIDIA Nemotron model (e.g. nvidia/llama-3.1-nemotron-70b-instruct, nvidia/nemotron-4-340b-instruct, nvidia/nemotron-3-ultra-550b-a55b)")
    parser.add_argument("--groq-model", type=str, default="openai/gpt-oss-120b", help="Groq model (e.g. openai/gpt-oss-120b, openai/gpt-oss-20b, llama-3.3-70b-versatile)")
    parser.add_argument("--mistral-model", type=str, default="mistral-small-latest", help="Mistral model (e.g. mistral-small-latest, codestral-latest)")
    parser.add_argument("--gemini-model", type=str, default="gemini-3.6-flash", help="Gemini model")
    parser.add_argument("--openrouter-model", type=str, default="nvidia/nemotron-3-ultra-550b-a55b", help="OpenRouter pinned model")
    args = parser.parse_args()

    models_config = {
        "nvidia": args.nvidia_model,
        "groq": args.groq_model,
        "mistral": args.mistral_model,
        "gemini": args.gemini_model,
        "openrouter": args.openrouter_model
    }

    run_provider_benchmark_suite(
        providers=args.providers,
        models=models_config,
        calls_per_provider=args.calls
    )
