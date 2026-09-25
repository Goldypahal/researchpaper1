"""
LLMSearchAgent Reasoning Engine.
Implements the Provider Adapter Architecture:

                 LLMSearchAgent
                      │
              Provider Adapter
          ┌───────────┼───────────┐
          ↓           ↓           ↓
        Groq       Mistral     Gemini
          │           │           │
       model A      model B     model C

Provider + Model is a FROZEN experimental variable across all seeds in an arm.
Mixing providers across seeds (e.g., seed 42 -> Gemini, seed 101 -> Groq) is strictly
forbidden as it introduces severe experimental contamination.
If studying model dependence, configure distinct experimental arms:
  LLM-Groq (e.g. llama-3.3-70b-versatile)
  LLM-Mistral (e.g. mistral-small-latest)
  LLM-Gemini (e.g. gemini-3.6-flash)
"""

import os
import json
import re
import time
import copy
import hashlib
from typing import Dict, Any, List, Optional, Tuple

try:
    from agent_scaffold.provider_adapters import (
        BaseProviderAdapter,
        create_provider_adapter,
        SimulationDisallowedError,
        extract_json_payload
    )
except ImportError:
    from provider_adapters import (
        BaseProviderAdapter,
        create_provider_adapter,
        SimulationDisallowedError,
        extract_json_payload
    )

SYSTEM_PROMPT = """You are an expert autonomous machine learning researcher specializing in neural architecture search, structural Transformer design, and sequence modeling optimization.
Your objective is to minimize the holdout validation cross-entropy loss and out-of-distribution generalization loss of a Small Transformer language model on an unseen synthetic Dyck-k or state-transition sequence modeling dataset.

You have access to a hierarchical design space covering both continuous optimization and discrete architectural topology:
1. Learning rate (lr): float in range [1e-5, 5e-2]
2. Weight decay (weight_decay): float in range [0.0, 0.2]
3. Number of layers (n_layers): int in {2, 4, 6, 8}
4. Number of attention heads (n_heads): int in {2, 4, 8}
5. Key/Value heads for Grouped-Query Attention (n_kv_heads): int in {1, 2, n_heads} (must divide n_heads)
6. Model dimension (d_model): int in {128, 256, 384} (must be divisible by n_heads)
7. Activation function (activation): "gelu", "relu", or "silu"
8. Normalization layer (norm_type): "layernorm" or "rmsnorm"
9. Positional encoding (pos_encoding): "learned", "sinusoidal", "rotary" (RoPE), or "none" (NoPE)
10. Feed-Forward architecture (ffn_type): "standard" (Linear-Act-Linear) or "swiglu" (Gated Linear Unit)
11. Residual Block Topology (topology): "pre_ln" (sequential Pre-LN), "post_ln" (sequential Post-LN), or "parallel" (PaLM/GPT-J parallel block)
12. Attention scale factor (scale_factor): float or null (default is 1/sqrt(d_k))

You will receive the full history of previous trials, including:
- Hypotheses tested
- Hyperparameters and architectural structures used
- Validation loss and out-of-distribution (OOD) accuracy achieved
- Execution errors or tracebacks if any occurred
- The current incumbent best loss

You must analyze the historical progression, diagnose whether previous attempts underfit, overfit, diverged, or hit capacity bottlenecks, and propose the NEXT logical architectural hypothesis and exact modifications.
You must also provide your OWN quantitative prediction for the expected loss reduction (predicted_delta_loss).

You must respond with ONLY a valid JSON object matching this schema:
{
  "hypothesis_text": "<Clear technical hypothesis explaining why this structural modification improves hierarchical sequence modeling>",
  "target_component": "<attention | ffn | topology | positional_encoding | normalization | optimizer>",
  "predicted_delta_loss": <float, your expected reduction in validation loss, e.g. 0.04>,
  "modifications": {
    "lr": <float>,
    "weight_decay": <float>,
    "n_layers": <int>,
    "n_heads": <int>,
    "n_kv_heads": <int or null>,
    "d_model": <int>,
    "activation": "<gelu | relu | silu>",
    "norm_type": "<layernorm | rmsnorm>",
    "pos_encoding": "<learned | sinusoidal | rotary | none>",
    "ffn_type": "<standard | swiglu>",
    "topology": "<pre_ln | post_ln | parallel>",
    "scale_factor": <float or null>
  },
  "reasoning": "<Detailed scientific rationale>"
}
"""


class LLMSearchAgent:
    """
    Autonomous Research Search Agent.
    Interacts with LLM reasoning models via Provider Adapter abstraction.
    Freezes (provider, model) as invariant experimental conditions.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        strict: bool = True,
        allow_simulation: bool = False,
        reasoning_mode: str = "full",
        base_url: Optional[str] = None
    ):
        self.reasoning_mode = reasoning_mode
        self.strict = strict
        self.allow_simulation = allow_simulation
        self.base_url = base_url

        # Instantiate provider adapter
        self.adapter = create_provider_adapter(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            strict=strict,
            allow_simulation=allow_simulation
        )

        # Freeze provider + model as immutable experimental variables
        self.provider = self.adapter.name
        self.model = getattr(self.adapter, "model_id", None) or str(self.adapter.model)
        self._last_usage = {}

        print(f"[LLMSearchAgent] Initialized with FROZEN variable: provider='{self.provider}', model='{self.model}', reasoning_mode='{self.reasoning_mode}'", flush=True)

    def verify_connection(self) -> Tuple[bool, Any]:
        """Verifies connectivity to the configured provider."""
        return self.adapter.verify_connection()

    def propose_experiment(
        self,
        iteration_idx: int,
        history: List[Dict[str, Any]],
        incumbent_best_loss: float,
        base_loss: float,
        task: str = "dyck"
    ) -> Dict[str, Any]:
        """
        Formulates the next hypothesis and hyperparameter modification proposal.
        """
        t_start = time.time()

        history_summary = []
        for item in history:
            it_idx = item.get("iteration") if item.get("iteration") is not None else item.get("iteration_index")
            hypo = item.get("hypothesis") or item.get("hypothesis_text") or item.get("reasoning") or ""
            cfg = item.get("config") or item.get("hyperparameters") or item.get("hyperparameters_tested") or item.get("modifications") or {}
            val_l = item.get("val_loss")
            ood_l = item.get("ood_loss")
            rel_g = item.get("relative_gain_pct") if item.get("relative_gain_pct") is not None else item.get("gap_rel")

            h_entry = {
                "iteration": it_idx,
                "hypothesis": hypo,
                "target_component": item.get("target_component", "architecture"),
                "configuration": cfg,
                "status": item.get("status", "SUCCESS"),
                "val_loss": val_l,
                "ood_loss": ood_l,
                "relative_gain_pct": rel_g
            }
            if item.get("traceback"):
                h_entry["error_traceback"] = str(item.get("traceback"))[:300]
            history_summary.append(h_entry)

        # Format historical context according to reasoning ablation mode
        if self.reasoning_mode == "no_history":
            history_text = "No historical context provided (Ablation Mode: Zero-Shot Proposal). Propose candidate based solely on initial task and baseline specifications."
        elif self.reasoning_mode == "history_no_reflection":
            numeric_history = [
                {
                    "iteration": h.get("iteration"),
                    "configuration": h.get("configuration"),
                    "status": h.get("status"),
                    "val_loss": h.get("val_loss"),
                    "ood_loss": h.get("ood_loss")
                }
                for h in history_summary
            ]
            history_text = json.dumps(numeric_history, indent=2)
        else:
            history_text = json.dumps(history_summary, indent=2)

        user_prompt = f"""Task: {task.upper()} sequence modeling
Current Iteration: {iteration_idx}
Baseline Loss (Iteration 0): {base_loss:.4f}
Current Incumbent Best Validation Loss: {incumbent_best_loss:.4f}
Reasoning Mode: {self.reasoning_mode}

Execution History so far:
{history_text}

Formulate Iteration {iteration_idx}'s hypothesis and concrete parameter modifications. Output ONLY valid JSON matching the specified schema.
"""
        prompt_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:16]

        if self.provider == "adaptive_simulation":
            if not self.allow_simulation:
                raise SimulationDisallowedError("Simulation disallowed in strict mode.")
            result = self._call_adaptive_simulation(iteration_idx, history_summary, incumbent_best_loss, base_loss, task)
            raw_response_text = json.dumps(result, indent=2)
            usage = {}
            latency = 0.05
        else:
            result, raw_response_text, usage, latency = self.adapter.complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.7
            )
            self._last_usage = usage

        # Normalize schema keys
        if not isinstance(result, dict):
            result = {"modifications": {}}
        if "modifications" not in result:
            result["modifications"] = {}
        if "hypothesis" not in result:
            result["hypothesis"] = result.get("hypothesis_text", f"Iteration {iteration_idx} modification")
        if "hypothesis_text" not in result:
            result["hypothesis_text"] = result["hypothesis"]
        if "predicted_delta_loss" not in result:
            result["predicted_delta_loss"] = 0.0

        critic_prompt_hash = None
        critic_raw_response = None

        # Dual-Agent Verification / Critic Refinement Pass
        if self.reasoning_mode == "critic_refine":
            critic_prompt = f"""You are a senior adversarial machine learning verification critic.
Review the following proposed optimization candidate:
{json.dumps(result, indent=2)}

Historical execution context:
{history_text}

Task: Check whether this modification repeats a known failed configuration, violates head-dimension divisibility, or risks divergence.
Refine the hypothesis, predicted_delta_loss, and hyperparameters to maximize stability and search efficiency.
Output ONLY the final verified/refined JSON matching the original schema.
"""
            critic_prompt_hash = hashlib.sha256(critic_prompt.encode("utf-8")).hexdigest()[:16]
            try:
                if self.provider == "adaptive_simulation":
                    refined = copy.deepcopy(result)
                    mods = refined.get("modifications", {})
                    d_m = mods.get("d_model", 256)
                    n_h = mods.get("n_heads", 4)
                    if d_m % n_h != 0:
                        mods["d_model"] = (d_m // n_h) * n_h
                    refined["predicted_delta_loss"] = round(float(refined.get("predicted_delta_loss", 0.02)) * 0.7, 4)
                    refined["hypothesis_text"] = f"[CRITIC-VERIFIED] {refined.get('hypothesis_text', '')}"
                    refined["hypothesis"] = refined["hypothesis_text"]
                    critic_raw_response = json.dumps(refined, indent=2)
                else:
                    refined, critic_raw_response, _, _ = self.adapter.complete(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=critic_prompt,
                        temperature=0.5
                    )
                if isinstance(refined, dict) and "modifications" in refined:
                    result = refined
                    result["critic_refined"] = True
            except Exception as e:
                print(f"[LLMSearchAgent] Critic refinement pass skipped due to error: {e}")

        total_latency = round(time.time() - t_start, 3)

        result["provider"] = self.provider
        result["model"] = self.model
        result["reasoning_mode"] = self.reasoning_mode
        result["prompt_hash"] = prompt_hash
        result["raw_prompt"] = user_prompt
        result["raw_response"] = raw_response_text
        result["latency_seconds"] = total_latency
        parse_audit = result.pop("_parse_audit", None) or getattr(self, "_last_usage", {})
        result["parse_audit"] = parse_audit
        result["token_usage"] = {
            "prompt_tokens": parse_audit.get("prompt_tokens", 0),
            "completion_tokens": parse_audit.get("completion_tokens", 0),
            "total_tokens": parse_audit.get("total_tokens", 0)
        } if parse_audit else getattr(self, "_last_usage", {})
        result["reasoning_effort"] = "medium" if self.provider == "gemini" else None
        if critic_prompt_hash:
            result["critic_prompt_hash"] = critic_prompt_hash
            result["critic_raw_response"] = critic_raw_response

        return result

    def generate_next_hypothesis(self, iteration_idx, history, incumbent_best_loss, base_loss, task="dyck"):
        """Backwards-compatible alias for propose_experiment."""
        return self.propose_experiment(iteration_idx, history, incumbent_best_loss, base_loss, task=task)

    def _call_adaptive_simulation(self, iteration_idx, history, incumbent_best_loss, base_loss, task="dyck"):
        """Offline simulation fallback for unit tests and CI sanity checks."""
        mode = self.reasoning_mode
        if mode == "no_history":
            proposals = [
                {
                    "hypothesis_text": "[ZERO-SHOT ABLATION] Broad structural scan: Test Pre-LN with RMSNorm and GELU (width=256).",
                    "target_component": "normalization",
                    "predicted_delta_loss": 0.03,
                    "modifications": {
                        "lr": 0.001, "weight_decay": 0.01, "n_layers": 4, "n_heads": 4,
                        "d_model": 256, "activation": "gelu", "norm_type": "rmsnorm",
                        "pos_encoding": "learned", "ffn_type": "standard", "topology": "pre_ln", "scale_factor": None
                    },
                    "reasoning": "Zero-shot testing of standard normalization baseline."
                },
                {
                    "hypothesis_text": "[ZERO-SHOT ABLATION] Rotary embedding scan: Evaluate relative position encoding for sequence tracking.",
                    "target_component": "positional_encoding",
                    "predicted_delta_loss": 0.035,
                    "modifications": {
                        "lr": 0.0015, "weight_decay": 0.02, "n_layers": 4, "n_heads": 4,
                        "d_model": 128, "activation": "silu", "norm_type": "layernorm",
                        "pos_encoding": "rotary", "ffn_type": "standard", "topology": "pre_ln", "scale_factor": None
                    },
                    "reasoning": "Zero-shot RoPE inductive bias trial."
                }
            ]
            return proposals[(iteration_idx - 1) % len(proposals)]
        elif mode == "history_no_reflection":
            decay_lr = max(1e-4, 0.003 * (0.75 ** (iteration_idx - 1)))
            step_wd = round(0.01 + 0.005 * iteration_idx, 4)
            return {
                "hypothesis_text": f"[NUMERIC HISTORY ONLY] Coordinate step #{iteration_idx}: lr={decay_lr:.5f}, weight_decay={step_wd}.",
                "target_component": "optimizer",
                "predicted_delta_loss": round(0.02 * (0.9 ** iteration_idx), 4),
                "modifications": {
                    "lr": round(decay_lr, 5), "weight_decay": step_wd, "n_layers": 4, "n_heads": 4,
                    "d_model": 256, "activation": "silu", "norm_type": "rmsnorm",
                    "pos_encoding": "learned", "ffn_type": "standard", "topology": "pre_ln", "scale_factor": None
                },
                "reasoning": "Numerical coordinate descent without qualitative verbal reflection."
            }
        else:
            return {
                "hypothesis_text": f"[SEMANTIC REFLECTION] For {task.upper()}, Rotary Position Embeddings (RoPE) and RMSNorm preserve token distance representations across nested bracket closures.",
                "target_component": "positional_encoding",
                "predicted_delta_loss": 0.045,
                "modifications": {
                    "lr": 0.001, "weight_decay": 0.01, "n_layers": 4, "n_heads": 4,
                    "d_model": 256, "activation": "gelu", "norm_type": "rmsnorm",
                    "pos_encoding": "rotary", "ffn_type": "swiglu", "topology": "pre_ln", "scale_factor": None
                },
                "reasoning": "Rotary encoding and SwiGLU inductive bias for hierarchical sequence grammar."
            }


# Backwards compatibility alias
LLMResearchAgent = LLMSearchAgent
