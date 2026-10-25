"""
agent_scaffold/llm_agent.py

Autonomous Research Search Agent for Neural Architecture Optimization.
Enforces:
  1. Single source of truth for the design space via search_space.render_search_space_for_prompt()
  2. Zero OOD leakage (evaluates only validation loss, strips all OOD metrics)
  3. Experimental decomposition modes:
     - task_framing: 'named' vs 'anonymized'
     - feedback: 'real' vs 'shuffled' vs 'none'
  4. Precise calibration definition: predicted_delta_loss is relative to incumbent best
  5. Robust one-shot re-prompting on schema/bound violations
"""

import os
import sys
import copy
import json
import time
import re
import random
import hashlib
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from search_space import (
    SEARCH_SPACE_SPEC,
    render_search_space_for_prompt,
    validate_proposal,
    sanitize_configuration,
    get_obfuscated_spec,
    deobfuscate_proposal
)
from agent_scaffold.provider_adapters import (
    BaseProviderAdapter,
    create_provider_adapter,
    SimulationDisallowedError,
    extract_json_payload,
    parse_llm_json
)


def build_system_prompt(task_framing: str = "named", obfuscated: bool = False) -> str:
    """Dynamically generates the system prompt from the single source of truth search space."""
    spec_text = render_search_space_for_prompt(get_obfuscated_spec() if obfuscated else SEARCH_SPACE_SPEC)
    
    if task_framing == "anonymized":
        task_desc = "an unseen synthetic sequence modeling dataset (Task A)"
    else:
        task_desc = "an unseen sequence modeling dataset"

    return f"""You are an expert autonomous machine learning researcher specializing in neural architecture search, Transformer optimization, and sequence modeling.
Your objective is to minimize the holdout validation cross-entropy loss of a Small Transformer language model on {task_desc}.

You have access to a formal architecture design space:
{spec_text}

You will receive the history of previous iterations, including:
- Hypotheses tested
- Architectural configurations evaluated
- Execution status (OK, TIMEOUT, CRASH, INVALID_PROPOSAL)
- Validation loss achieved
- Current incumbent best validation loss

You must analyze the historical progression and propose the NEXT architectural hypothesis and concrete modifications.
You must also provide your OWN quantitative prediction for expected loss reduction (predicted_delta_loss) relative to the CURRENT INCUMBENT BEST loss.

CRITICAL INSTRUCTIONS:
1. Output ONLY a valid JSON object starting with {{ and ending with }}. Do NOT wrap in markdown fences or commentary.
2. Keep the "reasoning" text concise (under 60 words).
3. In string fields, use single quotes for technical terms. NEVER use unescaped double quotes inside text strings.
4. All parameter modifications must strictly obey the allowed bounds and discrete choices.

Response schema:
{{
  "hypothesis_text": "<Clear technical hypothesis under 30 words>",
  "target_component": "<attention | ffn | topology | positional_encoding | normalization | optimizer>",
  "predicted_delta_loss": <float, expected reduction relative to current incumbent best>,
  "modifications": {{
    "lr": <float>,
    "weight_decay": <float>,
    "d_model": <int>,
    "n_layers": <int>,
    "n_heads": <int>,
    "d_ff": <int>,
    "activation": "<gelu | relu | silu>",
    "norm_type": "<layernorm | rmsnorm>",
    "pos_encoding": "<learned | sinusoidal | rotary | none>",
    "ffn_type": "<standard | swiglu>",
    "topology": "<pre_ln | post_ln | parallel>",
    "scale_factor": <float or null>,
    "n_kv_heads": <int or null>
  }},
  "reasoning": "<Concise rationale under 60 words>"
}}"""


class LLMSearchAgent:
    """
    Autonomous Research Search Agent.
    Interacts with LLM reasoning models via Provider Adapter abstraction.
    Supports decomposition arms:
      - task_framing: 'named' (Dyck / FSM) vs 'anonymized' (Task A)
      - feedback: 'real' (true val losses) vs 'shuffled' (permuted losses) vs 'none' (zero-shot)
      - obfuscated: hyperparameter names stripped to param_alpha, param_beta, etc.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        strict: bool = True,
        allow_simulation: bool = False,
        reasoning_mode: str = "full",
        task_framing: str = "named",
        feedback: str = "real",
        obfuscated: bool = False,
        base_url: Optional[str] = None
    ):
        self.reasoning_mode = reasoning_mode
        self.task_framing = task_framing
        self.feedback = feedback
        self.obfuscated = obfuscated
        self.strict = strict
        self.allow_simulation = allow_simulation
        self.base_url = base_url

        self.adapter = create_provider_adapter(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            strict=strict,
            allow_simulation=allow_simulation
        )

        self.provider = self.adapter.name
        self.model = getattr(self.adapter, "model_id", None) or str(self.adapter.model)
        self._last_usage = {}

        print(
            f"[LLMSearchAgent] Initialized: provider='{self.provider}', model='{self.model}', "
            f"framing='{self.task_framing}', feedback='{self.feedback}', obfuscated={self.obfuscated}",
            flush=True
        )

    def _prepare_history(self, raw_history: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], float]:
        """
        Prepares historical context under the assigned feedback condition.
        Crucial guarantees:
          1. OOD metrics are completely omitted.
          2. Shuffled feedback randomly permutes validation losses among past runs.
          3. Incumbent is recomputed from the presented validation losses.
        """
        if self.feedback == "none" or self.reasoning_mode == "no_history":
            return [], 99.0

        clean_history = []
        val_losses = []

        for item in raw_history:
            it_idx = item.get("iteration") if item.get("iteration") is not None else item.get("iteration_index", 0)
            status = item.get("status", "OK")
            val_l = item.get("val_loss")
            
            entry = {
                "iteration": it_idx,
                "hypothesis": item.get("hypothesis") or item.get("hypothesis_text") or "",
                "target_component": item.get("target_component", "architecture"),
                "configuration": item.get("config") or item.get("modifications") or {},
                "status": status,
                "val_loss": val_l
            }
            if item.get("error_trace"):
                entry["error_detail"] = str(item.get("error_trace"))[:200]
            clean_history.append(entry)
            if val_l is not None and isinstance(val_l, (int, float)) and val_l < 90.0:
                val_losses.append(val_l)

        # Shuffled feedback condition: permute validation losses
        if self.feedback == "shuffled" and len(val_losses) > 1:
            rng = random.Random(seed + len(clean_history) * 101)
            shuffled_vals = list(val_losses)
            rng.shuffle(shuffled_vals)
            v_idx = 0
            for entry in clean_history:
                if entry["val_loss"] is not None and entry["val_loss"] < 90.0:
                    entry["val_loss"] = round(shuffled_vals[v_idx], 6)
                    v_idx += 1

        # Recompute incumbent from visible losses
        visible_losses = [e["val_loss"] for e in clean_history if e["val_loss"] is not None and e["val_loss"] < 90.0]
        incumbent = min(visible_losses) if visible_losses else 99.0
        return clean_history, incumbent

    def propose_experiment(
        self,
        iteration_idx: int,
        history: List[Dict[str, Any]],
        incumbent_best_loss: float,
        base_loss: float,
        task: str = "dyck",
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Proposes the next architectural modification.
        Enforces strict validation and re-prompts once upon schema/bound errors.
        """
        t_start = time.time()
        system_prompt = build_system_prompt(task_framing=self.task_framing, obfuscated=self.obfuscated)

        # Build task heading based on task_framing
        if self.task_framing == "anonymized":
            task_header = "Synthetic sequence modeling (Task A)"
        else:
            task_header = f"{task.upper()} sequence modeling"

        # Prepare history according to feedback condition
        prepared_history, active_incumbent = self._prepare_history(history, seed=seed)
        if self.feedback == "none" or self.reasoning_mode == "no_history":
            history_text = "No history provided (Zero-Shot condition). Propose a candidate based on initial specification."
        else:
            history_text = json.dumps(prepared_history, indent=2)

        user_prompt = f"""Task: {task_header}
Current Iteration: {iteration_idx}
Initial Baseline Loss (Iteration 0): {base_loss:.4f}
Current Incumbent Best Validation Loss: {active_incumbent:.4f}

Execution History so far:
{history_text}

Propose Iteration {iteration_idx}'s hypothesis and exact modifications. Output ONLY a valid JSON object matching the schema."""

        prompt_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:16]

        parse_status = "UNKNOWN"
        attempts = 0
        raw_response_text = ""
        result = None
        usage = {}

        # First attempt
        attempts += 1
        if self.provider == "adaptive_simulation":
            if not self.allow_simulation:
                raise SimulationDisallowedError("Simulation disallowed in strict mode.")
            result = self._call_adaptive_simulation(iteration_idx, task=task)
            raw_response_text = json.dumps(result, indent=2)
            parse_status = "OK"
        else:
            result, raw_response_text, usage, latency = self.adapter.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )
            self._last_usage = usage
            parse_status = getattr(self.adapter, "_last_parse_status", "OK")

        if self.obfuscated and result:
            result = deobfuscate_proposal(result)

        # Strict validation
        is_valid, violations = validate_proposal(result) if result else (False, ["Null result"])

        # Re-prompt once if proposal is invalid
        if not is_valid and self.provider != "adaptive_simulation":
            attempts += 1
            reprompt = f"""Your previous proposal violated the following search space rules:
{json.dumps(violations, indent=2)}

Please correct these violations immediately.
Remember:
- d_model must be in {{128, 256, 512}} and divisible by n_heads.
- For pos_encoding='rotary', d_model // n_heads must be an even integer.
- Return ONLY the corrected JSON object."""
            try:
                corrected, raw_corr, usage_corr, _ = self.adapter.complete(
                    system_prompt=system_prompt,
                    user_prompt=reprompt,
                    temperature=0.3
                )
                if self.obfuscated and corrected:
                    corrected = deobfuscate_proposal(corrected)
                corr_valid, corr_violations = validate_proposal(corrected)
                if corr_valid:
                    result = corrected
                    raw_response_text = raw_corr
                    is_valid = True
                    violations = []
                    parse_status = "RETRY_OK"
                else:
                    violations = corr_violations
            except Exception as e:
                violations.append(f"Retry failed with exception: {e}")

        total_latency = round(time.time() - t_start, 3)

        # Package standardized proposal output
        out = {
            "hypothesis": (result.get("hypothesis_text") or result.get("hypothesis", f"Iteration {iteration_idx}")) if result else "Failed Proposal",
            "hypothesis_text": (result.get("hypothesis_text") or result.get("hypothesis", f"Iteration {iteration_idx}")) if result else "Failed Proposal",
            "target_component": result.get("target_component", "architecture") if result else "unknown",
            "predicted_delta_loss": float(result.get("predicted_delta_loss", 0.0)) if result and isinstance(result.get("predicted_delta_loss"), (int, float)) else 0.0,
            "modifications": result.get("modifications", {}) if (result and is_valid) else {},
            "reasoning": result.get("reasoning", "") if result else "",
            "is_valid": is_valid,
            "violations": violations,
            "status": "OK" if is_valid else "INVALID_PROPOSAL",
            "parse_status": parse_status,
            "parse_attempts": attempts,
            "provider": self.provider,
            "model": self.model,
            "task_framing": self.task_framing,
            "feedback": self.feedback,
            "reasoning_mode": self.reasoning_mode,
            "prompt_hash": prompt_hash,
            "raw_prompt": user_prompt,
            "raw_response": raw_response_text,
            "latency_seconds": total_latency,
            "token_usage": usage
        }
        return out

    def generate_next_hypothesis(self, iteration_idx, history, incumbent_best_loss, base_loss, task="dyck"):
        return self.propose_experiment(iteration_idx, history, incumbent_best_loss, base_loss, task=task)

    def _call_adaptive_simulation(self, iteration_idx: int, task: str = "dyck") -> Dict[str, Any]:
        """Offline simulation for unit testing only."""
        if self.task_framing == "anonymized":
            hypo = f"[ANONYMIZED TEST] Iteration {iteration_idx}: Test Pre-LN with RMSNorm."
        else:
            hypo = f"[NAMED TEST] Iteration {iteration_idx} on {task}: Test RoPE inductive bias."

        pos = "rotary" if (self.task_framing != "anonymized" and task == "dyck") else "learned"
        return {
            "hypothesis_text": hypo,
            "target_component": "positional_encoding",
            "predicted_delta_loss": 0.03,
            "modifications": {
                "lr": 0.001,
                "weight_decay": 0.01,
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 4,
                "d_ff": 1024,
                "activation": "gelu",
                "norm_type": "rmsnorm",
                "pos_encoding": pos,
                "ffn_type": "standard",
                "topology": "pre_ln",
                "scale_factor": None,
                "n_kv_heads": None
            },
            "reasoning": "Standard simulated proposal for test verification."
        }


LLMResearchAgent = LLMSearchAgent
