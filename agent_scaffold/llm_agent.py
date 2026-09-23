"""
LLM Agent Reasoning Engine.
Connects to Anthropic Claude (via anthropic SDK) or OpenAI GPT-4o (via openai SDK).

STRICT MODE (default): if no API key/provider is available, the agent raises
RuntimeError instead of silently falling back to a hardcoded simulation. This
prevents fake, scripted "hypotheses" from ever being logged as real agent output
without an explicit, deliberate opt-in.

Every response (real or simulated) is tagged with "provider" so downstream
trace logs are self-auditing: you can grep any trace file for
"adaptive_simulation" and know instantly whether that run is contaminated.

The agent receives:
- Task & dataset description
- Base Transformer specifications
- Full chronological trace of prior iterations (hypotheses, parameters, losses, crashes)
- Current incumbent best validation loss

The agent generates:
- Technical hypothesis text explaining the inductive bias
- Target component classification
- The agent's OWN predicted delta loss (predicted_delta_loss)
- Concrete dictionary of hyperparameter modifications
"""

import os
import json
import re
import time

SYSTEM_PROMPT = """You are an expert autonomous machine learning researcher specializing in neural architecture search and Transformer optimization.
Your objective is to minimize the holdout validation cross-entropy loss of a small Transformer language model on an unseen synthetic algebraic state-transition sequence modeling dataset.

You have access to the following mutable design space:
1. Learning rate (lr): float in range [1e-5, 5e-2]
2. Weight decay (weight_decay): float in range [0.0, 0.2]
3. Number of layers (n_layers): int in {2, 4, 6, 8}
4. Number of attention heads (n_heads): int in {2, 4, 8}
5. Model dimension (d_model): int in {128, 256, 512} (must be divisible by n_heads)
6. Activation function (activation): "gelu", "relu", or "silu"
7. Normalization layer (norm_type): "layernorm" or "rmsnorm"
8. Attention scale factor (scale_factor): float or null (default is 1/sqrt(d_k))

You will receive the full history of previous trials, including:
- Hypotheses tested
- Hyperparameters used
- Validation and training losses achieved
- Execution errors or tracebacks if any occurred
- The current incumbent best loss

You must analyze the historical progression, diagnose whether previous attempts underfit, overfit, diverged, or plateaued, and propose the NEXT logical hypothesis and exact modifications.
You must also provide your OWN quantitative prediction for the expected loss reduction (predicted_delta_loss).

You must respond with ONLY a valid JSON object matching this schema:
{
  "hypothesis_text": "<Clear technical hypothesis explaining why this modification will improve state-transition modeling>",
  "target_component": "<attention | optimizer | normalization | activation | depth_width>",
  "predicted_delta_loss": <float, your expected reduction in validation loss, e.g. 0.04>,
  "modifications": {
    "lr": <float>,
    "weight_decay": <float>,
    "n_layers": <int>,
    "n_heads": <int>,
    "d_model": <int>,
    "activation": "<gelu | relu | silu>",
    "norm_type": "<layernorm | rmsnorm>",
    "scale_factor": <float or null>
  },
  "reasoning": "<Detailed scientific rationale>"
}
"""


class SimulationDisallowedError(RuntimeError):
    """Raised when no real LLM provider is configured and strict mode forbids
    silently falling back to the hardcoded simulation."""
    pass


class LLMResearchAgent:
    def __init__(self, provider=None, model=None, api_key=None, strict=True,
                 allow_simulation=False):
        """
        strict: if True (default), refuses to silently use the simulation
            fallback when no API key/provider is found. This is what you want
            for any run whose output might be reported in the paper.
        allow_simulation: must be explicitly set True (together with
            strict=False, or on its own) to permit the simulation path.
            Two separate knobs so a single typo can't accidentally unlock it.
        """
        self.anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.nvidia_key = api_key if (api_key and str(api_key).startswith("nvapi-")) else os.environ.get("NVIDIA_API_KEY")
        self.strict = strict
        self.allow_simulation = allow_simulation

        if provider:
            self.provider = provider
        elif self.nvidia_key or (api_key and str(api_key).startswith("nvapi-")):
            self.provider = "nvidia"
        elif self.anthropic_key:
            self.provider = "anthropic"
        elif self.openai_key:
            self.provider = "openai"
        else:
            self.provider = "adaptive_simulation"

        if self.provider == "adaptive_simulation" and not self.allow_simulation:
            raise SimulationDisallowedError(
                "No ANTHROPIC_API_KEY, OPENAI_API_KEY, or NVIDIA_API_KEY found, and no provider "
                "was explicitly specified. Refusing to silently fall back to "
                "the scripted simulation, since output from that path is not "
                "real LLM reasoning and must never be reported as agent output.\n\n"
                "If this is genuinely intentional (e.g. a plumbing/CI smoke "
                "test with no API access), construct the agent with "
                "allow_simulation=True (or pass --allow-simulation on the "
                "command line) to opt in explicitly."
            )

        self.model = model
        if self.provider in ["nvidia", "nemotron"]:
            import openai
            self.nvidia_key = self.nvidia_key or api_key or os.environ.get("NVIDIA_API_KEY")
            self.client = openai.OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.nvidia_key
            )
            self.model = self.model or "nvidia/nemotron-3-ultra-550b-a55b"
            print(f"[LLM Agent] Initialized NVIDIA NIM Client (Model: {self.model})")
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.anthropic_key)
            self.model = self.model or "claude-3-5-sonnet-20241022"
            print(f"[LLM Agent] Initialized Anthropic Claude Client (Model: {self.model})")
        elif self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=self.openai_key)
            self.model = self.model or "gpt-4o"
            print(f"[LLM Agent] Initialized OpenAI Client (Model: {self.model})")
        else:
            self.client = None
            print("[LLM Agent] *** WARNING: RUNNING IN ADAPTIVE SIMULATION MODE ***")
            print("[LLM Agent] Output is scripted, NOT real LLM reasoning.")
            print("[LLM Agent] Every returned record will be tagged provider='adaptive_simulation'.")
            print("[LLM Agent] To connect live Claude reasoning, set ANTHROPIC_API_KEY in your environment.")

    def generate_next_hypothesis(self, iteration_idx, history, incumbent_best_loss, base_loss):
        """
        Generates the next hypothesis from the LLM based on execution history.
        Returns a dict that always includes a "provider" field identifying the
        real source of the hypothesis (anthropic / openai / adaptive_simulation).
        """
        history_summary = []
        for item in history:
            h_entry = {
                "iteration": item.get("iteration_index"),
                "hypothesis": item.get("hypothesis"),
                "target_component": item.get("target_component"),
                "hyperparameters_tested": item.get("hyperparameters"),
                "status": item.get("status"),
                "val_loss": item.get("val_loss"),
                "relative_gain_pct": item.get("relative_gain_pct")
            }
            if item.get("traceback"):
                h_entry["error_traceback"] = item.get("traceback")[:300]
            history_summary.append(h_entry)

        user_prompt = f"""Current Iteration: {iteration_idx}
Baseline Loss (Iteration 0): {base_loss:.4f}
Current Incumbent Best Validation Loss: {incumbent_best_loss:.4f}

Execution History so far:
{json.dumps(history_summary, indent=2)}

Formulate Iteration {iteration_idx}'s hypothesis and concrete parameter modifications. Output ONLY valid JSON.
"""

        if self.provider == "anthropic":
            result = self._call_anthropic(user_prompt)
        elif self.provider in ["nvidia", "nemotron"]:
            result = self._call_nvidia(user_prompt)
        elif self.provider == "openai":
            result = self._call_openai(user_prompt)
        else:
            result = self._call_adaptive_simulation(iteration_idx, history, incumbent_best_loss, base_loss)

        # Self-audit tag: every record, real or simulated, carries its true source.
        result["provider"] = self.provider
        result["model"] = self.model if self.provider != "adaptive_simulation" else "adaptive_simulation"
        return result

    def _call_nvidia(self, user_prompt, max_retries=5):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.6,
                    max_tokens=3000
                )
                content = resp.choices[0].message.content
                return self._extract_json(content)
            except Exception as e:
                last_err = e
                backoff = attempt * 3
                print(f"[LLM Agent] NVIDIA NIM call failed (attempt {attempt}/{max_retries}: {type(e).__name__} - {e}). Retrying in {backoff}s...")
                time.sleep(backoff)
        raise RuntimeError(f"NVIDIA NIM failed after {max_retries} attempts: {last_err}")

    def _call_anthropic(self, user_prompt):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.7,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        content = resp.content[0].text
        return self._extract_json(content)

    def _call_openai(self, user_prompt):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = resp.choices[0].message.content
        return self._extract_json(content)

    def _call_adaptive_simulation(self, iteration_idx, history, incumbent_best_loss, base_loss):
        """
        SIMULATION ONLY. Dynamically adapts based on previous iteration metrics
        rather than a static list, but this is still NOT real LLM reasoning.
        Only reachable if allow_simulation=True was explicitly passed.
        """
        last_trial = history[-1] if history else None
        last_status = last_trial.get("status") if last_trial else "SUCCESS"
        last_loss = last_trial.get("val_loss") if last_trial else base_loss

        if last_status == "DIVERGED" or (last_loss and last_loss > base_loss * 1.5):
            return {
                "hypothesis_text": "[SIMULATED] Previous trial experienced numerical divergence due to excessive step size. Dampen learning rate by 5x and apply gradient clipping with RMSNorm for variance stabilization.",
                "target_component": "optimizer",
                "predicted_delta_loss": 0.05,
                "modifications": {
                    "lr": 0.0003, "weight_decay": 0.02, "n_layers": 6, "n_heads": 4,
                    "d_model": 256, "activation": "gelu", "norm_type": "rmsnorm",
                    "scale_factor": None
                },
                "reasoning": "[SIMULATED] Divergence recovery heuristic."
            }
        elif iteration_idx == 1:
            return {
                "hypothesis_text": "[SIMULATED] RMSNorm removes mean-centering overhead and stabilizes gradient norms across Dyck bracket token state transitions.",
                "target_component": "normalization",
                "predicted_delta_loss": 0.04,
                "modifications": {
                    "lr": 0.001, "weight_decay": 0.01, "n_layers": 6, "n_heads": 4,
                    "d_model": 256, "activation": "gelu", "norm_type": "rmsnorm",
                    "scale_factor": None
                },
                "reasoning": "[SIMULATED] RMSNorm hypothesis."
            }
        elif iteration_idx == 2:
            return {
                "hypothesis_text": "[SIMULATED] SiLU (Swish) activation provides smooth non-linear gating, enabling better representation of non-local parity updates compared to GELU.",
                "target_component": "activation",
                "predicted_delta_loss": 0.03,
                "modifications": {
                    "lr": 0.001, "weight_decay": 0.01, "n_layers": 6, "n_heads": 4,
                    "d_model": 256, "activation": "silu", "norm_type": "rmsnorm",
                    "scale_factor": None
                },
                "reasoning": "[SIMULATED] Activation smoothing hypothesis."
            }
        elif iteration_idx == 3:
            return {
                "hypothesis_text": "[SIMULATED] Modulating attention scale factor from 1/sqrt(d_k) to 1.2/sqrt(d_k) sharpens attention probability distribution over long context horizons.",
                "target_component": "attention",
                "predicted_delta_loss": 0.025,
                "modifications": {
                    "lr": 0.0008, "weight_decay": 0.015, "n_layers": 6, "n_heads": 8,
                    "d_model": 256, "activation": "silu", "norm_type": "rmsnorm",
                    "scale_factor": 0.15
                },
                "reasoning": "[SIMULATED] Attention sharpening hypothesis."
            }
        else:
            decay_lr = max(1e-4, 0.001 * (0.8 ** (iteration_idx - 3)))
            return {
                "hypothesis_text": f"[SIMULATED] Iteration {iteration_idx}: Refine optimization dynamics with cosine learning rate decay (lr={decay_lr:.5f}) and regularization (weight_decay=0.03).",
                "target_component": "optimizer",
                "predicted_delta_loss": 0.015,
                "modifications": {
                    "lr": round(decay_lr, 5), "weight_decay": 0.03, "n_layers": 6,
                    "n_heads": 4, "d_model": 256, "activation": "silu",
                    "norm_type": "rmsnorm", "scale_factor": None
                },
                "reasoning": "[SIMULATED] Decay annealing hypothesis."
            }

    def _extract_json(self, text):
        # 1. Try direct JSON load
        try:
            return json.loads(text.strip(), strict=False)
        except Exception:
            pass

        # 2. Try markdown fenced code block
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip(), strict=False)
            except Exception:
                pass

        # 3. Find balanced braces starting from first '{'
        start = text.find('{')
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i+1]
                            try:
                                return json.loads(candidate, strict=False)
                            except Exception:
                                break

        # 4. Fallback to greedy regex
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0), strict=False)

        return json.loads(text, strict=False)
