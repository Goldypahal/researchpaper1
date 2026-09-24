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
import copy
import hashlib

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


class SimulationDisallowedError(RuntimeError):
    """Raised when no real LLM provider is configured and strict mode forbids
    silently falling back to the hardcoded simulation."""
    pass


class LLMResearchAgent:
    def __init__(self, provider=None, model=None, api_key=None, strict=True,
                 allow_simulation=False, reasoning_mode="full", base_url=None):
        """
        strict: if True (default), refuses to silently use the simulation
            fallback when no API key/provider is found. This is mandatory for
            any empirical research runs reported in the paper.
        allow_simulation: must be explicitly set True (together with
            strict=False, or on its own) to permit the simulation path.
        reasoning_mode: one of 'full', 'no_history', 'history_no_reflection', 'critic_refine'.
        """
        self.reasoning_mode = reasoning_mode
        self.strict = strict
        self.allow_simulation = allow_simulation
        self.base_url = base_url

        # Check Kaggle UserSecretsClient if available (running on Kaggle)
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            for key_name in [
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
                "NVIDIA_API_KEY", "HF_TOKEN"
            ]:
                try:
                    secret_val = user_secrets.get_secret(key_name)
                    if secret_val and not os.environ.get(key_name):
                        os.environ[key_name] = secret_val
                except Exception:
                    pass
        except Exception:
            pass

        self.anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.gemini_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.groq_key = api_key or os.environ.get("GROQ_API_KEY")
        self.openrouter_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.nvidia_key = api_key if (api_key and str(api_key).startswith("nvapi-")) else os.environ.get("NVIDIA_API_KEY")

        if provider:
            self.provider = provider
        elif self.anthropic_key:
            self.provider = "anthropic"
        elif self.openai_key:
            self.provider = "openai"
        elif self.gemini_key:
            self.provider = "gemini"
        elif self.groq_key:
            self.provider = "groq"
        elif self.openrouter_key:
            self.provider = "openrouter"
        elif self.nvidia_key or (api_key and str(api_key).startswith("nvapi-")):
            self.provider = "nvidia"
        elif os.environ.get("LOCAL_LLM_MODEL"):
            self.provider = "local"
        else:
            self.provider = "adaptive_simulation"

        if self.provider == "adaptive_simulation" and not self.allow_simulation:
            raise SimulationDisallowedError(
                "\n" + "=" * 78 + "\n"
                "[FATAL ERROR] REAL LLM REASONING REQUIRED BUT NO VALID PROVIDER FOUND\n"
                "=" * 78 + "\n"
                "The experiment attempted to initialize the LLM Research Agent, but no real\n"
                "LLM API key or local model was configured.\n\n"
                "Checked providers in environment & Kaggle Secrets:\n"
                f"  - Anthropic Claude (ANTHROPIC_API_KEY): {'FOUND' if self.anthropic_key else 'NOT FOUND'}\n"
                f"  - OpenAI GPT (OPENAI_API_KEY):           {'FOUND' if self.openai_key else 'NOT FOUND'}\n"
                f"  - Google Gemini (GEMINI_API_KEY):       {'FOUND' if self.gemini_key else 'NOT FOUND'}\n"
                f"  - Groq Llama (GROQ_API_KEY):             {'FOUND' if self.groq_key else 'NOT FOUND'}\n"
                f"  - OpenRouter (OPENROUTER_API_KEY):       {'FOUND' if self.openrouter_key else 'NOT FOUND'}\n"
                f"  - NVIDIA NIM (NVIDIA_API_KEY):           {'FOUND' if self.nvidia_key else 'NOT FOUND'}\n"
                f"  - Local HuggingFace (LOCAL_LLM_MODEL):   {os.environ.get('LOCAL_LLM_MODEL') or 'NOT SET'}\n\n"
                "STRICT MODE IS ACTIVE: Falling back to scripted/adaptive simulation is\n"
                "STRICTLY FORBIDDEN to protect the scientific integrity of empirical results.\n\n"
                "HOW TO FIX:\n"
                "1. If on Kaggle: Add your API key in Kaggle Notebook -> 'Add-ons' -> 'Secrets'\n"
                "   (e.g., GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY).\n"
                "2. If locally: export GEMINI_API_KEY='your-key' (or OPENAI_API_KEY / GROQ_API_KEY).\n"
                "3. If using open weights on Kaggle T4 GPU: set LOCAL_LLM_MODEL='Qwen/Qwen2.5-3B-Instruct'.\n"
                "4. Only for offline CI plumbing smoke tests: pass --allow-simulation explicitly.\n"
                "=" * 78 + "\n"
            )

        self.model = model
        self._init_client(api_key)

    def _init_client(self, api_key=None):
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.anthropic_key)
            self.model = self.model or "claude-3-5-sonnet-20241022"
            print(f"[LLM Agent] Initialized Anthropic Claude Client (Model: {self.model})", flush=True)

        elif self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=self.openai_key, base_url=self.base_url)
            self.model = self.model or "gpt-4o"
            print(f"[LLM Agent] Initialized OpenAI Client (Model: {self.model})", flush=True)

        elif self.provider == "gemini":
            import openai
            # Google Gemini exposes standard OpenAI-compatible API
            self.client = openai.OpenAI(
                base_url=self.base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=self.gemini_key
            )
            self.model = self.model or "gemini-2.0-flash"
            print(f"[LLM Agent] Initialized Google Gemini Client via OpenAI interface (Model: {self.model})", flush=True)

        elif self.provider == "groq":
            import openai
            self.client = openai.OpenAI(
                base_url=self.base_url or "https://api.groq.com/openai/v1",
                api_key=self.groq_key
            )
            self.model = self.model or "llama-3.3-70b-versatile"
            print(f"[LLM Agent] Initialized Groq Client (Model: {self.model})", flush=True)

        elif self.provider == "openrouter":
            import openai
            self.client = openai.OpenAI(
                base_url=self.base_url or "https://openrouter.ai/api/v1",
                api_key=self.openrouter_key
            )
            self.model = self.model or "anthropic/claude-3.5-sonnet"
            print(f"[LLM Agent] Initialized OpenRouter Client (Model: {self.model})", flush=True)

        elif self.provider in ["nvidia", "nemotron"]:
            import openai
            self.client = openai.OpenAI(
                base_url=self.base_url or "https://integrate.api.nvidia.com/v1",
                api_key=self.nvidia_key
            )
            self.model = self.model or "nvidia/nemotron-3-ultra-550b-a55b"
            print(f"[LLM Agent] Initialized NVIDIA NIM Client (Model: {self.model})", flush=True)

        elif self.provider in ["local", "hf"]:
            self._init_local_model()

        else:
            self.client = None
            print("[LLM Agent] *** CAUTION: RUNNING IN SCRIPTED ADAPTIVE SIMULATION MODE ***", flush=True)
            print("[LLM Agent] Output is scripted proxy heuristics, NOT real LLM reasoning.", flush=True)
            print("[LLM Agent] Every record will be tagged provider='adaptive_simulation'.", flush=True)

    def _init_local_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        model_id = self.model or os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        print(f"[LLM Agent] Loading local HuggingFace model on GPU: {model_id} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.model = model_id
        self.client = "local"
        print(f"[LLM Agent] Local model {model_id} loaded successfully on {model.device}.", flush=True)

    def propose_experiment(self, iteration_idx: int, history: list, incumbent_best_loss: float, base_loss: float, task: str = "dyck"):
        """
        Primary interface for autonomous LLM research proposals.
        Generates the next hypothesis from the LLM based on execution history,
        ablation mode, and task grammar specifications.
        Returns a dict containing hypothesis, modifications, prompt hash, raw response,
        and audit metadata.
        """
        import hashlib
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

        # Handle reasoning ablation modes
        if self.reasoning_mode == "no_history":
            history_text = "No historical context provided (Ablation Mode: Zero-Shot Proposal). Propose candidate based solely on initial task and baseline specifications."
        elif self.reasoning_mode == "history_no_reflection":
            # Strip verbal hypotheses and error explanations, keeping only strictly numeric metrics
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
        raw_response_text = ""

        if self.provider == "anthropic":
            result, raw_response_text = self._call_anthropic(user_prompt)
        elif self.provider in ["openai", "gemini", "groq", "openrouter", "nvidia", "nemotron"]:
            result, raw_response_text = self._call_openai_compatible(user_prompt)
        elif self.provider in ["local", "hf"]:
            result, raw_response_text = self._call_local_hf(user_prompt)
        else:
            if not self.allow_simulation:
                raise SimulationDisallowedError("Simulation disallowed in strict mode.")
            result = self._call_adaptive_simulation(iteration_idx, history_summary, incumbent_best_loss, base_loss, task)
            raw_response_text = json.dumps(result, indent=2)

        # Ensure result has standard keys
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
                if self.provider == "anthropic":
                    refined, critic_raw_response = self._call_anthropic(critic_prompt)
                elif self.provider in ["openai", "gemini", "groq", "openrouter", "nvidia", "nemotron"]:
                    refined, critic_raw_response = self._call_openai_compatible(critic_prompt)
                elif self.provider in ["local", "hf"]:
                    refined, critic_raw_response = self._call_local_hf(critic_prompt)
                else:
                    # Simulated critic refinement: sanity check head divisibility and dampen optimistic prediction
                    refined = copy.deepcopy(result)
                    mods = refined.get("modifications", {})
                    d_m = mods.get("d_model", 256)
                    n_h = mods.get("n_heads", 4)
                    if d_m % n_h != 0:
                        mods["d_model"] = (d_m // n_h) * n_h
                    # Calibrate prediction: critics dampen over-optimism by 30%
                    refined["predicted_delta_loss"] = round(float(refined.get("predicted_delta_loss", 0.02)) * 0.7, 4)
                    refined["hypothesis_text"] = f"[CRITIC-VERIFIED] {refined.get('hypothesis_text', '')}"
                    refined["hypothesis"] = refined["hypothesis_text"]
                    critic_raw_response = json.dumps(refined, indent=2)

                if isinstance(refined, dict) and "modifications" in refined:
                    result = refined
                    result["critic_refined"] = True
            except Exception as e:
                print(f"[LLM Agent] Critic refinement pass skipped due to error: {e}")

        latency = round(time.time() - t_start, 3)

        result["provider"] = self.provider
        result["model"] = self.model if self.provider != "adaptive_simulation" else "adaptive_simulation"
        result["reasoning_mode"] = self.reasoning_mode
        result["prompt_hash"] = prompt_hash
        result["raw_prompt"] = user_prompt
        result["raw_response"] = raw_response_text
        result["latency_seconds"] = latency
        if critic_prompt_hash:
            result["critic_prompt_hash"] = critic_prompt_hash
            result["critic_raw_response"] = critic_raw_response

        return result

    def generate_next_hypothesis(self, iteration_idx, history, incumbent_best_loss, base_loss, task="dyck"):
        """Backwards-compatible wrapper routing directly to propose_experiment."""
        return self.propose_experiment(iteration_idx, history, incumbent_best_loss, base_loss, task=task)

    def verify_connection(self):
        """Sends a minimal 1-shot test prompt to verify live provider connection."""
        test_prompt = "Return a JSON object with key 'status' set to 'connected'."
        print(f"[LLM Agent] Verifying live connection to provider='{self.provider}' (model='{self.model}')...", flush=True)
        try:
            if self.provider == "anthropic":
                res, _ = self._call_anthropic(test_prompt)
            elif self.provider in ["openai", "gemini", "groq", "openrouter", "nvidia", "nemotron"]:
                res, _ = self._call_openai_compatible(test_prompt)
            elif self.provider in ["local", "hf"]:
                res, _ = self._call_local_hf(test_prompt)
            else:
                return False, "Using adaptive_simulation (not a real LLM provider)"
            print(f"[LLM Agent] Live connection verified successfully: {res}", flush=True)
            return True, res
        except Exception as e:
            print(f"[LLM Agent] Live connection verification FAILED: {e}", flush=True)
            return False, str(e)

    def _call_openai_compatible(self, user_prompt, max_retries=5):
        """Unified caller for OpenAI, Gemini, Groq, OpenRouter, and NVIDIA NIM endpoints."""
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                # Add json response format if supported by provider
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                }
                if self.provider in ["openai", "groq"]:
                    kwargs["response_format"] = {"type": "json_object"}
                elif self.provider in ["nvidia", "nemotron"]:
                    kwargs["max_tokens"] = 3000
                    kwargs["temperature"] = 0.6

                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                return self._extract_json(content), content
            except Exception as e:
                last_err = e
                backoff = attempt * 3
                print(f"[LLM Agent] Provider '{self.provider}' call failed (attempt {attempt}/{max_retries}: {type(e).__name__} - {e}). Retrying in {backoff}s...", flush=True)
                time.sleep(backoff)
        raise RuntimeError(f"Provider '{self.provider}' failed after {max_retries} attempts: {last_err}")

    def _call_anthropic(self, user_prompt, max_retries=5):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.7,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                content = resp.content[0].text
                return self._extract_json(content), content
            except Exception as e:
                last_err = e
                backoff = attempt * 3
                print(f"[LLM Agent] Anthropic call failed (attempt {attempt}/{max_retries}: {type(e).__name__} - {e}). Retrying in {backoff}s...", flush=True)
                time.sleep(backoff)
        raise RuntimeError(f"Anthropic failed after {max_retries} attempts: {last_err}")

    def _call_local_hf(self, user_prompt):
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        out = self.pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)
        generated_text = out[0]["generated_text"][len(prompt):]
        return self._extract_json(generated_text), generated_text

    def _call_adaptive_simulation(self, iteration_idx, history, incumbent_best_loss, base_loss, task="dyck"):
        """
        SIMULATION ONLY. Dynamically adapts based on previous iteration metrics
        and reasoning mode, rather than a single static list.
        Only reachable if allow_simulation=True was explicitly passed.
        """
        mode = self.reasoning_mode

        # MODE 1: ZERO-SHOT / NO-HISTORY
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
                },
                {
                    "hypothesis_text": "[ZERO-SHOT ABLATION] Gated feed-forward scan: Evaluate SwiGLU gating for sequence feature projection.",
                    "target_component": "ffn",
                    "predicted_delta_loss": 0.04,
                    "modifications": {
                        "lr": 0.002, "weight_decay": 0.005, "n_layers": 4, "n_heads": 4,
                        "d_model": 256, "activation": "silu", "norm_type": "rmsnorm",
                        "pos_encoding": "learned", "ffn_type": "swiglu", "topology": "pre_ln", "scale_factor": None
                    },
                    "reasoning": "Zero-shot SwiGLU projection exploration."
                },
                {
                    "hypothesis_text": "[ZERO-SHOT ABLATION] Parallel block topology scan: Test parallel attention-FFN computation block.",
                    "target_component": "topology",
                    "predicted_delta_loss": 0.02,
                    "modifications": {
                        "lr": 0.0008, "weight_decay": 0.01, "n_layers": 6, "n_heads": 4,
                        "d_model": 256, "activation": "gelu", "norm_type": "rmsnorm",
                        "pos_encoding": "sinusoidal", "ffn_type": "standard", "topology": "parallel", "scale_factor": None
                    },
                    "reasoning": "Zero-shot parallel topology trial."
                }
            ]
            idx = (iteration_idx - 1) % len(proposals)
            return proposals[idx]

        # MODE 2: HISTORY WITHOUT VERBAL REFLECTION (Pure Numerical Coordinate Steps)
        elif mode == "history_no_reflection":
            decay_lr = max(1e-4, 0.003 * (0.75 ** (iteration_idx - 1)))
            step_wd = round(0.01 + 0.005 * iteration_idx, 4)
            d_ff_val = 512 if iteration_idx % 2 == 1 else 1024
            return {
                "hypothesis_text": f"[NUMERIC HISTORY ONLY] Coordinate step #{iteration_idx}: lr={decay_lr:.5f}, weight_decay={step_wd}, d_ff={d_ff_val}.",
                "target_component": "optimizer",
                "predicted_delta_loss": round(0.02 * (0.9 ** iteration_idx), 4),
                "modifications": {
                    "lr": round(decay_lr, 5), "weight_decay": step_wd, "n_layers": 4, "n_heads": 4,
                    "d_model": 256, "d_ff": d_ff_val, "activation": "silu", "norm_type": "rmsnorm",
                    "pos_encoding": "learned", "ffn_type": "standard", "topology": "pre_ln", "scale_factor": None
                },
                "reasoning": "Numerical coordinate descent without qualitative verbal reflection."
            }

        # MODE 3 & 4: FULL REFLECTION & CRITIC REFINE
        else:
            last_trial = history[-1] if history else None
            last_status = last_trial.get("status") if last_trial else "SUCCESS"
            last_loss = last_trial.get("val_loss") if last_trial else base_loss

            if last_status == "DIVERGED" or (last_loss and last_loss > base_loss * 1.5):
                return {
                    "hypothesis_text": "[SEMANTIC REFLECTION] Previous configuration diverged. Dampen learning rate by 3x and introduce RMSNorm for gradient variance stabilization.",
                    "target_component": "optimizer",
                    "predicted_delta_loss": 0.04,
                    "modifications": {
                        "lr": 0.0004, "weight_decay": 0.02, "n_layers": 4, "n_heads": 4,
                        "d_model": 256, "activation": "gelu", "norm_type": "rmsnorm",
                        "pos_encoding": "learned", "ffn_type": "standard", "topology": "pre_ln", "scale_factor": None
                    },
                    "reasoning": "Divergence recovery reflection heuristic."
                }
            elif iteration_idx == 1:
                return {
                    "hypothesis_text": f"[SEMANTIC REFLECTION] For {task.upper()} sequence modeling, Rotary Position Embeddings (RoPE) and RMSNorm preserve token distance representations across nested bracket closures.",
                    "target_component": "positional_encoding",
                    "predicted_delta_loss": 0.045,
                    "modifications": {
                        "lr": 0.001, "weight_decay": 0.01, "n_layers": 4, "n_heads": 4,
                        "d_model": 256, "activation": "gelu", "norm_type": "rmsnorm",
                        "pos_encoding": "rotary", "ffn_type": "swiglu", "topology": "pre_ln", "scale_factor": None
                    },
                    "reasoning": "Rotary encoding and SwiGLU inductive bias for hierarchical sequence grammar."
                }
            elif iteration_idx == 2:
                return {
                    "hypothesis_text": "[SEMANTIC REFLECTION] Incorporate SwiGLU feed-forward projection and increase depth to 6 layers to expand grammatical stack state tracking capacity.",
                    "target_component": "ffn",
                    "predicted_delta_loss": 0.035,
                    "modifications": {
                        "lr": 0.0008, "weight_decay": 0.015, "n_layers": 6, "n_heads": 4,
                        "d_model": 256, "activation": "silu", "norm_type": "rmsnorm",
                        "pos_encoding": "rotary", "ffn_type": "swiglu", "topology": "pre_ln", "scale_factor": None
                    },
                    "reasoning": "SwiGLU gating capacity expansion."
                }
            elif iteration_idx == 3:
                return {
                    "hypothesis_text": "[SEMANTIC REFLECTION] Sharpen attention logits via scale_factor 0.15 to facilitate crisp bracket matching over long context spans.",
                    "target_component": "attention",
                    "predicted_delta_loss": 0.025,
                    "modifications": {
                        "lr": 0.0006, "weight_decay": 0.02, "n_layers": 6, "n_heads": 4,
                        "d_model": 256, "activation": "silu", "norm_type": "rmsnorm",
                        "pos_encoding": "rotary", "ffn_type": "swiglu", "topology": "pre_ln", "scale_factor": 0.15
                    },
                    "reasoning": "Attention scale tuning for sharp boundary detection."
                }
            else:
                decay_lr = max(1e-4, 0.0008 * (0.8 ** (iteration_idx - 3)))
                return {
                    "hypothesis_text": f"[SEMANTIC REFLECTION] Iteration {iteration_idx}: Fine-tune learning rate schedule (lr={decay_lr:.5f}) with weight_decay=0.03 for stability.",
                    "target_component": "optimizer",
                    "predicted_delta_loss": 0.015,
                    "modifications": {
                        "lr": round(decay_lr, 5), "weight_decay": 0.03, "n_layers": 6,
                        "n_heads": 4, "d_model": 256, "activation": "silu",
                        "norm_type": "rmsnorm", "pos_encoding": "rotary", "ffn_type": "swiglu", "topology": "pre_ln", "scale_factor": None
                    },
                    "reasoning": "Convergence stabilization step."
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
