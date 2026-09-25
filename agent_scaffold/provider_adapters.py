"""
Provider Adapter Architecture for LLMSearchAgent.
Provides standardized, model-specific adapters for all supported reasoning providers:
  - Groq (e.g. llama-3.3-70b-versatile, llama-3.1-8b-instant)
  - Mistral (e.g. mistral-small-latest, codestral-latest)
  - Gemini (e.g. gemini-3.6-flash, gemini-2.5-flash)
  - OpenAI (e.g. gpt-4o, gpt-4o-mini)
  - Anthropic (e.g. claude-3-5-sonnet-20241022)
  - OpenRouter (Gateway with pinned underlying model/provider)
  - NVIDIA NIM (e.g. nvidia/nemotron-3-ultra-550b-a55b)
  - Local HuggingFace (on-GPU execution)
  - Adaptive Simulation (strictly gated offline stub for CI)

SCIENTIFIC EXPERIMENT RULE:
  Provider + Model is a FROZEN experimental variable across all seeds in an arm.
  Do NOT mix providers across seeds (e.g. seed 42 -> Gemini, seed 101 -> Groq).
  Different providers/models must be run as distinct experimental arms:
  LLM-Groq, LLM-Mistral, LLM-Gemini.
"""

import os
import json
import re
import time
from typing import Dict, Any, Tuple, Optional


class SimulationDisallowedError(RuntimeError):
    """Raised when no real LLM provider is configured and strict mode forbids
    silently falling back to the hardcoded simulation."""
    pass


def extract_json_payload(text: str) -> Dict[str, Any]:
    """Robust extractor for JSON objects returned by LLMs."""
    # 1. Direct parse
    try:
        return json.loads(text.strip(), strict=False)
    except Exception:
        pass

    # 2. Markdown fenced code block
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip(), strict=False)
        except Exception:
            pass

    # 3. Balanced braces extraction
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

    # 4. Fallback greedy regex
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0), strict=False)
        except Exception:
            pass

    return json.loads(text, strict=False)


class BaseProviderAdapter:
    """Base interface for all LLM reasoning provider adapters."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.name = "base"
        self._last_usage = {}

    def verify_connection(self) -> Tuple[bool, Any]:
        """Sends a minimal 1-shot test prompt to verify connectivity and credentials."""
        raise NotImplementedError

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 5
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], float]:
        """
        Executes completion against the provider.
        Returns:
            (parsed_json, raw_text, usage_metrics, latency_seconds)
        """
        raise NotImplementedError

    def get_audit_metadata(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "last_token_usage": self._last_usage
        }


class OpenAIBaseAdapter(BaseProviderAdapter):
    """Adapter for OpenAI-compatible HTTP endpoints (OpenAI, Groq, Mistral, Gemini, OpenRouter, NVIDIA)."""

    def __init__(self, name: str, default_model: str, model: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 120.0):
        super().__init__(model=model or default_model, api_key=api_key, base_url=base_url)
        self.name = name
        self.timeout = timeout
        if not self.api_key:
            raise SimulationDisallowedError(f"Missing API key for provider '{self.name}'. Please set the environment variable or Kaggle secret.")
        import openai
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def verify_connection(self) -> Tuple[bool, Any]:
        test_prompt = "Return a JSON object with key 'status' set to 'connected'."
        try:
            parsed, raw, usage, lat = self.complete(
                system_prompt="You are a ping test assistant. Return valid JSON only.",
                user_prompt=test_prompt,
                temperature=0.0,
                max_tokens=256,
                max_retries=2
            )
            return True, parsed
        except Exception as e:
            return False, str(e)

    def _build_kwargs(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        return kwargs

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 5
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], float]:
        t0 = time.time()
        last_err = None

        for attempt in range(1, max_retries + 1):
            try:
                kwargs = self._build_kwargs(system_prompt, user_prompt, temperature, max_tokens)
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                if not content:
                    finish_reason = getattr(resp.choices[0], "finish_reason", None)
                    raise RuntimeError(f"Provider '{self.name}' returned empty content (finish_reason={finish_reason}).")

                usage = getattr(resp, "usage", None)
                usage_dict = {}
                if usage:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                self._last_usage = usage_dict
                latency = round(time.time() - t0, 3)
                parsed = extract_json_payload(content)
                return parsed, content, usage_dict, latency

            except Exception as e:
                last_err = e
                backoff = attempt * 3
                print(f"[Adapter:{self.name}] Call failed (attempt {attempt}/{max_retries}: {type(e).__name__} - {e}). Retrying in {backoff}s...", flush=True)
                time.sleep(backoff)

        raise RuntimeError(f"Provider '{self.name}' ({self.model}) failed after {max_retries} attempts: {last_err}")


class GroqAdapter(OpenAIBaseAdapter):
    """Adapter for Groq hardware acceleration API (gpt-oss-120b, gpt-oss-20b, llama-3.3-70b-versatile, etc.)."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # Normalize model aliases
        m = model
        if m in ["gpt-oss-120b", "openai/gpt-oss-120b"]:
            m = "openai/gpt-oss-120b"
        elif m in ["gpt-oss-20b", "openai/gpt-oss-20b"]:
            m = "openai/gpt-oss-20b"
        super().__init__(
            name="groq",
            default_model="openai/gpt-oss-120b",
            model=m,
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            base_url=base_url or "https://api.groq.com/openai/v1"
        )

    def _build_kwargs(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        kwargs = super()._build_kwargs(system_prompt, user_prompt, temperature, max_tokens)
        kwargs["response_format"] = {"type": "json_object"}
        return kwargs


class MistralAdapter(OpenAIBaseAdapter):
    """Adapter for Mistral AI platform (mistral-small-latest, codestral-latest, mistral-large-latest)."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(
            name="mistral",
            default_model="mistral-small-latest",
            model=model,
            api_key=api_key or os.environ.get("MISTRAL_API_KEY"),
            base_url=base_url or "https://api.mistral.ai/v1"
        )

    def _build_kwargs(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        kwargs = super()._build_kwargs(system_prompt, user_prompt, temperature, max_tokens)
        kwargs["response_format"] = {"type": "json_object"}
        return kwargs


class GeminiAdapter(OpenAIBaseAdapter):
    """Adapter for Google Gemini API (gemini-3.6-flash, gemini-2.5-flash) via OpenAI compatibility endpoint."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(
            name="gemini",
            default_model="gemini-3.6-flash",
            model=model,
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=120.0
        )

    def _build_kwargs(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        kwargs = super()._build_kwargs(system_prompt, user_prompt, temperature, max_tokens)
        # Gemini 3.x utilizes internal thinking tokens within total token envelope
        kwargs["max_tokens"] = 8192
        kwargs["reasoning_effort"] = "medium"
        return kwargs


class OpenAIAdapter(OpenAIBaseAdapter):
    """Adapter for OpenAI models (gpt-4o, gpt-4o-mini)."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(
            name="openai",
            default_model="gpt-4o",
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url
        )

    def _build_kwargs(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        kwargs = super()._build_kwargs(system_prompt, user_prompt, temperature, max_tokens)
        kwargs["response_format"] = {"type": "json_object"}
        return kwargs


class AnthropicAdapter(BaseProviderAdapter):
    """Adapter for Anthropic Claude models (claude-3-5-sonnet-20241022)."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(
            model=model or "claude-3-5-sonnet-20241022",
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url
        )
        self.name = "anthropic"
        if not self.api_key:
            raise SimulationDisallowedError("Missing API key for Anthropic. Please set ANTHROPIC_API_KEY.")
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def verify_connection(self) -> Tuple[bool, Any]:
        try:
            parsed, raw, usage, lat = self.complete(
                system_prompt="You are a ping test assistant. Return valid JSON only.",
                user_prompt="Return a JSON object with key 'status' set to 'connected'.",
                temperature=0.0,
                max_tokens=256,
                max_retries=2
            )
            return True, parsed
        except Exception as e:
            return False, str(e)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 5
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], float]:
        t0 = time.time()
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                content = resp.content[0].text
                usage_dict = {
                    "input_tokens": getattr(resp.usage, "input_tokens", None),
                    "output_tokens": getattr(resp.usage, "output_tokens", None),
                }
                self._last_usage = usage_dict
                latency = round(time.time() - t0, 3)
                parsed = extract_json_payload(content)
                return parsed, content, usage_dict, latency
            except Exception as e:
                last_err = e
                backoff = attempt * 3
                print(f"[Adapter:anthropic] Call failed (attempt {attempt}/{max_retries}: {type(e).__name__} - {e}). Retrying in {backoff}s...", flush=True)
                time.sleep(backoff)
        raise RuntimeError(f"Anthropic failed after {max_retries} attempts: {last_err}")


class OpenRouterAdapter(OpenAIBaseAdapter):
    """Adapter for OpenRouter gateway. Requires explicit pinned model identifier for reproducibility."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(
            name="openrouter",
            default_model="nvidia/nemotron-3-ultra-550b-a55b",
            model=model,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url or "https://openrouter.ai/api/v1"
        )


class NvidiaAdapter(OpenAIBaseAdapter):
    """Adapter for NVIDIA NIM endpoints (Nemotron Super/Ultra, Nemotron 70B/340B)."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # Normalize Nemotron model aliases
        m = model
        if m in ["super", "nemotron-super", "nvidia/nemotron-3-super"]:
            m = "nvidia/nemotron-3-super"
        elif m in ["ultra", "nemotron-ultra", "nvidia/nemotron-3-ultra-550b-a55b"]:
            m = "nvidia/nemotron-3-ultra-550b-a55b"
        elif m in ["70b", "nemotron-70b", "nvidia/llama-3.1-nemotron-70b-instruct"]:
            m = "nvidia/llama-3.1-nemotron-70b-instruct"
        elif m in ["340b", "nemotron-340b", "nvidia/nemotron-4-340b-instruct"]:
            m = "nvidia/nemotron-4-340b-instruct"

        super().__init__(
            name="nvidia",
            default_model="nvidia/llama-3.1-nemotron-70b-instruct",
            model=m,
            api_key=api_key or os.environ.get("NVIDIA_API_KEY"),
            base_url=base_url or "https://integrate.api.nvidia.com/v1"
        )

    def _build_kwargs(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        kwargs = super()._build_kwargs(system_prompt, user_prompt, min(temperature, 0.6), min(max_tokens, 3000))
        return kwargs


class LocalHuggingFaceAdapter(BaseProviderAdapter):
    """Adapter for locally loaded HuggingFace open-weight models on GPU."""

    def __init__(self, model: Optional[str] = None):
        model_id = model or os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        super().__init__(model=model_id)
        self.name = "local"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        print(f"[Adapter:local] Loading HuggingFace model: {model_id} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.pipe = pipeline("text-generation", model=hf_model, tokenizer=tokenizer)
        print(f"[Adapter:local] Model {model_id} loaded on {hf_model.device}.", flush=True)

    def verify_connection(self) -> Tuple[bool, Any]:
        try:
            parsed, raw, usage, lat = self.complete(
                system_prompt="Return JSON only.",
                user_prompt="Return a JSON object with key 'status' set to 'connected'."
            )
            return True, parsed
        except Exception as e:
            return False, str(e)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 1
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], float]:
        t0 = time.time()
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        out = self.pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
        generated_text = out[0]["generated_text"][len(prompt):]
        latency = round(time.time() - t0, 3)
        parsed = extract_json_payload(generated_text)
        return parsed, generated_text, {}, latency


class AdaptiveSimulationAdapter(BaseProviderAdapter):
    """
    Offline simulation stub. Strictly gated by allow_simulation=True.
    Used exclusively for CI smoke-testing plumbing without incurring API tokens.
    """

    def __init__(self):
        super().__init__(model="adaptive_simulation")
        self.name = "adaptive_simulation"

    def verify_connection(self) -> Tuple[bool, Any]:
        return False, "Using adaptive_simulation (offline simulation stub, not a real LLM provider)"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 1
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], float]:
        t0 = time.time()
        payload = {
            "hypothesis_text": "[OFFLINE SIMULATION STUB] Pre-LN RMSNorm SwiGLU architectural proposal.",
            "target_component": "ffn",
            "predicted_delta_loss": 0.03,
            "modifications": {
                "lr": 0.001,
                "weight_decay": 0.01,
                "n_layers": 4,
                "n_heads": 4,
                "d_model": 256,
                "activation": "silu",
                "norm_type": "rmsnorm",
                "pos_encoding": "rotary",
                "ffn_type": "swiglu",
                "topology": "pre_ln",
                "scale_factor": None
            },
            "reasoning": "Offline deterministic test stub."
        }
        latency = round(time.time() - t0, 3)
        return payload, json.dumps(payload, indent=2), {}, latency


def create_provider_adapter(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    strict: bool = True,
    allow_simulation: bool = False
) -> BaseProviderAdapter:
    """
    Factory function for Provider Adapters.
    Freezes provider + model selection. Raises SimulationDisallowedError in strict mode
    if credentials are missing.
    """
    # Auto-load Kaggle UserSecrets if present
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for key_name in [
            "GROQ_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY"
        ]:
            if not os.environ.get(key_name):
                try:
                    val = user_secrets.get_secret(key_name)
                    if val:
                        os.environ[key_name] = val
                except Exception:
                    pass
    except Exception:
        pass

    # Detect provider if not specified
    if not provider:
        if api_key and str(api_key).startswith("gsk_"):
            provider = "groq"
        elif api_key and str(api_key).startswith("nvapi-"):
            provider = "nvidia"
        elif os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("MISTRAL_API_KEY"):
            provider = "mistral"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.environ.get("OPENROUTER_API_KEY"):
            provider = "openrouter"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("LOCAL_LLM_MODEL"):
            provider = "local"
        else:
            provider = "adaptive_simulation"

    provider = provider.lower()

    if provider == "adaptive_simulation" and not allow_simulation and strict:
        raise SimulationDisallowedError(
            "\n" + "=" * 78 + "\n"
            "[FATAL ERROR] REAL LLM REASONING REQUIRED BUT NO VALID PROVIDER FOUND\n"
            "=" * 78 + "\n"
            "STRICT MODE: Falling back to scripted/adaptive simulation is FORBIDDEN.\n"
            "Supported providers: Groq, Mistral, Gemini, OpenAI, Anthropic, OpenRouter, NVIDIA, Local.\n"
            "Please configure the respective API key in environment or Kaggle Secrets.\n"
            "=" * 78 + "\n"
        )

    # Instantiate adapter
    if provider == "groq":
        return GroqAdapter(model=model, api_key=api_key, base_url=base_url)
    elif provider == "mistral":
        return MistralAdapter(model=model, api_key=api_key, base_url=base_url)
    elif provider == "gemini":
        return GeminiAdapter(model=model, api_key=api_key, base_url=base_url)
    elif provider == "openai":
        return OpenAIAdapter(model=model, api_key=api_key, base_url=base_url)
    elif provider == "anthropic":
        return AnthropicAdapter(model=model, api_key=api_key, base_url=base_url)
    elif provider == "openrouter":
        return OpenRouterAdapter(model=model, api_key=api_key, base_url=base_url)
    elif provider in ["nvidia", "nemotron"]:
        return NvidiaAdapter(model=model, api_key=api_key, base_url=base_url)
    elif provider in ["local", "hf"]:
        return LocalHuggingFaceAdapter(model=model)
    elif provider == "adaptive_simulation":
        return AdaptiveSimulationAdapter()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
