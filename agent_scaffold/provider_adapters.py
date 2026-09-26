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


class LLMProposalParseError(RuntimeError):
    """Raised when an LLM fails to produce valid JSON adhering to the experiment schema."""
    def __init__(self, message: str, raw_text: str = "", audit_metadata: dict = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.audit_metadata = audit_metadata or {}


def parse_llm_json(text: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Rigorously parses LLM output into JSON.
    Tracks raw validity vs repair without pretending repaired responses were originally valid.

    Returns:
        (parsed_dict_or_None, audit_dict)
    """
    audit = {
        "json_valid": False,
        "repair_attempted": False,
        "repair_success": False,
        "error_msg": None,
        "raw_response": text
    }

    if not text or not isinstance(text, str) or not text.strip():
        audit["error_msg"] = "Empty or null text"
        return None, audit

    cleaned_text = text.strip()

    # 1. Direct raw parse attempt
    try:
        parsed = json.loads(cleaned_text, strict=False)
        if isinstance(parsed, dict):
            audit["json_valid"] = True
            return parsed, audit
    except Exception as e:
        raw_err = str(e)

    # 2. Markdown fenced code block raw attempt
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_text)
    if code_block:
        candidate_block = code_block.group(1).strip()
        try:
            parsed = json.loads(candidate_block, strict=False)
            if isinstance(parsed, dict):
                audit["json_valid"] = True
                return parsed, audit
        except Exception:
            pass

    # 3. Direct balanced braces raw attempt
    start = cleaned_text.find('{')
    candidate_braces = None
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cleaned_text)):
            c = cleaned_text[i]
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
                        candidate_braces = cleaned_text[start:i+1]
                        try:
                            parsed = json.loads(candidate_braces, strict=False)
                            if isinstance(parsed, dict):
                                audit["json_valid"] = True
                                return parsed, audit
                        except Exception:
                            pass
                        break

    # If all raw parse attempts failed, raw output is NOT valid JSON
    audit["json_valid"] = False
    audit["repair_attempted"] = True

    # Best text candidate for repair:
    s = candidate_braces or (code_block.group(1).strip() if code_block else None)
    if not s and start != -1:
        s = cleaned_text[start:]
    if not s:
        s = cleaned_text

    # Progressive deterministic repairs:
    try:
        # A. Strip any remaining markdown code block markers
        s_repaired = re.sub(r"```(?:json)?", "", s).strip()

        # B. Normalize Python literals to JSON
        s_repaired = re.sub(r':\s*None\b', ': null', s_repaired)
        s_repaired = re.sub(r':\s*True\b', ': true', s_repaired)
        s_repaired = re.sub(r':\s*False\b', ': false', s_repaired)

        # C. Strip trailing commas before closing braces/brackets
        s_repaired = re.sub(r',\s*([\}\]])', r'\1', s_repaired)

        # D. Add missing commas between key-value pairs across newlines
        s_repaired = re.sub(r'([0-9"truefalsenull\]\}])\s*\n\s*"([a-zA-Z0-9_]+)"\s*:', r'\1,\n"\2":', s_repaired)

        # E. Replace single quotes on keys and simple string values
        s_repaired = re.sub(r"(?<=[{,\s])'([a-zA-Z0-9_]+)'\s*:", r'"\1":', s_repaired)

        # F. Close truncated curly braces if unclosed due to token cutoff
        open_braces = s_repaired.count('{')
        close_braces = s_repaired.count('}')
        if open_braces > close_braces:
            if s_repaired.count('"') % 2 != 0:
                s_repaired += '"'
            s_repaired += '\n' + '}\n' * (open_braces - close_braces)

        parsed = json.loads(s_repaired, strict=False)
        if isinstance(parsed, dict):
            audit["repair_success"] = True
            return parsed, audit
    except Exception as e:
        audit["error_msg"] = str(e)

    # G. Character-level repair for unescaped double quotes inside free-form text fields (e.g. "reasoning": "using "swiglu"...")
    try:
        for _ in range(6):
            try:
                parsed = json.loads(s_repaired, strict=False)
                if isinstance(parsed, dict):
                    audit["repair_success"] = True
                    return parsed, audit
            except json.JSONDecodeError as jde:
                audit["error_msg"] = str(jde)
                if "delimiter" in str(jde).lower() or "char" in str(jde).lower():
                    pos = jde.pos
                    # Find the nearest preceding double quote before pos
                    prev_quote = s_repaired.rfind('"', 0, pos)
                    if prev_quote > 0:
                        prefix = s_repaired[:prev_quote].rstrip()
                        # If this quote is inside a text string value (not after ':', '{', ',', '[')
                        if not (prefix.endswith(':') or prefix.endswith('{') or prefix.endswith(',') or prefix.endswith('[')):
                            s_repaired = s_repaired[:prev_quote] + "'" + s_repaired[prev_quote+1:]
                            continue
                break
    except Exception as e:
        audit["error_msg"] = str(e)

    # Fallback greedy regex repair
    match = re.search(r"\{[\s\S]*\}", cleaned_text)
    if match:
        try:
            s_cand = re.sub(r',\s*([\}\]])', r'\1', match.group(0))
            s_cand = re.sub(r':\s*None\b', ': null', s_cand)
            parsed = json.loads(s_cand, strict=False)
            if isinstance(parsed, dict):
                audit["repair_success"] = True
                return parsed, audit
        except Exception as e:
            audit["error_msg"] = str(e)

    audit["repair_success"] = False
    return None, audit


def extract_json_payload(text: str) -> Dict[str, Any]:
    """Extractor for JSON objects returned by LLMs; raises JSONDecodeError if parsing fails."""
    parsed, audit = parse_llm_json(text)
    if parsed is not None:
        return parsed
    raise json.JSONDecodeError(audit.get("error_msg") or "Failed to extract valid JSON payload", text, 0)


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


# Module-level cache for local LLM model and tokenizer to prevent reloading weights across iterations/seeds
_CACHED_LOCAL_MODEL = None
_CACHED_LOCAL_TOKENIZER = None
_CACHED_MODEL_ID = None


class LocalHuggingFaceAdapter(BaseProviderAdapter):
    """
    Adapter for locally loaded HuggingFace open-weight models on GPU.
    Supports 4-bit quantization (bitsandbytes) for 7B-14B models on Kaggle Tesla T4.
    Direct GPU tensor generation eliminates pipeline overhead and max_length conflicts.
    Features progressive parsing, controlled deterministic retries, and comprehensive audit tracking.
    """

    def __init__(self, model: Optional[str] = None, quantization: Optional[str] = None):
        global _CACHED_LOCAL_MODEL, _CACHED_LOCAL_TOKENIZER, _CACHED_MODEL_ID
        model_id = model or os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        quant_mode = quantization or os.environ.get("LOCAL_LLM_QUANT", "4bit")
        super().__init__(model=model_id)
        self.model_id = model_id
        self.model = model_id  # Public string identifier for logging, serialization, and experiment registry
        self.name = "local"
        self.quant_mode = quant_mode

        if _CACHED_LOCAL_MODEL is not None and _CACHED_MODEL_ID == model_id:
            print(f"[Adapter:local] Reusing already loaded in-memory model: {model_id}", flush=True)
            self.hf_model = _CACHED_LOCAL_MODEL
            self.tokenizer = _CACHED_LOCAL_TOKENIZER
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[Adapter:local] Loading HuggingFace model '{model_id}' (quantization={self.quant_mode}) ...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True
            }

            # 4-bit NF4 Quantization
            if torch.cuda.is_available() and self.quant_mode == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    print(f"[Adapter:local] Using 4-bit NF4 quantization (bitsandbytes) for efficient VRAM utilization.", flush=True)
                except Exception as bnb_err:
                    print(f"[Adapter:local] bitsandbytes unavailable ({bnb_err}). Falling back to float16.", flush=True)
                    model_kwargs["torch_dtype"] = torch.float16
            elif torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.float32

            hf_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            # Remove conflicting default max_length from generation config
            if hasattr(hf_model, "generation_config") and hf_model.generation_config is not None:
                hf_model.generation_config.max_length = None

            self.hf_model = hf_model
            _CACHED_LOCAL_MODEL = self.hf_model
            _CACHED_LOCAL_TOKENIZER = self.tokenizer
            _CACHED_MODEL_ID = model_id
            print(f"[Adapter:local] Model '{model_id}' loaded successfully on device: {self.hf_model.device}.", flush=True)

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
        max_tokens: int = 1280,  # Generous budget preventing end-of-sequence truncation
        max_retries: int = 4
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], float]:
        import torch
        t0 = time.time()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                base_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                base_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            base_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        current_prompt = base_prompt
        last_generated = ""
        last_audit = {}

        for attempt in range(1, max_retries + 1):
            inputs = self.tokenizer(current_prompt, return_tensors="pt").to(self.hf_model.device)
            input_tok_len = inputs["input_ids"].shape[1]

            # Progressive sampling temperatures across retries:
            # Attempt 1: Standard exploration (temperature = 0.7)
            # Attempt 2: Moderate entropy with explicit syntax correction (temperature = 0.4)
            # Attempt 3: Low entropy focused generation (temperature = 0.1)
            # Attempt 4: Deterministic greedy decoding (temperature = 0.0)
            if attempt == 1:
                gen_temp = temperature
            elif attempt == 2:
                gen_temp = 0.4
            elif attempt == 3:
                gen_temp = 0.1
            else:
                gen_temp = 0.0

            do_sample = (gen_temp > 0.0)

            with torch.inference_mode():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=gen_temp if do_sample else None,
                    top_p=0.9 if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            gen_ids = outputs[0][input_tok_len:]
            last_generated = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_tok_len = len(gen_ids)

            parsed, audit = parse_llm_json(last_generated)
            audit["attempt"] = attempt
            audit["retry_attempted"] = (attempt > 1)
            audit["retry_success"] = (attempt > 1 and parsed is not None)
            audit["prompt_tokens"] = input_tok_len
            audit["completion_tokens"] = gen_tok_len
            audit["total_tokens"] = input_tok_len + gen_tok_len
            audit["latency_seconds"] = round(time.time() - t0, 3)
            audit["prompt"] = current_prompt
            audit["raw_output"] = last_generated
            last_audit = audit
            self._last_usage = audit

            if parsed is not None and isinstance(parsed, dict):
                parsed["_parse_audit"] = audit
                return parsed, last_generated, audit, audit["latency_seconds"]

            # If attempt failed to parse, trigger controlled retry with clean instruction
            if attempt < max_retries:
                err_msg = audit.get("error_msg", "syntax error")
                print(f"[Adapter:local] Attempt {attempt} returned malformed JSON ({err_msg}). Retrying (attempt {attempt+1}/{max_retries}, temp={gen_temp})...", flush=True)

                retry_user_prompt = (
                    f"{user_prompt}\n\n"
                    f"CRITICAL FIX REQUIRED (Attempt {attempt+1}/{max_retries}):\n"
                    f"Your previous output failed JSON parsing ({err_msg}).\n"
                    f"Follow these strict formatting rules:\n"
                    f"1. Inside text fields (hypothesis_text, reasoning), use SINGLE QUOTES 'like this' around architectural terms (e.g. 'swiglu', 'pre_ln'). NEVER use double quotes inside strings.\n"
                    f"2. Separate every key-value line with a comma. Do not put trailing commas before closing braces }} or brackets ]].\n"
                    f"3. Output ONLY the raw JSON object starting with {{ and ending with }}. No markdown blocks, no text before or after."
                )
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_user_prompt}
                ]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    try:
                        current_prompt = self.tokenizer.apply_chat_template(retry_messages, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        current_prompt = base_prompt
                else:
                    current_prompt = base_prompt

        # If all attempts failed, save raw output for post-mortem audit and raise structured error
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_dir = os.path.join(workspace_root, "experiment_results", "failed_llm_proposals")
        os.makedirs(out_dir, exist_ok=True)
        fail_file = os.path.join(out_dir, f"failed_llm_output_{int(time.time())}.txt")
        with open(fail_file, "w", encoding="utf-8") as ff:
            ff.write(f"=== RAW LLM RESPONSE ===\n{last_generated}\n\n=== AUDIT METADATA ===\n{json.dumps(last_audit, indent=2)}\n")
        print(f"[Adapter:local] Fatal: LLM failed to emit valid JSON after {max_retries} attempts. Saved raw output to: {fail_file}", flush=True)
        raise LLMProposalParseError(
            f"Local LLM failed to emit valid JSON after {max_retries} attempts ({last_audit.get('error_msg')}).",
            raw_text=last_generated,
            audit_metadata=last_audit
        )


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
        if os.environ.get("LOCAL_LLM_MODEL"):
            provider = "local"
        elif api_key and str(api_key).startswith("gsk_"):
            provider = "groq"
        elif api_key and str(api_key).startswith("nvapi-"):
            provider = "nvidia"
        elif os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("NVIDIA_API_KEY"):
            provider = "nvidia"
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
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    provider = "local"
                else:
                    provider = "adaptive_simulation"
            except Exception:
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
