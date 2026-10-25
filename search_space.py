"""
Formal Model Search Space Specification.
Single Source of Truth across all search arms:
  - LLM Agent (with prompt generation and schema alignment)
  - Random Search
  - Bayesian Optimization (Optuna TPE)
  - Evolutionary Algorithm (GA)
"""

import math
import random
from typing import Dict, Any, Optional, Tuple, List

SEARCH_SPACE_SPEC: Dict[str, Dict[str, Any]] = {
    # Level 1: Continuous Hyperparameters
    "lr": {"type": "float", "low": 1e-5, "high": 5e-2, "log": True, "default": 1e-3},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.2, "log": False, "default": 0.01},

    # Level 1: Discrete Architecture Hyperparameters
    "d_model": {"type": "categorical", "choices": [128, 256, 512], "default": 256},
    "n_layers": {"type": "categorical", "choices": [2, 4, 6, 8], "default": 6},
    "n_heads": {"type": "categorical", "choices": [2, 4, 8], "default": 4},
    "d_ff": {"type": "categorical", "choices": [256, 512, 1024, 2048], "default": 1024},
    "activation": {"type": "categorical", "choices": ["gelu", "relu", "silu"], "default": "gelu"},
    "norm_type": {"type": "categorical", "choices": ["layernorm", "rmsnorm"], "default": "layernorm"},
    "scale_factor": {"type": "categorical", "choices": [None, 0.05, 0.1, 0.15, 0.25], "default": None},

    # Level 2: Structural Architecture Topologies
    "pos_encoding": {"type": "categorical", "choices": ["learned", "sinusoidal", "rotary", "none"], "default": "learned"},
    "ffn_type": {"type": "categorical", "choices": ["standard", "swiglu"], "default": "standard"},
    "topology": {"type": "categorical", "choices": ["pre_ln", "post_ln", "parallel"], "default": "pre_ln"},
    "n_kv_heads": {"type": "categorical", "choices": [None, 1, 2], "default": None}
}


def render_search_space_for_prompt(spec: Dict[str, Any] = SEARCH_SPACE_SPEC) -> str:
    """Renders the exact search space description for inclusion in LLM prompts."""
    lines = []
    for i, (k, v) in enumerate(spec.items(), 1):
        if v["type"] == "float":
            scale_note = " (log scale)" if v.get("log") else ""
            lines.append(f"{i}. {k}: float in [{v['low']}, {v['high']}]{scale_note}")
        else:
            choices_repr = [("null" if c is None else repr(c)) for c in v["choices"]]
            lines.append(f"{i}. {k}: one of {{{', '.join(choices_repr)}}}")
    return "\n".join(lines)


def sanitize_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes and enforces physical architectural invariants:
      1. d_model must be divisible by n_heads
      2. n_kv_heads must divide n_heads
      3. For rotary (RoPE), head dimension d_k = d_model // n_heads must be even
    """
    clean = dict(config)
    d_model = clean.get("d_model", 256)
    n_heads = clean.get("n_heads", 4)

    # Ensure d_model divisible by n_heads
    if d_model % n_heads != 0:
        clean["d_model"] = max(n_heads * 32, (d_model // n_heads) * n_heads)

    # Enforce n_kv_heads constraint for GQA
    n_kv = clean.get("n_kv_heads")
    if n_kv == "same" or n_kv is None:
        clean["n_kv_heads"] = None
    elif isinstance(n_kv, int):
        if clean["n_heads"] % n_kv != 0:
            clean["n_kv_heads"] = None

    # Enforce RoPE even dimension constraint
    if clean.get("pos_encoding") == "rotary":
        d_k = clean["d_model"] // clean["n_heads"]
        if d_k % 2 != 0:
            # Snap to an even head dimension if physically impossible
            clean["pos_encoding"] = "sinusoidal"

    return clean


def sample_random_candidate(rng: Optional[random.Random] = None, spec: Dict[str, Any] = SEARCH_SPACE_SPEC) -> Dict[str, Any]:
    """Uniform / log-uniform random sampling strictly from the search space specification."""
    r = rng if rng is not None else random
    cand = {}

    for k, v in spec.items():
        if v["type"] == "float":
            if v.get("log"):
                log_min = math.log10(v["low"])
                log_max = math.log10(v["high"])
                cand[k] = round(10 ** r.uniform(log_min, log_max), 6)
            else:
                cand[k] = round(r.uniform(v["low"], v["high"]), 4)
        elif v["type"] == "categorical":
            cand[k] = r.choice(v["choices"])

    return sanitize_configuration(cand)


def sample_from_optuna(trial: Any, spec: Dict[str, Any] = SEARCH_SPACE_SPEC) -> Dict[str, Any]:
    """Dynamic Optuna suggestion directly derived from the single source of truth spec."""
    cand = {}
    for k, v in spec.items():
        if v["type"] == "float":
            cand[k] = trial.suggest_float(k, v["low"], v["high"], log=v.get("log", False))
        elif v["type"] == "categorical":
            # Optuna suggest_categorical handles strings, ints, floats, None
            cand[k] = trial.suggest_categorical(k, v["choices"])
    return sanitize_configuration(cand)


def get_default_baseline(spec: Dict[str, Any] = SEARCH_SPACE_SPEC) -> Dict[str, Any]:
    """Returns the standardized step-0 baseline architecture."""
    base = {k: v["default"] for k, v in spec.items()}
    return sanitize_configuration(base)


def validate_proposal(proposal: Dict[str, Any], spec: Dict[str, Any] = SEARCH_SPACE_SPEC) -> Tuple[bool, List[str]]:
    """
    Strict validation of an LLM proposal against schema and search space constraints.
    Zero-slack enforcement:
      - All float parameters must fall strictly within [low, high]
      - All categorical parameters must be in choices
      - Structural consistency: d_model divisible by n_heads, n_kv_heads divides n_heads,
        and RoPE requires even head dimension d_k.
    Returns:
        (is_valid: bool, violations: List[str])
    """
    violations = []
    if not isinstance(proposal, dict):
        return False, ["Proposal is not a dictionary"]

    # Top-level schema checks
    hypo = proposal.get("hypothesis") or proposal.get("hypothesis_text")
    if not hypo or not isinstance(hypo, str) or len(hypo.strip()) < 5:
        violations.append("Missing or trivial hypothesis text (min 5 characters required)")

    if "target_component" not in proposal:
        violations.append("Missing 'target_component'")

    pred_delta = proposal.get("predicted_delta_loss")
    if pred_delta is None or not isinstance(pred_delta, (int, float)):
        violations.append("Missing or non-numeric 'predicted_delta_loss'")

    mods = proposal.get("modifications")
    if not isinstance(mods, dict) or len(mods) == 0:
        violations.append("Missing or empty 'modifications' dictionary")
        return False, violations

    # Strict parameter-level checks
    for param_name, param_val in mods.items():
        if param_name not in spec:
            violations.append(f"Unknown parameter '{param_name}' not in search space specification")
            continue

        param_spec = spec[param_name]
        p_type = param_spec["type"]

        if p_type == "float":
            if not isinstance(param_val, (int, float)):
                violations.append(f"Parameter '{param_name}' must be float, got {type(param_val).__name__}")
            elif not (param_spec["low"] <= float(param_val) <= param_spec["high"]):
                violations.append(
                    f"Parameter '{param_name}'={param_val} strictly out of bounds [{param_spec['low']}, {param_spec['high']}]"
                )
        elif p_type == "categorical":
            allowed = param_spec["choices"]
            # Treat string 'null' or None uniformly
            matched = False
            for ch in allowed:
                if ch is None and (param_val is None or param_val == "null" or param_val == "None"):
                    matched = True
                    break
                elif ch == param_val:
                    matched = True
                    break
            if not matched:
                violations.append(
                    f"Parameter '{param_name}'={param_val} not in allowed choices {allowed}"
                )

    # Cross-parameter structural consistency
    d_model = mods.get("d_model", spec["d_model"]["default"])
    n_heads = mods.get("n_heads", spec["n_heads"]["default"])
    if isinstance(d_model, int) and isinstance(n_heads, int) and n_heads > 0:
        if d_model % n_heads != 0:
            violations.append(f"Structural violation: d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        d_k = d_model // n_heads
        pos_enc = mods.get("pos_encoding", spec["pos_encoding"]["default"])
        if pos_enc == "rotary" and d_k % 2 != 0:
            violations.append(f"RoPE violation: Rotary embedding requires an even head dimension d_k, but d_model={d_model} // n_heads={n_heads} = {d_k} is odd")

    n_kv = mods.get("n_kv_heads")
    if isinstance(n_kv, int) and isinstance(n_heads, int) and n_kv > 0:
        if n_heads % n_kv != 0:
            violations.append(f"Structural violation: n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv})")

    return (len(violations) == 0, violations)


# Obfuscation mapping for Wrong-Prior ablation (Phase 2.6a)
OBFUSCATION_PARAM_MAP = {
    "lr": "param_alpha",
    "weight_decay": "param_beta",
    "d_model": "param_gamma",
    "n_layers": "param_delta",
    "n_heads": "param_epsilon",
    "d_ff": "param_zeta",
    "activation": "param_eta",
    "norm_type": "param_theta",
    "scale_factor": "param_iota",
    "pos_encoding": "param_kappa",
    "ffn_type": "param_lambda",
    "topology": "param_mu",
    "n_kv_heads": "param_nu"
}
REVERSE_OBFUSCATION_PARAM_MAP = {v: k for k, v in OBFUSCATION_PARAM_MAP.items()}

OBFUSCATION_CHOICES_MAP = {
    "activation": {"gelu": "opt_A1", "relu": "opt_A2", "silu": "opt_A3"},
    "norm_type": {"layernorm": "opt_B1", "rmsnorm": "opt_B2"},
    "pos_encoding": {"learned": "opt_C1", "sinusoidal": "opt_C2", "rotary": "opt_C3", "none": "opt_C4"},
    "ffn_type": {"standard": "opt_D1", "swiglu": "opt_D2"},
    "topology": {"pre_ln": "opt_E1", "post_ln": "opt_E2", "parallel": "opt_E3"}
}

def get_obfuscated_spec() -> Dict[str, Dict[str, Any]]:
    """Generates an obfuscated search space specification where semantic names are stripped."""
    obf_spec = {}
    for orig_k, orig_v in SEARCH_SPACE_SPEC.items():
        new_k = OBFUSCATION_PARAM_MAP.get(orig_k, orig_k)
        new_v = dict(orig_v)
        if orig_k in OBFUSCATION_CHOICES_MAP:
            choice_map = OBFUSCATION_CHOICES_MAP[orig_k]
            new_v["choices"] = [choice_map.get(c, c) for c in orig_v["choices"]]
            new_v["default"] = choice_map.get(orig_v["default"], orig_v["default"])
        obf_spec[new_k] = new_v
    return obf_spec

def deobfuscate_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Translates an obfuscated proposal back into real parameter names and values."""
    if not isinstance(proposal, dict) or "modifications" not in proposal:
        return proposal
    deobf_proposal = dict(proposal)
    clean_mods = {}
    for k, v in proposal["modifications"].items():
        real_k = REVERSE_OBFUSCATION_PARAM_MAP.get(k, k)
        if real_k in OBFUSCATION_CHOICES_MAP:
            rev_choice_map = {v_opt: k_opt for k_opt, v_opt in OBFUSCATION_CHOICES_MAP[real_k].items()}
            real_v = rev_choice_map.get(v, v)
        else:
            real_v = v
        clean_mods[real_k] = real_v
    deobf_proposal["modifications"] = clean_mods
    return deobf_proposal
