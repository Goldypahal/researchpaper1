"""
Formal Model Search Space Specification (Phase 3).
Unifies Level 1 (Hyperparameters) and Level 2 (Structural Architecture Space).
Guarantees identical bounds and valid combinations across all search arms:
  - Arm 1: LLM Agent (with Ablations)
  - Arm 2: Random Search
  - Arm 3: Bayesian Optimization (TPE)
  - Arm 4: Evolutionary Algorithm (GA)
  - Arm 5: Human Baseline
"""

import math
import random
from typing import Dict, Any, Optional

SEARCH_SPACE_SPEC = {
    # Level 1: Continuous & Discrete Hyperparameters
    "lr": {"type": "float", "low": 1e-5, "high": 5e-2, "log": True, "default": 1e-3},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.2, "log": False, "default": 0.01},
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
    "n_kv_heads": {"type": "categorical", "choices": [1, 2, "same"], "default": "same"}
}


def sanitize_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes and enforces physical architectural invariants:
      1. d_model must be divisible by n_heads
      2. n_kv_heads must divide n_heads
      3. bounded parameter types
    """
    clean = dict(config)
    
    # Ensure d_model divisible by n_heads
    d_model = clean.get("d_model", 256)
    n_heads = clean.get("n_heads", 4)
    if d_model % n_heads != 0:
        clean["d_model"] = (d_model // n_heads) * n_heads
        if clean["d_model"] == 0:
            clean["d_model"] = n_heads * 32

    # Enforce n_kv_heads constraint for GQA
    n_kv = clean.get("n_kv_heads", "same")
    if n_kv == "same" or n_kv is None:
        clean["n_kv_heads"] = None
    elif isinstance(n_kv, int):
        if clean["n_heads"] % n_kv != 0:
            clean["n_kv_heads"] = 1  # Fallback to MQA if not divisible

    return clean


def sample_random_candidate(rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Uniform / log-uniform random sampling from the search space."""
    r = rng if rng is not None else random
    cand = {}
    
    # lr (log-uniform)
    log_min = math.log10(SEARCH_SPACE_SPEC["lr"]["low"])
    log_max = math.log10(SEARCH_SPACE_SPEC["lr"]["high"])
    cand["lr"] = round(10 ** r.uniform(log_min, log_max), 6)
    
    # weight_decay (uniform)
    cand["weight_decay"] = round(r.uniform(SEARCH_SPACE_SPEC["weight_decay"]["low"], SEARCH_SPACE_SPEC["weight_decay"]["high"]), 4)
    
    # Categoricals
    for param in ["d_model", "n_layers", "n_heads", "d_ff", "activation", "norm_type",
                  "scale_factor", "pos_encoding", "ffn_type", "topology", "n_kv_heads"]:
        cand[param] = r.choice(SEARCH_SPACE_SPEC[param]["choices"])
        
    return sanitize_configuration(cand)


def get_default_baseline() -> Dict[str, Any]:
    """Returns the standardized step-0 baseline architecture."""
    base = {k: v["default"] for k, v in SEARCH_SPACE_SPEC.items()}
    return sanitize_configuration(base)
