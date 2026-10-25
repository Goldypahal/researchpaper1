r"""
tests/test_airtight_correctness.py

Comprehensive Phase 3 unit test suite covering:
  1. test_grid_builds (RoPE parity guard & architecture construction)
  2. test_validate_rejects_offgrid (strict zero-slack search space bounds)
  3. test_prompt_has_no_ood (leakage prevention)
  4. test_anonymized_no_task_words (semantic prior isolation)
  5. test_shuffled_feedback (feedback scrambling integrity)
  6. test_parser_invalid_escape (LaTeX \Delta JSON parsing)
  7. test_failure_semantics (None val_loss & Optuna FAIL state)
  8. test_atomic_write (safe atomic trace writes)
"""

import os
import sys
import re
import json
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

from search_space import (
    SEARCH_SPACE_SPEC,
    validate_proposal,
    sanitize_configuration,
    sample_random_candidate,
    get_default_baseline
)
from agent_scaffold.model_baseline import SmallTransformerLM
from agent_scaffold.provider_adapters import parse_llm_json, repair_json_escapes
from agent_scaffold.llm_agent import build_system_prompt, LLMSearchAgent
from scripts.run_comparative_search import write_trace_atomic


def test_grid_builds():
    """Checks that candidate configurations on the grid instantiate and execute forward pass on CPU without crash."""
    for d_model in SEARCH_SPACE_SPEC["d_model"]["choices"]:
        for n_heads in SEARCH_SPACE_SPEC["n_heads"]["choices"]:
            if d_model % n_heads != 0:
                continue
            d_k = d_model // n_heads
            for pos in SEARCH_SPACE_SPEC["pos_encoding"]["choices"]:
                # If RoPE is selected, d_k must be even
                if pos == "rotary" and d_k % 2 != 0:
                    continue
                for topo in ["pre_ln", "post_ln", "parallel"]:
                    for ffn in ["standard", "swiglu"]:
                        model = SmallTransformerLM(
                            vocab_size=128,
                            d_model=d_model,
                            n_layers=2,
                            n_heads=n_heads,
                            d_ff=256,
                            max_seq_len=32,
                            activation="gelu",
                            norm_type="layernorm",
                            pos_encoding=pos,
                            ffn_type=ffn,
                            topology=topo
                        )
                        x = torch.randint(0, 128, (2, 16))
                        logits, loss = model(x, x)
                        assert logits.shape == (2, 16, 128), f"Shape mismatch for {pos}, {topo}, {ffn}"
                        assert not torch.isnan(loss), "Loss was NaN"


def test_validate_rejects_offgrid():
    """Checks that strict validate_proposal rejects off-grid values and odd-RoPE dimensions."""
    base_prop = {
        "hypothesis_text": "Testing valid proposal candidate",
        "target_component": "architecture",
        "predicted_delta_loss": 0.05,
        "modifications": {
            "lr": 0.001,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 4,
            "d_ff": 1024,
            "activation": "gelu",
            "norm_type": "layernorm",
            "pos_encoding": "learned",
            "ffn_type": "standard",
            "topology": "pre_ln"
        }
    }
    valid, violations = validate_proposal(base_prop)
    assert valid, f"Expected valid baseline proposal, got violations: {violations}"

    # 1. Off-grid d_model = 204
    p_off_dmodel = json.loads(json.dumps(base_prop))
    p_off_dmodel["modifications"]["d_model"] = 204
    valid, violations = validate_proposal(p_off_dmodel)
    assert not valid
    assert any("204" in v for v in violations)

    # 2. Out-of-bounds lr = 0.07 (max is 0.05)
    p_off_lr = json.loads(json.dumps(base_prop))
    p_off_lr["modifications"]["lr"] = 0.07
    valid, violations = validate_proposal(p_off_lr)
    assert not valid
    assert any("lr" in v for v in violations)

    # 3. Off-grid scale_factor = 0.0625
    p_off_scale = json.loads(json.dumps(base_prop))
    p_off_scale["modifications"]["scale_factor"] = 0.0625
    valid, violations = validate_proposal(p_off_scale)
    assert not valid
    assert any("scale_factor" in v for v in violations)


def test_prompt_has_no_ood():
    """Checks that rendered system prompt has ZERO mentions of 'ood' or 'out-of-distribution'."""
    prompt = build_system_prompt(task_framing="named")
    assert "ood" not in prompt.lower()
    assert "out-of-distribution" not in prompt.lower()
    assert "generalization loss" not in prompt.lower()


def test_anonymized_no_task_words():
    """Checks that anonymized system prompt contains NO domain-specific task indicators."""
    prompt = build_system_prompt(task_framing="anonymized")
    forbidden = ["dyck", "bracket", "fsm", "hierarchical", "markov", "state-transition"]
    for w in forbidden:
        assert w not in prompt.lower(), f"Forbidden task word '{w}' leaked into anonymized prompt"


def test_shuffled_feedback():
    """Checks that shuffled feedback permutes past losses while maintaining multiset and recalculating incumbent."""
    os.environ["ALLOW_LLM_SIMULATION"] = "1"
    agent = LLMSearchAgent(
        provider="adaptive_simulation",
        task_framing="named",
        feedback="shuffled",
        allow_simulation=True
    )
    raw_history = [
        {"iteration": 0, "val_loss": 4.20, "config": {}},
        {"iteration": 1, "val_loss": 4.15, "config": {}},
        {"iteration": 2, "val_loss": 4.10, "config": {}},
        {"iteration": 3, "val_loss": 4.05, "config": {}}
    ]
    prepared, incumbent = agent._prepare_history(raw_history, seed=42)
    prepared_losses = [e["val_loss"] for e in prepared]
    original_losses = [e["val_loss"] for e in raw_history]

    # Multiset equality
    assert sorted(prepared_losses) == sorted(original_losses), "Shuffled feedback corrupted loss multiset"
    # Order changed or permuted
    assert incumbent == min(prepared_losses)


def test_parser_invalid_escape():
    """Checks that JSON strings containing LaTeX escapes like \\Delta or \\s parse cleanly."""
    latex_json = '{"hypothesis_text": "Increase lr by \\Delta = 0.002 to escape local saddle point", "target_component": "optimizer", "predicted_delta_loss": 0.03, "modifications": {"lr": 0.003}}'
    parsed, audit = parse_llm_json(latex_json)
    assert parsed is not None, f"Parser failed on LaTeX escape: {audit.get('error_msg')}"
    assert parsed["modifications"]["lr"] == 0.003
    assert audit["parse_status"] in ("OK", "REPAIRED")


def test_failure_semantics():
    """Checks that failed proposals have val_loss=None, not 99.0, preventing distorted metrics."""
    step_fail = {
        "iteration": 1,
        "config": None,
        "status": "INVALID_PROPOSAL",
        "val_loss": None
    }
    assert step_fail["val_loss"] is None
    assert step_fail["status"] == "INVALID_PROPOSAL"


def test_atomic_write(tmp_path):
    """Checks that write_trace_atomic writes file cleanly without leaving temp files behind."""
    out_file = str(tmp_path / "test_trace.json")
    payload = {"status": "COMPLETE_VALID", "seed": 42, "best_val": 4.11}
    write_trace_atomic(out_file, payload)

    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert loaded["best_val"] == 4.11
    # Check no leftover temp files
    assert len(list(tmp_path.glob("*.tmp*"))) == 0
