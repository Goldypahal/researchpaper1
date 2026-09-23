# Frozen Historical Archive: v0.1 Experimental Benchmark

**Archive Timestamp**: 2026-09-23T23:11:00Z  
**Status**: PERMANENTLY FROZEN (Do Not Modify)

---

## 1. Purpose of This Archive

This directory permanently preserves the experimental traces, statistical results, figures, and manuscripts from the original v0.1 exploration ($N=10$ per arm, 3-iteration horizon) prior to the Phase 1 scientific corrections.

## 2. Core Audit Findings & Retraction of Inflated Metrics

A formal scientific audit revealed two fundamental issues in the original v0.1 reporting:

1. **Inflated Baseline Reference in `trace_logger.py`**:
   - Stated Baseline Loss: $L_{\text{base}} = 7.0653$
   - Calculation in Code: hardcoded $7.1105$ in `trace_logger.py` Line 176
   - **Corrected Values**:
     - Agent Best ($6.9578$): True gain is **$1.52\%$**, not the reported $2.15\%$.
     - Random Best ($6.9297$): True gain is **$1.92\%$**, not the reported $2.54\%$.
   - **Scientific Implication**: Random search outperformed the LLM agent under this budget and horizon. Rather than a flaw, this negative result forms the core basis of the revised paper: *"The Search Efficiency Gap: Measuring When LLM Reasoning Provides Value in Autonomous Model Optimization"*.

2. **Synthetic Task Specification**:
   - The docstring claimed a *"deterministic algebraic Dyck / state-transition token stream with bracket matching"*, but the implementation in `eval_harness.py` was a 4-state parity check with a modulo-17 shift.
   - Phase 1 replaces this with a true hierarchical Dyck-$k$ language task (with balanced parentheses, stack tracking, variable depth, and out-of-distribution depth splits) and a formal hidden finite state machine.

## 3. Preserved Artifacts

- `execution_traces/`: 20 raw JSON execution traces ($10 \times \text{Arm 1}$ and $10 \times \text{Arm 2}$).
- `experiment_results/`: Distribution reports and scaleup checkpoints.
- `figures/`: Original 5 publication figures.
- `generate_evaluation_figures.py`: Original visualization script.
- `section_4_empirical_results.md`: Original v0.1 empirical writeup.
