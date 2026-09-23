# Can AI Agents Improve AI Without Humans?
## Comprehensive Literature Synthesis, Architectural Taxonomy, and Empirical Self-Improvement Harness

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-Research%20Paper%201-success.svg)]()

This repository contains the codebase, empirical evaluation harness, experimental execution traces, publication figures, and literature synthesis for **Research Paper 1: Can AI Agents Improve AI Without Humans?**.

---

## 📌 Repository Overview

```
├── agent_scaffold/              # Autonomous agent loop, sandbox runner, & baseline implementations
│   ├── agent_loop.py            # Main agent interaction loop
│   ├── eval_harness.py          # Empirical evaluation harness & metrics
│   ├── llm_agent.py             # Agent LLM prompting and step reasoning
│   ├── model_baseline.py        # Baseline model architectures
│   ├── random_search_baseline.py# Stochastic search comparison baseline
│   ├── sandbox_runner.py        # Isolated execution sandbox
│   └── trace_logger.py          # Structured JSON trace recording
├── execution_traces/            # Empirical execution traces across test arms & seeds
│   ├── trace_exp_arm1_*.json    # Iterative LLM self-improvement trials
│   └── trace_exp_arm2_*.json    # Random search baseline comparison runs
├── experiment_results/          # Aggregated statistical benchmarks and reports
│   ├── benchmark_distribution_report.json
│   ├── benchmark_distribution_report.md
│   └── scaleup_n10_checkpoint.json
├── figures/                     # High-resolution publication figures & evaluation plots
│   ├── fig1_loss_distribution_comparison.png
│   ├── fig2_optimization_trajectories.png
│   ├── fig3_parameter_space_coverage.png
│   ├── fig4_prediction_calibration.png
│   └── fig_composite_evaluation.png
├── Complete_Literature_Analysis_Topic_1.docx # Complete systematic literature review
├── citation_audit.csv / .xlsx   # 44 curated literature citations with category & audit tags
├── differentiation_and_protocol.md # Methodological differentiation & validation protocol
├── human_baseline_protocol.md   # Human reference benchmark protocol
├── outline_and_synthesis.md     # Literature synthesis, formal taxonomy, & roadmap
├── section_4_empirical_results.md # Empirical validation & statistical analysis writeup
└── trace_schema.json            # JSON schema for agent execution traces
```

---

## 🔬 Core Research Questions & Taxonomy

We investigate three operational regimes of self-improvement:
1. **Type I: In-Context Ephemeral Refinement ($\Delta \mathcal{M}, \Delta \mathcal{P}$)** — Refinement through prompt-based self-critique without modifying model weights.
2. **Type II: Parametric Self-Training & RLAIF ($\Delta \mathcal{W}$)** — Self-generated rationales, critiques, or preferences updating parameters via reinforcement learning.
3. **Type III: Meta-Architectural & Open-Ended Self-Modification ($\Delta \mathcal{W}, \Delta \mathcal{T}, \Delta \text{Code}$)** — Self-modification of optimization algorithms, code scaffolds, and tool synthesis.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- `matplotlib`, `numpy`, `scipy` (for running benchmarks and plotting)

### Generate Figures
To reproduce the evaluation figures from experimental traces:
```bash
python generate_evaluation_figures.py
```

### Run Benchmarks
To execute the benchmark harness across baseline models:
```bash
python -m agent_scaffold.run_distribution_benchmark
```

---

## 📊 Citation & Reference
For detailed analysis, refer to [outline_and_synthesis.md](outline_and_synthesis.md) and [section_4_empirical_results.md](section_4_empirical_results.md).
