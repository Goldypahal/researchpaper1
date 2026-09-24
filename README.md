# The Search Efficiency Gap: Measuring When LLM Reasoning Improves (or Hinders) Autonomous Model Optimization
## Controlled Empirical Search Benchmarks, Hierarchical Task Families, and Agent Execution Traces

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-Research%20Paper%201-success.svg)]()

This repository contains the codebase, empirical evaluation harness, execution trace graphs, publication figures, and literature synthesis for **Research Paper 1**.

Rather than making unsupported macro-claims about fully autonomous scientific discovery, this work focuses on a concrete, rigorously measurable question:
> **When autonomous LLM-driven optimization is constrained to a fixed experimental budget, does semantic reasoning actually provide useful search efficiency over non-semantic search, and under what conditions does it fail?**

---

## 📌 Repository Overview

```
├── agent_scaffold/              # Autonomous agent loop, sandbox runner, & baseline implementations
│   ├── agent_loop.py            # Main agent interaction loop with task & reasoning ablation dispatcher
│   ├── eval_harness.py          # Deterministic evaluation harness (Dyck-k, Hidden FSM, Parity)
│   ├── llm_agent.py             # Agent LLM reasoning engine (Proposer + Critic + Ablation modes)
│   ├── model_baseline.py        # Baseline SmallTransformerLM architecture
│   ├── random_search_baseline.py# Baseline B: Uniform stochastic search
│   ├── bayesian_opt_baseline.py # Baseline C: Bayesian Optimization via Optuna TPE
│   ├── evolutionary_baseline.py # Baseline D: Genetic Algorithm (Tournament, Crossover, Mutation)
│   ├── run_comparative_benchmark.py # Multi-baseline comparative benchmark orchestrator
│   ├── sandbox_runner.py        # Isolated execution sandbox
│   └── trace_logger.py          # Dynamic baseline tracking & JSON trace graph generation
├── execution_traces/            # Current empirical execution traces
│   ├── trace_exp_arm1_*.json    # Iterative LLM self-improvement trials
│   └── trace_exp_arm2_*.json    # Random search baseline comparison runs
├── experiment_results/          # Statistical distribution reports and scaleup checkpoints
├── figures/                     # High-resolution publication figures & evaluation plots
├── v0.1-current-results/        # [FROZEN ARCHIVE] Immutable snapshot of initial N=10 v0.1 exploration
│   ├── ARCHIVE_MANIFEST.md      # Detailed audit findings, corrected baseline gains, & audit notes
│   ├── execution_traces/        # Original 20 JSON traces
│   ├── experiment_results/      # Original statistical reports
│   └── figures/                 # Original 5 evaluation plots
├── Complete_Literature_Analysis_Topic_1.docx # Systematic literature review
├── citation_audit.csv / .xlsx   # 44 curated literature citations with category & audit tags
├── differentiation_and_protocol.md # Methodological differentiation & validation protocol
├── human_baseline_protocol.md   # Human reference benchmark protocol (3-5 ML practitioners)
├── outline_and_synthesis.md     # Literature synthesis, formal taxonomy, & roadmap
├── section_4_empirical_results.md # Empirical validation & statistical analysis writeup
└── trace_schema.json            # Graph schema for agent execution traces (Nodes & Edges)
```

---

## 🔬 Benchmark Task Families in `eval_harness.py`

1. **Task A: True Dyck-$k$ Language with Stack Hierarchical Memory ($k=4$)**:
   - Explicit bracket pairs: `()`, `[]`, `{}`, `<>` interspersed with distractor tokens.
   - Guaranteed grammatical validity via recursive stack tracking.
   - **In-distribution split**: Nesting depth $d \in [1, 6]$.
   - **Out-of-distribution (OOD) split**: Nesting depth $d \in [7, 12]$ (tests whether the architecture learned true hierarchical stack memory or local $n$-gram shortcuts).
   - Evaluates both cross-entropy loss and exact next-token structural bracket prediction accuracy.
2. **Task B: Hidden Finite State Machine (Algebraic State Transitions)**:
   - Hidden transition table $T: S \times \Sigma \to S$ ($|S|=8, |\Sigma|=16$).
   - Emission table $E: S \times \Sigma \to \Sigma$.
   - Requires temporal state tracking across token sequences.
3. **Task C: Parity Legacy**:
   - 4-state parity baseline preserved for exact backwards comparability.

---

## 🚀 Quickstart

### Run Dyck-$k$ Evaluation
```bash
# In-distribution training & evaluation with OOD generalization test
python agent_scaffold/eval_harness.py --task dyck --dry-run
```

### Run Search Baselines
```bash
# Baseline B: Random Search
python agent_scaffold/random_search_baseline.py --task dyck --iterations 10 --seed 42

# Baseline C: Bayesian Optimization (Optuna TPE)
python agent_scaffold/bayesian_opt_baseline.py --task dyck --iterations 10 --seed 42

# Baseline D: Evolutionary Search (Genetic Algorithm)
python agent_scaffold/evolutionary_baseline.py --task dyck --iterations 10 --seed 42
```

### Run Unified Comparative Search Benchmark
```bash
python agent_scaffold/run_comparative_benchmark.py --task dyck --iterations 10 --seeds 42 101 777 --methods random tpe evolutionary
```

### Run Autonomous LLM Agent & Reasoning Ablations
```bash
# Full reasoning with execution history & reflection
python agent_scaffold/agent_loop.py --task dyck --iterations 10 --provider anthropic --model claude-3-5-sonnet-20241022 --reasoning-mode full

# Ablation 1: Zero-shot proposal (no history)
python agent_scaffold/agent_loop.py --task dyck --iterations 10 --reasoning-mode no_history

# Ablation 2: Number-only history without verbal reflection
python agent_scaffold/agent_loop.py --task dyck --iterations 10 --reasoning-mode history_no_reflection

# Ablation 3: Proposer-Critic refinement loop
python agent_scaffold/agent_loop.py --task dyck --iterations 10 --reasoning-mode critic_refine
```

---

## 📊 Core Competing Hypotheses

- **$H_1$ (Semantic Advantage)**: Agent search efficiency strictly dominates stochastic search under equal compute.
- **$H_2$ (Semantic Parity)**: Agent search matches random search with no significant efficiency advantage.
- **$H_3$ (Semantic Deficit)**: Agent inductive biases constrain exploration to suboptimal local modes, underperforming random search.
- **$H_4$ (Budget-Dependent Crossover)**: Search dominance flips as a function of the experimental budget $K$ (e.g. LLM advantages in low-budget regimes vs. Bayesian/evolutionary dominance in large-budget regimes).
