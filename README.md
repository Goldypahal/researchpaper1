# The Search Efficiency Gap: Measuring When Semantic LLM Reasoning Actually Improves (or Hinders) Autonomous Model Optimization

## Controlled Empirical Benchmarks, Matched-Compute Protocols, and Autonomous Research Trajectories

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-Empirical%20Research%20Program-blue.svg)]()

This repository contains the experimental infrastructure, mathematically formalized benchmarks, immutable evaluator, and statistical analysis pipeline for **Research Paper 1**.

Rather than asserting unverified claims about fully autonomous discovery, this empirical research program directly investigates:
> **Under what conditions does semantic LLM reasoning actually provide value over established search algorithms (Random Search, TPE Bayesian Optimization, Genetic Algorithms) for autonomous machine learning optimization under equal compute budgets, and why?**

---

## 📌 Repository Architecture

```
RESEARCHPAPER1
│
├── benchmarks/                  # Formal language benchmarks & Chomsky hierarchy levels
│   ├── dyck.py                  # True Dyck-k grammar with stack memory & verified disjoint splits
│   ├── fsm.py                   # Hidden Finite State Machine with algebraic state transitions
│   ├── parity_legacy.py         # 4-state parity baseline preserved for backward compatibility
│   ├── specifications.md        # Mathematical definitions, depth boundaries, & OOD gaps
│   └── tests/test_benchmarks.py # Unit tests asserting sequence validity, depth isolation, & determinism
│
├── evaluator/                   # Immutable Evaluator & Adversarial Anti-Gaming Suite
│   ├── immutable_evaluator.py   # Ground-truth evaluation harness with cryptographic hash isolation
│   ├── adversarial_gaming.py    # 10 adversarial attacks benchmarking evaluator tamper-resistance
│   └── adversarial_gaming_report.md # Formal security report (GamingRate = 0%, DetectionRate = 100%)
│
├── scientific_stats/            # Rigorous Statistical Framework & Efficiency Metrics
│   └── statistical_analysis.py  # Bootstrap CIs, paired tests, Cohen's d_z, Holm-Bonferroni, normalized gain AUC
│
├── search_space.py              # Formal Level 1 (hyperparameters) & Level 2 (structural NAS) search space
│
├── v0.1-current-results/        # [FROZEN ARCHIVE] Immutable historical exploratory run & audit manifest
├── v0.2-benchmark-validation/   # Multi-seed baseline Transformer empirical validation (Dyck & FSM)
├── v0.3-comparative-search/     # Matched-compute comparative search suite (Random vs TPE vs GA vs LLM)
├── v0.4-reasoning-ablation/     # Cognitive reasoning ablations (No-History, Reflection, Proposer-Critic)
├── v0.6-human/                  # Standardized human engineering protocol (interactive empirical subject CLI)
├── v0.7-real-ml/                # Cross-domain real ML small language modeling benchmark
├── reproducibility/             # Master one-command reproduction package (Dockerfile, requirements, seeds)
│   └── reproduce_all.py         # End-to-end automated empirical replication pipeline
│
├── research_questions_and_hypotheses.md # Pre-registered hypotheses H1–H7 & formal research questions
└── EXPERIMENT_REGISTRY.json     # Master audit registry logging every trajectory, commit, seed, & outcome
```

---

## 🔬 Formal Benchmark Tasks

1. **Task A: True Dyck-$k$ Language with Stack Hierarchical Memory ($k=4$)**:
   - Distinct bracket pairs: `()`, `[]`, `{}`, `<>` interspersed with distractor tokens.
   - Guaranteed grammatical validity via push/pop stack discipline.
   - **In-distribution ($D_{\text{train}}, D_{\text{val}}, D_{\text{test}}$)**: Nesting depth $d \in [1, 6]$.
   - **Out-of-distribution ($D_{\text{OOD}}$)**: Nesting depth $d \in [7, 12]$ (evaluates hierarchical generalization).
   - Zero sample leakage confirmed by cryptographic SHA-256 validation ($D_{\text{train}} \cap D_{\text{val}} = \emptyset$).
2. **Task B: Hidden Finite State Machine (Algebraic State Transitions)**:
   - Hidden transition table $T: S \times \Sigma \to S$ ($|S|=8, |\Sigma|=16$).
   - Observation emission table $E: S \times \Sigma \to \Sigma$.
   - **OOD Split**: Doubled state space $|S|=16$ testing temporal state tracking limits.
3. **Task C: Real ML Text Modeling**:
   - Character/Byte-level next-token prediction with in-distribution vs. cross-genre out-of-distribution shifts.

---

## 🔒 The Immutable Evaluator Principle

To prevent autonomous search algorithms from gaming metrics, modifying test sets, or inflating gains:
```
┌──────────────────────┐
│   Search Algorithm   │
│                      │
│ Random / TPE / GA /  │
│     LLM / Human      │
└──────────┬───────────┘
           │ Candidate Configuration θ
           ▼
┌──────────────────────┐
│ IMMUTABLE EVALUATOR  │
│                      │
│  - Data Splits       │  -> Evaluated independently in-memory
│  - Training Loop     │  -> Fixed step & compute budget
│  - Metrics & Gaps    │  -> Cross-entropy & structural bracket accuracy
│  - Hardware Monitor  │  -> Separates training compute vs total search cost
└──────────┬───────────┘
           │ Empirical Result (L_val, L_OOD, Acc, Compute Breakdown)
           ▼
```

- **Adversarial Benchmark**: Validated against 10 distinct exploit vectors (metric spoofing, NaN/Inf injection, dataset swapping, test-data inspection, checkpoint reuse, early stopping, cache poisoning, timeouts, resource manipulation). Evaluator demonstrated **0.00% Gaming Rate** and **100.00% Detection Rate**.

---

## 📊 Pre-Registered Competing Hypotheses

Before observing the final large-sample sweeps, hypotheses are pre-registered in [`research_questions_and_hypotheses.md`](research_questions_and_hypotheses.md):
- **$\mathcal{H}_1$ (Search Superiority)**: $L^*_{\text{val}}(\text{LLM}) < L^*_{\text{val}}(\text{Random})$ under matched compute budgets.
- **$\mathcal{H}_2$ (Value Beyond Sequential Optimizers)**: $L^*_{\text{val}}(\text{LLM}) < \min(L^*_{\text{val}}(\text{TPE}), L^*_{\text{val}}(\text{GA}))$.
- **$\mathcal{H}_3$ (Complexity-Dependent Value)**: The value of semantic reasoning increases as task grammar complexity escalates from regular state-spaces to context-free hierarchical trees.
- **$\mathcal{H}_4$ (Qualitative Verbal Reflection)**: Verbal reflection logs improve search efficiency ($\text{AUC}_{\text{gain}}$) over numeric history alone.
- **$\mathcal{H}_5$ (Adversarial Critic Calibration)**: Proposer-critic loops improve quantitative prediction calibration error.
- **$\mathcal{H}_6$ (OOD Inductive Bias)**: LLM-selected structural architectures generalize better out-of-distribution than stochastic selections.

---

## 🚀 One-Command Full Reproduction

To run the complete automated verification pipeline:
```bash
python reproducibility/reproduce_all.py
```
This executes:
1. Formal benchmark invariant & split isolation unit tests
2. Adversarial specification-gaming security verification
3. Multi-seed baseline Transformer empirical validation
4. Matched-compute comparative search trajectories across Random, TPE, GA, and LLM
5. Cognitive reasoning ablation suite
6. Statistical aggregation with 95% bootstrap CIs and paired significance tests
