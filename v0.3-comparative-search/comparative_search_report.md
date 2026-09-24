# Phase 8: Matched-Compute Comparative Search Benchmark Report

**Core Research Question**: *When does semantic LLM reasoning provide marginal value over established search algorithms (Random, TPE, GA) under equal compute budgets, and why?*

**Search Horizon**: $K=10$ candidate evaluations per trajectory.
**Seeds Evaluated**: `[42, 101, 202]` ($N=3$ replications per arm).
**Budget Constraint**: All candidates evaluated by the Immutable Evaluator under identical gradient step budgets.

---

## Benchmark: DYCK

### 1. Performance Summary Across Search Arms
| Search Arm | $L^*_{\text{val}}$ Mean [95% CI] | $L^*_{\text{OOD}}$ Mean [95% CI] | Gain (%) Mean | Normalized Gain AUC | Eval Wall-Clock (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RANDOM** | 4.2465 [4.2189, 4.2742] | 4.2447 [4.2161, 4.2739] | +0.15% | 0.0000 | 0.0s |

### 2. Hypothesis Testing & Effect Sizes
| Comparison | Mean Difference | Cohen's $d$ | Hedges' $g$ | Welch's $t$ $p$-value | Mann-Whitney $U$ $p$-value | Permutation $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

---

## 3. Scientific Takeaways
- **Search Paradigm Comparison**: Compares uniform stochastic exploration (Random), surrogate model optimization (TPE), population-based genetic selection (Evolutionary), and semantic hypothesis generation (LLM).
- **Task Complexity Scaling**: Evaluates whether the inductive biases discoverable by each search paradigm differ across Chomsky regular state tracking (Hidden FSM) versus context-free hierarchical grammar (Dyck-4).
- **Matched Evaluation Budget**: All candidate configurations are strictly bounded by identical gradient update horizons in the Immutable Evaluator, isolating search algorithm efficacy from training budget disparities.