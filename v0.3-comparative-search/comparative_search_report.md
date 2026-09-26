# Phase 8: Matched-Compute Comparative Search Benchmark Report

**Core Research Question**: *When does semantic LLM reasoning provide marginal value over established search algorithms (Random, TPE, GA) under equal compute budgets, and why?*

**Search Horizon**: $K=10$ candidate evaluations per trajectory.
**Seeds Evaluated**: `[42, 101, 202, 303, 404]` ($N=5$ replications per arm).
**Budget Constraint**: All candidates evaluated by the Immutable Evaluator under identical gradient step budgets.

---

## Benchmark: DYCK

### 1. Performance Summary Across Search Arms
| Search Arm | $L^*_{\text{val}}$ Mean [95% CI] | $L^*_{\text{OOD}}$ Mean [95% CI] | Gain (%) Mean | Normalized Gain AUC | Cand Train GPU (s) | LLM Infer (s) | Total Wall-Clock (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RANDOM** | 4.1681 [4.1552, 4.1807] | 4.1497 [4.1238, 4.1784] | +0.67% | 0.0054 | 36.1s | 0.0s | 36.1s |
| **TPE** | 4.1567 [4.1311, 4.1957] | 4.1445 [4.1069, 4.1798] | +0.94% | 0.0068 | 42.4s | 0.0s | 42.4s |
| **EVOLUTIONARY** | 4.1366 [4.0998, 4.1705] | 4.1216 [4.0813, 4.1670] | +1.42% | 0.0075 | 39.9s | 0.0s | 39.9s |
| **LLM** | 4.0986 [4.0667, 4.1302] | 4.0950 [4.0817, 4.1089] | +2.33% | 0.0218 | 127.2s | 127.2s | 127.2s |

### 2. Hypothesis Testing & Effect Sizes
| Comparison | Mean Difference | Cohen's $d$ | Hedges' $g$ | Welch's $t$ $p$-value | Mann-Whitney $U$ $p$-value | Permutation $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| tpe_vs_random | -0.0114 | -0.34 | -0.31 | 0.6140 | 0.2492 | 0.6507 |
| evolutionary_vs_random | -0.0315 | -0.93 | -0.84 | 0.2025 | 0.2492 | 0.2136 |
| llm_vs_random | -0.0695 | -2.18 | -1.97 | 0.0173 | 0.0159 | 0.0112 |
| llm_vs_tpe | -0.0581 | -1.34 | -1.21 | 0.0668 | 0.1508 | 0.0778 |
| llm_vs_evolutionary | -0.0380 | -0.87 | -0.79 | 0.2067 | 0.3095 | 0.2074 |

---

## Benchmark: FSM

### 1. Performance Summary Across Search Arms
| Search Arm | $L^*_{\text{val}}$ Mean [95% CI] | $L^*_{\text{OOD}}$ Mean [95% CI] | Gain (%) Mean | Normalized Gain AUC | Cand Train GPU (s) | LLM Infer (s) | Total Wall-Clock (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RANDOM** | 2.6574 [2.6347, 2.6859] | 2.9178 [2.8869, 2.9471] | +0.24% | 0.0015 | 37.0s | 0.0s | 37.0s |
| **TPE** | 2.6567 [2.6360, 2.6823] | 2.9047 [2.8741, 2.9291] | +0.26% | 0.0017 | 42.4s | 0.0s | 42.4s |
| **EVOLUTIONARY** | 2.6544 [2.6312, 2.6802] | 2.9120 [2.8745, 2.9345] | +0.35% | 0.0017 | 42.8s | 0.0s | 42.8s |
| **LLM** | 2.6531 [2.6294, 2.6814] | 2.9200 [2.8803, 2.9496] | +0.40% | 0.0034 | 87.8s | 87.8s | 87.8s |

### 2. Hypothesis Testing & Effect Sizes
| Comparison | Mean Difference | Cohen's $d$ | Hedges' $g$ | Welch's $t$ $p$-value | Mann-Whitney $U$ $p$-value | Permutation $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| tpe_vs_random | -0.0007 | -0.02 | -0.02 | 0.9728 | 1.0000 | 0.9448 |
| evolutionary_vs_random | -0.0029 | -0.09 | -0.08 | 0.8886 | 0.6905 | 0.8706 |
| llm_vs_random | -0.0043 | -0.13 | -0.12 | 0.8422 | 0.5476 | 0.7429 |
| llm_vs_tpe | -0.0036 | -0.11 | -0.10 | 0.8627 | 0.6905 | 0.7588 |
| llm_vs_evolutionary | -0.0013 | -0.04 | -0.04 | 0.9495 | 1.0000 | 0.9632 |

---

## 3. Scientific Takeaways
- **Search Paradigm Comparison**: Compares uniform stochastic exploration (Random), surrogate model optimization (TPE), population-based genetic selection (Evolutionary), and semantic hypothesis generation (LLM).
- **Task Complexity Scaling**: Evaluates whether the inductive biases discoverable by each search paradigm differ across Chomsky regular state tracking (Hidden FSM) versus context-free hierarchical grammar (Dyck-4).
- **Matched Evaluation Budget**: All candidate configurations are strictly bounded by identical gradient update horizons in the Immutable Evaluator, isolating search algorithm efficacy from training budget disparities.