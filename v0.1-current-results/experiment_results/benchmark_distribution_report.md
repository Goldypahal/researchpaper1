# Empirical Statistical Comparison: Arm 1 (LLM Agent) vs. Arm 2 (Random Search)

> **Sample Size**: N=10 per arm | **Date**: 2026-09-23
> **Provenance**: All Arm 1 trials executed via `nvidia/nemotron-3-ultra-550b-a55b` in strict mode.
> All Arm 2 trials sampled from the identical 8-dimensional search space.

## 1. Distribution Summary (N=10 per arm)

| Arm | N | Mean Loss | Std Dev | Median | IQR | Min (Best) | Max (Worst) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1 (LLM Agent)** | 10 | 7.0410 | 0.0385 | 7.0652 | 0.0450 | **6.9578** | 7.0653 |
| **Arm 2 (Random Search)** | 10 | 6.9926 | 0.0484 | 6.9868 | 0.0707 | **6.9297** | 7.0653 |

## 2. Statistical Significance Testing

- **Welch's Two-Sample t-test**: $t = 2.4748$, $p = 0.0241$
- **Mann-Whitney U Test**: $U = 77.0$, $p = 0.0407$
- **Effect Size (Cohen's d)**: $d = 1.1068$
- **Statistically Significant at $\alpha=0.05$**: **YES**

## 3. Individual Trial Logs

### Arm 1 (LLM Agent)

| Trial | Best Val Loss | Rel. Gain % | Dead-End Ratio | Status |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 7.0653 | 0.64% | 0.75 | SUCCESS |
| 2 | 7.0519 | 0.82% | 0.50 | SUCCESS |
| 3 | 7.0098 | 1.42% | 0.50 | SUCCESS |
| 4 | 7.0653 | 0.64% | 0.75 | SUCCESS |
| 5 | 6.9578 | 2.15% | 0.50 | SUCCESS |
| 6 | 7.0653 | 0.64% | 0.75 | SUCCESS |
| 7 | 7.0653 | 0.64% | 0.75 | SUCCESS |
| 8 | 7.0652 | 0.64% | 0.50 | SUCCESS |
| 9 | 7.0653 | 0.64% | 0.75 | SUCCESS |
| 10 | 6.9989 | 1.57% | 0.50 | SUCCESS |

### Arm 2 (Random Search Baseline)

| Seed | Best Val Loss | Status |
| :---: | :---: | :---: |
| 42 | 6.9918 | SUCCESS |
| 101 | 6.9818 | SUCCESS |
| 777 | 6.9297 | SUCCESS |
| 999 | 6.9496 | SUCCESS |
| 2026 | 7.0653 | SUCCESS |
| 1234 | 6.9462 | SUCCESS |
| 4321 | 7.0653 | SUCCESS |
| 5555 | 7.0312 | SUCCESS |
| 8888 | 6.9637 | SUCCESS |
| 9999 | 7.0018 | SUCCESS |

## 4. Key Takeaways for Research Paper 1

1. **Difference in Means**: Arm 1 validation loss is 0.0484 higher (worse) than Arm 2 on average.
2. **Statistical Power**: Across N=10 trials per arm, the p-value is $0.0241$. The difference between arms is statistically significant at $\alpha = 0.05$.
3. **Variance Profile**: Random search exhibits a spread with standard deviation $\sigma=0.0484$ vs. Agent $\sigma=0.0385$.
