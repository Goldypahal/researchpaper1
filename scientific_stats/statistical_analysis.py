"""
Rigorous Statistical Analysis & Search Efficiency Framework.
Phase 9 & Phase 10 Implementation.

Features:
- Parametric & Non-Parametric Descriptives: Mean, Std, Median, IQR, 95% Bootstrap CI
- Effect Size Quantification: Cohen's d, Hedges' g
- Hypothesis Significance Testing:
    - Welch's t-test (unequal variances)
    - Mann-Whitney U test (non-parametric rank-sum)
    - Exact/Monte-Carlo Permutation Test
    - Holm-Bonferroni Family-Wise Error Rate (FWER) Correction
- Search Efficiency & Compute-Normalized Metrics:
    - Delta Performance / GPU-hour
    - Delta Performance / Experiment
    - Delta Performance / 1k LLM Tokens
    - Area Under the Search Curve (AUC): AUC = integral_0^B P(b) db
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Tuple, Callable, Optional


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """Computes a non-parametric two-sided bootstrap confidence interval."""
    arr = np.asarray(data)
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_stats, 100 * alpha))
    high = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
    return round(low, 6), round(high, 6)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cohen's d effect size with pooled standard deviation."""
    x, y = np.asarray(x), np.asarray(y)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if sp < 1e-12:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / sp
    return round(float(d), 4)


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Hedges' g (unbiased sample effect size for small N)."""
    d = cohens_d(x, y)
    n = len(x) + len(y)
    correction = 1.0 - (3.0 / (4.0 * n - 9.0))
    return round(float(d * correction), 4)


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = 5000,
    seed: int = 42
) -> float:
    """Two-sample two-sided permutation test for difference in means."""
    x, y = np.asarray(x), np.asarray(y)
    observed_diff = np.abs(np.mean(x) - np.mean(y))
    combined = np.concatenate([x, y])
    n_x = len(x)
    rng = np.random.default_rng(seed)
    
    count = 0
    for _ in range(n_resamples):
        permuted = rng.permutation(combined)
        sim_diff = np.abs(np.mean(permuted[:n_x]) - np.mean(permuted[n_x:]))
        if sim_diff >= observed_diff:
            count += 1
            
    p_val = (count + 1) / (n_resamples + 1)
    return round(float(p_val), 6)


def holm_bonferroni_correction(p_values: List[float]) -> List[Tuple[float, bool]]:
    """
    Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (alpha=0.05).
    Returns list of (adjusted_p_value, is_significant).
    """
    m = len(p_values)
    if m == 0:
        return []
    
    sorted_indices = np.argsort(p_values)
    adjusted = [0.0] * m
    significant = [False] * m
    
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        adj_p = min(1.0, max(running_max, p * (m - rank)))
        running_max = adj_p
        adjusted[idx] = round(adj_p, 6)
        significant[idx] = (adj_p < 0.05)
        
    return list(zip(adjusted, significant))


def calculate_auc_search_curve(compute_points: List[float], performance_points: List[float]) -> float:
    r"""
    Calculates Area Under the Search Curve:
    AUC = \int_0^B P(b) db
    Higher AUC indicates finding high-performing solutions earlier in the compute budget.
    """
    x = np.asarray(compute_points, dtype=np.float64)
    y = np.asarray(performance_points, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    auc = np.trapz(y_sorted, x_sorted)
    return round(float(auc), 4)


def summarize_distribution(data: List[float], seed: int = 42) -> Dict[str, Any]:
    """Generates complete statistical summary of an experimental sample."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return {}
    
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    median_val = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr_val = q75 - q25
    ci_low, ci_high = bootstrap_ci(arr, stat_fn=np.mean, seed=seed)
    
    return {
        "n": len(arr),
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "median": round(median_val, 6),
        "iqr": round(iqr_val, 6),
        "q25": round(q25, 6),
        "q75": round(q75, 6),
        "ci_95": (ci_low, ci_high)
    }


def compare_two_search_arms(
    arm_a_name: str,
    arm_a_values: List[float],
    arm_b_name: str,
    arm_b_values: List[float],
    higher_is_better: bool = False
) -> Dict[str, Any]:
    """
    Conducts full comparative statistical hypothesis test between two search arms.
    """
    a = np.asarray(arm_a_values, dtype=np.float64)
    b = np.asarray(arm_b_values, dtype=np.float64)
    
    # Parametric Welch's t-test
    t_stat, t_pval = stats.ttest_ind(a, b, equal_var=False)
    
    # Non-parametric Mann-Whitney U test
    try:
        mw_stat, mw_pval = stats.mannwhitneyu(a, b, alternative='two-sided')
    except Exception:
        mw_stat, mw_pval = 0.0, 1.0
        
    # Non-parametric Permutation Test
    perm_pval = permutation_test(a, b)
    
    # Effect sizes
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    
    # Mean difference
    diff = float(np.mean(a) - np.mean(b))
    
    return {
        "comparison": f"{arm_a_name} vs {arm_b_name}",
        "mean_diff": round(diff, 6),
        "cohens_d": d,
        "hedges_g": g,
        "welch_t": {
            "statistic": round(float(t_stat), 4),
            "p_value": round(float(t_pval), 6)
        },
        "mann_whitney_u": {
            "statistic": round(float(mw_stat), 4),
            "p_value": round(float(mw_pval), 6)
        },
        "permutation_test": {
            "p_value": perm_pval
        }
    }
