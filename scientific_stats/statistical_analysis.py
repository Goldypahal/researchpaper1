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


def calculate_normalized_gain_auc(
    compute_points: List[float],
    incumbent_losses: List[float],
    base_loss: float
) -> float:
    r"""
    Calculates Area Under the Normalized Performance Gain Curve:
      Gain(b) = max(0.0, base_loss - L*(b)) / max(1e-8, base_loss)
      AUC_gain = \int_0^B Gain(b) db
    HIGHER is strictly better: reflects discovering larger performance gains earlier in the search budget.
    """
    x = np.asarray(compute_points, dtype=np.float64)
    losses = np.asarray(incumbent_losses, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    losses_sorted = losses[sort_idx]
    denom = max(1e-8, base_loss)
    gains = np.maximum(0.0, (base_loss - losses_sorted) / denom)
    auc = np.trapz(gains, x_sorted)
    return round(float(auc), 6)


def calculate_auc_search_curve(compute_points: List[float], performance_points: List[float], base_loss: Optional[float] = None) -> float:
    """Backwards-compatible wrapper routing to calculate_normalized_gain_auc."""
    if base_loss is not None:
        return calculate_normalized_gain_auc(compute_points, performance_points, base_loss)
    # If base_loss is not explicitly provided, treat performance_points as raw gain
    x = np.asarray(compute_points, dtype=np.float64)
    y = np.asarray(performance_points, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    sort_idx = np.argsort(x)
    return round(float(np.trapz(y[sort_idx], x[sort_idx])), 6)


def paired_difference_analysis(
    arm_a_values: List[float],
    arm_b_values: List[float],
    arm_a_name: str = "Arm_A",
    arm_b_name: str = "Arm_B"
) -> Dict[str, Any]:
    """
    Conducts paired hypothesis tests exploiting matched seed experimental units:
      D_i = Arm_A[i] - Arm_B[i]
    """
    a = np.asarray(arm_a_values, dtype=np.float64)
    b = np.asarray(arm_b_values, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Paired comparison requires identical sample sizes: len(a)={len(a)} != len(b)={len(b)}")

    diff = a - b
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    dz = mean_d / std_d if std_d > 1e-12 else 0.0

    try:
        t_stat, t_pval = stats.ttest_rel(a, b)
    except Exception:
        t_stat, t_pval = 0.0, 1.0

    try:
        w_stat, w_pval = stats.wilcoxon(diff, alternative='two-sided')
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    ci_low, ci_high = bootstrap_ci(diff, stat_fn=np.mean)

    return {
        "comparison": f"{arm_a_name} - {arm_b_name}",
        "paired_mean_difference": round(mean_d, 6),
        "paired_std_difference": round(std_d, 6),
        "paired_cohens_dz": round(float(dz), 4),
        "ci_95_difference": (ci_low, ci_high),
        "paired_t_test": {
            "statistic": round(float(t_stat), 4),
            "p_value": round(float(t_pval), 6)
        },
        "wilcoxon_signed_rank": {
            "statistic": round(float(w_stat), 4),
            "p_value": round(float(w_pval), 6)
        }
    }


def evaluate_hypothesis_calibration(
    predicted_deltas: List[float],
    actual_deltas: List[float]
) -> Dict[str, Any]:
    """
    Comprehensive metacognitive calibration analysis:
      1. Mean Absolute Error (MAE)
      2. Directional Accuracy (% of predictions correctly identifying gain vs loss)
      3. Spearman Rank Correlation (assessing relative ranking awareness)
      4. Confidence-Binned Expected Calibration
    """
    pred = np.asarray(predicted_deltas, dtype=np.float64)
    actual = np.asarray(actual_deltas, dtype=np.float64)
    if len(pred) == 0:
        return {}

    mae = float(np.mean(np.abs(pred - actual)))

    pred_sign = np.sign(pred)
    actual_sign = np.sign(actual)
    directional_acc = float(np.mean(pred_sign == actual_sign)) * 100.0

    try:
        spearman_rho, spearman_pval = stats.spearmanr(pred, actual)
    except Exception:
        spearman_rho, spearman_pval = 0.0, 1.0

    bins = [0.0, 0.02, 0.05, 0.10, float('inf')]
    bin_labels = ["[0, 0.02)", "[0.02, 0.05)", "[0.05, 0.10)", ">= 0.10"]
    binned_metrics = []

    for idx in range(len(bins) - 1):
        low, high = bins[idx], bins[idx + 1]
        mask = (np.abs(pred) >= low) & (np.abs(pred) < high)
        if np.sum(mask) > 0:
            binned_metrics.append({
                "bin": bin_labels[idx],
                "count": int(np.sum(mask)),
                "mean_predicted": round(float(np.mean(pred[mask])), 6),
                "mean_actual": round(float(np.mean(actual[mask])), 6),
                "bin_error": round(float(np.abs(np.mean(pred[mask]) - np.mean(actual[mask]))), 6)
            })

    return {
        "n_predictions": len(pred),
        "mae": round(mae, 6),
        "directional_accuracy_pct": round(directional_acc, 2),
        "spearman_rank_correlation": {
            "rho": round(float(spearman_rho), 4),
            "p_value": round(float(spearman_pval), 6)
        },
        "confidence_binned_calibration": binned_metrics
    }


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
