import numpy as np
import pandas as pd
from typing import List, Dict, Callable

def recording_level_bootstrap_ci(
    recording_ids: List[str],
    eval_fn: Callable[[List[str]], Dict[str, float]],
    n_iterations: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Performs recording-level bootstrap resampling to compute valid non-parametric

    confidence intervals that preserve intra-recording correlation.
    """
    rng = np.random.RandomState(seed)
    n_recordings = len(recording_ids)
    
    bootstrap_results = {}
    
    for i in range(n_iterations):
        sample_recs = rng.choice(recording_ids, size=n_recordings, replace=True).tolist()
        metrics = eval_fn(sample_recs)
        
        for k, v in metrics.items():
            if k not in bootstrap_results:
                bootstrap_results[k] = []
            bootstrap_results[k].append(v)
            
    summary = {}
    lower_pct = ((1.0 - ci_level) / 2.0) * 100.0
    upper_pct = (1.0 - (1.0 - ci_level) / 2.0) * 100.0
    
    for k, values in bootstrap_results.items():
        arr = np.array(values)
        summary[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, lower_pct)),
            "ci_upper": float(np.percentile(arr, upper_pct))
        }
        
    return summary

def paired_recording_level_bootstrap_comparison(
    recording_ids: List[str],
    eval_fn_a: Callable[[List[str]], Dict[str, float]],
    eval_fn_b: Callable[[List[str]], Dict[str, float]],
    n_iterations: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Performs paired recording-level bootstrap comparison between Model B (e.g. DANN)

    and Model A (e.g. Baseline CRNN).
    
    For each bootstrap iteration:
    1. Resamples the exact same list of recording IDs with replacement for both models.
    2. Evaluates Model A and Model B preserving duplicate recording occurrences.
    3. Computes the empirical paired difference: Delta = Metric_B - Metric_A.
    4. Computes empirical percentiles [2.5%, 97.5%] of the Delta distribution.
    """
    rng = np.random.RandomState(seed)
    n_recordings = len(recording_ids)
    
    delta_results = {}
    
    for i in range(n_iterations):
        sample_recs = rng.choice(recording_ids, size=n_recordings, replace=True).tolist()
        metrics_a = eval_fn_a(sample_recs)
        metrics_b = eval_fn_b(sample_recs)
        
        for k in metrics_a.keys():
            if k not in delta_results:
                delta_results[k] = []
            diff = metrics_b[k] - metrics_a[k]
            delta_results[k].append(diff)
            
    summary = {}
    lower_pct = ((1.0 - ci_level) / 2.0) * 100.0
    upper_pct = (1.0 - (1.0 - ci_level) / 2.0) * 100.0
    
    for k, values in delta_results.items():
        arr = np.array(values)
        ci_lower = float(np.percentile(arr, lower_pct))
        ci_upper = float(np.percentile(arr, upper_pct))
        includes_zero = bool(ci_lower <= 0.0 <= ci_upper)
        
        # Using sample standard deviation (ddof=1) across bootstrap replicates as the bootstrap standard error
        se = float(np.std(arr, ddof=1)) if len(arr) > 1 else float(np.std(arr))
        
        summary[k] = {
            "mean_difference": float(np.mean(arr)),
            "standard_error": se,
            "std_difference": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "includes_zero": includes_zero
        }
        
    return summary
