import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Set

def compute_temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Computes temporal Intersection-over-Union (IoU) between two time intervals."""
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union

def match_events_bipartite(
    gt_events: List[Dict[str, float]],
    pred_events: List[Dict[str, float]],
    iou_threshold: float = 0.1
) -> Tuple[int, int, int]:
    """Strict one-to-one event matching using maximum-weight bipartite matching

    (Hungarian algorithm via SciPy linear_sum_assignment).
    
    Guarantees:
    - Exactly one ground-truth event matches at most one predicted event.
    - Matches maximize total IoU among pairs where IoU >= iou_threshold.
    Returns:
        (true_positives, false_positives, false_negatives)
    """
    n_gt = len(gt_events)
    n_pred = len(pred_events)
    
    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0
    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt
        
    # Maximum-cardinality bipartite matching:
    # Any pair with IoU >= iou_threshold receives a large matching bonus (MATCH_BONUS = 1e4)
    # plus the actual IoU as secondary objective, ensuring the Hungarian algorithm first
    # maximizes the total number of valid TP matches, then maximizes total IoU among them.
    MATCH_BONUS = 10000.0
    DISALLOWED_COST = 1e9
    
    cost_matrix = np.full((n_gt, n_pred), DISALLOWED_COST, dtype=np.float64)
    for i, gt in enumerate(gt_events):
        for j, pred in enumerate(pred_events):
            iou = compute_temporal_iou(gt["begin_time"], gt["end_time"], pred["begin_time"], pred["end_time"])
            if iou >= iou_threshold:
                cost_matrix[i, j] = -(MATCH_BONUS + iou)
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 0.0: # Valid match with IoU >= iou_threshold
            tp += 1
            
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn

def calculate_metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

