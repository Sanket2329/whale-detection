import numpy as np
import scipy.signal as signal
from typing import Dict, List, Tuple, Optional
from src.evaluation.event_metrics import match_events_bipartite, calculate_metrics_from_counts

def extract_events_from_probabilities(
    probabilities: np.ndarray,
    threshold: float,
    target_sr: int = 250,
    hop_length: int = 64,
    min_duration: float = 0.5,
    filter_kernel_size: int = 7
) -> List[Dict[str, float]]:
    """Applies 1D median filtering and extracts contiguous intervals where p >= threshold.

    Guarantees exact parity between validation threshold optimization and final evaluation.
    """
    if len(probabilities) == 0:
        return []
        
    frame_dur = hop_length / target_sr
    smoothed = signal.medfilt(probabilities, kernel_size=filter_kernel_size)
    binary_mask = (smoothed >= threshold).astype(np.int32)
    
    diff = np.diff(np.pad(binary_mask, (1, 1), mode='constant'))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    events = []
    for s, e in zip(starts, ends):
        dur = (e - s) * frame_dur
        if dur >= min_duration:
            conf = float(np.mean(smoothed[s:e]))
            events.append({
                "begin_time": float(s * frame_dur),
                "end_time": float(e * frame_dur),
                "duration": float(dur),
                "confidence": conf
            })
    return events

def calibrate_event_level_thresholds(
    validation_probabilities_by_rec: Dict[str, np.ndarray],
    validation_gt_by_rec: Dict[str, List[Dict[str, float]]],
    class_names: List[str],
    candidate_thresholds: np.ndarray = None,
    iou_thresh: float = 0.1,
    target_sr: int = 250,
    hop_length: int = 64
) -> Tuple[Dict[str, Optional[float]], Dict[str, float], Dict[str, str], Dict[str, dict]]:
    """Performs Event-Level Threshold Calibration on validation recordings using Hungarian matching.

    
    For each class:
    - Verifies presence of positive ground-truth events in the validation set.
    - If no positive events exist, explicitly flags calibration_status as 'uncalibrated_no_positive_events'.
    - If positive events exist, sweeps candidate thresholds, extracts events, and calculates event-level F1.
    - Selects the threshold maximizing validation event-level F1.
    
    Returns:
        best_thresholds: {class_name: float or None}
        best_f1s: {class_name: float}
        calibration_status: {class_name: status_string}
        diagnostics: {class_name: dict_of_metrics}
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.arange(0.10, 0.91, 0.05)
        
    best_thresholds = {}
    best_f1s = {}
    calibration_status = {}
    diagnostics = {}
    
    rec_ids = list(validation_probabilities_by_rec.keys())
    
    for c_idx, c_name in enumerate(class_names):
        # Count total validation GT events for this class
        total_gt = sum(len(validation_gt_by_rec.get(r, {}).get(c_name, [])) for r in rec_ids)
        
        if total_gt == 0:
            best_thresholds[c_name] = None
            best_f1s[c_name] = 0.0
            calibration_status[c_name] = "uncalibrated_no_positive_events"
            diagnostics[c_name] = {
                "total_gt": 0,
                "best_threshold": None,
                "f1": 0.0,
                "active_frame_pct": 0.0
            }
            continue
            
        best_th = None
        best_f1 = -1.0
        best_m = {}
        
        # Sweep candidate thresholds
        for th in candidate_thresholds:
            c_tp = 0
            c_fp = 0
            c_fn = 0
            
            for r in rec_ids:
                prob_arr = validation_probabilities_by_rec[r][:, c_idx]
                gt_events = validation_gt_by_rec.get(r, {}).get(c_name, [])
                pred_events = extract_events_from_probabilities(
                    probabilities=prob_arr,
                    threshold=th,
                    target_sr=target_sr,
                    hop_length=hop_length
                )
                tp, fp, fn = match_events_bipartite(gt_events, pred_events, iou_threshold=iou_thresh)
                c_tp += tp
                c_fp += fp
                c_fn += fn
                
            m = calculate_metrics_from_counts(c_tp, c_fp, c_fn)
            # Prioritize event-level F1, breaking ties by preferring higher precision (higher threshold)
            if m["f1"] > best_f1 or (m["f1"] == best_f1 and best_th is not None and m["precision"] > best_m.get("precision", 0.0)):
                best_f1 = m["f1"]
                best_th = float(th)
                best_m = m
                
        # If no candidate threshold yielded any positive detection on validation data,
        # do NOT apply a 0.50 fallback. Explicitly mark as uncalibrated.
        if best_th is None or best_f1 <= 0.0:
            best_thresholds[c_name] = None
            best_f1s[c_name] = 0.0
            calibration_status[c_name] = "uncalibrated_no_validation_detection"
            diagnostics[c_name] = {
                "total_gt": total_gt,
                "best_threshold": None,
                "event_f1": 0.0,
                "event_precision": 0.0,
                "event_recall": 0.0,
                "active_frame_pct": 0.0
            }
        else:
            calibration_status[c_name] = "calibrated_event_f1"
            
            # Calculate active frame percentage at selected threshold using median-smoothed probabilities
            all_smoothed_frames = []
            for r in rec_ids:
                prob_arr = validation_probabilities_by_rec[r][:, c_idx]
                if len(prob_arr) > 0:
                    smoothed = signal.medfilt(prob_arr, kernel_size=7)
                    all_smoothed_frames.append(smoothed)
            concat_frames = np.concatenate(all_smoothed_frames, axis=0) if all_smoothed_frames else np.array([])
            active_pct = float((concat_frames >= best_th).mean() * 100.0) if len(concat_frames) > 0 else 0.0
            
            best_thresholds[c_name] = best_th
            best_f1s[c_name] = float(best_f1)
            diagnostics[c_name] = {
                "total_gt": total_gt,
                "best_threshold": best_th,
                "event_f1": best_f1,
                "event_precision": best_m.get("precision", 0.0),
                "event_recall": best_m.get("recall", 0.0),
                "smoothed_active_frame_pct": active_pct
            }
        
    return best_thresholds, best_f1s, calibration_status, diagnostics

# Deprecated legacy helper kept solely for backwards compatibility with older tests.
# Final evaluation and training pipelines strictly use calibrate_event_level_thresholds().
def find_best_thresholds_per_class(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    threshold_range: np.ndarray = None
) -> Tuple[Dict[str, Optional[float]], Dict[str, float], float]:
    """Frame-level baseline threshold search for testing/diagnostic parity."""
    if threshold_range is None:
        threshold_range = np.arange(0.10, 0.91, 0.05)
        
    best_thresholds = {}
    class_f1s = {}
    
    num_classes = len(class_names)
    for c in range(num_classes):
        c_name = class_names[c]
        y_true = targets[:, c]
        y_prob = probabilities[:, c]
        
        if np.sum(y_true) == 0:
            best_thresholds[c_name] = None
            class_f1s[c_name] = 0.0
            continue
            
        best_th = 0.5
        best_f1 = -1.0
        for th in threshold_range:
            y_pred = (y_prob >= th).astype(np.int32)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            score = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            if score > best_f1:
                best_f1 = score
                best_th = th
                
        best_thresholds[c_name] = float(best_th)
        class_f1s[c_name] = float(max(0.0, best_f1))
        
    macro_f1 = float(np.mean(list(class_f1s.values())))
    return best_thresholds, class_f1s, macro_f1
