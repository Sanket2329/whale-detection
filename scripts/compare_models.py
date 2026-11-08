import os
import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from typing import List, Dict

from src.data.dataset import (
    parse_annotations,
    normalise_recording_filename,
    CLASS_NAMES
)
from src.evaluation.event_metrics import match_events_bipartite, calculate_metrics_from_counts
from src.evaluation.confidence_intervals import paired_recording_level_bootstrap_comparison

def parse_args():
    parser = argparse.ArgumentParser(description="Paired Model Comparison Script")
    parser.add_argument("--crnn_dir", type=str, default="results/final_evaluation/crnn", help="CRNN evaluation directory")
    parser.add_argument("--dann_dir", type=str, default="results/final_evaluation/dann", help="DANN evaluation directory")
    parser.add_argument("--annotation_dir", type=str, default="data/Greenwich64S2015", help="Ground truth annotations directory")
    parser.add_argument("--output_dir", type=str, default="results/final_evaluation/comparison", help="Output comparison directory")
    parser.add_argument("--test_manifest", type=str, default="data/splits/target_test.csv", help="Test manifest CSV")
    parser.add_argument("--bootstrap_iterations", type=int, default=2000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def validate_and_load_predictions(
    preds_csv_path: str,
    manifest_rec_ids: List[str],
    class_names: List[str],
    model_name: str
) -> Dict[str, Dict[str, list]]:
    """Loads prediction CSV and strictly validates coverage, recording IDs, classes, and timestamps."""
    if not os.path.exists(preds_csv_path):
        raise FileNotFoundError(f"FATAL: Missing prediction CSV for {model_name} at: {preds_csv_path}")
        
    df_p = pd.read_csv(preds_csv_path)
    
    # Check for unexpected class names
    pred_classes = df_p["class_name"].unique().tolist() if not df_p.empty else []
    for c in pred_classes:
        if c not in class_names:
            raise ValueError(f"FATAL: Unknown class name '{c}' found in {model_name} predictions ({preds_csv_path})")
            
    # Check for unexpected recording IDs
    manifest_set = set(manifest_rec_ids)
    pred_rec_ids = df_p["recording_id"].astype(str).unique().tolist() if not df_p.empty else []
    for r in pred_rec_ids:
        if r not in manifest_set:
            raise ValueError(f"FATAL: Prediction recording ID '{r}' in {model_name} does not exist in target_test.csv!")
            
    # Initialize container for all manifest recordings
    preds_by_rec = {r: {c: [] for c in class_names} for r in manifest_rec_ids}
    
    # Populate and validate timestamps
    for idx, row in df_p.iterrows():
        r_id = str(row["recording_id"])
        c_name = str(row["class_name"])
        b_time = float(row["begin_time"])
        e_time = float(row["end_time"])
        
        if not (np.isfinite(b_time) and np.isfinite(e_time) and b_time >= 0.0 and e_time > b_time):
            raise ValueError(f"FATAL: Invalid timestamps for {model_name} row {idx}: begin={b_time}, end={e_time}")
            
        dur = float(row.get("duration", e_time - b_time))
        conf = float(row.get("confidence", 1.0))
        
        # Strict validation of duration, timestamp matching, and confidence bounds [0, 1]
        if not (np.isfinite(dur) and dur > 0.0 and np.isfinite(conf) and 0.0 <= conf <= 1.0):
            raise ValueError(
                f"FATAL: Invalid duration or confidence for {model_name} row {idx}: "
                f"duration={dur}, confidence={conf}"
            )
        if not np.isclose(dur, e_time - b_time, atol=1e-3):
            raise ValueError(
                f"FATAL: Duration mismatch for {model_name} row {idx}: "
                f"duration={dur} vs (end - begin)={e_time - b_time}"
            )
            
        preds_by_rec[r_id][c_name].append({
            "begin_time": b_time,
            "end_time": e_time,
            "confidence": conf
        })
        
    # Check coverage
    active_recs = len([r for r in manifest_rec_ids if any(preds_by_rec[r][c] for c in class_names)])
    print(f"Validated {model_name} predictions: {len(df_p)} events across {active_recs}/{len(manifest_rec_ids)} recordings.")
    return preds_by_rec

def create_eval_function(
    preds_by_rec: Dict[str, Dict[str, list]],
    gt_by_rec: Dict[str, Dict[str, list]],
    rec_duration_hours: Dict[str, float],
    class_names: List[str],
    iou_thresh: float = 0.1
):
    """Returns an evaluation function that processes sampled recordings preserving duplicate occurrences."""
    def eval_fn(sample_recs: List[str]) -> Dict[str, float]:
        sample_audio_hours = sum(rec_duration_hours.get(r, 0.0) for r in sample_recs)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        class_f1s = []
        class_precs = []
        class_recs = []
        
        for c in class_names:
            c_tp = 0
            c_fp = 0
            c_fn = 0
            
            for r in sample_recs:
                gts = gt_by_rec.get(r, {}).get(c, [])
                preds = preds_by_rec.get(r, {}).get(c, [])
                tp, fp, fn = match_events_bipartite(gts, preds, iou_threshold=iou_thresh)
                c_tp += tp
                c_fp += fp
                c_fn += fn
                
            m = calculate_metrics_from_counts(c_tp, c_fp, c_fn)
            class_f1s.append(m["f1"])
            class_precs.append(m["precision"])
            class_recs.append(m["recall"])
            
            total_tp += c_tp
            total_fp += c_fp
            total_fn += c_fn
            
        macro_f1 = float(np.mean(class_f1s))
        macro_prec = float(np.mean(class_precs))
        macro_rec = float(np.mean(class_recs))
        micro_m = calculate_metrics_from_counts(total_tp, total_fp, total_fn)
        fp_per_hour = total_fp / max(0.01, sample_audio_hours)
        
        res = {
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "micro_f1": micro_m["f1"],
            "false_positives_per_hour": fp_per_hour,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn
        }
        for c, f1, prec, rec in zip(class_names, class_f1s, class_precs, class_recs):
            res[f"f1_{c}"] = f1
            res[f"precision_{c}"] = prec
            res[f"recall_{c}"] = rec
        return res
    return eval_fn

def reconcile_point_estimates(computed_point: dict, saved_json_path: str, model_name: str, tol: float = 1e-4):
    """Reconciles computed point estimate with saved metrics_iou_01.json, halting on discrepancy."""
    if not os.path.exists(saved_json_path):
        print(f"Note: Saved metrics file not found at {saved_json_path}. Using computed point estimates.")
        return
        
    saved = json.load(open(saved_json_path))
    for k in ["macro_f1", "macro_precision", "macro_recall", "micro_f1", "false_positives_per_hour"]:
        val_comp = computed_point[k]
        val_saved = saved[k]
        diff = abs(val_comp - val_saved)
        if diff > tol:
            raise RuntimeError(
                f"FATAL Discrepancy for {model_name} on metric '{k}': "
                f"computed from predictions.csv = {val_comp:.6f}, saved in JSON = {val_saved:.6f} (diff={diff:.6f}). "
                "Halting execution to prevent comparing stale point estimates against fresh bootstrap draws."
            )
    print(f"Point estimate reconciliation passed for {model_name} (computed matches {saved_json_path}).")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load and strictly validate ground truth annotations
    df_manifest = pd.read_csv(args.test_manifest)
    rec_ids = [str(r) for r in df_manifest["recording_id"].tolist()]
    
    all_anns_df = parse_annotations(args.annotation_dir)
    gt_by_rec = {}
    rec_duration_hours = {}
    
    for _, row in df_manifest.iterrows():
        rec_id = str(row["recording_id"])
        fpath = row["file_path"]
        fname = normalise_recording_filename(fpath)
        
        info = sf.info(fpath)
        rec_duration_hours[rec_id] = info.duration / 3600.0
        
        match = all_anns_df[all_anns_df["wav_file"] == fname]
        gt_by_rec[rec_id] = {c: [] for c in CLASS_NAMES}
        for g in match.to_dict("records"):
            c = g["class_name"]
            gt_by_rec[rec_id][c].append({
                "begin_time": float(g["begin_time"]),
                "end_time": float(g["end_time"])
            })
            
    # Ground-truth integrity assertions
    total_gt = sum(len(events) for rec in gt_by_rec.values() for events in rec.values())
    if total_gt == 0:
        raise RuntimeError("FATAL: No ground-truth events matched the target test manifest.")
        
    zero_ann_recs = [r for r in rec_ids if not any(gt_by_rec[r][c] for c in CLASS_NAMES)]
    print(f"Ground-truth audit: Loaded {total_gt} reference events across {len(rec_ids)} recordings.")
    print(f"Recordings with >= 1 annotations: {len(rec_ids) - len(zero_ann_recs)} | Clean background recordings (0 annotations): {len(zero_ann_recs)}")
    
    # Print per-class ground truth breakdown
    print("Test ground-truth events by class:")
    for c in CLASS_NAMES:
        c_count = sum(len(gt_by_rec[r][c]) for r in rec_ids)
        print(f"  {c:14s}: {c_count:4d} events")
        
    # 2. Validate and load predictions from both models
    crnn_preds_csv = os.path.join(args.crnn_dir, "predictions.csv")
    dann_preds_csv = os.path.join(args.dann_dir, "predictions.csv")
    
    preds_crnn = validate_and_load_predictions(crnn_preds_csv, rec_ids, CLASS_NAMES, "CRNN Baseline")
    preds_dann = validate_and_load_predictions(dann_preds_csv, rec_ids, CLASS_NAMES, "CRNN DANN")
    
    # 3. Create evaluation closures and reconcile point estimates
    eval_fn_crnn = create_eval_function(preds_crnn, gt_by_rec, rec_duration_hours, CLASS_NAMES, iou_thresh=0.1)
    eval_fn_dann = create_eval_function(preds_dann, gt_by_rec, rec_duration_hours, CLASS_NAMES, iou_thresh=0.1)
    
    crnn_point = eval_fn_crnn(rec_ids)
    dann_point = eval_fn_dann(rec_ids)
    
    reconcile_point_estimates(crnn_point, os.path.join(args.crnn_dir, "metrics_iou_01.json"), "CRNN Baseline")
    reconcile_point_estimates(dann_point, os.path.join(args.dann_dir, "metrics_iou_01.json"), "CRNN DANN")
    
    # 4. Execute Paired Recording-Level Bootstrap
    print(f"\n=== Executing Paired Recording-Level Bootstrap Comparison ({args.bootstrap_iterations} draws) ===")
    paired_ci = paired_recording_level_bootstrap_comparison(
        recording_ids=rec_ids,
        eval_fn_a=eval_fn_crnn,
        eval_fn_b=eval_fn_dann,
        n_iterations=args.bootstrap_iterations,
        ci_level=0.95,
        seed=args.seed
    )
    
    paired_json_path = os.path.join(args.output_dir, "paired_bootstrap_comparison.json")
    with open(paired_json_path, "w") as f:
        json.dump(paired_ci, f, indent=4)
    print(f"Saved paired bootstrap results to {paired_json_path}")
    
    # 5. Build Final Reconciled Comparison Table
    rows = [
        {
            'Metric': 'Macro-F1 (IoU >= 0.1)',
            'CRNN Baseline': f"{crnn_point['macro_f1']:.4f}",
            'CRNN DANN': f"{dann_point['macro_f1']:.4f}",
            'Mean Paired Difference (DANN - CRNN)': f"{paired_ci['macro_f1']['mean_difference']:+.4f}",
            'Bootstrap Standard Error': f"{paired_ci['macro_f1']['standard_error']:.4f}",
            'Paired 95% Bootstrap CI': f"[{paired_ci['macro_f1']['ci_lower']:+.4f}, {paired_ci['macro_f1']['ci_upper']:+.4f}]",
            'Includes Zero': str(paired_ci['macro_f1']['includes_zero'])
        },
        {
            'Metric': 'Micro-F1 (IoU >= 0.1)',
            'CRNN Baseline': f"{crnn_point['micro_f1']:.4f}",
            'CRNN DANN': f"{dann_point['micro_f1']:.4f}",
            'Mean Paired Difference (DANN - CRNN)': f"{paired_ci['micro_f1']['mean_difference']:+.4f}",
            'Bootstrap Standard Error': f"{paired_ci['micro_f1']['standard_error']:.4f}",
            'Paired 95% Bootstrap CI': f"[{paired_ci['micro_f1']['ci_lower']:+.4f}, {paired_ci['micro_f1']['ci_upper']:+.4f}]",
            'Includes Zero': str(paired_ci['micro_f1']['includes_zero'])
        },
        {
            'Metric': 'Macro-Precision (IoU >= 0.1)',
            'CRNN Baseline': f"{crnn_point['macro_precision']:.4f}",
            'CRNN DANN': f"{dann_point['macro_precision']:.4f}",
            'Mean Paired Difference (DANN - CRNN)': f"{paired_ci['macro_precision']['mean_difference']:+.4f}",
            'Bootstrap Standard Error': f"{paired_ci['macro_precision']['standard_error']:.4f}",
            'Paired 95% Bootstrap CI': f"[{paired_ci['macro_precision']['ci_lower']:+.4f}, {paired_ci['macro_precision']['ci_upper']:+.4f}]",
            'Includes Zero': str(paired_ci['macro_precision']['includes_zero'])
        },
        {
            'Metric': 'Macro-Recall (IoU >= 0.1)',
            'CRNN Baseline': f"{crnn_point['macro_recall']:.4f}",
            'CRNN DANN': f"{dann_point['macro_recall']:.4f}",
            'Mean Paired Difference (DANN - CRNN)': f"{paired_ci['macro_recall']['mean_difference']:+.4f}",
            'Bootstrap Standard Error': f"{paired_ci['macro_recall']['standard_error']:.4f}",
            'Paired 95% Bootstrap CI': f"[{paired_ci['macro_recall']['ci_lower']:+.4f}, {paired_ci['macro_recall']['ci_upper']:+.4f}]",
            'Includes Zero': str(paired_ci['macro_recall']['includes_zero'])
        },
        {
            'Metric': 'False Positives / Hour',
            'CRNN Baseline': f"{crnn_point['false_positives_per_hour']:.2f}",
            'CRNN DANN': f"{dann_point['false_positives_per_hour']:.2f}",
            'Mean Paired Difference (DANN - CRNN)': f"{paired_ci['false_positives_per_hour']['mean_difference']:+.2f}",
            'Bootstrap Standard Error': f"{paired_ci['false_positives_per_hour']['standard_error']:.2f}",
            'Paired 95% Bootstrap CI': f"[{paired_ci['false_positives_per_hour']['ci_lower']:+.2f}, {paired_ci['false_positives_per_hour']['ci_upper']:+.2f}]",
            'Includes Zero': str(paired_ci['false_positives_per_hour']['includes_zero'])
        }
    ]
    
    for c in CLASS_NAMES:
        f1_key = f"f1_{c}"
        rows.append({
            'Metric': f"F1 ({c})",
            'CRNN Baseline': f"{crnn_point[f1_key]:.4f}",
            'CRNN DANN': f"{dann_point[f1_key]:.4f}",
            'Mean Paired Difference (DANN - CRNN)': f"{paired_ci[f1_key]['mean_difference']:+.4f}",
            'Bootstrap Standard Error': f"{paired_ci[f1_key]['standard_error']:.4f}",
            'Paired 95% Bootstrap CI': f"[{paired_ci[f1_key]['ci_lower']:+.4f}, {paired_ci[f1_key]['ci_upper']:+.4f}]",
            'Includes Zero': str(paired_ci[f1_key]['includes_zero'])
        })
    
    df_comp = pd.DataFrame(rows)
    comp_csv_path = os.path.join(args.output_dir, "model_comparison.csv")
    df_comp.to_csv(comp_csv_path, index=False)
    print("\n" + df_comp.to_string(index=False))
    print(f"\nFinal paired model comparison saved to {comp_csv_path}")

if __name__ == "__main__":
    main()
