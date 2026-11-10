import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.preprocessing.audio import preprocess_audio
from src.preprocessing.spectrogram import generate_spectrogram
from src.models.crnn import WhaleCRNN, WhaleCRNNDANN
from src.training.checkpoint import load_checkpoint
from src.data.dataset import (
    parse_annotations,
    normalise_recording_filename,
    CLASS_MAPPING,
    CLASS_NAMES
)
from src.evaluation.event_metrics import match_events_bipartite, calculate_metrics_from_counts
from src.evaluation.threshold import extract_events_from_probabilities
from src.evaluation.confidence_intervals import recording_level_bootstrap_ci

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4: Final Evaluation with Validated Thresholds")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--test_manifest", type=str, default="data/splits/target_test.csv", help="Test manifest CSV")
    parser.add_argument("--annotation_dir", type=str, default="data/Greenwich64S2015", help="Directory of ground truth Raven tables")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference (mps, cuda, cpu)")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum event duration in seconds")
    parser.add_argument("--max_duration", type=float, default=None, help="Optional maximum event duration in seconds to reject continuous floor activations")
    parser.add_argument("--bootstrap_iterations", type=int, default=2000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    return parser.parse_args()

def validate_frozen_thresholds(class_thresholds: dict, class_names: list) -> dict:
    """Validates frozen thresholds from checkpoint, rejecting null, out-of-range,

    or silently defaulted thresholds.
    """
    if not isinstance(class_thresholds, dict) or len(class_thresholds) == 0:
        raise ValueError("FATAL: Checkpoint contains empty or invalid class_thresholds dictionary!")
        
    validated = {}
    for c in class_names:
        th = class_thresholds.get(c)
        if th is None or pd.isna(th):
            print(f"   [Evaluation Notice] Class '{c}' is uncalibrated (threshold=None). Skipping event extraction.")
            validated[c] = None
        else:
            th_float = float(th)
            if not np.isfinite(th_float) or th_float <= 0.0 or th_float >= 1.0:
                raise ValueError(f"FATAL: Checkpoint threshold for '{c}' is invalid: {th_float} (must be in (0, 1))")
            validated[c] = th_float
    return validated

def run_continuous_recording_inference(
    model: torch.nn.Module,
    wav_path: str,
    device: torch.device,
    clip_duration: float = 30.0,
    stride: float = 15.0,
    target_sr: int = 250,
    hop_length: int = 64
) -> np.ndarray:
    """Generates full continuous frame-level probability trajectory for a recording."""
    with sf.SoundFile(wav_path) as sfile:
        orig_sr = sfile.samplerate
        total_samples = len(sfile)
        duration_sec = total_samples / orig_sr
        
    num_frames_total = int(np.ceil((duration_sec * target_sr) / hop_length))
    num_classes = len(CLASS_MAPPING)
    
    prob_accum = np.zeros((num_frames_total, num_classes), dtype=np.float32)
    count_accum = np.zeros(num_frames_total, dtype=np.float32)
    
    starts = np.arange(0, max(0.1, duration_sec - clip_duration + 1), stride)
    if len(starts) == 0:
        starts = np.array([0.0])
        
    model.eval()
    with torch.no_grad():
        for s in starts:
            with sf.SoundFile(wav_path) as sfile:
                start_frame = int(s * orig_sr)
                frames_to_read = int(clip_duration * orig_sr)
                sfile.seek(start_frame)
                audio = sfile.read(frames_to_read, dtype='float32')
                
            if len(audio) < frames_to_read:
                audio = np.pad(audio, (0, frames_to_read - len(audio)))
                
            processed = preprocess_audio(audio, original_sr=orig_sr, target_sr=target_sr)
            expected_len = int(clip_duration * target_sr)
            if len(processed) < expected_len:
                processed = np.pad(processed, (0, expected_len - len(processed)))
            else:
                processed = processed[:expected_len]
                
            spec = generate_spectrogram(processed, n_fft=256, hop_length=hop_length)
            spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(device)
            
            logits = model(spec_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy() # (118, num_classes)
            
            f_start = int(s * target_sr / hop_length)
            f_end = min(num_frames_total, f_start + probs.shape[0])
            p_len = f_end - f_start
            
            prob_accum[f_start:f_end] += probs[:p_len]
            count_accum[f_start:f_end] += 1.0
            
    count_accum[count_accum == 0] = 1.0
    return prob_accum / count_accum[:, None]

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
    print(f"=== Stage 4: Held-Out Final Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Manifest: {args.test_manifest}")
    print(f"Device: {device}")
    
    # 1. Load Checkpoint and Validate Frozen Thresholds
    ckpt = load_checkpoint(args.checkpoint, device="cpu")
    arch = ckpt.get("config", {}).get("model", {}).get("architecture", "crnn")
    
    if "dann" in arch:
        model = WhaleCRNNDANN(num_classes=8)
    else:
        model = WhaleCRNN(num_classes=8)
        
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    raw_thresholds = ckpt.get("class_thresholds", {})
    frozen_thresholds = validate_frozen_thresholds(raw_thresholds, CLASS_NAMES)
    print(f"Validated frozen class thresholds: {frozen_thresholds}")
    
    # 2. Load Held-Out Test Manifest and Annotations
    df_manifest = pd.read_csv(args.test_manifest)
    print(f"Loaded {len(df_manifest)} held-out test recordings.")
    
    all_anns_df = parse_annotations(args.annotation_dir)
    if all_anns_df.empty:
        raise RuntimeError("FATAL: No annotations were parsed from the annotation directory.")
        
    gt_by_rec = {}
    rec_duration_hours = {}
    
    for _, row in df_manifest.iterrows():
        rec_id = row["recording_id"]
        fpath = row["file_path"]
        fname = normalise_recording_filename(fpath)
        
        info = sf.info(fpath)
        rec_duration_hours[rec_id] = info.duration / 3600.0
        
        match = all_anns_df[all_anns_df["wav_file"] == fname]
        gt_events = match.to_dict("records")
        gt_by_rec[rec_id] = {c: [] for c in CLASS_NAMES}
        for g in gt_events:
            c = g["class_name"]
            gt_by_rec[rec_id][c].append({
                "begin_time": float(g["begin_time"]),
                "end_time": float(g["end_time"])
            })
            
    total_gt = sum(len(events) for recording in gt_by_rec.values() for events in recording.values())
    if total_gt == 0:
        raise RuntimeError("FATAL: Evaluator loaded zero ground-truth events.")
    print(f"Successfully loaded {total_gt} ground-truth test annotations.")
    
    # 3. Generate Continuous Probabilities & Extract Events
    pred_by_rec = {}
    all_pred_rows = []
    active_frame_stats = {c: [] for c in CLASS_NAMES}
    
    print("\nRunning inference across held-out test recordings...")
    for idx, row in df_manifest.iterrows():
        rec_id = row["recording_id"]
        fpath = row["file_path"]
        
        avg_probs = run_continuous_recording_inference(model, fpath, device)
        pred_by_rec[rec_id] = {c: [] for c in CLASS_NAMES}
        
        for c_idx, c_name in enumerate(CLASS_NAMES):
            th = frozen_thresholds[c_name]
            if th is None:
                continue
                
            prob_col = avg_probs[:, c_idx]
            active_pct = float((prob_col >= th).mean() * 100.0)
            active_frame_stats[c_name].append(active_pct)
            
            events = extract_events_from_probabilities(
                probabilities=prob_col,
                threshold=th,
                target_sr=250,
                hop_length=64,
                min_duration=args.min_duration,
                max_duration=args.max_duration
            )
            for e in events:
                e["class_name"] = c_name
                pred_by_rec[rec_id][c_name].append(e)
                all_pred_rows.append({
                    "recording_id": rec_id,
                    "file_path": fpath,
                    "class_name": c_name,
                    "begin_time": e["begin_time"],
                    "end_time": e["end_time"],
                    "duration": e["duration"],
                    "confidence": e["confidence"]
                })
                
    # Save predictions CSV
    preds_df = pd.DataFrame(all_pred_rows)
    preds_csv_path = os.path.join(args.output_dir, "predictions.csv")
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"Saved {len(all_pred_rows)} predictions to {preds_csv_path}")
    
    # Active frame diagnostics
    print("\n--- Test Active Frame Diagnostics ---")
    for c_name in CLASS_NAMES:
        pcts = active_frame_stats[c_name]
        mean_act = np.mean(pcts) if pcts else 0.0
        print(f"  {c_name:14s}: Mean Active Frames = {mean_act:.2f}% (th={frozen_thresholds[c_name]})")
        
    # 4. Metric Computation Function
    rec_ids = df_manifest["recording_id"].tolist()
    
    def compute_all_metrics(recordings: list[str], iou_thresh: float) -> dict:
        class_metrics = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0
        total_pred = 0
        
        sample_audio_hours = sum(rec_duration_hours[r] for r in recordings)
        
        for c in CLASS_NAMES:
            c_tp = 0
            c_fp = 0
            c_fn = 0
            
            for r in recordings:
                gts = gt_by_rec[r][c]
                preds = pred_by_rec[r][c]
                
                tp, fp, fn = match_events_bipartite(gts, preds, iou_threshold=iou_thresh)
                c_tp += tp
                c_fp += fp
                c_fn += fn
                
            m = calculate_metrics_from_counts(c_tp, c_fp, c_fn)
            m["gt_count"] = c_tp + c_fn
            m["pred_count"] = c_tp + c_fp
            m["calibrated"] = bool(frozen_thresholds[c] is not None)
            class_metrics[c] = m
            
            total_tp += c_tp
            total_fp += c_fp
            total_fn += c_fn
            total_gt += (c_tp + c_fn)
            total_pred += (c_tp + c_fp)
            
        macro_prec = float(np.mean([class_metrics[c]["precision"] for c in CLASS_NAMES]))
        macro_rec = float(np.mean([class_metrics[c]["recall"] for c in CLASS_NAMES]))
        macro_f1 = float(np.mean([class_metrics[c]["f1"] for c in CLASS_NAMES]))
        
        micro_metrics = calculate_metrics_from_counts(total_tp, total_fp, total_fn)
        fp_per_hour = total_fp / max(0.01, sample_audio_hours)
        
        return {
            "class_metrics": class_metrics,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "micro_f1": micro_metrics["f1"],
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "false_positives_per_hour": fp_per_hour
        }
        
    metrics_01 = compute_all_metrics(rec_ids, iou_thresh=0.1)
    metrics_05 = compute_all_metrics(rec_ids, iou_thresh=0.5)
    
    with open(os.path.join(args.output_dir, "metrics_iou_01.json"), "w") as f:
        json.dump(metrics_01, f, indent=4)
    with open(os.path.join(args.output_dir, "metrics_iou_05.json"), "w") as f:
        json.dump(metrics_05, f, indent=4)
        
    # 5. Recording-Level Bootstrap Resampling
    print(f"\nRunning {args.bootstrap_iterations} recording-level bootstrap iterations...")
    
    def bootstrap_eval_fn(sample_recs):
        m = compute_all_metrics(sample_recs, iou_thresh=0.1)
        return {
            "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "micro_f1": m["micro_f1"],
            "false_positives_per_hour": m["false_positives_per_hour"]
        }
        
    ci_summary = recording_level_bootstrap_ci(
        recording_ids=rec_ids,
        eval_fn=bootstrap_eval_fn,
        n_iterations=args.bootstrap_iterations,
        ci_level=0.95,
        seed=args.seed
    )
    
    with open(os.path.join(args.output_dir, "bootstrap_ci.json"), "w") as f:
        json.dump(ci_summary, f, indent=4)
        
    print(f"Final Evaluation Complete. Results saved in {args.output_dir}")
    print(f"IoU >= 0.1 Macro-F1: {metrics_01['macro_f1']:.4f} (95% CI: [{ci_summary['macro_f1']['ci_lower']:.4f}, {ci_summary['macro_f1']['ci_upper']:.4f}])")
    print(f"IoU >= 0.5 Macro-F1: {metrics_05['macro_f1']:.4f}")
    print(f"False Positives / Hour: {metrics_01['false_positives_per_hour']:.2f}")

if __name__ == "__main__":
    main()
