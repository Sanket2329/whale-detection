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
from src.evaluation.threshold import calibrate_event_level_thresholds
from scripts.evaluate import run_continuous_recording_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate Event-Level Thresholds using Hungarian Matching on Validation Data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--val_manifest", type=str, default="data/splits/target_val.csv", help="Validation manifest CSV")
    parser.add_argument("--annotation_dir", type=str, default="data/Greenwich64S2015", help="Directory of ground truth Raven tables")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output JSON path for calibrated thresholds")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference (mps, cuda, cpu)")
    parser.add_argument("--iou_thresh", type=float, default=0.1, help="IoU threshold for bipartite Hungarian matching")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum event duration in seconds")
    parser.add_argument("--max_duration", type=float, default=35.0, help="Maximum event duration in seconds (rejects floor activations)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
    print(f"=== Event-Level Hungarian Threshold Calibration ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation Manifest: {args.val_manifest}")
    print(f"Device: {device}")
    print(f"IoU Threshold: {args.iou_thresh}")
    print(f"Duration bounds: [{args.min_duration}s, {args.max_duration}s]")
    
    # 1. Load Checkpoint & Model
    ckpt = load_checkpoint(args.checkpoint, device="cpu")
    arch = ckpt.get("config", {}).get("model", {}).get("architecture", "crnn")
    
    if "dann" in arch:
        model = WhaleCRNNDANN(num_classes=8)
    else:
        model = WhaleCRNN(num_classes=8)
        
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    # 2. Load Validation Manifest & Annotations
    df_val = pd.read_csv(args.val_manifest)
    print(f"Loaded {len(df_val)} validation recordings.")
    
    all_anns_df = parse_annotations(args.annotation_dir)
    if all_anns_df.empty:
        raise RuntimeError("FATAL: No annotations were parsed from annotation directory.")
        
    gt_by_rec = {}
    for _, row in df_val.iterrows():
        rec_id = row["recording_id"]
        fpath = row["file_path"]
        fname = normalise_recording_filename(fpath)
        
        match = all_anns_df[all_anns_df["wav_file"] == fname]
        gt_events = match.to_dict("records")
        gt_by_rec[rec_id] = {c: [] for c in CLASS_NAMES}
        for g in gt_events:
            c = g["class_name"]
            gt_by_rec[rec_id][c].append({
                "begin_time": float(g["begin_time"]),
                "end_time": float(g["end_time"])
            })
            
    # 3. Compute continuous probability trajectories on validation recordings
    print("\nRunning inference across validation recordings...")
    val_probs_by_rec = {}
    for idx, row in df_val.iterrows():
        rec_id = row["recording_id"]
        fpath = row["file_path"]
        probs = run_continuous_recording_inference(model, fpath, device)
        val_probs_by_rec[rec_id] = probs
        if (idx + 1) % 10 == 0 or (idx + 1) == len(df_val):
            print(f"  Processed {idx + 1}/{len(df_val)} recordings...")
            
    # 4. Calibrate thresholds using bipartite Hungarian matching
    print("\nExecuting Hungarian matching sweep across candidate thresholds...")
    best_th, best_f1, status, diagnostics = calibrate_event_level_thresholds(
        validation_probabilities_by_rec=val_probs_by_rec,
        validation_gt_by_rec=gt_by_rec,
        class_names=CLASS_NAMES,
        iou_thresh=args.iou_thresh,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    print("\n=== Calibration Results ===")
    print(f"{'Class Name':15s} | {'Val GT':6s} | {'Calibrated Th':13s} | {'Event F1':8s} | {'Status'}")
    print("-" * 75)
    for c in CLASS_NAMES:
        gt_cnt = diagnostics[c]["total_gt"]
        th_val = f"{best_th[c]:.2f}" if best_th[c] is not None else "None"
        f1_val = f"{best_f1[c]:.4f}"
        print(f"{c:15s} | {gt_cnt:6d} | {th_val:13s} | {f1_val:8s} | {status[c]}")
        
    output_dict = {
        "calibrated_thresholds": best_th,
        "validation_event_f1": best_f1,
        "calibration_status": status,
        "diagnostics": diagnostics,
        "iou_thresh": args.iou_thresh,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration
    }
    
    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nCalibrated thresholds saved to: {args.output_path}")

if __name__ == "__main__":
    main()

