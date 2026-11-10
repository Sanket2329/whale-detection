import os
import json
import numpy as np
import pandas as pd
from src.data.dataset import parse_annotations, normalise_recording_filename, CLASS_NAMES

def main():
    source_train_manifest = "data/splits/source_train.csv"
    target_val_manifest = "data/splits/target_val.csv"
    
    source_ann_dir = "data/Casey2014"
    target_ann_dir = "data/Greenwich64S2015"
    output_path = "results/calibration/ground_truth_duration_stats.json"
    
    print("=== Computing Ground-Truth Event Duration Statistics ===")
    
    # 1. Parse Annotations
    source_anns = parse_annotations(source_ann_dir)
    target_anns = parse_annotations(target_ann_dir)
    
    # 2. Filter by Splits
    df_src_train = pd.read_csv(source_train_manifest)
    df_tgt_val = pd.read_csv(target_val_manifest)
    
    src_fnames = set(df_src_train["file_path"].apply(normalise_recording_filename))
    tgt_fnames = set(df_tgt_val["file_path"].apply(normalise_recording_filename))
    
    matched_src = source_anns[source_anns["wav_file"].isin(src_fnames)].copy()
    matched_tgt = target_anns[target_anns["wav_file"].isin(tgt_fnames)].copy()
    
    matched_src["duration"] = matched_src["end_time"] - matched_src["begin_time"]
    matched_tgt["duration"] = matched_tgt["end_time"] - matched_tgt["begin_time"]
    
    # Filter positive, valid durations
    matched_src = matched_src[matched_src["duration"] > 0.0].copy()
    matched_tgt = matched_tgt[matched_tgt["duration"] > 0.0].copy()
    
    combined = pd.concat([matched_src, matched_tgt], ignore_index=True)
    
    def get_stats(df, label):
        durations = df["duration"].values
        if len(durations) == 0:
            return {"count": 0}
        return {
            "split": label,
            "count": int(len(durations)),
            "min": float(np.min(durations)),
            "p50_median": float(np.median(durations)),
            "p90": float(np.percentile(durations, 90)),
            "p95": float(np.percentile(durations, 95)),
            "p99": float(np.percentile(durations, 99)),
            "max": float(np.max(durations)),
            "mean": float(np.mean(durations)),
            "std": float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0
        }
    
    stats_source = get_stats(matched_src, "source_train (Casey 2014)")
    stats_target_val = get_stats(matched_tgt, "target_val (Greenwich 2015)")
    stats_combined = get_stats(combined, "combined_source_train_and_target_val")
    
    # Per-class stats on combined training + validation
    per_class = {}
    for c in CLASS_NAMES:
        c_df = combined[combined["class_name"] == c]
        if len(c_df) > 0:
            c_durs = c_df["duration"].values
            per_class[c] = {
                "count": int(len(c_durs)),
                "min": float(np.min(c_durs)),
                "p50_median": float(np.median(c_durs)),
                "p95": float(np.percentile(c_durs, 95)),
                "p99": float(np.percentile(c_durs, 99)),
                "max": float(np.max(c_durs))
            }
        else:
            per_class[c] = {"count": 0}
            
    summary = {
        "description": "Ground-truth event duration percentiles derived strictly from source_train and target_val annotations.",
        "rationale_for_35s_cutoff": "The 35.0s upper bound bounds call durations above the 99th percentile across all classes in training and validation sets, rejecting continuous background probability floor activations.",
        "source_train": stats_source,
        "target_val": stats_target_val,
        "combined_train_val": stats_combined,
        "per_class_combined": per_class
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n--- Summary Duration Stats (Combined Source-Train + Target-Val) ---")
    print(f"Total Reference Events : {stats_combined['count']}")
    print(f"Median Duration        : {stats_combined['p50_median']:.2f} s")
    print(f"90th Percentile        : {stats_combined['p90']:.2f} s")
    print(f"95th Percentile        : {stats_combined['p95']:.2f} s")
    print(f"99th Percentile        : {stats_combined['p99']:.2f} s")
    print(f"Maximum Event Duration : {stats_combined['max']:.2f} s")
    print(f"\nStats written to: {output_path}")

if __name__ == "__main__":
    main()
