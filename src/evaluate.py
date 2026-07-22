import os
import argparse
import glob
import csv
import numpy as np

from src.dataset import parse_selection_files, CLASS_MAPPING

def parse_predicted_selections(pred_dir, class_name):
    """Parses all predicted selections for a specific class."""
    pred_path = os.path.join(pred_dir, class_name, "*.predictions.selections.txt")
    files = glob.glob(pred_path)
    
    events_by_wav = {}
    for f_path in files:
        # Expected name: [wav_filename].[class_name].predictions.selections.txt
        # e.g., 20150102-140944.Bm.Ant-A.predictions.selections.txt
        basename = os.path.basename(f_path)
        
        # Reconstruct original wav filename
        # Greenwich: 20150102-140944
        # Casey: 200_2013-12-25_06-00-00
        parts = basename.split(f".{class_name}.predictions.selections.txt")
        wav_name = parts[0] + ".wav"
        
        events_by_wav[wav_name] = []
        
        with open(f_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            if not header:
                continue
                
            try:
                start_samp_idx = header.index("Beg File Samp (samples)")
                end_samp_idx = header.index("End File Samp (samples)")
                conf_idx = header.index("Confidence")
            except ValueError:
                continue
                
            for row in reader:
                if len(row) <= max(start_samp_idx, end_samp_idx, conf_idx):
                    continue
                try:
                    start_samp = int(row[start_samp_idx])
                    end_samp = int(row[end_samp_idx])
                    conf = float(row[conf_idx])
                    
                    events_by_wav[wav_name].append({
                        'start_samp': start_samp,
                        'end_samp': end_samp,
                        'confidence': conf
                    })
                except ValueError:
                    continue
                    
        # Sort by start sample
        events_by_wav[wav_name].sort(key=lambda x: x['start_samp'])
        
    return events_by_wav

def calculate_iou(g_start, g_end, p_start, p_end):
    """Calculates the Intersection-over-Union (IoU) of two intervals."""
    intersection = max(0, min(g_end, p_end) - max(g_start, p_start))
    union = (g_end - g_start) + (p_end - p_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def evaluate_class_events(gt_events_by_wav, pred_events_by_wav, iou_thresh=0.1):
    """Matches predictions to ground truth via greedy IoU and calculates metrics."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Loop over all WAV files present in either ground truth or predictions
    all_wavs = set(gt_events_by_wav.keys()) | set(pred_events_by_wav.keys())
    
    for wav_file in all_wavs:
        gts = gt_events_by_wav.get(wav_file, [])
        preds = pred_events_by_wav.get(wav_file, [])
        
        if not gts:
            total_fp += len(preds)
            continue
        if not preds:
            total_fn += len(gts)
            continue
            
        # Build all candidate matches
        candidates = []
        for g_idx, gt in enumerate(gts):
            for p_idx, pred in enumerate(preds):
                iou = calculate_iou(gt['start_samp'], gt['end_samp'], pred['start_samp'], pred['end_samp'])
                if iou >= iou_thresh:
                    candidates.append((iou, g_idx, p_idx))
                    
        # Sort candidates by IoU descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        matched_gt = set()
        matched_pred = set()
        tp = 0
        
        for iou, g_idx, p_idx in candidates:
            if g_idx not in matched_gt and p_idx not in matched_pred:
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                tp += 1
                
        fp = len(preds) - len(matched_pred)
        fn = len(gts) - len(matched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    # Calculate Precision, Recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return total_tp, total_fp, total_fn, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate Predicted Selections")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth selections")
    parser.add_argument("--gt_name", type=str, required=True, help="Dataset name for ground truth (casey2014 or Greenwich64S2015)")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted selections")
    parser.add_argument("--iou_thresh", type=float, default=0.1, help="IoU threshold for event matching")
    
    args = parser.parse_args()
    
    # 1. Parse ground truth annotations
    print("Parsing ground truth selection files...")
    gt_all = parse_selection_files(args.gt_dir, args.gt_name)
    
    print("\n================ Evaluation Results ================")
    print(f"Dataset: {args.gt_name} | IoU Threshold: {args.iou_thresh}")
    print("-" * 75)
    print(f"{'Class Name':20s} | {'GT':5s} | {'Pred':5s} | {'TP':5s} | {'FP':5s} | {'FN':5s} | {'Precision':9s} | {'Recall':8s} | {'F1-Score':8s}")
    print("-" * 75)
    
    macro_f1s = []
    
    for cls_name, cls_idx in CLASS_MAPPING.items():
        # Filter ground-truth annotations for this specific class
        gt_cls = {}
        for wav_file, anns in gt_all.items():
            gt_cls[wav_file] = [a for a in anns if a['class'] == cls_name]
            
        # Parse predicted annotations for this specific class
        pred_cls = parse_predicted_selections(args.pred_dir, cls_name)
        
        # Calculate counts
        gt_count = sum(len(anns) for anns in gt_cls.values())
        pred_count = sum(len(anns) for anns in pred_cls.values())
        
        tp, fp, fn, prec, rec, f1 = evaluate_class_events(gt_cls, pred_cls, iou_thresh=args.iou_thresh)
        
        print(f"{cls_name:20s} | {gt_count:5d} | {pred_count:5d} | {tp:5d} | {fp:5d} | {fn:5d} | {prec:9.4f} | {rec:8.4f} | {f1:8.4f}")
        
        # Keep track for macro F1
        macro_f1s.append(f1)
        
    print("-" * 75)
    mean_macro_f1 = np.mean(macro_f1s)
    print(f"{'Macro Average F1-Score':59s} : {mean_macro_f1:.4f}")
    
if __name__ == "__main__":
    main()
