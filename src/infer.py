import os
import argparse
import glob
import numpy as np
import soundfile as sf
import scipy.signal as signal
import librosa
import torch
import csv
import json

from src.model import WhaleSEDModel
from src.dataset import CLASS_MAPPING, get_device

# Reverse mapping for outputting class names
REV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

def run_inference_on_file(model, wav_path, device, target_sr=250, clip_duration=30.0, hop_size_sec=15.0, thresholds=0.5):
    """Runs sliding window inference on a single WAV file and extracts detected events."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
        
    with sf.SoundFile(wav_path) as f:
        orig_sr = f.samplerate
        total_samples = len(f)
        duration_sec = total_samples / orig_sr
        y_full = f.read()
        
    if len(y_full.shape) > 1:
        y_full = y_full[:, 0]
        
    # 10Hz highpass filter on full audio
    nyq = 0.5 * orig_sr
    normal_cutoff = 10.0 / nyq
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    y_full = signal.filtfilt(b, a, y_full)
    
    # Resample full audio to target_sr
    if orig_sr != target_sr:
        y_full = librosa.resample(y_full, orig_sr=orig_sr, target_sr=target_sr, res_type='soxr_qq')
        
    # Segment parameters in target sampling rate
    clip_samples_target = int(clip_duration * target_sr)
    hop_samples_target = int(hop_size_sec * target_sr)
    total_samples_target = len(y_full)
    
    # Calculate STFT once for the entire file
    hop_length = 64
    stft_full = np.abs(librosa.stft(y_full, n_fft=256, hop_length=64))
    
    total_frames = stft_full.shape[1]
    
    # Accumulators for overlapping predictions
    num_classes = len(CLASS_MAPPING)
    pred_accumulator = np.zeros((total_frames, num_classes), dtype=np.float32)
    count_accumulator = np.zeros((total_frames, 1), dtype=np.float32)
    
    # Define sliding windows in target samples
    window_starts = list(range(0, total_samples_target - clip_samples_target + 1, hop_samples_target))
    if not window_starts:
        window_starts = [0]
        
    batch_size = 256
    model.eval()
    
    # We map target sample starts to target frame starts
    win_frames = 118 # expected frame length for 30s at 250Hz, n_fft=256, hop=64
    
    for b_start in range(0, len(window_starts), batch_size):
        b_ends = window_starts[b_start:b_start+batch_size]
        specs_batch = []
        frame_indices = []
        
        for start_samp in b_ends:
            f_idx = start_samp // hop_length
            end_f_idx = f_idx + win_frames
            
            # Slice the full spectrogram
            spec_slice = stft_full[:, f_idx:end_f_idx]
            
            # Pad if slice is shorter than win_frames
            if spec_slice.shape[1] < win_frames:
                spec_slice = np.pad(spec_slice, ((0, 0), (0, win_frames - spec_slice.shape[1])), 'constant')
                
            # Decibel conversion relative to max of the slice
            spec_db = librosa.amplitude_to_db(spec_slice, ref=np.max)
            spec_db = (spec_db + 80.0) / 80.0
            spec_db = np.clip(spec_db, 0.0, 1.0)
            
            specs_batch.append(spec_db)
            frame_indices.append(f_idx)
            
        # Run model
        batch_tensor = torch.tensor(np.array(specs_batch), dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            probs = model(batch_tensor).cpu().numpy()
            
        # Add to accumulators
        for idx, f_idx in enumerate(frame_indices):
            num_win_frames = probs.shape[1]
            end_idx = min(total_frames, f_idx + num_win_frames)
            len_valid = end_idx - f_idx
            
            pred_accumulator[f_idx:end_idx] += probs[idx, :len_valid]
            count_accumulator[f_idx:end_idx] += 1.0
            
    # Compute averaged probabilities
    count_accumulator = np.maximum(count_accumulator, 1.0)
    avg_probs = pred_accumulator / count_accumulator
    
    # Post-processing: apply temporal median filtering to smooth predictions
    smooth_probs = np.zeros_like(avg_probs)
    for c in range(num_classes):
        smooth_probs[:, c] = signal.medfilt(avg_probs[:, c], kernel_size=7)
        
    # Extract events
    detected_events = []
    frame_time_step = hop_length / target_sr
    
    for c in range(num_classes):
        cls_name = REV_CLASS_MAPPING[c]
        if isinstance(thresholds, dict):
            threshold = thresholds.get(cls_name, 0.5)
        else:
            threshold = thresholds
            
        active = (smooth_probs[:, c] >= threshold).astype(int)
        
        is_active = False
        start_frame = 0
        
        for f_idx in range(total_frames):
            if active[f_idx] == 1 and not is_active:
                start_frame = f_idx
                is_active = True
            elif active[f_idx] == 0 and is_active:
                end_frame = f_idx - 1
                is_active = False
                
                duration_frames = end_frame - start_frame + 1
                duration_sec = duration_frames * frame_time_step
                
                if duration_sec >= 0.5:
                    start_sec = start_frame * frame_time_step
                    end_sec = (end_frame + 1) * frame_time_step
                    beg_samp_orig = int(start_sec * orig_sr)
                    end_samp_orig = int(end_sec * orig_sr)
                    confidence = float(np.mean(avg_probs[start_frame:end_frame+1, c]))
                    
                    detected_events.append({
                        'class': cls_name,
                        'begin_time_sec': start_sec,
                        'end_time_sec': end_sec,
                        'begin_samp': beg_samp_orig,
                        'end_samp': end_samp_orig,
                        'confidence': confidence
                    })
                    
        if is_active:
            end_frame = total_frames - 1
            duration_sec = (end_frame - start_frame + 1) * frame_time_step
            if duration_sec >= 0.5:
                start_sec = start_frame * frame_time_step
                end_sec = (end_frame + 1) * frame_time_step
                beg_samp_orig = int(start_sec * orig_sr)
                end_samp_orig = int(end_sec * orig_sr)
                confidence = float(np.mean(avg_probs[start_frame:end_frame+1, c]))
                detected_events.append({
                    'class': cls_name,
                    'begin_time_sec': start_sec,
                    'end_time_sec': end_sec,
                    'begin_samp': beg_samp_orig,
                    'end_samp': end_samp_orig,
                    'confidence': confidence
                })
                
    detected_events.sort(key=lambda x: x['begin_time_sec'])
    return detected_events

def write_raven_selections(events, output_path, wav_filename):
    """Writes detected events to a Raven-compatible selections text file."""
    import time
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    headers = [
        "Selection", "View", "Channel", "Begin File", "End File", 
        "Begin Time (s)", "End Time (s)", "Beg File Samp (samples)", "End File Samp (samples)",
        "Confidence"
    ]
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(headers)
                
                for idx, ev in enumerate(events, 1):
                    writer.writerow([
                        idx,
                        "Spectrogram 1",
                        1,
                        wav_filename,
                        wav_filename,
                        f"{ev['begin_time_sec']:.3f}",
                        f"{ev['end_time_sec']:.3f}",
                        ev['begin_samp'],
                        ev['end_samp'],
                        f"{ev['confidence']:.4f}"
                    ])
            break # success
        except (TimeoutError, OSError) as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(0.2)

def main():
    parser = argparse.ArgumentParser(description="Run Inference and Save Selections")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weight checkpoint")
    parser.add_argument("--wav_path", type=str, required=True, help="Path to specific WAV file or folder of WAV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predicted selection files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of processed files")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Model
    model = WhaleSEDModel(num_classes=len(CLASS_MAPPING))
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    # Try to load adaptive thresholds from the JSON file next to the model path
    thresholds = args.threshold
    thresholds_json_path = args.model_path.replace(".pth", "_thresholds.json")
    if os.path.exists(thresholds_json_path):
        try:
            with open(thresholds_json_path, 'r', encoding='utf-8') as f:
                thresholds = json.load(f)
            print(f"Loaded class-specific optimal thresholds from: {thresholds_json_path}")
            print(f"Thresholds: {thresholds}")
        except Exception as e:
            print(f"Warning: failed to load thresholds file: {e}. Falling back to default threshold {args.threshold}")
    else:
        print(f"No threshold JSON found at {thresholds_json_path}, using global threshold {thresholds}")
    
    # Identify WAV files to process
    if os.path.isdir(args.wav_path):
        wav_files = sorted(glob.glob(os.path.join(args.wav_path, "*.wav")))
    else:
        wav_files = [args.wav_path]
        
    if args.limit is not None:
        wav_files = wav_files[:args.limit]
        print(f"Limiting execution to the first {args.limit} WAV files.")
        
    print(f"Processing {len(wav_files)} WAV files...")
    
    for w_path in wav_files:
        basename = os.path.basename(w_path)
        print(f"Running detection on: {basename}")
        
        events = run_inference_on_file(model, w_path, device, thresholds=thresholds)
        print(f"  Detected {len(events)} events.")
        
        for cls_name in CLASS_MAPPING.keys():
            cls_events = [ev for ev in events if ev['class'] == cls_name]
            cls_out_name = f"{basename.replace('.wav', '')}.{cls_name}.predictions.selections.txt"
            cls_out_path = os.path.join(args.output_dir, cls_name, cls_out_name)
            write_raven_selections(cls_events, cls_out_path, basename)
        
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
