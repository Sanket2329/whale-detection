import os
import argparse
import glob
import numpy as np
import soundfile as sf
import scipy.signal as signal
import librosa
import torch
import csv

from src.model import WhaleSEDModel
from src.dataset import CLASS_MAPPING, get_device

# Reverse mapping for outputting class names
REV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

def load_and_preprocess_segment(f_handle, start_samp, end_samp, orig_sr, target_sr, clip_samples, target_samples):
    """Loads and preprocesses a single audio segment from a WAV file handle."""
    f_handle.seek(start_samp)
    y = f_handle.read(end_samp - start_samp)
    
    if len(y.shape) > 1:
        y = y[:, 0]
        
    if len(y) < clip_samples:
        y = np.pad(y, (0, clip_samples - len(y)), 'constant')
    else:
        y = y[:clip_samples]
        
    # 10Hz highpass filter
    nyq = 0.5 * orig_sr
    normal_cutoff = 10.0 / nyq
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, y)
    
    # Resample to target_sr
    if orig_sr != target_sr:
        num_target_samples = int(len(y) * target_sr / orig_sr)
        y = signal.resample(y, num_target_samples)
        
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), 'constant')
    else:
        y = y[:target_samples]
        
    # Normalize amplitude
    std = np.std(y)
    if std > 1e-6:
        y = y / std * 0.1
        
    # Compute Spectrogram
    stft = np.abs(librosa.stft(y, n_fft=256, hop_length=64))
    spec = librosa.amplitude_to_db(stft, ref=np.max)
    spec = (spec + 80.0) / 80.0
    spec = np.clip(spec, 0.0, 1.0)
    
    return spec

def run_inference_on_file(model, wav_path, device, target_sr=250, clip_duration=30.0, hop_size_sec=15.0, threshold=0.5):
    """Runs sliding window inference on a single WAV file and extracts detected events."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
        
    with sf.SoundFile(wav_path) as f:
        orig_sr = f.samplerate
        total_samples = len(f)
        duration_sec = total_samples / orig_sr
        
    # Parameters
    hop_length = 64
    clip_samples = int(clip_duration * orig_sr)
    target_samples = int(clip_duration * target_sr)
    
    # Calculate total frames for accumulator
    total_samples_target = int(duration_sec * target_sr)
    total_frames = total_samples_target // hop_length + 1
    
    # Accumulators for overlapping predictions
    num_classes = len(CLASS_MAPPING)
    pred_accumulator = np.zeros((total_frames, num_classes), dtype=np.float32)
    count_accumulator = np.zeros((total_frames, 1), dtype=np.float32)
    
    # Define sliding windows in terms of original samples
    hop_samples = int(hop_size_sec * orig_sr)
    
    window_starts = list(range(0, total_samples - clip_samples + 1, hop_samples))
    if not window_starts:
        window_starts = [0]
        
    # Batch up sliding windows to run quickly
    batch_size = 64
    model.eval()
    
    with sf.SoundFile(wav_path) as f_handle:
        for b_start in range(0, len(window_starts), batch_size):
            b_ends = window_starts[b_start:b_start+batch_size]
            specs_batch = []
            frame_indices = []
            
            for start_samp in b_ends:
                end_samp = start_samp + clip_samples
                spec = load_and_preprocess_segment(f_handle, start_samp, end_samp, orig_sr, target_sr, clip_samples, target_samples)
                specs_batch.append(spec)
                
                # Corresponding target frame index
                target_start_samp = int((start_samp / orig_sr) * target_sr)
                f_idx = target_start_samp // hop_length
                frame_indices.append(f_idx)
                
            # Run model
            batch_tensor = torch.tensor(np.array(specs_batch), dtype=torch.float32).unsqueeze(1).to(device)
            with torch.no_grad():
                probs = model(batch_tensor).cpu().numpy() # shape: (B, 118, num_classes)
                
            # Add to accumulators
            for idx, f_idx in enumerate(frame_indices):
                num_win_frames = probs.shape[1] # usually 118
                end_idx = min(total_frames, f_idx + num_win_frames)
                len_valid = end_idx - f_idx
                
                pred_accumulator[f_idx:end_idx] += probs[idx, :len_valid]
                count_accumulator[f_idx:end_idx] += 1.0
                
    # Compute averaged probabilities
    count_accumulator = np.maximum(count_accumulator, 1.0)
    avg_probs = pred_accumulator / count_accumulator
    
    # Post-processing: apply temporal median filtering to smooth predictions
    # Filter size: 7 frames (approx 1.8 seconds)
    smooth_probs = np.zeros_like(avg_probs)
    for c in range(num_classes):
        # We can use a 1D median filter
        smooth_probs[:, c] = signal.medfilt(avg_probs[:, c], kernel_size=7)
        
    # Extract events
    detected_events = []
    frame_time_step = hop_length / target_sr
    
    for c in range(num_classes):
        cls_name = REV_CLASS_MAPPING[c]
        active = (smooth_probs[:, c] >= threshold).astype(int)
        
        # Find continuous runs of 1s
        is_active = False
        start_frame = 0
        
        for f_idx in range(total_frames):
            if active[f_idx] == 1 and not is_active:
                start_frame = f_idx
                is_active = True
            elif active[f_idx] == 0 and is_active:
                end_frame = f_idx - 1
                is_active = False
                
                # Verify duration > 0.5s to avoid single-frame spurious noise
                duration_frames = end_frame - start_frame + 1
                duration_sec = duration_frames * frame_time_step
                
                if duration_sec >= 0.5:
                    # Calculate event start/end times and samples in original file
                    start_sec = start_frame * frame_time_step
                    end_sec = (end_frame + 1) * frame_time_step
                    
                    # Convert to original samples
                    beg_samp_orig = int(start_sec * orig_sr)
                    end_samp_orig = int(end_sec * orig_sr)
                    
                    # Confidence is mean prob over active region
                    confidence = float(np.mean(avg_probs[start_frame:end_frame+1, c]))
                    
                    detected_events.append({
                        'class': cls_name,
                        'begin_time_sec': start_sec,
                        'end_time_sec': end_sec,
                        'begin_samp': beg_samp_orig,
                        'end_samp': end_samp_orig,
                        'confidence': confidence
                    })
                    
        # Handle trailing active run
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
                
    # Sort events by start time
    detected_events.sort(key=lambda x: x['begin_time_sec'])
    return detected_events

def write_raven_selections(events, output_path, wav_filename):
    """Writes detected events to a Raven-compatible selections text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    headers = [
        "Selection", "View", "Channel", "Begin File", "End File", 
        "Begin Time (s)", "End Time (s)", "Beg File Samp (samples)", "End File Samp (samples)",
        "Confidence"
    ]
    
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

def main():
    parser = argparse.ArgumentParser(description="Run Inference and Save Selections")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weight checkpoint")
    parser.add_argument("--wav_path", type=str, required=True, help="Path to specific WAV file or folder of WAV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predicted selection files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detection")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Model
    model = WhaleSEDModel(num_classes=len(CLASS_MAPPING))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Identify WAV files to process
    if os.path.isdir(args.wav_path):
        wav_files = glob.glob(os.path.join(args.wav_path, "*.wav"))
    else:
        wav_files = [args.wav_path]
        
    print(f"Processing {len(wav_files)} WAV files...")
    
    for w_path in wav_files:
        basename = os.path.basename(w_path)
        print(f"Running detection on: {basename}")
        
        events = run_inference_on_file(model, w_path, device, threshold=args.threshold)
        print(f"  Detected {len(events)} events.")
        
        # Write Raven selections file
        # Output filename matches class-wise structure or a single global predictions selections file.
        # We will write a single unified file containing all classes, indicating the class in the output name or table.
        # To make it fully compatible, we write it to output_dir with name [wav_basename].predictions.selections.txt
        out_name = basename.replace(".wav", ".predictions.selections.txt")
        out_path = os.path.join(args.output_dir, out_name)
        
        # Add class information to events or write class-specific files?
        # Actually, Raven selections support custom columns, but to be safe we can include the class name in the "View" column
        # or write class-specific selections files like AWI2015.Bm.Ant-A..predictions.selections.txt
        # Let's write class-specific selection files, as this mirrors the dataset annotation structure!
        # This makes it super easy to compare directly with the class-specific ground truth files.
        for cls_name in CLASS_MAPPING.keys():
            cls_events = [ev for ev in events if ev['class'] == cls_name]
            if not cls_events:
                # We still write an empty file with headers to maintain complete alignment
                cls_events = []
                
            cls_out_name = f"{basename.replace('.wav', '')}.{cls_name}.predictions.selections.txt"
            cls_out_path = os.path.join(args.output_dir, cls_name, cls_out_name)
            write_raven_selections(cls_events, cls_out_path, basename)
            
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
