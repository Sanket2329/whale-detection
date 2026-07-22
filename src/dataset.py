import os
import glob
import csv
import numpy as np
import soundfile as sf
import scipy.signal as signal
import librosa
import torch
from torch.utils.data import Dataset

# Mapping from whale call category to standard index
CLASS_MAPPING = {
    'Bm.Ant-A': 0,
    'Bm.Ant-B': 1,
    'Bm.Ant-Z': 2,
    'Bm.D': 3,
    'Bp.20Hz': 4,
    'Bp.20Plus': 5,
    'Bp.Downsweep': 6,
    'Unidentified': 7
}

def get_class_from_selection_file(filename):
    """Maps the selection file name to a standard class name."""
    filename = filename.lower()
    if 'ant-a' in filename:
        return 'Bm.Ant-A'
    elif 'ant-b' in filename:
        return 'Bm.Ant-B'
    elif 'ant-z' in filename:
        return 'Bm.Ant-Z'
    elif '.d.' in filename:
        return 'Bm.D'
    elif '20hz' in filename:
        return 'Bp.20Hz'
    elif '20plus' in filename:
        return 'Bp.20Plus'
    elif 'downsweep' in filename or '.ds.' in filename:
        return 'Bp.Downsweep'
    elif 'unidentified' in filename:
        return 'Unidentified'
    else:
        return None

def resolve_wav_filename(selection_file_val, dataset_name, wav_basenames):
    """Maps a filename from selection file to the actual wav file name."""
    if not selection_file_val.endswith('.wav'):
        selection_file_val += '.wav'
        
    if selection_file_val in wav_basenames:
        return selection_file_val
        
    if dataset_name == 'Greenwich64S2015':
        # Selection: 20150102-140944_AWI229-11_SV1057.wav -> 20150102-140944.wav
        prefix = selection_file_val.split('_')[0]
        mapped = prefix + '.wav'
        if mapped in wav_basenames:
            return mapped
            
    return None

def parse_selection_files(dataset_dir, dataset_name):
    """Parses selection text files and aggregates annotations by WAV file."""
    wav_dir = os.path.join(dataset_dir, 'wav')
    if not os.path.exists(wav_dir):
        raise FileNotFoundError(f"wav directory not found in {dataset_dir}")
        
    wav_files = os.listdir(wav_dir)
    wav_basenames = {f for f in wav_files if f.endswith('.wav')}
    
    # Store annotations grouped by resolved wav filename
    # annotations_by_wav = { wav_filename: [ {class, start_samp, end_samp, start_sec, end_sec} ] }
    annotations_by_wav = {}
    for wav_file in wav_basenames:
        annotations_by_wav[wav_file] = []
        
    selection_files = glob.glob(os.path.join(dataset_dir, '*.selections.txt'))
    
    for file_path in selection_files:
        filename = os.path.basename(file_path)
        cls_name = get_class_from_selection_file(filename)
        if not cls_name:
            continue
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            if not header:
                continue
                
            try:
                file_idx = header.index("Begin File")
                start_samp_idx = header.index("Beg File Samp (samples)")
                end_samp_idx = header.index("End File Samp (samples)")
                start_sec_idx = header.index("Begin Time (s)")
                end_sec_idx = header.index("End Time (s)")
            except ValueError as e:
                print(f"Skipping {filename}: missing required columns ({e})")
                continue
                
            for row in reader:
                if len(row) <= max(file_idx, start_samp_idx, end_samp_idx, start_sec_idx, end_sec_idx):
                    continue
                try:
                    orig_file = row[file_idx]
                    resolved_file = resolve_wav_filename(orig_file, dataset_name, wav_basenames)
                    if not resolved_file:
                        continue
                        
                    start_samp = int(row[start_samp_idx])
                    end_samp = int(row[end_samp_idx])
                    start_sec = float(row[start_sec_idx])
                    end_sec = float(row[end_sec_idx])
                    
                    annotations_by_wav[resolved_file].append({
                        'class': cls_name,
                        'start_samp': start_samp,
                        'end_samp': end_samp,
                        'start_sec': start_sec,
                        'end_sec': end_sec
                    })
                except ValueError:
                    continue
                    
    # Sort annotations for each wav file by start sample
    for wav_file in annotations_by_wav:
        annotations_by_wav[wav_file].sort(key=lambda x: x['start_samp'])
        
    return annotations_by_wav

def build_dataset_clips(dataset_dir, dataset_name, annotations_by_wav, clip_duration=30.0, neg_multiplier=1.0):
    """
    Constructs positive and negative 30-second clip definitions.
    Returns: List of dicts: {dataset_dir, dataset_name, wav_file, start_samp, end_samp, annotations}
    """
    wav_dir = os.path.join(dataset_dir, 'wav')
    clips = []
    
    for wav_file, annotations in annotations_by_wav.items():
        wav_path = os.path.join(wav_dir, wav_file)
        if not os.path.exists(wav_path):
            continue
            
        # Read WAV properties
        with sf.SoundFile(wav_path) as f:
            sr_orig = f.samplerate
            total_samples = len(f)
            
        clip_samples = int(clip_duration * sr_orig)
        
        # Positive clips: center around annotations
        positive_clips = []
        for ann in annotations:
            center = (ann['start_samp'] + ann['end_samp']) // 2
            clip_start = center - clip_samples // 2
            
            # Keep clip in bounds
            if clip_start < 0:
                clip_start = 0
            if clip_start + clip_samples > total_samples:
                clip_start = max(0, total_samples - clip_samples)
                
            clip_end = clip_start + clip_samples
            
            # Find all annotations that fall in this clip
            clip_anns = []
            for a in annotations:
                # Event overlaps with clip if: start of event < end of clip and end of event > start of clip
                if a['start_samp'] < clip_end and a['end_samp'] > clip_start:
                    clip_anns.append(a)
                    
            positive_clips.append({
                'dataset_dir': dataset_dir,
                'dataset_name': dataset_name,
                'wav_file': wav_file,
                'start_samp': clip_start,
                'end_samp': clip_end,
                'annotations': clip_anns,
                'is_positive': True
            })
            
        # Deduplicate positive clips that have identical start_samp
        unique_pos_clips = {}
        for c in positive_clips:
            unique_pos_clips[c['start_samp']] = c
        positive_clips = list(unique_pos_clips.values())
        clips.extend(positive_clips)
        
        # Negative clips: sample segments with no overlapping annotations
        if not positive_clips:
            # If no positives in this file, we can sample a couple of negatives
            num_negatives = int(2 * neg_multiplier)
        else:
            num_negatives = int(len(positive_clips) * neg_multiplier)
            
        # Find all unannotated intervals in the file
        annotated_mask = np.zeros(total_samples, dtype=bool)
        for ann in annotations:
            # Pad annotation intervals slightly to avoid edge transitions
            pad = int(1.0 * sr_orig)
            start = max(0, ann['start_samp'] - pad)
            end = min(total_samples, ann['end_samp'] + pad)
            annotated_mask[start:end] = True
            
        # Find continuous unannotated segments using numpy (vectorized and extremely fast)
        unannotated_runs = []
        if not np.any(annotated_mask):
            unannotated_runs.append((0, total_samples))
        else:
            true_idx = np.where(annotated_mask)[0]
            # Check leading gap
            first_true = true_idx[0]
            if first_true >= clip_samples:
                unannotated_runs.append((0, first_true))
                
            # Gaps between annotations
            diff_idx = np.diff(true_idx)
            gap_locs = np.where(diff_idx > clip_samples)[0]
            for idx in gap_locs:
                unannotated_runs.append((true_idx[idx] + 1, true_idx[idx + 1]))
                
            # Check trailing gap
            last_true = true_idx[-1]
            if total_samples - last_true - 1 >= clip_samples:
                unannotated_runs.append((last_true + 1, total_samples))
            
        # Sample negative clips from unannotated segments
        sampled_negs = 0
        if unannotated_runs and num_negatives > 0:
            # We will try to sample uniformly from the runs
            for _ in range(num_negatives * 2): # Try up to 2x budget in case of overlap checks
                if sampled_negs >= num_negatives:
                    break
                # Select a random run
                run_idx = np.random.randint(len(unannotated_runs))
                r_start, r_end = unannotated_runs[run_idx]
                
                # Pick random start within run
                neg_start = np.random.randint(r_start, r_end - clip_samples + 1)
                neg_end = neg_start + clip_samples
                
                clips.append({
                    'dataset_dir': dataset_dir,
                    'dataset_name': dataset_name,
                    'wav_file': wav_file,
                    'start_samp': neg_start,
                    'end_samp': neg_end,
                    'annotations': [],
                    'is_positive': False
                })
                sampled_negs += 1
                
    return clips

class WhaleDataset(Dataset):
    """PyTorch Dataset to load audio clips and generate spectrograms and label grids on the fly."""
    def __init__(self, clip_definitions, target_sr=250, clip_duration=30.0, augment=False):
        self.clips = clip_definitions
        self.target_sr = target_sr
        self.clip_duration = clip_duration
        self.augment = augment
        self.target_samples = int(clip_duration * target_sr)
        
        # Spectrogram parameters
        self.n_fft = 256
        self.hop_length = 64
        self.num_frames = self.target_samples // self.hop_length + 1 # 118 frames
        self.num_classes = len(CLASS_MAPPING)

    def __len__(self):
        return len(self.clips)

    def apply_highpass_filter(self, y, sr, cutoff=10.0):
        """Applies a 10Hz highpass Butterworth filter to clear hydrostatic noise."""
        nyq = 0.5 * sr
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, y)

    def resample_audio(self, y, orig_sr):
        """Resamples audio from orig_sr to target_sr using librosa (high-performance soxr)."""
        if orig_sr == self.target_sr:
            return y
        return librosa.resample(y, orig_sr=orig_sr, target_sr=self.target_sr)

    def apply_augmentation(self, y):
        """Applies dynamic audio augmentations (noise injection, scaling)."""
        # 1. Random gain scaling
        gain = np.random.uniform(0.8, 1.2)
        y = y * gain
        
        # 2. Add soft white noise
        if np.random.rand() < 0.3:
            noise_std = np.random.uniform(0.001, 0.005)
            y = y + np.random.normal(0, noise_std, len(y))
            
        return y

    def __getitem__(self, idx):
        c_def = self.clips[idx]
        wav_path = os.path.join(c_def['dataset_dir'], 'wav', c_def['wav_file'])
        
        # 1. Load audio segment directly
        try:
            with sf.SoundFile(wav_path) as f:
                orig_sr = f.samplerate
                f.seek(c_def['start_samp'])
                expected_len = c_def['end_samp'] - c_def['start_samp']
                y = f.read(expected_len)
        except Exception as e:
            # Return dummy zeros in case of load failure
            print(f"Error loading {wav_path} at segment {c_def['start_samp']}:{c_def['end_samp']}: {e}")
            y = np.zeros(int(self.clip_duration * 1000)) # assume standard length
            orig_sr = 1000
            
        # Ensure single channel and correct length
        if len(y.shape) > 1:
            y = y[:, 0]
            
        target_orig_len = int(self.clip_duration * orig_sr)
        if len(y) < target_orig_len:
            y = np.pad(y, (0, target_orig_len - len(y)), 'constant')
        elif len(y) > target_orig_len:
            y = y[:target_orig_len]
            
        # 2. Apply filtering
        y = self.apply_highpass_filter(y, orig_sr, cutoff=10.0)
        
        # 3. Resample to target sample rate
        y = self.resample_audio(y, orig_sr)
        
        # Ensure length matches target_samples
        if len(y) < self.target_samples:
            y = np.pad(y, (0, self.target_samples - len(y)), 'constant')
        else:
            y = y[:self.target_samples]
            
        # 4. Data Augmentation
        if self.augment:
            y = self.apply_augmentation(y)
            
        # Normalize audio amplitude
        std = np.std(y)
        if std > 1e-6:
            y = y / std * 0.1
            
        # 5. Compute Spectrogram
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        # Shape: (129, num_frames) -> e.g., (129, 118)
        # Convert to decibels (log scale)
        spec = librosa.amplitude_to_db(stft, ref=np.max)
        
        # Normalize spectrogram values to roughly [0, 1]
        spec = (spec + 80.0) / 80.0
        spec = np.clip(spec, 0.0, 1.0)
        
        # Convert to PyTorch tensor with shape (1, freq, time)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        
        # 6. Generate Frame-Level Label Grid
        label_grid = np.zeros((self.num_frames, self.num_classes), dtype=np.float32)
        
        # Clip coordinates
        clip_start_sec = c_def['start_samp'] / orig_sr
        clip_end_sec = c_def['end_samp'] / orig_sr
        
        # Time step per frame in seconds (at target sample rate)
        frame_time_step = self.hop_length / self.target_sr
        
        for ann in c_def['annotations']:
            cls_idx = CLASS_MAPPING.get(ann['class'])
            if cls_idx is None:
                continue
                
            # Event start and end time relative to the clip start
            ann_start_rel = ann['start_samp'] / orig_sr - clip_start_sec
            ann_end_rel = ann['end_samp'] / orig_sr - clip_start_sec
            
            # Map time to frame indices
            frame_start = int(max(0.0, ann_start_rel) / frame_time_step)
            frame_end = int(min(self.clip_duration, ann_end_rel) / frame_time_step)
            
            # Set label in range
            frame_start = min(self.num_frames - 1, frame_start)
            frame_end = min(self.num_frames - 1, frame_end)
            
            if frame_start <= frame_end:
                label_grid[frame_start:frame_end+1, cls_idx] = 1.0
                
        label_tensor = torch.tensor(label_grid, dtype=torch.float32)
        
        return spec_tensor, label_tensor

def get_device():
    """Returns the most appropriate PyTorch device available."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
