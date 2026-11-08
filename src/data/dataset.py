import os
import re
import glob
import pandas as pd
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from src.preprocessing.audio import preprocess_audio
from src.preprocessing.spectrogram import generate_spectrogram

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

CLASS_NAMES = list(CLASS_MAPPING.keys())

def normalise_recording_filename(value) -> str:
    """Resolve Raven filenames to the corresponding WAV filename.

    Example:
    20150102-140944_AWI229-11_SV1057.wav becomes 20150102-140944.wav
    """
    if pd.isna(value):
        return ""
    filename = os.path.basename(str(value).strip().replace("\\", "/"))
    if not filename:
        return ""
    stem, ext = os.path.splitext(filename)
    # Only strip descriptors from the known timestamp-based format (e.g. YYYYMMDD-HHMMSS)
    match = re.match(r"^(\d{8}-\d{6})(?:_.*)?$", stem)
    if match:
        return f"{match.group(1)}.wav"
    if ext.lower() == ".wav":
        return f"{stem}.wav"
    return ""

def parse_annotations(data_dir: str) -> pd.DataFrame:
    """Parses Raven selection table files (.selections.txt) in a dataset folder."""
    files = glob.glob(os.path.join(data_dir, "*.selections.txt"))
    records = []
    
    file_class_patterns = {
        'Ant-A': 'Bm.Ant-A',
        'Ant-B': 'Bm.Ant-B',
        'Ant-Z': 'Bm.Ant-Z',
        'Bm.D': 'Bm.D',
        '20Hz': 'Bp.20Hz',
        '20Plus': 'Bp.20Plus',
        'Downsweep': 'Bp.Downsweep',
        'Fin.DS': 'Bp.Downsweep',
        'Unidentified': 'Unidentified'
    }
    
    for f in files:
        fname = os.path.basename(f)
        call_class = None
        for pattern, cls in file_class_patterns.items():
            if pattern in fname:
                call_class = cls
                break
        if call_class is None:
            continue
            
        try:
            df = pd.read_csv(f, sep='\t')
            for _, row in df.iterrows():
                rec_file = row.get("Begin File")
                if pd.isna(rec_file) or not str(rec_file).strip():
                    rec_file = row.get("File", "")
                clean_file = normalise_recording_filename(rec_file)
                if not clean_file:
                    continue
                    
                # Raven timestamps:
                # "Begin Time (s)" is cumulative relative to the Raven sound sequence.
                # "Beg File Samp (samples)" is the true intra-file sample offset within this specific WAV file.
                # Convert intra-file sample offsets to file-local seconds using the native sample rate.
                beg_samp = float(row.get('Beg File Samp (samples)', 0.0))
                end_samp = float(row.get('End File Samp (samples)', 0.0))
                
                # Greenwich files are 253 Hz, Casey files are 1000 Hz
                native_sr = 253.0 if "greenwich" in data_dir.lower() or "awi" in fname.lower() else 1000.0
                begin_t = beg_samp / native_sr
                end_t = end_samp / native_sr
                
                records.append({
                    'wav_file': clean_file,
                    'begin_time': begin_t,
                    'end_time': end_t,
                    'class_name': call_class,
                    'class_idx': CLASS_MAPPING[call_class]
                })
        except Exception as exc:
            raise RuntimeError(f"Failed to parse annotation file {f}: {exc}") from exc
            
    return pd.DataFrame(records)

class WhaleClipDataset(Dataset):
    """Manifest-driven Dataset for Whale SED.

    - If labelled=True: Loads audio, generates STFT spectrogram, and creates multi-hot frame targets (118, 8).
    - If labelled=False: Target adaptation dataset. Strictly loads audio features without access to annotations.
    """
    def __init__(
        self,
        clips: list[dict],
        labelled: bool = True,
        clip_duration: float = 30.0,
        target_sr: int = 250,
        n_fft: int = 256,
        hop_length: int = 64
    ):
        self.clips = clips
        self.labelled = labelled
        self.clip_duration = clip_duration
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = (int(clip_duration * target_sr) // hop_length) + 1 # 118 frames

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        wav_path = clip["file_path"]
        start_sec = clip["start_sec"]
        
        # Read exact slice using soundfile to optimize I/O
        with sf.SoundFile(wav_path) as sfile:
            orig_sr = sfile.samplerate
            start_frame = int(start_sec * orig_sr)
            frames_to_read = int(self.clip_duration * orig_sr)
            sfile.seek(start_frame)
            audio = sfile.read(frames_to_read, dtype='float32')
            
        # Zero-pad if clip is short
        if len(audio) < frames_to_read:
            audio = np.pad(audio, (0, frames_to_read - len(audio)))
            
        # Standardized preprocessing (10Hz butterworth + soxr_hq resampling)
        processed_audio = preprocess_audio(audio, original_sr=orig_sr, target_sr=self.target_sr)
        expected_samples = int(self.clip_duration * self.target_sr)
        if len(processed_audio) < expected_samples:
            processed_audio = np.pad(processed_audio, (0, expected_samples - len(processed_audio)))
        else:
            processed_audio = processed_audio[:expected_samples]
            
        spec = generate_spectrogram(processed_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).float() # (1, 129, 118)
        
        if not self.labelled:
            return spec_tensor
            
        # Multi-hot ground truth label grid
        labels = np.zeros((self.num_frames, len(CLASS_MAPPING)), dtype=np.float32)
        frame_duration = self.hop_length / self.target_sr
        
        for ann in clip.get("annotations", []):
            c_idx = ann["class_idx"]
            rel_start = max(0.0, ann["begin_time"] - start_sec)
            rel_end = min(self.clip_duration, ann["end_time"] - start_sec)
            
            f_start = int(rel_start / frame_duration)
            f_end = min(self.num_frames, int(np.ceil(rel_end / frame_duration)))
            if f_end > f_start:
                labels[f_start:f_end, c_idx] = 1.0
                
        return spec_tensor, torch.from_numpy(labels).float()

def build_clips_from_manifest(
    manifest_csv: str,
    annotation_dir: str = None,
    labelled: bool = True,
    clip_duration: float = 30.0,
    stride: float = 15.0,
    max_clips: int = None,
    seed: int = 42
) -> list[dict]:
    """Generates 30s audio clips strictly from recordings in manifest_csv.

    - For unlabelled adaptation: generates sliding windows without any annotation dependencies.
    - For labelled data: aligns Raven selections to the recording windows.
    """
    df_manifest = pd.read_csv(manifest_csv)
    clips = []
    
    anns_df = None
    if labelled and annotation_dir:
        anns_df = parse_annotations(annotation_dir)
        
    for _, row in df_manifest.iterrows():
        fpath = row["file_path"]
        rec_id = row["recording_id"]
        fname = normalise_recording_filename(fpath)
        
        if not os.path.exists(fpath):
            continue
            
        try:
            info = sf.info(fpath)
            duration = info.duration
        except Exception:
            continue
            
        rec_anns = []
        if labelled and anns_df is not None and not anns_df.empty:
            match = anns_df[anns_df["wav_file"] == fname]
            rec_anns = match.to_dict("records")
            
        # Generate sliding windows
        starts = np.arange(0, max(0.1, duration - clip_duration + 1), stride)
        for s in starts:
            clip_dict = {
                "file_path": fpath,
                "recording_id": rec_id,
                "start_sec": s,
                "duration": clip_duration
            }
            if labelled:
                clip_anns = [
                    a for a in rec_anns
                    if not (a["end_time"] <= s or a["begin_time"] >= s + clip_duration)
                ]
                clip_dict["annotations"] = clip_anns
            clips.append(clip_dict)
            
    if max_clips and len(clips) > max_clips:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(clips))[:max_clips]
        clips = [clips[i] for i in idx]
        
    return clips

