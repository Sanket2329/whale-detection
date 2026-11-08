import numpy as np
import scipy.signal as signal
import librosa

TARGET_SR = 250

def preprocess_audio(audio: np.ndarray, original_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Standardised audio preprocessing: 10 Hz 4th-order Butterworth high-pass filter

    followed by high-quality resampling to target_sr (default 250 Hz).
    """
    audio = np.asarray(audio, dtype=np.float32)
    
    # 4th-order Butterworth high-pass filter at 10 Hz
    # Cutoff must be normalized by Nyquist frequency (0.5 * original_sr)
    nyquist = 0.5 * original_sr
    if nyquist > 10.0:
        sos = signal.butter(4, 10.0, btype="highpass", fs=original_sr, output="sos")
        audio = signal.sosfiltfilt(sos, audio)
    
    # Resample if sample rate differs
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr, res_type="soxr_hq")
        
    return audio.astype(np.float32)

def load_and_preprocess_audio(file_path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file using librosa with its native sampling rate,

    then apply standard highpass filtering and rational resampling.
    """
    audio, orig_sr = librosa.load(file_path, sr=None, mono=True)
    processed = preprocess_audio(audio, original_sr=orig_sr, target_sr=target_sr)
    return processed, target_sr

