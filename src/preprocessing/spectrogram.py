import numpy as np
import librosa

def generate_spectrogram(
    audio: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 64,
    top_db: float = 80.0
) -> np.ndarray:
    """Computes linear STFT magnitude spectrogram, converts to decibels,

    and normalizes into [0.0, 1.0] over specified top_db dynamic range.
    Output shape: (num_freq_bins, num_time_frames) e.g., (129, 118) for 30s @ 250Hz.
    """
    if len(audio) == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)
        
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(spec)
    
    max_val = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    db = librosa.amplitude_to_db(magnitude, ref=max_val, top_db=top_db)
    normalised = (db + top_db) / top_db
    
    return np.clip(normalised, 0.0, 1.0).astype(np.float32)

