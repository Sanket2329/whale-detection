import os
import glob
import hashlib
import numpy as np
import pandas as pd

def generate_checksum(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read first 64KB for speed while maintaining file differentiation
        hasher.update(f.read(65536))
    return hasher.hexdigest()

def create_recording_level_manifests(
    casey_dir: str = "data/casey2014",
    greenwich_dir: str = "data/Greenwich64S2015",
    output_dir: str = "data/splits",
    val_ratio: float = 0.2,
    seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 1. Source (Casey 2014): Split recordings into source_train and source_val
    casey_wavs = sorted(glob.glob(os.path.join(casey_dir, "wav", "*.wav")))
    rng.shuffle(casey_wavs)
    
    n_casey_val = max(1, int(len(casey_wavs) * val_ratio))
    casey_val_wavs = casey_wavs[:n_casey_val]
    casey_train_wavs = casey_wavs[n_casey_val:]
    
    source_train_rows = [{
        "recording_id": os.path.splitext(os.path.basename(p))[0],
        "file_path": p,
        "site": "casey",
        "year": 2014,
        "split": "source_train",
        "checksum": generate_checksum(p)
    } for p in casey_train_wavs]
    
    source_val_rows = [{
        "recording_id": os.path.splitext(os.path.basename(p))[0],
        "file_path": p,
        "site": "casey",
        "year": 2014,
        "split": "source_val",
        "checksum": generate_checksum(p)
    } for p in casey_val_wavs]

    pd.DataFrame(source_train_rows).to_csv(os.path.join(output_dir, "source_train.csv"), index=False)
    pd.DataFrame(source_val_rows).to_csv(os.path.join(output_dir, "source_val.csv"), index=False)

    # 2. Target (Greenwich 2015): Strictly partition recordings into adaptation, validation, and test
    greenwich_wavs = sorted(glob.glob(os.path.join(greenwich_dir, "wav", "*.wav")))
    rng.shuffle(greenwich_wavs)
    
    # 50% adaptation (unlabelled), 20% target validation, 30% final held-out test
    n_adapt = int(len(greenwich_wavs) * 0.5)
    n_val = int(len(greenwich_wavs) * 0.2)
    
    adapt_wavs = greenwich_wavs[:n_adapt]
    val_wavs = greenwich_wavs[n_adapt:n_adapt + n_val]
    test_wavs = greenwich_wavs[n_adapt + n_val:]

    # Safeguard: target_adaptation manifest contains NO annotation fields
    adapt_rows = [{
        "recording_id": os.path.splitext(os.path.basename(p))[0],
        "file_path": p,
        "site": "greenwich",
        "year": 2015,
        "split": "target_adaptation",
        "checksum": generate_checksum(p)
    } for p in adapt_wavs]

    target_val_rows = [{
        "recording_id": os.path.splitext(os.path.basename(p))[0],
        "file_path": p,
        "site": "greenwich",
        "year": 2015,
        "split": "target_val",
        "checksum": generate_checksum(p)
    } for p in val_wavs]

    target_test_rows = [{
        "recording_id": os.path.splitext(os.path.basename(p))[0],
        "file_path": p,
        "site": "greenwich",
        "year": 2015,
        "split": "target_test",
        "checksum": generate_checksum(p)
    } for p in test_wavs]

    pd.DataFrame(adapt_rows).to_csv(os.path.join(output_dir, "target_adaptation.csv"), index=False)
    pd.DataFrame(target_val_rows).to_csv(os.path.join(output_dir, "target_val.csv"), index=False)
    pd.DataFrame(target_test_rows).to_csv(os.path.join(output_dir, "target_test.csv"), index=False)

    print(f"Generated recording-level splits in {output_dir}:")
    print(f"  source_train: {len(source_train_rows)} recordings")
    print(f"  source_val:   {len(source_val_rows)} recordings")
    print(f"  target_adapt: {len(adapt_rows)} recordings (unlabelled)")
    print(f"  target_val:   {len(target_val_rows)} recordings")
    print(f"  target_test:  {len(target_test_rows)} recordings")

if __name__ == "__main__":
    create_recording_level_manifests()

