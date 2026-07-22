# Underwater Acoustic Sound Event Detection (SED) Pipeline

An automated Sound Event Detection (SED) system for passive acoustic monitoring of Antarctic blue and fin whale vocalizations in underwater hydrophone recordings. Designed for the Antarctic Blue & Fin Whale Acoustic Library (BioDCASE challenge).

This codebase implements a Convolutional Recurrent Neural Network (CRNN) to detect, classify, and localize 8 whale call categories in time, with confidence estimates, and evaluated under cross-site validation to test generalization.

---

## Codebase Architecture

```
whale-detection/
├── data/                      # Dataset directories
│   ├── casey2014/             # Casey site recordings & selections (2014)
│   └── Greenwich64S2015/      # Greenwich site recordings & selections (2015)
├── models/                    # Saved model weight checkpoints
├── reports/                   # Predictions & evaluation reports
│   └── technical_report.md    # Detailed methodology and experiment analysis
├── src/                       # Source scripts
│   ├── __init__.py
│   ├── dataset.py             # Data loading, resampling, high-pass filter, clip balancing
│   ├── model.py               # CRNN SED architecture
│   ├── train.py               # Model training loop & validation
│   ├── infer.py               # Sliding window inference & boundary extraction
│   └── evaluate.py            # IoU-based event matching & metric calculation
├── README.md                  # Project documentation
├── run_pipeline.sh            # E2E pipeline runner (train -> infer -> evaluate)
└── task.md                    # Task tracking checklist
```

---

## Setup Instructions

### 1. Python Environment
This project requires Python 3.11+. Install the required scientific, audio processing, and deep learning dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib librosa soundfile torch torchaudio pypdf
```

### 2. Apple Silicon Acceleration (macOS)
On macOS, PyTorch utilizes Metal Performance Shaders (MPS) for GPU acceleration. To resolve library loader lookup paths for PyTorch on Mac, ensure the environment variable `DYLD_LIBRARY_PATH` is exported:

```bash
export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/torch/lib
export PYTHONPATH=.
```

---

## How to Run the Pipeline

To run the complete pipeline end-to-end (training on `casey2014`, running sliding window inference on `Greenwich64S2015` and calculating metrics), simply execute:

```bash
./run_pipeline.sh
```

---

## Detailed Script Usage

### 1. Training
Train the model on a dataset and validate on another:
```bash
python3 src/train.py \
    --train_dir data/casey2014 \
    --train_name casey2014 \
    --val_dir data/Greenwich64S2015 \
    --val_name Greenwich64S2015 \
    --epochs 5 \
    --batch_size 128 \
    --subsample_train 0.05 \
    --subsample_val 0.1 \
    --save_path models/best_model.pth
```
- `--subsample_train` / `--subsample_val`: Subsamples dataset to speed up training cycles.

### 2. Sliding Window Inference
Detect events in a folder of WAV files:
```bash
python3 src/infer.py \
    --model_path models/best_model.pth \
    --wav_path data/Greenwich64S2015/wav \
    --output_dir reports/predictions/Greenwich64S2015 \
    --threshold 0.5
```
This saves predicted selections in a Raven-compatible format in `reports/predictions/Greenwich64S2015/[class]/`.

### 3. Evaluation
Compare predicted selection tables against ground-truth:
```bash
python3 src/evaluate.py \
    --gt_dir data/Greenwich64S2015 \
    --gt_name Greenwich64S2015 \
    --pred_dir reports/predictions/Greenwich64S2015 \
    --iou_thresh 0.1
```
Calculates event-level Precision, Recall, and F1-score for each of the 8 call categories.
