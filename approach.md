# Technical Approach: Whale Sound Event Detection

This document details the engineering design, technical rationale, and methodology chosen to solve the Underwater Acoustic Sound Event Detection (SED) challenge using the Antarctic Blue & Fin Whale Acoustic Library.

---

## 1. Problem Formulation
The goal is to detect and localize baleen whale vocalizations in continuous, long-duration hydrophone recordings. This is formulated as a **Supervised Sound Event Detection (SED)** task:
*   **Input**: A 30-second audio clip $x(t)$.
*   **Output**: A sequence of frame-level probability vectors $\hat{y}_t \in [0, 1]^C$ for time frames $t = 1, \dots, T$ and classes $C = 8$.
*   **Loss Function**: Frame-level Binary Cross Entropy (BCE) loss to support multi-label predictions (overlapping calls).
*   **Inference**: Sliding window prediction on 1-hour files, aggregation, median filtering, and thresholding to extract events.

---

## 2. Preprocessing & Signal Processing Pipeline

Whale vocalizations (Z-calls, D-calls, 20Hz pulses) are low-frequency sounds (10 to 120 Hz). Standard general audio features (like Mel spectrograms sampled at 16 kHz or 44.1 kHz) are highly suboptimal because they compress low frequencies and introduce massive data redundancy.

```
Raw Audio (1000 Hz / 253 Hz)
  │
  ▼
High-pass filter (10 Hz Butterworth)
  │
  ▼
Resample to 250 Hz (soxr)
  │
  ▼
Linear Magnitude STFT (n_fft=256, hop=64)
  │
  ▼
Log-Scale dB Amplitude Normalization
  │
  ▼
Normalized Spectrogram: (1, 129, 118)
```

1.  **High-Pass Filtering**: We apply a 4th-order Butterworth high-pass filter with a cutoff frequency of 10 Hz. This removes ocean hydrostatic pressure fluctuations, wave motion noise, and sensor self-noise which dominate the sub-10 Hz band.
2.  **Rational Resampling**: Casey files are sampled at 1000 Hz; Greenwich files are sampled at 253 Hz. We resample both to a common target of **250 Hz** using the C-based `soxr` library. This is crucial for cross-site generalization, as it ensures the network sees identical spectrogram structures regardless of source site.
3.  **Linear Frequency Spectrogram**: We compute a Short-Time Fourier Transform (STFT) with `n_fft = 256` and `hop_length = 64`. 
    - At 250 Hz, `n_fft = 256` gives a frequency bin width of $250 / 256 \approx 0.98$ Hz. This provides high resolution to distinguish very close frequency calls (e.g. 15 Hz vs 20 Hz).
    - `hop_length = 64` translates to a frame step size of $64 / 250 = 0.256$ seconds, providing high temporal resolution.
4.  **Decibel Scaling**: Magnitude spectrograms are converted to decibels: $dB = 20 \log_{10}(X)$. The values are normalized to $[0, 1]$ relative to an 80 dB dynamic range.

---

## 3. CNN-RNN (CRNN) Model Architecture

A CRNN architecture is highly suited for SED because the CNN layers learn local time-frequency features (like spectral sweeps and harmonics), while the RNN layers learn sequential constraints and event durations.

*   **Frequency-Only Pooling**: To avoid temporal alignment mismatches, we use 2D CNN layers with max-pooling applied *only along the frequency axis* (`MaxPool2d(2, 1)`). This reduces the frequency height from 129 to 4 bins after 5 layers, while keeping the time frame dimension constant at 118 frames.
*   **Temporal Context**: The output of the CNN is reshaped to `(batch_size, 118, 256 * 4) = (batch_size, 118, 1024)` and fed into a 2-layer Bidirectional GRU with 128 hidden units. The bidirectional nature allows the model to look forward and backward in time, resolving start/end boundaries more accurately.
*   **Sigmoid Classification**: A fully connected layer projects the GRU output at each frame to 8 output logits, followed by independent Sigmoids.

---

## 4. Addressing Class Imbalance & Mining Negatives

Whale calls occupy less than 10% of the continuous audio. A naive training set would contain mostly silence, leading the model to always predict zero.
1.  **Positive Centering**: For each ground-truth annotation, we extract a 30-second clip centered at the call's midpoint.
2.  **Vectorized Negative Mining**: We compute a binary mask of all annotated frames in a WAV file. Using numpy difference and gap-finding algorithms, we identify continuous unannotated segments of at least 30 seconds. We sample negative clips from these segments at a 1:1 ratio with positive clips to balance the training set.

---

## 5. Unsupervised Domain Adaptation (DANN)
To handle the cross-site domain shift (e.g., Casey to Greenwich), we implement a **Domain-Adversarial Neural Network (DANN)** using a **Gradient Reversal Layer (GRL)**. 
- During training, a secondary Domain Classifier branch attempts to classify the origin site of the audio.
- The GRL negates the gradients during the backward pass, forcing the CNN feature extractor to learn representations that are site-invariant, significantly improving cross-site generalization.

---

## 6. Inference Acceleration & ONNX Export
For real-time deployment and scaling to multi-year archives, pure PyTorch inference is too slow.
- The model graph is exported to **ONNX (Open Neural Network Exchange)**.
- Benchmarks demonstrate an approximate **fourfold (4x) speedup** during inference using ONNX Runtime compared to native PyTorch, providing massive cost savings for large-scale acoustic analysis.

---

## 7. Post-Processing & Event Boundary Extraction

During inference on a 1-hour WAV file:
1.  **Sliding Window**: A 30-second window slides with 50% overlap (15-second step).
2.  **Overlap Aggregation**: Probabilities in overlapping regions are averaged.
3.  **Median Filtering**: A 1D median filter of width 7 frames (~1.8 seconds) is applied to each class prediction. This smooths out temporary drops (dropout gaps) and removes high-frequency spurious noise.
4.  **Thresholding**: A threshold (default $P \ge 0.5$) is applied to identify contiguous active regions. Regions with a duration of at least 0.5 seconds are registered as events.
5.  **Confidence Estimation**: The event's confidence score is the average predicted probability of the class over the active event frames.
