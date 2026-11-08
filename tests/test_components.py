import unittest
import numpy as np
import torch
from src.preprocessing.audio import preprocess_audio
from src.preprocessing.spectrogram import generate_spectrogram
from src.models.crnn import WhaleCRNN, WhaleCRNNDANN
from src.models.dann import grad_reverse
from src.training.losses import FocalLoss
from src.evaluation.threshold import find_best_thresholds_per_class

class TestWhalePipeline(unittest.TestCase):
    def test_audio_preprocessing_shape_and_sr(self):
        # 1 second of 1000 Hz signal
        audio_1000 = np.sin(2 * np.pi * 30 * np.linspace(0, 1, 1000)).astype(np.float32)
        processed = preprocess_audio(audio_1000, original_sr=1000, target_sr=250)
        # Should be resampled to 250 samples
        self.assertEqual(len(processed), 250)

    def test_spectrogram_dimensions_and_bounds(self):
        # 30 seconds at 250 Hz = 7500 samples
        audio_30s = np.random.randn(7500).astype(np.float32)
        spec = generate_spectrogram(audio_30s, n_fft=256, hop_length=64)
        # Freq bins: 256//2 + 1 = 129
        # Frames: (7500 // 64) + 1 = 118
        self.assertEqual(spec.shape, (129, 118))
        self.assertTrue(np.all(spec >= 0.0) and np.all(spec <= 1.0))

    def test_crnn_forward_shape(self):
        model = WhaleCRNN(num_classes=8)
        dummy_input = torch.randn(2, 1, 129, 118)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (2, 118, 8))

    def test_gradient_reversal_layer(self):
        x = torch.tensor([2.0, -3.0], requires_grad=True)
        reversed_x = grad_reverse(x, alpha=1.5)
        loss = reversed_x.sum()
        loss.backward()
        # Incoming grad is [1.0, 1.0], after reversal with alpha=1.5 it must be [-1.5, -1.5]
        expected_grad = torch.tensor([-1.5, -1.5])
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_focal_loss_finite(self):
        criterion = FocalLoss()
        logits = torch.randn(4, 118, 8, requires_grad=True)
        targets = torch.randint(0, 2, (4, 118, 8)).float()
        loss = criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_threshold_optimizer(self):
        targets = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        probs = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.85]])
        th, f1s, macro = find_best_thresholds_per_class(targets, probs, ["clsA", "clsB"])
        self.assertIn("clsA", th)
        self.assertIn("clsB", th)
        self.assertGreaterEqual(macro, 0.9)

if __name__ == "__main__":
    unittest.main()

