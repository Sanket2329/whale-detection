import os
import unittest
import hashlib
import tempfile
import numpy as np
import pandas as pd
import torch
from src.preprocessing.audio import preprocess_audio
from src.preprocessing.spectrogram import generate_spectrogram
from src.models.crnn import WhaleCRNN
from src.training.checkpoint import save_checkpoint, load_checkpoint

class TestPipelineIntegrity(unittest.TestCase):
    def test_split_integrity_and_disjoint_recordings(self):
        """Check 1: Confirm that no recording appears in more than one manifest."""
        manifest_dir = "data/splits"
        splits = ["source_train.csv", "source_val.csv", "target_adaptation.csv", "target_val.csv", "target_test.csv"]
        
        seen_recordings = {}
        for s in splits:
            path = os.path.join(manifest_dir, s)
            self.assertTrue(os.path.exists(path), f"Manifest {path} missing")
            df = pd.read_csv(path)
            recordings = df["recording_id"].tolist()
            
            # Check internal uniqueness
            self.assertEqual(len(recordings), len(set(recordings)), f"Duplicate recordings found inside {s}")
            
            for rec in recordings:
                self.assertNotIn(rec, seen_recordings, f"Recording {rec} leaks across {seen_recordings.get(rec)} and {s}")
                seen_recordings[rec] = s

    def test_target_adaptation_manifest_annotation_isolation(self):
        """Check 2: target_adaptation.csv must not expose any annotation or label fields."""
        path = "data/splits/target_adaptation.csv"
        df = pd.read_csv(path)
        forbidden_cols = ["label", "call_type", "annotation", "begin_time", "end_time", "start_time", "confidence"]
        for col in forbidden_cols:
            self.assertNotIn(col, [c.lower() for c in df.columns], f"Forbidden annotation column '{col}' found in adaptation manifest!")

    def test_manifest_checksum_integrity(self):
        """Verify MD5 checksums stored in manifests match the on-disk audio headers."""
        for s in ["source_train.csv", "target_adaptation.csv"]:
            df = pd.read_csv(os.path.join("data/splits", s))
            sample_row = df.iloc[0]
            file_path = sample_row["file_path"]
            stored_checksum = sample_row["checksum"]
            
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                hasher.update(f.read(65536))
            computed_checksum = hasher.hexdigest()
            self.assertEqual(stored_checksum, computed_checksum, f"Checksum mismatch for {file_path}")

    def test_deterministic_preprocessing_parity(self):
        """Check 4: Confirm preprocessing produces deterministic and identical output across multiple calls."""
        rng = np.random.RandomState(1337)
        audio = rng.randn(10000).astype(np.float32)
        
        # Run preprocessing twice
        p1 = preprocess_audio(audio, original_sr=1000, target_sr=250)
        p2 = preprocess_audio(audio, original_sr=1000, target_sr=250)
        np.testing.assert_array_equal(p1, p2, "Audio preprocessing non-deterministic")
        
        s1 = generate_spectrogram(p1[:7500], n_fft=256, hop_length=64)
        s2 = generate_spectrogram(p2[:7500], n_fft=256, hop_length=64)
        np.testing.assert_array_equal(s1, s2, "Spectrogram generation non-deterministic")

    def test_checkpoint_reproducibility_and_reload(self):
        """Check 5: Saved checkpoint records config, seed, commit, and reproduces exact logits."""
        torch.manual_seed(42)
        model = WhaleCRNN(num_classes=8)
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        dummy_input = torch.randn(1, 1, 129, 118)
        
        with torch.no_grad():
            expected_logits = model(dummy_input)
            
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_ckpt.pt")
            config = {"name": "test_exp", "architecture": "crnn"}
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=1,
                best_metric=0.75,
                config=config,
                seed=42,
                class_thresholds={"class_0": 0.35}
            )
            
            loaded = load_checkpoint(ckpt_path)
            self.assertEqual(loaded["seed"], 42)
            self.assertEqual(loaded["epoch"], 1)
            self.assertIn("git_commit", loaded)
            self.assertEqual(loaded["config"], config)
            self.assertEqual(loaded["class_thresholds"]["class_0"], 0.35)
            
            # Recreate model & reload state
            new_model = WhaleCRNN(num_classes=8)
            new_model.load_state_dict(loaded["model_state"])
            new_model.eval()
            with torch.no_grad():
                reloaded_logits = new_model(dummy_input)
                
            self.assertTrue(torch.allclose(expected_logits, reloaded_logits, atol=1e-6))

if __name__ == "__main__":
    unittest.main()
