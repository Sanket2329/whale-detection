import unittest
from src.data.dataset import normalise_recording_filename

class TestAnnotationResolution(unittest.TestCase):
    def test_greenwich_raven_filename(self):
        self.assertEqual(
            normalise_recording_filename("20150102-140944_AWI229-11_SV1057.wav"),
            "20150102-140944.wav"
        )

    def test_shortened_filename(self):
        self.assertEqual(
            normalise_recording_filename("20150102-140944.wav"),
            "20150102-140944.wav"
        )

    def test_path_filename(self):
        self.assertEqual(
            normalise_recording_filename("/data/Greenwich64S2015/wav/20150102-140944_AWI229-11_SV1057.wav"),
            "20150102-140944.wav"
        )

    def test_unrelated_underscore_filename(self):
        self.assertEqual(
            normalise_recording_filename("whale_recording_01.wav"),
            "whale_recording_01.wav"
        )

if __name__ == "__main__":
    unittest.main()

