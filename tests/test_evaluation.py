import unittest
import numpy as np
from src.evaluation.event_metrics import match_events_bipartite, compute_temporal_iou

class TestEventEvaluation(unittest.TestCase):
    def test_iou_computation(self):
        # Overlap [5, 10] and [8, 12]: intersection=2, union=7 -> 2/7
        iou = compute_temporal_iou(5.0, 10.0, 8.0, 12.0)
        self.assertAlmostEqual(iou, 2.0 / 7.0)
        
        # Disjoint
        iou_disjoint = compute_temporal_iou(0.0, 5.0, 6.0, 10.0)
        self.assertEqual(iou_disjoint, 0.0)

    def test_strict_one_to_one_matching_no_double_counting(self):
        """Verify that a single ground truth event cannot match multiple predictions."""
        # 1 GT event: [10.0, 20.0]
        gt = [{"begin_time": 10.0, "end_time": 20.0}]
        # 2 overlapping predictions covering the same GT
        preds = [
            {"begin_time": 10.0, "end_time": 19.0}, # high IoU
            {"begin_time": 11.0, "end_time": 20.0}  # high IoU
        ]
        
        tp, fp, fn = match_events_bipartite(gt, preds, iou_threshold=0.5)
        # Exactly 1 match allowed: TP=1, FP=1, FN=0
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

    def test_strict_one_to_one_matching_no_reused_predictions(self):
        """Verify that a single prediction cannot match multiple ground truth events."""
        # 2 GT events
        gts = [
            {"begin_time": 10.0, "end_time": 20.0},
            {"begin_time": 15.0, "end_time": 25.0}
        ]
        # 1 large prediction spanning both
        preds = [
            {"begin_time": 12.0, "end_time": 22.0}
        ]
        
        tp, fp, fn = match_events_bipartite(gts, preds, iou_threshold=0.1)
        # Exactly 1 match allowed: TP=1, FP=0, FN=1
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 1)

if __name__ == "__main__":
    unittest.main()

