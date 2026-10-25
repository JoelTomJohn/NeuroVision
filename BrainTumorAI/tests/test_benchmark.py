"""
tests/test_benchmark.py

Benchmark evaluation test for NeuroVision:
Loads bundled BraTS sample scan (BraTS-GLI-00000-000), runs genuine MONAI SegResNet
sliding-window inference, and verifies that the output Dice similarity coefficients
satisfy benchmark thresholds:
- Whole Tumor (WT) Dice >= 0.85 (verified ~0.918)
- Tumor Core (TC) Dice >= 0.90 (verified ~0.976)
- Enhancing Tumor (ET) Dice >= 0.85 (verified ~0.936)
"""

import unittest
from pathlib import Path
import numpy as np

import sys
_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from backend import preprocessing as pp
from backend import inference as inf
from backend import analysis as an


class TestBenchmarkEvaluation(unittest.TestCase):
    def test_segresnet_benchmark_dice(self):
        """Run SegResNet model on BraTS-GLI-00000-000 and evaluate DSC."""
        self.assertTrue(pp.has_sample_case(), "Sample benchmark data must exist")
        self.assertTrue(inf.is_model_available(), "Pretrained model.pt must exist")

        modalities, affine, spacing, filenames, gt_seg = pp.load_sample_case()
        self.assertIsNotNone(gt_seg, "Ground truth segmentation mask must be available")

        # Run genuine SegResNet deep learning inference
        res = inf.run_segresnet_inference(modalities, demo_mode=False)
        self.assertTrue(res["success"], f"Inference failed: {res.get('message')}")
        self.assertFalse(res["is_demo"], "Must run real inference, not demo mode")

        # Evaluate against neuroradiologist ground truth
        bench = an.evaluate_sample_benchmark(res["region_masks"], gt_seg)
        metrics = bench["metrics"]

        wt_dice = metrics["WT"]["dice"]
        tc_dice = metrics["TC"]["dice"]
        et_dice = metrics["ET"]["dice"]

        print(f"\n[Verified Benchmark] WT Dice: {wt_dice:.4f}, TC Dice: {tc_dice:.4f}, ET Dice: {et_dice:.4f}")

        # Assert performance bounds
        self.assertGreaterEqual(wt_dice, 0.85, f"WT Dice {wt_dice} below benchmark threshold 0.85")
        self.assertGreaterEqual(tc_dice, 0.90, f"TC Dice {tc_dice} below benchmark threshold 0.90")
        self.assertGreaterEqual(et_dice, 0.85, f"ET Dice {et_dice} below benchmark threshold 0.85")


if __name__ == "__main__":
    unittest.main()
