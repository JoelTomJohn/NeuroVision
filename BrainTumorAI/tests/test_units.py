"""
tests/test_units.py

Automated unit tests for NeuroVision core modules:
- Preprocessing (NIfTI loading, normalization, sample case retrieval)
- Quantitative Analysis (volumetric calculation, bounding box, Dice score)
- Reporting (PDF report compilation via ReportLab)
- Viewer Utilities (axial slice extraction, discrete mask slicing)
"""

import unittest
from pathlib import Path
import numpy as np

# Ensure NeuroVision modules are importable
import sys
_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from backend import preprocessing as pp
from backend import analysis as an
from backend import report as rep
from frontend import viewer as vw
from backend import inference as inf


class TestPreprocessing(unittest.TestCase):
    def test_has_sample_case(self):
        """Verify the bundled benchmark BraTS case is detected."""
        self.assertTrue(pp.has_sample_case(), "Bundled benchmark case should exist")

    def test_load_sample_case(self):
        """Verify loading the bundled benchmark BraTS sample case."""
        modalities, affine, spacing, filenames, gt_seg = pp.load_sample_case()
        self.assertEqual(modalities.shape, (4, 240, 240, 155))
        self.assertEqual(modalities.dtype, np.float32)
        self.assertEqual(affine.shape, (4, 4))
        self.assertEqual(len(spacing), 3)
        self.assertIn("t1c", filenames)
        self.assertIn("t1", filenames)
        self.assertIn("t2", filenames)
        self.assertIn("flair", filenames)
        self.assertIsNotNone(gt_seg)
        self.assertEqual(gt_seg.shape, (240, 240, 155))

    def test_basic_normalize(self):
        """Verify basic min-max normalization."""
        arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        norm = pp.basic_normalize(arr)
        self.assertAlmostEqual(float(norm[0]), 0.0)
        self.assertAlmostEqual(float(norm[2]), 1.0)


class TestAnalysis(unittest.TestCase):
    def test_calculate_single_mask_measurements(self):
        """Verify volumetric and physical bounding-box measurement logic."""
        mask = np.zeros((100, 100, 100), dtype=np.uint8)
        # 10 x 20 x 30 voxels with 1mm isotropic spacing
        mask[10:20, 20:40, 30:60] = 1
        spacing = (1.0, 1.0, 1.0)
        meas = an.calculate_single_mask_measurements(mask, spacing)

        self.assertTrue(meas["available"])
        self.assertEqual(meas["voxel_count"], 10 * 20 * 30)  # 6000 voxels
        self.assertEqual(meas["volume_cm3"], 6.0)            # 6000 mm^3 = 6.0 cm^3
        self.assertEqual(meas["length_mm"], 10.0)
        self.assertEqual(meas["width_mm"], 20.0)
        self.assertEqual(meas["height_mm"], 30.0)

    def test_calculate_empty_mask(self):
        """Verify handling of empty or None mask."""
        meas = an.calculate_single_mask_measurements(None)
        self.assertFalse(meas["available"])
        self.assertEqual(meas["voxel_count"], 0)
        self.assertEqual(meas["volume_cm3"], 0.0)

    def test_calculate_dice_score(self):
        """Verify Dice Similarity Coefficient calculation."""
        a = np.zeros((10, 10, 10), dtype=np.uint8)
        b = np.zeros((10, 10, 10), dtype=np.uint8)
        a[2:6, 2:6, 2:6] = 1
        b[2:6, 2:6, 2:6] = 1

        # Identical masks -> Dice 1.0
        self.assertAlmostEqual(an.calculate_dice_score(a, b), 1.0)

        # Disjoint masks -> Dice 0.0
        c = np.zeros((10, 10, 10), dtype=np.uint8)
        c[7:9, 7:9, 7:9] = 1
        self.assertAlmostEqual(an.calculate_dice_score(a, c), 0.0)

    def test_calculate_confidence(self):
        """Verify confidence map summary metrics."""
        conf_map = np.ones((50, 50, 50), dtype=np.float32) * 0.85
        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1
        res = an.calculate_confidence(conf_map, mask)
        self.assertTrue(res["available"])
        self.assertAlmostEqual(res["mean"], 0.85)


class TestViewerUtilities(unittest.TestCase):
    def test_get_axial_slice(self):
        """Verify axial slice extraction and rotation."""
        vol = np.zeros((100, 100, 50), dtype=np.float32)
        vol[20, 30, 25] = 100.0
        sl = vw.get_axial_slice(vol, 25)
        self.assertIsNotNone(sl)
        self.assertEqual(sl.shape, (100, 100))

    def test_get_mask_slice_preserves_labels(self):
        """Verify discrete labels are preserved without normalization distortion."""
        mask = np.zeros((50, 50, 20), dtype=np.uint8)
        mask[10, 10, 5] = 1
        mask[20, 20, 5] = 2
        mask[30, 30, 5] = 3
        sl = vw.get_mask_slice(mask, 5)
        self.assertIsNotNone(sl)
        unique_labels = set(np.unique(sl))
        self.assertEqual(unique_labels, {0, 1, 2, 3})


class TestReportGeneration(unittest.TestCase):
    def test_generate_pdf_report(self):
        """Verify PDF summary report compiles successfully with valid ReportLab bytes."""
        meas = {
            "available": True,
            "volume_cm3": 25.5,
            "length_mm": 40.0,
            "width_mm": 35.0,
            "height_mm": 30.0,
            "regions": {
                "WT": {"volume_cm3": 25.5, "ratio_to_wt_pct": 100.0},
                "TC": {"volume_cm3": 18.2, "ratio_to_wt_pct": 71.4},
                "ET": {"volume_cm3": 12.0, "ratio_to_wt_pct": 47.1},
            },
        }
        conf = {
            "available": True,
            "mean": 0.89,
            "min": 0.55,
            "max": 0.99,
            "std": 0.07,
            "regions": {},
        }
        pdf_bytes = rep.generate_pdf_report(
            file_name="BraTS-GLI-00000-000",
            analysis_status="Analysis Verified",
            measurements=meas,
            confidence=conf,
            is_demo=False,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1500)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Must start with standard PDF magic number")


from backend import explainability as xai
from backend import batch_eval as be


class TestModelCheckpoint(unittest.TestCase):
    def test_model_availability(self):
        """Verify pretrained model weights checkpoint exists on disk."""
        self.assertTrue(inf.is_model_available(), "Pretrained model.pt must exist")

    def test_demo_mode_inference(self):
        """Verify demo inference returns expected dictionary schema."""
        vol = np.zeros((4, 64, 64, 32), dtype=np.float32)
        res = inf.run_segresnet_inference(vol, demo_mode=True)
        self.assertTrue(res["success"])
        self.assertTrue(res["is_demo"])
        self.assertIn("segmentation", res)
        self.assertIn("composite", res)
        self.assertIn("region_masks", res)
        self.assertIn("confidence_map", res)
        self.assertIn("entropy_map", res)
        self.assertIn("explainability", res)


class TestExplainability(unittest.TestCase):
    def test_boundary_entropy_map_math(self):
        """Verify Shannon boundary entropy calculation."""
        # 3 channels, 10x10x10 voxels
        # Max entropy at p = 0.5 -> entropy 1.0
        p_ambiguous = np.full((3, 10, 10, 10), 0.5, dtype=np.float32)
        ent = xai.compute_boundary_entropy_map(p_ambiguous)
        self.assertIsNotNone(ent)
        self.assertEqual(ent.shape, (10, 10, 10))
        self.assertAlmostEqual(float(ent[0, 0, 0]), 1.0, places=3)

        # Min entropy at p = 0.0 or p = 1.0 -> entropy near 0.0
        p_certain = np.full((3, 10, 10, 10), 0.0, dtype=np.float32)
        ent_zero = xai.compute_boundary_entropy_map(p_certain)
        self.assertAlmostEqual(float(ent_zero[0, 0, 0]), 0.0, places=3)

    def test_boundary_entropy_invalid_shape(self):
        """Verify invalid input raises ValueError."""
        p_invalid = np.zeros((10, 10, 10), dtype=np.float32)
        with self.assertRaises(ValueError):
            xai.compute_boundary_entropy_map(p_invalid)

    def test_summarize_explainability_diagnostics(self):
        """Verify explainability summary diagnostics."""
        probs = np.full((3, 20, 20, 20), 0.8, dtype=np.float32)
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        diag = xai.summarize_explainability_diagnostics(probs, {"WT": mask})
        self.assertTrue(diag["available"])
        self.assertIn("mean_boundary_entropy", diag)
        self.assertIn("peak_boundary_entropy", diag)
        self.assertIn("ambiguity_fraction_pct", diag)


class TestBatchEvaluation(unittest.TestCase):
    def test_discover_patient_cases_sample(self):
        """Verify patient case discovery on local BraTS directory."""
        data_dir = _project_dir / "data" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        cases = be.discover_patient_cases(data_dir)
        self.assertGreaterEqual(len(cases), 1)
        first = cases[0]
        self.assertEqual(first["case_id"], "BraTS-GLI-00000-000")
        self.assertTrue(first["is_complete"])
        self.assertEqual(len(first["missing_modalities"]), 0)

    def test_cohort_statistics_calculation(self):
        """Verify cohort statistics calculation (mean, std, median, min, max)."""
        mock_results = [
            {"case_id": "C1", "status": "SUCCESS", "wt_dice": 0.90, "tc_dice": 0.85, "et_dice": 0.80},
            {"case_id": "C2", "status": "SUCCESS", "wt_dice": 0.94, "tc_dice": 0.89, "et_dice": 0.86},
            {"case_id": "C3", "status": "FAIL_MISSING_FILES", "error_message": "Missing T1c"},
        ]
        stats = be.compute_cohort_statistics(mock_results)
        self.assertEqual(stats["total_cases_found"], 3)
        self.assertEqual(stats["successful_cases"], 2)
        self.assertEqual(stats["failed_cases"], 1)
        self.assertAlmostEqual(stats["metrics"]["wt_dice"]["mean"], 0.92, places=2)
        self.assertAlmostEqual(stats["metrics"]["wt_dice"]["min"], 0.90, places=2)
        self.assertAlmostEqual(stats["metrics"]["wt_dice"]["max"], 0.94, places=2)

    def test_evaluate_missing_files_case(self):
        """Verify incomplete case is recorded as failure without crashing."""
        case_info = {
            "case_id": "Mock_Incomplete",
            "is_complete": False,
            "missing_modalities": ["T1c", "FLAIR"],
        }
        res = be.evaluate_single_case(case_info)
        self.assertEqual(res["status"], "FAIL_MISSING_FILES")
        self.assertIn("Missing required sequences", res["error_message"])
        self.assertIsNone(res["wt_dice"])


class TestReportEdgeCases(unittest.TestCase):
    def test_report_generation_empty_measurements(self):
        """Verify report compiles cleanly when measurements and confidence are unavailable."""
        meas = {"available": False, "volume_cm3": 0.0}
        conf = {"available": False}
        pdf = rep.generate_pdf_report(
            file_name="Empty_Scan",
            analysis_status="Incomplete",
            measurements=meas,
            confidence=conf,
            is_demo=False,
            slice_image_bytes=None,
        )
        self.assertIsInstance(pdf, bytes)
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()

