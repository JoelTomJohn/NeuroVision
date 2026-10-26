"""
backend/batch_eval.py

NeuroVision — Multi-Patient Dataset Batch Evaluation Engine
Automated evaluation pipeline for multi-subject BraTS cohorts:
- Robust discovery of multimodal NIfTI sequences and ground-truth segmentation masks
- Per-case Dice similarity computation (WT, TC, ET, NCR, ED)
- Volumetric measurement and confidence extraction
- Strict empty-mask handling (conforms to BraTS challenge evaluation criteria)
- Error recording without silent case drops
- Machine-readable CSV and JSON reporting
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import nibabel as nib

# Add project root to path
_backend_dir = Path(__file__).resolve().parent
_project_dir = _backend_dir.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from backend import inference as inf
from backend import analysis as an


def _match_modality_file(case_folder: Path, patterns: List[str]) -> Optional[Path]:
    """Find a file matching any of the pattern strings (case-insensitive)."""
    for file in case_folder.iterdir():
        if file.is_file():
            name_lower = file.name.lower()
            if any(p.lower() in name_lower for p in patterns) and (
                name_lower.endswith(".nii") or name_lower.endswith(".nii.gz")
            ):
                return file
    return None


def discover_patient_cases(dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan a dataset directory for patient case folders containing BraTS sequences.
    Identifies T1c, T1, T2, FLAIR, and Seg files.
    """
    cases = []
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return cases

    for item in sorted(dataset_dir.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            case_id = item.name
            t1c_file = _match_modality_file(item, ["-t1c.", "-t1ce.", "t1c.", "t1ce."])
            t1_file = _match_modality_file(item, ["-t1n.", "-t1.", "t1n.", "t1."])
            t2_file = _match_modality_file(item, ["-t2w.", "-t2.", "t2w.", "t2."])
            flair_file = _match_modality_file(item, ["-t2f.", "-flair.", "t2f.", "flair."])
            seg_file = _match_modality_file(item, ["-seg.", "seg.", "ground_truth."])

            missing = []
            if not t1c_file: missing.append("T1c")
            if not t1_file: missing.append("T1")
            if not t2_file: missing.append("T2")
            if not flair_file: missing.append("FLAIR")
            if not seg_file: missing.append("Seg (Ground Truth)")

            cases.append({
                "case_id": case_id,
                "folder_path": str(item),
                "is_complete": len(missing) == 0,
                "missing_modalities": missing,
                "t1c_path": str(t1c_file) if t1c_file else None,
                "t1_path": str(t1_file) if t1_file else None,
                "t2_path": str(t2_file) if t2_file else None,
                "flair_path": str(flair_file) if flair_file else None,
                "seg_path": str(seg_file) if seg_file else None,
            })

    return cases


def evaluate_single_case(
    case_info: Dict[str, Any],
    demo_mode: bool = False,
) -> Dict[str, Any]:
    """
    Load multimodal sequences, run SegResNet inference, and evaluate against ground truth.
    Transparently catches and records failures (missing files, shape mismatches, corruption).
    """
    case_id = case_info["case_id"]

    if not case_info["is_complete"]:
        return {
            "case_id": case_id,
            "status": "FAIL_MISSING_FILES",
            "error_message": f"Missing required sequences: {', '.join(case_info['missing_modalities'])}",
            "wt_dice": None,
            "tc_dice": None,
            "et_dice": None,
            "ncr_dice": None,
            "ed_dice": None,
            "wt_volume_cm3": None,
            "tc_volume_cm3": None,
            "et_volume_cm3": None,
            "mean_confidence": None,
            "pred_voxels_wt": 0,
            "gt_voxels_wt": 0,
        }

    try:
        # Load NIfTI volumes
        t1c_img = nib.load(case_info["t1c_path"])
        t1_img = nib.load(case_info["t1_path"])
        t2_img = nib.load(case_info["t2_path"])
        flair_img = nib.load(case_info["flair_path"])
        seg_img = nib.load(case_info["seg_path"])

        t1c_data = np.asarray(t1c_img.get_fdata(), dtype=np.float32)
        t1_data = np.asarray(t1_img.get_fdata(), dtype=np.float32)
        t2_data = np.asarray(t2_img.get_fdata(), dtype=np.float32)
        flair_data = np.asarray(flair_img.get_fdata(), dtype=np.float32)
        gt_seg = np.asarray(seg_img.get_fdata(), dtype=np.float32)

        # Dimension verification
        ref_shape = t1c_data.shape
        for name, data in [("T1", t1_data), ("T2", t2_data), ("FLAIR", flair_data), ("GT", gt_seg)]:
            if data.shape != ref_shape:
                return {
                    "case_id": case_id,
                    "status": "FAIL_SHAPE_MISMATCH",
                    "error_message": f"Shape mismatch: T1c has {ref_shape} but {name} has {data.shape}",
                    "wt_dice": None,
                    "tc_dice": None,
                    "et_dice": None,
                    "ncr_dice": None,
                    "ed_dice": None,
                    "wt_volume_cm3": None,
                    "tc_volume_cm3": None,
                    "et_volume_cm3": None,
                    "mean_confidence": None,
                    "pred_voxels_wt": 0,
                    "gt_voxels_wt": 0,
                }

        # Stack modalities
        modalities = np.stack([t1c_data, t1_data, t2_data, flair_data], axis=0)
        voxel_spacing = tuple(float(v) for v in t1c_img.header.get_zooms()[:3])

        # Run inference
        res = inf.run_segresnet_inference(modalities, demo_mode=demo_mode)
        if not res.get("success"):
            return {
                "case_id": case_id,
                "status": "FAIL_INFERENCE_ERROR",
                "error_message": res.get("message", "Unknown inference failure"),
                "wt_dice": None,
                "tc_dice": None,
                "et_dice": None,
                "ncr_dice": None,
                "ed_dice": None,
                "wt_volume_cm3": None,
                "tc_volume_cm3": None,
                "et_volume_cm3": None,
                "mean_confidence": None,
                "pred_voxels_wt": 0,
                "gt_voxels_wt": 0,
            }

        region_masks = res["region_masks"]
        pred_wt = region_masks["WT"]
        pred_tc = region_masks["TC"]
        pred_et = region_masks["ET"]
        pred_ncr = region_masks["NCR"]
        pred_ed = region_masks["ED"]

        # Ground-truth binary targets (BraTS standard convention)
        gt_wt = (gt_seg > 0).astype(np.uint8)
        gt_tc = np.isin(gt_seg, [1, 3]).astype(np.uint8)
        gt_et = (gt_seg == 3).astype(np.uint8)
        gt_ncr = (gt_seg == 1).astype(np.uint8)
        gt_ed = (gt_seg == 2).astype(np.uint8)

        # Dice calculation
        dice_wt = an.calculate_dice_score(pred_wt, gt_wt)
        dice_tc = an.calculate_dice_score(pred_tc, gt_tc)
        dice_et = an.calculate_dice_score(pred_et, gt_et)
        dice_ncr = an.calculate_dice_score(pred_ncr, gt_ncr)
        dice_ed = an.calculate_dice_score(pred_ed, gt_ed)

        # Volumetrics
        meas = an.calculate_tumor_measurements(res.get("segmentation"), voxel_spacing, region_masks)
        conf = an.calculate_confidence(res.get("confidence_map"), res.get("segmentation"), region_masks)

        wt_vol = meas["regions"].get("WT", {}).get("volume_cm3", meas.get("volume_cm3", 0.0))
        tc_vol = meas["regions"].get("TC", {}).get("volume_cm3", 0.0)
        et_vol = meas["regions"].get("ET", {}).get("volume_cm3", 0.0)

        return {
            "case_id": case_id,
            "status": "SUCCESS",
            "error_message": "",
            "wt_dice": round(dice_wt, 4),
            "tc_dice": round(dice_tc, 4),
            "et_dice": round(dice_et, 4),
            "ncr_dice": round(dice_ncr, 4),
            "ed_dice": round(dice_ed, 4),
            "wt_volume_cm3": wt_vol,
            "tc_volume_cm3": tc_vol,
            "et_volume_cm3": et_vol,
            "mean_confidence": conf.get("mean"),
            "pred_voxels_wt": int(np.sum(pred_wt > 0)),
            "gt_voxels_wt": int(np.sum(gt_wt > 0)),
            "pred_voxels_tc": int(np.sum(pred_tc > 0)),
            "gt_voxels_tc": int(np.sum(gt_tc > 0)),
            "pred_voxels_et": int(np.sum(pred_et > 0)),
            "gt_voxels_et": int(np.sum(gt_et > 0)),
        }

    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "FAIL_EXCEPTION",
            "error_message": str(exc),
            "wt_dice": None,
            "tc_dice": None,
            "et_dice": None,
            "ncr_dice": None,
            "ed_dice": None,
            "wt_volume_cm3": None,
            "tc_volume_cm3": None,
            "et_volume_cm3": None,
            "mean_confidence": None,
            "pred_voxels_wt": 0,
            "gt_voxels_wt": 0,
        }


def compute_cohort_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate mean, standard deviation, median, min, and max across valid cases."""
    successful = [r for r in results if r["status"] == "SUCCESS"]

    stats = {
        "total_cases_found": len(results),
        "successful_cases": len(successful),
        "failed_cases": len(results) - len(successful),
        "metrics": {},
    }

    if not successful:
        return stats

    for metric_key in ["wt_dice", "tc_dice", "et_dice", "ncr_dice", "ed_dice", "wt_volume_cm3", "mean_confidence"]:
        values = [r[metric_key] for r in successful if r.get(metric_key) is not None]
        if values:
            arr = np.array(values, dtype=np.float64)
            stats["metrics"][metric_key] = {
                "count": len(arr),
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            }

    return stats


def run_batch_evaluation(
    dataset_dir: Path,
    output_csv: Optional[Path] = None,
    output_json: Optional[Path] = None,
    max_cases: Optional[int] = None,
    demo_mode: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute batch evaluation across an entire dataset directory and export results.
    """
    cases = discover_patient_cases(dataset_dir)
    if max_cases is not None:
        cases = cases[:max_cases]

    results = []
    print(f"[BatchEval] Discovered {len(cases)} case directories in {dataset_dir}")

    for idx, case_info in enumerate(cases, 1):
        case_id = case_info["case_id"]
        print(f"[BatchEval] [{idx}/{len(cases)}] Evaluating case: {case_id}...")
        res = evaluate_single_case(case_info, demo_mode=demo_mode)
        results.append(res)
        if res["status"] == "SUCCESS":
            print(f"            WT Dice: {res['wt_dice']} | TC Dice: {res['tc_dice']} | ET Dice: {res['et_dice']}")
        else:
            print(f"            FAILED ({res['status']}): {res['error_message']}")

    stats = compute_cohort_statistics(results)

    # Export CSV if requested
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "case_id", "status", "error_message",
            "wt_dice", "tc_dice", "et_dice", "ncr_dice", "ed_dice",
            "wt_volume_cm3", "tc_volume_cm3", "et_volume_cm3", "mean_confidence",
            "pred_voxels_wt", "gt_voxels_wt",
            "pred_voxels_tc", "gt_voxels_tc",
            "pred_voxels_et", "gt_voxels_et",
        ]
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"[BatchEval] CSV report saved to: {output_csv}")

    # Export JSON if requested
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_directory": str(dataset_dir),
            "summary_statistics": stats,
            "case_results": results,
        }
        with open(output_json, mode="w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[BatchEval] JSON summary saved to: {output_json}")

    return results, stats


def main():
    parser = argparse.ArgumentParser(description="NeuroVision Multi-Patient Dataset Batch Evaluation CLI")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing patient case folders")
    parser.add_argument("--output-csv", type=str, default="outputs/batch_evaluation_results.csv", help="Path to output CSV")
    parser.add_argument("--output-json", type=str, default="outputs/batch_evaluation_summary.json", help="Path to output JSON")
    parser.add_argument("--max-cases", type=int, default=None, help="Maximum number of cases to evaluate")
    parser.add_argument("--demo", action="store_true", help="Run with simulated demo inference (for testing CLI)")

    args = parser.parse_args()
    data_path = Path(args.data_dir).resolve()

    if not data_path.exists():
        print(f"Error: Dataset directory does not exist: {data_path}")
        sys.exit(1)

    run_batch_evaluation(
        dataset_dir=data_path,
        output_csv=Path(args.output_csv).resolve(),
        output_json=Path(args.output_json).resolve(),
        max_cases=args.max_cases,
        demo_mode=args.demo,
    )


if __name__ == "__main__":
    main()
