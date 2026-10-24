"""
backend/analysis.py

NeuroVision — AI-Assisted Brain MRI Analysis
Quantitative tumor measurements, volumetric analysis, confidence metrics,
and benchmark validation statistics.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np


def calculate_single_mask_measurements(
    mask: Optional[np.ndarray],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """
    Compute volume and bounding-box dimensions for an individual 3D binary mask.
    """
    if mask is None or not np.any(mask > 0):
        return {
            "available": False,
            "voxel_count": 0,
            "volume_cm3": 0.0,
            "length_mm": 0.0,
            "width_mm": 0.0,
            "height_mm": 0.0,
        }

    binary = mask > 0
    voxel_count = int(np.sum(binary))

    sx, sy, sz = voxel_spacing
    voxel_vol_mm3 = float(sx * sy * sz)
    volume_mm3 = voxel_count * voxel_vol_mm3
    volume_cm3 = volume_mm3 / 1000.0

    coords = np.argwhere(binary)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = (maxs - mins) + 1

    length_mm = float(extents[0] * sx)
    width_mm = float(extents[1] * sy)
    height_mm = float(extents[2] * sz)

    return {
        "available": True,
        "voxel_count": voxel_count,
        "volume_cm3": round(volume_cm3, 2),
        "length_mm": round(length_mm, 1),
        "width_mm": round(width_mm, 1),
        "height_mm": round(height_mm, 1),
    }


def calculate_tumor_measurements(
    segmentation: Optional[np.ndarray],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    region_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive measurements for Whole Tumor (WT) and sub-regions
    (Tumor Core TC, Enhancing Tumor ET, Edema ED, Non-enhancing Core NCR).
    """
    primary = calculate_single_mask_measurements(segmentation, voxel_spacing)
    if not primary["available"]:
        return {
            "available": False,
            "voxel_count": 0,
            "volume_cm3": None,
            "length_mm": None,
            "width_mm": None,
            "height_mm": None,
            "regions": {},
            "peak_slice": None,
        }

    # Identify slice with peak tumor area
    peak_slice = None
    if segmentation is not None and segmentation.ndim == 3:
        slice_sums = np.sum(segmentation > 0, axis=(0, 1))
        if np.any(slice_sums > 0):
            peak_slice = int(np.argmax(slice_sums))

    result = {
        "available": True,
        "voxel_count": primary["voxel_count"],
        "volume_cm3": primary["volume_cm3"],
        "length_mm": primary["length_mm"],
        "width_mm": primary["width_mm"],
        "height_mm": primary["height_mm"],
        "peak_slice": peak_slice,
        "regions": {},
    }

    if region_masks:
        wt_count = max(1, primary["voxel_count"])
        for reg_key in ["WT", "TC", "ET", "NCR", "ED"]:
            if reg_key in region_masks and region_masks[reg_key] is not None:
                reg_m = calculate_single_mask_measurements(region_masks[reg_key], voxel_spacing)
                ratio_pct = round((reg_m["voxel_count"] / wt_count) * 100.0, 1)
                reg_m["ratio_to_wt_pct"] = ratio_pct
                result["regions"][reg_key] = reg_m

    return result


def calculate_confidence(
    confidence_map: Optional[np.ndarray],
    mask: Optional[np.ndarray] = None,
    region_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Summarize model confidence (predicted probability proxy) across the predicted
    regions and individual subregions.
    """
    if confidence_map is None:
        return {
            "available": False,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "regions": {},
        }

    values = confidence_map
    if mask is not None and np.any(mask > 0):
        values = confidence_map[mask > 0]

    if values.size == 0:
        return {
            "available": False,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "regions": {},
        }

    result = {
        "available": True,
        "mean": round(float(np.mean(values)), 3),
        "min": round(float(np.min(values)), 3),
        "max": round(float(np.max(values)), 3),
        "std": round(float(np.std(values)), 3),
        "regions": {},
    }

    if region_masks:
        for reg_key in ["WT", "TC", "ET"]:
            if reg_key in region_masks and region_masks[reg_key] is not None:
                sub_mask = region_masks[reg_key] > 0
                if np.any(sub_mask):
                    sub_vals = confidence_map[sub_mask]
                    result["regions"][reg_key] = {
                        "mean": round(float(np.mean(sub_vals)), 3),
                        "min": round(float(np.min(sub_vals)), 3),
                        "max": round(float(np.max(sub_vals)), 3),
                    }

    return result


def calculate_dice_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute Dice Similarity Coefficient (DSC) between prediction and ground truth.
    Formula: 2 * |P ∩ G| / (|P| + |G|)
    """
    pred_bool = prediction > 0
    gt_bool = ground_truth > 0

    intersection = 2.0 * np.sum(pred_bool & gt_bool)
    denominator = np.sum(pred_bool) + np.sum(gt_bool)

    if denominator == 0:
        return 1.0  # Perfect agreement on empty mask
    return float(intersection / denominator)


def evaluate_sample_benchmark(
    pred_regions: Dict[str, np.ndarray],
    gt_seg: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate predicted regions against BraTS ground-truth segmentation.
    BraTS GT labels: 1 = NCR, 2 = ED, 3 = ET.
    Subregions:
    - WT (Whole Tumor): labels 1, 2, 3
    - TC (Tumor Core): labels 1, 3
    - ET (Enhancing Tumor): label 3
    """
    gt_wt = (gt_seg > 0).astype(np.uint8)
    gt_tc = np.isin(gt_seg, [1, 3]).astype(np.uint8)
    gt_et = (gt_seg == 3).astype(np.uint8)

    pred_wt = pred_regions.get("WT", np.zeros_like(gt_seg))
    pred_tc = pred_regions.get("TC", np.zeros_like(gt_seg))
    pred_et = pred_regions.get("ET", np.zeros_like(gt_seg))

    dice_wt = calculate_dice_score(pred_wt, gt_wt)
    dice_tc = calculate_dice_score(pred_tc, gt_tc)
    dice_et = calculate_dice_score(pred_et, gt_et)

    return {
        "available": True,
        "metrics": {
            "WT": {
                "label": "Whole Tumor (WT)",
                "dice": round(dice_wt, 4),
                "dice_pct": round(dice_wt * 100.0, 1),
                "pred_voxels": int(np.sum(pred_wt > 0)),
                "gt_voxels": int(np.sum(gt_wt > 0)),
            },
            "TC": {
                "label": "Tumor Core (TC)",
                "dice": round(dice_tc, 4),
                "dice_pct": round(dice_tc * 100.0, 1),
                "pred_voxels": int(np.sum(pred_tc > 0)),
                "gt_voxels": int(np.sum(gt_tc > 0)),
            },
            "ET": {
                "label": "Enhancing Tumor (ET)",
                "dice": round(dice_et, 4),
                "dice_pct": round(dice_et * 100.0, 1),
                "pred_voxels": int(np.sum(pred_et > 0)),
                "gt_voxels": int(np.sum(gt_et > 0)),
            },
        },
    }
