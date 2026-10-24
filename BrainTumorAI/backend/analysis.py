"""
backend/analysis.py

Turns a segmentation mask (+ optional probability/confidence map) into
the measurements shown on the Tumor Analysis and Confidence pages.

Both functions accept None gracefully and return a result dict with
is_demo/available flags so the frontend can label placeholder values
correctly.
"""

from typing import Optional, Tuple

import numpy as np


def calculate_tumor_measurements(
    segmentation: Optional[np.ndarray],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """
    Compute tumor volume and bounding-box dimensions from a binary/label
    segmentation mask.

    Parameters
    ----------
    segmentation : np.ndarray or None
        3D mask where tumor voxels are > 0.
    voxel_spacing : tuple of float
        (x, y, z) physical spacing in mm per voxel, from the NIfTI header.

    Returns
    -------
    dict with keys: available, voxel_count, volume_cm3, length_mm,
    width_mm, height_mm
    """
    if segmentation is None or not np.any(segmentation > 0):
        return {
            "available": False,
            "voxel_count": 0,
            "volume_cm3": None,
            "length_mm": None,
            "width_mm": None,
            "height_mm": None,
        }

    mask = segmentation > 0
    voxel_count = int(np.sum(mask))

    sx, sy, sz = voxel_spacing
    voxel_volume_mm3 = sx * sy * sz
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_cm3 = volume_mm3 / 1000.0

    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extent_voxels = (maxs - mins) + 1  # (x, y, z) extent in voxels

    length_mm = float(extent_voxels[0] * sx)
    width_mm = float(extent_voxels[1] * sy)
    height_mm = float(extent_voxels[2] * sz)

    return {
        "available": True,
        "voxel_count": voxel_count,
        "volume_cm3": round(volume_cm3, 2),
        "length_mm": round(length_mm, 1),
        "width_mm": round(width_mm, 1),
        "height_mm": round(height_mm, 1),
    }


def calculate_confidence(confidence_map: Optional[np.ndarray], mask: Optional[np.ndarray] = None) -> dict:
    """
    Summarize a voxel-wise confidence/probability map.

    If mask is provided, statistics are computed only within the
    predicted tumor region (more meaningful); otherwise over the whole
    map.

    Returns dict with keys: available, mean, min, max
    """
    if confidence_map is None:
        return {"available": False, "mean": None, "min": None, "max": None}

    values = confidence_map
    if mask is not None and np.any(mask > 0):
        values = confidence_map[mask > 0]

    if values.size == 0:
        return {"available": False, "mean": None, "min": None, "max": None}

    return {
        "available": True,
        "mean": round(float(np.mean(values)), 3),
        "min": round(float(np.min(values)), 3),
        "max": round(float(np.max(values)), 3),
    }
