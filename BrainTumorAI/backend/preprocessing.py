"""
backend/preprocessing.py

NeuroVision — AI-Assisted Brain MRI Analysis
NIfTI loading, validation, and multimodal preprocessing utilities.

Supports:
- Single and 4-modality BraTS NIfTI loading (.nii, .nii.gz)
- Coregistration and spatial shape verification
- Modality order verification: T1c -> T1 -> T2 -> FLAIR
- Bundled BraTS benchmark sample loading for demonstration and evaluation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

import numpy as np


class NiftiLoadError(Exception):
    """Raised when a NIfTI file cannot be read or is unusable."""


@dataclass
class NiftiVolume:
    data: np.ndarray
    affine: np.ndarray
    voxel_spacing: Tuple[float, float, float]
    shape: Tuple[int, int, int]
    filename: str


def load_nifti(file_bytes: bytes, filename: str) -> NiftiVolume:
    """Load a single NIfTI file uploaded through Streamlit."""

    import os
    import tempfile
    import nibabel as nib

    if not (
        filename.lower().endswith(".nii")
        or filename.lower().endswith(".nii.gz")
    ):
        raise NiftiLoadError(
            "Unsupported file extension. Please upload a .nii or .nii.gz file."
        )

    if not file_bytes:
        raise NiftiLoadError("The uploaded file is empty.")

    temp_path = None

    try:
        suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        img = nib.load(temp_path)
        data = np.asarray(img.get_fdata(), dtype=np.float32)

    except Exception as exc:
        raise NiftiLoadError(
            "This file could not be read as a NIfTI volume. "
            "Please make sure it is a valid .nii or .nii.gz file."
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if data.size == 0:
        raise NiftiLoadError("The NIfTI file contains no image data.")

    if data.ndim == 4:
        data = data[..., 0]

    if data.ndim != 3:
        raise NiftiLoadError(
            f"Expected a 3D MRI volume but this file has "
            f"{data.ndim} dimensions."
        )

    try:
        voxel_spacing = tuple(
            float(v) for v in img.header.get_zooms()[:3]
        )
    except Exception:
        voxel_spacing = (1.0, 1.0, 1.0)

    return NiftiVolume(
        data=data,
        affine=img.affine,
        voxel_spacing=voxel_spacing,
        shape=data.shape,
        filename=filename,
    )


def load_multimodal_nifti(
    modality_files: Dict[str, Tuple[bytes, str]]
):
    """
    Load the four BraTS MRI modalities.

    Expected dictionary:

        {
            "t1c": (file_bytes, filename),
            "t1":  (file_bytes, filename),
            "t2":  (file_bytes, filename),
            "flair": (file_bytes, filename),
        }

    Returns
    -------
    modalities : np.ndarray
        Shape: (4, X, Y, Z)

        Channel order:
        0 = T1c
        1 = T1
        2 = T2
        3 = FLAIR

    affine : np.ndarray
    voxel_spacing : tuple
    """

    required = ["t1c", "t1", "t2", "flair"]

    for key in required:
        if key not in modality_files:
            raise NiftiLoadError(
                f"Missing required MRI modality: {key.upper()}"
            )

    loaded = {}

    for key in required:
        file_bytes, filename = modality_files[key]

        loaded[key] = load_nifti(
            file_bytes,
            filename
        )

    reference = loaded["t1c"]

    # All four modalities must be spatially aligned.
    for key in required:
        if loaded[key].shape != reference.shape:
            raise NiftiLoadError(
                "The four MRI modalities must have identical dimensions. "
                f"T1c has {reference.shape}, but {key.upper()} has "
                f"{loaded[key].shape}."
            )

    # Check voxel spacing.
    for key in required:
        spacing = loaded[key].voxel_spacing

        if not np.allclose(
            spacing,
            reference.voxel_spacing,
            atol=1e-4
        ):
            raise NiftiLoadError(
                "The four MRI modalities must have matching voxel spacing."
            )

    # Model expects:
    # T1c, T1, T2, FLAIR
    modalities = np.stack(
        [
            loaded["t1c"].data,
            loaded["t1"].data,
            loaded["t2"].data,
            loaded["flair"].data,
        ],
        axis=0
    ).astype(np.float32)

    return (
        modalities,
        reference.affine,
        reference.voxel_spacing,
    )


def basic_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Simple min-max normalization.

    The real MONAI model uses nonzero, channel-wise normalization
    rather than this function.
    """

    v_min = np.nanmin(volume)
    v_max = np.nanmax(volume)

    if v_max - v_min < 1e-6:
        return np.zeros_like(volume)

    return (volume - v_min) / (v_max - v_min)


def describe_modalities_note() -> str:
    """Explain the required BraTS MRI modalities."""

    return (
        "NeuroVision requires four co-registered, skull-stripped 3D MRI "
        "modalities in BraTS format: T1c (contrast-enhanced), T1 (native), "
        "T2 (T2-weighted), and FLAIR (fluid-attenuated inversion recovery)."
    )


SAMPLE_CASE_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    / "BraTS-GLI-00000-000"
)


def has_sample_case() -> bool:
    """Check if the bundled benchmark BraTS sample case is available."""
    if not SAMPLE_CASE_DIR.exists():
        return False
    required_files = [
        "BraTS-GLI-00000-000-t1c.nii.gz",
        "BraTS-GLI-00000-000-t1n.nii.gz",
        "BraTS-GLI-00000-000-t2w.nii.gz",
        "BraTS-GLI-00000-000-t2f.nii.gz",
    ]
    return all((SAMPLE_CASE_DIR / f).exists() for f in required_files)


def load_sample_case() -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], Dict[str, str], Optional[np.ndarray]]:
    """
    Load the bundled benchmark BraTS sample case (BraTS-GLI-00000-000).

    Returns
    -------
    modalities : np.ndarray (4, X, Y, Z)
        Channel order: T1c, T1, T2, FLAIR
    affine : np.ndarray (4, 4)
    voxel_spacing : tuple of float
    filenames : dict of str
    gt_seg : Optional[np.ndarray] (X, Y, Z) ground truth mask if available
    """
    import nibabel as nib

    if not has_sample_case():
        raise NiftiLoadError(f"Bundled sample case not found at {SAMPLE_CASE_DIR}")

    t1c_path = SAMPLE_CASE_DIR / "BraTS-GLI-00000-000-t1c.nii.gz"
    t1_path = SAMPLE_CASE_DIR / "BraTS-GLI-00000-000-t1n.nii.gz"
    t2_path = SAMPLE_CASE_DIR / "BraTS-GLI-00000-000-t2w.nii.gz"
    flair_path = SAMPLE_CASE_DIR / "BraTS-GLI-00000-000-t2f.nii.gz"
    seg_path = SAMPLE_CASE_DIR / "BraTS-GLI-00000-000-seg.nii.gz"

    t1c_img = nib.load(str(t1c_path))
    t1_img = nib.load(str(t1_path))
    t2_img = nib.load(str(t2_path))
    flair_img = nib.load(str(flair_path))

    t1c_data = np.asarray(t1c_img.get_fdata(), dtype=np.float32)
    t1_data = np.asarray(t1_img.get_fdata(), dtype=np.float32)
    t2_data = np.asarray(t2_img.get_fdata(), dtype=np.float32)
    flair_data = np.asarray(flair_img.get_fdata(), dtype=np.float32)

    modalities = np.stack([t1c_data, t1_data, t2_data, flair_data], axis=0)

    try:
        voxel_spacing = tuple(float(v) for v in t1c_img.header.get_zooms()[:3])
    except Exception:
        voxel_spacing = (1.0, 1.0, 1.0)

    filenames = {
        "t1c": t1c_path.name,
        "t1": t1_path.name,
        "t2": t2_path.name,
        "flair": flair_path.name,
        "case_id": "BraTS-GLI-00000-000",
    }

    gt_seg = None
    if seg_path.exists():
        try:
            gt_img = nib.load(str(seg_path))
            gt_seg = np.asarray(gt_img.get_fdata(), dtype=np.float32)
        except Exception:
            gt_seg = None

    return modalities, t1c_img.affine, voxel_spacing, filenames, gt_seg
