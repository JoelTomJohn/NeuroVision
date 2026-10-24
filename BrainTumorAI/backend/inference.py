"""
backend/inference.py

NeuroVision — AI-Assisted Brain MRI Analysis
Deep learning inference pipeline powered by MONAI SegResNet.

Model Architecture:
- Architecture: MONAI SegResNet (3D encoder-decoder with residual blocks)
- Spatial Dimensions: 3D (X, Y, Z)
- Input Channels: 4 (T1c, T1, T2, FLAIR)
- Output Channels: 3 (Channel 0: Tumor Core, Channel 1: Whole Tumor, Channel 2: Enhancing Tumor)
- Checkpoint: Pretrained on BraTS (Brain Tumor Segmentation) benchmark

Inference Method:
- Spatial sliding-window inference with Gaussian importance weighting
- CPU & CUDA device compatibility (optimized for 8GB RAM CPU execution)
- Safe caching of model weights
"""

from pathlib import Path
from typing import Optional, TypedDict, Any, Dict
import warnings

import numpy as np
import torch
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference


class InferenceResult(TypedDict, total=False):
    success: bool
    is_demo: bool
    message: str
    segmentation: Optional[np.ndarray]
    composite: Optional[np.ndarray]
    probabilities: Optional[np.ndarray]
    confidence_map: Optional[np.ndarray]
    region_masks: Dict[str, np.ndarray]
    model_info: Dict[str, Any]


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model.pt"
)

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_model_available() -> bool:
    """Check if the pretrained model checkpoint exists on disk."""
    return MODEL_PATH.exists() and MODEL_PATH.is_file()


def _load_model():
    """Load and cache the MONAI SegResNet model weights."""
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained model checkpoint not found at: {MODEL_PATH}\n"
            "Please ensure model.pt is present in the models/ directory."
        )

    model = SegResNet(
        spatial_dims=3,
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=_device,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value

    model.load_state_dict(cleaned, strict=True)
    model.to(_device)
    model.eval()

    _model = model
    return _model


def _normalize_modalities(volume: np.ndarray) -> np.ndarray:
    """
    Standardize nonzero voxels per channel (zero-mean, unit variance).
    Matches BraTS benchmark preprocessing protocol.
    """
    volume = volume.astype(np.float32, copy=True)

    for channel in range(volume.shape[0]):
        data = volume[channel]
        nonzero = data != 0

        if np.any(nonzero):
            mean = float(data[nonzero].mean())
            std = float(data[nonzero].std())

            if std > 1e-6:
                data[nonzero] = (data[nonzero] - mean) / std
            else:
                data[nonzero] = data[nonzero] - mean

        volume[channel] = data

    return volume


def run_segresnet_inference(
    mri_volume: Optional[np.ndarray],
    demo_mode: bool = False,
) -> InferenceResult:
    """
    Execute AI segmentation on multimodal brain MRI volume.

    Parameters
    ----------
    mri_volume : np.ndarray or None
        4D array with shape (4, X, Y, Z) containing:
        0: T1c, 1: T1, 2: T2, 3: FLAIR.
    demo_mode : bool
        If True, returns simulated geometric demonstration masks.

    Returns
    -------
    InferenceResult dictionary with segmentation, region_masks, confidence_map, etc.
    """
    if mri_volume is None:
        return {
            "success": False,
            "is_demo": False,
            "message": "No MRI volume was provided. Please upload or load 4-modality MRI data.",
            "segmentation": None,
            "composite": None,
            "probabilities": None,
            "confidence_map": None,
            "region_masks": {},
            "model_info": {},
        }

    if demo_mode:
        return _run_demo_inference(mri_volume)

    try:
        if mri_volume.ndim != 4:
            raise ValueError(
                f"Expected 4D array with shape (4, X, Y, Z), but received shape {mri_volume.shape}."
            )

        if mri_volume.shape[0] != 4:
            raise ValueError(
                f"The model requires exactly 4 modalities (T1c, T1, T2, FLAIR). Received channel count: {mri_volume.shape[0]}."
            )

        normalized = _normalize_modalities(mri_volume)
        model = _load_model()

        input_tensor = torch.from_numpy(normalized).unsqueeze(0).to(_device)

        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=(240, 240, 160),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )

            probabilities = torch.sigmoid(logits)

        probs = probabilities[0].detach().cpu().numpy()

        # Model output channels (BraTS standard multi-label formulation):
        # Channel 0 = Tumor Core (TC: necrotic core + active enhancing tumor)
        # Channel 1 = Whole Tumor (WT: peritumoral edema + tumor core)
        # Channel 2 = Enhancing Tumor (ET: active gadolinium-enhancing region)
        pred_tc = (probs[0] > 0.5).astype(np.uint8)
        pred_wt = (probs[1] > 0.5).astype(np.uint8)
        pred_et = (probs[2] > 0.5).astype(np.uint8)

        # Enforce clinical hierarchical consistency:
        # ET must be within TC; TC must be within WT
        pred_tc = pred_tc | pred_et
        pred_wt = pred_wt | pred_tc

        # Disjoint subregions:
        # Non-enhancing core (NCR) = TC excluding ET
        pred_ncr = ((pred_tc > 0) & (pred_et == 0)).astype(np.uint8)
        # Peritumoral edema (ED) = WT excluding TC
        pred_ed = ((pred_wt > 0) & (pred_tc == 0)).astype(np.uint8)

        # Multiclass composite mask according to BraTS labeling convention:
        # 0 = Background
        # 1 = Non-enhancing / Necrotic Core (NCR)
        # 2 = Peritumoral Edema (ED)
        # 3 = Enhancing Tumor (ET)
        composite = np.zeros(pred_wt.shape, dtype=np.uint8)
        composite[pred_ncr > 0] = 1
        composite[pred_ed > 0] = 2
        composite[pred_et > 0] = 3

        # Primary whole-tumor binary mask
        segmentation = pred_wt.copy()

        # Maximum predicted confidence proxy across channels
        confidence_map = np.max(probs, axis=0).astype(np.float32)

        return {
            "success": True,
            "is_demo": False,
            "message": "AI multimodal brain tumor segmentation completed successfully using MONAI SegResNet.",
            "segmentation": segmentation,
            "composite": composite,
            "probabilities": probs,
            "confidence_map": confidence_map,
            "region_masks": {
                "WT": pred_wt,
                "TC": pred_tc,
                "ET": pred_et,
                "NCR": pred_ncr,
                "ED": pred_ed,
                "composite": composite,
            },
            "model_info": {
                "architecture": "MONAI SegResNet (3D)",
                "checkpoint": "Pretrained BraTS Model",
                "in_channels": 4,
                "out_channels": 3,
                "device": str(_device).upper(),
            },
        }

    except Exception as e:
        # Never silently replace real inference failures with demo data
        return {
            "success": False,
            "is_demo": False,
            "message": f"Real model inference failed: {e}",
            "segmentation": None,
            "composite": None,
            "probabilities": None,
            "confidence_map": None,
            "region_masks": {},
            "model_info": {},
        }


# Backwards compatibility alias for legacy call sites
run_nnunet_inference = run_segresnet_inference
run_inference = run_segresnet_inference


def _run_demo_inference(mri_volume: np.ndarray) -> InferenceResult:
    """Generate realistic synthetic ellipsoidal demo regions for UI testing."""
    shape = mri_volume.shape[-3:]

    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    rx_wt = max(6, shape[0] // 9)
    ry_wt = max(6, shape[1] // 11)
    rz_wt = max(4, shape[2] // 9)

    rx_tc = int(rx_wt * 0.70)
    ry_tc = int(ry_wt * 0.70)
    rz_tc = int(rz_wt * 0.70)

    rx_et = int(rx_wt * 0.45)
    ry_et = int(ry_wt * 0.45)
    rz_et = int(rz_wt * 0.45)

    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]

    dist_sq_wt = ((x - cx) / rx_wt) ** 2 + ((y - cy) / ry_wt) ** 2 + ((z - cz) / rz_wt) ** 2
    dist_sq_tc = ((x - cx) / rx_tc) ** 2 + ((y - cy) / ry_tc) ** 2 + ((z - cz) / rz_tc) ** 2
    dist_sq_et = ((x - cx) / rx_et) ** 2 + ((y - cy) / ry_et) ** 2 + ((z - cz) / rz_et) ** 2

    pred_wt = (dist_sq_wt <= 1.0).astype(np.uint8)
    pred_tc = (dist_sq_tc <= 1.0).astype(np.uint8)
    pred_et = (dist_sq_et <= 1.0).astype(np.uint8)

    pred_ncr = ((pred_tc > 0) & (pred_et == 0)).astype(np.uint8)
    pred_ed = ((pred_wt > 0) & (pred_tc == 0)).astype(np.uint8)

    composite = np.zeros(shape, dtype=np.uint8)
    composite[pred_ncr > 0] = 1
    composite[pred_ed > 0] = 2
    composite[pred_et > 0] = 3

    confidence_map = np.zeros(shape, dtype=np.float32)
    confidence_map[pred_wt > 0] = 0.88
    confidence_map[pred_tc > 0] = 0.93
    confidence_map[pred_et > 0] = 0.96

    return {
        "success": True,
        "is_demo": True,
        "message": "Demo mode active: synthetic placeholder tumor regions generated.",
        "segmentation": pred_wt,
        "composite": composite,
        "probabilities": None,
        "confidence_map": confidence_map,
        "region_masks": {
            "WT": pred_wt,
            "TC": pred_tc,
            "ET": pred_et,
            "NCR": pred_ncr,
            "ED": pred_ed,
            "composite": composite,
        },
        "model_info": {
            "architecture": "Demo Simulation (Synthetic Geometric Regions)",
            "checkpoint": "None (Demo Mode)",
            "in_channels": 4,
            "out_channels": 3,
            "device": "CPU",
        },
    }
