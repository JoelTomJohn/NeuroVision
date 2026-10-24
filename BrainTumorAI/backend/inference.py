from pathlib import Path
from typing import Optional, TypedDict, Any

import numpy as np
import torch
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference


class InferenceResult(TypedDict, total=False):
    success: bool
    is_demo: bool
    message: str
    segmentation: Optional[np.ndarray]
    probabilities: Optional[np.ndarray]
    confidence_map: Optional[np.ndarray]
    region_masks: dict
    volume_cm3: Optional[float]
    length_mm: Optional[float]
    width_mm: Optional[float]
    height_mm: Optional[float]


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model.pt"
)


_model = None
_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def _load_model():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
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

    model.load_state_dict(
        cleaned,
        strict=True,
    )

    model.to(_device)
    model.eval()

    _model = model

    return _model


def _normalize_modalities(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=True)

    for channel in range(volume.shape[0]):

        data = volume[channel]

        nonzero = data != 0

        if np.any(nonzero):

            mean = data[nonzero].mean()
            std = data[nonzero].std()

            if std > 0:
                data[nonzero] = (
                    data[nonzero] - mean
                ) / std
            else:
                data[nonzero] = (
                    data[nonzero] - mean
                )

        volume[channel] = data

    return volume


def run_nnunet_inference(
    mri_volume: Optional[np.ndarray],
    demo_mode: bool = False,
) -> InferenceResult:

    if mri_volume is None:
        return {
            "success": False,
            "is_demo": False,
            "message": "No MRI volume was provided.",
            "segmentation": None,
            "probabilities": None,
            "confidence_map": None,
        }

    if demo_mode:
        return _run_demo_inference(mri_volume)

    try:

        if mri_volume.ndim != 4:
            raise ValueError(
                "Expected four MRI modalities with shape "
                "(4, X, Y, Z)."
            )

        if mri_volume.shape[0] != 4:
            raise ValueError(
                "The model requires exactly four modalities: "
                "T1c, T1, T2 and FLAIR."
            )

        normalized = _normalize_modalities(
            mri_volume
        )

        model = _load_model()

        input_tensor = torch.from_numpy(
            normalized
        ).unsqueeze(0).to(_device)

        with torch.no_grad():

            logits = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

            probabilities = torch.sigmoid(
                logits
            )

        probs = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )

        # Model channel meanings verified on BraTS:
        # 0 = Tumor Core
        # 1 = Whole Tumor
        # 2 = Enhancing Tumor

        pred_tc = probs[0] > 0.5
        pred_wt = probs[1] > 0.5
        pred_et = probs[2] > 0.5

        # Current frontend uses Whole Tumor
        # as the primary segmentation mask.
        segmentation = pred_wt.astype(
            np.uint8
        )

        # Maximum predicted probability
        # across the three tumor channels.
        confidence_map = np.max(
            probs,
            axis=0,
        ).astype(np.float32)

        return {
            "success": True,
            "is_demo": False,
            "message": "Real brain tumor segmentation completed.",
            "segmentation": segmentation,
            "probabilities": probs,
            "confidence_map": confidence_map,
            "region_masks": {
                "TC": pred_tc.astype(np.uint8),
                "WT": pred_wt.astype(np.uint8),
                "ET": pred_et.astype(np.uint8),
            },
        }

    except Exception as e:

        return {
            "success": False,
            "is_demo": False,
            "message": f"Inference failed: {e}",
            "segmentation": None,
            "probabilities": None,
            "confidence_map": None,
        }


def _run_demo_inference(
    mri_volume: np.ndarray,
) -> InferenceResult:

    shape = mri_volume.shape[-3:]

    segmentation = np.zeros(
        shape,
        dtype=np.uint8,
    )

    cx = shape[0] // 2
    cy = shape[1] // 2
    cz = shape[2] // 2

    radius_x = max(
        5,
        shape[0] // 10,
    )

    radius_y = max(
        5,
        shape[1] // 12,
    )

    radius_z = max(
        3,
        shape[2] // 10,
    )

    x, y, z = np.ogrid[
        :shape[0],
        :shape[1],
        :shape[2],
    ]

    mask = (
        ((x - cx) / radius_x) ** 2
        + ((y - cy) / radius_y) ** 2
        + ((z - cz) / radius_z) ** 2
        <= 1
    )

    segmentation[mask] = 1

    confidence_map = np.zeros(
        shape,
        dtype=np.float32,
    )

    confidence_map[mask] = 0.92

    return {
        "success": True,
        "is_demo": True,
        "message": "Demo segmentation generated.",
        "segmentation": segmentation,
        "probabilities": None,
        "confidence_map": confidence_map,
        "region_masks": {
            "WT": segmentation.copy(),
        },
    }
