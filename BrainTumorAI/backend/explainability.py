"""
backend/explainability.py

NeuroVision — Explainability (XAI) and Uncertainty Quantification Engine
Provides interpretable diagnostics and uncertainty metrics for 3D multimodal brain MRI segmentation:
1. Voxel-wise Shannon Boundary Entropy (Decision boundary uncertainty quantification)
2. Modality Sensitivity / Occlusion Feature Attribution (Evaluates channel-specific contributions)
3. Transparent distinction between raw sigmoid heuristics, epistemic uncertainty, and clinical ground truth
"""

import warnings
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch


def compute_boundary_entropy_map(
    probabilities: Optional[np.ndarray],
    eps: float = 1e-7,
) -> Optional[np.ndarray]:
    """
    Compute normalized voxel-wise Shannon entropy across all predicted multi-label channels.
    
    Formula per channel c:
        H_c(x) = - [ p_c(x) * log2(p_c(x) + eps) + (1 - p_c(x)) * log2(1 - p_c(x) + eps) ]
    
    Total Normalized Entropy:
        H(x) = (1 / C) * sum_c H_c(x)  in [0.0, 1.0]

    Interpretation:
        H(x) == 1.0 indicates maximal model indecision / boundary ambiguity (p == 0.5).
        H(x) == 0.0 indicates absolute certainty (p close to 0.0 or 1.0).
    """
    if probabilities is None:
        return None

    if probabilities.ndim != 4:
        raise ValueError(
            f"Expected 4D array of probabilities with shape (C, X, Y, Z), but got {probabilities.shape}"
        )

    probs = np.clip(probabilities.astype(np.float32), eps, 1.0 - eps)
    num_channels = probs.shape[0]

    # Binary Shannon entropy across independent sigmoid channels
    channel_entropy = -(probs * np.log2(probs) + (1.0 - probs) * np.log2(1.0 - probs))
    mean_entropy = np.mean(channel_entropy, axis=0)

    # Normalize to [0.0, 1.0] (max binary entropy is 1.0 at p=0.5)
    normalized_entropy = np.clip(mean_entropy, 0.0, 1.0)
    return normalized_entropy


def compute_modality_occlusion_sensitivity(
    modalities: np.ndarray,
    region_masks: Dict[str, np.ndarray],
    model: Any = None,
    bounding_box: Optional[Tuple[slice, slice, slice]] = None,
) -> Dict[str, Any]:
    """
    Assess modality-specific feature attribution via systematic channel occlusion (ablation).
    
    Parameters
    ----------
    modalities : np.ndarray
        4D multimodal array with shape (4, X, Y, Z):
        0: T1c, 1: T1, 2: T2, 3: FLAIR.
    region_masks : Dict[str, np.ndarray]
        Dictionary containing baseline binary masks ('WT', 'TC', 'ET').
    model : Optional PyTorch model
        If None, loads the cached SegResNet model from backend.inference.
    bounding_box : Optional tuple of slice
        ROI bounding box to evaluate occlusion efficiently without full-volume convolution.
    """
    from backend import inference as inf
    from monai.inferers import sliding_window_inference

    modality_names = ["T1c", "T1", "T2", "FLAIR"]
    modality_roles = {
        "T1c": "Active contrast enhancement & neo-vascular breakdown (ET / TC)",
        "T1": "Anatomical parenchymal structure reference",
        "T2": "Hyperintense fluid & bulk lesion edema",
        "FLAIR": "Peritumoral vasogenic edema delineation (ED / WT)",
    }

    base_wt = int(np.sum(region_masks.get("WT", 0) > 0))
    base_tc = int(np.sum(region_masks.get("TC", 0) > 0))
    base_et = int(np.sum(region_masks.get("ET", 0) > 0))

    if base_wt == 0:
        return {
            "available": False,
            "message": "No tumor present in baseline prediction; cannot compute occlusion sensitivity.",
            "attributions": {},
        }

    # Use existing cached SegResNet
    if model is None:
        model = inf._load_model()

    device = next(model.parameters()).device

    # Crop to ROI if bounding box provided, or construct from WT mask
    wt_mask = region_masks.get("WT", np.zeros(modalities.shape[1:], dtype=np.uint8)) > 0
    if bounding_box is None and np.any(wt_mask):
        coords = np.argwhere(wt_mask)
        mins = np.maximum(0, coords.min(axis=0) - 16)
        maxs = np.minimum(modalities.shape[1:], coords.max(axis=0) + 16)
        bounding_box = (
            slice(int(mins[0]), int(maxs[0])),
            slice(int(mins[1]), int(maxs[1])),
            slice(int(mins[2]), int(maxs[2])),
        )

    # Prepare normalized baseline input
    norm_vol = inf._normalize_modalities(modalities)

    # Occlusion results dictionary
    occlusion_results = {}

    for ch_idx, mod_name in enumerate(modality_names):
        # Create occluded volume where channel ch_idx is zeroed out
        occluded = norm_vol.copy()
        occluded[ch_idx] = 0.0

        if bounding_box is not None:
            # Crop to ROI for rapid computation
            roi_crop = occluded[:, bounding_box[0], bounding_box[1], bounding_box[2]]
            # Pad to multiple of 16 if necessary
            target_shape = [max(32, int(np.ceil(s / 16.0) * 16)) for s in roi_crop.shape[1:]]
            padded = np.zeros((4, target_shape[0], target_shape[1], target_shape[2]), dtype=np.float32)
            padded[:, :roi_crop.shape[1], :roi_crop.shape[2], :roi_crop.shape[3]] = roi_crop
            
            inp_t = torch.from_numpy(padded).unsqueeze(0).to(device)
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    logits = sliding_window_inference(
                        inputs=inp_t,
                        roi_size=target_shape,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.25,
                        mode="gaussian",
                    )
                probs = torch.sigmoid(logits)[0, :, :roi_crop.shape[1], :roi_crop.shape[2], :roi_crop.shape[3]].cpu().numpy()
        else:
            inp_t = torch.from_numpy(occluded).unsqueeze(0).to(device)
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    logits = sliding_window_inference(
                        inputs=inp_t,
                        roi_size=(240, 240, 160),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.25,
                        mode="gaussian",
                    )
                probs = torch.sigmoid(logits)[0].cpu().numpy()

        pred_tc_occ = int(np.sum(probs[0] > 0.5))
        pred_wt_occ = int(np.sum(probs[1] > 0.5))
        pred_et_occ = int(np.sum(probs[2] > 0.5))

        # Relative drop in predicted volume upon channel occlusion
        drop_wt = max(0, base_wt - pred_wt_occ)
        drop_tc = max(0, base_tc - pred_tc_occ)
        drop_et = max(0, base_et - pred_et_occ)

        occlusion_results[mod_name] = {
            "channel_index": ch_idx,
            "clinical_role": modality_roles[mod_name],
            "drop_wt_voxels": drop_wt,
            "drop_tc_voxels": drop_tc,
            "drop_et_voxels": drop_et,
            "sensitivity_ratio_wt": round(drop_wt / max(1, base_wt), 3),
            "sensitivity_ratio_tc": round(drop_tc / max(1, base_tc), 3),
            "sensitivity_ratio_et": round(drop_et / max(1, base_et), 3),
        }

    # Calculate normalized percentage importance for Whole Tumor and Enhancing Tumor
    sum_drop_wt = sum(d["drop_wt_voxels"] for d in occlusion_results.values())
    sum_drop_et = sum(d["drop_et_voxels"] for d in occlusion_results.values())

    for mod_name, d in occlusion_results.items():
        d["wt_importance_pct"] = round((d["drop_wt_voxels"] / max(1, sum_drop_wt)) * 100.0, 1)
        d["et_importance_pct"] = round((d["drop_et_voxels"] / max(1, sum_drop_et)) * 100.0, 1)

    return {
        "available": True,
        "method": "Modality Occlusion Sensitivity (Channel Ablation Attribution)",
        "baseline_voxels": {"WT": base_wt, "TC": base_tc, "ET": base_et},
        "modality_attributions": occlusion_results,
    }


def summarize_explainability_diagnostics(
    probabilities: Optional[np.ndarray],
    region_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compile diagnostic interpretability metrics:
    - Voxel-wise Shannon boundary entropy summary
    - High-ambiguity voxel fraction (voxels near decision boundary 0.4 <= p <= 0.6)
    - Methodological disclosures
    """
    if probabilities is None:
        return {
            "available": False,
            "message": "Probabilities tensor is missing.",
            "boundary_entropy": None,
        }

    entropy_map = compute_boundary_entropy_map(probabilities)
    
    # Analyze entropy within tumor and peritumoral boundary
    wt_mask = None
    if region_masks and "WT" in region_masks:
        wt_mask = region_masks["WT"] > 0

    if wt_mask is not None and np.any(wt_mask):
        tumor_entropy = entropy_map[wt_mask]
        mean_tumor_entropy = float(np.mean(tumor_entropy))
        peak_tumor_entropy = float(np.max(tumor_entropy))
        # Ambiguity fraction: voxels where entropy exceeds 0.75 (corresponding to 0.25 <= p <= 0.75)
        ambiguous_voxels = int(np.sum(tumor_entropy > 0.75))
        ambiguity_pct = round((ambiguous_voxels / tumor_entropy.size) * 100.0, 1)
    else:
        mean_tumor_entropy = float(np.mean(entropy_map))
        peak_tumor_entropy = float(np.max(entropy_map))
        ambiguous_voxels = 0
        ambiguity_pct = 0.0

    return {
        "available": True,
        "mean_boundary_entropy": round(mean_tumor_entropy, 3),
        "peak_boundary_entropy": round(peak_tumor_entropy, 3),
        "ambiguity_voxel_count": ambiguous_voxels,
        "ambiguity_fraction_pct": ambiguity_pct,
        "entropy_map": entropy_map,
        "interpretability_notice": (
            "Shannon entropy quantifies spatial decision uncertainty: voxels with entropy > 0.75 "
            "represent boundary transitions where model confidence drops close to the 0.5 decision threshold."
        ),
    }
