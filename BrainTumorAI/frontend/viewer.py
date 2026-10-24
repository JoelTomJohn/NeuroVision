"""
frontend/viewer.py

NeuroVision — AI-Assisted Brain MRI Analysis
Axial slice extraction, multi-region radiological overlay generation,
synchronized slice navigation, and 3D volumetric scatter visualization.
"""

from typing import Optional, Tuple, Dict, Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from frontend.styles import COLORS, REGION_COLORS
from frontend.components import render_panel_label


def get_axial_slice(
    volume: np.ndarray,
    slice_index: int,
    is_mask: bool = False,
) -> np.ndarray:
    """
    Extract a single 2D axial slice from a 3D MRI volume, normalized to 0-255
    and rotated to standard radiological orientation (superior at top).

    If is_mask is True or array has integer/boolean dtype, normalization is bypassed
    to preserve exact discrete label values.
    """
    if volume is None:
        return None
    slice_index = int(np.clip(slice_index, 0, volume.shape[2] - 1))
    slice_2d = volume[:, :, slice_index]

    if is_mask or np.issubdtype(volume.dtype, np.integer) or np.issubdtype(volume.dtype, np.bool_):
        return np.rot90(slice_2d)

    img = slice_2d.astype(np.float32)
    img_min, img_max = np.nanmin(img), np.nanmax(img)
    if img_max - img_min > 1e-6:
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        img = np.zeros_like(img)

    return np.rot90(img)


def get_mask_slice(
    volume: Optional[np.ndarray],
    slice_index: int,
) -> Optional[np.ndarray]:
    """
    Extract a single 2D axial slice from a 3D segmentation mask.
    Preserves exact discrete integer label values (0, 1, 2, 3...) without
    intensity normalization, and applies standard radiological rotation (rot90).
    """
    if volume is None:
        return None
    slice_index = int(np.clip(slice_index, 0, volume.shape[2] - 1))
    return np.rot90(volume[:, :, slice_index])


def create_overlay(
    mri_slice: np.ndarray,
    mask_slice: Optional[np.ndarray],
    mask_color: tuple = (0, 201, 167),  # NeuroVision Teal for WT
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """
    Combine a grayscale MRI slice with a single-region segmentation mask slice.
    """
    base = np.stack([mri_slice] * 3, axis=-1).astype(np.uint8)

    if mask_slice is None:
        return base

    mask_bool = mask_slice > 0
    if not np.any(mask_bool):
        return base

    overlay = base.copy().astype(np.float32)
    color = np.array(mask_color, dtype=np.float32)
    overlay[mask_bool] = (
        overlay[mask_bool] * (1.0 - mask_alpha) + color * mask_alpha
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_composite_overlay(
    mri_slice: np.ndarray,
    composite_slice: Optional[np.ndarray],
    mask_alpha: float = 0.50,
) -> np.ndarray:
    """
    Combine a grayscale MRI slice with a multi-region BraTS composite mask:
    - Label 2 (ED / Peritumoral Edema): Teal (0, 201, 167)
    - Label 1 (NCR / Non-enhancing Core): Amber (245, 158, 11)
    - Label 3 (ET / Enhancing Tumor): Crimson (239, 68, 68)
    """
    base = np.stack([mri_slice] * 3, axis=-1).astype(np.uint8)

    if composite_slice is None or not np.any(composite_slice > 0):
        return base

    overlay = base.copy().astype(np.float32)

    color_ed = np.array(REGION_COLORS["ED"]["rgb"], dtype=np.float32)
    color_ncr = np.array(REGION_COLORS["NCR"]["rgb"], dtype=np.float32)
    color_et = np.array(REGION_COLORS["ET"]["rgb"], dtype=np.float32)

    ed_mask = composite_slice == 2
    ncr_mask = composite_slice == 1
    et_mask = composite_slice == 3

    if np.any(ed_mask):
        overlay[ed_mask] = overlay[ed_mask] * (1.0 - mask_alpha) + color_ed * mask_alpha

    if np.any(ncr_mask):
        overlay[ncr_mask] = overlay[ncr_mask] * (1.0 - mask_alpha) + color_ncr * mask_alpha

    if np.any(et_mask):
        overlay[et_mask] = overlay[et_mask] * (1.0 - mask_alpha) + color_et * mask_alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _image_figure(img: np.ndarray, height: int = 440) -> go.Figure:
    """Create a responsive borderless Plotly 2D image figure."""
    fig = go.Figure()
    fig.add_trace(go.Image(z=img if img.ndim == 3 else np.stack([img] * 3, axis=-1)))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def slice_navigation_controls(
    num_slices: int,
    state_key: str = "current_slice",
    peak_slice: Optional[int] = None,
) -> int:
    """
    Render slice slider, previous/next buttons, and optional 'Jump to Peak Tumor' button.
    All panels reading state_key stay synchronized in st.session_state.
    """
    if state_key not in st.session_state or st.session_state[state_key] is None:
        st.session_state[state_key] = peak_slice if peak_slice is not None else num_slices // 2

    st.session_state[state_key] = int(np.clip(st.session_state[state_key], 0, max(0, num_slices - 1)))
    slider_key = f"{state_key}_slider"

    cols = st.columns([1, 6, 1, 2] if peak_slice is not None else [1, 7, 1])

    with cols[0]:
        if st.button("◀ Prev", key=f"{state_key}_prev", use_container_width=True):
            st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
            st.session_state[slider_key] = st.session_state[state_key]
            st.rerun()

    with cols[1]:
        if slider_key not in st.session_state:
            st.session_state[slider_key] = st.session_state[state_key]

        selected_slice = st.slider(
            "Axial slice",
            min_value=0,
            max_value=max(num_slices - 1, 0),
            value=int(st.session_state[state_key]),
            key=slider_key,
            label_visibility="collapsed",
        )
        st.session_state[state_key] = selected_slice

    with cols[2]:
        if st.button("Next ▶", key=f"{state_key}_next", use_container_width=True):
            st.session_state[state_key] = min(num_slices - 1, st.session_state[state_key] + 1)
            st.session_state[slider_key] = st.session_state[state_key]
            st.rerun()

    if peak_slice is not None and len(cols) > 3:
        with cols[3]:
            if st.button("🎯 Peak Tumor", key=f"{state_key}_peak", use_container_width=True, help="Jump to slice with largest tumor cross-section"):
                st.session_state[state_key] = peak_slice
                st.session_state[slider_key] = peak_slice
                st.rerun()

    st.caption(f"Axial Slice: **{st.session_state[state_key] + 1}** / {num_slices}" + (f" (Peak lesion slice: #{peak_slice + 1})" if peak_slice is not None else ""))
    return st.session_state[state_key]


def render_radiological_legend(selected_overlay: str = "Composite") -> None:
    """Render color-coded radiological legend for active overlay mode."""
    if selected_overlay == "Composite":
        st.markdown(
            f"""
            <div class="legend-box">
                <span style="font-size:0.80rem; font-weight:700; color:{COLORS['text_muted']};">REGIONS:</span>
                <div class="legend-item">
                    <span class="legend-swatch" style="background:{REGION_COLORS['ET']['hex']};"></span>
                    <span>Enhancing Tumor (ET)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-swatch" style="background:{REGION_COLORS['NCR']['hex']};"></span>
                    <span>Tumor Core (NCR)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-swatch" style="background:{REGION_COLORS['ED']['hex']};"></span>
                    <span>Peritumoral Edema (ED)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif selected_overlay == "Whole Tumor (WT)":
        st.markdown(
            f"""
            <div class="legend-box">
                <div class="legend-item">
                    <span class="legend-swatch" style="background:{REGION_COLORS['WT']['hex']};"></span>
                    <span>Whole Tumor (WT): Full abnormal region (Edema + Core)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif selected_overlay == "Tumor Core (TC)":
        st.markdown(
            f"""
            <div class="legend-box">
                <div class="legend-item">
                    <span class="legend-swatch" style="background:{REGION_COLORS['TC']['hex']};"></span>
                    <span>Tumor Core (TC): Non-enhancing necrotic core + Active enhancing tumor</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif selected_overlay == "Enhancing Tumor (ET)":
        st.markdown(
            f"""
            <div class="legend-box">
                <div class="legend-item">
                    <span class="legend-swatch" style="background:{REGION_COLORS['ET']['hex']};"></span>
                    <span>Enhancing Tumor (ET): Hyperintense gadolinium-enhancing rim/core</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_viewer(
    volume: np.ndarray,
    mask: Optional[np.ndarray],
    display_mode: str,
    region_masks: Optional[Dict[str, np.ndarray]] = None,
    composite: Optional[np.ndarray] = None,
    peak_slice: Optional[int] = None,
) -> None:
    """
    Main MRI viewer renderer supporting single-modality or multi-region overlays.
    """
    num_slices = volume.shape[2]
    slice_idx = slice_navigation_controls(num_slices, peak_slice=peak_slice)

    mri_slice = get_axial_slice(volume, slice_idx)

    # Determine which mask slice to use based on region selector
    region_selection = "Composite"
    if region_masks and len(region_masks) > 1 and display_mode in ["Segmentation", "Overlay", "Side-by-side"]:
        reg_col, _ = st.columns([3, 5])
        with reg_col:
            region_selection = st.selectbox(
                "Region Overlay",
                ["Composite (All Regions)", "Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"],
                index=0,
                key="viewer_region_select",
            )

    selected_mask_slice = None
    use_composite = False
    color = REGION_COLORS["WT"]["rgb"]

    if region_selection.startswith("Composite") and composite is not None:
        selected_mask_slice = get_mask_slice(composite, slice_idx)
        use_composite = True
    elif region_selection.startswith("Whole Tumor") and region_masks and "WT" in region_masks:
        selected_mask_slice = get_mask_slice(region_masks["WT"], slice_idx)
        use_composite = False
        color = REGION_COLORS["WT"]["rgb"]
    elif region_selection.startswith("Tumor Core") and region_masks and "TC" in region_masks:
        selected_mask_slice = get_mask_slice(region_masks["TC"], slice_idx)
        use_composite = False
        color = REGION_COLORS["TC"]["rgb"]
    elif region_selection.startswith("Enhancing") and region_masks and "ET" in region_masks:
        selected_mask_slice = get_mask_slice(region_masks["ET"], slice_idx)
        use_composite = False
        color = REGION_COLORS["ET"]["rgb"]
    elif mask is not None:
        selected_mask_slice = get_mask_slice(mask, slice_idx)
        use_composite = False
        color = REGION_COLORS["WT"]["rgb"]

    # Render according to display mode
    if display_mode == "Original":
        st.plotly_chart(_image_figure(mri_slice), use_container_width=True)

    elif display_mode == "Segmentation":
        if selected_mask_slice is None or not np.any(selected_mask_slice > 0):
            st.plotly_chart(_image_figure(np.zeros_like(mri_slice)), use_container_width=True)
            if selected_mask_slice is None:
                st.info("No segmentation mask available yet. Click '🔬 Run AI Brain Tumor Segmentation' above to generate predictions.")
            else:
                peak_note = f" Click '🎯 Peak Tumor' (slice #{peak_slice + 1}) to jump to active lesion." if peak_slice is not None else ""
                st.caption(f"No tumor mask present on slice {slice_idx + 1}.{peak_note}")
        else:
            if use_composite:
                seg_img = create_composite_overlay(np.zeros_like(mri_slice), selected_mask_slice, mask_alpha=1.0)
            else:
                seg_img = create_overlay(np.zeros_like(mri_slice), selected_mask_slice, mask_color=color, mask_alpha=1.0)
            st.plotly_chart(_image_figure(seg_img), use_container_width=True)
            render_radiological_legend(region_selection.split(" ")[0])

    elif display_mode == "Overlay":
        if selected_mask_slice is None or not np.any(selected_mask_slice > 0):
            st.plotly_chart(_image_figure(mri_slice), use_container_width=True)
            if selected_mask_slice is None:
                st.caption("Showing MRI only — awaiting segmentation analysis.")
            else:
                peak_note = f" Click '🎯 Peak Tumor' (slice #{peak_slice + 1}) to view active tumor." if peak_slice is not None else ""
                st.caption(f"No tumor mask on slice {slice_idx + 1} — showing original MRI.{peak_note}")
        else:
            if use_composite:
                overlay_img = create_composite_overlay(mri_slice, selected_mask_slice, mask_alpha=0.45)
            else:
                overlay_img = create_overlay(mri_slice, selected_mask_slice, mask_color=color, mask_alpha=0.45)
            st.plotly_chart(_image_figure(overlay_img), use_container_width=True)
            render_radiological_legend(region_selection.split(" ")[0])

    elif display_mode == "Side-by-side":
        left, right = st.columns(2)
        with left:
            render_panel_label("ORIGINAL MRI (FLAIR)")
            st.plotly_chart(_image_figure(mri_slice), use_container_width=True)
        with right:
            render_panel_label("AI SEGMENTATION OVERLAY")
            if selected_mask_slice is None or not np.any(selected_mask_slice > 0):
                st.plotly_chart(_image_figure(mri_slice), use_container_width=True)
                if selected_mask_slice is None:
                    st.caption("Awaiting segmentation analysis.")
                else:
                    st.caption(f"No tumor mask on slice {slice_idx + 1}.")
            else:
                if use_composite:
                    right_img = create_composite_overlay(mri_slice, selected_mask_slice, mask_alpha=0.45)
                else:
                    right_img = create_overlay(mri_slice, selected_mask_slice, mask_color=color, mask_alpha=0.45)
                st.plotly_chart(_image_figure(right_img), use_container_width=True)
        render_radiological_legend(region_selection.split(" ")[0])


def create_3d_visualization(
    segmentation: Optional[np.ndarray],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    region_masks: Optional[Dict[str, np.ndarray]] = None,
    max_points: int = 12000,
) -> Optional[go.Figure]:
    """
    Build a multi-region 3D scatter visualization of tumor sub-regions using Plotly.
    Points are subsampled per region to ensure responsive 60fps rendering in browser.
    """
    if segmentation is None or not np.any(segmentation > 0):
        return None

    sx, sy, sz = voxel_spacing
    traces = []
    rng = np.random.default_rng(seed=42)

    if region_masks and "ET" in region_masks and "TC" in region_masks and "WT" in region_masks:
        # Enhancing Tumor (ET)
        et_mask = region_masks["ET"] > 0
        if np.any(et_mask):
            coords_et = np.argwhere(et_mask)
            if coords_et.shape[0] > max_points // 3:
                idx = rng.choice(coords_et.shape[0], size=max_points // 3, replace=False)
                coords_et = coords_et[idx]
            traces.append(
                go.Scatter3d(
                    x=coords_et[:, 0] * sx,
                    y=coords_et[:, 1] * sy,
                    z=coords_et[:, 2] * sz,
                    mode="markers",
                    name="Enhancing Tumor (ET)",
                    marker=dict(size=2.5, color=REGION_COLORS["ET"]["hex"], opacity=0.85),
                )
            )

        # Non-enhancing core (NCR) = TC and not ET
        ncr_mask = (region_masks["TC"] > 0) & (~et_mask)
        if np.any(ncr_mask):
            coords_ncr = np.argwhere(ncr_mask)
            if coords_ncr.shape[0] > max_points // 3:
                idx = rng.choice(coords_ncr.shape[0], size=max_points // 3, replace=False)
                coords_ncr = coords_ncr[idx]
            traces.append(
                go.Scatter3d(
                    x=coords_ncr[:, 0] * sx,
                    y=coords_ncr[:, 1] * sy,
                    z=coords_ncr[:, 2] * sz,
                    mode="markers",
                    name="Tumor Core (NCR)",
                    marker=dict(size=2.2, color=REGION_COLORS["NCR"]["hex"], opacity=0.70),
                )
            )

        # Edema (ED) = WT and not TC
        ed_mask = (region_masks["WT"] > 0) & (~(region_masks["TC"] > 0))
        if np.any(ed_mask):
            coords_ed = np.argwhere(ed_mask)
            if coords_ed.shape[0] > max_points // 3:
                idx = rng.choice(coords_ed.shape[0], size=max_points // 3, replace=False)
                coords_ed = coords_ed[idx]
            traces.append(
                go.Scatter3d(
                    x=coords_ed[:, 0] * sx,
                    y=coords_ed[:, 1] * sy,
                    z=coords_ed[:, 2] * sz,
                    mode="markers",
                    name="Peritumoral Edema (ED)",
                    marker=dict(size=1.8, color=REGION_COLORS["ED"]["hex"], opacity=0.35),
                )
            )
    else:
        # Single region fallback
        coords = np.argwhere(segmentation > 0)
        if coords.shape[0] > max_points:
            idx = rng.choice(coords.shape[0], size=max_points, replace=False)
            coords = coords[idx]
        traces.append(
            go.Scatter3d(
                x=coords[:, 0] * sx,
                y=coords[:, 1] * sy,
                z=coords[:, 2] * sz,
                mode="markers",
                name="Whole Tumor (WT)",
                marker=dict(size=2.0, color=REGION_COLORS["WT"]["hex"], opacity=0.65),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (mm)", backgroundcolor="rgba(14,24,42,0.5)", gridcolor="rgba(30,47,77,0.5)"),
            yaxis=dict(title="Y (mm)", backgroundcolor="rgba(14,24,42,0.5)", gridcolor="rgba(30,47,77,0.5)"),
            zaxis=dict(title="Z (mm)", backgroundcolor="rgba(14,24,42,0.5)", gridcolor="rgba(30,47,77,0.5)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=10, b=10),
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(14,24,42,0.85)",
            bordercolor="rgba(30,47,77,0.8)",
            borderwidth=1,
            font=dict(color="#F8FAFC", size=11),
        ),
    )
    return fig
