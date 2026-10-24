"""
frontend/viewer.py

Everything related to *looking at* an MRI volume:
- extracting a single axial slice from a 3D numpy array
- building an overlay image (MRI + tumor mask)
- rendering the four display modes (Original / Segmentation /
  Overlay / Side-by-side) with Plotly
- the slice slider + prev/next controls, kept in sync via
  st.session_state so both panels always show the same slice.

Nothing here talks to nnU-Net directly. It only ever receives numpy
arrays, so it can be wired to a real segmentation mask later without
any changes to this file.
"""

from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ----------------------------------------------------------------------
# Slice extraction
# ----------------------------------------------------------------------
def get_axial_slice(volume: np.ndarray, slice_index: int) -> np.ndarray:
    """Return a single axial (Z) slice from a 3D volume, normalized to
    0-255 for consistent display regardless of the scanner's intensity
    range."""
    slice_index = int(np.clip(slice_index, 0, volume.shape[2] - 1))
    img = volume[:, :, slice_index].astype(np.float32)

    img_min, img_max = np.nanmin(img), np.nanmax(img)
    if img_max - img_min > 1e-6:
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        img = np.zeros_like(img)

    # Rotate so the image displays in a conventional radiological
    # orientation (superior at top).
    return np.rot90(img)


# ----------------------------------------------------------------------
# Overlay creation
# ----------------------------------------------------------------------
def create_overlay(
    mri_slice: np.ndarray,
    mask_slice: Optional[np.ndarray],
    mask_color: tuple = (231, 76, 60),  # red-ish, distinguishable from grayscale MRI
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """
    Combine a grayscale MRI slice with a tumor segmentation mask slice
    into an RGB image where the tumor region is highlighted.

    This is a *visualization* of the segmentation mask, not a
    diagnosis. If mask_slice is None, the plain MRI (as RGB) is
    returned so the function is always safe to call.
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
        overlay[mask_bool] * (1 - mask_alpha) + color * mask_alpha
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Plotly rendering helpers
# ----------------------------------------------------------------------
def _image_figure(img: np.ndarray, height: int = 420) -> go.Figure:
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


# ----------------------------------------------------------------------
# Slice navigation controls (kept in sync across panels via session_state)
# ----------------------------------------------------------------------
def slice_navigation_controls(num_slices: int, state_key: str = "current_slice") -> int:
    """Renders a slider + prev/next buttons bound to a single
    session_state key, so every panel that reads state_key stays in
    sync automatically."""
    if state_key not in st.session_state:
        st.session_state[state_key] = num_slices // 2

    col_prev, col_slider, col_next = st.columns([1, 6, 1])

    with col_prev:
        if st.button("◀", key=f"{state_key}_prev", use_container_width=True):
            st.session_state[state_key] = max(0, st.session_state[state_key] - 1)

    with col_slider:
        st.session_state[state_key] = st.slider(
            "Axial slice",
            min_value=0,
            max_value=max(num_slices - 1, 0),
            value=int(st.session_state[state_key]),
            key=f"{state_key}_slider",
            label_visibility="collapsed",
        )

    with col_next:
        if st.button("▶", key=f"{state_key}_next", use_container_width=True):
            st.session_state[state_key] = min(num_slices - 1, st.session_state[state_key] + 1)

    st.caption(f"Slice: {st.session_state[state_key] + 1} / {num_slices}")
    return st.session_state[state_key]


# ----------------------------------------------------------------------
# Main render entry point used by the MRI Analysis page
# ----------------------------------------------------------------------
def render_viewer(volume: np.ndarray, mask: Optional[np.ndarray], display_mode: str) -> None:
    """
    display_mode: one of "Original", "Segmentation", "Overlay", "Side-by-side"

    The slice index is stored once in st.session_state["current_slice"]
    so that regardless of display mode, changing the slider updates
    every panel that is currently visible.
    """
    num_slices = volume.shape[2]
    slice_idx = slice_navigation_controls(num_slices)

    mri_slice = get_axial_slice(volume, slice_idx)
    mask_slice = get_axial_slice(mask, slice_idx) if mask is not None else None

    if display_mode == "Original":
        st.plotly_chart(_image_figure(mri_slice), use_container_width=True)

    elif display_mode == "Segmentation":
        if mask_slice is None:
            st.info("No segmentation mask available yet. Run analysis or connect the model.")
        else:
            seg_only = create_overlay(np.zeros_like(mri_slice), mask_slice)
            st.plotly_chart(_image_figure(seg_only), use_container_width=True)

    elif display_mode == "Overlay":
        overlay_img = create_overlay(mri_slice, mask_slice)
        st.plotly_chart(_image_figure(overlay_img), use_container_width=True)
        if mask_slice is None:
            st.caption("Showing MRI only — no segmentation mask to overlay yet.")

    elif display_mode == "Side-by-side":
        from frontend.components import render_panel_label  # local import avoids cycle

        left, right = st.columns(2)
        with left:
            render_panel_label("ORIGINAL MRI")
            st.plotly_chart(_image_figure(mri_slice), use_container_width=True)
        with right:
            render_panel_label("AI SEGMENTATION")
            right_img = create_overlay(mri_slice, mask_slice)
            st.plotly_chart(_image_figure(right_img), use_container_width=True)
            if mask_slice is None:
                st.caption("No segmentation mask yet — showing MRI only.")


# ----------------------------------------------------------------------
# 3D visualization
# ----------------------------------------------------------------------
def create_3d_visualization(
    segmentation: Optional[np.ndarray],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_points: int = 15000,
) -> Optional[go.Figure]:
    """
    Build a 3D scatter render of a segmentation mask using Plotly.

    Returns None if no segmentation is available, so the caller can
    show a placeholder message instead. Points are randomly subsampled
    above max_points to keep the browser responsive.
    """
    if segmentation is None or not np.any(segmentation > 0):
        return None

    coords = np.argwhere(segmentation > 0)
    if coords.shape[0] > max_points:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[idx]

    sx, sy, sz = voxel_spacing
    x = coords[:, 0] * sx
    y = coords[:, 1] * sy
    z = coords[:, 2] * sz

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=2, color="#E74C3C", opacity=0.6),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
