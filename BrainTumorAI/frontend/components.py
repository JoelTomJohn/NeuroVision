"""
frontend/components.py

NeuroVision — AI-Assisted Brain MRI Analysis
Reusable clinical UI components, metric cards, status badges, and empty states.
"""

from typing import List, Dict, Any, Optional
import streamlit as st
from frontend.styles import COLORS, REGION_COLORS


def demo_badge_html() -> str:
    """HTML badge for simulated demo data."""
    return '<span class="badge badge-demo">Demo Mode — Simulated Placeholder Data</span>'


def real_badge_html() -> str:
    """HTML badge for real deep-learning inference."""
    return '<span class="badge badge-real">MONAI SegResNet — Verified Active Inference</span>'


def status_badge_html(status: str) -> str:
    """Return status pill badge."""
    if status == "success":
        return '<span class="badge badge-real">Analysis Completed</span>'
    elif status == "demo":
        return demo_badge_html()
    else:
        return '<span class="badge" style="background:rgba(148,163,184,0.15); color:#94A3B8; border:1px solid rgba(148,163,184,0.3);">Awaiting Input</span>'


def render_header(title: str, subtitle: str) -> None:
    """Render standardized top clinical page header with NeuroVision branding."""
    st.markdown(
        f"""
        <div class="neuro-header">
            <div>
                <h1 class="neuro-brand-title">
                    🧠 Neuro<span class="brand-accent">Vision</span>
                </h1>
                <div class="neuro-brand-tagline">{subtitle}</div>
            </div>
            <div>
                <span class="prototype-pill">Research Prototype</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(text: str, icon: str = "") -> None:
    """Render consistent section header."""
    icon_str = f"<span>{icon}</span> " if icon else ""
    st.markdown(f'<div class="section-title">{icon_str}{text}</div>', unsafe_allow_html=True)


def render_disclaimer(text: Optional[str] = None) -> None:
    """Render discreet responsible-use disclaimer."""
    default_text = (
        "NeuroVision is an investigative AI research prototype developed for academic evaluation. "
        "It is NOT a certified medical device and does not substitute for certified clinical "
        "judgment or radiologist interpretation. Do not use for patient treatment decisions."
    )
    content = text or default_text
    st.markdown(
        f"""
        <div class="disclaimer-note">
            <strong>Responsible-Use Notice:</strong> {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: Any,
    unit: str = "",
    subtext: str = "",
    is_demo: bool = False,
    accent_color: Optional[str] = None,
) -> None:
    """Render clinical metric card."""
    accent_bar = f'style="border-top: 3px solid {accent_color};"' if accent_color else ""
    demo_html = f'<div style="margin-top:0.4rem;">{demo_badge_html()}</div>' if is_demo else ""
    subtext_html = f'<div class="metric-subtext">{subtext}</div>' if subtext else ""

    st.markdown(
        f"""
        <div class="metric-card" {accent_bar}>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
            {subtext_html}
            {demo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: List[Dict[str, Any]]) -> None:
    """Render horizontal row of metric cards."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_metric_card(
                label=m.get("label", ""),
                value=m.get("value", "—"),
                unit=m.get("unit", ""),
                subtext=m.get("subtext", ""),
                is_demo=m.get("is_demo", False),
                accent_color=m.get("accent_color"),
            )


def render_modality_cards() -> None:
    """Render 4 clinical modality explanation cards for start analysis section."""
    cols = st.columns(4)

    modalities = [
        {
            "name": "T1c",
            "full": "T1 Contrast-Enhanced",
            "desc": "Highlights active blood-brain barrier disruption and delineates vascularized enhancing tumor margins.",
            "format": ".nii / .nii.gz",
        },
        {
            "name": "T1",
            "full": "T1 Native",
            "desc": "Anatomical reference sequence showing baseline brain parenchyma, ventricles, and healthy gray/white matter boundaries.",
            "format": ".nii / .nii.gz",
        },
        {
            "name": "T2",
            "full": "T2-Weighted",
            "desc": "Detects abnormal tissue water content, hyperintense edema, and structural mass effect across axial planes.",
            "format": ".nii / .nii.gz",
        },
        {
            "name": "FLAIR",
            "full": "T2 FLAIR",
            "desc": "Suppresses free fluid (CSF) signals to clearly isolate hyperintense peritumoral vasogenic edema margins.",
            "format": ".nii / .nii.gz",
        },
    ]

    for col, m in zip(cols, modalities):
        with col:
            st.markdown(
                f"""
                <div class="clinical-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                        <span class="modality-badge">{m['name']}</span>
                        <span style="font-size:0.72rem; color:{COLORS['text_dim']};">{m['format']}</span>
                    </div>
                    <div class="clinical-card-header">{m['full']}</div>
                    <div class="clinical-card-body">{m['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_feature_grid() -> None:
    """Render clean feature overview cards on the Dashboard."""
    features = [
        {
            "icon": "🧠",
            "title": "Multimodal MRI Segmentation",
            "desc": "Deep learning segmentation isolating Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) from co-registered scans.",
        },
        {
            "icon": "📏",
            "title": "Quantitative Volumetric Analysis",
            "desc": "Automatic measurement of tumor volume (cm³), 3D orthogonal bounding-box extents (mm), and subregion fraction ratios.",
        },
        {
            "icon": "🎯",
            "title": "Confidence & Uncertainty",
            "desc": "Voxel-wise model probability maps and transparent explanations distinguishing probability heuristics from clinical certainty.",
        },
        {
            "icon": "🧊",
            "title": "Interactive 3D Volumetric Exploration",
            "desc": "Hardware-accelerated 3D point-cloud exploration enabling spatial rotation, zoom, and multi-region layer separation.",
        },
        {
            "icon": "📄",
            "title": "Clinical Summary PDF Reporting",
            "desc": "One-click generation of formatted radiological summary reports complete with scan metadata, measurements, and disclaimers.",
        },
    ]

    cols = st.columns(len(features))
    for col, f in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="clinical-card">
                    <div style="font-size:1.8rem; margin-bottom:0.5rem;">{f['icon']}</div>
                    <div class="clinical-card-header" style="font-size:0.95rem;">{f['title']}</div>
                    <div class="clinical-card-body" style="font-size:0.82rem;">{f['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_empty_analysis_state() -> None:
    """Render welcoming empty state inviting user to start first analysis."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">🔬</div>
            <div class="empty-state-title">No Scan Analyzed in Current Session</div>
            <div class="empty-state-text">
                Ready to begin. Upload your 4-modality brain MRI dataset (T1c, T1, T2, FLAIR) in NIfTI format, 
                or load the pre-bundled benchmark sample scan for immediate evaluation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_benchmark_evaluation_card(eval_res: Dict[str, Any]) -> None:
    """Render ground truth benchmark Dice score evaluation results."""
    if not eval_res.get("available"):
        return

    metrics = eval_res.get("metrics", {})
    st.markdown(
        f"""
        <div style="background:{COLORS['surface']}; border:1px solid {COLORS['accent_teal']}; border-radius:10px; padding:1.2rem; margin:1.0rem 0;">
            <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF; margin-bottom:0.4rem; display:flex; align-items:center; gap:0.5rem;">
                <span>🎯</span> Ground-Truth Benchmark Evaluation (BraTS-GLI-00000-000)
            </div>
            <div style="font-size:0.85rem; color:{COLORS['text_muted']}; margin-bottom:1.0rem;">
                Predictions rigorously compared against expert neuroradiologist ground-truth annotation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    if "WT" in metrics:
        wt = metrics["WT"]
        with cols[0]:
            render_metric_card(
                "Whole Tumor (WT) Dice",
                f"{wt['dice_pct']}%",
                unit=f"({wt['dice']:.4f})",
                subtext=f"Pred: {wt['pred_voxels']:,} / GT: {wt['gt_voxels']:,} voxels",
                accent_color=REGION_COLORS["WT"]["hex"],
            )
    if "TC" in metrics:
        tc = metrics["TC"]
        with cols[1]:
            render_metric_card(
                "Tumor Core (TC) Dice",
                f"{tc['dice_pct']}%",
                unit=f"({tc['dice']:.4f})",
                subtext=f"Pred: {tc['pred_voxels']:,} / GT: {tc['gt_voxels']:,} voxels",
                accent_color=REGION_COLORS["TC"]["hex"],
            )
    if "ET" in metrics:
        et = metrics["ET"]
        with cols[2]:
            render_metric_card(
                "Enhancing Tumor (ET) Dice",
                f"{et['dice_pct']}%",
                unit=f"({et['dice']:.4f})",
                subtext=f"Pred: {et['pred_voxels']:,} / GT: {et['gt_voxels']:,} voxels",
                accent_color=REGION_COLORS["ET"]["hex"],
            )
