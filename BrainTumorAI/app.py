"""
app.py

NeuroVision — AI-Assisted Brain MRI Analysis
A clinical research prototype for multimodal brain MRI segmentation,
visualization, and quantitative analysis.

Launch command:
    streamlit run app.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path regardless of launch directory
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import numpy as np
import streamlit as st

from frontend.styles import inject_global_css, COLORS, REGION_COLORS
from frontend.components import (
    render_header,
    render_section_heading,
    render_disclaimer,
    render_metric_row,
    render_metric_card,
    render_modality_cards,
    render_feature_grid,
    render_empty_analysis_state,
    render_benchmark_evaluation_card,
    status_badge_html,
    demo_badge_html,
    real_badge_html,
)
from frontend.viewer import (
    render_viewer,
    create_3d_visualization,
    get_axial_slice,
    create_composite_overlay,
)
from backend.preprocessing import (
    load_nifti,
    load_multimodal_nifti,
    load_sample_case,
    has_sample_case,
    NiftiLoadError,
    describe_modalities_note,
)
from backend.inference import (
    run_segresnet_inference,
    run_nnunet_inference,
    is_model_available,
)
from backend.analysis import (
    calculate_tumor_measurements,
    calculate_confidence,
    evaluate_sample_benchmark,
)
from backend.report import generate_pdf_report


# ========================================================================
# Page Configuration & Clinical Styling
# ========================================================================

st.set_page_config(
    page_title="NeuroVision — AI-Assisted Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


# ========================================================================
# Session State Initialization (Demo Mode default: OFF)
# ========================================================================

DEFAULTS = {
    "mri_volume": None,
    "mri_modalities": None,
    "mri_affine": None,
    "voxel_spacing": (1.0, 1.0, 1.0),
    "file_name": None,
    "file_size_kb": None,
    "modality_filenames": {},
    "analysis_results": None,
    "measurements": None,
    "confidence_summary": None,
    "demo_mode": False,              # Requirement: Default to OFF for real inference
    "current_slice": None,
    "last_report_bytes": None,
    "nav_page": "🏠 Dashboard",
    "gt_seg": None,                  # Separate ground truth (never uploaded as input)
    "benchmark_eval": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ========================================================================
# Sidebar Navigation & System Controls
# ========================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 0.8rem 0;">
            <div style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                🧠 Neuro<span style="color: #00B4D8;">Vision</span>
            </div>
            <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 0.15rem;">
                AI-Assisted Brain MRI Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active session status indicator
    if st.session_state["mri_volume"] is not None:
        case_label = st.session_state["file_name"] or "Active Scan"
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 0.4rem 0.6rem; font-size: 0.78rem; color: #10B981; margin-bottom: 0.8rem;">
                ● <b>Scan Loaded:</b> {case_label[:22]}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 6px; padding: 0.4rem 0.6rem; font-size: 0.78rem; color: #94A3B8; margin-bottom: 0.8rem;">
                ○ No MRI scan loaded
            </div>
            """,
            unsafe_allow_html=True,
        )

    nav_options = [
        "🏠 Dashboard",
        "🧠 MRI Analysis",
        "📊 Tumor Analysis",
        "🎯 Confidence",
        "🧊 3D Visualization",
        "📄 Clinical Report",
        "ℹ️ Technical Specs",
    ]

    # Handle programmatic navigation (e.g. from Dashboard CTA buttons)
    if st.session_state["nav_page"] not in nav_options:
        st.session_state["nav_page"] = "🏠 Dashboard"

    current_idx = nav_options.index(st.session_state["nav_page"])

    selected_page = st.radio(
        "Navigation",
        nav_options,
        index=current_idx,
        key="sidebar_nav_radio",
        label_visibility="collapsed",
    )
    if selected_page != st.session_state["nav_page"]:
        st.session_state["nav_page"] = selected_page

    st.divider()

    # Demo Mode Toggle (Default: OFF)
    st.markdown("##### ⚙️ Execution Mode")
    demo_toggle = st.toggle(
        "Demo Mode",
        value=st.session_state["demo_mode"],
        help="Toggle between real MONAI SegResNet inference and simulated demo data.",
    )
    st.session_state["demo_mode"] = demo_toggle

    if st.session_state["demo_mode"]:
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#F5B041; background:rgba(245,176,65,0.12); padding:0.4rem 0.6rem; border-radius:5px; border:1px solid rgba(245,176,65,0.3);">
                <b>Demo Mode: ON</b><br>Generates simulated placeholder tumor geometry for UI evaluation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#10B981; background:rgba(16,185,129,0.12); padding:0.4rem 0.6rem; border-radius:5px; border:1px solid rgba(16,185,129,0.3);">
                <b>Real Model: ACTIVE</b><br>Runs MONAI SegResNet 3D inference using the verified model checkpoint.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Fast Quick-Load for Presentation
    if has_sample_case():
        st.markdown("##### ⚡ Presentation Quick-Load")
        if st.button("Load BraTS Benchmark Scan", use_container_width=True, help="Load pre-bundled BraTS-GLI-00000-000 sample scan for demonstration"):
            try:
                with st.spinner("Loading benchmark dataset..."):
                    modalities, affine, spacing, fnames, gt = load_sample_case()
                    st.session_state["mri_modalities"] = modalities
                    st.session_state["mri_affine"] = affine
                    st.session_state["mri_volume"] = modalities[3]  # FLAIR
                    st.session_state["voxel_spacing"] = spacing
                    st.session_state["file_name"] = "BraTS-GLI-00000-000"
                    st.session_state["file_size_kb"] = 9600.0
                    st.session_state["modality_filenames"] = fnames
                    st.session_state["gt_seg"] = gt
                    st.session_state["current_slice"] = modalities.shape[3] // 2
                    st.session_state["analysis_results"] = None
                    st.session_state["measurements"] = None
                    st.session_state["confidence_summary"] = None
                    st.session_state["benchmark_eval"] = None
                    st.session_state["nav_page"] = "🧠 MRI Analysis"
                    st.success("BraTS sample loaded!")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not load sample: {e}")

    st.caption("Research prototype. Not for primary diagnostic use.")


# ========================================================================
# 1. Dashboard Page (Clean Medical-Technology Design)
# ========================================================================

def page_dashboard():
    # A. Header
    render_header(
        "NeuroVision",
        "AI-Assisted Brain MRI Analysis",
    )

    # B. Welcome Hero Banner
    st.markdown(
        f"""
        <div class="welcome-hero">
            <h2>Brain MRI Analysis Platform</h2>
            <p>
                NeuroVision is a clinical research prototype designed for multimodal 3D brain MRI segmentation, 
                sub-region volumetric measurement, and uncertainty quantification. Powered by a MONAI SegResNet 
                deep learning architecture, it enables automated delineation of whole tumor, tumor core, and enhancing 
                tumor boundaries from standard diagnostic sequences.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prominent Call to Action Button
    col_cta, col_empty = st.columns([3, 5])
    with col_cta:
        if st.button("🚀 Start New Analysis", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "🧠 MRI Analysis"
            st.rerun()

    st.write("")

    # C. Start Analysis Section: 4 Modality Requirement Explanation
    render_section_heading("Required MRI Modalities", "📋")
    st.markdown(
        """
        <div style="font-size:0.92rem; color:#94A3B8; margin-bottom:0.9rem;">
            Multimodal MRI analysis requires four co-registered, skull-stripped 3D volumes in NIfTI format 
            (<code>.nii</code> or <code>.nii.gz</code>) conforming to the BraTS protocol:
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_modality_cards()

    st.write("")

    # D. Feature Overview (Only Genuine Implemented Features)
    render_section_heading("Platform Capabilities", "✨")
    render_feature_grid()

    st.write("")

    # E. Recent Analyses / Current Session Status
    render_section_heading("Session Analysis Status", "📊")

    results = st.session_state.get("analysis_results")
    if results and results.get("success"):
        meas = st.session_state.get("measurements") or {}
        conf = st.session_state.get("confidence_summary") or {}
        is_demo = results.get("is_demo", False)

        st.markdown(
            f"""
            <div class="clinical-card" style="border-left: 4px solid {COLORS['success']};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.1rem; font-weight:700; color:#FFFFFF;">
                        Current Active Scan: {st.session_state['file_name'] or 'MRI Volume'}
                    </span>
                    <span>{status_badge_html('demo' if is_demo else 'success')}</span>
                </div>
                <div style="font-size:0.88rem; color:#94A3B8; margin-top:0.6rem;">
                    Whole Tumor Volume: <b>{meas.get('volume_cm3', '—')} cm³</b> | 
                    Mean Confidence: <b>{conf.get('mean', '—')}</b> | 
                    Modality Set: <b>T1c, T1, T2, FLAIR</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        c1, c2, _ = st.columns([2, 2, 4])
        with c1:
            if st.button("View Segmentation Viewer", use_container_width=True):
                st.session_state["nav_page"] = "🧠 MRI Analysis"
                st.rerun()
        with c2:
            if st.button("View Volumetric Metrics", use_container_width=True):
                st.session_state["nav_page"] = "📊 Tumor Analysis"
                st.rerun()
    else:
        render_empty_analysis_state()

    # F. Responsible-Use Note
    render_disclaimer()


# ========================================================================
# 2. MRI Analysis Page (Upload, Real Inference & Radiological Viewer)
# ========================================================================

def page_mri_analysis():
    render_header(
        "MRI Analysis",
        "Multimodal upload, deep learning inference, and interactive slice viewer",
    )

    render_section_heading("1. Input MRI Modalities (NIfTI)", "📁")

    st.markdown(
        """
        <div style="font-size:0.88rem; color:#94A3B8; margin-bottom:0.8rem;">
            Upload four co-registered 3D volumes (<code>.nii</code> or <code>.nii.gz</code>) or load 
            the bundled BraTS benchmark scan for immediate verification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sample Quick Load Button
    if has_sample_case():
        sc_col1, sc_col2 = st.columns([3, 5])
        with sc_col1:
            if st.button("⚡ Quick-Load BraTS Benchmark Scan", use_container_width=True):
                try:
                    with st.spinner("Loading benchmark dataset (BraTS-GLI-00000-000)..."):
                        modalities, affine, spacing, fnames, gt = load_sample_case()
                        st.session_state["mri_modalities"] = modalities
                        st.session_state["mri_affine"] = affine
                        st.session_state["mri_volume"] = modalities[3]  # FLAIR for baseline display
                        st.session_state["voxel_spacing"] = spacing
                        st.session_state["file_name"] = "BraTS-GLI-00000-000"
                        st.session_state["file_size_kb"] = 9600.0
                        st.session_state["modality_filenames"] = fnames
                        st.session_state["gt_seg"] = gt
                        st.session_state["current_slice"] = modalities.shape[3] // 2
                        st.session_state["analysis_results"] = None
                        st.session_state["measurements"] = None
                        st.session_state["confidence_summary"] = None
                        st.session_state["benchmark_eval"] = None
                        st.success("BraTS-GLI-00000-000 benchmark dataset loaded successfully.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Sample load failed: {e}")

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        t1c_file = st.file_uploader(
            "1. T1c (T1 Contrast-Enhanced)",
            type=["nii", "gz", "nii.gz"],
            key="t1c_upload",
            help="Required for enhancing tumor margin identification.",
        )
        t1_file = st.file_uploader(
            "2. T1 (T1 Native / Non-Contrast)",
            type=["nii", "gz", "nii.gz"],
            key="t1_upload",
            help="Anatomical reference sequence.",
        )

    with col_up2:
        t2_file = st.file_uploader(
            "3. T2 (T2-Weighted)",
            type=["nii", "gz", "nii.gz"],
            key="t2_upload",
            help="Highlights hyperintense fluid and edema boundary.",
        )
        flair_file = st.file_uploader(
            "4. FLAIR (T2 Fluid-Attenuated)",
            type=["nii", "gz", "nii.gz"],
            key="flair_upload",
            help="Suppresses normal CSF to highlight peritumoral edema.",
        )

    uploaded_files_present = all([t1c_file, t1_file, t2_file, flair_file])

    # If new manual files uploaded, load them into session
    if uploaded_files_present:
        st.success("All four MRI modalities provided.")
        with st.expander("Inspected Modality Files"):
            st.write(f"- **T1c:** {t1c_file.name}")
            st.write(f"- **T1:** {t1_file.name}")
            st.write(f"- **T2:** {t2_file.name}")
            st.write(f"- **FLAIR:** {flair_file.name}")

    # Analysis Action Section
    st.write("")
    render_section_heading("2. AI Segmentation Execution", "⚡")

    can_run = uploaded_files_present or (st.session_state["mri_modalities"] is not None)

    if not can_run:
        st.info("Please upload all four MRI modalities or click 'Quick-Load BraTS Benchmark Scan' above to proceed.")
    else:
        active_name = st.session_state["file_name"] or (flair_file.name if flair_file else "Active Scan")
        st.caption(f"Target Scan: **{active_name}** | Mode: **{'Demo Mode' if st.session_state['demo_mode'] else 'MONAI SegResNet (Real Inference)'}**")

        run_clicked = st.button("🔬 Run AI Brain Tumor Segmentation", type="primary", use_container_width=True)

        if run_clicked:
            # 1. Process files if uploaded via UI
            if uploaded_files_present:
                with st.spinner("Validating and stacking 4D MRI volume..."):
                    try:
                        modality_dict = {
                            "t1c": (t1c_file.getvalue(), t1c_file.name),
                            "t1": (t1_file.getvalue(), t1_file.name),
                            "t2": (t2_file.getvalue(), t2_file.name),
                            "flair": (flair_file.getvalue(), flair_file.name),
                        }
                        modalities, affine, spacing = load_multimodal_nifti(modality_dict)
                        st.session_state["mri_modalities"] = modalities
                        st.session_state["mri_affine"] = affine
                        st.session_state["mri_volume"] = modalities[3]  # FLAIR
                        st.session_state["voxel_spacing"] = spacing
                        st.session_state["file_name"] = flair_file.name
                        st.session_state["gt_seg"] = None  # User upload has no ground truth
                        st.session_state["current_slice"] = modalities.shape[3] // 2
                    except NiftiLoadError as e:
                        st.error(f"NIfTI Validation Error: {e}")
                        st.stop()
                    except Exception as e:
                        st.error(f"Unexpected error loading volumes: {e}")
                        st.stop()

            # 2. Run Deep Learning Model Inference
            mode_desc = "simulated demo pipeline" if st.session_state["demo_mode"] else "MONAI SegResNet 3D model (CPU-optimized)"
            with st.spinner(f"Executing {mode_desc}... This takes ~45 seconds on CPU."):
                res = run_segresnet_inference(
                    st.session_state["mri_modalities"],
                    demo_mode=st.session_state["demo_mode"],
                )
                st.session_state["analysis_results"] = res

                if not res.get("success"):
                    st.error(res.get("message", "Inference failed."))
                    st.stop()

                # 3. Calculate Multi-Region Measurements
                meas = calculate_tumor_measurements(
                    segmentation=res.get("segmentation"),
                    voxel_spacing=st.session_state["voxel_spacing"],
                    region_masks=res.get("region_masks"),
                )
                st.session_state["measurements"] = meas

                # Sync current slice to peak tumor slice if found
                if meas.get("peak_slice") is not None:
                    st.session_state["current_slice"] = meas["peak_slice"]

                # 4. Calculate Confidence Summary
                conf = calculate_confidence(
                    confidence_map=res.get("confidence_map"),
                    mask=res.get("segmentation"),
                    region_masks=res.get("region_masks"),
                )
                st.session_state["confidence_summary"] = conf

                # 5. Evaluate Benchmark Dice if ground truth is present
                if st.session_state.get("gt_seg") is not None:
                    eval_res = evaluate_sample_benchmark(
                        pred_regions=res.get("region_masks", {}),
                        gt_seg=st.session_state["gt_seg"],
                    )
                    st.session_state["benchmark_eval"] = eval_res

                st.session_state["last_report_bytes"] = None
                st.success("AI Segmentation Completed Successfully.")
                st.rerun()

    # MRI Viewer Section
    st.write("")
    render_section_heading("3. Interactive Slice Viewer", "🔍")

    volume = st.session_state.get("mri_volume")
    if volume is None:
        render_empty_analysis_state()
        return

    results = st.session_state.get("analysis_results") or {}
    mask = results.get("segmentation")
    composite = results.get("composite")
    region_masks = results.get("region_masks")
    meas = st.session_state.get("measurements") or {}
    peak_slice = meas.get("peak_slice")

    c_mode, c_status = st.columns([6, 4])
    with c_mode:
        display_mode = st.radio(
            "Viewer Display Mode",
            ["Original", "Segmentation", "Overlay", "Side-by-side"],
            index=2,  # Overlay default
            horizontal=True,
            key="viewer_display_mode",
        )
    with c_status:
        if results.get("success"):
            if results.get("is_demo"):
                st.markdown(f'<div style="text-align:right;">{demo_badge_html()}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:right;">{real_badge_html()}</div>', unsafe_allow_html=True)

    render_viewer(
        volume=volume,
        mask=mask,
        display_mode=display_mode,
        region_masks=region_masks,
        composite=composite,
        peak_slice=peak_slice,
    )

    # Benchmark Ground-Truth Evaluation Card if benchmark data was evaluated
    if st.session_state.get("benchmark_eval"):
        render_benchmark_evaluation_card(st.session_state["benchmark_eval"])


# ========================================================================
# 3. Tumor Analysis Page (Quantitative Multi-Region Breakdown)
# ========================================================================

def page_tumor_analysis():
    render_header(
        "Tumor Analysis",
        "Quantitative volumetric measurements, morphometry, and sub-region statistics",
    )

    meas = st.session_state.get("measurements")
    results = st.session_state.get("analysis_results") or {}
    is_demo = results.get("is_demo", False)

    if not meas or not meas.get("available"):
        render_empty_analysis_state()
        return

    if is_demo:
        st.markdown(demo_badge_html(), unsafe_allow_html=True)
        st.write("")

    # Primary Whole Tumor Overview Metrics
    render_section_heading("Primary Whole Tumor (WT) Metrics", "📏")
    render_metric_row([
        {
            "label": "Whole Tumor Volume",
            "value": meas.get("volume_cm3", "—"),
            "unit": "cm³",
            "subtext": "Total abnormal tissue burden",
            "is_demo": is_demo,
            "accent_color": REGION_COLORS["WT"]["hex"],
        },
        {
            "label": "Lesion Length (X)",
            "value": meas.get("length_mm", "—"),
            "unit": "mm",
            "subtext": "Right-to-Left physical extent",
            "is_demo": is_demo,
        },
        {
            "label": "Lesion Width (Y)",
            "value": meas.get("width_mm", "—"),
            "unit": "mm",
            "subtext": "Anterior-Posterior extent",
            "is_demo": is_demo,
        },
        {
            "label": "Lesion Height (Z)",
            "value": meas.get("height_mm", "—"),
            "unit": "mm",
            "subtext": "Inferior-Superior extent",
            "is_demo": is_demo,
        },
    ])

    st.write("")

    # Multi-Region Sub-Compartment Breakdown
    regions = meas.get("regions", {})
    if regions:
        render_section_heading("Hierarchical Sub-Region Breakdown (BraTS Standard)", "🧩")

        sub_cols = st.columns(3)

        # Whole Tumor Card
        wt_info = regions.get("WT", {})
        with sub_cols[0]:
            st.markdown(
                f"""
                <div class="clinical-card" style="border-top: 4px solid {REGION_COLORS['WT']['hex']};">
                    <div class="clinical-card-header" style="color:{REGION_COLORS['WT']['hex']};">Whole Tumor (WT)</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#FFFFFF; margin:0.3rem 0;">
                        {wt_info.get('volume_cm3', '—')} <span style="font-size:0.85rem; color:#94A3B8;">cm³</span>
                    </div>
                    <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
                        • <b>Voxel count:</b> {wt_info.get('voxel_count', 0):,}<br>
                        • <b>Dimensions:</b> {wt_info.get('length_mm', '—')} × {wt_info.get('width_mm', '—')} × {wt_info.get('height_mm', '—')} mm<br>
                        • <b>Share of WT:</b> 100.0%<br>
                        • <i>Includes edema and core tumor tissue</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Tumor Core Card
        tc_info = regions.get("TC", {})
        with sub_cols[1]:
            st.markdown(
                f"""
                <div class="clinical-card" style="border-top: 4px solid {REGION_COLORS['TC']['hex']};">
                    <div class="clinical-card-header" style="color:{REGION_COLORS['TC']['hex']};">Tumor Core (TC)</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#FFFFFF; margin:0.3rem 0;">
                        {tc_info.get('volume_cm3', '—')} <span style="font-size:0.85rem; color:#94A3B8;">cm³</span>
                    </div>
                    <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
                        • <b>Voxel count:</b> {tc_info.get('voxel_count', 0):,}<br>
                        • <b>Dimensions:</b> {tc_info.get('length_mm', '—')} × {tc_info.get('width_mm', '—')} × {tc_info.get('height_mm', '—')} mm<br>
                        • <b>Share of WT:</b> {tc_info.get('ratio_to_wt_pct', '—')}%<br>
                        • <i>Necrotic core + Enhancing tumor</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Enhancing Tumor Card
        et_info = regions.get("ET", {})
        with sub_cols[2]:
            st.markdown(
                f"""
                <div class="clinical-card" style="border-top: 4px solid {REGION_COLORS['ET']['hex']};">
                    <div class="clinical-card-header" style="color:{REGION_COLORS['ET']['hex']};">Enhancing Tumor (ET)</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#FFFFFF; margin:0.3rem 0;">
                        {et_info.get('volume_cm3', '—')} <span style="font-size:0.85rem; color:#94A3B8;">cm³</span>
                    </div>
                    <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
                        • <b>Voxel count:</b> {et_info.get('voxel_count', 0):,}<br>
                        • <b>Dimensions:</b> {et_info.get('length_mm', '—')} × {et_info.get('width_mm', '—')} × {et_info.get('height_mm', '—')} mm<br>
                        • <b>Share of WT:</b> {et_info.get('ratio_to_wt_pct', '—')}%<br>
                        • <i>Active contrast-enhancing neo-vasculature</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # Measurement Explanations Panel
    with st.expander("ℹ️ Clinical Measurement Interpretation Guide", expanded=True):
        st.markdown(
            """
            - **Tumor Volume (cm³):** Calculated by multiplying the total segmented voxel count by the individual voxel volume derived from the NIfTI spatial header spacing $(s_x \\times s_y \\times s_z)$. Essential for quantitative monitoring under RANO (Response Assessment in Neuro-Oncology) guidelines.
            - **Orthogonal Bounding Extents (mm):** Physical span along scanner coordinate axes $(X, Y, Z)$. Reflects maximum anatomical lesion diameter.
            - **Enhancing Tumor Ratio (%):** Proportion of the total lesion exhibiting active breakdown of the blood-brain barrier. High enhancing fractions are commonly associated with aggressive high-grade glioma activity.
            - **Peak Axial Slice:** The axial slice possessing the maximum cross-sectional tumor area, prioritized for radiologist review.
            """
        )

    render_disclaimer()


# ========================================================================
# 4. Confidence & Explainability Page
# ========================================================================

def page_confidence():
    render_header(
        "Confidence & Explainability",
        "Spatial model probability maps, heuristic confidence metrics, and transparent explainability",
    )

    conf = st.session_state.get("confidence_summary")
    results = st.session_state.get("analysis_results") or {}
    volume = st.session_state.get("mri_volume")
    conf_map = results.get("confidence_map")
    is_demo = results.get("is_demo", False)

    if not conf or not conf.get("available") or conf_map is None:
        render_empty_analysis_state()
        return

    if is_demo:
        st.markdown(demo_badge_html(), unsafe_allow_html=True)
        st.write("")

    # Confidence Metric Cards
    render_section_heading("Confidence Summary Across Tumor Region", "🎯")
    render_metric_row([
        {
            "label": "Mean Confidence Proxy",
            "value": conf.get("mean", "—"),
            "unit": "",
            "subtext": "Average sigmoid activation over tumor",
            "is_demo": is_demo,
            "accent_color": COLORS["accent_teal"],
        },
        {
            "label": "Minimum Boundary Agreement",
            "value": conf.get("min", "—"),
            "unit": "",
            "subtext": "Lowest probability at tumor periphery",
            "is_demo": is_demo,
        },
        {
            "label": "Peak Core Agreement",
            "value": conf.get("max", "—"),
            "unit": "",
            "subtext": "Maximum probability in central mass",
            "is_demo": is_demo,
        },
        {
            "label": "Spatial Dispersion (Std)",
            "value": conf.get("std", "—"),
            "unit": "",
            "subtext": "Standard deviation across lesion voxels",
            "is_demo": is_demo,
        },
    ])

    st.write("")

    # Spatial Heatmap Slice Viewer
    render_section_heading("Spatial Probability Heatmap", "🗺️")

    num_slices = volume.shape[2]
    slice_idx = st.session_state.get("current_slice") or (num_slices // 2)

    slice_idx = st.slider(
        "Axial slice",
        0,
        num_slices - 1,
        int(slice_idx),
        key="conf_heatmap_slider",
    )
    st.session_state["current_slice"] = slice_idx

    conf_slice = np.rot90(conf_map[:, :, slice_idx])

    import plotly.graph_objects as go
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=conf_slice,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title=dict(text="Probability", font=dict(color="#FFFFFF", size=11)),
                tickfont=dict(color="#F8FAFC"),
                outlinecolor="rgba(30,47,77,0.8)",
            ),
        )
    )
    fig_heat.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_heat.update_xaxes(visible=False)
    fig_heat.update_yaxes(visible=False)

    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Viridis color scale: 0.0 (background/low probability) to 1.0 (highest model confidence).")

    st.write("")

    # Honest Explainability Panel (Requirement 7)
    render_section_heading("Model Interpretability & Methodological Boundaries", "🔍")

    st.markdown(
        """
        <div class="clinical-card">
            <h4 style="margin-top:0; color:#FFFFFF;">Model Interpretation & Honest Disclosure</h4>
            <div style="font-size:0.88rem; color:#94A3B8; line-height:1.65;">
                <p>
                    <b>1. Input MRI Modalities:</b> The pipeline processes four distinct 3D modalities (T1c, T1, T2, FLAIR). 
                    Each modality provides complementary radiological contrast: T1 native captures baseline anatomy; T1c reveals 
                    active neo-vascular leakage; T2 highlights gross hyperintensity; and FLAIR isolates infiltrative peritumoral vasogenic edema.
                </p>
                <p>
                    <b>2. Segmentation Hierarchy:</b> According to the BraTS protocol, predictions represent nested sub-compartments:
                    Whole Tumor (WT: edema + core), Tumor Core (TC: necrotic core + enhancing tumor), and Enhancing Tumor (ET: viable active tumor margins).
                </p>
                <p>
                    <b>3. What Confidence Values Do & Do Not Represent:</b> The displayed confidence metric is computed directly from 
                    the model's sigmoid output activations (0.0 to 1.0) on spatial sliding-window patches. 
                    <b>It is a spatial probability heuristic, NOT a Bayesian calibrated posterior or a clinical measure of diagnostic truth.</b> 
                    A high probability indicates strong agreement with features learned from the BraTS dataset, not guaranteed histopathological accuracy.
                </p>
                <p>
                    <b>4. Important Limitations:</b>
                    <br>• The model assumes co-registered, skull-stripped volumes resampled to 1mm³ isotropic resolution.
                    <br>• Variations across 1.5T vs 3.0T scanner field strengths, head coils, or motion artifacts can introduce boundary uncertainty.
                    <br>• NeuroVision does NOT predict genetic mutations (e.g. IDH status, 1p/19q codeletion), assign WHO tumor grades, or forecast patient overall survival.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer()


# ========================================================================
# 5. 3D Visualization Page (Multi-Region Point Cloud)
# ========================================================================

def page_3d_visualization():
    render_header(
        "3D Visualization",
        "Interactive volumetric 3D exploration of segmented tumor sub-regions",
    )

    results = st.session_state.get("analysis_results") or {}
    mask = results.get("segmentation")
    region_masks = results.get("region_masks")
    is_demo = results.get("is_demo", False)

    if mask is None or not np.any(mask > 0):
        render_empty_analysis_state()
        return

    if is_demo:
        st.markdown(demo_badge_html(), unsafe_allow_html=True)
        st.write("")

    fig_3d = create_3d_visualization(
        segmentation=mask,
        voxel_spacing=st.session_state["voxel_spacing"],
        region_masks=region_masks,
    )

    if fig_3d is None:
        render_empty_analysis_state()
        return

    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown(
        """
        <div style="font-size:0.84rem; color:#94A3B8; text-align:center;">
            <b>Interactive Controls:</b> Click and drag to orbit / rotate 3D camera • Scroll wheel to zoom in/out • Right-click and drag to pan • Double click to reset orientation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer()


# ========================================================================
# 6. Report Page (Clinical PDF Generation & Download)
# ========================================================================

def page_report():
    render_header(
        "Clinical Summary Report",
        "Generate and download a standardized radiological summary report in PDF format",
    )

    file_name = st.session_state.get("file_name")
    results = st.session_state.get("analysis_results") or {}
    meas = st.session_state.get("measurements") or {"available": False}
    conf = st.session_state.get("confidence_summary") or {"available": False}
    is_demo = results.get("is_demo", False)

    if not file_name or not results.get("success"):
        render_empty_analysis_state()
        return

    # Report Preview Card
    render_section_heading("Report Metadata & Parameters", "📋")

    status_str = "Demo Simulation (Placeholder Data)" if is_demo else "Completed (Real Model Inference)"

    st.markdown(
        f"""
        <div class="clinical-card">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; font-size:0.88rem; color:#94A3B8;">
                <div><b>Scan Identifier:</b> <span style="color:#FFFFFF;">{file_name}</span></div>
                <div><b>Status:</b> <span style="color:#FFFFFF;">{status_str}</span></div>
                <div><b>Whole Tumor Volume:</b> <span style="color:#FFFFFF;">{meas.get('volume_cm3', '—')} cm³</span></div>
                <div><b>Peak Lesion Slice:</b> <span style="color:#FFFFFF;">#{meas.get('peak_slice', 0) + 1}</span></div>
                <div><b>Mean Confidence:</b> <span style="color:#FFFFFF;">{conf.get('mean', '—')}</span></div>
                <div><b>Generated At:</b> <span style="color:#FFFFFF;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Generate PDF Action
    col_gen, _ = st.columns([3, 5])
    with col_gen:
        if st.button("📄 Build PDF Summary Report", type="primary", use_container_width=True):
            with st.spinner("Compiling clinical PDF document with ReportLab..."):
                try:
                    # Render axial thumbnail for PDF snapshot
                    slice_bytes = None
                    volume = st.session_state.get("mri_volume")
                    composite = results.get("composite")
                    peak_idx = meas.get("peak_slice") or (volume.shape[2] // 2)

                    if volume is not None and composite is not None:
                        import io
                        from PIL import Image as PILImage
                        mri_slice = get_axial_slice(volume, peak_idx)
                        comp_slice = get_axial_slice(composite, peak_idx)
                        overlay_rgb = create_composite_overlay(mri_slice, comp_slice)
                        pil_img = PILImage.fromarray(overlay_rgb)
                        b_buf = io.BytesIO()
                        pil_img.save(b_buf, format="PNG")
                        slice_bytes = b_buf.getvalue()

                    pdf_bytes = generate_pdf_report(
                        file_name=file_name,
                        analysis_status=status_str,
                        measurements=meas,
                        confidence=conf,
                        is_demo=is_demo,
                        slice_image_bytes=slice_bytes,
                        model_name="MONAI SegResNet (BraTS Benchmark Pretrained)",
                    )
                    st.session_state["last_report_bytes"] = pdf_bytes
                    st.success("PDF report compiled successfully.")
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

    if st.session_state.get("last_report_bytes"):
        clean_name = file_name.replace(".nii.gz", "").replace(".nii", "")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"NeuroVision_Report_{clean_name}_{timestamp_str}.pdf"

        st.write("")
        st.download_button(
            label="⬇️ Download NeuroVision Clinical Report (PDF)",
            data=st.session_state["last_report_bytes"],
            file_name=download_name,
            mime="application/pdf",
            type="primary",
        )

    render_disclaimer()


# ========================================================================
# 7. Technical Specs & Model Architecture Page
# ========================================================================

def page_technical_specs():
    render_header(
        "Model Architecture & Technical Specifications",
        "Detailed neural network configuration, training provenance, and benchmark validation",
    )

    st.markdown(
        """
        <div class="clinical-card">
            <h3 style="margin-top:0; color:#FFFFFF;">MONAI SegResNet Architecture</h3>
            <div style="font-size:0.88rem; color:#94A3B8; line-height:1.65;">
                <p>
                    NeuroVision deploys a verified <b>SegResNet</b> (3D Encoder-Decoder with Residual Connections) 
                    implemented using the MONAI framework. SegResNet incorporates an asymmetric encoder-decoder structure 
                    with residual skip connections and group normalization to handle high-resolution volumetric medical data efficiently.
                </p>
                <div style="background:#070D18; border:1px solid #1E2F4D; border-radius:8px; padding:1.0rem; margin:1.0rem 0;">
                    <b>Technical Configuration:</b><br>
                    • <b>Spatial Dimensions:</b> 3D (X, Y, Z)<br>
                    • <b>Initial Filter Dimension:</b> 16<br>
                    • <b>Input Channels:</b> 4 (T1c, T1, T2, FLAIR)<br>
                    • <b>Output Channels:</b> 3 (Channel 0: Tumor Core, Channel 1: Whole Tumor, Channel 2: Enhancing Tumor)<br>
                    • <b>Downsampling Blocks:</b> (1, 2, 2, 4)<br>
                    • <b>Upsampling Blocks:</b> (1, 1, 1)<br>
                    • <b>Dropout Rate:</b> 0.2<br>
                    • <b>Inference Window Size:</b> (240, 240, 160) with 50% Gaussian overlap
                </div>
                <p>
                    <b>Data Preprocessing Protocol:</b>
                    <br>Each channel is independently standardized by computing the zero-mean, unit-variance transform across nonzero voxels:
                    $$x_{\\text{norm}} = \\frac{x - \\mu_{\\text{nonzero}}}{\\sigma_{\\text{nonzero}}}$$
                </p>
                <p>
                    <b>Hardware & Execution Profile:</b>
                    <br>Optimized for standard consumer hardware (8 GB RAM, Intel integrated graphics, CPU execution). 
                    No discrete NVIDIA GPU is mandatory. Typical 3D sliding-window inference duration on CPU is 40–50 seconds.
                </p>
                <p>
                    <b>Future Roadmap:</b>
                    <br>Integration of automated nnU-Net v2 dynamic topology planning, self-configuring deep supervision, and calibrated Bayesian uncertainty ensembles.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer()


# ========================================================================
# Application Router
# ========================================================================

PAGES = {
    "🏠 Dashboard": page_dashboard,
    "🧠 MRI Analysis": page_mri_analysis,
    "📊 Tumor Analysis": page_tumor_analysis,
    "🎯 Confidence": page_confidence,
    "🧊 3D Visualization": page_3d_visualization,
    "📄 Clinical Report": page_report,
    "ℹ️ Technical Specs": page_technical_specs,
}

page_func = PAGES.get(st.session_state["nav_page"], page_dashboard)
page_func()
