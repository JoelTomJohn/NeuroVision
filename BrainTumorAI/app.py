"""
app.py

BrainTumorAI — AI-Assisted Brain Tumor Analysis System

Run with:
    streamlit run app.py

This file handles navigation + page composition.
Styling lives in frontend/styles.py, reusable widgets in
frontend/components.py, MRI rendering in frontend/viewer.py,
and model logic lives under backend/.
"""

from datetime import datetime

import numpy as np
import streamlit as st

from frontend.styles import inject_global_css
from frontend.components import (
    render_header,
    render_section_heading,
    render_info_card,
    render_disclaimer,
    render_metric_row,
    render_pipeline_diagram,
    render_upload_summary,
    status_badge_html,
    demo_badge_html,
)
from frontend.viewer import render_viewer, create_3d_visualization
from backend.preprocessing import (
    load_nifti,
    load_multimodal_nifti,
    NiftiLoadError,
    describe_modalities_note,
)
from backend.inference import run_nnunet_inference
from backend.analysis import calculate_tumor_measurements, calculate_confidence
from backend.report import generate_pdf_report


# ========================================================================
# Page config + global styles
# ========================================================================

st.set_page_config(
    page_title="BrainTumorAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


# ========================================================================
# Session state defaults
# ========================================================================

DEFAULTS = {
    "mri_volume": None,
    "mri_modalities": None,
    "mri_affine": None,
    "voxel_spacing": (1.0, 1.0, 1.0),
    "file_name": None,
    "file_size_kb": None,
    "analysis_results": None,
    "measurements": None,
    "confidence_summary": None,
    "demo_mode": True,
    "current_slice": None,
    "last_report_bytes": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ========================================================================
# Sidebar navigation
# ========================================================================

with st.sidebar:
    st.markdown("### 🧠 NeuroVision")
    st.caption("AI-Assisted Brain Tumor Analysis")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "🧠 MRI Analysis",
            "📊 Tumor Analysis",
            "🎯 Confidence",
            "🧊 3D Visualization",
            "📄 Report",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.session_state["demo_mode"] = st.toggle(
        "Demo Mode",
        value=st.session_state["demo_mode"],
    )

    if st.session_state["demo_mode"]:
        st.caption(
            "Demo Mode: ON — placeholder data is used where "
            "the real model output is unavailable."
        )
    else:
        st.caption(
            "Demo Mode: OFF — real model inference will be used "
            "when the model is connected."
        )

    st.divider()
    st.caption(
        "Research / educational prototype. Not for clinical use."
    )


# ========================================================================
# Dashboard page
# ========================================================================

def page_dashboard():
    render_header(
        "BrainTumorAI",
        "AI-Assisted Brain Tumor Analysis",
    )

    st.write(
        "Upload a compatible brain MRI scan to analyze tumor regions "
        "using an AI-assisted segmentation workflow."
    )

    render_info_card(
        "The system uses four co-registered MRI modalities "
        "(T1c, T1, T2 and FLAIR) for brain tumor segmentation."
    )

    st.write("")

    render_section_heading("Workflow")

    col_a, col_b, col_c = st.columns([1, 2, 1])

    with col_b:
        render_pipeline_diagram()

    st.write("")

    render_section_heading("System Status")

    status_cols = st.columns(3)

    with status_cols[0]:
        st.markdown(
            f"**MRI loaded:** "
            f"{'Yes' if st.session_state['mri_volume'] is not None else 'No'}"
        )

    with status_cols[1]:
        results = st.session_state["analysis_results"]
        ran = results is not None and results.get("success")
        st.markdown(
            f"**Analysis run:** {'Yes' if ran else 'No'}"
        )

    with status_cols[2]:
        st.markdown(
            f"**Demo Mode:** "
            f"{'ON' if st.session_state['demo_mode'] else 'OFF'}"
        )


# ========================================================================
# MRI Analysis page
# ========================================================================

def page_mri_analysis():

    render_header(
        "MRI Analysis",
        "Upload, preview, and analyze a brain MRI scan",
    )

    # --------------------------------------------------------------------
    # Upload section
    # --------------------------------------------------------------------

    render_section_heading("Upload Brain MRI")

    st.info(
        "The AI model requires four co-registered MRI modalities: "
        "T1c, T1, T2 and FLAIR."
    )

    col1, col2 = st.columns(2)

    with col1:

        t1c_file = st.file_uploader(
            "T1c (T1 contrast-enhanced)",
            type=["nii", "gz"],
            key="t1c_upload",
        )

        t1_file = st.file_uploader(
            "T1",
            type=["nii", "gz"],
            key="t1_upload",
        )

    with col2:

        t2_file = st.file_uploader(
            "T2",
            type=["nii", "gz"],
            key="t2_upload",
        )

        flair_file = st.file_uploader(
            "FLAIR",
            type=["nii", "gz"],
            key="flair_upload",
        )

    all_modalities_uploaded = all(
        [
            t1c_file,
            t1_file,
            t2_file,
            flair_file,
        ]
    )

    # --------------------------------------------------------------------
    # When all four files are uploaded
    # --------------------------------------------------------------------

    if all_modalities_uploaded:

        st.success(
            "All four MRI modalities are ready."
        )

        with st.expander("Uploaded files"):

            st.write(
                f"**T1c:** {t1c_file.name}"
            )

            st.write(
                f"**T1:** {t1_file.name}"
            )

            st.write(
                f"**T2:** {t2_file.name}"
            )

            st.write(
                f"**FLAIR:** {flair_file.name}"
            )

        analyze_clicked = st.button(
            "Analyze MRI",
            type="primary",
            use_container_width=True,
        )

        # ---------------------------------------------------------------
        # Analyze button
        # ---------------------------------------------------------------

        if analyze_clicked:

            # -----------------------------------------------------------
            # Load MRI files
            # -----------------------------------------------------------

            with st.spinner(
                "Loading and preprocessing MRI modalities..."
            ):

                try:

                    modality_files = {
                        "t1c": (
                            t1c_file.getvalue(),
                            t1c_file.name,
                        ),
                        "t1": (
                            t1_file.getvalue(),
                            t1_file.name,
                        ),
                        "t2": (
                            t2_file.getvalue(),
                            t2_file.name,
                        ),
                        "flair": (
                            flair_file.getvalue(),
                            flair_file.name,
                        ),
                    }

                    (
                        modalities,
                        affine,
                        voxel_spacing,
                    ) = load_multimodal_nifti(
                        modality_files
                    )

                    # Store the complete 4-channel MRI volume.
                    st.session_state[
                        "mri_modalities"
                    ] = modalities

                    st.session_state[
                        "mri_affine"
                    ] = affine

                    # Use FLAIR for the main viewer.
                    st.session_state[
                        "mri_volume"
                    ] = modalities[3]

                    st.session_state[
                        "voxel_spacing"
                    ] = voxel_spacing

                    st.session_state[
                        "file_name"
                    ] = flair_file.name

                    total_size_kb = (
                        len(t1c_file.getvalue())
                        + len(t1_file.getvalue())
                        + len(t2_file.getvalue())
                        + len(flair_file.getvalue())
                    ) / 1024.0

                    st.session_state[
                        "file_size_kb"
                    ] = total_size_kb

                    st.session_state[
                        "current_slice"
                    ] = modalities.shape[3] // 2

                    st.success(
                        "All MRI modalities loaded successfully."
                    )

                    st.write(
                        f"**Input shape:** "
                        f"`{modalities.shape}`"
                    )

                    st.write(
                        f"**Voxel spacing:** "
                        f"`{voxel_spacing}`"
                    )

                except NiftiLoadError as e:

                    st.error(str(e))
                    st.stop()

                except Exception as e:

                    st.error(
                        f"Unexpected error while loading "
                        f"MRI files: {e}"
                    )

                    st.stop()

            # -----------------------------------------------------------
            # AI inference
            # -----------------------------------------------------------

            with st.spinner(
                "Running AI-assisted segmentation workflow..."
            ):

                result = run_nnunet_inference(
                    st.session_state[
                        "mri_modalities"
                    ],
                    demo_mode=st.session_state[
                        "demo_mode"
                    ],
                )

                st.session_state[
                    "analysis_results"
                ] = result

                # -------------------------------------------------------
                # Tumor measurements
                # -------------------------------------------------------

                measurements = (
                    calculate_tumor_measurements(
                        result.get("segmentation"),
                        st.session_state[
                            "voxel_spacing"
                        ],
                    )
                )

                st.session_state[
                    "measurements"
                ] = measurements

                # -------------------------------------------------------
                # Confidence
                # -------------------------------------------------------

                confidence = calculate_confidence(
                    result.get("confidence_map"),
                    result.get("segmentation"),
                )

                st.session_state[
                    "confidence_summary"
                ] = confidence

            # -----------------------------------------------------------
            # Result message
            # -----------------------------------------------------------

            if result.get("success"):

                if result.get("is_demo"):

                    st.warning(
                        "Demo segmentation generated.",
                        icon="⚠️",
                    )

                    st.markdown(
                        demo_badge_html(),
                        unsafe_allow_html=True,
                    )

                else:

                    st.success(
                        "AI segmentation result generated successfully."
                    )

            else:

                st.info(
                    result.get(
                        "message",
                        "Analysis unavailable.",
                    )
                )

    else:

        st.warning(
            "Please upload all four MRI modalities "
            "(T1c, T1, T2 and FLAIR) before starting analysis."
        )

    # ====================================================================
    # MRI Viewer
    # ====================================================================

    st.write("")

    render_section_heading(
        "MRI Viewer"
    )

    volume = st.session_state[
        "mri_volume"
    ]

    if volume is None:

        st.info(
            "Upload all four MRI modalities and "
            "analyze them to view the MRI here."
        )

        return

    results = (
        st.session_state[
            "analysis_results"
        ]
        or {}
    )

    mask = results.get(
        "segmentation"
    )

    display_mode = st.radio(
        "Display mode",
        [
            "Original",
            "Segmentation",
            "Overlay",
            "Side-by-side",
        ],
        horizontal=True,
    )

    if results.get("is_demo"):

        st.markdown(
            demo_badge_html(),
            unsafe_allow_html=True,
        )

    render_viewer(
        volume,
        mask,
        display_mode,
    )

    with st.expander(
        "About multi-modal MRI support"
    ):

        st.write(
            describe_modalities_note()
        )


# ========================================================================
# Tumor Analysis page
# ========================================================================

def page_tumor_analysis():

    render_header(
        "Tumor Analysis",
        "Quantitative measurements from the segmentation mask",
    )

    measurements = (
        st.session_state[
            "measurements"
        ]
    )

    results = (
        st.session_state[
            "analysis_results"
        ]
        or {}
    )

    is_demo = results.get(
        "is_demo",
        False,
    )

    if (
        not measurements
        or not measurements.get("available")
    ):

        render_info_card(
            "No tumor measurements available yet. "
            "Upload an MRI and run analysis from "
            "the MRI Analysis page."
        )

        return

    if is_demo:

        st.markdown(
            demo_badge_html(),
            unsafe_allow_html=True,
        )

    render_section_heading(
        "Tumor Dimensions"
    )

    render_metric_row(
        [
            {
                "label": "Tumor Volume",
                "value": measurements[
                    "volume_cm3"
                ],
                "unit": "cm³",
                "is_demo": is_demo,
            },
            {
                "label": "Tumor Length",
                "value": measurements[
                    "length_mm"
                ],
                "unit": "mm",
                "is_demo": is_demo,
            },
            {
                "label": "Tumor Width",
                "value": measurements[
                    "width_mm"
                ],
                "unit": "mm",
                "is_demo": is_demo,
            },
            {
                "label": "Tumor Height",
                "value": measurements[
                    "height_mm"
                ],
                "unit": "mm",
                "is_demo": is_demo,
            },
        ]
    )

    st.write("")

    render_section_heading(
        "Voxel Statistics"
    )

    render_metric_row(
        [
            {
                "label": "Tumor Voxel Count",
                "value": f"{measurements['voxel_count']:,}",
                "unit": "",
                "is_demo": is_demo,
            },
        ]
    )

    render_disclaimer(
        "These measurements are derived from an AI segmentation "
        "result and require professional clinical interpretation. "
        "This is a research prototype, not a diagnosis."
    )


# ========================================================================
# Confidence page
# ========================================================================

def page_confidence():

    render_header(
        "Confidence / Uncertainty",
        "Model confidence over the predicted tumor region",
    )

    confidence = (
        st.session_state[
            "confidence_summary"
        ]
    )

    results = (
        st.session_state[
            "analysis_results"
        ]
        or {}
    )

    is_demo = results.get(
        "is_demo",
        False,
    )

    render_info_card(
        "Confidence is derived from the model's prediction "
        "probabilities. The displayed value is a confidence "
        "proxy and not clinically calibrated uncertainty."
    )

    if (
        not confidence
        or not confidence.get("available")
    ):

        st.info(
            "No confidence data available yet. "
            "Upload an MRI and run analysis first."
        )

        return

    if is_demo:

        st.markdown(
            demo_badge_html(),
            unsafe_allow_html=True,
        )

    st.write("")

    render_section_heading(
        "Confidence Summary"
    )

    render_metric_row(
        [
            {
                "label": "Average Confidence",
                "value": confidence["mean"],
                "unit": "",
                "is_demo": is_demo,
            },
            {
                "label": "Minimum Confidence",
                "value": confidence["min"],
                "unit": "",
                "is_demo": is_demo,
            },
            {
                "label": "Maximum Confidence",
                "value": confidence["max"],
                "unit": "",
                "is_demo": is_demo,
            },
        ]
    )

    st.write("")

    render_section_heading(
        "Confidence Map"
    )

    volume = st.session_state[
        "mri_volume"
    ]

    conf_map = results.get(
        "confidence_map"
    )

    if (
        volume is not None
        and conf_map is not None
    ):

        import plotly.graph_objects as go

        slice_idx = (
            st.session_state.get(
                "current_slice"
            )
            or volume.shape[2] // 2
        )

        slice_idx = st.slider(
            "Axial slice",
            0,
            volume.shape[2] - 1,
            int(slice_idx),
            key="confidence_slice",
        )

        conf_slice = np.rot90(
            conf_map[
                :,
                :,
                slice_idx,
            ]
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=conf_slice,
                colorscale="Turbo",
                zmin=0,
                zmax=1,
            )
        )

        fig.update_layout(
            height=420,
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_xaxes(
            visible=False
        )

        fig.update_yaxes(
            visible=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ========================================================================
# 3D Visualization page
# ========================================================================

def page_3d_visualization():

    render_header(
        "3D Visualization",
        "Volumetric view of the segmented tumor region",
    )

    results = (
        st.session_state[
            "analysis_results"
        ]
        or {}
    )

    mask = results.get(
        "segmentation"
    )

    is_demo = results.get(
        "is_demo",
        False,
    )

    if mask is None:

        render_info_card(
            "3D tumor visualization will appear "
            "after model segmentation."
        )

        return

    if is_demo:

        st.markdown(
            demo_badge_html(),
            unsafe_allow_html=True,
        )

    fig = create_3d_visualization(
        mask,
        st.session_state[
            "voxel_spacing"
        ],
    )

    if fig is None:

        render_info_card(
            "3D tumor visualization will appear "
            "after model segmentation."
        )

        return

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Drag to rotate, scroll to zoom, "
        "double-click to reset the camera."
    )


# ========================================================================
# Report page
# ========================================================================

def page_report():

    render_header(
        "Report",
        "Generate a summary PDF report of the analysis",
    )

    file_name = st.session_state[
        "file_name"
    ]

    results = (
        st.session_state[
            "analysis_results"
        ]
        or {}
    )

    measurements = (
        st.session_state[
            "measurements"
        ]
        or {
            "available": False
        }
    )

    confidence = (
        st.session_state[
            "confidence_summary"
        ]
        or {
            "available": False
        }
    )

    is_demo = results.get(
        "is_demo",
        False,
    )

    if not file_name:

        render_info_card(
            "Upload and analyze an MRI first "
            "to generate a report."
        )

        return

    render_section_heading(
        "Summary"
    )

    status_label = (
        "Completed (demo)"
        if is_demo
        else (
            "Completed"
            if results.get("success")
            else "Not run"
        )
    )

    st.markdown(
        f"""
        <div class="info-card">
        <b>MRI file name:</b> {file_name}<br>
        <b>Analysis status:</b> {status_label}<br>
        <b>Tumor volume:</b>
            {measurements.get('volume_cm3', '—')} cm³<br>
        <b>Tumor dimensions (L×W×H):</b>
            {measurements.get('length_mm', '—')}
            ×
            {measurements.get('width_mm', '—')}
            ×
            {measurements.get('height_mm', '—')}
            mm<br>
        <b>Confidence (avg):</b>
            {confidence.get('mean', '—')}<br>
        <b>Date/time:</b>
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_demo:

        st.markdown(
            demo_badge_html(),
            unsafe_allow_html=True,
        )

    st.write("")

    if st.button(
        "Generate PDF Report",
        type="primary",
    ):

        try:

            pdf_bytes = generate_pdf_report(
                file_name=file_name,
                analysis_status=status_label,
                measurements=measurements,
                confidence=confidence,
                is_demo=is_demo,
            )

            st.session_state[
                "last_report_bytes"
            ] = pdf_bytes

            st.success(
                "Report generated."
            )

        except Exception:

            st.error(
                "The PDF report could not be generated. "
                "Please try again."
            )

    if st.session_state.get(
        "last_report_bytes"
    ):

        st.download_button(
            "Download PDF Report",
            data=st.session_state[
                "last_report_bytes"
            ],
            file_name=(
                "BrainTumorAI_Report_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            ),
            mime="application/pdf",
        )

    render_disclaimer(
        "BrainTumorAI is an AI-assisted research/educational "
        "prototype and is NOT a replacement for professional "
        "medical diagnosis."
    )


# ========================================================================
# Router
# ========================================================================

PAGES = {
    "🏠 Dashboard": page_dashboard,
    "🧠 MRI Analysis": page_mri_analysis,
    "📊 Tumor Analysis": page_tumor_analysis,
    "🎯 Confidence": page_confidence,
    "🧊 3D Visualization": page_3d_visualization,
    "📄 Report": page_report,
}

PAGES[page]()
