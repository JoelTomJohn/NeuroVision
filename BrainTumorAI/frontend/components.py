"""
frontend/components.py

Small, reusable Streamlit UI building blocks used across every page:
metric cards, badges, the workflow pipeline diagram, and status
indicators. Keeping these in one place keeps page files short and
consistent-looking.
"""

import streamlit as st


def demo_badge_html() -> str:
    """HTML snippet for a 'Demo data' badge. Use wherever placeholder
    values are shown so the user never mistakes them for real output."""
    return '<span class="badge badge-demo">Demo data — awaiting model inference</span>'


def status_badge_html(status: str) -> str:
    """Return a badge for 'success' / 'pending' / 'demo' style statuses."""
    mapping = {
        "success": '<span class="badge badge-success">Completed</span>',
        "pending": '<span class="badge badge-pending">Not run yet</span>',
        "demo": demo_badge_html(),
    }
    return mapping.get(status, mapping["pending"])


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_section_heading(text: str) -> None:
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)


def render_info_card(text: str) -> None:
    st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)


def render_disclaimer(text: str) -> None:
    st.markdown(f'<div class="disclaimer-box">{text}</div>', unsafe_allow_html=True)


def render_metric_card(label: str, value, unit: str = "", is_demo: bool = False) -> None:
    """A single metric card, e.g. Tumor Volume: 14.2 cm3."""
    demo_html = f'<div style="margin-top:0.4rem;">{demo_badge_html()}</div>' if is_demo else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
            {demo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: list) -> None:
    """metrics: list of dicts with keys label, value, unit, is_demo."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_metric_card(
                label=m.get("label", ""),
                value=m.get("value", "—"),
                unit=m.get("unit", ""),
                is_demo=m.get("is_demo", False),
            )


def render_pipeline_diagram() -> None:
    """Vertical workflow diagram shown on the Dashboard page."""
    steps = [
        "MRI Upload",
        "Preprocessing",
        "nnU-Net v2 Segmentation",
        "Tumor Analysis",
        "3D Visualization",
        "AI-Assisted Report",
    ]
    for i, step in enumerate(steps):
        st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)
        if i < len(steps) - 1:
            st.markdown('<div class="pipeline-arrow">&#8595;</div>', unsafe_allow_html=True)


def render_panel_label(text: str) -> None:
    st.markdown(f'<div class="panel-label">{text}</div>', unsafe_allow_html=True)


def render_upload_summary(file_name: str, file_size_kb: float, file_format: str) -> None:
    render_info_card(
        f"<b>File name:</b> {file_name}<br>"
        f"<b>File size:</b> {file_size_kb:.1f} KB<br>"
        f"<b>Format:</b> {file_format}"
    )
