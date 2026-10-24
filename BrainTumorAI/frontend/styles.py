"""
frontend/styles.py

Central place for all visual styling (CSS) and small style-related
constants used across the app. Keeping this separate means you can
re-theme the whole dashboard without touching page logic.
"""

import streamlit as st

# ----------------------------------------------------------------------
# Color palette (clean clinical / AI look: deep blue + teal accent)
# ----------------------------------------------------------------------
COLORS = {
    "primary": "#1B4F72",       # deep medical blue
    "primary_light": "#2E86C1",
    "accent": "#17A589",        # teal accent
    "background": "#0E1117",
    "surface": "#161B22",
    "surface_light": "#1E2530",
    "text": "#E6EDF3",
    "text_muted": "#9AA7B4",
    "success": "#2ECC71",
    "warning": "#F5B041",
    "error": "#E74C3C",
    "border": "#2A313C",
    "demo_badge": "#F5B041",
}


def inject_global_css() -> None:
    """Inject the app-wide CSS. Call once, near the top of app.py."""
    st.markdown(
        f"""
        <style>
        /* ---- General typography & spacing ---- */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            font-weight: 600;
            letter-spacing: -0.02em;
        }}

        /* ---- App header ---- */
        .app-title {{
            font-size: 2.1rem;
            font-weight: 700;
            color: {COLORS['text']};
            margin-bottom: 0;
        }}
        .app-subtitle {{
            font-size: 1.0rem;
            color: {COLORS['text_muted']};
            margin-top: 0.15rem;
            margin-bottom: 1.4rem;
        }}

        /* ---- Section headings ---- */
        .section-heading {{
            font-size: 1.35rem;
            font-weight: 600;
            color: {COLORS['text']};
            border-left: 4px solid {COLORS['accent']};
            padding-left: 0.6rem;
            margin: 1.2rem 0 0.9rem 0;
        }}

        /* ---- Generic card ---- */
        .metric-card {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 1.0rem 1.1rem;
            text-align: left;
        }}
        .metric-card .metric-label {{
            font-size: 0.80rem;
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .metric-card .metric-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: {COLORS['text']};
            margin-top: 0.15rem;
        }}
        .metric-card .metric-unit {{
            font-size: 0.85rem;
            color: {COLORS['text_muted']};
            margin-left: 0.25rem;
        }}

        /* ---- Info / workflow card ---- */
        .info-card {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-left: 4px solid {COLORS['primary_light']};
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            color: {COLORS['text']};
            font-size: 0.92rem;
        }}

        /* ---- Badges ---- */
        .badge {{
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        .badge-demo {{
            background-color: rgba(245, 176, 65, 0.15);
            color: {COLORS['demo_badge']};
            border: 1px solid rgba(245, 176, 65, 0.4);
        }}
        .badge-success {{
            background-color: rgba(46, 204, 113, 0.15);
            color: {COLORS['success']};
            border: 1px solid rgba(46, 204, 113, 0.4);
        }}
        .badge-pending {{
            background-color: rgba(154, 167, 180, 0.15);
            color: {COLORS['text_muted']};
            border: 1px solid rgba(154, 167, 180, 0.4);
        }}

        /* ---- Workflow pipeline ---- */
        .pipeline-step {{
            background-color: {COLORS['surface_light']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            text-align: center;
            font-size: 0.85rem;
            color: {COLORS['text']};
        }}
        .pipeline-arrow {{
            text-align: center;
            color: {COLORS['text_muted']};
            font-size: 1.1rem;
        }}

        /* ---- Panel labels for side-by-side viewer ---- */
        .panel-label {{
            text-align: center;
            font-weight: 600;
            color: {COLORS['text']};
            background-color: {COLORS['surface_light']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 0.35rem 0;
            margin-bottom: 0.4rem;
            font-size: 0.9rem;
        }}

        /* ---- Sidebar tweaks ---- */
        section[data-testid="stSidebar"] {{
            border-right: 1px solid {COLORS['border']};
        }}

        /* ---- Disclaimer box ---- */
        .disclaimer-box {{
            background-color: rgba(231, 76, 60, 0.08);
            border: 1px solid rgba(231, 76, 60, 0.35);
            border-radius: 8px;
            padding: 0.8rem 1.0rem;
            color: {COLORS['text']};
            font-size: 0.85rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
