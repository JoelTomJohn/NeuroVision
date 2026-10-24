"""
frontend/styles.py

NeuroVision — AI-Assisted Brain MRI Analysis
Centralized clinical design system, custom CSS, and visual styling tokens.
"""

import streamlit as st

# ----------------------------------------------------------------------
# Color Palette: Deep clinical navy, crisp white, restrained teal accents
# ----------------------------------------------------------------------
COLORS = {
    "primary": "#0B192C",          # Deep navy
    "primary_surface": "#0F1E36",  # Elevated card navy
    "primary_light": "#1E6091",    # Clinical steel blue
    "accent_teal": "#00B4D8",      # Radiant medical teal
    "accent_cyan": "#00C9A7",      # Whole tumor / edema accent
    "accent_orange": "#F59E0B",    # Tumor core accent
    "accent_red": "#EF4444",       # Enhancing tumor accent
    "background": "#070D18",       # Deep canvas
    "surface": "#0E182A",          # Main card surface
    "surface_light": "#16233B",    # Hover/nested surface
    "border": "#1E2F4D",           # Card borders
    "border_subtle": "#162238",
    "text": "#F8FAFC",             # High contrast light text
    "text_muted": "#94A3B8",       # Muted slate text
    "text_dim": "#64748B",
    "success": "#10B981",          # Completed green
    "warning": "#F5B041",          # Demo badge / warning
    "error": "#E74C3C",            # Error red
}

# Sub-region radiological colors for masks and legends
REGION_COLORS = {
    "WT": {"name": "Whole Tumor", "hex": "#00C9A7", "rgb": (0, 201, 167), "desc": "Total abnormal tissue (Edema + Core)"},
    "TC": {"name": "Tumor Core", "hex": "#F59E0B", "rgb": (245, 158, 11), "desc": "Necrotic core + Enhancing tumor"},
    "ET": {"name": "Enhancing Tumor", "hex": "#EF4444", "rgb": (239, 68, 68), "desc": "Active contrast-enhancing margins"},
    "ED": {"name": "Peritumoral Edema", "hex": "#38BDF8", "rgb": (56, 189, 248), "desc": "Infiltrating edema on FLAIR/T2"},
    "NCR": {"name": "Non-Enhancing Core", "hex": "#FB923C", "rgb": (251, 146, 60), "desc": "Hypointense necrotic/cystic core"},
}


def inject_global_css() -> None:
    """Inject application-wide clinical CSS styles."""
    st.markdown(
        f"""
        <style>
        /* ---- Main Canvas & Typography ---- */
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3.5rem;
            max-width: 1240px;
        }}

        body, button, input, select, textarea {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}

        h1, h2, h3, h4 {{
            letter-spacing: -0.025em;
            color: {COLORS['text']};
        }}

        /* ---- App Header & Brand ---- */
        .neuro-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 1.1rem;
            margin-bottom: 1.6rem;
        }}
        .neuro-brand-title {{
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0;
            line-height: 1.2;
        }}
        .neuro-brand-title span.brand-accent {{
            color: {COLORS['accent_teal']};
        }}
        .neuro-brand-tagline {{
            font-size: 0.95rem;
            color: {COLORS['text_muted']};
            margin-top: 0.2rem;
            font-weight: 400;
        }}
        .prototype-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(0, 180, 216, 0.12);
            border: 1px solid rgba(0, 180, 216, 0.35);
            color: {COLORS['accent_teal']};
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        .prototype-pill::before {{
            content: "";
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: {COLORS['accent_teal']};
        }}

        /* ---- Hero / Welcome Banner ---- */
        .welcome-hero {{
            background: linear-gradient(135deg, rgba(15, 30, 54, 0.92) 0%, rgba(14, 24, 42, 0.85) 100%);
            border: 1px solid {COLORS['border']};
            border-left: 5px solid {COLORS['accent_teal']};
            border-radius: 12px;
            padding: 1.8rem 2.0rem;
            margin-bottom: 2.0rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }}
        .welcome-hero h2 {{
            font-size: 1.65rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 0 0 0.5rem 0;
        }}
        .welcome-hero p {{
            font-size: 0.98rem;
            color: {COLORS['text_muted']};
            line-height: 1.55;
            margin: 0 0 1.2rem 0;
            max-width: 900px;
        }}

        /* ---- Section Titles ---- */
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: {COLORS['text']};
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 1.8rem 0 1.0rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid {COLORS['border_subtle']};
        }}

        /* ---- Metric Cards ---- */
        .metric-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 1.1rem 1.2rem;
            transition: transform 0.15s ease, border-color 0.15s ease;
            position: relative;
            overflow: hidden;
        }}
        .metric-card:hover {{
            border-color: {COLORS['accent_teal']};
        }}
        .metric-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, {COLORS['accent_teal']}, {COLORS['primary_light']});
        }}
        .metric-label {{
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS['text_muted']};
            margin-bottom: 0.35rem;
        }}
        .metric-value {{
            font-size: 1.85rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.1;
        }}
        .metric-unit {{
            font-size: 0.90rem;
            font-weight: 500;
            color: {COLORS['text_muted']};
            margin-left: 0.3rem;
        }}
        .metric-subtext {{
            font-size: 0.78rem;
            color: {COLORS['text_dim']};
            margin-top: 0.4rem;
        }}

        /* ---- Clinical Cards & Features ---- */
        .clinical-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 1.2rem 1.3rem;
            height: 100%;
        }}
        .clinical-card-header {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.45rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        .clinical-card-body {{
            font-size: 0.88rem;
            color: {COLORS['text_muted']};
            line-height: 1.5;
        }}

        /* ---- Modality Badges ---- */
        .modality-badge {{
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            background: {COLORS['surface_light']};
            border: 1px solid {COLORS['border']};
            color: {COLORS['accent_teal']};
        }}

        /* ---- Empty State Container ---- */
        .empty-state {{
            background: {COLORS['surface']};
            border: 1px dashed {COLORS['border']};
            border-radius: 12px;
            padding: 2.5rem 1.8rem;
            text-align: center;
            color: {COLORS['text_muted']};
            margin: 1.0rem 0;
        }}
        .empty-state-icon {{
            font-size: 2.6rem;
            margin-bottom: 0.8rem;
            opacity: 0.85;
        }}
        .empty-state-title {{
            font-size: 1.2rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.4rem;
        }}
        .empty-state-text {{
            font-size: 0.90rem;
            color: {COLORS['text_muted']};
            max-width: 520px;
            margin: 0 auto 1.2rem auto;
            line-height: 1.5;
        }}

        /* ---- Legend Container ---- */
        .legend-box {{
            display: flex;
            flex-wrap: wrap;
            gap: 1.0rem;
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 0.75rem 1.0rem;
            margin: 0.8rem 0;
            align-items: center;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: {COLORS['text']};
        }}
        .legend-swatch {{
            width: 13px;
            height: 13px;
            border-radius: 3px;
        }}

        /* ---- Responsible Use Disclaimer Note ---- */
        .disclaimer-note {{
            background: rgba(14, 24, 42, 0.75);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-left: 4px solid {COLORS['warning']};
            border-radius: 8px;
            padding: 0.85rem 1.1rem;
            color: {COLORS['text_muted']};
            font-size: 0.82rem;
            line-height: 1.5;
            margin-top: 1.8rem;
        }}
        .disclaimer-note strong {{
            color: #FFFFFF;
        }}

        /* ---- Badges ---- */
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .badge-demo {{
            background: rgba(245, 176, 65, 0.15);
            color: {COLORS['warning']};
            border: 1px solid rgba(245, 176, 65, 0.4);
        }}
        .badge-real {{
            background: rgba(16, 185, 129, 0.15);
            color: {COLORS['success']};
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}

        /* ---- Sidebar Styling ---- */
        section[data-testid="stSidebar"] {{
            background-color: #070D18;
            border-right: 1px solid {COLORS['border']};
        }}

        /* ---- Buttons ---- */
        div.stButton > button:first-child {{
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.15s ease;
        }}
        div.stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['accent_teal']} 0%, {COLORS['primary_light']} 100%);
            border: none;
            color: #FFFFFF;
            box-shadow: 0 2px 10px rgba(0, 180, 216, 0.3);
        }}
        div.stButton > button[kind="primary"]:hover {{
            box-shadow: 0 4px 16px rgba(0, 180, 216, 0.45);
            transform: translateY(-1px);
        }}

        /* ---- Panel labels for side-by-side viewer ---- */
        .panel-label {{
            text-align: center;
            font-weight: 700;
            color: #FFFFFF;
            background-color: {COLORS['surface_light']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 0.45rem 0;
            margin-bottom: 0.4rem;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
