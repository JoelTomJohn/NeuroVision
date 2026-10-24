"""
backend/report.py

Builds the PDF summary report using ReportLab. Pulls together
everything already computed elsewhere (file info, measurements,
confidence summary) — it does not run any analysis itself.
"""

import io
from datetime import datetime
from typing import Optional


def generate_pdf_report(
    file_name: str,
    analysis_status: str,
    measurements: dict,
    confidence: dict,
    is_demo: bool,
) -> bytes:
    """
    Build a PDF report and return it as raw bytes (ready for
    st.download_button).

    Parameters mirror what is already stored in st.session_state on the
    Report page — this function has no Streamlit dependency so it can
    be tested or reused independently.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#1B4F72")
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#555555")
    )
    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1B4F72"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = styles["Normal"]
    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#7A2E27"),
        backColor=colors.HexColor("#FBEAE8"),
        borderPadding=8,
    )

    story = []

    # ---- Title ----
    story.append(Paragraph("BrainTumorAI", title_style))
    story.append(Paragraph("AI-Assisted Brain Tumor Analysis Report", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#2A313C")))

    demo_note = (
        " (DEMO DATA — placeholder values, not a real model prediction)"
        if is_demo
        else ""
    )

    # ---- 1. Scan information ----
    story.append(Paragraph("1. Scan Information", heading_style))
    scan_table = Table(
        [
            ["MRI file name", file_name or "—"],
            ["Analysis status", analysis_status + demo_note],
            ["Date / time of analysis", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ],
        colWidths=[55 * mm, 115 * mm],
    )
    scan_table.setStyle(_table_style())
    story.append(scan_table)

    # ---- 2. Segmentation summary ----
    story.append(Paragraph("2. Segmentation Summary", heading_style))
    seg_text = (
        "A segmentation mask was generated." if measurements.get("available") else
        "No segmentation mask is available yet."
    )
    if is_demo and measurements.get("available"):
        seg_text += " This mask is synthetic demo data, not derived from real tumor tissue."
    story.append(Paragraph(seg_text, body_style))

    # ---- 3. Tumor measurements ----
    story.append(Paragraph("3. Tumor Measurements", heading_style))
    if measurements.get("available"):
        m_table = Table(
            [
                ["Metric", "Value"],
                ["Tumor volume", f"{measurements['volume_cm3']} cm3"],
                ["Length", f"{measurements['length_mm']} mm"],
                ["Width", f"{measurements['width_mm']} mm"],
                ["Height", f"{measurements['height_mm']} mm"],
                ["Voxel count", f"{measurements['voxel_count']}"],
            ],
            colWidths=[55 * mm, 115 * mm],
        )
        m_table.setStyle(_table_style(header=True))
        story.append(m_table)
    else:
        story.append(Paragraph("No measurements available yet.", body_style))

    # ---- 4. Confidence / uncertainty ----
    story.append(Paragraph("4. Confidence / Uncertainty", heading_style))
    if confidence.get("available"):
        c_table = Table(
            [
                ["Metric", "Value"],
                ["Average confidence", f"{confidence['mean']}"],
                ["Minimum confidence", f"{confidence['min']}"],
                ["Maximum confidence", f"{confidence['max']}"],
            ],
            colWidths=[55 * mm, 115 * mm],
        )
        c_table.setStyle(_table_style(header=True))
        story.append(c_table)
    else:
        story.append(Paragraph("No confidence data available yet.", body_style))

    # ---- 5. Visualization summary ----
    story.append(Paragraph("5. Visualization Summary", heading_style))
    story.append(
        Paragraph(
            "Original MRI, segmentation overlay, and (when available) 3D tumor "
            "visualization were reviewed in the application's MRI Analysis and "
            "3D Visualization pages.",
            body_style,
        )
    )

    # ---- 6. Disclaimer ----
    story.append(Paragraph("6. Disclaimer", heading_style))
    story.append(
        Paragraph(
            "BrainTumorAI is an AI-assisted research and educational prototype "
            "developed for academic demonstration purposes. It is NOT a medical "
            "device and is NOT a replacement for professional medical diagnosis. "
            "All results, including any segmentation, measurements, and confidence "
            "values in this report, require professional clinical interpretation "
            "and must not be used for real patient care decisions.",
            disclaimer_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _table_style(header: bool = False):
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle

    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B4F72")))
        style.append(("TEXTCOLOR", (0, 0), (-1, 0), colors.white))
        style.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
    return TableStyle(style)
