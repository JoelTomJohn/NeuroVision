"""
backend/report.py

NeuroVision — AI-Assisted Brain MRI Analysis
Automated clinical-style PDF summary report generator built with ReportLab.
"""

import io
from datetime import datetime
from typing import Optional, Dict, Any


def generate_pdf_report(
    file_name: str,
    analysis_status: str,
    measurements: dict,
    confidence: dict,
    is_demo: bool,
    slice_image_bytes: Optional[bytes] = None,
    model_name: str = "MONAI SegResNet (BraTS Benchmark Pretrained)",
) -> bytes:
    """
    Build a professional PDF summary report and return raw bytes ready
    for Streamlit download.
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
        Image,
        KeepTogether,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
    )

    styles = getSampleStyleSheet()

    brand_primary = colors.HexColor("#0B192C")
    brand_secondary = colors.HexColor("#1E6091")
    brand_accent = colors.HexColor("#00A896")
    text_dark = colors.HexColor("#1A202C")
    text_muted = colors.HexColor("#4A5568")

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        textColor=brand_primary,
        fontName="Helvetica-Bold",
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=brand_secondary,
        fontName="Helvetica",
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=brand_primary,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=text_dark,
        fontName="Helvetica",
    )
    disclaimer_style = ParagraphStyle(
        "ReportDisclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#78350F"),
        backColor=colors.HexColor("#FEF3C7"),
        borderColor=colors.HexColor("#F59E0B"),
        borderWidth=0.5,
        borderPadding=7,
        spaceBefore=8,
    )
    badge_demo_style = ParagraphStyle(
        "DemoBadge",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#B45309"),
        backColor=colors.HexColor("#FEF3C7"),
        borderPadding=3,
    )

    story = []

    # ---- Header & Branding ----
    story.append(Paragraph("NeuroVision", title_style))
    story.append(Paragraph("AI-Assisted Brain MRI Analysis & Volumetric Summary Report", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=brand_secondary, spaceBefore=4, spaceAfter=8))

    # ---- 1. Scan & System Information ----
    story.append(Paragraph("1. Scan & System Information", heading_style))

    mode_label = "Demo Simulation (Placeholder Data)" if is_demo else "Real Model Inference (Active)"

    info_data = [
        ["Patient / Case Identifier", file_name or "Unspecified Scan"],
        ["Analysis Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["Input Modalities", "Multimodal 3D MRI (T1c, T1, T2, FLAIR)"],
        ["AI Model Architecture", model_name],
        ["Execution Mode", mode_label],
        ["Analysis Status", analysis_status],
    ]

    info_table = Table(info_data, colWidths=[55 * mm, 120 * mm])
    info_table.setStyle(_clean_table_style(header=False))
    story.append(info_table)
    story.append(Spacer(1, 6))

    # ---- 2. Multi-Region Volumetric Measurements ----
    story.append(Paragraph("2. Volumetric & Morphometric Measurements", heading_style))

    if measurements.get("available"):
        regions = measurements.get("regions", {})

        meas_rows = [
            ["Tumor Sub-Region", "Volume (cm³)", "Voxel Count", "Extent (L × W × H mm)", "WT Share (%)"]
        ]

        if "WT" in regions:
            wt = regions["WT"]
            meas_rows.append([
                "Whole Tumor (WT)",
                f"{wt.get('volume_cm3', '—')} cm³",
                f"{wt.get('voxel_count', 0):,}",
                f"{wt.get('length_mm', '—')} × {wt.get('width_mm', '—')} × {wt.get('height_mm', '—')}",
                "100.0%",
            ])
        else:
            meas_rows.append([
                "Whole Tumor (WT)",
                f"{measurements.get('volume_cm3', '—')} cm³",
                f"{measurements.get('voxel_count', 0):,}",
                f"{measurements.get('length_mm', '—')} × {measurements.get('width_mm', '—')} × {measurements.get('height_mm', '—')}",
                "100.0%",
            ])

        if "TC" in regions:
            tc = regions["TC"]
            meas_rows.append([
                "Tumor Core (TC)",
                f"{tc.get('volume_cm3', '—')} cm³",
                f"{tc.get('voxel_count', 0):,}",
                f"{tc.get('length_mm', '—')} × {tc.get('width_mm', '—')} × {tc.get('height_mm', '—')}",
                f"{tc.get('ratio_to_wt_pct', '—')}%",
            ])

        if "ET" in regions:
            et = regions["ET"]
            meas_rows.append([
                "Enhancing Tumor (ET)",
                f"{et.get('volume_cm3', '—')} cm³",
                f"{et.get('voxel_count', 0):,}",
                f"{et.get('length_mm', '—')} × {et.get('width_mm', '—')} × {et.get('height_mm', '—')}",
                f"{et.get('ratio_to_wt_pct', '—')}%",
            ])

        if "ED" in regions:
            ed = regions["ED"]
            meas_rows.append([
                "Peritumoral Edema (ED)",
                f"{ed.get('volume_cm3', '—')} cm³",
                f"{ed.get('voxel_count', 0):,}",
                f"{ed.get('length_mm', '—')} × {ed.get('width_mm', '—')} × {ed.get('height_mm', '—')}",
                f"{ed.get('ratio_to_wt_pct', '—')}%",
            ])

        m_table = Table(meas_rows, colWidths=[45 * mm, 30 * mm, 25 * mm, 50 * mm, 25 * mm])
        m_table.setStyle(_clean_table_style(header=True))
        story.append(m_table)
    else:
        story.append(Paragraph("No segmentation measurements available. Run analysis prior to report generation.", body_style))

    story.append(Spacer(1, 6))

    # ---- 3. Model Confidence & Uncertainty Proxy ----
    story.append(Paragraph("3. Model Confidence / Uncertainty Quantification", heading_style))

    if confidence.get("available"):
        conf_rows = [
            ["Metric", "Value", "Clinical Significance"],
            ["Mean Confidence Proxy", f"{confidence.get('mean', '—')}", "Average sigmoid output over predicted tumor voxels"],
            ["Minimum Confidence", f"{confidence.get('min', '—')}", "Lowest regional agreement at boundary periphery"],
            ["Maximum Confidence", f"{confidence.get('max', '—')}", "Peak certainty within core tumor mass"],
            ["Standard Deviation", f"{confidence.get('std', '—')}", "Spatial dispersion of model certainty"],
        ]
        c_table = Table(conf_rows, colWidths=[50 * mm, 30 * mm, 95 * mm])
        c_table.setStyle(_clean_table_style(header=True))
        story.append(c_table)
    else:
        story.append(Paragraph("No confidence metrics available for this scan.", body_style))

    story.append(Spacer(1, 6))

    # ---- 4. Visual Snapshot (if provided) ----
    if slice_image_bytes is not None:
        try:
            story.append(Paragraph("4. Key Axial Slice Visualization", heading_style))
            img_stream = io.BytesIO(slice_image_bytes)
            img_flowable = Image(img_stream, width=110 * mm, height=110 * mm)
            img_flowable.hAlign = "CENTER"
            story.append(img_flowable)
            story.append(Paragraph("<font size=7 color='#64748B'>Axial slice with NeuroVision AI multi-region segmentation overlay (Crimson: Enhancing Tumor, Orange: Tumor Core, Teal: Whole Tumor / Edema).</font>", subtitle_style))
            story.append(Spacer(1, 6))
        except Exception:
            pass

    # ---- 5. Responsible-Use & Regulatory Disclaimer ----
    story.append(KeepTogether([
        Paragraph("5. Responsible-Use & Regulatory Disclaimer", heading_style),
        Paragraph(
            "<b>RESEARCH PROTOTYPE NOTICE:</b> NeuroVision is an investigational software prototype "
            "designed for multimodal brain MRI segmentation, visualization, and quantitative research. "
            "It is <b>NOT</b> an FDA-cleared or CE-marked medical device and is <b>NOT</b> intended for "
            "direct diagnostic, prognostic, or clinical treatment planning purposes. All segmentation "
            "boundaries, volumetric statistics, and confidence scores are algorithmic outputs that must "
            "be reviewed and verified by a board-certified neuroradiologist or qualified medical professional.",
            disclaimer_style,
        )
    ]))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _clean_table_style(header: bool = False):
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle

    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
    ]
    if header:
        style.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B192C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ])
    return TableStyle(style)
