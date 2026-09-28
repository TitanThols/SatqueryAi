"""
api/report_generator.py
Generates a PDF report from an ExecutionTrace using ReportLab.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def generate_pdf_report(trace, out_path: Path) -> None:
    """Generate PDF report from ExecutionTrace dataclass."""
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    style_title   = ParagraphStyle("title",   parent=styles["Heading1"], fontSize=18, spaceAfter=6)
    style_h2      = ParagraphStyle("h2",      parent=styles["Heading2"], fontSize=13, spaceAfter=4)
    style_body    = ParagraphStyle("body",    parent=styles["Normal"],   fontSize=10, spaceAfter=3)
    style_mono    = ParagraphStyle("mono",    parent=styles["Code"],     fontSize=8,  spaceAfter=3,
                                   fontName="Courier")
    style_warning = ParagraphStyle("warn",    parent=styles["Normal"],   fontSize=10,
                                   textColor=colors.orange)
    style_error   = ParagraphStyle("err",     parent=styles["Normal"],   fontSize=10,
                                   textColor=colors.red)

    story = []

    # ── Header ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("SatQuery AI — Analysis Report", style_title))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", style_body))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 6 * mm))

    # ── Summary table ─────────────────────────────────────────────────────────
    story.append(Paragraph("Analysis Summary", style_h2))
    summary_data = [
        ["Field", "Value"],
        ["Query",         trace.query],
        ["Task",          trace.task],
        ["Model Label",   trace.model_label],
        ["Confidence",    f"{trace.confidence:.3f}"],
        ["Duration",      f"{trace.total_duration_s:.2f} s"],
    ]
    if trace.error:
        summary_data.append(["Error", trace.error])

    tbl = Table(summary_data, colWidths=[50 * mm, 120 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f0f0"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP", (1, 0), (1, -1), True),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6 * mm))

    # ── Answer ────────────────────────────────────────────────────────────────
    if trace.answer:
        story.append(Paragraph("Answer", style_h2))
        story.append(Paragraph(trace.answer, style_body))
        story.append(Spacer(1, 4 * mm))

    # ── Validation messages ───────────────────────────────────────────────────
    if trace.validation_warnings:
        story.append(Paragraph("Warnings", style_h2))
        for w in trace.validation_warnings:
            story.append(Paragraph(f"⚠ {w}", style_warning))
        story.append(Spacer(1, 4 * mm))

    if trace.validation_errors:
        story.append(Paragraph("Validation Errors", style_h2))
        for e in trace.validation_errors:
            story.append(Paragraph(f"✗ {e}", style_error))
        story.append(Spacer(1, 4 * mm))

    # ── Tool execution trace ──────────────────────────────────────────────────
    story.append(Paragraph("Execution Trace", style_h2))
    for i, tc in enumerate(trace.tool_calls):
        story.append(Paragraph(f"Step {i+1}: {tc['tool_name']}", style_h2))
        trace_data = [
            ["Duration", f"{tc['duration_s']:.3f} s"],
            ["Params",   json.dumps({k: v for k, v in tc["params"].items()
                                      if k not in ("change_mask_array",)}, indent=2)],
        ]
        if tc.get("error"):
            trace_data.append(["Error", tc["error"]])
        if tc.get("output"):
            out_str = json.dumps(
                {k: v for k, v in tc["output"].items()
                 if k not in ("change_mask_array", "prob_map")},
                indent=2,
            )
            trace_data.append(["Output", out_str])

        tbl2 = Table(trace_data, colWidths=[35 * mm, 135 * mm])
        tbl2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f4f8")),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("FONTNAME",   (1, 0), (1, -1), "Courier"),
            ("FONTSIZE",   (1, 0), (1, -1), 7),
        ]))
        story.append(tbl2)
        story.append(Spacer(1, 3 * mm))

    # ── Result images ─────────────────────────────────────────────────────────
    if trace.image_outputs:
        story.append(Paragraph("Result Images", style_h2))
        for img_path in trace.image_outputs:
            p = Path(img_path)
            if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                try:
                    story.append(Paragraph(p.name, style_body))
                    story.append(RLImage(str(p), width=120 * mm, height=80 * mm, kind="proportional"))
                    story.append(Spacer(1, 3 * mm))
                except Exception:
                    story.append(Paragraph(f"[Image: {p.name}]", style_body))

    # ── Confidence explanation ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Confidence Methodology", style_h2))
    story.append(Paragraph(
        "VQA/Caption confidence: exp(mean log-probability) over generated tokens. "
        "Change map confidence: mean change-probability over all pixels. "
        "Fusion confidence: mean max-class sigmoid probability. "
        "Confidence is never invented; if computation fails, 0.0 is reported.",
        style_body,
    ))

    doc.build(story)
