# ============================================================
# PDF REPORT GENERATOR — IMAGE SESSIONS
# Single-page snapshot report for image analysis sessions
# Output: sessions/<id>/image_analysis_report.pdf
# ============================================================

import json
import csv
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.enums import TA_LEFT


# ============================================================
# COLORS
# ============================================================

NAVY       = colors.HexColor("#0f172a")
BLUE       = colors.HexColor("#2563eb")
GREY       = colors.HexColor("#64748b")
LIGHT_GREY = colors.HexColor("#f1f5f9")
AMBER      = colors.HexColor("#f59e0b")


# ============================================================
# STYLES
# ============================================================

def make_styles():
    s = getSampleStyleSheet()

    s.add(ParagraphStyle(
        name="TitleBig", parent=s["Title"],
        fontSize=20, leading=24, textColor=NAVY,
        alignment=TA_LEFT, spaceAfter=2,
    ))
    s.add(ParagraphStyle(
        name="Subtitle", parent=s["Normal"],
        fontSize=9.5, leading=12, textColor=GREY, spaceAfter=10,
    ))
    s.add(ParagraphStyle(
        name="SectionH", parent=s["Heading2"],
        fontSize=12, leading=15, textColor=NAVY,
        spaceBefore=8, spaceAfter=5,
    ))
    s.add(ParagraphStyle(
        name="BodyTxt", parent=s["Normal"],
        fontSize=9, leading=12,
        textColor=colors.HexColor("#334155"),
    ))
    s.add(ParagraphStyle(
        name="TinyNote", parent=s["Normal"],
        fontSize=7.5, leading=10, textColor=GREY,
    ))
    s.add(ParagraphStyle(
        name="Disclaimer", parent=s["Normal"],
        fontSize=8, leading=11,
        textColor=colors.HexColor("#7c2d12"),
    ))
    return s


# ============================================================
# SECTION BUILDERS
# ============================================================

def build_header(report, styles):
    story = []
    story.append(Paragraph("Image Analysis Report", styles["TitleBig"]))

    src = report.get("source_file", "unknown")
    src_name = Path(src).name if src else "unknown"
    start = report.get("session_start", "")
    start = start[:19].replace("T", " ") if start else "unknown"

    story.append(Paragraph(
        f"Source: <b>{src_name}</b> · Analyzed {start}",
        styles["Subtitle"]
    ))
    return story


def build_disclaimer(styles):
    text = (
        "<b>⚠️ Research / Observational Use Only.</b> "
        "The metrics in this report are heuristic indicators, not "
        "measurements of truthfulness, guilt, or intent. Nonverbal cues have "
        "weak and contested links to deception. Do not use as sole basis for "
        "any investigative or legal decision."
    )
    t = Table([[Paragraph(text, styles["Disclaimer"])]], colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7ed")),
        ("BOX", (0, 0), (-1, -1), 0.6, AMBER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 10)]


def build_feature_table(report, styles):
    story = [Paragraph("Extracted Features", styles["SectionH"])]

    rows = [
        ["Feature", "Value"],
        ["Face Detected",
         "Yes" if report.get("face_detected") else "No"],
        ["Image Size",
         f"{report.get('image_width', 0)} × {report.get('image_height', 0)} px"],
        ["EAR (Eye Aspect Ratio)",
         f"{report.get('ear', 0):.3f}"],
        ["MAR (Mouth Aspect Ratio)",
         f"{report.get('mar', 0):.3f}"],
        ["Brow Raise",
         f"{report.get('brow', 0):.3f}"],
        ["Gaze Direction",
         report.get("gaze_horizontal", "UNKNOWN")],
        ["Head Yaw",
         f"{report.get('yaw', 0):.1f}°"],
        ["Head Pitch",
         f"{report.get('pitch', 0):.1f}°"],
        ["Head Roll",
         f"{report.get('roll', 0):.1f}°"],
        ["Dominant Emotion",
         report.get("emotion_dominant", "unknown").capitalize()],
        ["Emotion Confidence",
         f"{report.get('emotion_confidence', 0):.2f}"],
        ["Blendshape Channels",
         f"{report.get('blendshape_count', 0)}"],
    ]

    t = Table(rows, colWidths=[6*cm, 8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 1), (0, -1), LIGHT_GREY),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (0, -1), NAVY),
        ("ROWBACKGROUNDS", (1, 1), (1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    return story


def build_emotion_breakdown(report, styles):
    scores = report.get("emotion_scores", {})
    if not scores:
        return []

    story = [Spacer(1, 10),
             Paragraph("Emotion Breakdown", styles["SectionH"])]

    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    rows = [["Emotion", "Score", "Relative"]]
    for name, value in items:
        bar_width = max(1, int(round(value * 30)))
        bar = "█" * bar_width
        rows.append([
            name.capitalize(),
            f"{value:.3f}",
            bar
        ])

    t = Table(rows, colWidths=[4*cm, 3*cm, 9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (2, 1), (2, -1), "Courier-Bold"),
        ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#2563eb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Emotion scores are the relative share of blendshape activation "
        "assigned to each emotion class. Values near 1.0 indicate a clear "
        "single expression; values spread across several classes indicate a "
        "mixed or ambiguous expression. These are not measures of internal "
        "state.",
        styles["TinyNote"]
    ))
    return story


def build_annotated_image(session_dir, styles):
    img_path = session_dir / "annotated.png"
    if not img_path.exists():
        return []

    story = [Spacer(1, 10),
             Paragraph("Annotated Image", styles["SectionH"])]

    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(img_path)
        w, h = pil_img.size
        aspect = h / w

        target_w_cm = 11.5
        target_h_cm = target_w_cm * aspect

        if target_h_cm > 15:
            target_h_cm = 15
            target_w_cm = target_h_cm / aspect

        story.append(Image(str(img_path),
                           width=target_w_cm * cm,
                           height=target_h_cm * cm))
        story.append(Spacer(1, 8))
    except Exception as e:
        story.append(Paragraph(f"Could not load image: {e}",
                               styles["BodyTxt"]))

    return story


def build_interpretation(styles):
    story = [Spacer(1, 10),
             Paragraph("Interpretation Guide", styles["SectionH"])]

    guide = """
    <b>EAR (Eye Aspect Ratio)</b> — Ratio of eye height to width. Typical open
    eye ≈ 0.25–0.35; closed ≈ 0.10–0.20.<br/><br/>

    <b>MAR (Mouth Aspect Ratio)</b> — Ratio of mouth height to width. Small
    values (&lt;0.10) indicate a closed or pressed mouth; larger values
    indicate an open mouth.<br/><br/>

    <b>Brow Raise</b> — Normalized brow height above the eyes. Higher values
    indicate raised brows (surprise, questioning); lower or negative values
    indicate lowered brows (focus, anger, concentration).<br/><br/>

    <b>Gaze Direction</b> — Estimated horizontal gaze: LEFT, CENTER, or RIGHT,
    derived from iris position within the eye opening.<br/><br/>

    <b>Head Pose (Yaw / Pitch / Roll)</b> — Rotation angles of the head.
    Yaw: left/right turn. Pitch: up/down nod. Roll: sideways tilt.<br/><br/>

    <b>Emotion</b> — Approximate classification from MediaPipe's 52 facial
    blendshape channels. Treat as one signal among many, not as ground truth.
    """
    story.append(Paragraph(guide, styles["BodyTxt"]))
    return story


def build_footer(styles):
    story = [Spacer(1, 14),
             Paragraph(
                 f"Report generated "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
                 "Behavioral Analysis System · Research use only",
                 styles["TinyNote"]
             )]
    return story


# ============================================================
# BUILDER
# ============================================================

def generate_image_report(session_path, output_path=None):
    """
    Generate a single-page snapshot PDF report for an image session.
    Default output filename: image_analysis_report.pdf
    Returns the output path.
    """
    session_dir = Path(session_path)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    report_path = session_dir / "session_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing session_report.json in {session_dir}"
        )

    with open(report_path, "r") as f:
        report = json.load(f)

    if output_path is None:
        output_path = session_dir / "image_analysis_report.pdf"
    output_path = Path(output_path)

    styles = make_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
        title="Image Analysis Report",
        author="Behavioral Analysis System",
    )

    story = []
    story += build_header(report, styles)
    story += build_disclaimer(styles)
    story += build_feature_table(report, styles)
    story += build_emotion_breakdown(report, styles)
    story += build_annotated_image(session_dir, styles)
    story += build_interpretation(styles)
    story += build_footer(styles)

    doc.build(story)
    return output_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python img_report.py <session_id>")
        print("       (example: python img_report.py img_2026-09-22_172627)")

        sessions_dir = Path("sessions")
        if sessions_dir.exists():
            img_sessions = sorted(
                [d.name for d in sessions_dir.iterdir()
                 if d.is_dir() and d.name.startswith("img_")],
                reverse=True
            )
            if img_sessions:
                print("\nAvailable image sessions:")
                for i, s in enumerate(img_sessions[:10]):
                    print(f"  [{i}] {s}")
        sys.exit(1)

    session_id = sys.argv[1]
    path = Path("sessions") / session_id
    print(f"Generating image report for: {session_id}")

    # Always save as image_analysis_report.pdf so it never collides
    output_pdf = path / "image_analysis_report.pdf"
    out = generate_image_report(path, output_path=output_pdf)
    print(f"[OK] PDF saved: {out}")