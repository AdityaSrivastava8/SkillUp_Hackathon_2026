# ============================================================
# PDF REPORT GENERATOR — No page gaps
# Reads a session folder and produces a professional PDF
# ============================================================

import json
import csv
from pathlib import Path
from datetime import datetime
import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
LIGHT_BLUE = colors.HexColor("#dbeafe")
GREY       = colors.HexColor("#64748b")
LIGHT_GREY = colors.HexColor("#f1f5f9")
AMBER      = colors.HexColor("#f59e0b")
RED        = colors.HexColor("#dc2626")
GREEN      = colors.HexColor("#16a34a")


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def fig_to_image(fig, width_cm, height_cm):
    """
    Convert matplotlib figure to ReportLab Image.
    Saves WITHOUT bbox_inches='tight' so output pixel dims match
    the target box -> no padding, no gaps.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


def make_line_chart(df, x_col, y_cols, title, colors_list=None,
                    ylabel="", xlabel="Elapsed (s)",
                    fig_w_in=8.5, fig_h_in=2.1):
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")

    if colors_list is None:
        colors_list = ["#2563eb", "#f59e0b", "#dc2626", "#16a34a",
                       "#8b5cf6", "#0ea5e9"]

    for i, col in enumerate(y_cols):
        if col not in df.columns:
            continue
        ax.plot(df[x_col], df[col],
                label=col, linewidth=1.6,
                color=colors_list[i % len(colors_list)])

    ax.set_title(title, fontsize=10, fontweight="bold", color="#0f172a", pad=6)
    ax.set_xlabel(xlabel, fontsize=8, color="#64748b", labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, color="#64748b", labelpad=2)
    ax.tick_params(labelsize=7, colors="#64748b", length=3)
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    if len(y_cols) > 1:
        ax.legend(fontsize=7, loc="best", framealpha=0.9,
                  borderpad=0.4, labelspacing=0.3)
    fig.tight_layout(pad=0.4)
    return fig


def make_bar_chart(labels, values, title, color="#2563eb",
                   xlabel="", horizontal=False,
                   fig_w_in=8.5, fig_h_in=2.1):
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")

    if horizontal:
        ax.barh(labels, values, color=color)
        ax.set_xlabel(xlabel, fontsize=8, color="#64748b", labelpad=2)
    else:
        ax.bar(labels, values, color=color)
        ax.set_ylabel(xlabel, fontsize=8, color="#64748b", labelpad=2)

    ax.set_title(title, fontsize=10, fontweight="bold", color="#0f172a", pad=6)
    ax.tick_params(labelsize=7, colors="#64748b", length=3)
    ax.grid(True, alpha=0.25, axis="x" if horizontal else "y")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    fig.tight_layout(pad=0.4)
    return fig


# ============================================================
# REPORT BUILDER
# ============================================================

class ReportBuilder:

    def __init__(self, session_path, output_path=None):
        self.session = Path(session_path)
        if not self.session.exists():
            raise FileNotFoundError(f"Session not found: {self.session}")

        self.report = load_json(self.session / "session_report.json")
        self.df = load_csv(self.session / "video_features.csv")
        self.events = load_csv(self.session / "behavioral_events.csv")
        self.summary = load_csv(self.session / "session_summary.csv")
        self.heatmap_path = self.session / "gaze_heatmap.png"

        if output_path is None:
            output_path = self.session / "report.pdf"
        self.output_path = Path(output_path)

        self.styles = self._make_styles()

    def _make_styles(self):
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
            spaceBefore=6, spaceAfter=4,
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

    # --------------------------------------------------------
    # SECTION BUILDERS
    # --------------------------------------------------------

    def _header(self):
        story = []
        story.append(Paragraph("Behavioral Analysis Report",
                               self.styles["TitleBig"]))
        story.append(Paragraph(
            f"Session ID: <b>{self.report.get('session_id', 'N/A')}</b> · "
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles["Subtitle"]
        ))

        data = [
            ["Duration", f"{self.report.get('duration_seconds', 0):.1f} s",
             "Face Detected", f"{self.report.get('face_detection_pct', 0):.1f}%"],
            ["Total Frames", f"{self.report.get('total_frames', 0)}",
             "Blinks", f"{self.report.get('blink_count', 0)}"],
            ["Blink Rate", f"{self.report.get('blink_rate_per_min', 0):.1f}/min",
             "Gaze Aversions", f"{self.report.get('gaze_aversion_events', 0)}"],
        ]
        t = Table(data, colWidths=[3.5*cm, 4.2*cm, 3.5*cm, 4.2*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT_GREY),
            ("BACKGROUND", (2, 0), (2, -1), LIGHT_GREY),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
            ("TEXTCOLOR", (2, 0), (2, -1), NAVY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t)
        return story

    def _disclaimer(self):
        text = (
            "<b>⚠️ Research / Observational Use Only.</b> "
            "The metrics in this report are heuristic indicators, not "
            "measurements of truthfulness, guilt, or intent. Nonverbal cues "
            "have weak and contested links to deception. Do not use as sole "
            "basis for any investigative or legal decision. If used in an "
            "investigative context, this report and the model behind it may "
            "be discoverable by the defense."
        )
        t = Table([[Paragraph(text, self.styles["Disclaimer"])]],
                  colWidths=[17*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7ed")),
            ("BOX", (0, 0), (-1, -1), 0.6, AMBER),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return [Spacer(1, 6), t, Spacer(1, 6)]

    def _kpi_section(self):
        story = [Paragraph("Key Performance Indicators",
                           self.styles["SectionH"])]

        def kpi_cell(label, value, sub=""):
            txt = f"<font size='13' color='#0f172a'><b>{value}</b></font>"
            if sub:
                txt += f"<br/><font size='7.5' color='#64748b'>{sub}</font>"
            txt += f"<br/><font size='6.5' color='#64748b'>{label}</font>"
            return Paragraph(txt, self.styles["BodyTxt"])

        s = self.report

        data = [[
            kpi_cell("STRESS INDEX", f"{s.get('final_stress_index', 0):.0f}",
                     "/ 100"),
            kpi_cell("ENGAGEMENT", f"{s.get('final_engagement', 0):.0f}", "%"),
            kpi_cell("AVG EAR", f"{s.get('avg_ear', 0):.3f}", ""),
            kpi_cell("AVG MAR", f"{s.get('avg_mar', 0):.3f}", ""),
        ], [
            kpi_cell("AVG YAW", f"{s.get('avg_yaw', 0):.1f}", "°"),
            kpi_cell("AVG PITCH", f"{s.get('avg_pitch', 0):.1f}", "°"),
            kpi_cell("AVG VELOCITY",
                     f"{s.get('avg_landmark_velocity', 0):.3f}", ""),
            kpi_cell("AVG FLOW",
                     f"{s.get('avg_optical_flow', 0):.2f}", ""),
        ]]

        t = Table(data, colWidths=[4.25*cm]*4, rowHeights=[1.7*cm, 1.7*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        return story

    def _time_series_section(self):
        story = [Spacer(1, 6),
                 Paragraph("Time Series", self.styles["SectionH"])]

        if self.df.empty or "elapsed_seconds" not in self.df.columns:
            story.append(Paragraph("No time-series data available.",
                                   self.styles["BodyTxt"]))
            return story

        df = self.df

        cols = [c for c in ["ear", "blink_rate"] if c in df.columns]
        if cols:
            fig = make_line_chart(df, "elapsed_seconds", cols,
                                  "Eye Aspect Ratio & Blink Rate",
                                  ylabel="Value")
            story.append(fig_to_image(fig, 17, 4.2))
            story.append(Spacer(1, 3))

        cols = [c for c in ["yaw", "pitch", "roll"] if c in df.columns]
        if cols:
            fig = make_line_chart(df, "elapsed_seconds", cols,
                                  "Head Pose (Yaw / Pitch / Roll)",
                                  colors_list=["#2563eb", "#f59e0b", "#dc2626"],
                                  ylabel="Degrees")
            story.append(fig_to_image(fig, 17, 4.2))
            story.append(Spacer(1, 3))

        cols = [c for c in ["stress_index", "engagement"] if c in df.columns]
        if cols:
            fig = make_line_chart(df, "elapsed_seconds", cols,
                                  "Stress Index & Engagement",
                                  colors_list=["#dc2626", "#16a34a"],
                                  ylabel="Score (0-100)")
            story.append(fig_to_image(fig, 17, 4.2))

        return story

    def _motion_section(self):
        if not all(c in self.df.columns for c in
                   ["landmark_velocity", "jitter_index", "optical_flow"]):
            return []

        story = [Spacer(1, 6),
                 Paragraph("Motion Analysis", self.styles["SectionH"])]

        df = self.df
        fig = make_line_chart(
            df, "elapsed_seconds",
            ["landmark_velocity", "jitter_index", "optical_flow"],
            "Frame-to-Frame Motion Signals",
            colors_list=["#2563eb", "#f59e0b", "#8b5cf6"],
            ylabel="Magnitude"
        )
        story.append(fig_to_image(fig, 17, 4.2))
        story.append(Spacer(1, 2))
        story.append(Paragraph(
            "<b>Landmark velocity</b>: avg per-frame landmark displacement, "
            "normalized by face width. "
            "<b>Jitter index</b>: std-dev of recent velocities — high values "
            "indicate restlessness or tremor. "
            "<b>Optical flow</b>: dense pixel motion magnitude in the face region.",
            self.styles["TinyNote"]
        ))
        return story

    def _emotion_section(self):
        if "emotion_dominant" not in self.df.columns:
            return []

        story = [Spacer(1, 6),
                 Paragraph("Emotion Distribution", self.styles["SectionH"])]

        df = self.df
        counts = df["emotion_dominant"].value_counts()

        labels = list(counts.index)
        values = list(counts.values)

        fig, ax = plt.subplots(figsize=(8.5, 2.4), dpi=140)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8fafc")

        color_map = {
            "angry": "#e5484d", "disgust": "#8b5cf6", "fear": "#a855f7",
            "happy": "#2ea043", "sad": "#3b82f6", "surprise": "#f5a623",
            "neutral": "#6b7280",
        }
        bar_colors = [color_map.get(e, "#888") for e in labels]
        bars = ax.bar(labels, values, color=bar_colors)
        total = sum(values)
        for bar, v in zip(bars, values):
            pct = 100 * v / total
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.015,
                    f"{pct:.1f}%", ha="center", fontsize=7.5, color="#334155")
        ax.set_title("Emotion Distribution (% of frames)",
                     fontsize=10, fontweight="bold", color="#0f172a", pad=6)
        ax.tick_params(labelsize=7, colors="#64748b", length=3)
        ax.grid(True, alpha=0.25, axis="y")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        fig.tight_layout(pad=0.4)
        story.append(fig_to_image(fig, 17, 4.4))
        story.append(Spacer(1, 2))

        if len(counts) > 0:
            dom = counts.idxmax()
            pct = 100 * counts.max() / counts.sum()
            story.append(Paragraph(
                f"<b>Dominant emotion:</b> {dom.capitalize()} "
                f"({pct:.1f}% of frames)",
                self.styles["BodyTxt"]
            ))
            story.append(Spacer(1, 3))
            story.append(Paragraph(
                "Emotion labels are derived from MediaPipe blendshape "
                "coefficients using weighted heuristic rules. They are "
                "approximate and should not be treated as ground truth "
                "psychological state.",
                self.styles["TinyNote"]
            ))

        return story

    def _heatmap_section(self):
        if not self.heatmap_path.exists():
            return []

        story = [Spacer(1, 6),
                 Paragraph("Gaze Heatmap", self.styles["SectionH"])]

        story.append(Paragraph(
            "Accumulated gaze points over the session. Warmer colors indicate "
            "regions of more frequent attention. This is a visual summary of "
            "eye orientation — it does not indicate truthfulness, guilt, or "
            "intent.",
            self.styles["BodyTxt"]
        ))
        story.append(Spacer(1, 4))

        story.append(Image(str(self.heatmap_path),
                           width=12*cm, height=9*cm))
        return story

    def _events_section(self):
        if self.events.empty:
            return []

        story = [Spacer(1, 6),
                 Paragraph("Behavioral Events", self.styles["SectionH"])]

        ev_counts = self.events["event"].value_counts().reset_index()
        ev_counts.columns = ["Event", "Count"]

        fig = make_bar_chart(
            list(ev_counts["Event"]),
            list(ev_counts["Count"]),
            "Events by Type",
            color="#2563eb",
            xlabel="Count",
            horizontal=True,
        )
        story.append(fig_to_image(fig, 15, 3.8))
        story.append(Spacer(1, 4))

        story.append(Paragraph("Event Log (first 25)",
                               self.styles["SectionH"]))

        rows = [["Timestamp", "Elapsed (s)", "Event", "Detail"]]
        for _, row in self.events.head(25).iterrows():
            rows.append([
                str(row.get("timestamp", ""))[:19],
                f"{row.get('elapsed_seconds', 0):.1f}",
                str(row.get("event", "")),
                str(row.get("detail", ""))[:60],
            ])

        t = Table(rows, colWidths=[4*cm, 2.2*cm, 3.5*cm, 7*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        return story

    def _footer(self):
        story = [Spacer(1, 6),
                 Paragraph("Summary & Interpretation Guide",
                           self.styles["SectionH"])]

        guide = """
        <b>Stress Index (0-100)</b> — Composite heuristic. Not a measurement of
        deception. High values can reflect discomfort, caffeine, dry eyes, room
        temperature, cognitive load, or simply being watched.<br/><br/>

        <b>Engagement</b> — Combination of face presence and centered gaze.<br/><br/>

        <b>Emotions</b> — Approximate labels from facial blendshapes. Common
        confusions: neutral ↔ sad; surprise ↔ fear.<br/><br/>

        <b>Motion Analysis</b> — Landmark velocity captures facial movement;
        jitter captures high-frequency motion; optical flow captures pixel-level
        motion in the face region.<br/><br/>

        <b>Gaze Heatmap</b> — Where the subject's eyes were oriented. Attention
        proxy, not a truthfulness indicator.<br/><br/>

        <b>Person ID</b> — ByteTrack-assigned ID. Persists while a person
        remains in view.<br/><br/>

        <b>Z-scores</b> — How unusual a value is for this person relative to
        their own first 10 seconds.<br/><br/>

        <b>Legal / Ethical Note</b> — If used in any investigative context, this
        report and the model behind it may be discoverable by the defense.
        Consult counsel before operational use.
        """
        story.append(Paragraph(guide, self.styles["BodyTxt"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            f"Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            "Behavioral Analysis System · Research use only",
            self.styles["TinyNote"]
        ))
        return story

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    def build(self):
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            leftMargin=1.5*cm,
            rightMargin=1.5*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm,
            title="Behavioral Analysis Report",
            author="Behavioral Analysis System",
        )

        story = []
        story += self._header()
        story += self._disclaimer()
        story += self._kpi_section()
        story += self._time_series_section()
        story += self._motion_section()
        story += self._emotion_section()
        story += self._heatmap_section()
        story += self._events_section()
        story += self._footer()

        doc.build(story)
        return self.output_path


# ============================================================
# CLI ENTRY POINT
# ============================================================

def generate_report(session_path, output_path=None):
    """Generate a PDF report for a single session. Returns the output path."""
    builder = ReportBuilder(session_path, output_path)
    return builder.build()


def list_sessions(base="sessions"):
    p = Path(base)
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()], reverse=True)


if __name__ == "__main__":
    import sys

    sessions = list_sessions()
    if not sessions:
        print("No sessions found in ./sessions/")
        sys.exit(1)

    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target in sessions:
            session_name = target
        else:
            print(f"Session '{target}' not found.")
            print("Available sessions:")
            for s in sessions:
                print(f"  - {s}")
            sys.exit(1)
    else:
        print("Available sessions:")
        for i, s in enumerate(sessions):
            print(f"  [{i}] {s}")
        choice = input("\nEnter number (default 0): ").strip() or "0"
        try:
            session_name = sessions[int(choice)]
        except (ValueError, IndexError):
            print("Invalid choice.")
            sys.exit(1)

    session_path = Path("sessions") / session_name
    print(f"\nGenerating report for: {session_name}")
    out = generate_report(session_path)
    print(f"[OK] PDF saved: {out}")