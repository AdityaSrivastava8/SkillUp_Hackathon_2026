# ============================================================
# UNIFIED BEHAVIORAL ANALYSIS APP
# Single Streamlit UI: Image / Video / Live / Browse
# ============================================================

import streamlit as st
import pandas as pd
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================

PYTHON = sys.executable
BASE_DIR = Path(__file__).parent
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Behavioral Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    .page-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .page-header .icon { font-size: 2.4rem; line-height: 1; }
    .page-header .title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #fafafa;
        margin: 0;
    }
    .page-header .title span {
        background: linear-gradient(90deg, #4f8cff, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .page-subtitle {
        color: #8b95a5;
        font-size: 0.95rem;
        margin: 0 0 1.6rem 0;
    }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #262d38;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8b95a5 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6edf3 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid #262d38;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: #8b95a5;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #4f8cff !important;
        border-bottom: 2px solid #4f8cff !important;
    }
    .info-card {
        display: flex;
        gap: 14px;
        background: #161b22;
        border: 1px solid #262d38;
        border-left: 3px solid #f5a623;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 0.5rem 0 1.5rem 0;
    }
    .info-card .icon { font-size: 1.3rem; flex-shrink: 0; }
    .info-card .body {
        color: #b6c1ce;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .info-card .body strong { color: #f5a623; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def run_subprocess(cmd, timeout_sec=600):
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout_sec}s"
    except Exception as e:
        return False, "", str(e)


def get_latest_session(prefix=None):
    if not SESSIONS_DIR.exists():
        return None
    dirs = [d for d in SESSIONS_DIR.iterdir() if d.is_dir()]
    if prefix:
        dirs = [d for d in dirs if d.name.startswith(prefix)]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def section(title):
    st.markdown(f"### {title}")


def render_kpis(report, events_df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{report.get('duration_seconds', 0):.1f} s")
    c2.metric("Face Detected", f"{report.get('face_detection_pct', 0):.1f}%")
    c3.metric("Blinks",
              f"{report.get('blink_count', 0)}",
              f"{report.get('blink_rate_per_min', 0):.1f}/min")
    c4.metric("Events", f"{len(events_df)}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Stress Index",
              f"{report.get('final_stress_index', 0):.0f}/100")
    c6.metric("Engagement",
              f"{report.get('final_engagement', 0):.0f}%")
    c7.metric("Avg EAR", f"{report.get('avg_ear', 0):.3f}")
    c8.metric("Dominant Emotion",
              report.get("dominant_emotion_overall", "—").capitalize())


def render_pdf_buttons(session_dir, script_name, pdf_filename,
                       session_name, context="default"):
    section("Report")

    if script_name == "report_generator.py":
        st.caption("The PDF includes time-series charts, emotion distribution, "
                   "gaze heatmap, and the event log.")
    else:
        st.caption("One-page snapshot PDF with extracted features, emotion "
                   "breakdown, and annotated image.")

    pdf_path = session_dir / pdf_filename

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("📄 Generate PDF Report",
                     key=f"gen_pdf_{context}_{session_name}",
                     use_container_width=True):
            with st.spinner("Building PDF..."):
                ok, out, err = run_subprocess(
                    [PYTHON, script_name, session_name],
                    timeout_sec=120,
                )
                if ok:
                    st.success("✓ PDF generated")
                    st.rerun()
                else:
                    st.error(f"PDF generation failed:\n{err[-500:]}")

    with col_btn2:
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF",
                    data=f.read(),
                    file_name=f"report_{session_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_pdf_{context}_{session_name}",
                )
        else:
            st.caption("Click Generate to create the PDF first.")


def render_video_results(session_dir, context="default"):
    """Time-series charts + preview + heatmap + PDF for video/live sessions."""
    report = load_json(session_dir / "session_report.json")
    df = load_csv(session_dir / "video_features.csv")
    events_df = load_csv(session_dir / "behavioral_events.csv")

    # Warn if the video had poor face detection
    face_pct = report.get("face_detection_pct", 0)
    if face_pct < 5:
        st.warning(
            "**Very few frames had a detectable face.**\n\n"
            f"Face detection rate: **{face_pct:.1f}%**\n\n"
            "The analyzer needs a clear view of a person's face for accurate "
            "results. Try a video with better lighting and a front-facing person."
        )
    else:
        st.success(f"✓ Analysis complete — {session_dir.name}")

    render_kpis(report, events_df)
    # --- Time series ---
    if "elapsed_seconds" in df.columns and len(df) > 1:
        section("Time Series")
        fig = go.Figure()
        for col, color in [("ear", "#4f8cff"),
                           ("stress_index", "#e5484d"),
                           ("engagement", "#2ea043")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["elapsed_seconds"], y=df[col],
                    name=col, line=dict(width=2, color=color),
                ))
        fig.update_layout(
            height=340, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True,
                        key=f"ts_chart_{context}_{session_dir.name}")

    # --- Emotion distribution ---
    if "emotion_dominant" in df.columns:
        section("Emotion Distribution")
        counts = df["emotion_dominant"].value_counts().reset_index()
        counts.columns = ["Emotion", "Frames"]
        fig2 = px.bar(counts, x="Emotion", y="Frames", color="Emotion",
                      text="Frames")
        fig2.update_layout(
            height=300, showlegend=False,
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True,
                        key=f"emo_chart_{context}_{session_dir.name}")

    # --- Preview + heatmap ---
    col_a, col_b = st.columns(2)
    preview = session_dir / "annotated_preview.png"
    heatmap = session_dir / "gaze_heatmap.png"
    with col_a:
        if preview.exists():
            st.image(str(preview), caption="Peak-stress frame",
                     use_container_width=True)
    with col_b:
        if heatmap.exists():
            st.image(str(heatmap), caption="Gaze heatmap",
                     use_container_width=True)

    # --- PDF buttons ---
    render_pdf_buttons(
        session_dir=session_dir,
        script_name="report_generator.py",
        pdf_filename="report.pdf",
        session_name=session_dir.name,
        context=context,
    )


def render_image_results(session_dir, context="default"):
    """Annotated image + features + PDF for image sessions."""
    report = load_json(session_dir / "session_report.json")

    if not report.get("face_detected"):
        # Friendly info message — not an error
        st.warning(
            "**No face detected in the uploaded image.**\n\n"
            "The analyzer couldn't find a face in the photo. This usually means:\n"
            "- The image is a screenshot or contains no person\n"
            "- The face is too small, blurred, or at an extreme angle\n"
            "- The image is very dark or heavily stylized\n\n"
            "**Try again with a clear, front-facing photo of a person.**"
        )
        return

    st.success(f"✓ Analysis complete — {session_dir.name}")

    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Emotion", report.get("emotion_dominant", "—").capitalize())
    c2.metric("Confidence", f"{report.get('emotion_confidence', 0):.2f}")
    c3.metric("EAR", f"{report.get('ear', 0):.3f}")
    c4.metric("MAR", f"{report.get('mar', 0):.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Yaw", f"{report.get('yaw', 0):.1f}°")
    c6.metric("Pitch", f"{report.get('pitch', 0):.1f}°")
    c7.metric("Roll", f"{report.get('roll', 0):.1f}°")
    c8.metric("Gaze", report.get("gaze_horizontal", "—"))

    # --- Annotated image ---
    section("Annotated Image")
    annotated = session_dir / "annotated.png"
    if annotated.exists():
        st.image(str(annotated), use_container_width=True)

    # --- PDF buttons ---
    render_pdf_buttons(
        session_dir=session_dir,
        script_name="img_report.py",
        pdf_filename="image_analysis_report.pdf",
        session_name=session_dir.name,
        context=context,
    )
    # ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="page-header">
    <div class="icon">🧠</div>
    <div class="title">Behavioral <span>Analysis</span> System</div>
</div>
<div class="page-subtitle">Analyze images, videos, or live webcam — extract 25+ behavioral signals and download a report.</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="info-card">
    <div class="icon">⚠️</div>
    <div class="body">
        <strong>Research / Observational Use Only</strong>
        The metrics in this tool are heuristic indicators, not measurements
        of truthfulness, guilt, or intent. Nonverbal cues have weak and
        contested links to deception. Do not use as sole basis for any
        investigative or legal decision.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab_image, tab_video, tab_live, tab_browse = st.tabs([
    "🖼️  Upload Image",
    "🎥  Upload Video",
    "🔴  Live Webcam",
    "📚  Browse Sessions",
])


# ============================================================
# TAB 1 — IMAGE
# ============================================================

with tab_image:

    st.markdown("#### Analyze a single photo")
    st.caption("Upload a clear, front-facing photo. The system extracts facial "
               "features, gaze direction, head pose, and emotion from that one frame.")

    uploaded_img = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_uploader",
    )

    if uploaded_img is not None:
        save_path = UPLOAD_DIR / uploaded_img.name
        with open(save_path, "wb") as f:
            f.write(uploaded_img.getbuffer())

        col_preview, col_info = st.columns([2, 1])
        with col_preview:
            st.image(str(save_path), caption="Uploaded image",
                     use_container_width=True)
        with col_info:
            st.markdown("**Ready to analyze**")
            st.caption(f"File: `{uploaded_img.name}`")
            st.caption(f"Size: {uploaded_img.size / 1024:.1f} KB")

            if st.button("🔬 Run Analysis",
                         type="primary",
                         use_container_width=True,
                         key="analyze_img_btn"):
                with st.spinner("Analyzing image..."):
                    ok, out, err = run_subprocess(
                        [PYTHON, "batch_analysis.py", str(save_path)],
                        timeout_sec=120,
                    )
                    if ok:
                        latest = get_latest_session(prefix="img_")
                        if latest:
                            st.session_state["last_img_session"] = str(latest)
                            st.session_state["img_context"] = "img_tab"
                            st.rerun()
                        else:
                            st.error("Analysis ran but no session folder found.")
                    else:
                        st.error(f"Analysis failed:\n{err[-800:]}")

    if "last_img_session" in st.session_state:
        # Only show in Image tab if the context is image-related
        ctx = st.session_state.get("img_context", "img_tab")
        if ctx.startswith("img"):
            st.markdown("---")
            session_path = Path(st.session_state["last_img_session"])
            if session_path.exists():
                render_image_results(session_path, context=ctx)


# ============================================================
# TAB 2 — VIDEO
# ============================================================

with tab_video:

    st.markdown("#### Analyze a video file")
    st.caption("Upload an MP4, AVI, MOV, MKV, or WebM file. The system "
               "samples at 10 fps and extracts features across the full clip.")

    uploaded_vid = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="vid_uploader",
    )

    if uploaded_vid is not None:
        save_path = UPLOAD_DIR / uploaded_vid.name
        with open(save_path, "wb") as f:
            f.write(uploaded_vid.getbuffer())

        col_preview, col_info = st.columns([2, 1])
        with col_preview:
            st.video(str(save_path))
        with col_info:
            st.markdown("**Ready to analyze**")
            st.caption(f"File: `{uploaded_vid.name}`")
            st.caption(f"Size: {uploaded_vid.size / 1024 / 1024:.1f} MB")
            st.caption("Analysis time scales with video length.")

            if st.button("🔬 Run Analysis",
                         type="primary",
                         use_container_width=True,
                         key="analyze_vid_btn"):
                with st.spinner("Analyzing video — this may take a minute..."):
                    ok, out, err = run_subprocess(
                        [PYTHON, "batch_video_analysis.py", str(save_path)],
                        timeout_sec=900,
                    )
                    if ok:
                        latest = get_latest_session(prefix="vid_")
                        if latest:
                            st.session_state["last_vid_session"] = str(latest)
                            st.session_state["vid_context"] = "vid_tab"
                            st.rerun()
                        else:
                            st.error("Analysis ran but no session folder found.")
                    else:
                        st.error(f"Analysis failed:\n{err[-800:]}")

    if "last_vid_session" in st.session_state:
        ctx = st.session_state.get("vid_context", "vid_tab")
        if ctx.startswith("vid"):
            st.markdown("---")
            session_path = Path(st.session_state["last_vid_session"])
            if session_path.exists():
                render_video_results(session_path, context=ctx)


# ============================================================
# TAB 3 — LIVE
# ============================================================

with tab_live:

    st.markdown("#### Live webcam analysis")
    st.caption("Launches the live analyzer in a separate window. The "
               "dashboard will detect the new session when you finish recording.")

    st.info(
        "**How it works:**\n"
        "1. Click **Start Live Session** — a new camera window opens.\n"
        "2. Record for as long as you like (aim for 30+ seconds).\n"
        "3. Press **Q** in the camera window to stop.\n"
        "4. Return here — the session will be listed below.\n"
        "5. Click **View** to see results."
    )

    if st.button("🔴 Start Live Session",
                 type="primary",
                 use_container_width=True,
                 key="start_live_btn"):
        try:
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen(
                [PYTHON, "analyzer.py"],
                cwd=str(BASE_DIR),
                creationflags=creation_flags,
            )
            st.success("✓ Analyzer launched in a new window. "
                       "Press Q there when done.")
            st.caption("Refresh this page after the session ends to see "
                       "the new folder below.")
        except Exception as e:
            st.error(f"Could not launch analyzer: {e}")

    st.markdown("---")
    st.markdown("##### Recent live sessions")

    live_sessions = sorted(
        [d for d in SESSIONS_DIR.iterdir()
         if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )[:5]

    if not live_sessions:
        st.caption("No live sessions yet.")
    else:
        for s in live_sessions:
            cols = st.columns([3, 2, 1, 1])
            with cols[0]:
                st.markdown(f"**`{s.name}`**")
            with cols[1]:
                # Parse folder name → "Sep 22 · 6:33 PM (5 min ago)"
                try:
                    dt = datetime.strptime(s.name, "%Y-%m-%d_%H%M%S")
                    friendly = dt.strftime("%b %d · %I:%M %p").replace(" 0", " ")
                    # Remove leading zero in the hour for cleaner look
                    friendly = friendly.lstrip("0")

                    # Time-ago
                    delta = datetime.now() - dt
                    secs = int(delta.total_seconds())
                    if secs < 60:
                        ago = "just now"
                    elif secs < 3600:
                        ago = f"{secs // 60} min ago"
                    elif secs < 86400:
                        ago = f"{secs // 3600} hr ago"
                    else:
                        ago = f"{secs // 86400} day(s) ago"

                    st.caption(f"{friendly}  ·  {ago}")
                except Exception:
                    st.caption("—")
            with cols[2]:
                if (s / "session_report.json").exists():
                    st.caption("✓ complete")
                else:
                    st.caption("incomplete")
            with cols[3]:
                if st.button("View", key=f"view_live_{s.name}"):
                    st.session_state["last_vid_session"] = str(s)
                    st.session_state["vid_context"] = "live_tab"
                    st.rerun()

    if ("last_vid_session" in st.session_state and
            st.session_state.get("vid_context") == "live_tab"):
        session_path = Path(st.session_state["last_vid_session"])
        if session_path.exists() and (session_path / "session_report.json").exists():
            st.markdown("---")
            render_video_results(session_path, context="live_tab")


# ============================================================
# TAB 4 — BROWSE
# ============================================================

with tab_browse:

    st.markdown("#### All recorded sessions")
    st.caption("Click any session to view its details.")

    all_sessions = sorted(
        [d for d in SESSIONS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if not all_sessions:
        st.info("No sessions yet. Analyze an image, video, or start a live session.")
    else:
        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            filter_type = st.selectbox(
                "Filter by type",
                ["All", "Image (img_*)", "Video (vid_*)", "Live (numeric)"],
            )
        with col_f2:
            st.caption(f"Total: {len(all_sessions)} sessions")

        filtered = all_sessions
        if filter_type == "Image (img_*)":
            filtered = [s for s in all_sessions if s.name.startswith("img_")]
        elif filter_type == "Video (vid_*)":
            filtered = [s for s in all_sessions if s.name.startswith("vid_")]
        elif filter_type == "Live (numeric)":
            filtered = [s for s in all_sessions if s.name[0].isdigit()]

        for s in filtered[:30]:
            has_pdf = ((s / "report.pdf").exists() or
                       (s / "image_analysis_report.pdf").exists())
            size_mb = sum(f.stat().st_size for f in s.iterdir()
                          if f.is_file()) / 1024 / 1024

            cols = st.columns([3, 2, 1, 1, 1])
            with cols[0]:
                st.markdown(f"**`{s.name}`**")
            with cols[1]:
                try:
                    dt = datetime.strptime(s.name, "%Y-%m-%d_%H%M%S")
                    friendly = dt.strftime("%b %d · %I:%M %p").lstrip("0")
                    delta = datetime.now() - dt
                    secs = int(delta.total_seconds())
                    if secs < 60:
                        ago = "just now"
                    elif secs < 3600:
                        ago = f"{secs // 60} min ago"
                    elif secs < 86400:
                        ago = f"{secs // 3600} hr ago"
                    else:
                        ago = f"{secs // 86400} day(s) ago"
                    st.caption(f"{friendly} · {ago}")
                except Exception:
                    st.caption("—")
            with cols[2]:
                st.caption(f"{size_mb:.1f} MB")
            with cols[3]:
                st.caption("📄 PDF" if has_pdf else "—")
            with cols[4]:
                if st.button("Open", key=f"browse_{s.name}"):
                    if s.name.startswith("img_"):
                        st.session_state["last_img_session"] = str(s)
                        st.session_state["img_context"] = "browse_img"
                    else:
                        st.session_state["last_vid_session"] = str(s)
                        st.session_state["vid_context"] = "browse_vid"
                    st.rerun()

        # Show whichever session was opened from Browse tab
        img_ctx = st.session_state.get("img_context", "")
        vid_ctx = st.session_state.get("vid_context", "")

        if img_ctx == "browse_img" and "last_img_session" in st.session_state:
            session_path = Path(st.session_state["last_img_session"])
            if session_path.exists():
                st.markdown("---")
                render_image_results(session_path, context="browse_img")
        elif vid_ctx == "browse_vid" and "last_vid_session" in st.session_state:
            session_path = Path(st.session_state["last_vid_session"])
            if session_path.exists():
                st.markdown("---")
                render_video_results(session_path, context="browse_vid")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    f"Behavioral Analysis System · {datetime.now().strftime('%Y-%m-%d')} · "
    "Research use only. Not for investigative or legal decisions."
)