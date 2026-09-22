# ============================================================
# BEHAVIORAL ANALYSIS — STREAMLIT DASHBOARD
# With Emotion, Motion, Person Tracking, Gaze Heatmap
# ============================================================

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Behavioral Analysis Review",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .page-header .icon { font-size: 2.2rem; line-height: 1; }
    .page-header .title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #fafafa;
        letter-spacing: -0.5px;
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
        font-size: 0.9rem;
        margin: 0 0 1.6rem 0;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 600;
        color: #e6edf3;
        margin: 1.8rem 0 0.9rem 0;
    }
    .section-header::before {
        content: "";
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, #4f8cff, #8b5cf6);
        border-radius: 2px;
    }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #262d38;
        border-radius: 10px;
        padding: 14px 16px;
        transition: border-color 0.15s ease;
    }
    div[data-testid="stMetric"]:hover { border-color: #4f8cff; }
    div[data-testid="stMetric"] label {
        color: #8b95a5 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 1.55rem !important;
        font-weight: 600 !important;
    }
    .info-card {
        display: flex;
        gap: 14px;
        background: #161b22;
        border: 1px solid #262d38;
        border-left: 3px solid #f5a623;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 0.5rem 0 1.2rem 0;
    }
    .info-card .info-icon {
        font-size: 1.3rem;
        line-height: 1.4;
        flex-shrink: 0;
    }
    .info-card .info-body {
        color: #b6c1ce;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .info-card .info-body strong {
        color: #f5a623;
        display: block;
        margin-bottom: 4px;
        font-size: 0.92rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid #262d38;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: #8b95a5;
        font-weight: 500;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #4f8cff !important;
        background: transparent !important;
        border-bottom: 2px solid #4f8cff !important;
    }
    section[data-testid="stSidebar"] {
        background: #0b0e14;
        border-right: 1px solid #1c222c;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #262d38;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "sad":      "😢",
    "surprise": "😲",
    "neutral":  "😐",
    "unknown":  "❓",
}

EMOTION_COLORS = {
    "angry":    "#e5484d",
    "disgust":  "#8b5cf6",
    "fear":     "#a855f7",
    "happy":    "#2ea043",
    "sad":      "#3b82f6",
    "surprise": "#f5a623",
    "neutral":  "#6b7280",
    "unknown":  "#4b5563",
}


# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def list_sessions(base="sessions"):
    p = Path(base)
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()], reverse=True)


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="page-header">
    <div class="icon">🧠</div>
    <div class="title">Behavioral <span>Analysis</span></div>
</div>
<div class="page-subtitle">Post-session analytics · CV-based behavioral + emotion + motion feature extraction</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("### 📂 Sessions")

sessions = list_sessions()

if not sessions:
    st.sidebar.warning("No sessions found. Run analyzer.py first.")
    st.error("No session folders found in ./sessions/")
    st.info("Run `python analyzer.py`, record a session, then refresh this page.")
    st.stop()

selected_session = st.sidebar.selectbox(
    "Choose a session",
    options=sessions,
    index=0,
    label_visibility="collapsed",
)

SESSION_PATH = Path("sessions") / selected_session
st.sidebar.caption(f"`{SESSION_PATH}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Options")
show_disclaimer = st.sidebar.checkbox("Show ethical disclaimer", value=True)

if st.sidebar.button("🔄 Refresh session list", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# LOAD FILES
# ============================================================

report_path   = SESSION_PATH / "session_report.json"
features_path = SESSION_PATH / "video_features.csv"
events_path   = SESSION_PATH / "behavioral_events.csv"
summary_path  = SESSION_PATH / "session_summary.csv"
heatmap_path  = SESSION_PATH / "gaze_heatmap.png"

missing = [f.name for f in [report_path, features_path, events_path, summary_path]
           if not f.exists()]

if missing:
    st.error(f"Missing files in this session: {', '.join(missing)}")
    st.stop()

report     = load_json(report_path)
df         = load_csv(features_path)
events_df  = load_csv(events_path)
summary_df = load_csv(summary_path)

HAS_EMOTION   = any(c.startswith("emotion_") for c in df.columns)
HAS_MOTION    = all(c in df.columns for c in
                    ["landmark_velocity", "jitter_index", "optical_flow"])
HAS_HEATMAP   = heatmap_path.exists()
HAS_TRACKING  = "person_id" in df.columns


# ============================================================
# DISCLAIMER
# ============================================================

if show_disclaimer:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">⚠️</div>
        <div class="info-body">
            <strong>Research / Observational Use Only</strong>
            The metrics shown here (stress index, gaze aversion, blink bursts,
            emotion labels, motion analysis, etc.) are heuristic indicators,
            not measurements of truthfulness, guilt, or intent. Facial and
            nonverbal cues have weak and contested links to deception. Do not
            use as sole basis for any investigative or legal decision.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# KPI ROW 1
# ============================================================

section("📊 Session Overview")

c1, c2, c3, c4 = st.columns(4, gap="small")

with c1:
    st.metric("Duration", f"{report.get('duration_seconds', 0):.1f} s")

with c2:
    st.metric("Face Detection",
              f"{report.get('face_detection_pct', 0):.1f}%")

with c3:
    st.metric("Blinks",
              f"{report.get('blink_count', 0)}",
              f"{report.get('blink_rate_per_min', 0):.1f}/min")

with c4:
    st.metric("Events Logged", f"{len(events_df)}")


# ============================================================
# KPI ROW 2
# ============================================================

c5, c6, c7, c8 = st.columns(4, gap="small")

with c5:
    st.metric("Stress Index",
              f"{report.get('final_stress_index', 0):.0f} / 100")

with c6:
    st.metric("Engagement",
              f"{report.get('final_engagement', 0):.0f}%")

with c7:
    # Dominant emotion
    if HAS_EMOTION and len(df) > 0 and "emotion_dominant" in df.columns:
        dom_counts = df["emotion_dominant"].value_counts()
        if len(dom_counts) > 0:
            dom_emotion = dom_counts.idxmax()
            dom_pct = 100.0 * dom_counts.max() / dom_counts.sum()
        else:
            dom_emotion = "unknown"
            dom_pct = 0.0
        emoji = EMOTION_EMOJI.get(dom_emotion, "❓")
        st.metric("Dominant Emotion",
                  f"{emoji} {dom_emotion.capitalize()}",
                  f"{dom_pct:.0f}% of frames")
    else:
        st.metric("Dominant Emotion", "—", "no emotion data")

with c8:
    st.metric("Gaze Aversions",
              f"{report.get('gaze_aversion_events', 0)}")


# ============================================================
# BASELINE
# ============================================================

baseline = report.get("baseline", {})
if baseline:
    with st.expander("🎯 Subject Baseline (calibrated in first 10s)"):
        bcols = st.columns(4)
        bcols[0].metric("EAR baseline", f"{baseline.get('ear_mean', 0):.3f}")
        bcols[1].metric("Yaw baseline", f"{baseline.get('yaw_mean', 0):.2f}°")
        bcols[2].metric("Pitch baseline", f"{baseline.get('pitch_mean', 0):.2f}°")
        bcols[3].metric("Blink-rate baseline",
                        f"{baseline.get('blink_rate', 0):.1f}/min")


# ============================================================
# TABS
# ============================================================

st.markdown("---")

tab_labels = [
    "📈  Time Series",
    "🎯  Behavioral Scores",
]
if HAS_MOTION:
    tab_labels.append("🏃  Motion Analysis")
if HAS_EMOTION:
    tab_labels.append("😊  Emotions")
if HAS_HEATMAP:
    tab_labels.append("🔥  Gaze Heatmap")
tab_labels += [
    "🚨  Events",
    "📄  Raw Data",
]

tabs = st.tabs(tab_labels)


# ============================================================
# TAB 0 — TIME SERIES
# ============================================================

with tabs[0]:

    section("Facial & Head Dynamics")

    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in ["float64", "int64"] and c != "elapsed_seconds"
        and not c.startswith("emotion_")
    ]

    default_signals = [
        c for c in ["ear", "mar", "brow", "blink_rate",
                    "yaw", "pitch", "roll", "hme_yaw", "hme_pitch"]
        if c in numeric_cols
    ]

    selected = st.multiselect(
        "Signals to plot",
        options=numeric_cols,
        default=default_signals
    )

    if selected:
        fig = go.Figure()
        for col in selected:
            fig.add_trace(go.Scatter(
                x=df["elapsed_seconds"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=2),
            ))
        fig.update_layout(
            height=440,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Elapsed (s)",
            yaxis_title="Value",
            legend=dict(orientation="h", y=-0.25),
            hovermode="x unified",
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    section("Gaze State Distribution")

    if "gaze_horizontal" in df.columns:
        gaze_counts = df["gaze_horizontal"].value_counts().reset_index()
        gaze_counts.columns = ["Gaze", "Count"]

        color_map = {
            "CENTER":  "#2ea043",
            "LEFT":    "#f5a623",
            "RIGHT":   "#e5484d",
            "UNKNOWN": "#6b7280",
        }
        fig2 = px.bar(
            gaze_counts,
            x="Gaze",
            y="Count",
            color="Gaze",
            color_discrete_map=color_map,
            text="Count",
        )
        fig2.update_layout(
            height=320,
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 1 — BEHAVIORAL SCORES
# ============================================================

with tabs[1]:

    section("Composite Behavioral Scores")

    if "stress_index" in df.columns and "engagement" in df.columns:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["elapsed_seconds"],
            y=df["stress_index"],
            name="Stress Index",
            line=dict(color="#e5484d", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(229,72,77,0.12)",
        ))

        fig.add_trace(go.Scatter(
            x=df["elapsed_seconds"],
            y=df["engagement"],
            name="Engagement",
            line=dict(color="#2ea043", width=2.5),
            yaxis="y2",
        ))

        fig.update_layout(
            height=420,
            xaxis_title="Elapsed (s)",
            yaxis=dict(title="Stress Index (0-100)", range=[0, 105]),
            yaxis2=dict(
                title="Engagement (%)",
                overlaying="y",
                side="right",
                range=[0, 105],
            ),
            legend=dict(orientation="h", y=-0.22),
            hovermode="x unified",
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    section("Anomaly Z-Scores (self-referential)")

    z_cols = [c for c in ["ear_z", "yaw_z"] if c in df.columns]

    if z_cols:
        fig_z = go.Figure()
        for col in z_cols:
            fig_z.add_trace(go.Scatter(
                x=df["elapsed_seconds"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=2),
            ))
        fig_z.add_hline(y=2,  line_dash="dash", line_color="#f5a623",
                        annotation_text="+2σ", annotation_font_color="#f5a623")
        fig_z.add_hline(y=-2, line_dash="dash", line_color="#f5a623",
                        annotation_text="-2σ", annotation_font_color="#f5a623")
        fig_z.update_layout(
            height=360,
            xaxis_title="Elapsed (s)",
            yaxis_title="Z-score",
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_z, use_container_width=True)
        st.caption(
            "Values beyond ±2σ indicate unusual behavior relative to this "
            "subject's own baseline."
        )


# ============================================================
# TAB — MOTION ANALYSIS (conditional)
# ============================================================

tab_idx = 2

if HAS_MOTION:
    with tabs[tab_idx]:

        section("Frame-to-Frame Motion")

        fig_m = go.Figure()

        fig_m.add_trace(go.Scatter(
            x=df["elapsed_seconds"], y=df["landmark_velocity"],
            name="Landmark Velocity", line=dict(color="#4f8cff", width=2),
        ))
        fig_m.add_trace(go.Scatter(
            x=df["elapsed_seconds"], y=df["jitter_index"],
            name="Jitter Index", line=dict(color="#f5a623", width=2),
        ))
        fig_m.add_trace(go.Scatter(
            x=df["elapsed_seconds"], y=df["optical_flow"],
            name="Optical Flow", line=dict(color="#8b5cf6", width=2),
            yaxis="y2",
        ))

        fig_m.update_layout(
            height=440,
            xaxis_title="Elapsed (s)",
            yaxis=dict(title="Velocity / Jitter", range=[0, None]),
            yaxis2=dict(
                title="Optical Flow",
                overlaying="y",
                side="right",
            ),
            legend=dict(orientation="h", y=-0.22),
            hovermode="x unified",
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_m, use_container_width=True)

        section("Motion Statistics")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Avg Landmark Velocity",
                      f"{df['landmark_velocity'].mean():.4f}",
                      help="Mean normalized landmark displacement per frame")
        with c2:
            st.metric("Avg Jitter Index",
                      f"{df['jitter_index'].mean():.4f}",
                      help="High-frequency motion energy (tremor / restlessness)")
        with c3:
            st.metric("Avg Optical Flow",
                      f"{df['optical_flow'].mean():.4f}",
                      help="Dense flow magnitude in face region")

        st.caption(
            "High jitter + high velocity together usually indicate restlessness. "
            "High velocity with low jitter = smooth head movement. "
            "Low values across the board = still / calm."
        )

    tab_idx += 1


# ============================================================
# TAB — EMOTIONS (conditional)
# ============================================================

if HAS_EMOTION:
    with tabs[tab_idx]:

        section("Emotion Timeline")

        if "emotion_dominant" in df.columns and "elapsed_seconds" in df.columns:
            fig_tl = go.Figure()
            for emo in EMOTION_NAMES:
                mask = df["emotion_dominant"] == emo
                if not mask.any():
                    continue
                fig_tl.add_trace(go.Scatter(
                    x=df.loc[mask, "elapsed_seconds"],
                    y=[emo] * mask.sum(),
                    mode="markers",
                    name=emo,
                    marker=dict(
                        size=10,
                        color=EMOTION_COLORS.get(emo, "#888"),
                        symbol="square",
                    ),
                    hovertemplate=(
                        "%{x:.2f}s<br>"
                        f"<b>{emo}</b><br>"
                        "confidence=%{customdata:.2f}"
                        "<extra></extra>"
                    ),
                    customdata=df.loc[mask, "emotion_confidence"]
                    if "emotion_confidence" in df.columns
                    else [0] * mask.sum(),
                ))
            fig_tl.update_layout(
                height=300,
                xaxis_title="Elapsed (s)",
                yaxis_title="",
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

        section("Emotion Distribution")

        if "emotion_dominant" in df.columns:
            counts = df["emotion_dominant"].value_counts().reset_index()
            counts.columns = ["Emotion", "Frames"]
            counts["Pct"] = (100.0 * counts["Frames"]
                             / counts["Frames"].sum()).round(1)

            fig_ed = px.bar(
                counts,
                x="Emotion",
                y="Frames",
                color="Emotion",
                color_discrete_map=EMOTION_COLORS,
                text="Pct",
            )
            fig_ed.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_ed.update_layout(
                height=360,
                showlegend=False,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_ed, use_container_width=True)

        section("Average Emotion Scores")

        avg_cols = [c for c in df.columns if c.startswith("emotion_")
                    and c not in ("emotion_dominant", "emotion_confidence")]

        if avg_cols:
            rows = []
            for c in avg_cols:
                name = c.replace("emotion_", "")
                rows.append({
                    "Emotion": name,
                    "Average": float(df[c].mean()),
                })
            avg_df = pd.DataFrame(rows).sort_values("Average", ascending=False)

            fig_avg = go.Figure(go.Bar(
                x=avg_df["Average"],
                y=avg_df["Emotion"],
                orientation="h",
                marker=dict(
                    color=[EMOTION_COLORS.get(e, "#888")
                           for e in avg_df["Emotion"]]
                ),
                text=avg_df["Average"].round(3),
                textposition="outside",
            ))
            fig_avg.update_layout(
                height=340,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                xaxis_title="Average score (0-1)",
                yaxis_title="",
                margin=dict(l=20, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_avg, use_container_width=True)

        if "emotion_confidence" in df.columns:
            section("Emotion Confidence Over Time")

            fig_conf = go.Figure(go.Scatter(
                x=df["elapsed_seconds"],
                y=df["emotion_confidence"],
                mode="lines",
                line=dict(color="#4f8cff", width=2),
                fill="tozeroy",
                fillcolor="rgba(79,140,255,0.12)",
            ))
            fig_conf.update_layout(
                height=280,
                xaxis_title="Elapsed (s)",
                yaxis_title="Confidence (0-1)",
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        st.caption(
            "Emotion labels are derived from MediaPipe blendshape coefficients "
            "using weighted heuristic rules. They are approximate and should "
            "not be treated as ground truth psychological state."
        )

    tab_idx += 1


# ============================================================
# TAB — GAZE HEATMAP (conditional)
# ============================================================

if HAS_HEATMAP:
    with tabs[tab_idx]:

        section("Gaze Attention Heatmap")

        st.caption(
            "Accumulated gaze points across the session. Warmer colors = "
            "where the subject focused more often. This is a self-referential "
            "visual — it does not indicate truthfulness."
        )

        col_a, col_b = st.columns([3, 1], gap="large")

        with col_a:
            st.image(
                str(heatmap_path),
                caption="Gaze heatmap (640×480)",
                use_container_width=True,
            )

        with col_b:
            # Summary stats derived from raw heatmap
            try:
                npy_path = SESSION_PATH / "gaze_heatmap.npy"
                if npy_path.exists():
                    raw = np.load(str(npy_path))
                    coverage = float((raw > 0.1).mean()) * 100
                    peak_val = float(raw.max())
                    center_of_mass = np.unravel_index(np.argmax(raw), raw.shape)
                    ch, cw = raw.shape
                    cx_pct = 100.0 * center_of_mass[1] / cw
                    cy_pct = 100.0 * center_of_mass[0] / ch

                    st.metric("Coverage", f"{coverage:.1f}%",
                              help="% of heatmap area with activity")
                    st.metric("Peak Intensity", f"{peak_val:.2f}")
                    st.metric("Focus Center X", f"{cx_pct:.0f}%",
                              help="Horizontal position of max focus")
                    st.metric("Focus Center Y", f"{cy_pct:.0f}%",
                              help="Vertical position of max focus")
                else:
                    st.info("Raw heatmap data (.npy) not found.")
            except Exception as e:
                st.warning(f"Could not compute heatmap stats: {e}")

            st.download_button(
                "⬇️ Download heatmap PNG",
                data=open(heatmap_path, "rb").read(),
                file_name=f"gaze_heatmap_{selected_session}.png",
                mime="image/png",
                use_container_width=True,
            )

    tab_idx += 1


# ============================================================
# EVENTS TAB
# ============================================================

with tabs[tab_idx]:

    section("Behavioral Events")

    if len(events_df) == 0:
        st.info("No discrete behavioral events were logged in this session.")
    else:
        ev_counts = events_df["event"].value_counts().reset_index()
        ev_counts.columns = ["Event", "Count"]

        fig_ev = px.bar(
            ev_counts,
            x="Count",
            y="Event",
            orientation="h",
            text="Count",
            color="Event",
        )
        fig_ev.update_layout(
            height=max(240, 60 * len(ev_counts)),
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_ev, use_container_width=True)

        section("Event Timeline")

        if "elapsed_seconds" in events_df.columns:
            fig_tl = px.scatter(
                events_df,
                x="elapsed_seconds",
                y="event",
                color="event",
                hover_data=["detail", "timestamp"],
            )
            fig_tl.update_layout(
                height=340,
                xaxis_title="Elapsed (s)",
                yaxis_title="",
                showlegend=False,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

        section("Event Details")
        st.dataframe(events_df, use_container_width=True, hide_index=True)

tab_idx += 1


# ============================================================
# RAW DATA TAB
# ============================================================

with tabs[tab_idx]:

    section("Raw Feature Data")
    st.dataframe(df, use_container_width=True, height=500, hide_index=True)

    col_dl1, col_dl2, col_dl3 = st.columns(3, gap="small")

    with col_dl1:
        st.download_button(
            "⬇️ Features CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="video_features_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        if len(events_df) > 0:
            st.download_button(
                "⬇️ Events CSV",
                data=events_df.to_csv(index=False).encode("utf-8"),
                file_name="behavioral_events_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col_dl3:
        st.download_button(
            "⬇️ JSON Report",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="session_report_export.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# INTERPRETATION GUIDE
# ============================================================

st.markdown("---")
with st.expander("📖 How to interpret these metrics"):

    st.markdown("""
    **Stress Index (0-100)** — Composite heuristic from blink rate, gaze
    aversion, head motion energy, and blink bursts. Not a measurement of
    deception. High stress can mean anything: discomfort, caffeine, dry eyes,
    room temperature, cognitive load, or simply being watched.

    ---

    **Engagement (%)** — Combination of face presence and centered gaze.

    ---

    **Motion Analysis** — Three complementary signals:
    - *Landmark velocity*: average per-frame displacement of face landmarks,
      normalized by face width. Higher = more facial movement.
    - *Jitter index*: standard deviation of recent velocities. Higher =
      higher-frequency motion (restlessness, tremor) vs. smooth motion.
    - *Optical flow*: dense pixel-level motion magnitude in the face region.
      Captures motion that landmark tracking may miss.

    ---

    **Emotions** — Derived from 52 MediaPipe blendshape coefficients using
    weighted heuristic rules. Approximate; treat as one signal among many.
    Common confusions: *neutral* ↔ *sad*, *surprise* ↔ *fear*.

    ---

    **Gaze Heatmap** — Accumulation of gaze points over the session. Warmer
    colors = more focus. Useful for showing where attention was directed,
    but does not indicate intent or truthfulness.

    ---

    **Person ID** — ByteTrack-assigned ID from YOLO. Persists across frames
    while a person remains in view.

    ---

    **Z-scores** — How unusual a value is *for this person* relative to their
    own first 10 seconds. Controls for individual differences.

    ---

    **Legal / Ethical Note** — If used in any investigative context, this
    tool's outputs and the model behind them may be discoverable by the
    defense. Consult counsel before operational use.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    "Behavioral Analysis System · Research use only"
)