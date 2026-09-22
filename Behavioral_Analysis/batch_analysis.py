# ============================================================
# BATCH ANALYSIS — IMAGE PROCESSOR
# Runs the same feature extraction as analyzer.py, but on a
# single uploaded image. Produces a session folder compatible
# with the existing dashboard and report generator.
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import math
import json
import csv
import os
import urllib.request
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "face_landmarker.task"

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_COLORS_BGR = {
    "angry":    (77, 72, 229),
    "disgust":  (139, 92, 246),
    "fear":     (168, 168, 247),
    "happy":    (67, 160, 46),
    "sad":      (246, 130, 59),
    "surprise": (35, 166, 245),
    "neutral":  (110, 114, 107),
    "unknown":  (100, 100, 100),
}


# ============================================================
# AUTO-DOWNLOAD MODEL
# ============================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        raise


# ============================================================
# MEDIAPIPE SETUP (IMAGE MODE)
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# For images we use IMAGE mode (one-shot, no video timestamps)
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,
)

detector = FaceLandmarker.create_from_options(options)


# ============================================================
# LANDMARKS
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
MOUTH = [61, 291, 13, 14, 78, 308]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [300, 293, 334, 296, 336]


# ============================================================
# FEATURE FUNCTIONS (same math as analyzer.py)
# ============================================================

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    v1 = distance(p2, p6)
    v2 = distance(p3, p5)
    h = distance(p1, p4)
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def calculate_gaze(landmarks):
    li_x = np.mean([landmarks[i][0] for i in LEFT_IRIS])
    li_y = np.mean([landmarks[i][1] for i in LEFT_IRIS])
    ri_x = np.mean([landmarks[i][0] for i in RIGHT_IRIS])
    ri_y = np.mean([landmarks[i][1] for i in RIGHT_IRIS])

    lr = (li_x - landmarks[362][0]) / (landmarks[263][0] - landmarks[362][0] + 1e-6)
    rr = (ri_x - landmarks[33][0]) / (landmarks[133][0] - landmarks[33][0] + 1e-6)
    hr = (lr + rr) / 2
    vr = (li_y + ri_y) / 2

    if hr < 0.40:
        h = "LEFT"
    elif hr > 0.60:
        h = "RIGHT"
    else:
        h = "CENTER"

    return h, vr, hr


def calculate_mar(landmarks):
    left, right, top, bottom, _, _ = [landmarks[i] for i in MOUTH]
    horiz = distance(left, right) + 1e-6
    vert = distance(top, bottom)
    return vert / horiz


def calculate_brow_raise(landmarks):
    lb = np.mean([landmarks[i].y for i in LEFT_BROW])
    rb = np.mean([landmarks[i].y for i in RIGHT_BROW])
    le = landmarks[362].y
    re = landmarks[263].y
    return ((le - lb) + (re - rb)) / 2.0


def calculate_head_pose(points, frame_w, frame_h):
    image_points = np.array([
        points[1], points[152], points[33],
        points[263], points[61], points[291]
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    distortion = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rvec, tvec = cv2.solvePnPRefineLM(
        model_points, image_points, camera_matrix, distortion, rvec, tvec
    )
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy < 1e-6:
        return 0.0, 0.0, 0.0

    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    if pitch > 90: pitch -= 180
    elif pitch < -90: pitch += 180
    if yaw > 90: yaw -= 180
    elif yaw < -90: yaw += 180
    if roll > 90: roll -= 180
    elif roll < -90: roll += 180

    return pitch, yaw, roll


def classify_emotion(blendshapes):
    if not blendshapes:
        return "unknown", 0.0, {e: 0.0 for e in EMOTION_NAMES}

    bs = {c.category_name: c.score for c in blendshapes}
    g = lambda n: bs.get(n, 0.0)

    happy = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2
    happy += 0.3 * (g("cheekSquintLeft") + g("cheekSquintRight")) / 2

    sad = (g("mouthFrownLeft") + g("mouthFrownRight")) / 2
    sad += 0.5 * g("browInnerUp")
    sad += 0.3 * (g("mouthLowerDownLeft") + g("mouthLowerDownRight")) / 2

    angry = (g("browDownLeft") + g("browDownRight")) / 2
    angry += 0.4 * (g("mouthPressLeft") + g("mouthPressRight")) / 2
    angry += 0.3 * (g("noseSneerLeft") + g("noseSneerRight")) / 2

    surprise = g("jawOpen")
    surprise += 0.5 * (g("eyeWideLeft") + g("eyeWideRight")) / 2
    surprise += 0.4 * (g("browOuterUpLeft") + g("browOuterUpRight")) / 2
    surprise += 0.3 * g("browInnerUp")

    fear = (g("eyeWideLeft") + g("eyeWideRight")) / 2
    fear += 0.6 * g("browInnerUp")
    fear += 0.4 * (g("browOuterUpLeft") + g("browOuterUpRight")) / 2
    fear += 0.2 * g("jawOpen")

    disgust = (g("noseSneerLeft") + g("noseSneerRight")) / 2
    disgust += (g("mouthUpperUpLeft") + g("mouthUpperUpRight")) / 2
    disgust += 0.3 * (g("browDownLeft") + g("browDownRight")) / 2

    raw = {"happy": happy, "sad": sad, "angry": angry,
           "surprise": surprise, "fear": fear, "disgust": disgust}

    total = sum(raw.values())

    if total < 0.15:
        scores = {e: 0.0 for e in EMOTION_NAMES}
        scores["neutral"] = 1.0
        return "neutral", 1.0, scores

    scores = {e: raw[e] / total for e in raw}
    neutral_weight = max(0.0, 1.0 - total)
    active = 1.0 - neutral_weight
    for e in scores:
        scores[e] *= active
    scores["neutral"] = neutral_weight

    for e in EMOTION_NAMES:
        scores.setdefault(e, 0.0)

    dominant = max(scores, key=scores.get)
    return (dominant,
            round(scores[dominant], 3),
            {k: round(v, 3) for k, v in scores.items()})


# ============================================================
# ANNOTATED IMAGE
# ============================================================

def draw_annotations(frame, points, emotion, emotion_conf, gaze,
                     yaw, pitch, roll, ear, mar, brow):
    """Draw landmarks, head-pose axes, and a result panel."""
    h, w = frame.shape[:2]

    # Face mesh points
    for p in points[::3]:
        cv2.circle(frame, p, 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    # Top-left info panel
    panel_w, panel_h = 320, 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h),
                  (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    emo_color = EMOTION_COLORS_BGR.get(emotion, (255, 180, 100))
    y = 45
    cv2.putText(frame, f"EMOTION: {emotion.upper()}", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, emo_color, 2, cv2.LINE_AA)
    y += 26
    cv2.putText(frame, f"confidence: {emotion_conf:.2f}", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)

    y += 30
    lines = [
        ("EAR",  f"{ear:.3f}"),
        ("MAR",  f"{mar:.3f}"),
        ("Brow", f"{brow:.3f}"),
        ("Gaze", gaze),
        ("Yaw",  f"{yaw:.1f}°"),
        ("Pitch", f"{pitch:.1f}°"),
        ("Roll", f"{roll:.1f}°"),
    ]
    for label, val in lines:
        cv2.putText(frame, f"{label}:", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 160, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, val, (130, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        y += 26

    # Head-pose axis at face center
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    nose_tip = points[1]

    axis_len = 60
    yaw_r = math.radians(yaw)
    pitch_r = math.radians(pitch)
    x_end = int(nose_tip[0] + axis_len * math.sin(yaw_r))
    y_end = int(nose_tip[1] - axis_len * math.sin(pitch_r))
    cv2.arrowedLine(frame, nose_tip, (x_end, y_end),
                    (0, 200, 255), 2, tipLength=0.25, line_type=cv2.LINE_AA)

    return frame


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def analyze_image(image_path, output_dir=None, verbose=True):
    """
    Analyze a single image. Produces a session folder with:
      - session_report.json
      - video_features.csv       (single row)
      - session_summary.csv      (single row)
      - annotated.png            (image with overlay)
      - gaze_heatmap.png         (single-blob heatmap)

    Returns the session folder path.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not decode image: {image_path}")

    # Prepare session folder
    if output_dir is None:
        session_id = "img_" + datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_dir = Path("sessions") / session_id
    else:
        session_dir = Path(output_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Session folder: {session_dir}")

    h, w = frame.shape[:2]

    # MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if not result.face_landmarks:
        # Save a "no face" report
        report = {
            "session_id":     session_dir.name,
            "session_type":   "image",
            "session_start":  datetime.now().isoformat(),
            "source_file":    str(image_path),
            "face_detected":  False,
            "error":          "No face detected in image",
        }
        with open(session_dir / "session_report.json", "w") as f:
            json.dump(report, f, indent=2)

        _write_empty_csvs(session_dir)

        # Clean, exit-code-0 message (not an error)
        print("[OK] No face detected in the uploaded image.")
        print(f"[INFO] Session folder: {session_dir}")
        return session_dir

    # ---- Extract features ----
    landmarks = result.face_landmarks[0]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    left_ear = calculate_ear(points, LEFT_EYE)
    right_ear = calculate_ear(points, RIGHT_EYE)
    ear = (left_ear + right_ear) / 2

    mar = calculate_mar(points)
    brow = calculate_brow_raise(landmarks)

    gaze, gaze_v, gaze_h = calculate_gaze(points)
    pitch, yaw, roll = calculate_head_pose(points, w, h)

    emotion, emotion_conf, emotion_scores = "unknown", 0.0, {e: 0.0 for e in EMOTION_NAMES}
    bs_count = 0
    if result.face_blendshapes and len(result.face_blendshapes) > 0:
        bs_count = len(result.face_blendshapes[0])
        emotion, emotion_conf, emotion_scores = classify_emotion(result.face_blendshapes[0])

    # ---- Annotated image ----
    annotated = draw_annotations(
        frame.copy(), points, emotion, emotion_conf, gaze,
        yaw, pitch, roll, ear, mar, brow
    )
    cv2.imwrite(str(session_dir / "annotated.png"), annotated)

    # ---- Gaze heatmap (single-blob) ----
    _make_single_blob_heatmap(
        session_dir / "gaze_heatmap.png",
        points, gaze_h, gaze_v, w, h
    )

    # ---- Feature row (identical schema to live CSV) ----
    ts = datetime.now().isoformat()
    feature_row = {
        "timestamp": ts,
        "elapsed_seconds": 0.0,
        "face_detected": 1,
        "person_id": -1,
        "ear": round(ear, 4),
        "mar": round(mar, 4),
        "brow": round(brow, 4),
        "blink_count": 0,
        "blink_rate": 0.0,
        "yaw": round(yaw, 2),
        "pitch": round(pitch, 2),
        "roll": round(roll, 2),
        "gaze_horizontal": gaze,
        "landmark_velocity": 0.0,
        "jitter_index": 0.0,
        "optical_flow": 0.0,
        "stress_index": 0.0,
        "engagement": 0.0,
        "hme_yaw": 0.0,
        "hme_pitch": 0.0,
        "aversion_ratio": 0.0,
        "ear_z": 0.0,
        "yaw_z": 0.0,
        "emotion_dominant": emotion,
        "emotion_confidence": round(emotion_conf, 3),
    }
    for e in EMOTION_NAMES:
        feature_row[f"emotion_{e}"] = emotion_scores.get(e, 0.0)

    # Write feature CSV
    with open(session_dir / "video_features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_row.keys()))
        writer.writeheader()
        writer.writerow(feature_row)

    # Session summary
    with open(session_dir / "session_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_start", "session_type", "source_file",
            "duration_seconds", "total_frames", "face_detection_percentage",
            "average_ear", "average_mar", "average_yaw", "average_pitch",
            "dominant_emotion_overall",
        ])
        writer.writerow([
            ts, "image", str(image_path),
            0.0, 1, 100.0,
            round(ear, 4), round(mar, 4),
            round(yaw, 2), round(pitch, 2),
            emotion,
        ])

    # Behavioral events CSV (empty but valid)
    with open(session_dir / "behavioral_events.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["timestamp", "elapsed_seconds", "event", "detail"])

    # JSON report
    report = {
        "session_id":              session_dir.name,
        "session_type":            "image",
        "session_start":           ts,
        "source_file":             str(image_path),
        "face_detected":           True,
        "image_width":             w,
        "image_height":            h,
        "duration_seconds":        0.0,
        "total_frames":            1,
        "face_detection_pct":      100.0,
        "ear":                     round(ear, 4),
        "mar":                     round(mar, 4),
        "brow":                    round(brow, 4),
        "yaw":                     round(yaw, 2),
        "pitch":                   round(pitch, 2),
        "roll":                    round(roll, 2),
        "gaze_horizontal":         gaze,
        "emotion_dominant":        emotion,
        "emotion_confidence":      round(emotion_conf, 3),
        "emotion_scores":          emotion_scores,
        "blendshape_count":        bs_count,
    }
    with open(session_dir / "session_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print("✓ Analysis complete")
        print(f"  Emotion:   {emotion} ({emotion_conf:.2f})")
        print(f"  EAR:       {ear:.3f}")
        print(f"  MAR:       {mar:.3f}")
        print(f"  Gaze:      {gaze}")
        print(f"  Head pose: yaw={yaw:.1f}° pitch={pitch:.1f}° roll={roll:.1f}°")
        print(f"  Files:     {session_dir}")

    return session_dir


# ============================================================
# HELPERS
# ============================================================

def _write_empty_csvs(session_dir):
    """Write minimal valid CSVs when no face was detected."""
    cols = ["timestamp", "elapsed_seconds", "face_detected", "ear", "mar",
            "brow", "blink_count", "blink_rate", "yaw", "pitch", "roll",
            "gaze_horizontal", "stress_index", "engagement", "emotion_dominant",
            "emotion_confidence"]
    with open(session_dir / "video_features.csv", "w", newline="") as f:
        csv.writer(f).writerow(cols)

    with open(session_dir / "session_summary.csv", "w", newline="") as f:
        csv.writer(f).writerow([
            "session_start", "session_type", "duration_seconds",
            "total_frames", "face_detection_percentage"
        ])

    with open(session_dir / "behavioral_events.csv", "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "elapsed_seconds", "event", "detail"])


def _make_single_blob_heatmap(out_path, points, gaze_h, gaze_v, w, h,
                              heatmap_size=(240, 320)):
    """
    For single-image mode, render a heatmap with a single bright blob at the
    estimated gaze point. Keeps the dashboard's heatmap tab functional.
    """
    H, W = heatmap_size
    heatmap = np.zeros((H, W), dtype=np.float32)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    face_cx = (min(xs) + max(xs)) / 2
    face_cy = (min(ys) + max(ys)) / 2
    face_w = max(1, max(xs) - min(xs))
    face_h = max(1, max(ys) - min(ys))

    gx = face_cx + (gaze_h - 0.5) * face_w
    gy = face_cy + (gaze_v - 0.5) * face_h

    hx = int(gx / w * W)
    hy = int(gy / h * H)
    hx = max(0, min(W - 1, hx))
    hy = max(0, min(H - 1, hy))

    radius = 30
    x0 = max(0, hx - radius); x1 = min(W, hx + radius + 1)
    y0 = max(0, hy - radius); y1 = min(H, hy + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - hx) ** 2 + (yy - hy) ** 2
    heatmap[y0:y1, x0:x1] = np.exp(-dist2 / (2 * (radius / 2) ** 2))

    norm = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored = cv2.resize(colored, (640, 480), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_path), colored)


# ============================================================
# CLI ENTRY
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python batch_analysis.py <path_to_image>")
        sys.exit(1)

    try:
        analyze_image(sys.argv[1])
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)