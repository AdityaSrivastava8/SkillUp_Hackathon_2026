# ============================================================
# BATCH VIDEO ANALYSIS
# Runs the same pipeline as analyzer.py, but on a video file.
# Samples at 10 fps. Produces a full session folder compatible
# with the existing dashboard and report generator.
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import json
import urllib.request
import os
from datetime import datetime
from collections import deque
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "face_landmarker.task"

EAR_THRESHOLD = 0.22
HISTORY_LENGTH = 150
BASELINE_SECONDS = 10
YOLO_EVERY_N_FRAMES = 8
TARGET_SAMPLE_FPS = 10

HEATMAP_SIZE = (240, 320)
HEATMAP_DECAY = 0.999

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# ============================================================
# AUTO-DOWNLOAD MODEL
# ============================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
)

detector = FaceLandmarker.create_from_options(options)

YOLO_MODEL = YOLO("yolov8n.pt")

WATCHED_CLASSES = {
    0:  "person",
    67: "cell phone",
    39: "bottle",
    41: "cup",
    73: "book",
    76: "scissors",
    43: "knife",
}


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
# FEATURE FUNCTIONS
# ============================================================

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    v1 = distance(p2, p6)
    v2 = distance(p3, p5)
    h = distance(p1, p4)
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def calculate_gaze(landmarks):
    li_x = np.mean([landmarks[i][0] for i in LEFT_IRIS])
    li_y = np.mean([landmarks[i][1] for i in LEFT_IRIS])
    ri_x = np.mean([landmarks[i][0] for i in RIGHT_IRIS])
    ri_y = np.mean([landmarks[i][1] for i in RIGHT_IRIS])

    # Horizontal: iris position within eye width (0..1)
    lr = (li_x - landmarks[362][0]) / (landmarks[263][0] - landmarks[362][0] + 1e-6)
    rr = (ri_x - landmarks[33][0]) / (landmarks[133][0] - landmarks[33][0] + 1e-6)
    hr = (lr + rr) / 2

    # Vertical: iris position within eye height (0..1)
    left_top = landmarks[385][1]
    left_bot = landmarks[380][1]
    right_top = landmarks[160][1]
    right_bot = landmarks[144][1]

    lv = (li_y - left_top) / (left_bot - left_top + 1e-6)
    rv = (ri_y - right_top) / (right_bot - right_top + 1e-6)
    vr = (lv + rv) / 2

    hr = max(0.0, min(1.0, hr))
    vr = max(0.0, min(1.0, vr))

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

    happy = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2 + 0.3 * (g("cheekSquintLeft") + g("cheekSquintRight")) / 2
    sad = (g("mouthFrownLeft") + g("mouthFrownRight")) / 2 + 0.5 * g("browInnerUp") + 0.3 * (g("mouthLowerDownLeft") + g("mouthLowerDownRight")) / 2
    angry = (g("browDownLeft") + g("browDownRight")) / 2 + 0.4 * (g("mouthPressLeft") + g("mouthPressRight")) / 2 + 0.3 * (g("noseSneerLeft") + g("noseSneerRight")) / 2
    surprise = g("jawOpen") + 0.5 * (g("eyeWideLeft") + g("eyeWideRight")) / 2 + 0.4 * (g("browOuterUpLeft") + g("browOuterUpRight")) / 2 + 0.3 * g("browInnerUp")
    fear = (g("eyeWideLeft") + g("eyeWideRight")) / 2 + 0.6 * g("browInnerUp") + 0.4 * (g("browOuterUpLeft") + g("browOuterUpRight")) / 2 + 0.2 * g("jawOpen")
    disgust = (g("noseSneerLeft") + g("noseSneerRight")) / 2 + (g("mouthUpperUpLeft") + g("mouthUpperUpRight")) / 2 + 0.3 * (g("browDownLeft") + g("browDownRight")) / 2

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
    return dominant, round(scores[dominant], 3), {k: round(v, 3) for k, v in scores.items()}


def motion_energy(history):
    return float(np.std(list(history))) if len(history) >= 5 else 0.0


def detect_blink_burst(blink_times, window=3.0, threshold=3):
    if len(blink_times) < threshold:
        return False
    recent = [t for t in blink_times if blink_times[-1] - t <= window]
    return len(recent) >= threshold


def zscore(value, history, min_samples=30):
    if len(history) < min_samples:
        return 0.0
    arr = np.array(history)
    mu, sigma = arr.mean(), arr.std() + 1e-6
    return float((value - mu) / sigma)


def compute_stress_index(br, av, hy, hp, burst):
    s = 0.0
    if br > 20:
        s += min((br - 20) * 1.2, 25)
    s += min(av * 40, 25)
    s += min(hy * 0.8, 15)
    s += min(hp * 0.8, 15)
    if burst:
        s += 20
    return min(s, 100.0)


def compute_engagement(face_pct, gaze_center):
    return 0.5 * face_pct + 0.5 * (gaze_center * 100)


def landmark_velocity(prev, curr):
    if prev is None or curr is None or len(prev) != len(curr):
        return 0.0
    xs = [p[0] for p in curr]
    face_w = max(1, max(xs) - min(xs))
    disps = [distance(prev[i], curr[i]) for i in range(0, len(curr), 5)]
    return float(np.mean(disps) / face_w) if disps else 0.0


def jitter_index(vel_hist, recent_n=15):
    if len(vel_hist) < recent_n:
        return 0.0
    recent = list(vel_hist)[-recent_n:]
    return float(np.std(recent))


def optical_flow_energy(prev_gray, curr_gray, bbox=None):
    if prev_gray is None or curr_gray is None:
        return 0.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(prev_gray.shape[1], x2)
        y2 = min(prev_gray.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        prev_crop = prev_gray[y1:y2, x1:x2]
        curr_crop = curr_gray[y1:y2, x1:x2]
    else:
        prev_crop = prev_gray
        curr_crop = curr_gray
    if prev_crop.size == 0 or curr_crop.size == 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_crop, curr_crop, None,
        pyr_scale=0.5, levels=2, winsize=13,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def run_tracking(frame):
    results = YOLO_MODEL.track(
        frame, persist=True, tracker="bytetrack.yaml",
        conf=0.4, verbose=False
    )
    dets = []
    if not results:
        return dets
    r = results[0]
    if r.boxes is None or r.boxes.id is None:
        return dets
    fh, fw = frame.shape[:2]
    area = fw * fh
    for box, cls_id, tid in zip(r.boxes, r.boxes.cls, r.boxes.id):
        cls_id = int(cls_id)
        if cls_id not in WATCHED_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if (x2 - x1) * (y2 - y1) > 0.7 * area:
            continue
        dets.append({
            "track_id": int(tid),
            "label": WATCHED_CLASSES[cls_id],
            "conf": float(box.conf[0]),
            "box": (x1, y1, x2, y2),
        })
    return dets


# ------------------------------------------------------------
# Gaze heatmap (FIXED)
# ------------------------------------------------------------

def gaze_point_in_frame(face_box, hr, vr, frame_w, frame_h):
    """
    Compute the gaze point in frame coordinates.
    Uses frame dimensions (not face dimensions) so gaze direction spreads
    visibly across the frame.
    """
    x1, y1, x2, y2 = face_box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    dx = (hr - 0.5) * frame_w * 1.5
    dy = (vr - 0.5) * frame_h * 1.5

    gx = cx + dx
    gy = cy + dy

    gx = max(0, min(frame_w - 1, gx))
    gy = max(0, min(frame_h - 1, gy))

    return int(gx), int(gy)


def accumulate_gaze(heatmap, x, y, frame_shape):
    H, W = heatmap.shape
    fh, fw = frame_shape[:2]
    hx = max(0, min(W - 1, int(x / fw * W)))
    hy = max(0, min(H - 1, int(y / fh * H)))
    radius = 14
    x0 = max(0, hx - radius); x1 = min(W, hx + radius + 1)
    y0 = max(0, hy - radius); y1 = min(H, hy + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - hx) ** 2 + (yy - hy) ** 2
    blob = np.exp(-dist2 / (2 * (radius / 2) ** 2))
    heatmap[y0:y1, x0:x1] += blob.astype(np.float32)


def render_heatmap(heatmap, size=(240, 320)):
    if heatmap.max() < 1e-6:
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)
    norm = np.clip(heatmap / heatmap.max() * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.resize(colored, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def log_event(event_log, event_type, detail, elapsed):
    event_log.append({
        "timestamp": datetime.now().isoformat(),
        "elapsed": round(elapsed, 2),
        "event": event_type,
        "detail": detail,
    })


# ============================================================
# MAIN FUNCTION
# ============================================================

def analyze_video(video_path, output_dir=None, verbose=True):
    """
    Analyze a video file. Produces a session folder with the same
    outputs as analyzer.py (live webcam mode).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    duration_est = total_frames / native_fps if native_fps > 0 else 0

    if verbose:
        print(f"Video: {video_path.name}")
        print(f"  Resolution: {frame_w}x{frame_h}  FPS: {native_fps:.1f}")
        print(f"  Duration: {duration_est:.1f}s  Frames: {total_frames}")
        print(f"  Sampling at {TARGET_SAMPLE_FPS} fps ...")

    # Session folder
    if output_dir is None:
        session_id = "vid_" + datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_dir = Path("sessions") / session_id
    else:
        session_dir = Path(output_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Session folder: {session_dir}")

    # CSV
    csv_path = session_dir / "video_features.csv"
    csv_handle = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_handle)

    EMOTION_COLUMNS = ["emotion_dominant", "emotion_confidence"]
    EMOTION_COLUMNS += [f"emotion_{e}" for e in EMOTION_NAMES]

    csv_writer.writerow([
        "timestamp", "elapsed_seconds", "face_detected", "person_id",
        "ear", "mar", "brow", "blink_count", "blink_rate",
        "yaw", "pitch", "roll", "gaze_horizontal",
        "landmark_velocity", "jitter_index", "optical_flow",
        "stress_index", "engagement", "hme_yaw", "hme_pitch",
        "aversion_ratio", "ear_z", "yaw_z",
    ] + EMOTION_COLUMNS)

    # State
    ear_history = deque(maxlen=HISTORY_LENGTH)
    yaw_history = deque(maxlen=HISTORY_LENGTH)
    pitch_history = deque(maxlen=HISTORY_LENGTH)
    roll_history = deque(maxlen=HISTORY_LENGTH)
    mar_history = deque(maxlen=HISTORY_LENGTH)
    brow_history = deque(maxlen=HISTORY_LENGTH)
    vel_history = deque(maxlen=HISTORY_LENGTH)
    flow_history = deque(maxlen=HISTORY_LENGTH)

    gaze_states = deque(maxlen=300)
    gaze_aversion_events = []
    gaze_aversion_start = None

    emotion_counts = {e: 0 for e in EMOTION_NAMES}
    emotion_scores_history = deque(maxlen=500)

    gaze_heatmap = np.zeros(HEATMAP_SIZE, dtype=np.float32)

    event_log = []
    baseline_ready = False
    baseline_stats = {}
    blink_times = []
    face_frames = 0
    blink_count = 0
    blink_active = False
    processed_frames = 0
    timestamp_ms = 0
    logged_objects = set()
    burst_flag = False

    prev_face_points = None
    prev_gray = None

    # Latest values
    ear = mar = brow = 0.0
    yaw = pitch = roll = 0.0
    gaze_horizontal = "UNKNOWN"
    gaze_h_ratio = 0.5
    gaze_v_ratio = 0.5
    landmark_vel = jitter = flow_energy = 0.0
    stress_index = engagement = hme_yaw = hme_pitch = 0.0
    aversion_ratio = ear_z = yaw_z = blink_rate = face_percentage = 0.0
    current_emotion = "neutral"
    current_emotion_conf = 0.0
    current_emotion_scores = {e: 0.0 for e in EMOTION_NAMES}

    # Best frame tracking (highest stress)
    best_stress = -1.0
    best_frame = None

    # Sampling: only process every Nth frame
    sample_every = max(1, int(round(native_fps / TARGET_SAMPLE_FPS)))

    start_time = time.time()
    frame_idx = 0
    elapsed = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        processed_frames += 1
        elapsed = processed_frames / TARGET_SAMPLE_FPS

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(elapsed * 1000)
        result = detector.detect_for_video(mp_img, timestamp_ms)

        face_detected = 0
        face_box = None
        person_id = -1

        if result.face_landmarks:
            face_detected = 1
            face_frames += 1

            landmarks = result.face_landmarks[0]
            points = [(int(lm.x * frame_w), int(lm.y * frame_h))
                      for lm in landmarks]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            face_box = (min(xs), min(ys), max(xs), max(ys))

            # EAR
            left_ear = calculate_ear(points, LEFT_EYE)
            right_ear = calculate_ear(points, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2
            ear_history.append(ear)

            # Blink
            if ear < EAR_THRESHOLD:
                if not blink_active:
                    blink_active = True
            else:
                if blink_active:
                    blink_count += 1
                    blink_times.append(elapsed)
                    blink_active = False

            # Gaze
            gaze_horizontal, gaze_v_ratio, gaze_h_ratio = calculate_gaze(points)

            # MAR
            mar = calculate_mar(points)
            mar_history.append(mar)

            # Brow
            brow = calculate_brow_raise(landmarks)
            brow_history.append(brow)

            # Gaze aversion events
            gaze_states.append(gaze_horizontal)
            if gaze_horizontal in ("LEFT", "RIGHT"):
                if gaze_aversion_start is None:
                    gaze_aversion_start = elapsed
            else:
                if gaze_aversion_start is not None:
                    duration = elapsed - gaze_aversion_start
                    if duration >= 0.4:
                        gaze_aversion_events.append((gaze_aversion_start, elapsed))
                        if duration >= 1.5:
                            log_event(event_log, "GAZE_AVERSION",
                                      f"{gaze_horizontal} {duration:.2f}s", elapsed)
                    gaze_aversion_start = None

            # Head pose
            pitch, yaw, roll = calculate_head_pose(points, frame_w, frame_h)
            yaw_history.append(yaw)
            pitch_history.append(pitch)
            roll_history.append(roll)

            # Motion
            landmark_vel = landmark_velocity(prev_face_points, points)
            vel_history.append(landmark_vel)
            jitter = jitter_index(vel_history)
            flow_energy = optical_flow_energy(prev_gray, gray, face_box)
            flow_history.append(flow_energy)
            prev_face_points = points

            # Emotion
            if result.face_blendshapes and len(result.face_blendshapes) > 0:
                dominant, conf, scores = classify_emotion(result.face_blendshapes[0])
                if dominant != "unknown":
                    if dominant != current_emotion:
                        log_event(event_log, "EMOTION_CHANGE",
                                  f"{current_emotion} -> {dominant} ({conf:.2f})",
                                  elapsed)
                    current_emotion = dominant
                    current_emotion_conf = conf
                    current_emotion_scores = scores
                    emotion_scores_history.append(scores)
                    emotion_counts[dominant] += 1

            # Heatmap accumulation (FIXED)
            gx, gy = gaze_point_in_frame(
                face_box, gaze_h_ratio, gaze_v_ratio, frame_w, frame_h
            )
            accumulate_gaze(gaze_heatmap, gx, gy, frame.shape)

        # Fade heatmap
        gaze_heatmap *= HEATMAP_DECAY
        prev_gray = gray

        # Metrics
        blink_rate = (blink_count / elapsed) * 60 if elapsed > 1 else 0
        burst_now = detect_blink_burst(blink_times)
        if burst_now and not burst_flag:
            log_event(event_log, "BLINK_BURST", ">=3 blinks in 3s", elapsed)
        burst_flag = burst_now

        face_percentage = (face_frames / processed_frames * 100
                           if processed_frames > 0 else 0)

        if len(gaze_states) > 10:
            aversion_ratio = sum(1 for g in gaze_states
                                 if g in ("LEFT", "RIGHT")) / len(gaze_states)
        else:
            aversion_ratio = 0.0

        hme_yaw = motion_energy(yaw_history)
        hme_pitch = motion_energy(pitch_history)
        stress_index = compute_stress_index(blink_rate, aversion_ratio,
                                            hme_yaw, hme_pitch, burst_flag)
        engagement = compute_engagement(face_percentage, 1.0 - aversion_ratio)
        ear_z = zscore(ear, ear_history)
        yaw_z = zscore(yaw, yaw_history)

        # Baseline
        if not baseline_ready and elapsed >= BASELINE_SECONDS and len(ear_history) > 30:
            baseline_stats = {
                "ear_mean":   float(np.mean(ear_history)),
                "yaw_mean":   float(np.mean(yaw_history)),
                "pitch_mean": float(np.mean(pitch_history)),
                "blink_rate": blink_rate,
            }
            baseline_ready = True
            log_event(event_log, "BASELINE_SET",
                      json.dumps(baseline_stats), elapsed)

        # YOLO tracking
        if processed_frames % YOLO_EVERY_N_FRAMES == 0:
            tracks = run_tracking(frame)
            for det in tracks:
                if det["label"] == "person" and face_box is not None:
                    fx1, fy1, fx2, fy2 = face_box
                    fcx = (fx1 + fx2) // 2
                    fcy = (fy1 + fy2) // 2
                    x1, y1, x2, y2 = det["box"]
                    if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                        person_id = det["track_id"]
                if det["label"] in ("cell phone", "bottle", "knife"):
                    key = f"{det['track_id']}_{det['label']}"
                    if key not in logged_objects:
                        log_event(event_log, "OBJECT_DETECTED",
                                  f"#{det['track_id']} {det['label']}", elapsed)
                        logged_objects.add(key)

        # Best-stress frame
        if stress_index > best_stress and face_detected:
            best_stress = stress_index
            best_frame = frame.copy()

        # CSV row
        emotion_row = [current_emotion, round(current_emotion_conf, 3)]
        emotion_row += [round(current_emotion_scores.get(e, 0.0), 3)
                        for e in EMOTION_NAMES]

        csv_writer.writerow([
            datetime.now().isoformat(),
            round(elapsed, 2),
            face_detected,
            person_id,
            round(ear, 4), round(mar, 4), round(brow, 4),
            blink_count, round(blink_rate, 2),
            round(yaw, 2), round(pitch, 2), round(roll, 2),
            gaze_horizontal,
            round(landmark_vel, 4), round(jitter, 4), round(flow_energy, 4),
            round(stress_index, 2), round(engagement, 2),
            round(hme_yaw, 3), round(hme_pitch, 3),
            round(aversion_ratio, 4),
            round(ear_z, 3), round(yaw_z, 3),
        ] + emotion_row)

        # Progress
        if verbose and processed_frames % 30 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"  {pct:5.1f}%  processed={processed_frames}  "
                  f"face={face_percentage:.0f}%  emotion={current_emotion}")

    cap.release()
    csv_handle.close()

    # ---------------------------------------------------------
    # FINAL METRICS
    # ---------------------------------------------------------
    avg_ear = sum(ear_history) / len(ear_history) if ear_history else 0
    avg_mar = sum(mar_history) / len(mar_history) if mar_history else 0
    avg_yaw = sum(yaw_history) / len(yaw_history) if yaw_history else 0
    avg_pitch = sum(pitch_history) / len(pitch_history) if pitch_history else 0
    avg_vel = sum(vel_history) / len(vel_history) if vel_history else 0
    avg_flow = sum(flow_history) / len(flow_history) if flow_history else 0

    session_duration = elapsed
    final_blink_rate = (blink_count / session_duration * 60
                        if session_duration > 0 else 0)

    dominant_emotion = (max(emotion_counts, key=emotion_counts.get)
                        if sum(emotion_counts.values()) > 0 else "unknown")

    avg_emotion_scores = {}
    if emotion_scores_history:
        for e in EMOTION_NAMES:
            avg_emotion_scores[e] = round(
                float(np.mean([s.get(e, 0.0) for s in emotion_scores_history])), 3
            )

    # ---------------------------------------------------------
    # SAVE OUTPUTS
    # ---------------------------------------------------------

    # Session summary
    with open(session_dir / "session_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_start", "session_type", "source_file",
            "duration_seconds", "total_frames", "face_detection_percentage",
            "total_blinks", "average_blink_rate",
            "average_ear", "average_mar", "average_yaw", "average_pitch",
            "average_landmark_velocity", "average_optical_flow",
            "final_stress_index", "final_engagement",
            "gaze_aversion_events", "dominant_emotion_overall",
        ] + [f"avg_emotion_{e}" for e in EMOTION_NAMES])
        writer.writerow([
            datetime.now().isoformat(), "video", str(video_path),
            round(session_duration, 2), processed_frames,
            round(face_percentage, 2), blink_count,
            round(final_blink_rate, 2),
                        round(avg_ear, 4), round(avg_mar, 4),
            round(avg_yaw, 2), round(avg_pitch, 2),
            round(avg_vel, 4), round(avg_flow, 4),
            round(stress_index, 2), round(engagement, 2),
            len(gaze_aversion_events), dominant_emotion,
        ] + [avg_emotion_scores.get(e, 0.0) for e in EMOTION_NAMES])

    # Events
    with open(session_dir / "behavioral_events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "elapsed_seconds", "event", "detail"])
        for e in event_log:
            w.writerow([e["timestamp"], e["elapsed"], e["event"], e["detail"]])

    # Heatmap
    heatmap_render = render_heatmap(gaze_heatmap, size=(480, 640))
    cv2.imwrite(str(session_dir / "gaze_heatmap.png"), heatmap_render)
    np.save(str(session_dir / "gaze_heatmap.npy"),
            (gaze_heatmap / (gaze_heatmap.max() + 1e-6)).astype(np.float32))

    # Best-stress preview image
    if best_frame is not None:
        cv2.imwrite(str(session_dir / "annotated_preview.png"), best_frame)

    # JSON report
    report = {
        "session_id":               session_dir.name,
        "session_type":             "video",
        "session_start":            datetime.now().isoformat(),
        "source_file":              str(video_path),
        "video_native_fps":         round(native_fps, 2),
        "video_total_frames":       total_frames,
        "sample_fps":               TARGET_SAMPLE_FPS,
        "processed_frames":         processed_frames,
        "duration_seconds":         round(session_duration, 2),
        "face_detection_pct":       round(face_percentage, 2),
        "blink_count":              blink_count,
        "blink_rate_per_min":       round(final_blink_rate, 2),
        "avg_ear":                  round(avg_ear, 4),
        "avg_mar":                  round(avg_mar, 4),
        "avg_yaw":                  round(avg_yaw, 2),
        "avg_pitch":                round(avg_pitch, 2),
        "avg_landmark_velocity":    round(avg_vel, 4),
        "avg_optical_flow":         round(avg_flow, 4),
        "final_stress_index":       round(stress_index, 2),
        "final_engagement":         round(engagement, 2),
        "gaze_aversion_events":     len(gaze_aversion_events),
        "dominant_emotion_overall": dominant_emotion,
        "avg_emotion_scores":       avg_emotion_scores,
        "baseline":                 baseline_stats,
        "events":                   event_log,
    }
    with open(session_dir / "session_report.json", "w") as f:
        json.dump(report, f, indent=2)

    detector.close()

    if verbose:
        print()
        print("=" * 40)
        print("VIDEO ANALYSIS COMPLETE")
        print("=" * 40)
        print(f"Session:      {session_dir}")
        print(f"Duration:     {session_duration:.1f}s")
        print(f"Frames:       {processed_frames}")
        print(f"Face found:   {face_percentage:.1f}%")
        print(f"Blinks:       {blink_count}  ({final_blink_rate:.1f}/min)")
        print(f"Avg EAR:      {avg_ear:.3f}")
        print(f"Avg MAR:      {avg_mar:.3f}")
        print(f"Stress:       {stress_index:.1f}/100")
        print(f"Engagement:   {engagement:.1f}%")
        print(f"Emotion:      {dominant_emotion}")
        print(f"Events:       {len(event_log)}")
        print("=" * 40)
        print("Files saved:")
        print("  video_features.csv")
        print("  session_summary.csv")
        print("  behavioral_events.csv")
        print("  session_report.json")
        print("  gaze_heatmap.png")
        print("  annotated_preview.png")
        print("=" * 40)

    return session_dir


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python batch_video_analysis.py <path_to_video>")
        print("       (supports .mp4, .avi, .mov, .mkv, .webm)")
        sys.exit(1)

    try:
        analyze_video(sys.argv[1])
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)