# ============================================================
# ENHANCED BEHAVIORAL ANALYSIS SYSTEM
# OpenCV + MediaPipe + YOLO (ByteTrack) + Blendshape Emotion
# + Person Tracking + Gaze Heatmap + Motion Analysis
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
import threading
from datetime import datetime
from collections import deque
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "face_landmarker.task"
EAR_THRESHOLD = 0.22
GRAPH_WIDTH = 340
GRAPH_HEIGHT = 150
HISTORY_LENGTH = 150
LOG_INTERVAL = 3
BASELINE_SECONDS = 10
YOLO_EVERY_N_FRAMES = 4

HEATMAP_SIZE = (240, 320)
HEATMAP_DECAY = 0.995

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
# THREADED CAMERA
# ============================================================

class ThreadedCamera:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Could not open camera.")
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame.copy() if self.frame is not None else None

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


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
        exit(1)


# ============================================================
# SESSION FOLDER
# ============================================================

SESSION_ID = datetime.now().strftime("%Y-%m-%d_%H%M%S")
SESSION_DIR = Path("sessions") / SESSION_ID
SESSION_DIR.mkdir(parents=True, exist_ok=True)
print(f"Session folder: {SESSION_DIR}")


# ============================================================
# MEDIAPIPE & YOLO
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
# CAMERA INIT & WINDOW
# ============================================================

try:
    cap = ThreadedCamera(0, width=1280, height=720)
except RuntimeError as e:
    print(e)
    exit()

WINDOW_NAME = "Enhanced Behavioral Analysis"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("Enhanced Behavioral Analysis started.")
print("Press Q to quit. Press F to toggle fullscreen.")


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
# HELPER FUNCTIONS
# ============================================================

def draw_panel(frame, x, y, w, h, bg_color=(20, 20, 25), alpha=0.6):
    x, y, w, h = int(x), int(y), int(w), int(h)
    frame_h, frame_w = frame.shape[:2]
    x2 = min(x + w, frame_w)
    y2 = min(y + h, frame_h)
    x, y = max(0, x), max(0, y)
    if x >= x2 or y >= y2:
        return
    sub_img = frame[y:y2, x:x2]
    rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 0)
    frame[y:y2, x:x2] = res


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    v1 = distance(p2, p6)
    v2 = distance(p3, p5)
    h = distance(p1, p4)
    if h == 0:
        return 0
    return (v1 + v2) / (2.0 * h)


def calculate_gaze(landmarks):
    li_x = np.mean([landmarks[i][0] for i in LEFT_IRIS])
    li_y = np.mean([landmarks[i][1] for i in LEFT_IRIS])
    ri_x = np.mean([landmarks[i][0] for i in RIGHT_IRIS])
    ri_y = np.mean([landmarks[i][1] for i in RIGHT_IRIS])

    lr = (li_x - landmarks[362][0]) / (landmarks[263][0] - landmarks[362][0] + 1e-6)
    rr = (ri_x - landmarks[33][0]) / (landmarks[133][0] - landmarks[33][0] + 1e-6)
    hr = (lr + rr) / 2

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


def calculate_brow_raise(landmarks, points):
    lb = np.mean([landmarks[i].y for i in LEFT_BROW])
    rb = np.mean([landmarks[i].y for i in RIGHT_BROW])
    le = landmarks[362].y
    re = landmarks[263].y
    return ((le - lb) + (re - rb)) / 2.0


def motion_energy(history):
    if len(history) < 5:
        return 0.0
    return float(np.std(list(history)))


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


def compute_stress_index(blink_rate, gaze_aversion_ratio, hme_yaw, hme_pitch, burst_flag):
    score = 0.0
    if blink_rate > 20:
        score += min((blink_rate - 20) * 1.2, 25)
    score += min(gaze_aversion_ratio * 40, 25)
    score += min(hme_yaw * 0.8, 15)
    score += min(hme_pitch * 0.8, 15)
    if burst_flag:
        score += 20
    return min(score, 100.0)


def compute_engagement(face_pct, gaze_center_ratio):
    return 0.5 * face_pct + 0.5 * (gaze_center_ratio * 100)


def landmark_velocity(prev_points, curr_points):
    if prev_points is None or curr_points is None:
        return 0.0
    if len(prev_points) != len(curr_points):
        return 0.0
    xs = [p[0] for p in curr_points]
    face_w = max(1, max(xs) - min(xs))
    disps = [distance(prev_points[i], curr_points[i])
             for i in range(0, len(curr_points), 5)]
    return float(np.mean(disps) / face_w) if disps else 0.0


def jitter_index(vel_history, recent_n=15):
    if len(vel_history) < recent_n:
        return 0.0
    recent = list(vel_history)[-recent_n:]
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
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        verbose=False,
    )
    detections = []
    if not results:
        return detections
    r = results[0]
    if r.boxes is None or r.boxes.id is None:
        return detections

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h

    for box, cls_id, tid in zip(r.boxes, r.boxes.cls, r.boxes.id):
        cls_id = int(cls_id)
        if cls_id not in WATCHED_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > 0.7 * frame_area:
            continue

        detections.append({
            "track_id": int(tid),
            "label":    WATCHED_CLASSES[cls_id],
            "conf":     float(box.conf[0]),
            "box":      (x1, y1, x2, y2),
        })
    return detections


def gaze_point_in_frame(face_box, hr, vr, frame_w, frame_h):
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
    hx = int(x / fw * W)
    hy = int(y / fh * H)
    hx = max(0, min(W - 1, hx))
    hy = max(0, min(H - 1, hy))
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
    norm = heatmap / heatmap.max()
    norm = np.clip(norm * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.resize(colored, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def classify_emotion_from_blendshapes(blendshapes):
    if not blendshapes:
        return "unknown", 0.0, {e: 0.0 for e in EMOTION_NAMES}

    bs = {cat.category_name: cat.score for cat in blendshapes}

    def g(name): return bs.get(name, 0.0)

    happy = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2 + 0.3 * (g("cheekSquintLeft") + g("cheekSquintRight")) / 2
    sad = (g("mouthFrownLeft") + g("mouthFrownRight")) / 2 + 0.5 * g("browInnerUp") + 0.3 * (g("mouthLowerDownLeft") + g("mouthLowerDownRight")) / 2
    angry = (g("browDownLeft") + g("browDownRight")) / 2 + 0.4 * (g("mouthPressLeft") + g("mouthPressRight")) / 2 + 0.3 * (g("noseSneerLeft") + g("noseSneerRight")) / 2
    surprise = g("jawOpen") + 0.5 * (g("eyeWideLeft") + g("eyeWideRight")) / 2 + 0.4 * (g("browOuterUpLeft") + g("browOuterUpRight")) / 2 + 0.3 * g("browInnerUp")
    fear = (g("eyeWideLeft") + g("eyeWideRight")) / 2 + 0.6 * g("browInnerUp") + 0.4 * (g("browOuterUpLeft") + g("browOuterUpRight")) / 2 + 0.2 * g("jawOpen")
    disgust = (g("noseSneerLeft") + g("noseSneerRight")) / 2 + (g("mouthUpperUpLeft") + g("mouthUpperUpRight")) / 2 + 0.3 * (g("browDownLeft") + g("browDownRight")) / 2

    raw = {"happy": happy, "sad": sad, "angry": angry, "surprise": surprise, "fear": fear, "disgust": disgust}
    total = sum(raw.values())

    if total < 0.15:
        scores = {e: 0.0 for e in EMOTION_NAMES}
        scores["neutral"] = 1.0
        return "neutral", 1.0, scores

    scores = {e: raw[e] / total for e in raw}
    neutral_weight = max(0.0, 1.0 - total)
    active_scale = 1.0 - neutral_weight
    for e in scores:
        scores[e] *= active_scale
    scores["neutral"] = neutral_weight

    for e in EMOTION_NAMES:
        scores.setdefault(e, 0.0)

    dominant = max(scores, key=scores.get)
    return (dominant, round(scores[dominant], 3), {k: round(v, 3) for k, v in scores.items()})


def draw_graph_fast(frame, values, x, y, width, height, title):
    draw_panel(frame, x, y, width, height, bg_color=(20, 20, 25), alpha=0.65)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
    cv2.putText(frame, title, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    if len(values) < 2:
        return

    vals = np.array(values)
    min_v, max_v = vals.min(), vals.max()
    if max_v - min_v < 1e-3:
        max_v += 1e-3

    x_coords = np.linspace(x + 5, x + width - 5, len(vals), dtype=np.int32)
    y_coords = (y + height - 10 - ((vals - min_v) / (max_v - min_v)) * (height - 35)).astype(np.int32)

    pts = np.vstack((x_coords, y_coords)).T.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)


def color_for_score(s):
    if s < 33: return (0, 200, 0)
    if s < 66: return (0, 200, 255)
    return (0, 0, 255)


def log_event(event_type, detail, elapsed):
    event_log.append({
        "timestamp": datetime.now().isoformat(),
        "elapsed": round(elapsed, 2),
        "event": event_type,
        "detail": detail,
    })
    print(f"[EVENT {elapsed:6.2f}s] {event_type}: {detail}")


# ============================================================
# CSV SETUP
# ============================================================

csv_file = str(SESSION_DIR / "video_features.csv")
csv_handle = open(csv_file, "w", newline="")
csv_writer = csv.writer(csv_handle)

EMOTION_COLUMNS = ["emotion_dominant", "emotion_confidence"] + [f"emotion_{e}" for e in EMOTION_NAMES]

csv_writer.writerow([
    "timestamp", "elapsed_seconds", "face_detected",
    "person_id",
    "ear", "mar", "brow", "blink_count", "blink_rate",
    "yaw", "pitch", "roll", "gaze_horizontal",
    "landmark_velocity", "jitter_index", "optical_flow",
    "stress_index", "engagement", "hme_yaw", "hme_pitch",
    "aversion_ratio", "ear_z", "yaw_z",
] + EMOTION_COLUMNS)


# ============================================================
# STATE
# ============================================================

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

emotion_history = deque(maxlen=300)
emotion_confidence_history = deque(maxlen=300)
emotion_scores_history = deque(maxlen=300)
emotion_counts = {e: 0 for e in EMOTION_NAMES}

gaze_heatmap = np.zeros(HEATMAP_SIZE, dtype=np.float32)

cached_tracks = []
prev_face_points = None
prev_gray = None
current_person_id = -1

event_log = []
baseline_ready = False
baseline_stats = {}
blink_times = []
total_frames = 0
face_frames = 0
blink_count = 0
blink_active = False
start_time = time.time()
last_log_time = 0
timestamp_ms = 0
_logged_objects = set()
_burst_flag = False

ear = mar = brow = yaw = pitch = roll = 0.0
gaze_horizontal = "UNKNOWN"
gaze_h_ratio = 0.5
gaze_v_ratio = 0.5
landmark_vel = jitter = flow_energy = 0.0
stress_index = engagement = hme_yaw = hme_pitch = aversion_ratio = ear_z = yaw_z = blink_rate = face_percentage = 0.0
current_emotion = "neutral"
current_emotion_conf = 0.0
current_emotion_scores = {e: 0.0 for e in EMOTION_NAMES}
emotion_bs_count = 0
# ============================================================
# MAIN LOOP
# ============================================================

is_fullscreen = False

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR: Could not read frame.")
        break

    total_frames += 1
    elapsed = time.time() - start_time
    timestamp_ms += 33
    frame_h, frame_w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    face_detected = 0
    emotion_bs_count = 0
    face_box = None

    if result.face_landmarks:
        face_detected = 1
        face_frames += 1
        landmarks = result.face_landmarks[0]
        points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]

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
        brow = calculate_brow_raise(landmarks, points)
        brow_history.append(brow)

        # Gaze aversion
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
                        log_event("GAZE_AVERSION",
                                  f"{gaze_horizontal} {duration:.2f}s", elapsed)
                gaze_aversion_start = None

        # Head pose
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

        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            rvec, tvec = cv2.solvePnPRefineLM(
                model_points, image_points, camera_matrix, distortion, rvec, tvec
            )
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            if sy >= 1e-6:
                pitch = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
                yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
                roll = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
                if pitch > 90: pitch -= 180
                elif pitch < -90: pitch += 180
                if yaw > 90: yaw -= 180
                elif yaw < -90: yaw += 180
                if roll > 90: roll -= 180
                elif roll < -90: roll += 180

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
            emotion_bs_count = len(result.face_blendshapes[0])
            dominant, conf, scores = classify_emotion_from_blendshapes(result.face_blendshapes[0])
            if dominant != "unknown":
                if dominant != current_emotion:
                    log_event("EMOTION_CHANGE",
                              f"{current_emotion} -> {dominant} ({conf:.2f})", elapsed)
                current_emotion = dominant
                current_emotion_conf = conf
                current_emotion_scores = scores
                emotion_history.append(dominant)
                emotion_confidence_history.append(conf)
                emotion_scores_history.append(scores)
                emotion_counts[dominant] += 1

        # Gaze heatmap
        gx, gy = gaze_point_in_frame(
            face_box, gaze_h_ratio, gaze_v_ratio, frame_w, frame_h
        )
        accumulate_gaze(gaze_heatmap, gx, gy, frame.shape)

        for p in points[::5]:
            cv2.circle(frame, p, 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    gaze_heatmap *= HEATMAP_DECAY
    prev_gray = gray

    # ========================================================
    # METRICS
    # ========================================================

    blink_rate = (blink_count / elapsed) * 60 if elapsed > 1 else 0
    burst_now = detect_blink_burst(blink_times)
    if burst_now and not _burst_flag:
        log_event("BLINK_BURST", ">=3 blinks in 3s", elapsed)
    _burst_flag = burst_now

    face_percentage = face_frames / total_frames * 100 if total_frames > 0 else 0
    aversion_ratio = (
        sum(1 for g in gaze_states if g in ("LEFT", "RIGHT")) / len(gaze_states)
        if len(gaze_states) > 10 else 0.0
    )
    hme_yaw = motion_energy(yaw_history)
    hme_pitch = motion_energy(pitch_history)
    stress_index = compute_stress_index(
        blink_rate, aversion_ratio, hme_yaw, hme_pitch, _burst_flag
    )
    engagement = compute_engagement(face_percentage, 1.0 - aversion_ratio)
    ear_z, yaw_z = zscore(ear, ear_history), zscore(yaw, yaw_history)

    if not baseline_ready and elapsed >= BASELINE_SECONDS and len(ear_history) > 30:
        baseline_stats = {
            "ear_mean":   float(np.mean(ear_history)),
            "yaw_mean":   float(np.mean(yaw_history)),
            "pitch_mean": float(np.mean(pitch_history)),
            "blink_rate": blink_rate,
        }
        baseline_ready = True
        log_event("BASELINE_SET", json.dumps(baseline_stats), elapsed)

    # ========================================================
    # YOLO + ByteTrack
    # ========================================================

    if total_frames % YOLO_EVERY_N_FRAMES == 0:
        cached_tracks = run_tracking(frame)

    for det in cached_tracks:
        x1, y1, x2, y2 = det["box"]
        tid = det["track_id"]
        label = f'#{tid} {det["label"]} {det["conf"]:.2f}'
        color = (0, 200, 255) if det["label"] == "person" else (255, 100, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if det["label"] == "person" and face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
            if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                current_person_id = tid

        if det["label"] in ("cell phone", "bottle", "knife"):
            key = f"{tid}_{det['label']}"
            if key not in _logged_objects:
                log_event("OBJECT_DETECTED", f"#{tid} {det['label']}", elapsed)
                _logged_objects.add(key)

    # ========================================================
    # HUD — Left panel
    # ========================================================

    draw_panel(frame, 10, 10, 260, 480, bg_color=(15, 15, 20), alpha=0.6)

    y_off = 32
    hud_lines = [
        (f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}",   (0, 255, 0)),
        (f"Person ID: {current_person_id}",          (0, 255, 255)),
        (f"EAR: {ear:.3f}",                          (0, 255, 0)),
        (f"Blinks: {blink_count}",                   (0, 255, 0)),
        (f"Blink Rate: {blink_rate:.1f}/min",        (0, 255, 255)),
        (f"Yaw: {yaw:.1f}",                          (255, 255, 0)),
        (f"Pitch: {pitch:.1f}",                      (255, 255, 0)),
        (f"Roll: {roll:.1f}",                        (255, 255, 0)),
        (f"Gaze: {gaze_horizontal}",                 (0, 255, 255)),
        (f"Face: {face_percentage:.0f}%",            (255, 255, 255)),
    ]

    for line, color in hud_lines:
        cv2.putText(frame, line, (20, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y_off += 26

    y_off += 6
    cv2.putText(frame, f"Stress: {stress_index:.0f}/100", (20, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color_for_score(stress_index), 2, cv2.LINE_AA)
    y_off += 8
    cv2.rectangle(frame, (20, y_off), (250, y_off + 12), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, y_off),
                  (20 + int(230 * stress_index / 100), y_off + 12),
                  color_for_score(stress_index), -1)

    y_off += 32
    cv2.putText(frame, f"Engagement: {engagement:.0f}%", (20, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color_for_score(100 - engagement), 2, cv2.LINE_AA)

    y_off += 28
    cv2.putText(frame, f"Vel: {landmark_vel:.3f}  Jitter: {jitter:.3f}",
                (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 255), 1, cv2.LINE_AA)
    y_off += 22
    cv2.putText(frame, f"Flow: {flow_energy:.2f}",
                (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 255), 1, cv2.LINE_AA)
    y_off += 22
    cv2.putText(frame, f"MAR: {mar:.2f}  Brow: {brow:.2f}",
                (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 255), 1, cv2.LINE_AA)
    y_off += 22
    cv2.putText(frame, f"Motion: {hme_yaw:.1f} / {hme_pitch:.1f}",
                (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 255), 1, cv2.LINE_AA)
    y_off += 22
    cv2.putText(frame, f"Baseline: {'READY' if baseline_ready else 'calibrating...'}",
                (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 255, 180), 1, cv2.LINE_AA)

    # --------------------------------------------------------
    # Emotion panel — top-center (no gender)
    # --------------------------------------------------------
    emo_color = EMOTION_COLORS_BGR.get(current_emotion, (255, 180, 100))
    center_x = max(280, (frame_w - 260) // 2)

    draw_panel(frame, center_x, 10, 260, 80, bg_color=(15, 15, 20), alpha=0.6)
    cv2.putText(frame, f"EMOTION: {current_emotion.upper()}",
                (center_x + 12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, emo_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {current_emotion_conf:.2f}",
                (center_x + 12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Blendshapes: {emotion_bs_count}",
                (center_x + 12, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)

    # --------------------------------------------------------
    # Right column — graphs + heatmap
    # --------------------------------------------------------
    graph_x = max(10, frame_w - GRAPH_WIDTH - 15)

    draw_graph_fast(frame, ear_history, graph_x, 10,
                    GRAPH_WIDTH, GRAPH_HEIGHT, "EAR - Eye Aspect Ratio")
    draw_graph_fast(frame, yaw_history, graph_x, 20 + GRAPH_HEIGHT,
                    GRAPH_WIDTH, GRAPH_HEIGHT, "Yaw - Head Rotation")

    hm_display = render_heatmap(gaze_heatmap, size=HEATMAP_SIZE)
    hm_h, hm_w = HEATMAP_SIZE

    hm_x = max(0, frame_w - hm_w - 15)
    hm_y = max(0, frame_h - hm_h - 15)

    if hm_x > 0 and hm_y > 0 and hm_x + hm_w <= frame_w and hm_y + hm_h <= frame_h:
        roi = frame[hm_y:hm_y + hm_h, hm_x:hm_x + hm_w]
        blended = cv2.addWeighted(roi, 0.55, hm_display, 0.45, 0)
        frame[hm_y:hm_y + hm_h, hm_x:hm_x + hm_w] = blended

        cv2.rectangle(frame, (hm_x - 1, hm_y - 1),
                      (hm_x + hm_w, hm_y + hm_h), (80, 80, 80), 1)
        cv2.putText(frame, "Gaze Heatmap", (hm_x + 8, hm_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ========================================================
    # CSV LOG
    # ========================================================

    if elapsed - last_log_time >= LOG_INTERVAL / 30:
        emotion_row = [current_emotion, round(current_emotion_conf, 3)]
        emotion_row += [round(current_emotion_scores.get(e, 0.0), 3)
                        for e in EMOTION_NAMES]

        csv_writer.writerow([
            datetime.now().isoformat(),
            round(elapsed, 2),
            face_detected,
            current_person_id,
            round(ear, 4),
            round(mar, 4),
            round(brow, 4),
            blink_count,
            round(blink_rate, 2),
            round(yaw, 2),
            round(pitch, 2),
            round(roll, 2),
            gaze_horizontal,
            round(landmark_vel, 4),
            round(jitter, 4),
            round(flow_energy, 4),
            round(stress_index, 2),
            round(engagement, 2),
            round(hme_yaw, 3),
            round(hme_pitch, 3),
            round(aversion_ratio, 4),
            round(ear_z, 3),
            round(yaw_z, 3),
        ] + emotion_row)

        last_log_time = elapsed

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("f") or key == ord("F"):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL)

    if key == ord("q") or key == ord("Q"):
        break


# ============================================================
# SESSION SUMMARY
# ============================================================

session_duration = time.time() - start_time

average_ear = sum(ear_history) / len(ear_history) if ear_history else 0
average_yaw = sum(yaw_history) / len(yaw_history) if yaw_history else 0
average_pitch = sum(pitch_history) / len(pitch_history) if pitch_history else 0
average_mar = sum(mar_history) / len(mar_history) if mar_history else 0
average_vel = sum(vel_history) / len(vel_history) if vel_history else 0
average_flow = sum(flow_history) / len(flow_history) if flow_history else 0

final_blink_rate = (blink_count / session_duration * 60
                    if session_duration > 0 else 0)

dominant_emotion_overall = (
    max(emotion_counts, key=emotion_counts.get)
    if sum(emotion_counts.values()) > 0 else "unknown"
)

avg_emotion_scores = {}
if emotion_scores_history:
    for e in EMOTION_NAMES:
        avg_emotion_scores[e] = round(
            float(np.mean([s.get(e, 0.0) for s in emotion_scores_history])), 3
        )


# ============================================================
# SAVE SESSION SUMMARY CSV
# ============================================================

with open(SESSION_DIR / "session_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "session_start", "duration_seconds", "total_frames",
        "face_detection_percentage", "total_blinks",
        "average_blink_rate", "average_ear", "average_mar",
        "average_yaw", "average_pitch",
        "average_landmark_velocity", "average_optical_flow",
        "final_stress_index", "final_engagement",
        "gaze_aversion_events", "dominant_emotion_overall",
    ] + [f"avg_emotion_{e}" for e in EMOTION_NAMES])

    writer.writerow([
        datetime.now().isoformat(),
        round(session_duration, 2),
        total_frames,
        round(face_percentage, 2),
        blink_count,
        round(final_blink_rate, 2),
        round(average_ear, 4),
        round(average_mar, 4),
        round(average_yaw, 2),
        round(average_pitch, 2),
        round(average_vel, 4),
        round(average_flow, 4),
        round(stress_index, 2),
        round(engagement, 2),
        len(gaze_aversion_events),
        dominant_emotion_overall,
    ] + [avg_emotion_scores.get(e, 0.0) for e in EMOTION_NAMES])


# ============================================================
# SAVE BEHAVIORAL EVENTS CSV
# ============================================================

with open(SESSION_DIR / "behavioral_events.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "elapsed_seconds", "event", "detail"])
    for e in event_log:
        w.writerow([e["timestamp"], e["elapsed"], e["event"], e["detail"]])


# ============================================================
# SAVE GAZE HEATMAP
# ============================================================

heatmap_render = render_heatmap(gaze_heatmap, size=(480, 640))
cv2.imwrite(str(SESSION_DIR / "gaze_heatmap.png"), heatmap_render)

raw_norm = gaze_heatmap / (gaze_heatmap.max() + 1e-6)
np.save(str(SESSION_DIR / "gaze_heatmap.npy"), raw_norm.astype(np.float32))


# ============================================================
# SAVE JSON REPORT
# ============================================================

report = {
    "session_id":              SESSION_ID,
    "session_start":           datetime.now().isoformat(),
    "duration_seconds":        round(session_duration, 2),
    "total_frames":            total_frames,
    "face_detection_pct":      round(face_percentage, 2),
    "blink_count":             blink_count,
    "blink_rate_per_min":      round(final_blink_rate, 2),
    "avg_ear":                 round(average_ear, 4),
    "avg_mar":                 round(average_mar, 4),
    "avg_yaw":                 round(average_yaw, 2),
    "avg_pitch":               round(average_pitch, 2),
    "avg_landmark_velocity":   round(average_vel, 4),
    "avg_optical_flow":        round(average_flow, 4),
    "final_stress_index":      round(stress_index, 2),
    "final_engagement":        round(engagement, 2),
    "gaze_aversion_events":    len(gaze_aversion_events),
    "dominant_emotion_overall": dominant_emotion_overall,
    "avg_emotion_scores":      avg_emotion_scores,
    "baseline":                baseline_stats,
    "events":                  event_log,
}

with open(SESSION_DIR / "session_report.json", "w") as f:
    json.dump(report, f, indent=2)


# ============================================================
# CLEANUP
# ============================================================

csv_handle.close()
detector.close()
cap.release()
cv2.destroyAllWindows()


# ============================================================
# FINAL SUMMARY PRINT
# ============================================================

print()
print("======================================")
print("SESSION COMPLETE")
print("======================================")
print(f"Session folder: {SESSION_DIR}")
print(f"Duration: {session_duration:.1f} seconds")
print(f"Total blinks: {blink_count}")
print(f"Average blink rate: {final_blink_rate:.1f}/min")
print(f"Average EAR: {average_ear:.3f}")
print(f"Average MAR: {average_mar:.3f}")
print(f"Average yaw: {average_yaw:.2f}")
print(f"Average pitch: {average_pitch:.2f}")
print(f"Average landmark velocity: {average_vel:.4f}")
print(f"Average optical flow: {average_flow:.4f}")
print(f"Face detection: {face_percentage:.1f}%")
print(f"Final stress index: {stress_index:.1f}/100")
print(f"Final engagement: {engagement:.1f}%")
print(f"Gaze aversion events: {len(gaze_aversion_events)}")
print(f"Dominant emotion: {dominant_emotion_overall}")
print("======================================")
print("Saved files:")
print(f"  {SESSION_DIR}/video_features.csv")
print(f"  {SESSION_DIR}/session_summary.csv")
print(f"  {SESSION_DIR}/behavioral_events.csv")
print(f"  {SESSION_DIR}/session_report.json")
print(f"  {SESSION_DIR}/gaze_heatmap.png")
print(f"  {SESSION_DIR}/gaze_heatmap.npy")
print("======================================")