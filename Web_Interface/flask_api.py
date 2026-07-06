# Flask API for Piano Playing Level Classification
# Receives a video file, runs MediaPipe hand tracking per frame,
# extracts features, and runs ensemble inference across all 5 LSTM models.
#
# Endpoints:
#   POST /predict      - accepts a video file, returns classification result as JSON
#   GET  /video/<name> - streams an annotated video file
#   GET  /health       - returns API status and number of loaded models

# Libraries
import os
import gc
import uuid
import subprocess
import warnings
from collections import deque
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import imageio.v2 as iio
import imageio_ffmpeg

from flask import Flask, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename


# Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "Tugas Akhir Patrick",
                            "OUTPUT_PIANO_LSTM", "best_model")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"mp4", "mov"}
MAX_MB      = 50

os.makedirs(UPLOAD_DIR, exist_ok=True)


# Feature extraction parameters — must match the training notebook exactly
SEQUENCE_LENGTH = 30   # number of frames per input sequence
STRIDE          = 5    # step size when sliding the window across frames
FEATURE_SIZE    = 83   # 21 landmarks x 3 coords + 10 joint angles + 10 fingertip distances

# Joint triplets used to compute finger bend angles via dot product
JOINTS = [
    [2,  1,  3], [3,  2,  4],
    [6,  5,  7], [7,  6,  8],
    [10, 9,  11],[11, 10, 12],
    [14, 13, 15],[15, 14, 16],
    [18, 17, 19],[19, 18, 20],
]

LABEL_NAMES   = ["good", "needs_improvement", "poor"]
LABEL_DISPLAY = {
    "good":             "Good",
    "needs_improvement":"Needs Improvement",
    "poor":             "Poor",
}

# BGR colors used when drawing the result label on the annotated video
LABEL_COLOR = {
    "good":             (86,  205, 86),
    "needs_improvement":(60,  180, 245),
    "poor":             (60,  60,  239),
}

FINGER_INFO = [
    (4,  "Thumb"),
    (8,  "Index Finger"),
    (12, "Middle Finger"),
    (16, "Ring Finger"),
    (20, "Pinky"),
]

# C major scale note mapping per hand (with thumb-crossing fingering).
# Each finger maps to a list because thumb/index/middle cover 2 notes after the crossover.
#
# Right hand (ascending Do→Do):
#   C(Do)=Thumb, D(Re)=Index, E(Mi)=Middle, F(Fa)=Thumb-cross, G(Sol)=Index, A(La)=Middle, B(Si)=Ring, C(Do)=Pinky
# Left hand (ascending Do→Do):
#   C(Do)=Pinky, D(Re)=Ring, E(Mi)=Middle, F(Fa)=Index, G(Sol)=Thumb, A(La)=Middle-cross, B(Si)=Index, C(Do)=Thumb
#
# MediaPipe reports handedness assuming a mirrored (selfie) view, so for a front-facing
# performance recording: MediaPipe "Left" label = pianist's RIGHT hand, and vice-versa.
FINGER_NOTE_RIGHT = {
    4:  ["Do", "Fa"],    # Thumb  → C then crosses under to F
    8:  ["Re", "Sol"],   # Index  → D then G
    12: ["Mi", "La"],    # Middle → E then A
    16: ["Si"],          # Ring   → B
    20: ["Do"],          # Pinky  → C (upper octave)
}
FINGER_NOTE_LEFT = {
    4:  ["Sol", "Do"],   # Thumb  → G then crosses over to C (upper octave)
    8:  ["Fa", "Si"],    # Index  → F then B
    12: ["Mi", "La"],    # Middle → E then A (after crossover)
    16: ["Re"],          # Ring   → D
    20: ["Do"],          # Pinky  → C (lower octave)
}

# PIP (proximal interphalangeal) joint for each fingertip — used for position check
_PIP = {4: 3, 8: 6, 12: 10, 16: 14, 20: 18}


class FingertipPressTracker:
    """
    Detects piano key presses using THREE signals per finger:

    1. VELOCITY signal — fingertip Y-velocity relative to wrist.
       Detects the downward movement of a fast press.

    2. POSITION signal — fingertip Y is below the PIP (middle knuckle).
       Structural gate: prevents false positives from whole-hand drift.

    3. STATIC PRESS signal — fingertip stays below PIP for N consecutive frames.
       Catches slow/deliberate presses where velocity is too small to trigger
       the velocity gate alone. This is the main fix for missed slow presses.

    State machine per finger:
      IDLE      → PRESSING  : (velocity > thr AND tip below PIP)
                              OR static_press (slow deliberate press)
      PRESSING  → PRESSED   : sustained for CONFIRM_FRAMES AND still below PIP
      PRESSING  → IDLE      : velocity reverses AND tip no longer below PIP
      PRESSED   → RELEASING : velocity < -thr AND tip no longer below PIP
                              (held for at least HOLD_FRAMES to suppress flicker)
      RELEASING → PRESSING  : velocity > thr AND tip below PIP again
      RELEASING → IDLE      : tip no longer below PIP AND near-zero velocity
    """
    SMOOTH         = 3    # shorter window = more responsive to fast presses
    CONFIRM_FRAMES = 3    # confirm press 1 frame faster than before
    HOLD_FRAMES    = 4    # hold state longer to suppress flicker on release
    STATIC_FRAMES  = 6    # frames tip must stay below PIP to auto-confirm slow press

    # MCP (knuckle) landmarks — additional reference for press depth
    _MCP = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}

    def __init__(self):
        self.tip_y   = {idx: deque(maxlen=15) for idx, _ in FINGER_INFO}
        self.wrist_y = deque(maxlen=15)
        self.states  = {idx: 'idle' for idx, _ in FINGER_INFO}
        self.press_f = {idx: 0      for idx, _ in FINGER_INFO}
        self.hold_f  = {idx: 0      for idx, _ in FINGER_INFO}
        self.below_f = {idx: 0      for idx, _ in FINGER_INFO}  # consecutive frames below PIP

    def _velocity(self, buf) -> float:
        h = list(buf)
        if len(h) < 2:
            return 0.0
        diffs = [h[i] - h[i-1] for i in range(max(1, len(h) - self.SMOOTH), len(h))]
        return float(np.mean(diffs)) if diffs else 0.0

    def _below_pip(self, tip_idx: int, hand_lm, pos_thr: float) -> bool:
        """True when fingertip Y is below its PIP joint by at least pos_thr."""
        tip_y = hand_lm.landmark[tip_idx].y
        pip_y = hand_lm.landmark[_PIP[tip_idx]].y
        return (tip_y - pip_y) > pos_thr

    def update(self, hand_lm) -> dict:
        """
        Call once per frame. Returns {tip_idx: bool} — True = finger is pressing.

        Dataset analysis findings that shaped these thresholds:
        - hand_h ranges 0.002–0.155 across videos (close-up vs landscape).
          Using hand_h * factor alone makes press_thr ~0.0003 for close-up videos
          (velocity gate dies). A minimum floor of 0.004 fixes this.
        - Poor-technique players have flat fingers: tip_y - pip_y averages -0.014
          (tip ABOVE PIP). Strict pos_thr blocks all their presses.
          Two separate thresholds: permissive for velocity trigger, strict for static.
        - Resting velocity ~0.003, press velocity ~0.008+. press_thr=0.004 floor
          sits cleanly between noise floor and genuine press signal.
        """
        wrist_y = hand_lm.landmark[0].y
        self.wrist_y.append(wrist_y)
        dw = self._velocity(self.wrist_y)

        mid_mcp_y = hand_lm.landmark[9].y
        hand_h    = abs(wrist_y - mid_mcp_y) or 0.01

        # Velocity threshold: floor of 0.004 prevents near-zero threshold on close-up videos.
        # Scales up for larger hand_h (landscape / farther-away camera) so fast presses
        # in those videos still require proportionally more movement to trigger.
        press_thr = max(0.004, hand_h * 0.30)
        rel_thr   = press_thr * 0.40

        # Permissive position gate for velocity-triggered presses:
        # allows tip up to 0.5 * hand_h ABOVE PIP — catches flat-finger (poor) technique.
        pos_thr_vel = -hand_h * 0.5

        # Strict position gate for static (slow) presses:
        # requires tip to be genuinely below PIP to avoid always-on false positives.
        pos_thr_static = hand_h * 0.01

        result = {}
        for tip_idx, _ in FINGER_INFO:
            y = hand_lm.landmark[tip_idx].y
            self.tip_y[tip_idx].append(y)

            dt         = self._velocity(self.tip_y[tip_idx])
            dy_rel     = dt - dw
            down_vel   = self._below_pip(tip_idx, hand_lm, pos_thr_vel)    # permissive
            down_static= self._below_pip(tip_idx, hand_lm, pos_thr_static) # strict

            # Static counter uses the strict threshold to prevent always-on triggers
            if down_static:
                self.below_f[tip_idx] += 1
            else:
                self.below_f[tip_idx] = 0

            static_press = self.below_f[tip_idx] >= self.STATIC_FRAMES

            state = self.states[tip_idx]

            if state == 'idle':
                # Velocity trigger: uses permissive position gate (catches flat fingers)
                # Static trigger: uses strict position gate (tip must be below PIP)
                if (dy_rel > press_thr and down_vel) or static_press:
                    state = 'pressing'
                    self.press_f[tip_idx] = 1

            elif state == 'pressing':
                self.press_f[tip_idx] += 1
                if self.press_f[tip_idx] >= self.CONFIRM_FRAMES and down_vel:
                    state = 'pressed'
                    self.hold_f[tip_idx] = 0
                elif dy_rel < -rel_thr and not down_vel:
                    state = 'idle'
                    self.press_f[tip_idx] = 0

            elif state == 'pressed':
                self.hold_f[tip_idx] += 1
                if self.hold_f[tip_idx] >= self.HOLD_FRAMES and not static_press:
                    if dy_rel < -rel_thr and not down_vel:
                        state = 'releasing'

            elif state == 'releasing':
                if (dy_rel > press_thr and down_vel) or static_press:
                    state = 'pressing'
                    self.press_f[tip_idx] = 1
                elif not down_vel and abs(dy_rel) < press_thr * 0.5:
                    state = 'idle'
                    self.press_f[tip_idx] = 0
                    self.below_f[tip_idx] = 0

            self.states[tip_idx] = state
            result[tip_idx] = state in ('pressing', 'pressed')

        return result

# Short explanation shown in the UI info tooltip for each classification label
LABEL_EXPLANATION = {
    "good":
        "Your fingers are curling properly, pressing the keys with the fingertip, "
        "and your hand stays stable throughout — all signs of good technique.",
    "needs_improvement":
        "Your technique is on the right track, but the AI detected some inconsistencies "
        "— such as uneven finger pressure or slight wrist instability. More focused practice will fix this.",
    "poor":
        "The AI detected that your fingers are often flat (not curved) when pressing the keys. "
        "Try to keep your fingers slightly curved, like you are gently holding a ball, "
        "and press each key with the very tip of your finger.",
}


# MediaPipe hand tracking modules
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Visual style for drawing hand landmarks and connections on each frame
LANDMARK_SPEC   = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)


# Load all 5 fold models into memory at startup so inference requests are fast
print("=" * 55)
print("Loading ensemble models...")
print("=" * 55)

models = []
for fold in range(1, 6):
    path = os.path.join(MODEL_DIR, f"best_model_fold_{fold}.h5")
    if os.path.isfile(path):
        m = tf.keras.models.load_model(path)
        models.append(m)
        print(f"  [OK] Fold {fold}: {path}")
    else:
        print(f"  [SKIP] Fold {fold}: file not found at {path}")

if not models:
    raise RuntimeError("No models loaded. Check MODEL_DIR.")

print(f"\nEnsemble ready: {len(models)} models loaded.\n")


def calculate_angle(a, b, c):
    """Compute the angle at point b formed by vectors b->a and b->c, in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_frame_features(hand_landmarks):
    """
    Build the 83-dimensional feature vector for a single frame.
    Concatenates raw landmark coordinates, joint angles, and fingertip distances.
    """
    features = []
    coords   = []

    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
        features.extend([lm.x, lm.y, lm.z])

    coords = np.array(coords)

    for joint in JOINTS:
        features.append(calculate_angle(coords[joint[0]], coords[joint[1]], coords[joint[2]]))

    fingertips = [4, 8, 12, 16, 20]
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            features.append(float(np.linalg.norm(coords[fingertips[i]] - coords[fingertips[j]])))

    return features


def detect_pressed_fingers(pressing_map: dict, finger_note_map: dict) -> list:
    """
    Returns list of pressing fingers with their note(s).
    finger_note_map: FINGER_NOTE_RIGHT or FINGER_NOTE_LEFT based on detected handedness.
    note_label: joined string, e.g. 'Do/Fa' for fingers that cover 2 notes.
    """
    return [
        {
            "tip_idx":    tip_idx,
            "finger":     fname,
            "notes":      finger_note_map[tip_idx],
            "note_label": "/".join(finger_note_map[tip_idx]),
        }
        for tip_idx, fname in FINGER_INFO
        if pressing_map.get(tip_idx, False)
    ]


def extract_and_annotate(input_path: str, raw_out: str):
    """
    Pass 1: reads the input video frame by frame, runs MediaPipe hand detection,
    draws landmarks (with index labels) and fingertip markers onto each frame,
    and writes the result to raw_out using the mp4v codec.
    Returns a tuple of:
      - all_features: np.ndarray of shape (n_frames, FEATURE_SIZE)
      - finger_activity: list of per-frame pressed-finger dicts
    """
    cap = cv2.VideoCapture(input_path)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (width, height))

    all_features    = []
    finger_activity = []
    tracker         = FingertipPressTracker()

    with mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_lm  = results.multi_hand_landmarks[0]
                features = extract_frame_features(hand_lm)

                # Determine handedness: MediaPipe assumes a mirrored view, so for a
                # standard front-facing recording "Left" label = pianist's right hand.
                mp_label = (
                    results.multi_handedness[0].classification[0].label
                    if results.multi_handedness else "Left"
                )
                finger_note_map = FINGER_NOTE_RIGHT if mp_label == "Left" else FINGER_NOTE_LEFT
                hand_label_text = "R" if mp_label == "Left" else "L"

                mp_drawing.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    LANDMARK_SPEC,
                    CONNECTION_SPEC,
                )

                # Draw landmark index number next to each point
                for idx, lm in enumerate(hand_lm.landmark):
                    cx = int(lm.x * width)
                    cy = int(lm.y * height)
                    cv2.putText(frame, str(idx), (cx + 6, cy - 6),
                                cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

                pressing_map       = tracker.update(hand_lm)
                pressed_this_frame = detect_pressed_fingers(pressing_map, finger_note_map)
                pressing_set       = {p["tip_idx"] for p in pressed_this_frame}

                for tip_idx, fname in FINGER_INFO:
                    lm  = hand_lm.landmark[tip_idx]
                    cx  = int(lm.x * width)
                    cy  = int(lm.y * height)
                    pressing = tip_idx in pressing_set
                    color    = (0, 60, 255) if pressing else (0, 255, 255)
                    cv2.circle(frame, (cx, cy), 10, color, -1)
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2)
                    # Show note name(s) above fingertip when pressing
                    if pressing:
                        note_label = "/".join(finger_note_map[tip_idx])
                        cv2.putText(frame, note_label, (cx - 12, cy - 14),
                                    cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 60, 255), 1, cv2.LINE_AA)

                # Draw hand label (R / L) near the wrist
                wrist = hand_lm.landmark[0]
                wx, wy = int(wrist.x * width), int(wrist.y * height)
                cv2.putText(frame, hand_label_text, (wx - 18, wy + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

                # Draw a bounding box around the detected hand
                xs = [lm.x for lm in hand_lm.landmark]
                ys = [lm.y for lm in hand_lm.landmark]
                x1 = max(0, int(min(xs) * width)  - 15)
                y1 = max(0, int(min(ys) * height) - 15)
                x2 = min(width,  int(max(xs) * width)  + 15)
                y2 = min(height, int(max(ys) * height) + 15)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

                finger_activity.append(pressed_this_frame)

            else:
                features = [0.0] * FEATURE_SIZE
                finger_activity.append([])

            all_features.append(features)
            writer.write(frame)

    cap.release()
    writer.release()

    return np.array(all_features), finger_activity


def finalize_video(raw_path: str, final_path: str,
                   display: str, confidence: float, label: str) -> None:
    """
    Pass 2: reads the raw annotated video, draws a semi-transparent label overlay
    showing the classification result on every frame, then re-encodes to H.264
    using imageio-ffmpeg's bundled binary so the output is playable in browsers.
    H.264 requires even pixel dimensions, so frames are padded if necessary.
    """
    cap    = cv2.VideoCapture(raw_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    enc_w = width  + (width  % 2)
    enc_h = height + (height % 2)

    color  = LABEL_COLOR.get(label, (255, 255, 255))
    # Primary text: the human-readable class name (e.g. "Good", "Needs Improvement")
    text1  = display
    text2  = f"Confidence: {confidence:.1f}%"
    font   = cv2.FONT_HERSHEY_DUPLEX
    scale1 = max(1.0,  width / 720)
    scale2 = max(0.5,  width / 1300)
    thick1 = max(2, int(width / 400))
    thick2 = max(1, int(width / 900))
    pad    = 12
    box_h  = 72
    box_w  = max(320, int(width * 0.40))

    writer = iio.get_writer(
        final_path,
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        ffmpeg_log_level="quiet",
        output_params=[
            "-pix_fmt",  "yuv420p",
            "-movflags", "+faststart",
            "-preset",   "fast",
        ],
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Semi-transparent dark background pill behind the label text
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
        # Colored left-side accent bar matching the label
        cv2.rectangle(frame, (pad, pad), (pad + 5, pad + box_h), color, -1)
        # Class name in the label color, large and prominent
        cv2.putText(frame, text1, (pad + 14, pad + 44),
                    font, scale1, color, thick1, cv2.LINE_AA)
        # Confidence in a neutral light color beneath the class name
        cv2.putText(frame, text2, (pad + 14, pad + 64),
                    font, scale2, (200, 200, 200), thick2, cv2.LINE_AA)

        if enc_w != width or enc_h != height:
            frame = cv2.copyMakeBorder(frame, 0, enc_h - height, 0, enc_w - width,
                                       cv2.BORDER_CONSTANT, value=0)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.close()


def mux_audio(video_no_audio: str, audio_source: str, output_path: str) -> None:
    """
    Pass 3: copies the audio track from the original upload into the annotated video.
    Uses the FFmpeg binary bundled with imageio-ffmpeg, so no system FFmpeg is needed.
    Raises subprocess.CalledProcessError if the source has no audio track.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-i", video_no_audio,
            "-i", audio_source,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


def ensemble_predict(all_features: np.ndarray) -> dict:
    """
    Builds sliding-window sequences from the per-frame features, runs inference
    on each of the 5 loaded models, and averages their softmax outputs (soft voting).
    Returns the predicted label, confidence, and per-class probabilities.
    """
    sequences = []
    for i in range(0, len(all_features) - SEQUENCE_LENGTH + 1, STRIDE):
        sequences.append(all_features[i : i + SEQUENCE_LENGTH])

    # If the video is too short to fill one full window, pad with zeros
    if not sequences:
        padded = np.zeros((SEQUENCE_LENGTH, FEATURE_SIZE))
        n = min(len(all_features), SEQUENCE_LENGTH)
        padded[:n] = all_features[:n]
        sequences = [padded]

    X = np.array(sequences)

    all_probs = []
    per_model = []

    for idx, model in enumerate(models):
        preds = model.predict(X, verbose=0)
        avg   = np.mean(preds, axis=0)
        all_probs.append(avg)

        fold_label = LABEL_NAMES[int(np.argmax(avg))]
        per_model.append({
            "fold":          idx + 1,
            "label":         fold_label,
            "probabilities": {
                LABEL_NAMES[i]: round(float(avg[i]) * 100, 2)
                for i in range(3)
            },
        })

    ensemble_avg = np.mean(all_probs, axis=0)
    best_idx     = int(np.argmax(ensemble_avg))
    label        = LABEL_NAMES[best_idx]
    confidence   = float(ensemble_avg[best_idx])

    return {
        "label":         label,
        "display":       LABEL_DISPLAY[label],
        "confidence":    round(confidence * 100, 2),
        "probabilities": {
            LABEL_NAMES[i]: round(float(ensemble_avg[i]) * 100, 2)
            for i in range(3)
        },
        "per_model":     per_model,
        "model_count":   len(models),
    }


def build_analysis_summary(finger_activity: list, total_frames: int, detection_frames: int) -> dict:
    """
    Aggregates per-frame finger-press activity into a summary.
    finger_frames[name] = set of unique frame indices where that finger was pressing.
    Percentages use total_frames as denominator so they never exceed 100%.
    """
    finger_frames = {}   # fname → set of frame indices
    press_frames  = set()

    for frame_idx, frame_presses in enumerate(finger_activity):
        for press in frame_presses:
            key = f'{press["note_label"]} ({press["finger"]})'
            finger_frames.setdefault(key, set()).add(frame_idx)
            press_frames.add(frame_idx)

    detection_rate = round(detection_frames / total_frames * 100, 1) if total_frames > 0 else 0
    base           = total_frames if total_frames > 0 else 1

    return {
        "total_frames":     total_frames,
        "detection_frames": detection_frames,
        "detection_rate":   detection_rate,
        "press_frames":     len(press_frames),
        "finger_activity":  {
            f: round(len(frames) / base * 100, 1)
            for f, frames in finger_frames.items()
        },
    }


def build_explanation(label: str, confidence: float, per_model: list, analysis: dict) -> str:
    """
    Short, human-friendly explanation of why the video got its result.
    Max 4 sentences — one per topic: why, reliability, video quality, finger activity.
    """
    total      = analysis.get("total_frames", 0)
    detected   = analysis.get("detection_frames", 0)
    rate       = analysis.get("detection_rate", 0)
    press      = analysis.get("press_frames", 0)
    finger_act = analysis.get("finger_activity", {})
    n_models   = len(per_model)
    n_agree    = sum(1 for m in per_model if m["label"] == label)
    press_pct  = round(press / total * 100, 1) if total > 0 else 0

    sorted_fingers = sorted(finger_act.items(), key=lambda x: x[1], reverse=True)
    most_active    = sorted_fingers[0] if sorted_fingers else None

    sentences = []

    # 1 — Why this label (the core human reason)
    if label == "good":
        sentences.append(
            "The AI detected that your fingers were <strong>curled and pressing the keys "
            "with the fingertip</strong> in a consistent way throughout the video, "
            "and your wrist stayed relatively stable — which are the main indicators of good technique."
        )
    elif label == "needs_improvement":
        sentences.append(
            "The AI noticed <strong>some inconsistencies</strong> in your finger movement — "
            "for example, a finger occasionally pressing at an angle or the wrist shifting "
            "slightly more than expected. Your technique is on the right track; "
            "focused practice on consistency will move you to the next level."
        )
    else:
        sentences.append(
            "The AI frequently detected <strong>flat fingers</strong> — meaning your fingers "
            "were not curled enough when pressing the keys. "
            "Try to keep your fingers slightly curved (like gently holding a ball) "
            "and press each key with the very tip of your finger."
        )

    # 2 — How reliable is this result
    if n_agree == n_models:
        sentences.append(
            f"All <strong>{n_models} AI models</strong> agreed on this result "
            f"(confidence <strong>{confidence:.1f}%</strong>), so this prediction is very consistent."
        )
    elif n_agree >= n_models // 2 + 1:
        sentences.append(
            f"<strong>{n_agree} of {n_models} AI models</strong> agreed "
            f"(confidence <strong>{confidence:.1f}%</strong>)."
        )
    else:
        sentences.append(
            f"Only <strong>{n_agree} of {n_models} AI models</strong> agreed "
            f"(confidence <strong>{confidence:.1f}%</strong>) — treat this as an estimate."
        )

    # 3 — Video quality (only if not great)
    if rate < 90:
        if rate >= 60:
            sentences.append(
                f"The hand was visible in <strong>{rate}% of frames</strong> — "
                f"recording with better lighting may improve accuracy."
            )
        else:
            sentences.append(
                f"The hand was only visible in <strong>{rate}% of frames</strong>, "
                f"which is low. Try recording again with the hand clearly in frame and better lighting."
            )

    # 4 — Key press activity
    if most_active and press > 0:
        sentences.append(
            f"Key presses were detected in <strong>{press_pct}% of the video</strong>, "
            f"with <strong>{most_active[0]}</strong> being the most active finger "
            f"({most_active[1]}% of the video)."
        )
    elif press == 0:
        sentences.append(
            "No key presses were detected — make sure the hand is close enough to the camera."
        )

    return " ".join(sentences)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart form upload with field name 'video'.
    Runs the full three-pass pipeline (feature extraction, H.264 encoding, audio mux)
    and returns a JSON object with the classification result and annotated video filename.
    """
    if "video" not in request.files:
        return jsonify({"error": "Field 'video' not found in request."}), 400

    file = request.files["video"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXT).upper()}"}), 400

    uid       = uuid.uuid4().hex
    orig_stem = os.path.splitext(secure_filename(file.filename))[0]

    input_path  = os.path.join(UPLOAD_DIR, f"{uid}_input.mp4")
    raw_path    = os.path.join(UPLOAD_DIR, f"{uid}_raw.mp4")
    silent_path = os.path.join(UPLOAD_DIR, f"{uid}_silent.mp4")

    file.save(input_path)

    try:
        # Check 1: ensure OpenCV can open the file (not corrupt)
        cap_check = cv2.VideoCapture(input_path)
        if not cap_check.isOpened():
            cap_check.release()
            return jsonify({"error": "Video file could not be read. Make sure the file is not corrupted and the format is supported."}), 422

        vid_w      = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h      = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps    = cap_check.get(cv2.CAP_PROP_FPS)
        vid_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_check.release()

        # Check 2: video must have valid dimensions
        if vid_w == 0 or vid_h == 0:
            return jsonify({"error": "Video has invalid dimensions (width or height is 0)."}), 422

        # Check 3: video must be long enough to form at least one sequence
        if 0 < vid_frames < SEQUENCE_LENGTH:
            return jsonify({
                "error": f"Video is too short. At least {SEQUENCE_LENGTH} frames are required; "
                         f"this video only has {vid_frames} frames."
            }), 422

        all_features, finger_activity = extract_and_annotate(input_path, raw_path)

        if len(all_features) == 0:
            return jsonify({"error": "No frames could be extracted from the video."}), 422

        # Check 4: at least 1 frame must have a hand detected
        detection_frames = sum(1 for f in all_features if any(v != 0.0 for v in f))
        if detection_frames == 0:
            return jsonify({
                "error": "No hand was detected in the video. "
                         "Make sure the hand is clearly visible and well-lit."
            }), 422

        result = ensemble_predict(all_features)

        result["analysis"] = build_analysis_summary(
            finger_activity, len(all_features), detection_frames
        )

        # Short one-liner shown in the badge hover tooltip
        result["short_explanation"] = LABEL_EXPLANATION[result["label"]]

        # Dynamic plain-language explanation built from actual video data
        result["explanation"] = build_explanation(
            result["label"], result["confidence"],
            result["per_model"], result["analysis"]
        )

        # Output filename encodes both the original name and the predicted label
        final_name = f"{orig_stem}_{result['label']}.mp4"
        final_path = os.path.join(UPLOAD_DIR, final_name)

        finalize_video(raw_path, silent_path,
                       result["display"], result["confidence"], result["label"])
        os.remove(raw_path)

        try:
            mux_audio(silent_path, input_path, final_path)
            os.remove(silent_path)
        except Exception:
            # Source video has no audio track; use the silent version as-is
            os.rename(silent_path, final_path)

        result["annotated_video"] = final_name

        return jsonify(result), 200

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        gc.collect()


@app.route("/video/<filename>")
def serve_video(filename):
    """Serves an annotated video file from the uploads directory. Supports HTTP Range requests for seeking."""
    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.route("/health")
def health():
    """Returns a simple status response indicating how many models are loaded."""
    return jsonify({"status": "ok", "models_loaded": len(models)}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
