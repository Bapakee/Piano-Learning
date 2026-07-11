# ============================================================
# Flask API — Piano Playing Level Classifier
# ============================================================
# This file is the "brain" of the application.
# It receives a piano video from the web page, analyses each
# frame of the video, and returns a classification result
# (Good / Needs Improvement / Poor) along with a detailed
# explanation and an annotated video.
#
# How it works (step by step):
#   1. The user uploads a video through the web page.
#   2. This API reads the video frame by frame.
#   3. MediaPipe detects the hand and 21 key points (landmarks)
#      in each frame.
#   4. Features (angles, distances, coordinates) are extracted
#      from the landmarks.
#   5. Five trained AI models each make a prediction.
#   6. The results are averaged (ensemble) to get the final label.
#   7. An annotated video is generated and sent back together
#      with the JSON result.
#
# Endpoints:
#   POST /predict      — upload a video, get the classification result
#   GET  /video/<name> — download the annotated video
#   GET  /health       — check whether the API is running
# ============================================================


# ── Standard Python libraries ──────────────────────────────
import os           # file and folder operations
import gc           # free up memory after each request
import uuid         # generate unique random file names
import subprocess   # run external programs (ffmpeg for audio)
import warnings     # suppress non-critical warning messages
from collections import deque   # sliding window for velocity smoothing
warnings.filterwarnings("ignore")

# Hide TensorFlow startup messages — they are not needed in production
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Third-party libraries ──────────────────────────────────
import cv2                      # read / write / draw on video frames
import numpy as np              # math and array operations
import mediapipe as mp          # hand landmark detection
import tensorflow as tf         # load and run the LSTM models
import imageio.v2 as iio        # write the final H.264 video
import imageio_ffmpeg           # bundled ffmpeg binary (no system install needed)

from flask import Flask, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename   # sanitise uploaded file names


# ============================================================
# FILE PATHS
# ============================================================
# BASE_DIR  — the folder where this script lives
# MODEL_DIR — where the five trained AI model files are stored
# UPLOAD_DIR— temporary folder for uploaded and processed videos
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "Tugas Akhir Patrick",
                            "OUTPUT_PIANO_LSTM", "best_model")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")

# Only MP4 and MOV files are accepted
ALLOWED_EXT = {"mp4", "mov"}

# Maximum upload size in megabytes
MAX_MB      = 50

# Create the uploads folder if it does not exist yet
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================
# These values must exactly match what was used during model
# training — changing them would break the predictions.
#
# SEQUENCE_LENGTH — how many consecutive frames are fed to the
#                   AI at once (like a short video clip of 30 frames)
# STRIDE          — how many frames to skip before starting the
#                   next clip (avoids too much overlap)
# FEATURE_SIZE    — total number of numbers describing one frame:
#                   21 landmarks × 3 coordinates (x, y, z) = 63
#                   + 10 joint bend angles = 73
#                   + 10 fingertip-to-fingertip distances = 83
SEQUENCE_LENGTH = 30
STRIDE          = 5
FEATURE_SIZE    = 83

# ── Joint triplets for angle calculation ───────────────────
# Each triplet (a, b, c) defines three landmark points.
# The angle is measured at point b — this tells us how bent
# each finger joint is.
# Example: [2, 1, 3] measures the angle at landmark 1
# (the base of the thumb) using landmarks 2 and 3 as sides.
JOINTS = [
    [2,  1,  3], [3,  2,  4],     # thumb joints
    [6,  5,  7], [7,  6,  8],     # index finger joints
    [10, 9,  11],[11, 10, 12],    # middle finger joints
    [14, 13, 15],[15, 14, 16],    # ring finger joints
    [18, 17, 19],[19, 18, 20],    # pinky joints
]


# ============================================================
# LABEL DEFINITIONS
# ============================================================
# The three possible classification results and how they are
# displayed in the UI and coloured in the annotated video.

LABEL_NAMES   = ["good", "needs_improvement", "poor"]

# Human-readable display name for each label
LABEL_DISPLAY = {
    "good":             "Good",
    "needs_improvement":"Needs Improvement",
    "poor":             "Poor",
}

# BGR colours used to draw the result banner on the video
# (OpenCV uses Blue-Green-Red order, not Red-Green-Blue)
LABEL_COLOR = {
    "good":             (86,  205, 86),    # green
    "needs_improvement":(60,  180, 245),   # orange
    "poor":             (60,  60,  239),   # red
}


# ============================================================
# FINGER AND NOTE DEFINITIONS
# ============================================================
# Maps each fingertip landmark index (MediaPipe numbering) to
# its common finger name.
# Landmark 4  = Thumb tip
# Landmark 8  = Index finger tip
# Landmark 12 = Middle finger tip
# Landmark 16 = Ring finger tip
# Landmark 20 = Pinky tip
FINGER_INFO = [
    (4,  "Thumb"),
    (8,  "Index Finger"),
    (12, "Middle Finger"),
    (16, "Ring Finger"),
    (20, "Pinky"),
]

# ── C major scale note mapping (with thumb-crossing) ───────
# In a C major scale the thumb crosses under (right hand) or
# the fingers cross over (left hand) to play all 8 notes.
# Because of this some fingers cover TWO notes, shown as a list.
#
# IMPORTANT — MediaPipe assumes the camera is a selfie/mirror
# view. For a normal front-facing recording the labels are
# flipped: MediaPipe "Left" = the pianist's RIGHT hand.
FINGER_NOTE_RIGHT = {
    4:  ["Do", "Fa"],    # Thumb: plays C, then crosses under to play F
    8:  ["Re", "Sol"],   # Index: plays D, then G
    12: ["Mi", "La"],    # Middle: plays E, then A
    16: ["Si"],          # Ring: plays B
    20: ["Do"],          # Pinky: plays the upper C
}
FINGER_NOTE_LEFT = {
    4:  ["Sol", "Do"],   # Thumb: plays G, then crosses over to upper C
    8:  ["Fa", "Si"],    # Index: plays F, then B
    12: ["Mi", "La"],    # Middle: plays E, then A (after crossover)
    16: ["Re"],          # Ring: plays D
    20: ["Do"],          # Pinky: plays the lower C
}

# PIP = Proximal Interphalangeal joint (the middle knuckle of each finger).
# Used to check whether the fingertip is bent downward far enough
# to indicate a genuine key press.
_PIP = {4: 3, 8: 6, 12: 10, 16: 14, 20: 18}


# ============================================================
# FINGERTIP PRESS TRACKER
# ============================================================
class FingertipPressTracker:
    """
    Decides, frame by frame, whether each fingertip is pressing
    a piano key.

    Three signals are combined to make this decision:

    1. VELOCITY — how fast the fingertip is moving downward
       relative to the wrist.  A sudden downward burst = press.

    2. POSITION (permissive) — is the fingertip below its middle
       knuckle (PIP)?  This gate is relaxed so that flat-finger
       (Poor) players are still detected.

    3. STATIC PRESS — has the fingertip stayed below the middle
       knuckle for several frames in a row?  This catches slow,
       deliberate presses that are too gentle to trigger the
       velocity gate alone.

    Each finger runs its own small state machine with four
    states: IDLE → PRESSING → PRESSED → RELEASING → IDLE.
    This prevents a single noisy frame from being counted as
    a full press.
    """

    # Number of recent frames used to smooth velocity
    SMOOTH         = 3

    # Frames of downward movement required before calling it a press
    CONFIRM_FRAMES = 3

    # Minimum frames in PRESSED state before allowing a release
    # (prevents flickering on/off for the same press)
    HOLD_FRAMES    = 4

    # Frames the tip must stay below PIP to auto-confirm a slow press
    STATIC_FRAMES  = 6

    # MCP = Metacarpophalangeal joint (the large knuckle at the base of each finger)
    # Stored here for possible future use as a depth reference
    _MCP = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}

    def __init__(self):
        # Store the last 15 Y-positions for each fingertip and the wrist
        self.tip_y   = {idx: deque(maxlen=15) for idx, _ in FINGER_INFO}
        self.wrist_y = deque(maxlen=15)

        # Current state for each finger ('idle', 'pressing', 'pressed', 'releasing')
        self.states  = {idx: 'idle' for idx, _ in FINGER_INFO}

        # Counters used by the state machine
        self.press_f = {idx: 0 for idx, _ in FINGER_INFO}   # frames in pressing state
        self.hold_f  = {idx: 0 for idx, _ in FINGER_INFO}   # frames held in pressed state
        self.below_f = {idx: 0 for idx, _ in FINGER_INFO}   # consecutive frames tip below PIP

    def _velocity(self, buf) -> float:
        """
        Calculates the average speed of recent movement from a
        position buffer.  Positive = moving downward (pressing).
        Negative = moving upward (releasing).
        """
        h = list(buf)
        if len(h) < 2:
            return 0.0
        # Take only the last SMOOTH differences to reduce noise
        diffs = [h[i] - h[i-1] for i in range(max(1, len(h) - self.SMOOTH), len(h))]
        return float(np.mean(diffs)) if diffs else 0.0

    def _below_pip(self, tip_idx: int, hand_lm, pos_thr: float) -> bool:
        """
        Returns True when the fingertip Y-coordinate is below the
        middle knuckle (PIP) by at least pos_thr units.
        In image coordinates Y increases downward, so a larger Y
        means the tip is lower (closer to the keys).
        A negative pos_thr makes the gate permissive — the tip
        only needs to be within that distance ABOVE the PIP.
        """
        tip_y = hand_lm.landmark[tip_idx].y
        pip_y = hand_lm.landmark[_PIP[tip_idx]].y
        return (tip_y - pip_y) > pos_thr

    def update(self, hand_lm) -> dict:
        """
        Called once per frame with the detected hand landmarks.
        Returns a dictionary: {fingertip_index: True/False}
        True  = this finger is currently pressing a key.
        False = this finger is not pressing.

        Why the thresholds are what they are:
        ─────────────────────────────────────
        • hand_h (wrist-to-knuckle distance) varies a lot
          between close-up and wide-angle recordings (0.002–0.155).
          Using hand_h × factor alone makes press_thr near-zero
          for close-up videos.  A floor of 0.004 prevents this.

        • Players with Poor technique press with flat fingers —
          the tip can sit ABOVE the PIP joint.  A permissive
          pos_thr_vel (negative) ensures these players are still
          detected.

        • The strict pos_thr_static (small positive) is only used
          for the slow/static press check so random noise does not
          trigger an always-on false positive.
        """
        # Record the wrist position for this frame
        wrist_y = hand_lm.landmark[0].y
        self.wrist_y.append(wrist_y)
        dw = self._velocity(self.wrist_y)   # wrist vertical velocity (used to remove wrist drift)

        # Hand height = vertical distance from wrist to middle knuckle
        mid_mcp_y = hand_lm.landmark[9].y
        hand_h    = abs(wrist_y - mid_mcp_y) or 0.01   # guard against zero

        # Press threshold: how fast the fingertip must move downward to trigger a press.
        # Scaled to hand size but never below 0.004 (handles close-up cameras).
        press_thr = max(0.004, hand_h * 0.30)

        # Release threshold: a gentler version — only need to slow down, not reverse hard.
        rel_thr   = press_thr * 0.40

        # Permissive position gate: allows detection even when the finger is flat.
        # The fingertip may be up to half the hand height ABOVE the PIP joint.
        pos_thr_vel = -hand_h * 0.5

        # Strict position gate: for the slow-press check the tip must be
        # just below the PIP joint (a tiny margin above is not enough).
        pos_thr_static = hand_h * 0.01

        result = {}
        for tip_idx, _ in FINGER_INFO:
            # Record this fingertip's Y position
            y = hand_lm.landmark[tip_idx].y
            self.tip_y[tip_idx].append(y)

            # dt = raw fingertip velocity; dy_rel = velocity relative to wrist
            # (removing wrist drift so whole-hand movement is not counted as a press)
            dt      = self._velocity(self.tip_y[tip_idx])
            dy_rel  = dt - dw

            # Two position checks with different strictness levels
            down_vel    = self._below_pip(tip_idx, hand_lm, pos_thr_vel)     # permissive gate
            down_static = self._below_pip(tip_idx, hand_lm, pos_thr_static)  # strict gate

            # Count how many frames in a row the tip is below PIP (strict gate)
            if down_static:
                self.below_f[tip_idx] += 1
            else:
                self.below_f[tip_idx] = 0

            # If the tip has been below PIP for enough frames → treat as a slow press
            static_press = self.below_f[tip_idx] >= self.STATIC_FRAMES

            state = self.states[tip_idx]

            if state == 'idle':
                # Start a press if: fast downward movement (velocity gate, permissive position)
                # OR the finger has been stationary below PIP for long enough (static gate)
                if (dy_rel > press_thr and down_vel) or static_press:
                    state = 'pressing'
                    self.press_f[tip_idx] = 1

            elif state == 'pressing':
                self.press_f[tip_idx] += 1
                if self.press_f[tip_idx] >= self.CONFIRM_FRAMES and down_vel:
                    # Confirmed: finger has been pressing for enough frames
                    state = 'pressed'
                    self.hold_f[tip_idx] = 0
                elif dy_rel < -rel_thr and not down_vel:
                    # Movement reversed before confirmation → false alarm, go back to idle
                    state = 'idle'
                    self.press_f[tip_idx] = 0

            elif state == 'pressed':
                self.hold_f[tip_idx] += 1
                # Only consider releasing after HOLD_FRAMES to suppress flickering
                if self.hold_f[tip_idx] >= self.HOLD_FRAMES and not static_press:
                    if dy_rel < -rel_thr and not down_vel:
                        state = 'releasing'

            elif state == 'releasing':
                if (dy_rel > press_thr and down_vel) or static_press:
                    # Finger pressed down again before fully releasing
                    state = 'pressing'
                    self.press_f[tip_idx] = 1
                elif not down_vel and abs(dy_rel) < press_thr * 0.5:
                    # Finger is back to resting position → fully idle
                    state = 'idle'
                    self.press_f[tip_idx] = 0
                    self.below_f[tip_idx] = 0

            self.states[tip_idx] = state
            # A finger counts as "pressing" when in either the pressing or pressed state
            result[tip_idx] = state in ('pressing', 'pressed')

        return result


# ============================================================
# TOOLTIP EXPLANATIONS
# ============================================================
# Short, one-paragraph explanation shown in the ⓘ badge tooltip.
# Written for non-technical users — no AI jargon.
LABEL_EXPLANATION = {
    "good":
        "Your fingers are curled properly and most of the keys are pressed with the fingertip. "
        "Your hand stays stable throughout as well which is also signs of good technique usage.",
    "needs_improvement":
        "Your technique is on the right track, but the AI detected some inconsistencies in your finger movement "
        "— such as uneven finger pressure or slight wrist instability. Practice with a slower tempo is recommended to improve further.",
    "poor":
        "The AI has detected that your fingers are often flat (not curved) when pressing the keys. "
        "Try keeping your fingers slightly curved, like you are gently holding a ball, "
        "Try slower tempo, but make sure to press each key with the very tip of your finger.",
}


# ============================================================
# MEDIAPIPE SETUP
# ============================================================
# Load the MediaPipe hand-tracking solution and define how
# landmarks and connections are drawn on the video frames.
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Yellow dots for landmarks, white lines for connections
LANDMARK_SPEC   = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)


# ============================================================
# LOAD AI MODELS
# ============================================================
# All five LSTM models (one per cross-validation fold) are
# loaded into memory once at startup.  Keeping them in memory
# makes every subsequent prediction fast.
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


# ============================================================
# FEATURE EXTRACTION HELPERS
# ============================================================

def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) at point b formed by
    the line segments b→a and b→c.

    Used to measure how bent each finger joint is:
      0°  = completely straight finger
      90° = finger bent at a right angle
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    # Clamp to [-1, 1] to avoid floating-point errors in arccos
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_frame_features(hand_landmarks):
    """
    Converts the 21 hand landmarks detected by MediaPipe into
    the 83-number feature vector used by the LSTM models.

    Breakdown:
      • 63 numbers — raw (x, y, z) coordinates of all 21 landmarks
      • 10 numbers — bend angles of the 10 finger joints
      • 10 numbers — distances between pairs of fingertips
                     (e.g. thumb-to-index, thumb-to-middle, …)
    Total: 63 + 10 + 10 = 83
    """
    features = []
    coords   = []

    # Step 1: collect raw coordinates
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
        features.extend([lm.x, lm.y, lm.z])   # adds 63 numbers

    coords = np.array(coords)

    # Step 2: compute bend angles for each joint triplet (adds 10 numbers)
    for joint in JOINTS:
        features.append(calculate_angle(coords[joint[0]], coords[joint[1]], coords[joint[2]]))

    # Step 3: compute distances between every pair of fingertips
    # There are C(5,2) = 10 unique pairs (adds 10 numbers)
    fingertips = [4, 8, 12, 16, 20]
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            features.append(float(np.linalg.norm(coords[fingertips[i]] - coords[fingertips[j]])))

    return features   # list of 83 float values


def detect_pressed_fingers(pressing_map: dict, finger_note_map: dict) -> list:
    """
    Translates the {tip_index: True/False} map from the tracker
    into a human-readable list of which fingers are pressing
    and which notes they are playing.

    Returns a list of dicts, one entry per pressing finger.
    """
    return [
        {
            "tip_idx":    tip_idx,
            "finger":     fname,
            "notes":      finger_note_map[tip_idx],
            "note_label": "/".join(finger_note_map[tip_idx]),   # e.g. "Do/Fa"
        }
        for tip_idx, fname in FINGER_INFO
        if pressing_map.get(tip_idx, False)
    ]


# ============================================================
# VIDEO PROCESSING — PASS 1: EXTRACT FEATURES AND ANNOTATE
# ============================================================

def extract_and_annotate(input_path: str, raw_out: str):
    """
    Reads the uploaded video frame by frame and does two things
    at the same time:

    1. FEATURE EXTRACTION — extracts the 83 numbers per frame
       that describe the hand's shape and posture.

    2. ANNOTATION — draws on each frame:
       • Yellow dots and white lines for all 21 landmarks
       • Landmark index numbers (0–20) next to each dot
       • A coloured circle at each fingertip:
           Yellow = not pressing   Red = pressing a key
       • The note name (e.g. "Do") above a pressing fingertip
       • R / L label near the wrist to indicate handedness
       • A light bounding box around the detected hand

    The annotated frames are saved to raw_out (mp4v codec,
    not yet browser-compatible — that happens in Pass 2).

    Returns:
      all_features    — numpy array of shape (n_frames, 83)
      finger_activity — list with one entry per frame;
                        each entry lists which fingers were pressing
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

    # Open the MediaPipe hand detector
    # max_num_hands=1: only track one hand at a time (the pianist's playing hand)
    with mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break   # end of video

            # MediaPipe requires RGB input; OpenCV reads BGR by default
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                # A hand was detected in this frame
                hand_lm  = results.multi_hand_landmarks[0]
                features = extract_frame_features(hand_lm)

                # Determine which hand this is.
                # MediaPipe labels assume a mirror/selfie camera, so for a normal
                # front-facing recording the labels are reversed:
                #   MediaPipe "Left"  → pianist's RIGHT hand
                #   MediaPipe "Right" → pianist's LEFT hand
                mp_label = (
                    results.multi_handedness[0].classification[0].label
                    if results.multi_handedness else "Left"
                )
                finger_note_map = FINGER_NOTE_RIGHT if mp_label == "Left" else FINGER_NOTE_LEFT
                hand_label_text = "R" if mp_label == "Left" else "L"

                # Draw the standard MediaPipe skeleton (dots + lines)
                mp_drawing.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    LANDMARK_SPEC,
                    CONNECTION_SPEC,
                )

                # Draw the landmark index number next to each dot
                for idx, lm in enumerate(hand_lm.landmark):
                    cx = int(lm.x * width)
                    cy = int(lm.y * height)
                    cv2.putText(frame, str(idx), (cx + 6, cy - 6),
                                cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

                # Run the press tracker and get which fingers are pressing
                pressing_map       = tracker.update(hand_lm)
                pressed_this_frame = detect_pressed_fingers(pressing_map, finger_note_map)
                pressing_set       = {p["tip_idx"] for p in pressed_this_frame}

                # Draw a coloured circle at each fingertip
                for tip_idx, fname in FINGER_INFO:
                    lm  = hand_lm.landmark[tip_idx]
                    cx  = int(lm.x * width)
                    cy  = int(lm.y * height)
                    pressing = tip_idx in pressing_set
                    # Red (BGR: 0,60,255) = pressing; Yellow (BGR: 0,255,255) = not pressing
                    color    = (0, 60, 255) if pressing else (0, 255, 255)
                    cv2.circle(frame, (cx, cy), 10, color, -1)            # filled circle
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2)         # black outline

                    # Show note name above the fingertip when it is pressing
                    if pressing:
                        note_label = "/".join(finger_note_map[tip_idx])
                        cv2.putText(frame, note_label, (cx - 12, cy - 14),
                                    cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 60, 255), 1, cv2.LINE_AA)

                # Draw R / L near the wrist to show which hand is detected
                wrist = hand_lm.landmark[0]
                wx, wy = int(wrist.x * width), int(wrist.y * height)
                cv2.putText(frame, hand_label_text, (wx - 18, wy + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

                # Draw a bounding box around the hand (padding of 15 pixels)
                xs = [lm.x for lm in hand_lm.landmark]
                ys = [lm.y for lm in hand_lm.landmark]
                x1 = max(0, int(min(xs) * width)  - 15)
                y1 = max(0, int(min(ys) * height) - 15)
                x2 = min(width,  int(max(xs) * width)  + 15)
                y2 = min(height, int(max(ys) * height) + 15)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

                finger_activity.append(pressed_this_frame)

            else:
                # No hand detected in this frame — use zeroes as placeholder features
                features = [0.0] * FEATURE_SIZE
                finger_activity.append([])

            all_features.append(features)
            writer.write(frame)   # save this annotated frame

    cap.release()
    writer.release()

    return np.array(all_features), finger_activity


# ============================================================
# VIDEO PROCESSING — PASS 2: ADD RESULT BANNER + H.264 ENCODE
# ============================================================

def finalize_video(raw_path: str, final_path: str,
                   display: str, confidence: float, label: str) -> None:
    """
    Reads the raw annotated video from Pass 1 and overlays a
    semi-transparent result banner (e.g. "Good  Confidence: 91.3%")
    on every frame.

    Then re-encodes the video to H.264 (libx264) so it plays
    in all modern browsers.  H.264 requires the width and height
    to be even numbers — any odd-dimension video is padded by 1px.

    The bundled imageio-ffmpeg binary is used, so no separate
    FFmpeg installation is required on the machine.
    """
    cap    = cv2.VideoCapture(raw_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure even dimensions for H.264 compatibility
    enc_w = width  + (width  % 2)
    enc_h = height + (height % 2)

    color  = LABEL_COLOR.get(label, (255, 255, 255))
    text1  = display                          # e.g. "Good"
    text2  = f"Confidence: {confidence:.1f}%" # e.g. "Confidence: 91.3%"

    # Scale font size and thickness proportionally to video width
    font   = cv2.FONT_HERSHEY_DUPLEX
    scale1 = max(1.0,  width / 720)
    scale2 = max(0.5,  width / 1300)
    thick1 = max(2, int(width / 400))
    thick2 = max(1, int(width / 900))
    pad    = 12
    box_h  = 72
    box_w  = max(320, int(width * 0.40))

    # Use imageio-ffmpeg to write directly in H.264 format
    writer = iio.get_writer(
        final_path,
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        ffmpeg_log_level="quiet",
        output_params=[
            "-pix_fmt",  "yuv420p",     # widely compatible pixel format
            "-movflags", "+faststart",  # allows streaming before download completes
            "-preset",   "fast",        # balance between encoding speed and file size
        ],
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw a semi-transparent dark rectangle as the banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)   # 60% dark, 40% original

        # Coloured left-side accent bar (green / orange / red depending on label)
        cv2.rectangle(frame, (pad, pad), (pad + 5, pad + box_h), color, -1)

        # Large label text in the label colour
        cv2.putText(frame, text1, (pad + 14, pad + 44),
                    font, scale1, color, thick1, cv2.LINE_AA)

        # Smaller confidence text in neutral light grey
        cv2.putText(frame, text2, (pad + 14, pad + 64),
                    font, scale2, (200, 200, 200), thick2, cv2.LINE_AA)

        # Pad to even dimensions if necessary
        if enc_w != width or enc_h != height:
            frame = cv2.copyMakeBorder(frame, 0, enc_h - height, 0, enc_w - width,
                                       cv2.BORDER_CONSTANT, value=0)

        # Convert BGR → RGB before passing to imageio
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.close()


# ============================================================
# VIDEO PROCESSING — PASS 3: RE-ATTACH ORIGINAL AUDIO
# ============================================================

def mux_audio(video_no_audio: str, audio_source: str, output_path: str) -> None:
    """
    Copies the original audio track from the uploaded video
    into the annotated video file.

    This is the final step so the viewer can hear the piano
    while watching the annotated playback.

    Uses the FFmpeg binary bundled with imageio-ffmpeg —
    no system-level FFmpeg installation is required.

    Raises CalledProcessError if the source has no audio track
    (silent video); the caller catches this and keeps the
    silent version instead.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-i", video_no_audio,   # annotated video (no audio)
            "-i", audio_source,     # original upload (source of audio)
            "-c:v", "copy",         # copy video stream as-is (no re-encode)
            "-c:a", "aac",          # encode audio to AAC (browser-compatible)
            "-map", "0:v:0",        # take video from first input
            "-map", "1:a:0",        # take audio from second input
            "-shortest",            # end at the shorter of the two streams
            output_path,
        ],
        check=True,
        capture_output=True,
    )


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(all_features: np.ndarray) -> dict:
    """
    Runs the five AI models on the extracted features and
    combines their predictions by averaging (soft voting).

    How it works:
    1. The per-frame features are grouped into overlapping clips
       of SEQUENCE_LENGTH (30) frames, stepping STRIDE (5) frames
       at a time — like a sliding window across the video.
    2. Each model predicts a probability for each class on
       every clip.
    3. The probabilities are averaged across all clips and
       then across all five models.
    4. The class with the highest average probability wins.

    If the video is shorter than 30 frames, it is zero-padded
    so at least one sequence can be formed.
    """
    # Build sliding-window sequences from the feature array
    sequences = []
    for i in range(0, len(all_features) - SEQUENCE_LENGTH + 1, STRIDE):
        sequences.append(all_features[i : i + SEQUENCE_LENGTH])

    # If video is too short for even one full window, pad with zeroes
    if not sequences:
        padded = np.zeros((SEQUENCE_LENGTH, FEATURE_SIZE))
        n = min(len(all_features), SEQUENCE_LENGTH)
        padded[:n] = all_features[:n]
        sequences = [padded]

    X = np.array(sequences)   # shape: (n_sequences, 30, 83)

    all_probs = []   # averaged probability vector from each model
    per_model = []   # per-model label and probabilities (for the explanation text)

    for idx, model in enumerate(models):
        preds = model.predict(X, verbose=0)   # shape: (n_sequences, 3)
        avg   = np.mean(preds, axis=0)         # average across all sequences → shape (3,)
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

    # Average across all five models → final ensemble probability
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


# ============================================================
# ANALYSIS SUMMARY
# ============================================================

def build_analysis_summary(finger_activity: list, total_frames: int, detection_frames: int) -> dict:
    """
    Summarises finger-press activity across the entire video.

    For each finger it counts how many unique frames that finger
    was detected as pressing, then converts that to a percentage
    of the total frame count.

    Returns a dictionary used both by the web page (detection
    stats card) and by build_explanation() to personalise the
    result text.
    """
    finger_frames = {}   # key: "Note (FingerName)", value: set of frame indices
    press_frames  = set()  # all frames where any key press was detected

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


# ============================================================
# HUMAN-FRIENDLY EXPLANATION BUILDER
# ============================================================

def build_explanation(label: str, confidence: float, per_model: list, analysis: dict) -> str:
    """
    Generates a personalised, plain-language explanation of why
    the video received its classification label.

    Structure (up to 4 sentences):
      1. Why this label  — the main technical reason in simple words
      2. Reliability     — how many models agreed and the confidence %
      3. Video quality   — only included when detection rate is below 90%
      4. Finger activity — which finger was most active and how often
    """
    total      = analysis.get("total_frames", 0)
    detected   = analysis.get("detection_frames", 0)
    rate       = analysis.get("detection_rate", 0)
    press      = analysis.get("press_frames", 0)
    finger_act = analysis.get("finger_activity", {})
    n_models   = len(per_model)
    n_agree    = sum(1 for m in per_model if m["label"] == label)
    press_pct  = round(press / total * 100, 1) if total > 0 else 0

    # Find the most active finger (highest press percentage)
    sorted_fingers = sorted(finger_act.items(), key=lambda x: x[1], reverse=True)
    most_active    = sorted_fingers[0] if sorted_fingers else None

    sentences = []

    # ── Sentence 1: Why this label ──────────────────────────
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
    else:  # poor
        sentences.append(
            "The AI frequently detected <strong>flat fingers</strong> — meaning your fingers "
            "were not curled enough when pressing the keys. "
            "Try to keep your fingers slightly curved (like gently holding a ball) "
            "and press each key with the very tip of your finger."
        )

    # ── Sentence 2: How reliable is this result ─────────────
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

    # ── Sentence 3: Video quality warning (only if poor) ────
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

    # ── Sentence 4: Key press activity ──────────────────────
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


# ============================================================
# FLASK APPLICATION
# ============================================================
app = Flask(__name__)

# Enforce the maximum upload size at the Flask layer as well
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    """Returns True if the file extension is in the allowed set (mp4, mov)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── /predict endpoint ───────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Main endpoint — receives a video, runs the full pipeline,
    and returns the classification result as JSON.

    Pipeline:
      1. Validate the uploaded file (format, size, readability)
      2. Pass 1 — extract features and generate raw annotated video
      3. Run ensemble prediction on the extracted features
      4. Pass 2 — add result banner and encode to H.264
      5. Pass 3 — re-attach original audio
      6. Return JSON with label, confidence, analysis, and video filename

    All temporary files are cleaned up in the finally block.
    """
    if "video" not in request.files:
        return jsonify({"error": "Field 'video' not found in request."}), 400

    file = request.files["video"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXT).upper()}"}), 400

    # Generate a unique ID for this request to avoid filename collisions
    uid       = uuid.uuid4().hex
    orig_stem = os.path.splitext(secure_filename(file.filename))[0]

    # Temporary file paths used during the three-pass pipeline
    input_path  = os.path.join(UPLOAD_DIR, f"{uid}_input.mp4")   # the raw upload
    raw_path    = os.path.join(UPLOAD_DIR, f"{uid}_raw.mp4")      # Pass 1 output
    silent_path = os.path.join(UPLOAD_DIR, f"{uid}_silent.mp4")   # Pass 2 output (no audio)

    file.save(input_path)

    try:
        # ── Validation checks ──────────────────────────────
        # Check 1: can OpenCV actually open this file?
        cap_check = cv2.VideoCapture(input_path)
        if not cap_check.isOpened():
            cap_check.release()
            return jsonify({"error": "Video file could not be read. Make sure the file is not corrupted and the format is supported."}), 422

        vid_w      = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h      = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps    = cap_check.get(cv2.CAP_PROP_FPS)
        vid_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_check.release()

        # Check 2: video dimensions must be non-zero
        if vid_w == 0 or vid_h == 0:
            return jsonify({"error": "Video has invalid dimensions (width or height is 0)."}), 422

        # Check 3: video must be long enough for at least one LSTM sequence
        if 0 < vid_frames < SEQUENCE_LENGTH:
            return jsonify({
                "error": f"Video is too short. At least {SEQUENCE_LENGTH} frames are required; "
                         f"this video only has {vid_frames} frames."
            }), 422

        # ── Pass 1: feature extraction + annotation ────────
        all_features, finger_activity = extract_and_annotate(input_path, raw_path)

        if len(all_features) == 0:
            return jsonify({"error": "No frames could be extracted from the video."}), 422

        # Check 4: at least one frame must contain a detected hand
        detection_frames = sum(1 for f in all_features if any(v != 0.0 for v in f))
        if detection_frames == 0:
            return jsonify({
                "error": "No hand was detected in the video. "
                         "Make sure the hand is clearly visible and well-lit."
            }), 422

        # ── Ensemble prediction ────────────────────────────
        result = ensemble_predict(all_features)

        # ── Build analysis summary ─────────────────────────
        result["analysis"] = build_analysis_summary(
            finger_activity, len(all_features), detection_frames
        )

        # Short one-liner for the ⓘ badge tooltip
        result["short_explanation"] = LABEL_EXPLANATION[result["label"]]

        # Longer personalised explanation for the "Why is the result?" panel
        result["explanation"] = build_explanation(
            result["label"], result["confidence"],
            result["per_model"], result["analysis"]
        )

        # Final video filename encodes the original name + predicted label
        final_name = f"{orig_stem}_{result['label']}.mp4"
        final_path = os.path.join(UPLOAD_DIR, final_name)

        # ── Pass 2: add banner + H.264 encode ─────────────
        finalize_video(raw_path, silent_path,
                       result["display"], result["confidence"], result["label"])
        os.remove(raw_path)   # raw intermediate no longer needed

        # ── Pass 3: re-attach audio ────────────────────────
        try:
            mux_audio(silent_path, input_path, final_path)
            os.remove(silent_path)
        except Exception:
            # Video has no audio track → keep the silent version as the final output
            os.rename(silent_path, final_path)

        result["annotated_video"] = final_name

        return jsonify(result), 200

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    finally:
        # Always clean up the original upload to save disk space
        if os.path.exists(input_path):
            os.remove(input_path)
        gc.collect()   # free memory held by numpy arrays and model outputs


# ── /video/<filename> endpoint ──────────────────────────────
@app.route("/video/<filename>")
def serve_video(filename):
    """
    Serves an annotated video file so the browser can play it.
    Supports HTTP Range requests, which allows the browser to
    seek to any position in the video without re-downloading it.
    """
    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="video/mp4", conditional=True)


# ── /health endpoint ────────────────────────────────────────
@app.route("/health")
def health():
    """
    Simple health check endpoint.
    Returns {"status": "ok", "models_loaded": N} so operators
    can verify the API is up and all models were loaded.
    """
    return jsonify({"status": "ok", "models_loaded": len(models)}), 200


# ── Start the server ────────────────────────────────────────
if __name__ == "__main__":
    # host="0.0.0.0" makes the API accessible from other machines on the same network.
    # debug=False keeps the server stable in production (no auto-reload, no debugger).
    app.run(host="0.0.0.0", port=5000, debug=False)
