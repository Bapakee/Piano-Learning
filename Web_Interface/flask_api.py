# ============================================================
# Flask API — Piano Playing Level Classifier
# ============================================================
# This file is the "brain" of the application.
#
# What it does in simple terms:
#   The user uploads a piano video from the web page.
#   This program watches the video, detects the player's hand,
#   and decides whether their technique is Good, Needs Improvement,
#   or Poor.  It also creates an annotated version of the video
#   showing which fingers are pressing which keys.
#
# Step-by-step flow:
#   1. User uploads a video through the web page.
#   2. This program reads the video one image (frame) at a time.
#   3. A tool called MediaPipe finds the hand and marks 21 points
#      on it (fingertips, knuckles, wrist) in each frame.
#   4. Numbers describing the hand shape are calculated from
#      those 21 points (angles, distances, positions).
#   5. Five AI models each guess the skill level.
#   6. The five guesses are averaged to get the final answer.
#   7. An annotated video is created and sent back to the page.
#
# Web addresses (endpoints) this program listens to:
#   POST /predict      — upload a video, get the result
#   GET  /video/<name> — download the annotated video
#   GET  /health       — check if the program is running
# ============================================================


# ── Tools this program uses ────────────────────────────────
import os           # work with files and folders
import gc           # free up memory after each request
import uuid         # create unique random file names so files don't overwrite each other
import subprocess   # run other programs (used to handle audio with ffmpeg)
import warnings     # hide unimportant warning messages
from collections import deque   # a list that automatically removes old items when it gets too long
warnings.filterwarnings("ignore")

# Hide TensorFlow startup messages — they are not useful here
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Extra tools that must be installed separately ──────────
import cv2                      # read, write, and draw on video frames
import numpy as np              # do math on lists of numbers quickly
import mediapipe as mp          # detect hands and their key points
import tensorflow as tf         # load and run the trained AI models
import imageio.v2 as iio        # save the final video in a web-friendly format
import imageio_ffmpeg           # a built-in video converter (no need to install ffmpeg separately)

from flask import Flask, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename   # make uploaded file names safe to use


# ============================================================
# FOLDER LOCATIONS
# ============================================================
# Where to find the AI model files and where to save uploaded videos.
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "Tugas Akhir Patrick",
                            "OUTPUT_PIANO_LSTM", "best_model")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")

# Only these video formats are accepted
ALLOWED_EXT = {"mp4", "mov"}

# Maximum video file size the user can upload (in megabytes)
MAX_MB      = 50

# Create the uploads folder automatically if it does not exist yet
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# AI MODEL SETTINGS
# ============================================================
# These numbers must match exactly what was used when training the AI.
# Changing them would break the predictions.
#
# SEQUENCE_LENGTH — how many frames the AI looks at in one go
#                   (like watching a 1-second clip at 30fps)
# STRIDE          — how many frames to skip before the next clip starts
#                   (so clips overlap and nothing is missed)
# FEATURE_SIZE    — how many numbers describe one frame:
#                   21 hand points × 3 coordinates (x, y, z) = 63
#                   + 10 finger bend angles                   = 73
#                   + 10 distances between fingertips         = 83
SEQUENCE_LENGTH = 30
STRIDE          = 5
FEATURE_SIZE    = 83

# ── Which three landmark points to use for each finger bend angle ──
# Each row has three landmark numbers: [vertex_joint, side_a, side_b].
# The angle is measured at the FIRST number (the vertex / actual joint),
# using the other two landmarks as the two sides of the angle.
# Example: [2, 1, 3] measures the angle at landmark 2,
# using landmark 1 and landmark 3 as the two sides.
JOINTS = [
    [2,  1,  3], [3,  2,  4],     # thumb
    [6,  5,  7], [7,  6,  8],     # index finger
    [10, 9,  11],[11, 10, 12],    # middle finger
    [14, 13, 15],[15, 14, 16],    # ring finger
    [18, 17, 19],[19, 18, 20],    # pinky
]


# ============================================================
# SKILL LEVEL LABELS
# ============================================================
# The three possible results the AI can give.

LABEL_NAMES   = ["good", "needs_improvement", "poor"]

# The text shown on screen for each result
LABEL_DISPLAY = {
    "good":             "Good",
    "needs_improvement":"Needs Improvement",
    "poor":             "Poor",
}

# The colour of the result banner drawn on the video
# (OpenCV uses Blue-Green-Red order instead of the usual Red-Green-Blue)
LABEL_COLOR = {
    "good":             (86,  205, 86),    # green
    "needs_improvement":(60,  180, 245),   # orange
    "poor":             (60,  60,  239),   # red
}


# ============================================================
# FINGER NAMES AND PIANO NOTE NAMES
# ============================================================
# Each fingertip has a number in MediaPipe's system.
# This table connects those numbers to finger names.
#   Landmark 4  = Thumb tip
#   Landmark 8  = Index finger tip
#   Landmark 12 = Middle finger tip
#   Landmark 16 = Ring finger tip
#   Landmark 20 = Pinky tip
FINGER_INFO = [
    (4,  "Thumb"),
    (8,  "Index Finger"),
    (12, "Middle Finger"),
    (16, "Ring Finger"),
    (20, "Pinky"),
]

# ── Which piano note each finger plays (C major scale) ─────
# Some fingers can play two notes because the thumb crosses
# under the hand (right hand) or fingers cross over (left hand).
# Those fingers are listed with two notes, e.g. ["Do", "Fa"].
#
# Note: MediaPipe labels hands as if the camera is a mirror.
# For a normal front-facing camera the labels are flipped:
#   MediaPipe says "Left"  → it is actually the RIGHT hand.
#   MediaPipe says "Right" → it is actually the LEFT hand.
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

# PIP = the middle knuckle of each finger (the one in the middle of the finger).
# The program checks whether the fingertip is lower than this knuckle
# to decide if a key is being pressed.
_PIP = {4: 3, 8: 6, 12: 10, 16: 14, 20: 18}


# ============================================================
# KEY PRESS DETECTOR
# ============================================================
class FingertipPressTracker:
    """
    Watches the hand in each video frame and decides whether
    each finger is pressing a piano key.

    A key press is detected from the fingertip's DOWNWARD MOTION,
    not from its static position. This matters because players with
    good (curled) technique keep their fingertips permanently below
    the middle knuckle, so a position-only check would wrongly mark
    every finger as "pressing" all the time.

    Two checks are combined:

    1. SPEED CHECK — Is the fingertip actively moving downward fast
       enough (relative to the wrist)? A sudden downward movement is
       what actually signals a key press.

    2. POSITION CHECK — Is the fingertip roughly at or below the
       middle knuckle? A loose gate so flat-finger players still count.

    Each finger goes through three stages:
      IDLE → PRESSING → PRESSED → (finger lifts up) → IDLE
    A press starts on a downward motion, is held (latched) through the
    whole keystroke, and is released only when the finger clearly moves
    back up. This prevents a single shaky frame from being counted.
    """

    # How many recent frames to use when calculating finger speed
    SMOOTH         = 3

    # How many frames of downward movement before confirming a press
    CONFIRM_FRAMES = 2

    # Minimum DOWNWARD speed (normalised units) to START a press, and minimum
    # UPWARD speed to RELEASE it. These correspond to the descending velocity of
    # a key depression and the ascending velocity of a key release described for
    # a keystroke cycle in Oku & Furuya (2022), Sensors 22(13):4891.
    # Lower = more sensitive; higher = stricter.
    PRESS_VEL      = 0.0015
    RELEASE_VEL    = 0.0015

    # The large knuckle at the base of each finger (kept for possible future use)
    _MCP = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}

    def __init__(self):
        # Remember the last 15 vertical positions of each fingertip and the wrist
        self.tip_y   = {idx: deque(maxlen=15) for idx, _ in FINGER_INFO}
        self.wrist_y = deque(maxlen=15)

        # The current stage of each finger (idle / pressing / pressed / releasing)
        self.states  = {idx: 'idle' for idx, _ in FINGER_INFO}

        # Counter tracking how long each finger has been in the "pressing" stage
        self.press_f = {idx: 0 for idx, _ in FINGER_INFO}

    def _velocity(self, buf) -> float:
        """
        Calculates how fast something is moving up or down.
        A positive number means moving downward (toward the keys).
        A negative number means moving upward (lifting off).
        Uses only the most recent few positions to reduce noise.
        """
        h = list(buf)
        if len(h) < 2:
            return 0.0
        diffs = [h[i] - h[i-1] for i in range(max(1, len(h) - self.SMOOTH), len(h))]
        return float(np.mean(diffs)) if diffs else 0.0

    def _below_pip(self, tip_idx: int, hand_lm, pos_thr: float) -> bool:
        """
        Returns True when the fingertip is low enough compared to
        the middle knuckle of the same finger.

        In video coordinates, Y increases downward (0 = top of screen).
        So a bigger Y value means the point is lower on screen (closer to the keys).

        pos_thr is a margin: a negative value makes the check loose
        (the tip only needs to be close to the knuckle level, not below it).
        """
        tip_y = hand_lm.landmark[tip_idx].y
        pip_y = hand_lm.landmark[_PIP[tip_idx]].y
        return (tip_y - pip_y) > pos_thr

    def update(self, hand_lm) -> dict:
        """
        Called once for every video frame.
        Looks at where all the fingers are and decides which ones
        are currently pressing a key.

        Returns a dictionary: {finger_number: True or False}
          True  = this finger is pressing a key right now.
          False = this finger is not pressing.

        How the keystroke is modelled (Oku & Furuya, 2022):
        ─────────────────────────────────────────────
        • A press STARTS on a downward (descending) motion of the fingertip,
          not on a static low position — curled, good-technique fingertips sit
          permanently below the knuckle and would otherwise always look pressed.

        • Once pressed, the finger is LATCHED (kept red) through the hold phase,
          where the fingertip is down but barely moving (velocity ≈ 0).

        • The press is RELEASED only on an upward (ascending) motion — the moment
          the finger lifts off the key.

        • A loose position gate (pos_thr_vel, negative) still requires the
          fingertip to be roughly at/below the knuckle, so flat-finger (Poor)
          players are covered while an obviously-raised finger is not.
        """
        # Save the wrist's vertical position for this frame
        wrist_y = hand_lm.landmark[0].y
        self.wrist_y.append(wrist_y)
        dw = self._velocity(self.wrist_y)   # how fast the whole wrist is moving (used to cancel wrist drift)

        # Hand height = vertical distance from wrist to the middle finger's base knuckle
        mid_mcp_y = hand_lm.landmark[9].y
        hand_h    = abs(wrist_y - mid_mcp_y) or 0.01   # never let this be zero

        # Loose position check: the fingertip just needs to be near the knuckle level
        # (allows detection of flat-finger players)
        pos_thr_vel = -hand_h * 0.5

        result = {}
        for tip_idx, _ in FINGER_INFO:
            # Save this fingertip's vertical position for this frame
            y = hand_lm.landmark[tip_idx].y
            self.tip_y[tip_idx].append(y)

            # dy_rel = fingertip speed minus wrist speed. This removes any
            # up-and-down movement of the whole hand, so only the finger's own
            # motion is counted.
            #   dy_rel > 0  → descending (pressing down)
            #   dy_rel < 0  → ascending (lifting up)
            dt      = self._velocity(self.tip_y[tip_idx])
            dy_rel  = dt - dw

            # Loose gate: is the fingertip roughly at or below the knuckle level?
            down_vel = self._below_pip(tip_idx, hand_lm, pos_thr_vel)

            state = self.states[tip_idx]

            if state == 'idle':
                # Start a press on a clear downward (descending) motion.
                # No static "held-down" trigger → a curled, still finger stays yellow.
                if dy_rel > self.PRESS_VEL and down_vel:
                    state = 'pressing'
                    self.press_f[tip_idx] = 1

            elif state == 'pressing':
                self.press_f[tip_idx] += 1
                if self.press_f[tip_idx] >= self.CONFIRM_FRAMES:
                    # Downward motion held for enough frames — confirm as pressed
                    state = 'pressed'
                elif dy_rel < -self.RELEASE_VEL:
                    # Moved back up before confirmation → just a wobble, not a press
                    state = 'idle'
                    self.press_f[tip_idx] = 0

            elif state == 'pressed':
                # LATCH: stay pressed through the hold phase (velocity ≈ 0);
                # release only on a clear upward (ascending) motion — the key lift.
                if dy_rel < -self.RELEASE_VEL:
                    state = 'idle'
                    self.press_f[tip_idx] = 0

            self.states[tip_idx] = state
            # A finger is counted as "pressing" when it is in the pressing or pressed stage
            result[tip_idx] = state in ('pressing', 'pressed')

        return result


# ============================================================
# SHORT TOOLTIP EXPLANATIONS
# ============================================================
# These short texts appear when the user hovers over the ⓘ icon.
# They explain the result in simple everyday language.
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
# MEDIAPIPE HAND TRACKER SETUP
# ============================================================
# Load the MediaPipe hand detection tool and set how the
# hand skeleton is drawn on the video.
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Yellow dots for the 21 hand points; white lines connecting them
LANDMARK_SPEC   = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)


# ============================================================
# LOAD THE AI MODELS
# ============================================================
# All five trained models are loaded into memory once when the
# program starts.  Keeping them ready means every video upload
# gets a fast response without waiting for models to load.
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
# HAND SHAPE MEASUREMENT HELPERS
# ============================================================

def calculate_angle(vertex, point_a, point_b):
    """
    Measures a finger-joint angle, in degrees.

    The angle is measured AT `vertex` (the first landmark of the triplet),
    between the two rays vertex→point_a and vertex→point_b.
    This follows the supervisor's convention where the vertex is the
    FIRST landmark in each JOINTS row.

      0°  = the two sides point in the same direction
      90° = the two sides are at a right angle

    The calculation uses the full 3D coordinates (x, y, z).
    """
    vertex  = np.array(vertex)
    point_a = np.array(point_a)
    point_b = np.array(point_b)

    vec_a = point_a - vertex
    vec_b = point_b - vertex

    # Keep the value inside a safe range to avoid math errors
    cos = np.dot(vec_a, vec_b) / (
        np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-6
    )
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_frame_features(hand_landmarks):
    """
    Turns the 21 hand points from MediaPipe into 83 numbers
    that describe the hand's shape for that single frame.

    The 83 numbers come from three things:
      • 63 numbers — the x, y, z position of each of the 21 hand points
      • 10 numbers — how bent each of the 10 finger joints is (in degrees)
      • 10 numbers — the distance between each pair of fingertips
                     (thumb-to-index, thumb-to-middle, etc. → 10 pairs total)
    Total: 63 + 10 + 10 = 83
    """
    features = []
    coords   = []

    # Step 1: collect the position of every hand point (adds 63 numbers)
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
        features.extend([lm.x, lm.y, lm.z])

    coords = np.array(coords)

    # Step 2: measure how bent each finger joint is (adds 10 numbers)
    for joint in JOINTS:
        features.append(calculate_angle(coords[joint[1]], coords[joint[0]], coords[joint[2]]))
        # joint[0]=vertex (the joint), joint[1] and joint[2]=the two sides

    # Step 3: measure the straight-line distance between every pair of fingertips
    # There are 10 unique pairs from 5 fingertips: C(5,2) = 10 (adds 10 numbers)
    fingertips = [4, 8, 12, 16, 20]
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            features.append(float(np.linalg.norm(coords[fingertips[i]] - coords[fingertips[j]])))

    return features   # a list of 83 numbers


def detect_pressed_fingers(pressing_map: dict, finger_note_map: dict) -> list:
    """
    Takes the True/False result from the press tracker and converts
    it into a readable list of which fingers are pressing and what
    note each finger is playing.

    Returns a list where each item describes one pressing finger,
    including its name and the note it is playing.
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
# VIDEO STEP 1 — READ THE VIDEO, DETECT THE HAND, DRAW ON IT
# ============================================================

def extract_and_annotate(input_path: str, raw_out: str):
    """
    Reads the uploaded video one frame at a time and does two jobs:

    JOB 1 — Collect hand shape numbers:
      For every frame where a hand is found, record 83 numbers
      describing the hand's shape and posture.

    JOB 2 — Draw on the video:
      • Yellow dots and white lines showing all 21 hand points
      • Small numbers (0–20) labelling each hand point
      • A coloured circle at each fingertip:
            Yellow = finger is NOT pressing a key
            Red    = finger IS pressing a key
      • The note name (e.g. "Do") shown above a pressing fingertip
      • An "R" or "L" near the wrist showing which hand it is
      • A light grey box drawn around the whole hand

    The annotated video is saved to raw_out.
    (It cannot be played in a browser yet — that is fixed in Step 2.)

    Returns:
      all_features    — an array of shape numbers, one row per frame
      finger_activity — a list showing which fingers pressed in each frame
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

    # Start the MediaPipe hand detector
    # max_num_hands=1: only track one hand (the playing hand)
    with mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break   # reached the end of the video

            # MediaPipe needs colours in RGB order; OpenCV reads them in BGR order
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                # A hand was found in this frame
                hand_lm  = results.multi_hand_landmarks[0]
                features = extract_frame_features(hand_lm)

                # Work out which hand this is.
                # MediaPipe assumes a selfie/mirror camera, so for a normal
                # front-facing camera the hand labels are switched:
                #   MediaPipe "Left"  → the pianist's RIGHT hand
                #   MediaPipe "Right" → the pianist's LEFT hand
                mp_label = (
                    results.multi_handedness[0].classification[0].label
                    if results.multi_handedness else "Left"
                )
                finger_note_map = FINGER_NOTE_RIGHT if mp_label == "Left" else FINGER_NOTE_LEFT
                hand_label_text = "R" if mp_label == "Left" else "L"

                # Draw the hand skeleton (yellow dots + white connecting lines)
                mp_drawing.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    LANDMARK_SPEC,
                    CONNECTION_SPEC,
                )

                # Draw the landmark number next to each hand point
                for idx, lm in enumerate(hand_lm.landmark):
                    cx = int(lm.x * width)
                    cy = int(lm.y * height)
                    cv2.putText(frame, str(idx), (cx + 6, cy - 6),
                                cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

                # Calculate all 10 joint angles for the angle panel
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
                angles = [
                    calculate_angle(coords[side_a], coords[vertex], coords[side_b])
                    for vertex, side_a, side_b in JOINTS
                ]

                # ── Draw the angle panel below the classification banner area ──
                # The banner (added in Pass 2) occupies y=12..84, so start at y=90.
                # Panel layout: 5 rows (one per finger), 2 angles per row (2 joints per finger).
                PAD      = 12          # left margin — same as the banner
                ROW_H    = 26          # height of each row in pixels
                PANEL_Y  = 90          # top of the panel (just below banner)
                PANEL_W  = 320         # width of the panel box
                PANEL_H  = ROW_H * 5 + 10  # 5 fingers × row height + padding

                # Semi-transparent dark background
                overlay = frame.copy()
                cv2.rectangle(overlay, (PAD, PANEL_Y),
                              (PAD + PANEL_W, PANEL_Y + PANEL_H), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

                # Finger names and their two angle values (indices into the angles list)
                finger_rows = [
                    ("Thumb",  angles[0], angles[1]),
                    ("Index",  angles[2], angles[3]),
                    ("Middle", angles[4], angles[5]),
                    ("Ring",   angles[6], angles[7]),
                    ("Pinky",  angles[8], angles[9]),
                ]
                for row_i, (fname, ang1, ang2) in enumerate(finger_rows):
                    row_y = PANEL_Y + 6 + ROW_H * row_i + 12
                    # Finger name in white
                    cv2.putText(frame, fname, (PAD + 6, row_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                    # Two angles in green — J1 and J2 for that finger
                    angle_text = f"J1:{ang1:5.1f}  J2:{ang2:5.1f}"
                    cv2.putText(frame, angle_text, (PAD + 96, row_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)

                # ── Draw the distance panel below the angle panel ──
                # The 10 pairwise 3D distances between the 5 fingertips (C(5,2)=10) —
                # the same "distance feature" fed to the model. Shown as 5 rows × 2 cols.
                DIST_Y = PANEL_Y + PANEL_H + 6            # just below the angle panel
                DIST_H = ROW_H * 6 + 8                    # 1 title row + 5 data rows

                overlay = frame.copy()
                cv2.rectangle(overlay, (PAD, DIST_Y),
                              (PAD + PANEL_W, DIST_Y + DIST_H), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

                # Panel title
                cv2.putText(frame, "Fingertip Distances", (PAD + 6, DIST_Y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

                # The 10 pairwise normalised 3D distances between the 5 fingertips
                # (C(5,2)=10) — exactly the "distance feature" fed to the model.
                # Values are in MediaPipe's normalised units (0–1), scale-invariant.
                _tips     = [4, 8, 12, 16, 20]
                _tip_lbl  = {4: "T", 8: "I", 12: "M", 16: "R", 20: "P"}
                dist_items = []
                for a_i in range(len(_tips)):
                    for b_i in range(a_i + 1, len(_tips)):
                        d = float(np.linalg.norm(coords[_tips[a_i]] - coords[_tips[b_i]]))
                        dist_items.append((f"{_tip_lbl[_tips[a_i]]}-{_tip_lbl[_tips[b_i]]}", d))

                # Lay out the 10 values as 5 rows × 2 columns, in cyan
                for k, (lbl, d) in enumerate(dist_items):
                    col = k % 2
                    row = k // 2
                    dx  = PAD + 8 + col * 150
                    dy  = DIST_Y + 20 + ROW_H * (row + 1)
                    cv2.putText(frame, f"{lbl}:{d:4.2f}", (dx, dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 120), 1, cv2.LINE_AA)

                # Ask the press tracker which fingers are pressing right now
                pressing_map       = tracker.update(hand_lm)
                pressed_this_frame = detect_pressed_fingers(pressing_map, finger_note_map)
                pressing_set       = {p["tip_idx"] for p in pressed_this_frame}

                # Draw a coloured circle on each fingertip
                for tip_idx, fname in FINGER_INFO:
                    lm  = hand_lm.landmark[tip_idx]
                    cx  = int(lm.x * width)
                    cy  = int(lm.y * height)
                    pressing = tip_idx in pressing_set
                    # Red = pressing a key; Yellow = not pressing
                    color    = (0, 60, 255) if pressing else (0, 255, 255)
                    cv2.circle(frame, (cx, cy), 10, color, -1)    # filled circle
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2) # black outline

                    # When pressing, show the note name above the fingertip.
                    # Small text, no background box; a thin dark outline keeps it
                    # readable over any background.
                    if pressing:
                        note_label = "/".join(finger_note_map[tip_idx])
                        (tw, th), _ = cv2.getTextSize(note_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        tx, ty = cx - tw // 2, cy - 14
                        cv2.putText(frame, note_label, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)   # outline
                        cv2.putText(frame, note_label, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)  # fill

                # Draw R or L near the wrist to show which hand was detected
                wrist = hand_lm.landmark[0]
                wx, wy = int(wrist.x * width), int(wrist.y * height)
                cv2.putText(frame, hand_label_text, (wx - 18, wy + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

                # Draw a light grey box around the hand (with a 15-pixel margin)
                xs = [lm.x for lm in hand_lm.landmark]
                ys = [lm.y for lm in hand_lm.landmark]
                x1 = max(0, int(min(xs) * width)  - 15)
                y1 = max(0, int(min(ys) * height) - 15)
                x2 = min(width,  int(max(xs) * width)  + 15)
                y2 = min(height, int(max(ys) * height) + 15)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

                finger_activity.append(pressed_this_frame)

            else:
                # No hand was found in this frame — use all zeroes as placeholder numbers
                features = [0.0] * FEATURE_SIZE
                finger_activity.append([])

            all_features.append(features)
            writer.write(frame)   # save this frame to the output video

    cap.release()
    writer.release()

    return np.array(all_features), finger_activity


# ============================================================
# VIDEO STEP 2 — ADD THE RESULT BANNER AND MAKE IT WEB-READY
# ============================================================

def finalize_video(raw_path: str, final_path: str,
                   display: str, confidence: float, label: str) -> None:
    """
    Takes the annotated video from Step 1 and adds a result banner
    (for example: "Good   Confidence: 91.3%") in the top-left corner
    of every frame.

    It also converts the video to the H.264 format so it can be
    played directly inside a web browser.  H.264 requires the video
    width and height to be even numbers, so any odd-sized video is
    padded by one pixel automatically.

    A built-in video converter (imageio-ffmpeg) is used, so no
    separate program needs to be installed on the computer.
    """
    cap    = cv2.VideoCapture(raw_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make sure width and height are even numbers (required by H.264)
    enc_w = width  + (width  % 2)
    enc_h = height + (height % 2)

    color  = LABEL_COLOR.get(label, (255, 255, 255))
    text1  = display                          # e.g. "Good"
    text2  = f"Confidence: {confidence:.1f}%" # e.g. "Confidence: 91.3%"

    # Make the text size proportional to the video width so it always looks good
    font   = cv2.FONT_HERSHEY_DUPLEX
    scale1 = max(1.0,  width / 720)
    scale2 = max(0.5,  width / 1300)
    thick1 = max(2, int(width / 400))
    thick2 = max(1, int(width / 900))
    pad    = 12
    box_h  = 72
    box_w  = max(320, int(width * 0.40))

    # Open a video writer that saves directly in H.264 format
    writer = iio.get_writer(
        final_path,
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        ffmpeg_log_level="quiet",
        output_params=[
            "-pix_fmt",  "yuv420p",     # colour format that works in all browsers
            "-movflags", "+faststart",  # lets the video start playing before it finishes downloading
            "-preset",   "fast",        # good balance between encoding speed and file size
        ],
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw a semi-transparent dark rectangle as the background for the result text
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)   # 60% dark box, 40% original frame

        # Draw a thin coloured bar on the left side of the banner (green / orange / red)
        cv2.rectangle(frame, (pad, pad), (pad + 5, pad + box_h), color, -1)

        # Draw the result label in large coloured text
        cv2.putText(frame, text1, (pad + 14, pad + 44),
                    font, scale1, color, thick1, cv2.LINE_AA)

        # Draw the confidence percentage in smaller light grey text
        cv2.putText(frame, text2, (pad + 14, pad + 64),
                    font, scale2, (200, 200, 200), thick2, cv2.LINE_AA)

        # Pad the frame to even dimensions if needed
        if enc_w != width or enc_h != height:
            frame = cv2.copyMakeBorder(frame, 0, enc_h - height, 0, enc_w - width,
                                       cv2.BORDER_CONSTANT, value=0)

        # Convert colours from BGR (OpenCV format) to RGB (imageio format) before saving
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.close()


# ============================================================
# VIDEO STEP 3 — PUT THE ORIGINAL AUDIO BACK INTO THE VIDEO
# ============================================================

def mux_audio(video_no_audio: str, audio_source: str, output_path: str) -> None:
    """
    Copies the piano audio from the original uploaded video into
    the annotated video, so the viewer can hear the music while
    watching the hand analysis.

    This is the last step in the video pipeline.

    The built-in ffmpeg tool handles this — no separate installation needed.

    If the uploaded video had no audio (e.g. a screen recording with
    no sound), this step will fail and the caller will simply keep
    the silent version of the video instead.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-i", video_no_audio,   # the annotated video (without audio)
            "-i", audio_source,     # the original upload (source of the audio)
            "-c:v", "copy",         # keep the video exactly as-is (no re-encoding)
            "-c:a", "aac",          # convert audio to AAC format (plays in all browsers)
            "-map", "0:v:0",        # take the video track from the annotated video
            "-map", "1:a:0",        # take the audio track from the original upload
            "-shortest",            # stop at whichever track ends first
            output_path,
        ],
        check=True,
        capture_output=True,
    )


# ============================================================
# AI PREDICTION — COMBINE ALL FIVE MODELS
# ============================================================

def ensemble_predict(all_features: np.ndarray) -> dict:
    """
    Feeds the hand shape numbers into all five AI models and
    combines their answers to get one final result.

    How it works:
    1. The frame-by-frame numbers are grouped into overlapping
       30-frame clips (like short video snippets), shifting 5
       frames forward each time to cover the whole video.
    2. Each of the five models guesses the skill level for
       every clip, giving a percentage for each of the three
       possible labels.
    3. The percentages are averaged across all clips and all
       five models.
    4. The label with the highest average percentage wins.

    If the video is too short to form even one 30-frame clip,
    the clip is padded with zeroes so the models can still run.
    """
    # Cut the video features into overlapping 30-frame clips
    sequences = []
    for i in range(0, len(all_features) - SEQUENCE_LENGTH + 1, STRIDE):
        sequences.append(all_features[i : i + SEQUENCE_LENGTH])

    # If the video is shorter than 30 frames, pad it with zeroes
    if not sequences:
        padded = np.zeros((SEQUENCE_LENGTH, FEATURE_SIZE))
        n = min(len(all_features), SEQUENCE_LENGTH)
        padded[:n] = all_features[:n]
        sequences = [padded]

    X = np.array(sequences)   # shape: (number of clips, 30 frames, 83 numbers)

    all_probs = []   # the averaged guess from each model
    per_model = []   # each model's individual guess (used in the explanation)

    for idx, model in enumerate(models):
        preds = model.predict(X, verbose=0)   # shape: (number of clips, 3 labels)
        avg   = np.mean(preds, axis=0)         # average across all clips → 3 numbers
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

    # Average across all five models → the final combined answer
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
# STATISTICS SUMMARY
# ============================================================

def build_analysis_summary(finger_activity: list, total_frames: int, detection_frames: int) -> dict:
    """
    Counts how often each finger pressed a key across the whole video
    and turns those counts into percentages.

    For example: "Do (Thumb) was pressing in 35% of the video frames."

    This information is shown on the results page and is also used
    to personalise the explanation text.
    """
    finger_frames = {}   # tracks which frames each finger-note combination was pressed
    press_frames  = set()  # tracks all frames where any key press happened

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
# EXPLANATION TEXT BUILDER
# ============================================================

def build_explanation(label: str, confidence: float, per_model: list, analysis: dict) -> str:
    """
    Writes a personalised, easy-to-understand explanation of why
    the video got the result it did.

    The explanation has up to four sentences:
      1. The main reason for the result (in simple language)
      2. How confident and consistent the AI was
      3. A warning if the hand was hard to see in the video
      4. Which finger was pressed the most and how often
    """
    total      = analysis.get("total_frames", 0)
    detected   = analysis.get("detection_frames", 0)
    rate       = analysis.get("detection_rate", 0)
    press      = analysis.get("press_frames", 0)
    finger_act = analysis.get("finger_activity", {})
    n_models   = len(per_model)
    n_agree    = sum(1 for m in per_model if m["label"] == label)
    press_pct  = round(press / total * 100, 1) if total > 0 else 0

    # Find the finger that was pressed most often
    sorted_fingers = sorted(finger_act.items(), key=lambda x: x[1], reverse=True)
    most_active    = sorted_fingers[0] if sorted_fingers else None

    sentences = []

    # ── Sentence 1: The main reason for this result ────────
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

    # ── Sentence 2: How confident and consistent the result is ─
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

    # ── Sentence 3: Warn if the hand was hard to see ──────
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

    # ── Sentence 4: Which finger pressed the most ─────────
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
# WEB SERVER (FLASK APPLICATION)
# ============================================================
app = Flask(__name__)

# Reject uploads that are larger than the allowed maximum size
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    """Checks whether the uploaded file has an accepted format (mp4 or mov)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── /predict — the main upload endpoint ─────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    This is what runs when the user presses the upload button.
    It receives the video, runs through all the steps, and sends
    back the result as a JSON message.

    Steps:
      1. Check that the uploaded file is valid (right format, not corrupted)
      2. Step 1 — read the video, detect the hand, collect shape numbers
      3. Run all five AI models on the shape numbers
      4. Step 2 — draw the result banner, convert to web-ready format
      5. Step 3 — put the audio back in
      6. Send back the result (label, confidence, stats, video filename)

    All temporary files are deleted at the end, whether or not
    the process succeeded.
    """
    if "video" not in request.files:
        return jsonify({"error": "Field 'video' not found in request."}), 400

    file = request.files["video"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXT).upper()}"}), 400

    # Create a unique ID for this upload so file names never clash
    uid       = uuid.uuid4().hex
    orig_stem = os.path.splitext(secure_filename(file.filename))[0]

    # Temporary file names for each stage of the pipeline
    input_path  = os.path.join(UPLOAD_DIR, f"{uid}_input.mp4")   # the original upload
    raw_path    = os.path.join(UPLOAD_DIR, f"{uid}_raw.mp4")      # after Step 1
    silent_path = os.path.join(UPLOAD_DIR, f"{uid}_silent.mp4")   # after Step 2 (no audio yet)

    file.save(input_path)

    try:
        # ── Check 1: can the video be opened? ──────────────
        cap_check = cv2.VideoCapture(input_path)
        if not cap_check.isOpened():
            cap_check.release()
            return jsonify({"error": "Video file could not be read. Make sure the file is not corrupted and the format is supported."}), 422

        vid_w      = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h      = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps    = cap_check.get(cv2.CAP_PROP_FPS)
        vid_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_check.release()

        # ── Check 2: does the video have valid dimensions? ──
        if vid_w == 0 or vid_h == 0:
            return jsonify({"error": "Video has invalid dimensions (width or height is 0)."}), 422

        # ── Check 3: is the video long enough? ─────────────
        if 0 < vid_frames < SEQUENCE_LENGTH:
            return jsonify({
                "error": f"Video is too short. At least {SEQUENCE_LENGTH} frames are required; "
                         f"this video only has {vid_frames} frames."
            }), 422

        # ── Step 1: detect hand, collect numbers, draw annotations ──
        all_features, finger_activity = extract_and_annotate(input_path, raw_path)

        if len(all_features) == 0:
            return jsonify({"error": "No frames could be extracted from the video."}), 422

        # ── Check 4: was a hand visible at any point? ───────
        detection_frames = sum(1 for f in all_features if any(v != 0.0 for v in f))
        if detection_frames == 0:
            return jsonify({
                "error": "No hand was detected in the video. "
                         "Make sure the hand is clearly visible and well-lit."
            }), 422

        # ── Run all five AI models and combine their answers ─
        result = ensemble_predict(all_features)

        # ── Build the statistics summary ────────────────────
        result["analysis"] = build_analysis_summary(
            finger_activity, len(all_features), detection_frames
        )

        # Short one-line explanation for the ⓘ tooltip
        result["short_explanation"] = LABEL_EXPLANATION[result["label"]]

        # Longer personalised explanation for the "Why is the result?" panel
        result["explanation"] = build_explanation(
            result["label"], result["confidence"],
            result["per_model"], result["analysis"]
        )

        # The final video file name includes the original name and the result label
        final_name = f"{orig_stem}_{result['label']}.mp4"
        final_path = os.path.join(UPLOAD_DIR, final_name)

        # ── Step 2: add result banner, convert to H.264 ────
        finalize_video(raw_path, silent_path,
                       result["display"], result["confidence"], result["label"])
        os.remove(raw_path)   # delete the Step 1 intermediate file

        # ── Step 3: put the audio back in ──────────────────
        try:
            mux_audio(silent_path, input_path, final_path)
            os.remove(silent_path)
        except Exception:
            # The video had no audio — just use the silent version as the final output
            os.rename(silent_path, final_path)

        result["annotated_video"] = final_name

        return jsonify(result), 200

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    finally:
        # Always delete the original upload to keep the server's disk clean
        if os.path.exists(input_path):
            os.remove(input_path)
        gc.collect()   # free memory used by the large arrays and model outputs


# ── /video/<filename> — serve a video file ──────────────────
@app.route("/video/<filename>")
def serve_video(filename):
    """
    Sends an annotated video file to the browser so it can be played.
    Supports seeking (the user can jump to any point in the video
    without waiting for the whole file to download first).
    """
    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="video/mp4", conditional=True)


# ── /health — check if the server is running ────────────────
@app.route("/health")
def health():
    """
    A simple check that tells you whether the API is running
    and how many AI models were loaded successfully.
    Returns something like: {"status": "ok", "models_loaded": 5}
    """
    return jsonify({"status": "ok", "models_loaded": len(models)}), 200


# ── Start the server ────────────────────────────────────────
if __name__ == "__main__":
    # host="0.0.0.0" allows other computers on the same network to reach this server.
    # debug=False keeps the server stable and does not auto-restart when the code changes.
    app.run(host="0.0.0.0", port=5000, debug=False)
