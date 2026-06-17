# Flask API for Piano Playing Level Classification
# Receives a video file, runs MediaPipe hand tracking per frame,
# extracts features, and runs ensemble inference across all 5 BiLSTM models.
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
MODEL_DIR   = os.path.join(BASE_DIR, "..", "Tugas Akhir Ko Pat",
                            "OUTPUT_PIANO_BILSTM", "best_model")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB      = 200

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


def extract_and_annotate(input_path: str, raw_out: str) -> np.ndarray:
    """
    Pass 1: reads the input video frame by frame, runs MediaPipe hand detection,
    draws landmarks and fingertip markers onto each frame, and writes the result
    to raw_out using the mp4v codec. Returns the feature array of shape
    (n_frames, FEATURE_SIZE) for use in prediction.
    """
    cap = cv2.VideoCapture(input_path)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (width, height))

    all_features = []

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

                mp_drawing.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    LANDMARK_SPEC,
                    CONNECTION_SPEC,
                )

                # Highlight fingertip landmarks with a filled circle
                for tip_idx in [4, 8, 12, 16, 20]:
                    lm = hand_lm.landmark[tip_idx]
                    cx = int(lm.x * width)
                    cy = int(lm.y * height)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2)

                # Draw a bounding box around the detected hand
                xs = [lm.x for lm in hand_lm.landmark]
                ys = [lm.y for lm in hand_lm.landmark]
                x1 = max(0, int(min(xs) * width)  - 15)
                y1 = max(0, int(min(ys) * height) - 15)
                x2 = min(width,  int(max(xs) * width)  + 15)
                y2 = min(height, int(max(ys) * height) + 15)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

            else:
                features = [0.0] * FEATURE_SIZE

            all_features.append(features)
            writer.write(frame)

    cap.release()
    writer.release()

    return np.array(all_features)


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
    text1  = f"Level: {display}"
    text2  = f"Confidence: {confidence:.1f}%"
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale1 = max(0.8,  width / 900)
    scale2 = max(0.55, width / 1300)
    thick1 = max(2, int(width / 480))
    thick2 = max(1, int(width / 800))
    pad    = 10
    box_h  = 70
    box_w  = max(350, int(width * 0.42))

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

        # Semi-transparent dark background behind the label text
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h), color, 2)
        cv2.putText(frame, text1, (pad + 10, pad + 38),
                    font, scale1, color, thick1, cv2.LINE_AA)
        cv2.putText(frame, text2, (pad + 10, pad + 60),
                    font, scale2, (220, 220, 220), thick2, cv2.LINE_AA)

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
        return jsonify({"error": "Field 'video' tidak ditemukan dalam request."}), 400

    file = request.files["video"]

    if not file.filename:
        return jsonify({"error": "Tidak ada file yang dipilih."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Format tidak didukung. Gunakan: {', '.join(ALLOWED_EXT).upper()}"}), 400

    uid       = uuid.uuid4().hex
    orig_stem = os.path.splitext(secure_filename(file.filename))[0]

    input_path  = os.path.join(UPLOAD_DIR, f"{uid}_input.mp4")
    raw_path    = os.path.join(UPLOAD_DIR, f"{uid}_raw.mp4")
    silent_path = os.path.join(UPLOAD_DIR, f"{uid}_silent.mp4")

    file.save(input_path)

    try:
        all_features = extract_and_annotate(input_path, raw_path)

        if len(all_features) == 0:
            return jsonify({"error": "Tidak ada frame yang berhasil diekstrak dari video."}), 422

        result = ensemble_predict(all_features)

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
