"""
Piano Playing Level Classifier — Flask API
Ensemble prediction menggunakan semua 5 model BiLSTM (K-Fold)
Endpoint:
  POST /predict       → upload video, kembalikan JSON hasil klasifikasi
  GET  /video/<name>  → stream annotated video
"""

import os
import gc
import uuid
import json
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from flask import Flask, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename

# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "Tugas Akhir Ko Pat",
                           "OUTPUT_PIANO_BILSTM", "best_model")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB      = 200

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================================================
# PARAMETER FITUR (identik dengan notebook training)
# =========================================================
SEQUENCE_LENGTH = 30
STRIDE          = 5
FEATURE_SIZE    = 83   # 21×3 landmark + 10 sudut + 10 jarak fingertip

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

# =========================================================
# MEDIAPIPE
# =========================================================
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# =========================================================
# LOAD SEMUA 5 MODEL SAAT STARTUP
# =========================================================
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
        print(f"  [SKIP] Fold {fold}: file tidak ditemukan — {path}")

if not models:
    raise RuntimeError("Tidak ada model yang berhasil dimuat. Periksa MODEL_DIR.")

print(f"\nEnsemble siap: {len(models)} model dimuat.\n")

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_frame_features(hand_landmarks):
    features = []
    coords   = []

    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
        features.extend([lm.x, lm.y, lm.z])

    coords = np.array(coords)

    for joint in JOINTS:
        features.append(calculate_angle(
            coords[joint[0]], coords[joint[1]], coords[joint[2]]
        ))

    fingertips = [4, 8, 12, 16, 20]
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            features.append(float(np.linalg.norm(
                coords[fingertips[i]] - coords[fingertips[j]]
            )))

    return features


# =========================================================
# PROCESS VIDEO: ekstrak fitur + tulis annotated video
# =========================================================
def process_video(input_path: str, output_path: str) -> np.ndarray:
    """Kembalikan array (n_frames, FEATURE_SIZE) dan tulis video teranotasi."""
    cap = cv2.VideoCapture(input_path)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_frames = []

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
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            else:
                features = [0.0] * FEATURE_SIZE

            all_frames.append(features)
            writer.write(frame)

    cap.release()
    writer.release()

    return np.array(all_frames)


# =========================================================
# ENSEMBLE PREDICTION
# =========================================================
def ensemble_predict(all_frames: np.ndarray) -> dict:
    """Rata-rata softmax dari semua model (soft voting)."""
    # bangun sequences
    sequences = []
    for i in range(0, len(all_frames) - SEQUENCE_LENGTH + 1, STRIDE):
        sequences.append(all_frames[i : i + SEQUENCE_LENGTH])

    # padding jika video terlalu pendek
    if not sequences:
        padded = np.zeros((SEQUENCE_LENGTH, FEATURE_SIZE))
        n = min(len(all_frames), SEQUENCE_LENGTH)
        padded[:n] = all_frames[:n]
        sequences = [padded]

    X = np.array(sequences)   # (n_seq, 30, 83)

    # prediksi tiap model lalu rata-rata
    all_probs = []
    for model in models:
        preds = model.predict(X, verbose=0)        # (n_seq, 3)
        avg   = np.mean(preds, axis=0)             # (3,)
        all_probs.append(avg)

    ensemble_avg = np.mean(all_probs, axis=0)      # (3,)
    idx          = int(np.argmax(ensemble_avg))
    label        = LABEL_NAMES[idx]
    confidence   = float(ensemble_avg[idx])

    return {
        "label":         label,
        "display":       LABEL_DISPLAY[label],
        "confidence":    round(confidence * 100, 2),
        "probabilities": {
            LABEL_NAMES[i]: round(float(ensemble_avg[i]) * 100, 2)
            for i in range(3)
        },
        "model_count":   len(models),
    }


# =========================================================
# FLASK APP
# =========================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------------------------------------------------------
# POST /predict
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "Field 'video' tidak ditemukan dalam request."}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"error": "Tidak ada file yang dipilih."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Format tidak didukung. Gunakan: {', '.join(ALLOWED_EXT).upper()}"}), 400

    uid         = uuid.uuid4().hex
    input_path  = os.path.join(UPLOAD_DIR, f"{uid}_input.mp4")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_annotated.mp4")

    file.save(input_path)

    try:
        all_frames = process_video(input_path, output_path)

        if len(all_frames) == 0:
            return jsonify({"error": "Tidak ada frame yang berhasil diekstrak dari video."}), 422

        result = ensemble_predict(all_frames)
        result["annotated_video"] = f"{uid}_annotated.mp4"

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        gc.collect()


# ---------------------------------------------------------
# GET /video/<filename>
# ---------------------------------------------------------
@app.route("/video/<filename>")
def serve_video(filename):
    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="video/mp4")


# ---------------------------------------------------------
# GET /health
# ---------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": len(models)}), 200


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
