"""
Piano Playing Level Classifier — Flask API
Ensemble prediction menggunakan semua 5 model BiLSTM (K-Fold)

Endpoint:
  POST /predict       → upload video, kembalikan JSON hasil klasifikasi
  GET  /video/<name>  → stream annotated video
  GET  /health        → cek status API
"""

import os
import gc
import uuid
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import subprocess
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import imageio.v2 as iio
import imageio_ffmpeg              # untuk mendapatkan path binary FFmpeg bawaan

from flask import Flask, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename

# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "Tugas Akhir Ko Pat",
                          "OUTPUT_PIANO_BILSTM", "best_model")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB     = 200

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================================================
# PARAMETER FITUR (identik dengan notebook training)
# =========================================================
SEQUENCE_LENGTH = 30
STRIDE          = 5
FEATURE_SIZE    = 83   # 21×3 landmark + 10 sudut sendi + 10 jarak fingertip

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

# warna BGR per label untuk overlay teks di video
LABEL_COLOR = {
    "good":             (86,  205, 86),   # hijau
    "needs_improvement":(60,  180, 245),  # kuning-oranye → pakai kuning muda
    "poor":             (60,  60,  239),  # merah
}

# =========================================================
# MEDIAPIPE
# =========================================================
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# custom drawing spec agar lebih jelas di video
LANDMARK_SPEC   = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)

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
        print(f"  [SKIP] Fold {fold}: tidak ditemukan — {path}")

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
# PASS 1 — ekstrak fitur + tulis raw annotated video
# =========================================================
def extract_and_annotate(input_path: str, raw_out: str) -> np.ndarray:
    """
    Baca video input, jalankan MediaPipe per frame.
    Gambar landmark + nomor fingertip + bounding box tangan.
    Tulis ke raw_out (mp4v, belum di-encode H.264).
    Kembalikan array fitur (n_frames, FEATURE_SIZE).
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

                # --- gambar koneksi tangan ---
                mp_drawing.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    LANDMARK_SPEC,
                    CONNECTION_SPEC,
                )

                # --- sorot fingertip (landmark 4,8,12,16,20) ---
                for tip_idx in [4, 8, 12, 16, 20]:
                    lm  = hand_lm.landmark[tip_idx]
                    cx  = int(lm.x * width)
                    cy  = int(lm.y * height)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2)

                # --- bounding box tangan ---
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

# =========================================================
# PASS 2 — label overlay + encode H.264 via imageio-ffmpeg
# =========================================================
def finalize_video(raw_path: str, final_path: str,
                   display: str, confidence: float, label: str) -> None:
    """
    Baca raw_path (mp4v dari OpenCV), tambahkan label overlay per frame,
    tulis final_path sebagai H.264 MP4 menggunakan imageio-ffmpeg
    (bundled binary — tidak perlu install FFmpeg di sistem).
    """
    cap    = cv2.VideoCapture(raw_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # H.264 memerlukan dimensi genap
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
            "-pix_fmt",   "yuv420p",
            "-movflags",  "+faststart",
            "-preset",    "fast",
        ],
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- label overlay ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h), color, 2)
        cv2.putText(frame, text1, (pad + 10, pad + 38),
                    font, scale1, color, thick1, cv2.LINE_AA)
        cv2.putText(frame, text2, (pad + 10, pad + 60),
                    font, scale2, (220, 220, 220), thick2, cv2.LINE_AA)

        # padding dimensi jika ganjil, lalu BGR→RGB untuk imageio
        if enc_w != width or enc_h != height:
            frame = cv2.copyMakeBorder(frame, 0, enc_h - height, 0, enc_w - width,
                                       cv2.BORDER_CONSTANT, value=0)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.close()

# =========================================================
# PASS 3 — mux audio dari video asli ke video anotasi
# =========================================================
def mux_audio(video_no_audio: str, audio_source: str, output_path: str) -> None:
    """
    Gabungkan video teranotasi (tanpa audio) dengan audio dari video asli.
    Menggunakan binary FFmpeg bawaan imageio-ffmpeg — tidak perlu install sistem.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-i", video_no_audio,   # video H.264 tanpa audio
            "-i", audio_source,     # video asli (sumber audio)
            "-c:v", "copy",         # salin stream video apa adanya
            "-c:a", "aac",          # encode audio ke AAC
            "-map", "0:v:0",        # ambil video dari input pertama
            "-map", "1:a:0",        # ambil audio dari input kedua
            "-shortest",            # selesai saat stream terpendek habis
            output_path,
        ],
        check=True,
        capture_output=True,
    )

# =========================================================
# ENSEMBLE PREDICTION
# =========================================================
def ensemble_predict(all_features: np.ndarray) -> dict:
    sequences = []
    for i in range(0, len(all_features) - SEQUENCE_LENGTH + 1, STRIDE):
        sequences.append(all_features[i : i + SEQUENCE_LENGTH])

    if not sequences:
        padded = np.zeros((SEQUENCE_LENGTH, FEATURE_SIZE))
        n = min(len(all_features), SEQUENCE_LENGTH)
        padded[:n] = all_features[:n]
        sequences = [padded]

    X = np.array(sequences)   # (n_seq, 30, 83)

    all_probs    = []
    per_model    = []

    for idx, model in enumerate(models):
        preds = model.predict(X, verbose=0)   # (n_seq, 3)
        avg   = np.mean(preds, axis=0)        # (3,)
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

    ensemble_avg = np.mean(all_probs, axis=0)   # (3,)
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

    if not file.filename:
        return jsonify({"error": "Tidak ada file yang dipilih."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Format tidak didukung. Gunakan: {', '.join(ALLOWED_EXT).upper()}"}), 400

    uid         = uuid.uuid4().hex
    # nama dasar dari file asli (tanpa ekstensi, karakter aman)
    orig_stem   = os.path.splitext(secure_filename(file.filename))[0]

    input_path  = os.path.join(UPLOAD_DIR, f"{uid}_input.mp4")
    raw_path    = os.path.join(UPLOAD_DIR, f"{uid}_raw.mp4")       # pass 1 (mp4v, no audio)
    silent_path = os.path.join(UPLOAD_DIR, f"{uid}_silent.mp4")    # pass 2 (H.264, no audio)
    # final_path ditentukan setelah label diketahui

    file.save(input_path)

    try:
        # --- Pass 1: ekstrak fitur + gambar landmark MediaPipe ---
        all_features = extract_and_annotate(input_path, raw_path)

        if len(all_features) == 0:
            return jsonify({"error": "Tidak ada frame yang berhasil diekstrak dari video."}), 422

        # --- Prediksi ensemble ---
        result = ensemble_predict(all_features)

        # --- nama file final: (nama_file_awal)_(hasil_klasifikasi).mp4 ---
        final_name  = f"{orig_stem}_{result['label']}.mp4"
        final_path  = os.path.join(UPLOAD_DIR, final_name)

        # --- Pass 2: label overlay + encode H.264 (belum ada audio) ---
        finalize_video(
            raw_path, silent_path,
            result["display"], result["confidence"], result["label"]
        )
        os.remove(raw_path)

        # --- Pass 3: mux audio dari video asli ---
        try:
            mux_audio(silent_path, input_path, final_path)
            os.remove(silent_path)
        except Exception:
            # jika video asli tidak punya audio, pakai video tanpa audio
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


# ---------------------------------------------------------
# GET /video/<filename>
# ---------------------------------------------------------
@app.route("/video/<filename>")
def serve_video(filename):
    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        abort(404)
    return send_file(
        path,
        mimetype="video/mp4",
        conditional=True,   # support Range requests (seek)
    )


# ---------------------------------------------------------
# GET /health
# ---------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "models_loaded": len(models),
    }), 200


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
