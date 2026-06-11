# =========================
# 1. INSTALL (Colab only)
# =========================
#!pip install mediapipe opencv-python tensorflow scikit-learn numpy

# =========================
# 2. IMPORT
# =========================
import os
import cv2
import numpy as np
import mediapipe as mp

# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 3. SETUP MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

# =========================
# 4. PARAMETER
# =========================
DATASET_PATH = "Dataset"   # ganti sesuai lokasi kamu
MAX_FRAMES = 30
FEATURES = 21 * 3 * 2  # 2 tangan

label_map = {
    "good": 0,
    "needs_improvement": 1,
    "poor": 2
}

# =========================
# 5. EXTRACT KEYPOINTS
# =========================
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            keypoints = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])

                # padding jika hanya 1 tangan
                if len(results.multi_hand_landmarks) == 1:
                    keypoints.extend([0] * (21 * 3))
            else:
                keypoints = [0] * FEATURES

            frames.append(keypoints)

    cap.release()

    # =========================
    # SAMPLING / PADDING
    # =========================
    if len(frames) > MAX_FRAMES:
        idx = np.linspace(0, len(frames)-1, MAX_FRAMES).astype(int)
        frames = [frames[i] for i in idx]
    else:
        while len(frames) < MAX_FRAMES:
            frames.append([0]*FEATURES)

    return np.array(frames)

# =========================
# 6. LOAD DATASET
# =========================
X = []
y = []

for label in label_map:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            path = os.path.join(folder, file)
            print("Processing:", path)

            kp = extract_keypoints(path)
            X.append(kp)
            y.append(label_map[label])

X = np.array(X)
# y = to_categorical(y)

print("X shape:", X.shape)
# print("y shape:", y.shape)

