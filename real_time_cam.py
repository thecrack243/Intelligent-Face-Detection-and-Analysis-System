import os

# Reduce TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from tensorflow.keras.models import load_model


# =========================================================
# Configuration
# =========================================================
YOLO_MODEL_PATH = "models/WIDER_FACE/runs/detect/face_detection/weights/best.pt"
EMOTION_MODEL_PATH = "models/FER2013/best_fer2013_model.keras"

WEBCAM_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

YOLO_IMAGE_SIZE = 640
FACE_CONFIDENCE_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.45

# Higher number = faster but less smooth prediction update
# 3 or 5 is good for laptop CPU
PROCESS_EVERY_N_FRAMES = 3

MIN_FACE_SIZE = 25

SCREENSHOT_FOLDER = "webcam_screenshots"
WINDOW_NAME = "Real-Time Face Expression Recognition"


# =========================================================
# Emotion Labels
# Same order as your training LabelEncoder
# ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# =========================================================
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# =========================================================
# Sentiment Color
# =========================================================
SENTIMENT_COLORS = {
    "Positive": (0, 255, 0),
    "Negative": (0, 0, 255),
    "Neutral": (0, 255, 255),
    "Unknown": (180, 180, 180)
}


# =========================================================
# Sentiment Analysis
# =========================================================
def get_sentiment(emotion):
    if emotion in ["happy", "surprise"]:
        return "Positive"
    elif emotion in ["angry", "disgust", "fear", "sad"]:
        return "Negative"
    elif emotion == "neutral":
        return "Neutral"
    else:
        return "Unknown"


# =========================================================
# Safe Text Drawing Function
# =========================================================
def draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x = max(0, x)
    y = max(text_h + 10, y)

    cv2.rectangle(
        frame,
        (x, y - text_h - 10),
        (x + text_w + 12, y + 5),
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 6, y - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )


# =========================================================
# Load Models
# =========================================================
print("Loading YOLOv8 face detection model...")
face_model = YOLO(YOLO_MODEL_PATH)

print("Loading emotion recognition model...")
emotion_model = load_model(EMOTION_MODEL_PATH)

print("Models loaded successfully.")


# =========================================================
# Emotion Prediction Function
# =========================================================
def predict_emotion(face_crop_bgr):
    try:
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))

        normalized = resized.astype("float32") / 255.0
        input_img = normalized.reshape(1, 48, 48, 1)

        # Faster than model.predict() for real-time webcam
        prediction = emotion_model(input_img, training=False).numpy()[0]

        pred_index = int(np.argmax(prediction))
        emotion = EMOTION_LABELS[pred_index]
        confidence = float(prediction[pred_index]) * 100

        return emotion, confidence

    except Exception as e:
        print(f"Emotion prediction error: {e}")
        return "unknown", 0.0


# =========================================================
# Face Detection + Emotion Recognition
# =========================================================
def detect_faces_and_emotions(frame):
    detections = []

    results = face_model(
        frame,
        conf=FACE_CONFIDENCE_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False
    )

    h, w, _ = frame.shape

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            face_confidence = float(box.conf[0]) * 100

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_width = x2 - x1
            face_height = y2 - y1

            if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                continue

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            emotion, emotion_confidence = predict_emotion(face_crop)
            sentiment = get_sentiment(emotion)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "face_confidence": face_confidence,
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
                "sentiment": sentiment
            })

    return detections


# =========================================================
# Draw Results on Frame
# =========================================================
def draw_results(frame, detections, fps):
    output = frame.copy()

    for i, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["bbox"]
        face_confidence = detection["face_confidence"]
        emotion = detection["emotion"]
        emotion_confidence = detection["emotion_confidence"]
        sentiment = detection["sentiment"]

        color = SENTIMENT_COLORS.get(sentiment, (0, 255, 0))

        # Bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        # Main label
        label_text = f"Face {i}: {emotion} {emotion_confidence:.1f}% | {sentiment}"
        draw_label(output, label_text, x1, y1 - 8, color)

        # Face confidence
        face_text = f"Detection: {face_confidence:.1f}%"
        cv2.putText(
            output,
            face_text,
            (x1, min(y2 + 25, output.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Top information panel
    panel_height = 105
    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (470, panel_height),
        (0, 0, 0),
        -1
    )

    output = cv2.addWeighted(overlay, 0.55, output, 0.45, 0)

    info_lines = [
        f"FPS: {fps:.1f}",
        f"Faces Detected: {len(detections)}",
        f"Confidence Threshold: {FACE_CONFIDENCE_THRESHOLD}",
        "Press 's' = Save Screenshot | Press 'q' = Quit"
    ]

    y = 35
    for line in info_lines:
        cv2.putText(
            output,
            line,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )
        y += 22

    return output


# =========================================================
# Save Screenshot
# =========================================================
def save_screenshot(frame):
    os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"webcam_detection_{timestamp}.png"
    save_path = os.path.join(SCREENSHOT_FOLDER, filename)

    cv2.imwrite(save_path, frame)
    print(f"Screenshot saved: {save_path}")


# =========================================================
# Start Webcam
# =========================================================
cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Webcam started.")
print("Press 's' to save screenshot.")
print("Press 'q' to quit.")
print(f"Processing every {PROCESS_EVERY_N_FRAMES} frame(s).")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1100, 700)

frame_count = 0
last_detections = []
last_time = time.time()
fps = 0.0


# =========================================================
# Main Loop
# =========================================================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from webcam.")
        break

    frame_count += 1

    # Calculate FPS
    current_time = time.time()
    time_diff = current_time - last_time

    if time_diff > 0:
        fps = 1.0 / time_diff

    last_time = current_time

    # Run YOLO + emotion model only every N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        last_detections = detect_faces_and_emotions(frame)

    display_frame = draw_results(frame, last_detections, fps)

    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        save_screenshot(display_frame)

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")