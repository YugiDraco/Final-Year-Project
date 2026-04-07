"""
face_detection.py — Face detection with MTCNN (preferred) or OpenCV Haar cascade (fallback)
# NEW  (UPDATED: Handles Python 3.14 / Pillow 12 environments where facenet-pytorch
         may not be installable — falls back to OpenCV Haar cascade automatically)
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

from config import FACE_PAD_RATIO, MIN_FACE_SIZE, DEVICE, ALIGN_EYES

# ── Lazy MTCNN loader ─────────────────────────────────────────────────────────
_mtcnn = None

def _get_mtcnn():
    """Lazy-load MTCNN. Returns None if unavailable (Python 3.14 / Pillow 12 compat)."""
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            _mtcnn = MTCNN(
                keep_all=True,
                min_face_size=MIN_FACE_SIZE,
                thresholds=[0.6, 0.7, 0.7],
                device=DEVICE,
                post_process=False,
            )
        except Exception:
            _mtcnn = "unavailable"
    return None if _mtcnn == "unavailable" else _mtcnn


def mtcnn_available() -> bool:
    return _get_mtcnn() is not None


# ── Face Alignment ────────────────────────────────────────────────────────────
def align_face(pil_image: Image.Image, landmarks: np.ndarray) -> Image.Image:
    """
    Align face by rotating so eyes are horizontal.
    landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    left_eye  = landmarks[0]
    right_eye = landmarks[1]

    # Calculate angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate around center of eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))


# ── OpenCV Haar Cascade fallback ──────────────────────────────────────────────
def _detect_faces_opencv(pil_image: Image.Image):
    """Use OpenCV frontal face Haar cascade as a reliable fallback."""
    img_bgr  = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )

    if len(faces) == 0:
        return None, None  # no face detected

    boxes = []
    probs = []
    for (x, y, w, h) in faces:
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
        probs.append(0.95)   # Haar cascade doesn't give probabilities

    return np.array(boxes), np.array(probs)


# ── Main face detector ────────────────────────────────────────────────────────
def detect_faces(pil_image: Image.Image) -> Tuple[List[Image.Image], List[dict], bool]:
    """
    Detect all faces in a PIL image, align them (if enabled), and return padded crops.
    """
    pil_rgb = pil_image.convert("RGB")
    boxes, probs, landmarks = None, None, None

    # ── Try MTCNN ─────────────────────────────────────────────────────────
    mtcnn = _get_mtcnn()
    if mtcnn is not None:
        try:
            # detect returns (boxes, probs, landmarks)
            boxes, probs, landmarks = mtcnn.detect(pil_rgb, landmarks=True)
        except Exception:
            boxes, probs, landmarks = None, None, None

    # ── Fallback: OpenCV Haar Cascade ──────────────────────────────────────
    if boxes is None or len(boxes) == 0:
        boxes, probs = _detect_faces_opencv(pil_rgb)
        landmarks = [None] * len(boxes) if boxes is not None else None

    # ── Final fallback: full image ─────────────────────────────────────────
    if boxes is None or len(boxes) == 0:
        return [pil_rgb], [{"box": None, "confidence": None, "face_id": 0}], True

    w, h = pil_rgb.size
    face_crops = []
    face_info  = []

    for i, (box, prob, ldmk) in enumerate(zip(boxes, probs, landmarks)):
        if prob is not None and prob < 0.70:
            continue

        # Optional Eye Alignment (Feature 1)
        working_img = pil_rgb
        if ALIGN_EYES and ldmk is not None:
            working_img = align_face(pil_rgb, ldmk)

        x1, y1, x2, y2 = [float(v) for v in box]
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * FACE_PAD_RATIO
        pad_y = bh * FACE_PAD_RATIO

        x1c = max(0, int(x1 - pad_x))
        y1c = max(0, int(y1 - pad_y))
        x2c = min(w, int(x2 + pad_x))
        y2c = min(h, int(y2 + pad_y))

        crop = working_img.crop((x1c, y1c, x2c, y2c))
        face_crops.append(crop)
        face_info.append({
            "face_id":    i,
            "box":        [x1c, y1c, x2c, y2c],
            "confidence": round(float(prob), 4) if prob is not None else None,
            "aligned":    (ALIGN_EYES and ldmk is not None)
        })

    if not face_crops:
        return [pil_rgb], [{"box": None, "confidence": None, "face_id": 0}], True

    return face_crops, face_info, False


def draw_face_boxes(pil_image: Image.Image, face_info: List[dict]) -> Image.Image:
    """Draw bounding boxes on the image for all detected faces."""
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    for fi in face_info:
        if fi.get("box") is None:
            continue
        x1, y1, x2, y2 = fi["box"]
        conf  = fi.get("confidence")
        label = f"Face #{fi['face_id']+1}" + (f" ({conf:.2f})" if conf else "")
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (57, 255, 20), 2)
        cv2.putText(img_bgr, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (57, 255, 20), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
