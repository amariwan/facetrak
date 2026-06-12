import cv2
import numpy as np

GOOD_THRESHOLD = 0.55


def score(face_bgr: np.ndarray, yaw: float = 0.0,
          pitch: float = 0.0) -> float:
    if face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var(), 200.0) / 200.0

    mean_bright = float(np.mean(gray))
    bright_score = 1.0 - abs(mean_bright - 128) / 128.0
    bright_score = max(0.0, bright_score)

    pose_penalty = (abs(yaw) + abs(pitch)) / 60.0
    pose_score = max(0.0, 1.0 - pose_penalty)

    return round(sharpness * 0.4 + bright_score * 0.3 + pose_score * 0.3, 3)


def label(q: float) -> str:
    if q >= 0.75:
        return "excellent"
    if q >= GOOD_THRESHOLD:
        return "good"
    if q >= 0.35:
        return "fair"
    return "poor"
