"""Registration quality diagnostics (explicit rejection reasons).

Builds on recog/quality.py score/label without modifying it: explains WHY
a capture sample was accepted or rejected during face registration.
"""

from __future__ import annotations

import cv2
import numpy as np

from facetrak.recog.quality import GOOD_THRESHOLD, label as quality_label, score as quality_score

MIN_CROP_PX = 3


def diagnose(face_bgr: np.ndarray | None, yaw: float = 0.0,
             pitch: float = 0.0,
             threshold: float = GOOD_THRESHOLD) -> dict:
    """Assess one capture sample. Never raises; pure function of pixels."""
    reasons: list[str] = []
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return {"quality": 0.0, "label": "poor", "accepted": False,
                "reasons": ["empty-crop"],
                "components": {"sharpness": 0.0, "brightness": 0.0,
                               "pose": 0.0}}
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < MIN_CROP_PX or w < MIN_CROP_PX:
        return {"quality": 0.0, "label": "poor", "accepted": False,
                "reasons": [f"crop-too-small:{w}x{h}"],
                "components": {"sharpness": 0.0, "brightness": 0.0,
                               "pose": 0.0}}

    sharp = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 200.0) / 200.0
    mean_bright = float(np.mean(gray))
    bright = max(0.0, 1.0 - abs(mean_bright - 128) / 128.0)
    pose = max(0.0, 1.0 - (abs(yaw) + abs(pitch)) / 60.0)

    if sharp < 0.25:
        reasons.append(f"blurry:{sharp:.2f}")
    if mean_bright < 40:
        reasons.append(f"too-dark:{mean_bright:.0f}")
    elif mean_bright > 215:
        reasons.append(f"too-bright:{mean_bright:.0f}")
    if abs(yaw) > 25 or abs(pitch) > 20:
        reasons.append(f"head-turned:yaw={yaw:.0f},pitch={pitch:.0f}")

    q = quality_score(face_bgr, yaw, pitch)
    accepted = bool(q >= threshold)
    if not accepted and not reasons:
        reasons.append(f"below-threshold:{q:.2f}<{threshold:.2f}")
    return {"quality": q, "label": quality_label(q), "accepted": accepted,
            "reasons": reasons,
            "components": {"sharpness": round(sharp, 3),
                           "brightness": round(bright, 3),
                           "pose": round(pose, 3)}}
