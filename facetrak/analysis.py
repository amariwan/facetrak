"""Per-face analysis: head pose (solvePnP) + expression metrics (blendshapes).

Uses the MediaPipe FaceLandmarker with blendshape output to derive smile,
eye openness, mouth/brow activity, a coarse emotion label, and an attention
flag (is the person looking roughly at the camera).
"""
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/latest/"
              "face_landmarker.task")
_MODEL_PATH = Path("face_landmarker.task")

# Generic 3D face model points: nose, chin, eye corners, mouth corners.
_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)
_LANDMARK_IDXS = [4, 152, 33, 263, 61, 291]

_ATTENTION_YAW = 25.0    # degrees within which we call it "attentive"
_ATTENTION_PITCH = 20.0
_EYES_CLOSED_SCORE = 0.5


@dataclass
class FaceMetrics:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    smile: float = 0.0
    mouth_open: float = 0.0
    brow_raise: float = 0.0
    eye_left: float = 1.0     # openness: 1 open, 0 closed
    eye_right: float = 1.0
    emotion: str = "neutral"
    attentive: bool = False
    blendshapes: dict[str, float] = field(default_factory=dict)

    @property
    def eyes_closed(self) -> bool:
        return self.eye_left < 0.5 and self.eye_right < 0.5


def _classify_emotion(bs: dict[str, float]) -> str:
    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    brow_up = bs.get("browInnerUp", 0)
    brow_down = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
    jaw_open = bs.get("jawOpen", 0)

    if brow_up > 0.5 and jaw_open > 0.25:
        return "surprised"
    if smile > 0.45:
        return "happy"
    if brow_down > 0.5:
        return "angry"
    if frown > 0.35 or (brow_up > 0.45 and smile < 0.1):
        return "sad"
    return "neutral"


class FaceAnalyzer:
    def __init__(self):
        self.landmarker = None
        self.ready = False
        self._cam_matrix: np.ndarray | None = None
        self._last = FaceMetrics()

    def load_model(self, img_w: int, img_h: int):
        if not _MODEL_PATH.exists():
            logger.info("Downloading face landmarker model (~5MB)...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        try:
            opts = vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(_MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=True,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(opts)
            fl = img_w * 0.8
            self._cam_matrix = np.array([
                [fl, 0, img_w / 2],
                [0, fl, img_h / 2],
                [0, 0, 1],
            ], dtype=np.float64)
            self.ready = True
        except Exception:
            logger.warning("Face landmarker unavailable, analysis disabled",
                           exc_info=True)
            self.ready = False

    def analyze(self, rgb_frame: np.ndarray) -> FaceMetrics:
        """Analyze the most prominent face; returns last metrics on miss."""
        if not self.ready or self.landmarker is None:
            return self._last
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=rgb_frame)
            result = self.landmarker.detect(mp_img)
        except (cv2.error, ValueError, RuntimeError):
            return self._last
        if not result.face_landmarks:
            return self._last

        m = FaceMetrics()
        m.yaw, m.pitch, m.roll = self._solve_pose(
            result.face_landmarks[0], rgb_frame.shape)

        if result.face_blendshapes:
            bs = {c.category_name: c.score
                  for c in result.face_blendshapes[0]}
            m.blendshapes = bs
            m.smile = (bs.get("mouthSmileLeft", 0)
                       + bs.get("mouthSmileRight", 0)) / 2
            m.mouth_open = bs.get("jawOpen", 0.0)
            m.brow_raise = bs.get("browInnerUp", 0.0)
            m.eye_left = 1.0 - bs.get("eyeBlinkLeft", 0.0)
            m.eye_right = 1.0 - bs.get("eyeBlinkRight", 0.0)
            m.emotion = _classify_emotion(bs)

        m.attentive = (abs(m.yaw) < _ATTENTION_YAW
                       and abs(m.pitch) < _ATTENTION_PITCH)
        self._last = m
        return m

    def _solve_pose(self, landmarks, shape) -> tuple[float, float, float]:
        h, w = shape[:2]
        pts = np.array([(landmarks[i].x * w, landmarks[i].y * h)
                        for i in _LANDMARK_IDXS], dtype=np.float64)
        try:
            ok, rvec, _ = cv2.solvePnP(
                _3D_MODEL, pts, self._cam_matrix, None,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return self._last.yaw, self._last.pitch, self._last.roll
            rmat, _ = cv2.Rodrigues(rvec)
        except cv2.error:
            return self._last.yaw, self._last.pitch, self._last.roll
        sy = np.sqrt(rmat[1, 0] ** 2 + rmat[2, 0] ** 2)
        if sy <= 1e-6:
            return self._last.yaw, self._last.pitch, self._last.roll
        yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
        roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
        return yaw, pitch, roll
