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

# MediaPipe FaceLandmarker (478-pt mesh) iris landmark indices
_L_IRIS = 468   # left iris centre
_R_IRIS = 473   # right iris centre
_L_OUT  = 33    # left eye outer corner
_L_INN  = 133   # left eye inner corner
_R_INN  = 362   # right eye inner corner
_R_OUT  = 263   # right eye outer corner
_L_TOP  = 159   # left eye lid top
_L_BOT  = 145   # left eye lid bottom
_R_TOP  = 386
_R_BOT  = 374


def _eye_gaze(iris_idx: int, outer: int, inner: int,
              top: int, bot: int,
              lm, w: int, h: int) -> tuple[float, float]:
    """Return (horiz, vert) gaze in [-1,1] for one eye."""
    def p(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float64)
    eye_w = np.linalg.norm(p(inner) - p(outer)) + 1e-6
    eye_h = np.linalg.norm(p(top) - p(bot)) + 1e-6
    mid_h = (p(outer) + p(inner)) / 2
    mid_v = (p(top) + p(bot)) / 2
    iris = p(iris_idx)
    gh = float(np.clip((iris[0] - mid_h[0]) / (eye_w / 2), -1, 1))
    gv = float(np.clip((iris[1] - mid_v[1]) / (eye_h / 2), -1, 1))
    return gh, gv


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
    gaze_h: float = 0.0        # -1=left, 0=centre, 1=right
    gaze_v: float = 0.0        # -1=up, 0=centre, 1=down
    gaze_label: str = "centre"

    @property
    def eyes_closed(self) -> bool:
        return self.eye_left < 0.5 and self.eye_right < 0.5


def _gaze_label(h: float, v: float) -> str:
    parts = []
    if v < -0.25: parts.append("up")
    elif v > 0.25: parts.append("down")
    if h < -0.35: parts.append("left")
    elif h > 0.35: parts.append("right")
    return "-".join(parts) if parts else "centre"


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

        # Gaze — iris landmarks require >=478 points
        lm0 = result.face_landmarks[0]
        if len(lm0) >= 478:
            lh, lv = _eye_gaze(_L_IRIS, _L_OUT, _L_INN, _L_TOP, _L_BOT,
                                lm0, rgb_frame.shape[1], rgb_frame.shape[0])
            rh, rv = _eye_gaze(_R_IRIS, _R_INN, _R_OUT, _R_TOP, _R_BOT,
                                lm0, rgb_frame.shape[1], rgb_frame.shape[0])
            rh = -rh   # right iris is mirrored
            m.gaze_h = round((lh + rh) / 2, 3)
            m.gaze_v = round((lv + rv) / 2, 3)
            m.gaze_label = _gaze_label(m.gaze_h, m.gaze_v)

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
