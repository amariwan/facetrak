import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/latest/"
              "face_landmarker.task")
_MODEL_PATH = Path("face_landmarker.task")

_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)

_LANDMARK_IDXS = [4, 152, 33, 263, 61, 291]


class HeadPoseEstimator:
    def __init__(self):
        self.landmarker = None
        self.ready = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self._cam_matrix: np.ndarray | None = None

    def load_model(self, img_w: int, img_h: int):
        if not _MODEL_PATH.exists():
            self._download()
        try:
            opts = vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(_MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=5,
                output_face_blendshapes=False,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(opts)
            self.ready = True
            fl = img_w * 0.8
            self._cam_matrix = np.array([
                [fl, 0, img_w / 2],
                [0, fl, img_h / 2],
                [0, 0, 1],
            ], dtype=np.float64)
        except Exception:
            self.ready = False

    @staticmethod
    def _download():
        import urllib.request
        print("[Pose] downloading model (~5MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

    def estimate(self, rgb_frame: np.ndarray) -> tuple[float, float, float]:
        if not self.ready or self.landmarker is None:
            return 0.0, 0.0, 0.0
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_img)
            if not result.face_landmarks:
                return 0.0, 0.0, 0.0
            lm = result.face_landmarks[0]
            h, w = rgb_frame.shape[:2]
            pts = np.array([(lm[i].x * w, lm[i].y * h)
                            for i in _LANDMARK_IDXS], dtype=np.float64)
            _, rvec, _ = cv2.solvePnP(
                _3D_MODEL, pts, self._cam_matrix, None,
                flags=cv2.SOLVEPNP_ITERATIVE)
            rmat, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rmat[1, 0]**2 + rmat[2, 0]**2)
            if sy > 1e-6:
                self.yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
                self.pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
                self.roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
            return self.yaw, self.pitch, self.roll
        except Exception:
            return 0.0, 0.0, 0.0
