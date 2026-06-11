import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from pathlib import Path
from . import config
from .facedb import FaceDatabase
from .servo import PanTiltController
from .pose import HeadPoseEstimator
from .recorder import VideoRecorder
from .notifier import Notifier

_DETECTOR_URL = ("https://storage.googleapis.com/mediapipe-models/"
                 "face_detector/blaze_face_short_range/float16/latest/"
                 "face_detector.task")
_DETECTOR_PATH = Path("face_detector.task")

_SAMPLE_COUNT = 5


class FaceEngine:
    def __init__(self):
        self.cfg = config.load()
        self.db = FaceDatabase()
        self.servo = PanTiltController(self.cfg)
        self.pose = HeadPoseEstimator()
        self.recorder = VideoRecorder()
        self.notifier = Notifier()
        self.detector = None
        self.cap = None
        self.running = False
        self.blur_enabled = self.cfg.get("blur_unknown", False)
        self.servo_enabled = False
        self.recog_threshold = self.cfg.get("recog_threshold", 0.55)

        self._frame = None
        self._detections = []
        self._detect_w = self.cfg.get("detect_width", 480)
        self._samples_buffer = []

        self.last_face_center = (0, 0)
        self.last_face_size = (0, 0)
        self.current_pan = 90.0
        self.current_tilt = 90.0
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0

        self.overlay_text = ""
        self.status_extra = ""
        self.current_cam_idx = self.cfg.get("camera", 0)

    def _ensure_detector(self):
        if self.detector is not None:
            return
        if not _DETECTOR_PATH.exists():
            import urllib.request
            print("[Engine] downloading face detector (~100KB)...")
            urllib.request.urlretrieve(_DETECTOR_URL, _DETECTOR_PATH)
        opts = vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(_DETECTOR_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        self.detector = vision.FaceDetector.create_from_options(opts)

    def _open_capture(self, source):
        if isinstance(source, str) and ("://" in source or source.startswith("/")):
            return cv2.VideoCapture(source)
        cid = int(source)
        cap = cv2.VideoCapture(cid, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cid)
        return cap

    def start(self, cam_idx: int | None = None) -> bool:
        if cam_idx is not None:
            self.current_cam_idx = cam_idx
            self.cfg["camera"] = cam_idx
            config.save(self.cfg)
        src = config.source(self.cfg, self.current_cam_idx)
        self.cap = self._open_capture(src)
        if not self.cap or not self.cap.isOpened():
            return False
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._ensure_detector()
        self.db.load()
        self.pose.load_model(w, h)
        self.running = True
        return True

    def switch_camera(self, cam_idx: int) -> bool:
        was_running = self.running
        if was_running:
            was_rec = self.recorder.recording
            if was_rec:
                self.recorder.stop()
            self.cap.release()
            self.cap = None
            self.running = False
        ok = self.start(cam_idx)
        if ok and was_running:
            if was_rec:
                h, w = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                self.recorder.start(w, h)
        return ok

    def stop(self):
        self.running = False
        if self.recorder.recording:
            self.recorder.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.servo.connected:
            self.servo.disconnect()
        config.save(self.cfg)

    def step(self) -> np.ndarray | None:
        if not self.running or self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self._run_detection(frame)
        face = self._handle_best_face(frame)
        self._draw_overlay(frame, face)
        self.recorder.write(frame)
        self._frame = frame
        return frame

    def _run_detection(self, frame: np.ndarray):
        if self.detector is None:
            self._detections = []
            return
        h, w = frame.shape[:2]
        scale = self._detect_w / w
        small = cv2.resize(frame, (self._detect_w, int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)
        self._detections = []
        if result.detections:
            for d in result.detections:
                rect = d.bounding_box
                scale_inv = 1.0 / scale
                self._detections.append({
                    "x": int(rect.origin_x * scale_inv),
                    "y": int(rect.origin_y * scale_inv),
                    "w": int(rect.width * scale_inv),
                    "h": int(rect.height * scale_inv),
                    "score": d.categories[0].score,
                })

    def _handle_best_face(self, frame: np.ndarray) -> dict | None:
        if not self._detections:
            self.last_face_center = (0, 0)
            self.last_face_size = (0, 0)
            return None

        best = max(self._detections, key=lambda d: d["score"])
        x, y, fw, fh = best["x"], best["y"], best["w"], best["h"]
        cx, cy = x + fw // 2, y + fh // 2
        self.last_face_center = (cx, cy)
        self.last_face_size = (fw, fh)

        if self.servo_enabled:
            h, w = frame.shape[:2]
            self.current_pan, self.current_tilt = self.servo.update(
                cx - w // 2, cy - h // 2, w, h)

        face_crop = frame[max(0, y):y + fh, max(0, x):x + fw]
        name = None
        if face_crop.size > 0:
            name, sim = self.db.predict(face_crop, self.recog_threshold)
            best["name"] = name
            best["sim"] = sim
            if name and self.cfg.get("notifications", True):
                self.notifier.notify(name)

            if self.blur_enabled and name is None:
                k = max(1, min(fw, fh) // 6) | 1
                frame[y:y + fh, x:x + fw] = cv2.GaussianBlur(
                    frame[y:y + fh, x:x + fw], (k, k), 0)

            if self._samples_buffer is not None and name is None and face_crop.size > 0:
                small = cv2.resize(face_crop, (128, 128))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = np.mean(gray)
                if sharpness > 30 and 40 < brightness < 220:
                    self._samples_buffer.append(small)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_yaw, self.current_pitch, self.current_roll = \
            self.pose.estimate(rgb)

        return best

    def _draw_overlay(self, frame: np.ndarray, face: dict | None):
        h, w = frame.shape[:2]
        if face:
            x, y, fw, fh = face["x"], face["y"], face["w"], face["h"]
            name = face.get("name")
            color = (0, 255, 0) if name else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            label = "Unknown"
            if name:
                sim = face.get("sim", 0)
                label = f"{name} ({sim:.2f})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.recorder.recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        lines = [
            f"Pan: {self.current_pan:.1f}  Tilt: {self.current_tilt:.1f}",
            f"Yaw: {self.current_yaw:.1f}  Pitch: {self.current_pitch:.1f}  Roll: {self.current_roll:.1f}",
        ]
        if face:
            cx, cy = self.last_face_center
            lines.append(f"Face: ({cx}, {cy})  {self.last_face_size[0]}x{self.last_face_size[1]}")
        if self.overlay_text:
            lines.append(self.overlay_text)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, h - 60 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def capture_samples(self) -> list[np.ndarray]:
        self._samples_buffer = []
        return self._samples_buffer

    def register(self, name: str):
        if not self._samples_buffer or len(self._samples_buffer) < 2:
            return False
        self.db.register(name, self._samples_buffer)
        self._samples_buffer = []
        return True

    def toggle_record(self):
        if self.recorder.recording:
            self.recorder.stop()
        else:
            if self._frame is not None:
                h, w = self._frame.shape[:2]
                self.recorder.start(w, h)

    def toggle_blur(self):
        self.blur_enabled = not self.blur_enabled
        self.cfg["blur_unknown"] = self.blur_enabled
        config.save(self.cfg)
        return self.blur_enabled

    def toggle_servo(self):
        self.servo_enabled = not self.servo_enabled
        self.servo.enabled = self.servo_enabled
        return self.servo_enabled

    def set_overlay(self, text: str):
        self.overlay_text = text
