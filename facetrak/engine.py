"""FaceTrak engine: detection, recognition, tracking, analysis pipeline.

Per frame:
  1. YuNet detects all faces on a downscaled frame (fast).
  2. The IoU tracker assigns stable IDs and smooths identities.
  3. SFace embeddings identify each tracked face (rate-limited per track).
  4. The FaceAnalyzer extracts pose + expression for the primary face.
  5. Servo follows the primary face; presence events are logged.
"""
import datetime
import logging
from pathlib import Path

import cv2
import numpy as np

from . import config
from .analysis import FaceAnalyzer, FaceMetrics
from .detection import YuNetDetector
from .events import PresenceLog
from .facedb import FaceDatabase
from .notifier import Notifier
from .recorder import VideoRecorder
from .servo import PanTiltController
from .tracker import FaceTracker, Track

logger = logging.getLogger(__name__)

_MAX_SAMPLES = 20            # embeddings collected per registration
_MIN_REGISTER_SAMPLES = 3
_EMBED_EVERY_N_FRAMES = 5    # re-identify a named track this often
_DEFAULT_THRESHOLD = 0.36    # SFace cosine similarity

_COLOR_KNOWN = (0, 255, 0)
_COLOR_UNKNOWN = (0, 165, 255)
_COLOR_HUD = (200, 200, 200)


class FaceEngine:
    def __init__(self):
        self.cfg = config.load()
        self.db = FaceDatabase()
        self.servo = PanTiltController(self.cfg)
        self.analyzer = FaceAnalyzer()
        self.recorder = VideoRecorder()
        self.notifier = Notifier()
        self.tracker = FaceTracker()
        self.presence = PresenceLog()
        self.detector: YuNetDetector | None = None
        self.cap = None
        self.running = False

        self.blur_enabled = self.cfg.get("blur_unknown", False)
        self.servo_enabled = False
        threshold = self.cfg.get("recog_threshold", _DEFAULT_THRESHOLD)
        # Legacy configs carry the old HOG-scale threshold (~0.55), which
        # would make SFace reject nearly everything — migrate it.
        if threshold > 0.5:
            threshold = _DEFAULT_THRESHOLD
            self.cfg["recog_threshold"] = threshold
        self.recog_threshold = threshold

        self._frame: np.ndarray | None = None
        self._detect_w = self.cfg.get("detect_width", 480)
        self._frame_count = 0
        self._samples_buffer: list[np.ndarray] = []
        self._capturing = False

        self.metrics = FaceMetrics()
        self.last_face_center = (0, 0)
        self.last_face_size = (0, 0)
        self.current_pan = 90.0
        self.current_tilt = 90.0

        self.overlay_text = ""
        self.current_cam_idx = self.cfg.get("camera", 0)

    # ── lifecycle ────────────────────────────────────────

    def _open_capture(self, source):
        if isinstance(source, str) and ("://" in source
                                        or source.startswith("/")):
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
        if self.detector is None:
            self.detector = YuNetDetector()
        self.db.load()
        self.analyzer.load_model(w, h)
        self.tracker.reset()
        self.running = True
        return True

    def switch_camera(self, cam_idx: int) -> bool:
        was_running = self.running
        was_rec = False
        if was_running:
            was_rec = self.recorder.recording
            if was_rec:
                self.recorder.stop()
            self.cap.release()
            self.cap = None
            self.running = False
        ok = self.start(cam_idx)
        if ok and was_running and was_rec:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.recorder.start(w, h)
        return ok

    def stop(self):
        self.running = False
        for track in self.tracker.tracks:
            self.presence.left(track.name, track.track_id,
                               track.dwell, track.blink_count)
        self.tracker.reset()
        if self.recorder.recording:
            self.recorder.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.servo.connected:
            self.servo.disconnect()
        config.save(self.cfg)

    # ── per-frame pipeline ───────────────────────────────

    def step(self) -> np.ndarray | None:
        if not self.running or self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self._frame_count += 1

        small, scale = self._downscale(frame)
        detections = self.detector.detect(small, scale=1.0 / scale)
        active, ended = self.tracker.update(detections)
        for track in active:
            if not track.announced:
                self.presence.appeared(track.name, track.track_id)
                track.announced = True
        self._log_ended(ended)
        self._identify(small, active)

        primary = self.tracker.largest()
        self._update_primary(frame, primary)
        self._draw_overlay(frame, active)

        self.recorder.write(frame)
        self._frame = frame
        return frame

    def _downscale(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = self._detect_w / w
        small = cv2.resize(frame, (self._detect_w, int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)
        return small, scale

    def _identify(self, small: np.ndarray, tracks: list[Track]):
        """Run recognition on tracks, rate-limited once a name is stable."""
        for track in tracks:
            needs_id = (track.name is None
                        or self._frame_count % _EMBED_EVERY_N_FRAMES == 0)
            if not needs_id:
                continue
            emb = self.db.embed(small, track.det.row)
            name, sim = self.db.predict(emb, self.recog_threshold)
            track.vote(name, sim)
            if track.name and self.cfg.get("notifications", True):
                self.notifier.notify(track.name)
            if self._capturing and track.name is None and emb is not None:
                if len(self._samples_buffer) < _MAX_SAMPLES:
                    self._samples_buffer.append(emb)

    def _log_ended(self, ended: list[Track]):
        for track in ended:
            self.presence.left(track.name, track.track_id,
                               track.dwell, track.blink_count)

    def _update_primary(self, frame: np.ndarray, primary: Track | None):
        if primary is None:
            self.last_face_center = (0, 0)
            self.last_face_size = (0, 0)
            return
        cx, cy = primary.det.center
        self.last_face_center = (cx, cy)
        self.last_face_size = (primary.det.w, primary.det.h)

        h, w = frame.shape[:2]
        if self.servo_enabled:
            self.current_pan, self.current_tilt = self.servo.update(
                cx - w // 2, cy - h // 2, w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.metrics = self.analyzer.analyze(rgb)
        primary.update_blink(self.metrics.eyes_closed)

    # ── rendering ────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray, tracks: list[Track]):
        h, w = frame.shape[:2]
        for track in tracks:
            self._draw_track(frame, track, w, h)
        if self.recorder.recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        self._draw_hud(frame, h)

    def _draw_track(self, frame: np.ndarray, track: Track, w: int, h: int):
        x, y, fw, fh = track.bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + fw), min(h, y + fh)
        known = track.name is not None
        color = _COLOR_KNOWN if known else _COLOR_UNKNOWN

        if self.blur_enabled and not known and x2 > x1 and y2 > y1:
            k = max(1, min(fw, fh) // 6) | 1
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(
                frame[y1:y2, x1:x2], (k, k), 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (f"#{track.track_id} {track.name} ({track.sim:.2f})"
                 if known else f"#{track.track_id} Unknown")
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"{track.dwell:.0f}s", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_hud(self, frame: np.ndarray, h: int):
        m = self.metrics
        lines = [
            f"Pan: {self.current_pan:.1f}  Tilt: {self.current_tilt:.1f}",
            f"Yaw: {m.yaw:.1f}  Pitch: {m.pitch:.1f}  Roll: {m.roll:.1f}",
            (f"Emotion: {m.emotion}  Smile: {m.smile:.2f}  "
             f"Attentive: {'yes' if m.attentive else 'no'}"),
            f"Faces: {len(self.tracker.active)}",
        ]
        if self._capturing:
            lines.append(f"Capturing {len(self._samples_buffer)}"
                         f"/{_MAX_SAMPLES} samples")
        if self.overlay_text:
            lines.append(self.overlay_text)
        y0 = h - 20 * len(lines) - 8
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, y0 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_HUD, 1)

    # ── commands ─────────────────────────────────────────

    def capture_samples(self):
        """Begin collecting registration embeddings on subsequent frames."""
        self._samples_buffer = []
        self._capturing = True

    def register(self, name: str) -> bool:
        self._capturing = False
        samples, self._samples_buffer = self._samples_buffer, []
        if len(samples) < _MIN_REGISTER_SAMPLES:
            return False
        ok = self.db.register(name, samples)
        if ok:
            # Re-identify everyone now that the database changed.
            for track in self.tracker.tracks:
                track.name = None
        return ok

    def snapshot(self) -> str | None:
        if self._frame is None:
            return None
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"snapshot_{ts}.png"
        cv2.imwrite(path, self._frame)
        return str(Path(path).resolve())

    def live_faces(self) -> list[dict]:
        """Structured view of all currently tracked faces (for MCP/UI)."""
        return [
            {
                "id": t.track_id,
                "name": t.name or "unknown",
                "similarity": round(t.sim, 3),
                "bbox": t.bbox,
                "dwell_s": round(t.dwell, 1),
                "blinks": t.blink_count,
            }
            for t in self.tracker.active
        ]

    def toggle_record(self):
        if self.recorder.recording:
            self.recorder.stop()
        elif self._frame is not None:
            h, w = self._frame.shape[:2]
            self.recorder.start(w, h)

    def toggle_blur(self) -> bool:
        self.blur_enabled = not self.blur_enabled
        self.cfg["blur_unknown"] = self.blur_enabled
        config.save(self.cfg)
        return self.blur_enabled

    def toggle_servo(self) -> bool:
        self.servo_enabled = not self.servo_enabled
        self.servo.enabled = self.servo_enabled
        return self.servo_enabled

    def set_overlay(self, text: str):
        self.overlay_text = text

    # Backwards-compatible pose accessors (used by UI/MCP status lines).
    @property
    def current_yaw(self) -> float:
        return self.metrics.yaw

    @property
    def current_pitch(self) -> float:
        return self.metrics.pitch

    @property
    def current_roll(self) -> float:
        return self.metrics.roll
