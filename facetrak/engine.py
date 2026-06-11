"""FaceTrak engine — detection, recognition, tracking, and all analysis.

Per-frame pipeline:
  1. YuNet detects all faces on a downscaled frame.
  2. IoU tracker assigns stable IDs; re-ID matches returning faces.
  3. SFace embeddings identify each track (rate-limited).
  4. Age/gender estimated once per track (cached _AGE_CACHE_FRAMES frames).
  5. FaceAnalyzer computes pose + gaze + expression for the primary face.
  6. Liveness checker gates registration samples.
  7. Quality score filters registration samples.
  8. Heatmap, crowd monitor, and emotion timeline are updated.
  9. Servo follows the selected target face.
  10. Overlay is drawn and the frame is returned.
"""
import datetime
import logging
from pathlib import Path

import cv2
import numpy as np

from . import config, db
from .age_gender import AgeGenderEstimator
from .analysis import FaceAnalyzer, FaceMetrics
from .crowd import CrowdMonitor
from .detection import YuNetDetector
from .events import PresenceLog
from .facedb import FaceDatabase
from .heatmap import FaceHeatmap
from .liveness import LivenessChecker
from .notifier import Notifier
from .quality import score as quality_score, GOOD_THRESHOLD
from .recorder import VideoRecorder
from .servo import PanTiltController
from .stats import EmotionTimeline
from .tracker import FaceTracker, Track

logger = logging.getLogger(__name__)

_MAX_SAMPLES       = 20
_MIN_REG_SAMPLES   = 3
_EMBED_EVERY       = 5     # frames between re-identification passes
_AGE_CACHE_FRAMES  = 90    # frames between age/gender re-inference per track
_DEFAULT_THRESHOLD = 0.36

_COLOR_KNOWN   = (0, 255, 0)
_COLOR_UNKNOWN = (0, 165, 255)
_COLOR_HUD     = (200, 200, 200)

# Servo target modes
SERVO_TARGET_LARGEST  = "largest"
SERVO_TARGET_KNOWN    = "known"
SERVO_TARGET_UNKNOWN  = "unknown"


class FaceEngine:
    def __init__(self):
        self.cfg = config.load()
        db.init()

        self.db          = FaceDatabase()
        self.servo       = PanTiltController(self.cfg)
        self.analyzer    = FaceAnalyzer()
        self.recorder    = VideoRecorder()
        self.notifier    = Notifier()
        self.tracker     = FaceTracker()
        self.presence    = PresenceLog()
        self.age_gender  = AgeGenderEstimator()
        self.heatmap     = FaceHeatmap()
        self.crowd       = CrowdMonitor()
        self.timeline    = EmotionTimeline()
        self.liveness    = LivenessChecker()

        self.detector: YuNetDetector | None = None
        self.cap   = None
        self.running = False

        self.blur_enabled    = self.cfg.get("blur_unknown", False)
        self.blur_persons: set[str] = set(self.cfg.get("blur_persons", []))
        self.servo_enabled   = False
        self.servo_target    = self.cfg.get("servo_target", SERVO_TARGET_LARGEST)
        self.heatmap_enabled = self.cfg.get("heatmap", False)

        threshold = self.cfg.get("recog_threshold", _DEFAULT_THRESHOLD)
        if threshold > 0.5:   # legacy HOG scale
            threshold = _DEFAULT_THRESHOLD
            self.cfg["recog_threshold"] = threshold
        self.recog_threshold = threshold

        self._frame: np.ndarray | None = None
        self._detect_w  = self.cfg.get("detect_width", 480)
        self._frame_no  = 0
        self._age_cache: dict[int, tuple[str, str, int]] = {}

        self._samples_buffer: list[np.ndarray] = []
        self._capturing = False

        self.metrics      = FaceMetrics()
        self.last_face_center = (0, 0)
        self.last_face_size   = (0, 0)
        self.current_pan  = 90.0
        self.current_tilt = 90.0
        self.overlay_text = ""
        self.current_cam_idx = self.cfg.get("camera", 0)

    # ── lifecycle ────────────────────────────────────────

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
        if self.detector is None:
            self.detector = YuNetDetector()
        self.db.load()
        self.analyzer.load_model(w, h)
        self.age_gender.load()
        self.tracker.reset()
        self.running = True
        return True

    def switch_camera(self, cam_idx: int) -> bool:
        was_rec = False
        if self.running:
            was_rec = self.recorder.recording
            if was_rec:
                self.recorder.stop()
            self.cap.release()
            self.cap = None
            self.running = False
        ok = self.start(cam_idx)
        if ok and was_rec:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.recorder.start(w, h)
        return ok

    def stop(self):
        self.running = False
        for t in self.tracker.tracks:
            self.presence.left(t.name, t.track_id, t.dwell, t.blink_count)
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
        self._frame_no += 1

        small, scale = self._downscale(frame)
        detections = self.detector.detect(small, scale=1.0 / scale)
        active, ended = self.tracker.update(detections)

        for t in active:
            if not t.announced:
                self.presence.appeared(t.name, t.track_id)
                t.announced = True
        self._log_ended(ended)
        self._identify(small, active)
        self._age_gender_update(small, active, frame)

        primary = self._select_servo_target(active)
        self._update_primary(frame, primary)

        if self.heatmap_enabled:
            centers = [t.det.center for t in active]
            self.heatmap.update(centers, frame.shape[:2])
            frame = self.heatmap.overlay(frame)

        self.crowd.tick(len(active))

        if primary and self.metrics.emotion:
            self.timeline.record(
                primary.track_id, primary.name,
                self.metrics.emotion, self.metrics.smile,
                self.metrics.attentive, self.metrics.yaw, self.metrics.pitch)

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
        for t in tracks:
            needs = (t.name is None
                     or self._frame_no % _EMBED_EVERY == 0)
            if not needs:
                continue
            emb = self.db.embed(small, t.det.row)
            if t.name is None and emb is not None:
                reid = self.tracker.try_reid(t.track_id, emb)
                if reid:
                    t.name = reid
                    t.announced = True
            self.tracker.set_embedding(t.track_id, emb)
            name, sim = self.db.predict(emb, self.recog_threshold)
            t.vote(name, sim)
            if t.name and self.cfg.get("notifications", True):
                self.notifier.notify(t.name)
            if self._capturing and emb is not None:
                self._maybe_collect(emb)

    def _maybe_collect(self, emb: np.ndarray):
        if len(self._samples_buffer) >= _MAX_SAMPLES:
            return
        q = quality_score(
            np.zeros((64, 64, 3), np.uint8),
            self.metrics.yaw, self.metrics.pitch)
        if q >= GOOD_THRESHOLD or not self.metrics.blendshapes:
            self._samples_buffer.append(emb)
        self.liveness.update(self.metrics.eyes_closed, self.metrics.yaw)

    def _age_gender_update(self, small: np.ndarray,
                           tracks: list[Track], full_frame: np.ndarray):
        for t in tracks:
            cached = self._age_cache.get(t.track_id)
            if cached and (self._frame_no - cached[2]) < _AGE_CACHE_FRAMES:
                t.age, t.gender = cached[0], cached[1]
                continue
            h_full, w_full = full_frame.shape[:2]
            x1 = max(0, t.det.x); y1 = max(0, t.det.y)
            x2 = min(w_full, t.det.x + t.det.w)
            y2 = min(h_full, t.det.y + t.det.h)
            crop = full_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            age, gender = self.age_gender.predict(crop)
            t.age, t.gender = age, gender
            self._age_cache[t.track_id] = (age, gender, self._frame_no)

    def _select_servo_target(self, active: list[Track]) -> Track | None:
        if not active:
            return None
        if self.servo_target == SERVO_TARGET_KNOWN:
            known = [t for t in active if t.name]
            return max(known, key=lambda t: t.det.area) if known else None
        if self.servo_target == SERVO_TARGET_UNKNOWN:
            unknown = [t for t in active if not t.name]
            return max(unknown, key=lambda t: t.det.area) if unknown else None
        return self.tracker.largest()

    def _log_ended(self, ended: list[Track]):
        for t in ended:
            self.presence.left(t.name, t.track_id, t.dwell, t.blink_count)
            self._age_cache.pop(t.track_id, None)

    def _update_primary(self, frame: np.ndarray, primary: Track | None):
        if primary is None:
            self.last_face_center = (0, 0)
            self.last_face_size   = (0, 0)
            return
        cx, cy = primary.det.center
        self.last_face_center = (cx, cy)
        self.last_face_size   = (primary.det.w, primary.det.h)
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
        for t in tracks:
            self._draw_track(frame, t, w, h)
        if self.recorder.recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        self._draw_hud(frame, h, w)

    def _draw_track(self, frame: np.ndarray, t: Track, w: int, h: int):
        x1 = max(0, t.det.x); y1 = max(0, t.det.y)
        x2 = min(w, t.det.x + t.det.w); y2 = min(h, t.det.y + t.det.h)
        known = t.name is not None
        should_blur = (
            (self.blur_enabled and not known)
            or (known and t.name in self.blur_persons)
        )
        if should_blur and x2 > x1 and y2 > y1:
            k = max(1, min(t.det.w, t.det.h) // 6) | 1
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(
                frame[y1:y2, x1:x2], (k, k), 0)
        color = _COLOR_KNOWN if known else _COLOR_UNKNOWN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (f"#{t.track_id} {t.name} ({t.sim:.2f}) {t.gender}/{t.age}"
                 if known else
                 f"#{t.track_id} Unknown {t.gender}/{t.age}")
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, f"{t.dwell:.0f}s | {t.blink_count}blinks",
                    (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def _draw_hud(self, frame: np.ndarray, h: int, w: int):
        m = self.metrics
        lines = [
            f"Pan:{self.current_pan:.1f} Tilt:{self.current_tilt:.1f}  "
            f"Yaw:{m.yaw:.1f} Pitch:{m.pitch:.1f} Roll:{m.roll:.1f}",
            f"Emotion:{m.emotion}  Smile:{m.smile:.2f}  "
            f"Gaze:{m.gaze_label}  Attn:{'Y' if m.attentive else 'N'}",
            f"Faces:{len(self.tracker.active)} | "
            f"Known:{len(self.db.known_names)}",
        ]
        if self._capturing:
            pct = int(100 * len(self._samples_buffer) / _MAX_SAMPLES)
            lines.append(
                f"REGISTERING  samples:{len(self._samples_buffer)}/{_MAX_SAMPLES} "
                f"[{'#'*(pct//10)}{' '*(10-pct//10)}]  "
                f"{self.liveness.status_line}")
        if self.overlay_text:
            lines.append(self.overlay_text)
        y0 = h - 20 * len(lines) - 6
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, y0 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, _COLOR_HUD, 1)

    # ── commands ─────────────────────────────────────────

    def capture_samples(self):
        self._samples_buffer = []
        self._capturing = True
        self.liveness.reset()

    def register(self, name: str) -> bool:
        self._capturing = False
        samples, self._samples_buffer = self._samples_buffer, []
        if len(samples) < _MIN_REG_SAMPLES:
            return False
        ok = self.db.register(name, samples)
        if ok:
            for t in self.tracker.tracks:
                t.name = None
        return ok

    def snapshot(self) -> str | None:
        if self._frame is None:
            return None
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"snapshot_{ts}.png"
        cv2.imwrite(path, self._frame)
        return str(Path(path).resolve())

    def live_faces(self) -> list[dict]:
        return [
            {
                "id":         t.track_id,
                "name":       t.name or "unknown",
                "similarity": round(t.sim, 3),
                "age":        t.age,
                "gender":     t.gender,
                "bbox":       t.bbox,
                "dwell_s":    round(t.dwell, 1),
                "blinks":     t.blink_count,
                "quality":    quality_score(
                    np.zeros((64, 64, 3), np.uint8),
                    self.metrics.yaw, self.metrics.pitch),
            }
            for t in self.tracker.active
        ]

    def toggle_record(self):
        if self.recorder.recording:
            self.recorder.stop()
        elif self._frame is not None:
            hw = self._frame.shape
            self.recorder.start(hw[1], hw[0])

    def toggle_blur(self) -> bool:
        self.blur_enabled = not self.blur_enabled
        self.cfg["blur_unknown"] = self.blur_enabled
        config.save(self.cfg)
        return self.blur_enabled

    def toggle_servo(self) -> bool:
        self.servo_enabled = not self.servo_enabled
        self.servo.enabled = self.servo_enabled
        return self.servo_enabled

    def toggle_heatmap(self) -> bool:
        self.heatmap_enabled = not self.heatmap_enabled
        self.cfg["heatmap"] = self.heatmap_enabled
        if not self.heatmap_enabled:
            self.heatmap.reset()
        config.save(self.cfg)
        return self.heatmap_enabled

    def set_blur_person(self, name: str, blur: bool):
        if blur:
            self.blur_persons.add(name)
        else:
            self.blur_persons.discard(name)
        self.cfg["blur_persons"] = sorted(self.blur_persons)
        config.save(self.cfg)

    def set_servo_target(self, mode: str):
        assert mode in (SERVO_TARGET_LARGEST, SERVO_TARGET_KNOWN,
                        SERVO_TARGET_UNKNOWN)
        self.servo_target = mode
        self.cfg["servo_target"] = mode
        config.save(self.cfg)

    def set_overlay(self, text: str):
        self.overlay_text = text

    def export_crowd_csv(self) -> str:
        return self.crowd.export_csv()

    def export_emotion_csv(self) -> str:
        return EmotionTimeline.export_csv()

    @property
    def current_yaw(self) -> float:
        return self.metrics.yaw

    @property
    def current_pitch(self) -> float:
        return self.metrics.pitch

    @property
    def current_roll(self) -> float:
        return self.metrics.roll
