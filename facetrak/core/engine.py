import datetime
import logging
import time

import cv2
import numpy as np

from facetrak.ai import GestureDetector, ObjectDetector, PoseEstimator
from facetrak.analysis import FaceAnalyzer  # None when MediaPipe missing
from facetrak.analysis.liveness import LivenessChecker
from facetrak.camera import CameraSource
from facetrak.camera import from_config as cameras_from_config
from facetrak.core.paths import data_dir
from facetrak.events import ZoneManager, load_masks
from facetrak.hardware.recorder import VideoRecorder
from facetrak.hardware.servo import PanTiltController
from facetrak.integrations.mqtt import MQTTPublisher
from facetrak.models import FaceDetection, FaceMetrics
from facetrak.models.track import Track
from facetrak.platform.resources import ResourceGuard
from facetrak.recog import (
    GOOD_THRESHOLD,
    AgeGenderEstimator,
    FaceDatabase,
    FaceTracker,
    YuNetDetector,
    quality_score,
    SCRFDDetector,
    ArcFaceRecognizer,
    InsightFaceAdapter,
)
from facetrak.device.management import ModelRegistry
from facetrak.sensors import AudioMonitor, DepthEstimator, PIRSensor
from facetrak.storage import (
    CrowdMonitor,
    EmotionTimeline,
    PresenceLog,
)
from facetrak.storage import db as dbmod
from facetrak.storage import (
    init as db_init,
)
from facetrak.utils.notifier import Notifier
from facetrak.viz.heatmap import FaceHeatmap
from facetrak.viz.overlay import draw_overlay

from . import config as cfgmod

logger = logging.getLogger(__name__)

_MAX_SAMPLES       = 20
_MIN_REG_SAMPLES   = 3
_EMBED_EVERY       = 5
_AGE_CACHE_FRAMES  = 90
_DEFAULT_THRESHOLD = 0.36

SERVO_TARGET_LARGEST  = "largest"
SERVO_TARGET_KNOWN    = "known"
SERVO_TARGET_UNKNOWN  = "unknown"


class FaceEngine:
    def __init__(self):
        self.cfg = cfgmod.load()
        db_init()

        self.db          = FaceDatabase()
        self.servo       = PanTiltController(self.cfg)
        self.analyzer    = FaceAnalyzer() if FaceAnalyzer else None
        self.recorder    = VideoRecorder()
        self.notifier    = Notifier()
        self.tracker     = FaceTracker()
        self.presence    = PresenceLog()
        self.age_gender  = AgeGenderEstimator()
        self.heatmap     = FaceHeatmap()
        self.crowd       = CrowdMonitor()
        self.timeline    = EmotionTimeline()
        self.liveness    = LivenessChecker()

        self.object_detector = ObjectDetector()
        self.pose_estimator  = PoseEstimator()
        self.gesture_detector = GestureDetector()

        self.objects_enabled  = self.cfg.get("objects_enabled", False)
        self.pose_enabled     = self.cfg.get("pose_enabled", False)
        self.gestures_enabled = self.cfg.get("gestures_enabled", False)

        self.audio_monitor = AudioMonitor()
        self.pir_sensor    = PIRSensor(pin=self.cfg.get("pir_gpio_pin", 17))
        self.depth_estimator = DepthEstimator()

        self.audio_enabled = self.cfg.get("audio_enabled", False)
        self.pir_enabled   = self.cfg.get("pir_enabled", False)
        self.depth_enabled = self.cfg.get("depth_enabled", False)

        self.last_audio_events: list = []
        self.last_motion: bool = False

        # Detector backend: YuNetDetector or SCRFDDetector
        self.detector: YuNetDetector | SCRFDDetector | None = None
        self.cam: CameraSource | None = None
        self.cameras: list[CameraSource] = []
        self.running = False

        # InsightFace adapter for optional models
        self.insightface_adapter: InsightFaceAdapter | None = None
        self.arcface: ArcFaceRecognizer | None = None

        # Model registry
        self.model_registry: ModelRegistry | None = None

        self.blur_enabled    = self.cfg.get("blur_unknown", False)
        self.blur_persons: set[str] = set(self.cfg.get("blur_persons", []))
        self.servo_enabled   = False
        self.servo_target    = self.cfg.get("servo_target", SERVO_TARGET_LARGEST)
        self.heatmap_enabled = self.cfg.get("heatmap", False)

        threshold = self.cfg.get("recog_threshold", _DEFAULT_THRESHOLD)
        if threshold > 0.5:
            logger.warning(
                "recog_threshold %r out of range, resetting to %s",
                threshold, _DEFAULT_THRESHOLD)
            threshold = _DEFAULT_THRESHOLD
            self.cfg["recog_threshold"] = threshold
        self.recog_threshold = threshold
        try:
            self._detect_every = max(
                1, int(self.cfg.get("detect_every", 1)))
        except (TypeError, ValueError):
            self._detect_every = 1
        self._detect_adaptive = bool(self.cfg.get("detect_adaptive", True))
        try:
            self._detect_target_ms = float(
                self.cfg.get("detect_target_ms", 33.0))
        except (TypeError, ValueError):
            self._detect_target_ms = 33.0
        self._step_ema_ms = 0.0
        self._last_detections: list = []
        self.resources = ResourceGuard(
            self.cfg.get("resources", {}).get("max_memory_mb", 0))

        self.zones = ZoneManager.from_config(self.cfg)
        self.masks = load_masks(self.cfg)
        self.mqtt = MQTTPublisher(self.cfg)
        self._subscribers: list = []
        self._recent_events: list[dict] = []

        self.perf = {
            "frames": 0, "drops": 0,
            "fps": 0.0, "step_ms": 0.0, "detect_ms": 0.0,
        }
        self._last_step_t: float | None = None

        self._frame: np.ndarray | None = None
        try:
            self._detect_w = int(self.cfg.get("detect_width", 480))
        except (TypeError, ValueError):
            self._detect_w = 480
        self._detect_w = max(160, min(1280, self._detect_w))
        self._frame_no  = 0
        self._age_cache: dict[int, tuple[str, str, int]] = {}

        self._samples_buffer: list[np.ndarray] = []
        self._capturing = False

        # InsightFace adapter (optional)
        if self.cfg.get("insightface", {}).get("enabled", False):
            self.insightface_adapter = InsightFaceAdapter(self.cfg.get("insightface", {}))
            if self.insightface_adapter and self.insightface_adapter.available:
                logger.info("InsightFace adapter initialized: %s", self.insightface_adapter.provider_name)
            else:
                logger.warning("InsightFace adapter not available: %s", self.insightface_adapter.init_error if self.insightface_adapter else "not created")

        # ArcFace recognizer (optional)
        if self.cfg.get("recognition_backend") == "arcface" and self.insightface_adapter and self.insightface_adapter.available:
            self.arcface = ArcFaceRecognizer(adapter=self.insightface_adapter)
            if self.arcface.ready:
                logger.info("ArcFace recognizer ready: %dd embeddings", self.arcface.embedding_dim)
            else:
                logger.warning("ArcFace recognizer not ready")

        # Model registry
        self.model_registry = ModelRegistry(self.cfg)

        self.metrics      = FaceMetrics()
        self.last_face_center = (0, 0)
        self.last_face_size   = (0, 0)
        self.current_pan  = 90.0
        self.current_tilt = 90.0
        self.overlay_text = ""
        self.current_cam_idx = self.cfg.get("camera", 0)

    # ── lifecycle ────────────────────────────────────────

    def start(self, cam_idx: int | None = None) -> bool:
        if cam_idx is not None:
            self.current_cam_idx = cam_idx
            self.cfg["camera"] = cam_idx
            cfgmod.save(self.cfg)

        try:
            self.cameras = cameras_from_config(self.cfg)
        except Exception as exc:
            logger.error("Failed to open cameras: %s", exc)
            return False

        if not self.cameras:
            logger.error("No cameras configured")
            return False
        idx = self.current_cam_idx
        if not isinstance(idx, int) or isinstance(idx, bool) \
                or not 0 <= idx < len(self.cameras):
            logger.error("Invalid camera index %r, falling back to 0", idx)
            idx = 0
            self.current_cam_idx = 0
        self.cam = self.cameras[idx]

        ok, probe = self.cam.read()
        if not ok or probe is None:
            logger.error("Primary camera did not return a frame")
            return False

        w, h = self.cam.resolution
        # Initialize detector based on backend config
        if self.detector is None:
            detector_backend = self.cfg.get("detector_backend", "yunet")
            if detector_backend == "scrfd" and self.insightface_adapter and self.insightface_adapter.available:
                self.detector = SCRFDDetector(adapter=self.insightface_adapter)
                if not self.detector.ready:
                    logger.warning("SCRFD not ready, falling back to YuNet")
                    self.detector = YuNetDetector()
            else:
                self.detector = YuNetDetector()

        self.db.load()
        if self.analyzer is not None:
            self.analyzer.load_model(w, h)
        self.age_gender.load()
        self.tracker.reset()
        if self.objects_enabled:
            self.object_detector.load()
        if self.pose_enabled:
            self.pose_estimator.load()
        if self.gestures_enabled:
            self.gesture_detector.load()
        if self.audio_enabled:
            self.audio_monitor.start()
        if self.pir_enabled:
            self.pir_sensor.start()
        if self.depth_enabled:
            self.depth_estimator.load()
        if self.mqtt.enabled:
            self.mqtt.start()
        self.running = True
        return True

    def switch_camera(self, cam_idx: int) -> bool:
        was_rec = False
        if self.running:
            was_rec = self.recorder.recording
            if was_rec:
                self.recorder.stop()
            for c in self.cameras:
                c.release()
            self.cameras = []
            self.cam = None
            self.running = False
        ok = self.start(cam_idx)
        if ok and was_rec:
            w, h = self.cam.resolution
            self.recorder.start(w, h)
        return ok

    def stop(self):
        self.running = False
        for t in self.tracker.tracks:
            self.presence.left(t.name, t.track_id, t.dwell, t.blink_count)
        self.tracker.reset()
        if self.recorder.recording:
            self.recorder.stop()
        for c in self.cameras:
            c.release()
        self.cameras = []
        self.cam = None
        self.pose_estimator.release()
        self.gesture_detector.release()
        self.audio_monitor.stop()
        self.pir_sensor.stop()
        self.mqtt.stop()
        if self.servo.connected:
            self.servo.disconnect()
        # Release InsightFace components
        if self.insightface_adapter:
            self.insightface_adapter.release()
        self.insightface_adapter = None
        self.arcface = None
        cfgmod.save(self.cfg)

    # ── per-frame pipeline ───────────────────────────────

    def step(self) -> np.ndarray | None:
        if not self.running or self.cam is None:
            return None
        t0 = time.monotonic()
        ret, frame = self.cam.read()
        if not ret or frame is None:
            self.perf["drops"] += 1
            return None
        if self.cfg.get("mirror", False):
            frame = cv2.flip(frame, 1)
        self._frame_no += 1

        small, scale = self._downscale(frame)
        for mask in self.masks:
            mask.apply(small)
        if self.detector is None or not self.detector.ready:
            detections = list(self._last_detections)
        elif self._detect_every > 1 and \
                self._frame_no % self._detect_every != 0:
            detections = list(self._last_detections)
        else:
            t_det = time.monotonic()
            detections = self.detector.detect(small, scale=1.0 / scale)
            self.perf["detect_ms"] = \
                (time.monotonic() - t_det) * 1000.0
            self._last_detections = detections
        active, ended = self.tracker.update(detections)

        for t in active:
            if not t.announced:
                self.presence.appeared(t.name, t.track_id)
                t.announced = True
        self._log_ended(ended)
        self._process_zones(active, frame)
        primary = self._select_servo_target(active)
        self._update_primary(frame, primary)
        self._identify(small, active, frame)
        self._age_gender_update(small, active, frame)

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

        frame = draw_overlay(
            frame, active, self.metrics, len(self.tracker.active),
            len(self.db.known_names), self.current_pan, self.current_tilt,
            recording=self.recorder.recording,
            blur_enabled=self.blur_enabled, blur_persons=self.blur_persons,
            capturing=self._capturing,
            samples_buffer_len=len(self._samples_buffer),
            max_samples=_MAX_SAMPLES,
            liveness_status=self.liveness.status_line,
            overlay_text=self.overlay_text,
        )

        if self.audio_enabled and self.audio_monitor.enabled:
            self.last_audio_events = self.audio_monitor.poll_all()
            for ev in self.last_audio_events:
                logger.debug("AudioEvent: %s @ %.1f dBFS", ev.event, ev.rms_db)

        if self.pir_enabled and self.pir_sensor.enabled:
            motion_events = self.pir_sensor.poll_all()
            if motion_events:
                self.last_motion = motion_events[-1].active

        if self.depth_enabled and self.depth_estimator.enabled:
            depth_map = self.depth_estimator.estimate(frame)
            if depth_map is not None:
                self.depth_estimator.draw_overlay(frame, depth_map)
                face_bboxes = [(t.det.x, t.det.y, t.det.w, t.det.h) for t in active]
                self.depth_estimator.draw_face_distances(frame, depth_map, face_bboxes)

        if self.objects_enabled and self.object_detector.enabled:
            objects = self.object_detector.detect(frame)
            self.object_detector.draw(frame, objects)

        if self.pose_enabled or self.gestures_enabled:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.pose_enabled and self.pose_estimator.enabled:
                pose = self.pose_estimator.process(rgb)
                if pose:
                    self.pose_estimator.draw(frame, pose)
            if self.gestures_enabled and self.gesture_detector.enabled:
                hands = self.gesture_detector.process(rgb)
                if hands:
                    self.gesture_detector.draw(frame, hands)

        self.recorder.write(frame)
        self._frame = frame
        self.perf["frames"] += 1
        dt = time.monotonic() - t0
        self.perf["step_ms"] = dt * 1000.0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        self.perf["fps"] = (0.9 * self.perf["fps"] + 0.1 * inst_fps
                            if self.perf["fps"] else inst_fps)
        self._adapt_detection(dt * 1000.0)
        if self.perf["frames"] % 300 == 0:
            self.perf["resources"] = self.resources.check()
        return frame

    def _adapt_detection(self, step_ms: float):
        """Bounded adaptive inference frequency (Pi optimization)."""
        ema = self._step_ema_ms
        self._step_ema_ms = (0.9 * ema + 0.1 * step_ms) if ema else step_ms
        if not self._detect_adaptive or self.perf["frames"] % 30:
            return
        if self._step_ema_ms > self._detect_target_ms and \
                self._detect_every < 4:
            self._detect_every += 1
            logger.info("Adaptive detection: every %d frames (EMA %.0f ms)",
                        self._detect_every, self._step_ema_ms)
        elif self._step_ema_ms < self._detect_target_ms * 0.55 and \
                self._detect_every > 1:
            self._detect_every -= 1
            logger.info("Adaptive detection: every %d frames (EMA %.0f ms)",
                        self._detect_every, self._step_ema_ms)
        self.perf["detect_every"] = self._detect_every

    def _downscale(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = self._detect_w / w
        small = cv2.resize(frame, (self._detect_w, int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)
        return small, scale

    def _align_face_for_arcface(self, frame: np.ndarray, det: FaceDetection) -> np.ndarray | None:
        """Align face for ArcFace using 5-point landmarks from detector."""
        if det.kps is not None and self.detector is not None and hasattr(self.detector, 'align_face'):
            return self.detector.align_face(frame, det)

        # Fallback: simple crop and resize to 112x112
        x, y, w, h = det.x, det.y, det.w, det.h
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    def _identify(self, small: np.ndarray, tracks: list, full_frame: np.ndarray):
        # Choose embedding extractor based on recognition backend
        recognition_backend = self.cfg.get("recognition_backend", "sface")

        for t in tracks:
            needs = (t.name is None
                     or self._frame_no % _EMBED_EVERY == 0)
            if not needs:
                continue

            if recognition_backend == "arcface" and self.arcface and self.arcface.ready:
                # ArcFace: need aligned face crop. ArcFace DB wiring is
                # optional; without a registered ArcFace gallery we can only
                # extract the embedding and skip identification.
                aligned = self._align_face_for_arcface(full_frame, t.det)
                emb = self.arcface.extract(aligned) if aligned is not None else None
                name, sim = None, 0.0
            else:
                # SFace (default)
                emb = self.db.embed(small, t.det.row)
                if t.name is None and emb is not None:
                    reid = self.tracker.try_reid(t.track_id, emb)
                    if reid:
                        t.name = reid
                        t.announced = True
                name, sim = self.db.predict(emb, self.recog_threshold)

            self.tracker.set_embedding(t.track_id, emb)
            t.vote(name, sim)
            if t.name and self.cfg.get("notifications", True):
                self.notifier.notify(t.name)
            if self._capturing and emb is not None:
                self._maybe_collect(emb, t, full_frame)

    def _maybe_collect(self, emb: np.ndarray, track: Track, frame: np.ndarray):
        if len(self._samples_buffer) >= _MAX_SAMPLES:
            return
        x1 = max(0, track.det.x); y1 = max(0, track.det.y)
        x2 = min(frame.shape[1], track.det.x + track.det.w)
        y2 = min(frame.shape[0], track.det.y + track.det.h)
        crop = frame[y1:y2, x1:x2]
        q = quality_score(crop, self.metrics.yaw, self.metrics.pitch)
        if q >= GOOD_THRESHOLD or not self.metrics.blendshapes:
            self._samples_buffer.append(emb)
        self.liveness.update(self.metrics.eyes_closed, self.metrics.yaw)

    def _age_gender_update(self, small: np.ndarray,
                           tracks: list, full_frame: np.ndarray):
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

    def _select_servo_target(self, active: list):
        if not active:
            return None
        if self.servo_target == SERVO_TARGET_KNOWN:
            known = [t for t in active if t.name]
            return max(known, key=lambda t: t.det.area) if known else None
        if self.servo_target == SERVO_TARGET_UNKNOWN:
            unknown = [t for t in active if not t.name]
            return max(unknown, key=lambda t: t.det.area) if unknown else None
        return self.tracker.largest()

    def _log_ended(self, ended: list):
        for t in ended:
            self.presence.left(t.name, t.track_id, t.dwell, t.blink_count)
            self._age_cache.pop(t.track_id, None)
            self.zones.forget_track(t.track_id)

    def _process_zones(self, active: list, frame: np.ndarray):
        if not self.zones.zones:
            return
        h, w = frame.shape[:2]
        for t in active:
            cx, cy = t.det.center
            for ev in self.zones.update(t.track_id, cx / w, cy / h,
                                        name=t.name):
                self._emit_event(ev)

    def _emit_event(self, ev: dict):
        try:
            dbmod.log_event(ev.get("kind", "event"), zone=ev.get("zone"),
                            track_id=ev.get("track_id"), name=ev.get("name"))
        except Exception as exc:
            logger.warning("Could not persist event: %s", exc)
        max_ev = self.cfg.get("events", {}).get("max_events", 1000)
        try:
            max_ev = max(1, int(max_ev))
        except (TypeError, ValueError):
            max_ev = 1000
        self._recent_events.append(ev)
        del self._recent_events[:-max_ev]
        if self.mqtt.enabled:
            payload = {"zone": ev.get("zone"),
                       "track_id": ev.get("track_id")}
            if self.mqtt.include_names:
                payload["name"] = ev.get("name")
            self.mqtt.publish(ev.get("kind", "event"), payload)
        for sub in list(self._subscribers):
            try:
                sub(ev)
            except Exception:
                logger.debug("Event subscriber failed", exc_info=True)

    # ── public event/perf/data API ──

    def on_event(self, callback):
        """Subscribe to zone/presence events (in-process fan-out)."""
        self._subscribers.append(callback)
        return callback

    def recent_events(self, limit: int = 100) -> list[dict]:
        return list(self._recent_events[-limit:])

    @property
    def current_frame(self) -> np.ndarray | None:
        return self._frame

    def perf_stats(self) -> dict:
        stats = dict(self.perf)
        stats["tracks"] = len(self.tracker.active)
        stats["detect_every"] = self._detect_every
        stats["detector_ready"] = bool(
            self.detector is not None and self.detector.ready)
        if self.cam is not None:
            stats["camera"] = self.cam.health.to_dict()
        return stats

    def purge_retention(self) -> dict:
        days = self.cfg.get("retention_days", 30)
        return dbmod.purge_older_than(days)

    def delete_runtime_data(self, include_faces: bool = False) -> dict:
        """Erase stored runtime data. Faces only with explicit consent."""
        result = dbmod.delete_all()
        if include_faces:
            from facetrak.recog.facedb import FACE_DIR
            removed = 0
            for p in FACE_DIR.glob("*.npy"):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
            result["face_files"] = removed
            self.db.load()
        return result

    def _update_primary(self, frame: np.ndarray, primary: Track | None):
        if primary is None:
            self.last_face_center = (0, 0)
            self.last_face_size   = (0, 0)
            if self.servo_enabled:
                moved = self.servo.note_target_lost()
                if moved is not None:
                    self.current_pan, self.current_tilt = moved
            return
        cx, cy = primary.det.center
        self.last_face_center = (cx, cy)
        self.last_face_size   = (primary.det.w, primary.det.h)
        h, w = frame.shape[:2]
        if self.servo_enabled:
            self.current_pan, self.current_tilt = self.servo.update(
                cx - w // 2, cy - h // 2, w, h)
        if self.analyzer is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.metrics = self.analyzer.analyze(rgb)
            primary.update_blink(self.metrics.eyes_closed)

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
        path = data_dir(self.cfg) / f"snapshot_{ts}.png"
        cv2.imwrite(str(path), self._frame)
        return str(path.resolve())

    def live_faces(self) -> list[dict]:
        frame = self._frame
        faces = []
        for t in self.tracker.active:
            q = 0.0
            if frame is not None:
                h, w = frame.shape[:2]
                x1 = max(0, t.det.x)
                y1 = max(0, t.det.y)
                x2 = min(w, t.det.x + t.det.w)
                y2 = min(h, t.det.y + t.det.h)
                if x2 > x1 and y2 > y1:
                    q = quality_score(
                        frame[y1:y2, x1:x2],
                        self.metrics.yaw, self.metrics.pitch)
            faces.append(
                {
                    "id":         t.track_id,
                    "name":       t.name or "unknown",
                    "similarity": round(t.sim, 3),
                    "age":        t.age,
                    "gender":     t.gender,
                    "bbox":       t.bbox,
                    "dwell_s":    round(t.dwell, 1),
                    "blinks":     t.blink_count,
                    "quality":    q,
                }
            )
        return faces

    def toggle_record(self):
        if self.recorder.recording:
            self.recorder.stop()
        elif self._frame is not None and self.cam is not None:
            w, h = self.cam.resolution
            self.recorder.start(w, h)

    def toggle_blur(self) -> bool:
        self.blur_enabled = not self.blur_enabled
        self.cfg["blur_unknown"] = self.blur_enabled
        cfgmod.save(self.cfg)
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
        cfgmod.save(self.cfg)
        return self.heatmap_enabled

    def toggle_mirror(self) -> bool:
        val = not self.cfg.get("mirror", False)
        self.cfg["mirror"] = val
        cfgmod.save(self.cfg)
        return val

    def set_blur_person(self, name: str, blur: bool):
        if blur:
            self.blur_persons.add(name)
        else:
            self.blur_persons.discard(name)
        self.cfg["blur_persons"] = sorted(self.blur_persons)
        cfgmod.save(self.cfg)

    def set_servo_target(self, mode: str):
        assert mode in (SERVO_TARGET_LARGEST, SERVO_TARGET_KNOWN,
                        SERVO_TARGET_UNKNOWN)
        self.servo_target = mode
        self.cfg["servo_target"] = mode
        cfgmod.save(self.cfg)

    def set_overlay(self, text: str):
        self.overlay_text = text

    def toggle_objects(self) -> bool:
        self.objects_enabled = not self.objects_enabled
        if self.objects_enabled and not self.object_detector.enabled:
            self.object_detector.load()
        self.cfg["objects_enabled"] = self.objects_enabled
        cfgmod.save(self.cfg)
        return self.objects_enabled

    def toggle_pose(self) -> bool:
        self.pose_enabled = not self.pose_enabled
        if self.pose_enabled and not self.pose_estimator.enabled:
            self.pose_estimator.load()
        self.cfg["pose_enabled"] = self.pose_enabled
        cfgmod.save(self.cfg)
        return self.pose_enabled

    def toggle_audio(self) -> bool:
        self.audio_enabled = not self.audio_enabled
        if self.audio_enabled and not self.audio_monitor.enabled:
            self.audio_monitor.start()
        elif not self.audio_enabled:
            self.audio_monitor.stop()
        self.cfg["audio_enabled"] = self.audio_enabled
        cfgmod.save(self.cfg)
        return self.audio_enabled

    def toggle_pir(self) -> bool:
        self.pir_enabled = not self.pir_enabled
        if self.pir_enabled and not self.pir_sensor.enabled:
            self.pir_sensor.start()
        elif not self.pir_enabled:
            self.pir_sensor.stop()
        self.cfg["pir_enabled"] = self.pir_enabled
        cfgmod.save(self.cfg)
        return self.pir_enabled

    def toggle_depth(self) -> bool:
        self.depth_enabled = not self.depth_enabled
        if self.depth_enabled and not self.depth_estimator.enabled:
            self.depth_estimator.load()
        self.cfg["depth_enabled"] = self.depth_enabled
        cfgmod.save(self.cfg)
        return self.depth_enabled

    def toggle_gestures(self) -> bool:
        self.gestures_enabled = not self.gestures_enabled
        if self.gestures_enabled and not self.gesture_detector.enabled:
            self.gesture_detector.load()
        self.cfg["gestures_enabled"] = self.gestures_enabled
        cfgmod.save(self.cfg)
        return self.gestures_enabled

    def export_crowd_csv(self) -> str:
        return self.crowd.export_csv(data_dir(self.cfg) / "crowd_stats.csv")

    def export_emotion_csv(self) -> str:
        return EmotionTimeline.export_csv(
            data_dir(self.cfg) / "emotion_log.csv")

    @property
    def current_yaw(self) -> float:
        return self.metrics.yaw

    @property
    def current_pitch(self) -> float:
        return self.metrics.pitch

    @property
    def current_roll(self) -> float:
        return self.metrics.roll
