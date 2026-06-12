import datetime
import logging
from pathlib import Path

import cv2
import numpy as np

from . import config as cfgmod
from facetrak.recog import (
    YuNetDetector, FaceTracker, FaceDatabase,
    quality_score, GOOD_THRESHOLD,
    AgeGenderEstimator,
)
from facetrak.analysis import FaceAnalyzer, LivenessChecker
from facetrak.viz.overlay import draw_overlay
from facetrak.viz.heatmap import FaceHeatmap
from facetrak.storage import (
    init as db_init,
    PresenceLog, CrowdMonitor, EmotionTimeline,
)
from facetrak.hardware.servo import PanTiltController
from facetrak.hardware.recorder import VideoRecorder
from facetrak.camera import CameraSource, from_config as cameras_from_config
from facetrak.ai import ObjectDetector, PoseEstimator, GestureDetector
from facetrak.sensors import AudioMonitor, PIRSensor, DepthEstimator
from facetrak.utils.notifier import Notifier
from facetrak.models import FaceMetrics
from facetrak.models.track import Track

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

        self.detector: YuNetDetector | None = None
        self.cam: CameraSource | None = None
        self.cameras: list[CameraSource] = []
        self.running = False

        self.blur_enabled    = self.cfg.get("blur_unknown", False)
        self.blur_persons: set[str] = set(self.cfg.get("blur_persons", []))
        self.servo_enabled   = False
        self.servo_target    = self.cfg.get("servo_target", SERVO_TARGET_LARGEST)
        self.heatmap_enabled = self.cfg.get("heatmap", False)

        threshold = self.cfg.get("recog_threshold", _DEFAULT_THRESHOLD)
        if threshold > 0.5:
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

        idx = min(self.current_cam_idx, len(self.cameras) - 1)
        self.cam = self.cameras[idx]

        ok, probe = self.cam.read()
        if not ok or probe is None:
            logger.error("Primary camera did not return a frame")
            return False

        w, h = self.cam.resolution
        if self.detector is None:
            self.detector = YuNetDetector()
        self.db.load()
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
        if self.servo.connected:
            self.servo.disconnect()
        cfgmod.save(self.cfg)

    # ── per-frame pipeline ───────────────────────────────

    def step(self) -> np.ndarray | None:
        if not self.running or self.cam is None:
            return None
        ret, frame = self.cam.read()
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
        return frame

    def _downscale(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = self._detect_w / w
        small = cv2.resize(frame, (self._detect_w, int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)
        return small, scale

    def _identify(self, small: np.ndarray, tracks: list, full_frame: np.ndarray):
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
                    frame[t.det.y:t.det.y+t.det.h, t.det.x:t.det.x+t.det.w],
                    self.metrics.yaw, self.metrics.pitch),
            }
            for t in self.tracker.active
        ]

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
