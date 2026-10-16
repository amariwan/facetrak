"""PrivateFrame-Inspired Video Face Anonymization.

Pipeline:
1. Analyze video - detect/track faces, produce analysis JSON with timestamps
2. Edit masks (optional) - modify detection boxes in JSON
3. Render - apply Gaussian blur or mosaic to face regions using stored masks

No re-inference during render. Stable mask tracking across frames via IoU + embedding association.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from facetrak.models import FaceDetection
from facetrak.recog import FaceTracker, FaceDatabase, YuNetDetector, SCRFDDetector
from facetrak.recog.insightface_adapter import InsightFaceAdapter
from facetrak.recog.scrfd import create_scrfd_detector
from facetrak.recog.arcface import ArcFaceRecognizer, create_arcface_recognizer

logger = logging.getLogger(__name__)


class RedactionMethod(str, Enum):
    GAUSSIAN = "gaussian"
    MOSAIC = "mosaic"


@dataclass
class RedactionConfig:
    method: RedactionMethod = RedactionMethod.GAUSSIAN
    box_scale: float = 1.15          # Scale face box for redaction margin
    gaussian_ksize: int = 51         # Gaussian kernel size (odd)
    gaussian_sigma: float = 30.0     # Gaussian sigma
    mosaic_block: int = 20           # Mosaic block size


@dataclass
class AnalysisConfig:
    detector_backend: str = "yunet"       # "yunet" | "scrfd"
    recognition_backend: str = "none"     # "none" | "sface" | "arcface"
    detect_every: int = 1                 # Frame sampling
    max_faces: int = 50
    min_face_size: int = 20
    track_iou_threshold: float = 0.3
    track_max_missed: int = 15
    embed_every: int = 5                  # Embedding extraction interval


@dataclass
class FaceTrack:
    """Track data stored in analysis JSON."""
    track_id: int
    frame_idx: int
    timestamp: float
    bbox: list[int]           # [x, y, w, h] in frame coordinates
    score: float
    kps: list[list[float]] | None = None  # 5-point landmarks
    embedding: list[float] | None = None  # Recognition embedding
    name: str | None = None               # Matched identity
    similarity: float = 0.0
    redact: bool = True                   # Whether to redact this face


@dataclass
class VideoAnalysis:
    """Complete video analysis result (serializable to JSON)."""
    video_path: str
    video_hash: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    config: dict
    tracks: list[FaceTrack] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    version: str = "1.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "VideoAnalysis":
        d = json.loads(data)
        # Convert track dicts to FaceTrack
        d["tracks"] = [FaceTrack(**t) for t in d["tracks"]]
        d["config"] = d.get("config", {})
        return cls(**d)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> "VideoAnalysis":
        return cls.from_json(path.read_text())


class VideoAnonymizer:
    """Video face anonymization pipeline inspired by InsightFace PrivateFrame."""

    def __init__(
        self,
        analysis_cfg: AnalysisConfig | None = None,
        redaction_cfg: RedactionConfig | None = None,
        insightface_adapter: InsightFaceAdapter | None = None,
        face_db: FaceDatabase | None = None,
        reference_embeddings: dict[str, list[np.ndarray]] | None = None,
    ):
        self.analysis_cfg = analysis_cfg or AnalysisConfig()
        self.redaction_cfg = redaction_cfg or RedactionConfig()
        self.adapter = insightface_adapter
        self.face_db = face_db
        self.reference_embeddings = reference_embeddings or {}

        # Initialize detectors
        self.yunet = YuNetDetector()
        self.scrfd = None
        self.arcface = None

        if self.adapter and self.adapter.available:
            if self.analysis_cfg.detector_backend == "scrfd":
                self.scrfd = create_scrfd_detector({}, self.adapter)
            if self.analysis_cfg.recognition_backend == "arcface":
                self.arcface = create_arcface_recognizer({}, self.adapter)

        self.tracker = FaceTracker()
        self._cancelled = False
        self._progress_callback: Any = None

    def set_progress_callback(self, callback: Any) -> None:
        """Set callback(progress: float, status: str, current_frame: int, total_frames: int)."""
        self._progress_callback = callback

    def cancel(self) -> None:
        self._cancelled = True

    def _report_progress(self, progress: float, status: str, current: int, total: int) -> None:
        if self._progress_callback:
            self._progress_callback(progress, status, current, total)

    def _compute_video_hash(self, path: Path) -> str:
        """Compute SHA256 of video file for integrity."""
        import hashlib
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def analyze(
        self,
        video_path: Path,
        output_json: Path | None = None,
        max_frames: int | None = None,
    ) -> VideoAnalysis:
        """Phase 1: Analyze video - detect and track faces."""
        self._cancelled = False

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        if max_frames:
            frame_count = min(frame_count, max_frames)

        video_hash = self._compute_video_hash(video_path)

        # Select detector
        if self.analysis_cfg.detector_backend == "scrfd" and self.scrfd and self.scrfd.ready:
            detector = self.scrfd
        else:
            detector = self.yunet

        if not detector.ready:
            raise RuntimeError("No detector available")

        # Reset tracker
        self.tracker = FaceTracker()
        self.tracker._IOU_MATCH = self.analysis_cfg.track_iou_threshold
        self.tracker._MAX_MISSED = self.analysis_cfg.track_max_missed

        tracks: list[FaceTrack] = []
        frame_idx = 0

        self._report_progress(0.0, "analyzing", 0, frame_count)

        while frame_idx < frame_count and not self._cancelled:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # Sample frames
            if frame_idx % self.analysis_cfg.detect_every != 0:
                frame_idx += 1
                continue

            # Detect
            small, scale = self._downscale(frame)
            detections = detector.detect(small, scale=1.0 / scale)

            # Filter by min face size
            detections = [d for d in detections if d.w >= self.analysis_cfg.min_face_size and d.h >= self.analysis_cfg.min_face_size]
            detections = detections[:self.analysis_cfg.max_faces]

            # Track
            active, ended = self.tracker.update(detections)

            # Process active tracks
            for track in active:
                face_track = FaceTrack(
                    track_id=track.track_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    bbox=[track.det.x, track.det.y, track.det.w, track.det.h],
                    score=track.det.score,
                )

                # Extract landmarks if available
                if hasattr(track.det, 'kps') and track.det.kps is not None:
                    face_track.kps = track.det.kps.tolist()

                # Extract embedding periodically
                if self.analysis_cfg.recognition_backend != "none" and frame_idx % self.analysis_cfg.embed_every == 0:
                    emb = self._extract_embedding(frame, track.det)
                    if emb is not None:
                        face_track.embedding = emb.tolist()
                        # Try to identify
                        name, sim = self._identify(emb)
                        if name:
                            face_track.name = name
                            face_track.similarity = sim
                            # Check if this person should NOT be redacted (reference match)
                            face_track.redact = name not in self.reference_embeddings

                tracks.append(face_track)

            frame_idx += 1

            if frame_idx % 30 == 0:
                self._report_progress(frame_idx / frame_count, "analyzing", frame_idx, frame_count)

        cap.release()

        if self._cancelled:
            raise InterruptedError("Analysis cancelled")

        # Create analysis result
        analysis = VideoAnalysis(
            video_path=str(video_path),
            video_hash=video_hash,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            config={
                "analysis": asdict(self.analysis_cfg),
                "redaction": asdict(self.redaction_cfg),
            },
            tracks=tracks,
        )

        if output_json:
            analysis.save(output_json)
            logger.info("Analysis saved to %s (%d face tracks)", output_json, len(tracks))

        self._report_progress(1.0, "complete", frame_count, frame_count)
        return analysis

    def _downscale(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        detect_w = 640
        scale = detect_w / w
        small = cv2.resize(frame, (detect_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return small, scale

    def _extract_embedding(self, frame: np.ndarray, det: FaceDetection) -> np.ndarray | None:
        """Extract recognition embedding from face detection."""
        if self.analysis_cfg.recognition_backend == "arcface" and self.arcface and self.arcface.ready:
            # Align using SCRFD landmarks or fallback
            if hasattr(det, 'kps') and det.kps is not None:
                aligned = self.scrfd.align_face(frame, det) if self.scrfd else None
            else:
                aligned = None
            if aligned is not None:
                return self.arcface.extract(aligned)

        elif self.analysis_cfg.recognition_backend == "sface" and self.face_db:
            # Use SFace via FaceDatabase
            try:
                emb = self.face_db.embed(frame, det.row)
                return emb
            except Exception:
                pass

        return None

    def _identify(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Identify face against reference embeddings."""
        if not self.reference_embeddings:
            return None, 0.0

        best_name = None
        best_sim = 0.0

        for name, ref_embs in self.reference_embeddings.items():
            for ref_emb in ref_embs:
                if ref_emb is not None and ref_emb.shape == embedding.shape:
                    sim = float(np.dot(embedding, ref_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

        return best_name, best_sim

    def render(
        self,
        video_path: Path,
        analysis: VideoAnalysis,
        output_path: Path,
        redaction_cfg: RedactionConfig | None = None,
    ) -> dict:
        """Phase 2: Render anonymized video using analysis JSON (no re-inference)."""
        self._cancelled = False
        cfg = redaction_cfg or self.redaction_cfg

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Verify analysis matches video
        if analysis.width != width or analysis.height != height:
            logger.warning("Analysis dimensions (%dx%d) != video (%dx%d)", analysis.width, analysis.height, width, height)

        # Organize tracks by frame
        tracks_by_frame: dict[int, list[FaceTrack]] = {}
        for t in analysis.tracks:
            tracks_by_frame.setdefault(t.frame_idx, []).append(t)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_path}")

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        redacted_count = 0

        self._report_progress(0.0, "rendering", 0, total_frames)

        while True:
            ret, frame = cap.read()
            if not ret or self._cancelled:
                break

            # Get tracks for this frame
            frame_tracks = tracks_by_frame.get(frame_idx, [])

            # Apply redaction
            for track in frame_tracks:
                if not track.redact:
                    continue

                x, y, w, h = track.bbox
                # Scale box
                cx, cy = x + w // 2, y + h // 2
                new_w = int(w * cfg.box_scale)
                new_h = int(h * cfg.box_scale)
                x1 = max(0, cx - new_w // 2)
                y1 = max(0, cy - new_h // 2)
                x2 = min(width, x1 + new_w)
                y2 = min(height, y1 + new_h)

                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]

                if cfg.method == RedactionMethod.GAUSSIAN:
                    ksize = cfg.gaussian_ksize
                    if ksize % 2 == 0:
                        ksize += 1
                    blurred = cv2.GaussianBlur(roi, (ksize, ksize), cfg.gaussian_sigma)
                    frame[y1:y2, x1:x2] = blurred
                elif cfg.method == RedactionMethod.MOSAIC:
                    block = cfg.mosaic_block
                    small = cv2.resize(roi, (max(1, new_w // block), max(1, new_h // block)), interpolation=cv2.INTER_LINEAR)
                    mosaiced = cv2.resize(small, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = mosaiced

                redacted_count += 1

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 30 == 0:
                self._report_progress(frame_idx / total_frames, "rendering", frame_idx, total_frames)

        cap.release()
        writer.release()

        if self._cancelled:
            output_path.unlink(missing_ok=True)
            raise InterruptedError("Rendering cancelled")

        self._report_progress(1.0, "complete", total_frames, total_frames)
        return {
            "frames_processed": frame_idx,
            "faces_redacted": redacted_count,
            "output": str(output_path),
        }

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        redaction_cfg: RedactionConfig | None = None,
        max_frames: int | None = None,
    ) -> dict:
        """Full pipeline: analyze -> render."""
        output_dir.mkdir(parents=True, exist_ok=True)

        video_stem = video_path.stem
        analysis_path = output_dir / f"{video_stem}_privateframe.json"
        output_video = output_dir / f"{video_stem}_redacted.mp4"

        # Phase 1: Analyze
        logger.info("Analyzing video: %s", video_path)
        analysis = self.analyze(video_path, analysis_path, max_frames)

        # Phase 2: Render
        logger.info("Rendering anonymized video: %s", output_video)
        result = self.render(video_path, analysis, output_video, redaction_cfg)

        result["analysis_json"] = str(analysis_path)
        result["analysis"] = {
            "total_tracks": len(analysis.tracks),
            "unique_ids": len(set(t.track_id for t in analysis.tracks)),
            "identified": len([t for t in analysis.tracks if t.name]),
            "redacted": len([t for t in analysis.tracks if t.redact]),
        }

        return result


def create_anonymizer_from_config(config: dict, face_db: FaceDatabase | None = None) -> VideoAnonymizer:
    """Create VideoAnonymizer from FaceTrak config dict."""
    insightface_cfg = config.get("insightface", {})
    adapter = None
    if insightface_cfg.get("enabled", False):
        adapter = create_adapter(insightface_cfg)

    anon_cfg = config.get("anonymization", {})
    analysis_cfg = AnalysisConfig(
        detector_backend=anon_cfg.get("detector_backend", "yunet"),
        recognition_backend=anon_cfg.get("recognition_backend", "none"),
        detect_every=anon_cfg.get("detect_every", 1),
        max_faces=anon_cfg.get("max_faces", 50),
        min_face_size=anon_cfg.get("min_face_size", 20),
    )
    redaction_cfg = RedactionConfig(
        method=RedactionMethod(anon_cfg.get("method", "gaussian")),
        box_scale=anon_cfg.get("box_scale", 1.15),
        gaussian_ksize=anon_cfg.get("gaussian_ksize", 51),
        gaussian_sigma=anon_cfg.get("gaussian_sigma", 30.0),
        mosaic_block=anon_cfg.get("mosaic_block", 20),
    )

    # Load reference embeddings for selective redaction
    reference_embeddings = {}
    if face_db:
        for name in face_db.known_names:
            # Get embeddings from face database
            pass  # Would need to load from .npy files

    return VideoAnonymizer(analysis_cfg, redaction_cfg, adapter, face_db, reference_embeddings)


# CLI entry point for standalone use
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video face anonymization (PrivateFrame-inspired)")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--method", choices=["gaussian", "mosaic"], default="gaussian")
    parser.add_argument("--box-scale", type=float, default=1.15)
    parser.add_argument("--detector", choices=["yunet", "scrfd"], default="yunet")
    parser.add_argument("--recognition", choices=["none", "sface", "arcface"], default="none")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--config", help="FaceTrak config.json path")
    args = parser.parse_args()

    config = {}
    if args.config:
        import json
        config = json.loads(Path(args.config).read_text())

    anonymizer = create_anonymizer_from_config(config)
    anonymizer.analysis_cfg.detector_backend = args.detector
    anonymizer.analysis_cfg.recognition_backend = args.recognition
    anonymizer.redaction_cfg.method = RedactionMethod(args.method)
    anonymizer.redaction_cfg.box_scale = args.box_scale

    result = anonymizer.process(Path(args.input), Path(args.output_dir), max_frames=args.max_frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()