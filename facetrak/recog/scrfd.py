"""SCRFD Face Detector using InsightFace ONNX models.

Supports multiple model sizes: 500M, 2.5G, 10G.
Provides 5-point face landmarks for alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from facetrak.models import FaceDetection
from facetrak.recog.insightface_adapter import InsightFaceAdapter, create_adapter

logger = logging.getLogger(__name__)


@dataclass
class SCRFDConfig:
    model_key: str = "det_10g"      # det_500m, det_2.5g, det_10g
    input_size: tuple[int, int] = (640, 640)
    score_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_num: int = 20
    use_kps: bool = True            # Use 5-point landmarks for alignment


class SCRFDDetector:
    """SCRFD face detector with 5-point landmark output.

    Compatible with FaceTrak's FaceDetection model and FaceTracker.
    """

    def __init__(self, config: SCRFDConfig | None = None, adapter: InsightFaceAdapter | None = None):
        self.cfg = config or SCRFDConfig()
        self.adapter = adapter
        self._session: Any | None = None
        self._input_name: str | None = None
        self._output_names: list[str] = []
        self._ready = False

        if self.adapter and self.adapter.available:
            self._load_model()

    def _load_model(self) -> bool:
        """Load SCRFD model from adapter."""
        session = self.adapter.get_session(self.cfg.model_key)
        if not session:
            logger.error("Failed to get SCRFD session for %s", self.cfg.model_key)
            return False

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [o.name for o in session.get_outputs()]
        self._ready = True

        logger.info(
            "SCRFD detector loaded: model=%s, input_size=%s, score_thresh=%.2f",
            self.cfg.model_key, self.cfg.input_size, self.cfg.score_threshold
        )
        return True

    @property
    def ready(self) -> bool:
        return self._ready and self._session is not None

    def detect(self, frame: np.ndarray, scale: float = 1.0) -> list[FaceDetection]:
        """Detect faces in frame.

        Args:
            frame: BGR image (H, W, 3)
            scale: Scale factor from original frame to this frame

        Returns:
            List of FaceDetection with 5-point landmarks in .kps attribute
        """
        if not self.ready:
            return []

        h, w = frame.shape[:2]

        # Preprocess: resize with letterbox to maintain aspect ratio
        input_h, input_w = self.cfg.input_size
        scale_x = input_w / w
        scale_y = input_h / h
        scale_factor = min(scale_x, scale_y)

        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox to input size
        input_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        input_img[:new_h, :new_w] = resized

        # BGR to RGB, HWC to CHW, normalize
        input_img = input_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        # Inference
        try:
            outputs = self._session.run(self._output_names, {self._input_name: input_img})
        except Exception as exc:
            logger.error("SCRFD inference failed: %s", exc)
            return []

        # Parse outputs
        # SCRFD outputs: [scores, bboxes, kpss] - shapes vary by model
        # scores: (1, N, 1) or (1, N)
        # bboxes: (1, N, 4)
        # kpss: (1, N, 5, 2) - 5 keypoints (x, y)
        detections = self._postprocess(
            outputs, frame.shape[1], frame.shape[0],
            scale, scale_factor, new_w, new_h, input_w, input_h)
        return detections

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        frame_w: int,
        frame_h: int,
        scale: float,
        scale_factor: float,
        new_w: int,
        new_h: int,
        input_w: int,
        input_h: int,
    ) -> list[FaceDetection]:
        """Post-process SCRFD outputs to FaceDetection list."""
        if len(outputs) < 2:
            return []

        # Handle different output formats
        scores = outputs[0]
        bboxes = outputs[1]
        kpss = outputs[2] if len(outputs) > 2 and self.cfg.use_kps else None

        # Squeeze batch dimension
        if scores.ndim == 3:
            scores = scores[0]  # (N, 1) or (N,)
        if bboxes.ndim == 3:
            bboxes = bboxes[0]  # (N, 4)
        if kpss is not None and kpss.ndim == 4:
            kpss = kpss[0]  # (N, 5, 2)

        # Flatten scores
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]

        # Filter by score threshold
        valid = scores >= self.cfg.score_threshold
        if not np.any(valid):
            return []

        scores = scores[valid]
        bboxes = bboxes[valid]
        if kpss is not None:
            kpss = kpss[valid]

        # Sort by score descending
        order = np.argsort(-scores)
        scores = scores[order]
        bboxes = bboxes[order]
        if kpss is not None:
            kpss = kpss[order]

        # Limit to max_num
        if len(scores) > self.cfg.max_num:
            scores = scores[:self.cfg.max_num]
            bboxes = bboxes[:self.cfg.max_num]
            if kpss is not None:
                kpss = kpss[:self.cfg.max_num]

        # Apply NMS
        keep = self._nms(bboxes, scores, self.cfg.nms_threshold)
        scores = scores[keep]
        bboxes = bboxes[keep]
        if kpss is not None:
            kpss = kpss[keep]

        # Convert bboxes from input space to original frame space
        # bboxes are in input image coordinates (after letterbox)
        # Need to map back to resized image, then to original
        detections = []
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            x1, y1, x2, y2 = bbox

            # Map from input space to resized space
            x1 = x1 / scale_factor
            y1 = y1 / scale_factor
            x2 = x2 / scale_factor
            y2 = y2 / scale_factor

            # Map from resized space to original frame space
            x1 = x1 * scale
            y1 = y1 * scale
            x2 = x2 * scale
            y2 = y2 * scale

            x = int(max(0, x1))
            y = int(max(0, y1))
            w = int(max(0, min(x2 - x1, frame_w * scale - x1)))
            h = int(max(0, min(y2 - y1, frame_h * scale - y1)))

            if w <= 0 or h <= 0:
                continue

            # Prepare landmark row for FaceDetection (compatible with YuNet format)
            # YuNet row has 15 elements: [x, y, w, h, kp0_x, kp0_y, ..., kp4_x, kp4_y, score]
            row = np.zeros(15, dtype=np.float32)
            row[0] = x / scale
            row[1] = y / scale
            row[2] = w / scale
            row[3] = h / scale

            det_kps: np.ndarray | None = None
            if kpss is not None and i < len(kpss):
                kps = kpss[i]  # (5, 2)
                # Map keypoints back to original frame space
                mapped = []
                for j, (kx, ky) in enumerate(kps):
                    kx = kx / scale_factor * scale
                    ky = ky / scale_factor * scale
                    row[4 + j * 2] = kx / scale
                    row[5 + j * 2] = ky / scale
                    mapped.append([kx, ky])
                det_kps = np.array(mapped, dtype=np.float32)
            else:
                # No landmarks - fill with center-ish points
                cx, cy = x + w // 2, y + h // 2
                for j in range(5):
                    row[4 + j * 2] = cx / scale
                    row[5 + j * 2] = cy / scale

            row[14] = float(score)
            detections.append(FaceDetection(
                row=row, x=x, y=y, w=w, h=h,
                score=float(score), kps=det_kps))

        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    def align_face(self, frame: np.ndarray, detection: FaceDetection, output_size: int = 112) -> np.ndarray | None:
        """Align face using 5-point landmarks (similar to ArcFace alignment).

        Returns aligned face chip (112x112) for recognition.
        """
        if detection.kps is None:
            # Fallback: simple crop and resize
            x, y, w, h = detection.x, detection.y, detection.w, detection.h
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                return None
            return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        # 5-point landmarks from SCRFD
        # Standard ArcFace 5-point reference (normalized to 112x112)
        ref_pts = np.array([
            [38.2946, 51.6963],   # left eye
            [73.5318, 51.5014],   # right eye
            [56.0252, 71.7366],   # nose tip
            [41.5493, 92.3655],   # left mouth corner
            [70.7299, 92.2041],   # right mouth corner
        ], dtype=np.float32)

        src_pts = detection.kps.astype(np.float32)  # (5, 2)

        # Estimate similarity transform
        M, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
        if M is None:
            return None

        # Warp
        aligned = cv2.warpAffine(frame, M, (output_size, output_size), borderValue=0.0)
        return aligned


def create_scrfd_detector(config: dict | None = None, adapter: InsightFaceAdapter | None = None) -> SCRFDDetector:
    """Factory to create SCRFDDetector from config dict."""
    if config:
        cfg = SCRFDConfig(
            model_key=config.get("scrfd_model", "det_10g"),
            input_size=tuple(config.get("scrfd_input_size", [640, 640])),
            score_threshold=config.get("scrfd_score_threshold", 0.5),
            nms_threshold=config.get("scrfd_nms_threshold", 0.4),
            max_num=config.get("scrfd_max_num", 20),
            use_kps=config.get("scrfd_use_kps", True),
        )
    else:
        cfg = SCRFDConfig()
    return SCRFDDetector(cfg, adapter)


def create_alignment_matrix(
    src_pts: np.ndarray, output_size: int = 112) -> np.ndarray | None:
    """Estimate similarity transform from 5-point landmarks to ArcFace template."""
    ref_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)
    src = np.asarray(src_pts, dtype=np.float32)
    if src.shape != (5, 2):
        return None
    scale = output_size / 112.0
    return cv2.estimateAffinePartial2D(
        src, ref_pts * scale, method=cv2.LMEDS)[0]