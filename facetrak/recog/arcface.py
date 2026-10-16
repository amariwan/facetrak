"""ArcFace Recognition Backend using InsightFace ONNX models.

Provides 512-dim embeddings (vs SFace 128-dim).
Supports model version tracking and embedding compatibility management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import cv2

from facetrak.recog.insightface_adapter import InsightFaceAdapter

logger = logging.getLogger(__name__)


@dataclass
class ArcFaceConfig:
    model_key: str = "w600k_r50"    # w600k_r50, w600k_mbf, glintr100
    input_size: tuple[int, int] = (112, 112)
    embedding_dim: int = 512
    norm_embedding: bool = True


class ArcFaceRecognizer:
    """ArcFace face recognition using ONNX Runtime.

    Produces 512-dim embeddings. Not compatible with SFace 128-dim embeddings.
    """

    def __init__(self, config: ArcFaceConfig | None = None, adapter: InsightFaceAdapter | None = None):
        self.cfg = config or ArcFaceConfig()
        self.adapter = adapter
        self._session: Any | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._ready = False

        if self.adapter and self.adapter.available:
            self._load_model()

    def _load_model(self) -> bool:
        """Load ArcFace model from adapter."""
        session = self.adapter.get_session(self.cfg.model_key)
        if not session:
            logger.error("Failed to get ArcFace session for %s", self.cfg.model_key)
            return False

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name
        self._ready = True

        # Verify embedding dimension
        output_shape = session.get_outputs()[0].shape
        if isinstance(output_shape[-1], int) and output_shape[-1] != self.cfg.embedding_dim:
            logger.warning(
                "ArcFace model output dim %d != configured %d",
                output_shape[-1], self.cfg.embedding_dim
            )
            self.cfg.embedding_dim = output_shape[-1]

        logger.info(
            "ArcFace recognizer loaded: model=%s, embedding_dim=%d, input_size=%s",
            self.cfg.model_key, self.cfg.embedding_dim, self.cfg.input_size
        )
        return True

    @property
    def ready(self) -> bool:
        return self._ready and self._session is not None

    @property
    def embedding_dim(self) -> int:
        return self.cfg.embedding_dim

    def extract(self, aligned_face: np.ndarray) -> np.ndarray | None:
        """Extract embedding from aligned face (112x112 RGB).

        Args:
            aligned_face: Aligned face chip (H, W, 3) RGB, typically 112x112

        Returns:
            Normalized embedding vector (embedding_dim,) or None on failure
        """
        if not self.ready:
            return None

        if aligned_face is None or aligned_face.size == 0:
            return None

        try:
            # Ensure correct size
            h, w = aligned_face.shape[:2]
            if (h, w) != self.cfg.input_size:
                aligned_face = cv2.resize(aligned_face, self.cfg.input_size, interpolation=cv2.INTER_LINEAR)

            # Preprocess: RGB to BGR, HWC to CHW, normalize to [-1, 1]
            # ArcFace typically expects input in [-1, 1] range
            input_img = aligned_face[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
            input_img = (input_img - 127.5) / 127.5
            input_img = np.expand_dims(input_img, axis=0)

            # Inference
            embedding = self._session.run([self._output_name], {self._input_name: input_img})[0]
            embedding = embedding.flatten().astype(np.float32)

            # L2 normalize
            if self.cfg.norm_embedding:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            return embedding

        except Exception as exc:
            logger.error("ArcFace extraction failed: %s", exc)
            return None

    def extract_batch(self, aligned_faces: list[np.ndarray]) -> list[np.ndarray | None]:
        """Extract embeddings from multiple aligned faces."""
        return [self.extract(face) for face in aligned_faces]

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two normalized embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))

    def verify(self, emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.4) -> tuple[bool, float]:
        """Verify if two embeddings match.

        ArcFace typically uses higher thresholds than SFace.
        Recommended: 0.4-0.5 for 512-dim embeddings.
        """
        sim = self.similarity(emb1, emb2)
        return sim >= threshold, sim


def create_arcface_recognizer(config: dict | None = None, adapter: InsightFaceAdapter | None = None) -> ArcFaceRecognizer:
    """Factory to create ArcFaceRecognizer from config dict."""
    if config:
        cfg = ArcFaceConfig(
            model_key=config.get("arcface_model", "w600k_r50"),
            input_size=tuple(config.get("arcface_input_size", [112, 112])),
            embedding_dim=config.get("arcface_embedding_dim", 512),
            norm_embedding=config.get("arcface_norm_embedding", True),
        )
    else:
        cfg = ArcFaceConfig()
    return ArcFaceRecognizer(cfg, adapter)


# Compatibility layer: ArcFace embeddings are NOT compatible with SFace
# This class manages dual-database support during transition
class DualEmbeddingDatabase:
    """Manage both SFace (128-d) and ArcFace (512-d) embeddings.

    During migration, stores both embedding types per identity.
    """

    def __init__(self, sface_db, arcface_recognizer: ArcFaceRecognizer | None = None):
        self.sface_db = sface_db  # Existing FaceDatabase
        self.arcface = arcface_recognizer
        self._arcface_embeddings: dict[str, list[np.ndarray]] = {}  # name -> list of 512-d embeddings

    def register_sface(self, name: str, embeddings: list[np.ndarray]) -> bool:
        """Register using SFace (existing path)."""
        return self.sface_db.register(name, embeddings)

    def register_arcface(self, name: str, embeddings: list[np.ndarray]) -> bool:
        """Register using ArcFace embeddings."""
        if not self.arcface or not self.arcface.ready:
            return False
        valid = [e for e in embeddings if e is not None and e.shape[0] == self.arcface.embedding_dim]
        if len(valid) < 3:
            return False
        self._arcface_embeddings[name] = valid
        # Also register with SFace for backward compat (extract from aligned faces if needed)
        return True

    def predict_sface(self, embedding: np.ndarray, threshold: float = 0.36) -> tuple[str | None, float]:
        """Predict using SFace database."""
        return self.sface_db.predict(embedding, threshold)

    def predict_arcface(self, embedding: np.ndarray, threshold: float = 0.45) -> tuple[str | None, float]:
        """Predict using ArcFace database."""
        if not self.arcface or not self.arcface.ready or not self._arcface_embeddings:
            return None, 0.0

        best_name = None
        best_sim = 0.0

        for name, embs in self._arcface_embeddings.items():
            for emb in embs:
                sim = self.arcface.similarity(embedding, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        if best_sim < threshold:
            return None, best_sim
        return best_name, best_sim

    def predict(self, embedding: np.ndarray, backend: str = "auto", threshold: float = 0.36) -> tuple[str | None, float]:
        """Predict using specified or best available backend."""
        if backend == "sface" or (backend == "auto" and not self.arcface):
            return self.predict_sface(embedding, threshold)
        elif backend == "arcface":
            arc_thresh = threshold if threshold > 0.3 else 0.45
            return self.predict_arcface(embedding, arc_thresh)
        else:
            # Auto: try both, prefer ArcFace if available and confident
            if self.arcface and self.arcface.ready and self._arcface_embeddings:
                name, sim = self.predict_arcface(embedding, 0.45)
                if name:
                    return name, sim
            return self.predict_sface(embedding, threshold)

    def get_known_names(self, backend: str = "auto") -> list[str]:
        """Get known names from specified backend."""
        if backend == "arcface" and self._arcface_embeddings:
            return list(self._arcface_embeddings.keys())
        return self.sface_db.known_names