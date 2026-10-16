"""InsightFace ONNX Runtime Adapter.

Optional adapter for InsightFace model packages (buffalo_l, antelopev2, etc.)
using ONNX Runtime. Gracefully degrades when insightface/onnxruntime not available.

Models are NOT auto-downloaded. User must provide licensed model files.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Provider priority order (first available wins)
DEFAULT_PROVIDERS = [
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "TensorrtExecutionProvider",
    "CPUExecutionProvider",
]

# Known InsightFace model packs and their expected files
MODEL_PACKS = {
    "buffalo_l": {
        "det_10g": "det_10g.onnx",           # SCRFD 10G detector
        "det_2.5g": "det_2.5g.onnx",         # SCRFD 2.5G detector
        "det_500m": "det_500m.onnx",         # SCRFD 500M detector
        "w600k_r50": "w600k_r50.onnx",       # ArcFace ResNet50
        "w600k_mbf": "w600k_mbf.onnx",       # ArcFace MobileFaceNet
        "genderage": "genderage.onnx",        # Gender/Age
        "landmark_2d_106": "landmark_2d_106.onnx",
        "landmark_3d_68": "landmark_3d_68.onnx",
    },
    "buffalo_m": {
        "det_10g": "det_10g.onnx",
        "det_2.5g": "det_2.5g.onnx",
        "w600k_r50": "w600k_r50.onnx",
        "genderage": "genderage.onnx",
    },
    "buffalo_s": {
        "det_10g": "det_10g.onnx",
        "w600k_r50": "w600k_r50.onnx",
        "genderage": "genderage.onnx",
    },
    "antelopev2": {
        "det_10g": "det_10g.onnx",
        "glintr100": "glintr100.onnx",       # ArcFace ResNet100
        "genderage": "genderage.onnx",
    },
    "raccoon_s": {
        "det": "det.onnx",
        "rec": "rec.onnx",
    },
    "raccoon_l": {
        "det": "det.onnx",
        "rec": "rec.onnx",
    },
}


@dataclass
class ModelManifest:
    """Model manifest with version, checksum, license, and compatibility info."""
    name: str
    version: str
    file_path: str
    sha256: str
    license: str
    license_url: str
    backend: str  # "scrfd", "arcface", "genderage", "landmark"
    input_shape: tuple[int, ...]
    output_shapes: list[tuple[int, ...]]
    embedding_dim: int | None = None
    commercial_license_required: bool = True
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "file_path": self.file_path,
            "sha256": self.sha256,
            "license": self.license,
            "license_url": self.license_url,
            "backend": self.backend,
            "input_shape": list(self.input_shape),
            "output_shapes": [list(s) for s in self.output_shapes],
            "embedding_dim": self.embedding_dim,
            "commercial_license_required": self.commercial_license_required,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelManifest":
        return cls(
            name=d["name"],
            version=d["version"],
            file_path=d["file_path"],
            sha256=d["sha256"],
            license=d["license"],
            license_url=d["license_url"],
            backend=d["backend"],
            input_shape=tuple(d["input_shape"]),
            output_shapes=[tuple(s) for s in d["output_shapes"]],
            embedding_dim=d.get("embedding_dim"),
            commercial_license_required=d.get("commercial_license_required", True),
            source_url=d.get("source_url", ""),
        )


class InsightFaceAdapter:
    """ONNX Runtime adapter for InsightFace models.

    Usage:
        adapter = InsightFaceAdapter(config)
        if adapter.available:
            session = adapter.get_session("det_10g")
            # run inference
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        # Accept both flat keys (insightface_enabled) and nested engine
        # config keys (enabled) so FaceEngine wiring actually activates.
        self.enabled = bool(cfg.get("insightface_enabled",
                                   cfg.get("enabled", False)))
        self.provider_name = cfg.get(
            "insightface_provider",
            cfg.get("provider", "CPUExecutionProvider"))
        self.model_pack = cfg.get("insightface_model_pack",
                                  cfg.get("model_pack", "buffalo_l"))
        self.models_dir = Path(cfg.get("insightface_models_dir",
                                       cfg.get("models_dir",
                                               "models/insightface")))
        self.allow_auto_download = bool(cfg.get("insightface_auto_download",
                                                cfg.get("auto_download", False)))
        self.require_license_consent = bool(cfg.get(
            "insightface_require_license_consent",
            cfg.get("require_license_consent", True)))

        self._sessions: dict[str, Any] = {}
        self._manifests: dict[str, ModelManifest] = {}
        self._ort = None
        self._available = False
        self._init_error: str | None = None

        if self.enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize ONNX Runtime and verify model directory."""
        try:
            import onnxruntime as ort
            self._ort = ort
        except ImportError as exc:
            self._init_error = f"onnxruntime not installed: {exc}"
            logger.warning(self._init_error)
            return

        # Verify provider availability
        available_providers = ort.get_available_providers()
        if self.provider_name not in available_providers:
            logger.warning(
                "Provider %s not available (available: %s), falling back to CPUExecutionProvider",
                self.provider_name, available_providers
            )
            self.provider_name = "CPUExecutionProvider"

        # Check model directory
        if not self.models_dir.exists():
            if self.allow_auto_download:
                logger.info("Model directory %s does not exist, creating", self.models_dir)
                self.models_dir.mkdir(parents=True, exist_ok=True)
            else:
                self._init_error = (
                    f"Model directory {self.models_dir} does not exist. "
                    "Set insightface_models_dir or enable insightface_auto_download."
                )
                logger.warning(self._init_error)
                return

        self._available = True
        logger.info(
            "InsightFace adapter initialized: provider=%s, model_pack=%s, models_dir=%s",
            self.provider_name, self.model_pack, self.models_dir
        )

    @property
    def available(self) -> bool:
        return self._available and self.enabled

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def get_provider_options(self) -> dict[str, Any]:
        """Get provider-specific options."""
        if self.provider_name == "CUDAExecutionProvider":
            return {"device_id": 0, "arena_extend_strategy": "kNextPowerOfTwo"}
        elif self.provider_name == "OpenVINOExecutionProvider":
            return {"device_type": "CPU_FP32"}
        elif self.provider_name == "TensorrtExecutionProvider":
            return {"device_id": 0, "trt_fp16_enable": True}
        return {}

    def get_session(self, model_key: str) -> Any | None:
        """Get or create ONNX Runtime session for a model."""
        if not self.available:
            return None

        if model_key in self._sessions:
            return self._sessions[model_key]

        # Resolve model file
        model_file = self._resolve_model_file(model_key)
        if not model_file or not model_file.exists():
            logger.error("Model file not found for %s: %s", model_key, model_file)
            return None

        # Verify checksum if manifest exists
        if model_key in self._manifests:
            if not self._verify_checksum(model_file, self._manifests[model_key].sha256):
                logger.error("Checksum mismatch for %s", model_key)
                return None

        try:
            sess_options = self._ort.SessionOptions()
            sess_options.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True

            providers = [(self.provider_name, self.get_provider_options())]
            session = self._ort.InferenceSession(
                str(model_file), sess_options, providers=providers
            )
            self._sessions[model_key] = session
            logger.info("Loaded InsightFace model: %s from %s", model_key, model_file)
            return session
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_key, exc)
            return None

    def _resolve_model_file(self, model_key: str) -> Path | None:
        """Resolve model file path from model pack."""
        pack = MODEL_PACKS.get(self.model_pack, {})
        filename = pack.get(model_key)
        if not filename:
            # Try direct lookup
            filename = model_key if model_key.endswith(".onnx") else f"{model_key}.onnx"
        return self.models_dir / filename

    def _verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify SHA256 checksum of model file."""
        if not expected_sha256:
            return True
        import hashlib
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_sha256

    def register_manifest(self, manifest: ModelManifest) -> None:
        """Register a model manifest for verification."""
        self._manifests[manifest.name] = manifest

    def load_manifests(self, manifest_path: Path) -> bool:
        """Load model manifests from JSON file."""
        if not manifest_path.exists():
            return False
        import json
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            for m in data.get("models", []):
                manifest = ModelManifest.from_dict(m)
                self._manifests[manifest.name] = manifest
            logger.info("Loaded %d model manifests from %s", len(self._manifests), manifest_path)
            return True
        except Exception as exc:
            logger.error("Failed to load manifests: %s", exc)
            return False

    def list_available_models(self) -> list[str]:
        """List models that have files present."""
        pack = MODEL_PACKS.get(self.model_pack, {})
        available = []
        for key, filename in pack.items():
            if (self.models_dir / filename).exists():
                available.append(key)
        return available

    def get_model_info(self, model_key: str) -> dict | None:
        """Get model metadata."""
        session = self.get_session(model_key)
        if not session:
            return None
        return {
            "name": model_key,
            "input": [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()],
            "output": [{"name": o.name, "shape": o.shape, "type": o.type} for o in session.get_outputs()],
            "provider": self.provider_name,
        }

    def release(self) -> None:
        """Release all sessions."""
        self._sessions.clear()
        self._available = False


def create_adapter(config: dict | None = None) -> InsightFaceAdapter:
    """Factory function to create InsightFaceAdapter."""
    return InsightFaceAdapter(config)


# Convenience: pre-configured manifests for common models
BUILTIN_MANIFESTS = {
    "buffalo_l": {
        "models": [
            {
                "name": "det_10g",
                "version": "1.0",
                "file_path": "models/insightface/det_10g.onnx",
                "sha256": "",  # User must fill in
                "license": "Apache-2.0 (OpenMMLab)",
                "license_url": "https://github.com/deepinsight/insightface/blob/master/detection/scrfd/LICENSE",
                "backend": "scrfd",
                "input_shape": [1, 3, 640, 640],
                "output_shapes": [[1, 17640, 15], [1, 17640, 4], [1, 17640, 10]],
                "embedding_dim": None,
                "commercial_license_required": True,
                "source_url": "https://github.com/deepinsight/insightface/tree/master/detection/scrfd",
            },
            {
                "name": "w600k_r50",
                "version": "1.0",
                "file_path": "models/insightface/w600k_r50.onnx",
                "sha256": "",
                "license": "Apache-2.0",
                "license_url": "https://github.com/deepinsight/insightface/tree/master/recognition/arcface",
                "backend": "arcface",
                "input_shape": [1, 3, 112, 112],
                "output_shapes": [[1, 512]],
                "embedding_dim": 512,
                "commercial_license_required": True,
                "source_url": "https://github.com/deepinsight/insightface/tree/master/recognition/arcface",
            },
            {
                "name": "genderage",
                "version": "1.0",
                "file_path": "models/insightface/genderage.onnx",
                "sha256": "",
                "license": "Apache-2.0",
                "license_url": "https://github.com/deepinsight/insightface/tree/master/recognition/arcface",
                "backend": "genderage",
                "input_shape": [1, 3, 112, 112],
                "output_shapes": [[1, 2], [1, 100]],
                "embedding_dim": None,
                "commercial_license_required": True,
                "source_url": "https://github.com/deepinsight/insightface/tree/master/recognition/arcface",
            },
        ]
    }
}