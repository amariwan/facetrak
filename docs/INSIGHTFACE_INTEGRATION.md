# FaceTrak × InsightFace 2.1 Integration Plan

## Overview

This document tracks the integration of reusable InsightFace components into FaceTrak while preserving the existing architecture.

**Goal**: Extend FaceTrak with InsightFace's best reusable functions (SCRFD detector, ArcFace recognition, PrivateFrame anonymization workflow) without replacing the existing YuNet/SFace/MediaPipe pipeline.

## Architecture Principles

- **Optional adapter pattern**: InsightFace components are opt-in via config flags
- **Graceful fallback**: All InsightFace features degrade to existing pipeline if unavailable
- **License compliance**: No auto-download of non-commercial models; explicit consent required
- **Linux-first**: ARM64/AMD64 compatibility, headless operation, Docker support
- **Model versioning**: Strict separation of model and embedding versions

## Implementation Phases

### Phase 1: InsightFace Backend Foundation ✅ IN PROGRESS
- [ ] `facetrak/recog/insightface_adapter.py` - Optional ONNX Runtime adapter
- [ ] Provider selection (CPU, CUDA, CoreML, OpenVINO, TensorRT)
- [ ] ARM64/AMD64 compatibility checks
- [ ] Config flags: `insightface_enabled`, `insightface_provider`, `insightface_model_pack`

### Phase 2: SCRFD Detector Integration
- [ ] `facetrak/recog/scrfd.py` - SCRFD detector with 5-point landmarks
- [ ] Support multiple model sizes (500M, 1G, 2.5G, 10G)
- [ ] Config: `detector_backend` ("yunet" | "scrfd")
- [ ] Benchmark: YuNet vs SCRFD on ARM64/AMD64

### Phase 3: ArcFace Recognition Backend
- [ ] `facetrak/recog/arcface.py` - ArcFace embedding extractor
- [ ] Model version tracking (embedding dim, model hash)
- [ ] Config: `recognition_backend` ("sface" | "arcface")
- [ ] Migration path for existing face database

### Phase 4: Face Analysis Enhancements
- [ ] 5-point face alignment (from SCRFD landmarks)
- [ ] Quality assessment for registration (sharpness, brightness, pose, occlusion)
- [ ] Explicit rejection reasons for unusable samples
- [ ] Optional RGB liveness backend (InsightFace's liveness model)
- [ ] Threshold validation & calibration utilities

### Phase 5: PrivateFrame-Inspired Anonymization
- [ ] `facetrak/video/anonymize.py` - Video face anonymization pipeline
- [ ] Gaussian blur & mosaic redaction
- [ ] Stable mask tracking across frames (IoU + embedding association)
- [ ] Analysis JSON with timestamps, editable masks
- [ ] Render without re-inference
- [ ] Batch video processing with progress/cancel/error handling
- [ ] Configurable retention & data minimization

### Phase 6: Model Management Enhancement
- [ ] `facetrak/device/management.py` - Enhanced model registry
- [ ] Model manifest (name, version, hash, license, source, backend_compat)
- [ ] Checksum validation on load
- [ ] License metadata & provenance tracking
- [ ] Installation/activation status
- [ ] Offline installation support
- [ ] License gates: no auto-download without consent

### Phase 7: Benchmark & Performance
- [ ] `scripts/benchmark_insightface.py` - Comprehensive benchmarks
- [ ] YuNet vs SCRFD (FPS, P50/P95 latency, memory, model load time)
- [ ] SFace vs ArcFace (embedding quality, verification accuracy)
- [ ] Liveness comparison (MediaPipe vs InsightFace RGB)
- [ ] ARM64 & AMD64 results

### Phase 8: Dashboard Integration
- [ ] Detector backend selector (YuNet/SCRFD)
- [ ] Recognition backend selector (SFace/ArcFace)
- [ ] Model status & license status panels
- [ ] Backend performance metrics
- [ ] Quality diagnostics for registration
- [ ] Anonymization editor (edit masks in JSON, re-render)
- [ ] Video batch job queue with progress

### Phase 9: Security & Licensing Audit
- [ ] `docs/INSIGHTFACE_LICENSE_AUDIT.md` - Complete license audit
- [ ] Code component license verification
- [ ] Model weight license verification
- [ ] No non-commercial models by default
- [ ] No covert identification
- [ ] No auto biometric registration
- [ ] No cross-camera re-identification
- [ ] Consent/RBAC/Retention preserved

### Phase 10: Testing & Acceptance
- [ ] Unit tests for all new adapters
- [ ] API integration tests
- [ ] Video fixture tests (anonymization)
- [ ] Mask & rendering tests
- [ ] Backend switching tests
- [ ] Model validation tests
- [ ] License gate tests
- [ ] Linux AMD64/ARM64 headless tests
- [ ] Docker smoke tests
- [ ] Security regression tests

## Deliverables Checklist

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Integrated Source Code | 🔄 |
| 2 | Updated Model/Backend Interfaces | ⏳ |
| 3 | PrivateFrame-Inspired Privacy Workflow | ⏳ |
| 4 | Functional Dashboard Integration | ⏳ |
| 5 | Regression Test Suite | ⏳ |
| 6 | ARM64/AMD64 Compatibility Report | ⏳ |
| 7 | Benchmark Report | ⏳ |
| 8 | `docs/INSIGHTFACE_INTEGRATION.md` | ✅ |
| 9 | `docs/INSIGHTFACE_LICENSE_AUDIT.md` | ⏳ |
| 10 | Final PASS/FAIL/BLOCKED Report | ⏳ |

## Configuration Additions

```json
{
  "insightface_enabled": false,
  "insightface_provider": "CPUExecutionProvider",
  "insightface_model_pack": "buffalo_l",
  "detector_backend": "yunet",
  "recognition_backend": "sface",
  "liveness_backend": "mediapipe",
  "anonymization": {
    "enabled": false,
    "method": "gaussian",
    "box_scale": 1.15,
    "gaussian_sigma": 30,
    "mosaic_block": 20
  },
  "model_registry": {
    "auto_download": false,
    "require_license_consent": true,
    "offline_models_dir": "./models"
  }
}
```

## License Compliance Notes

| Component | Code License | Model License | Commercial Use |
|-----------|-------------|---------------|----------------|
| InsightFace Python SDK | MIT | N/A | ✅ |
| SCRFD detector code | Apache-2.0 (OpenMMLab) | Non-commercial research only | ❌ (models) |
| ArcFace (buffalo_*) | Apache-2.0 | Non-commercial research only | ❌ (models) |
| PrivateFrame workflow | MIT | N/A | ✅ |
| InSwapper | MIT | Non-commercial research only | ❌ (models) |

**Policy**: Code components (MIT/Apache-2.0) can be integrated. Model weights require explicit commercial license from InsightFace. FaceTrak will only bundle code; models must be provided by user with valid license.

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| InsightFace models not commercially usable | HIGH | Document clearly; only bundle code; user provides licensed models |
| ONNX Runtime provider unavailable on ARM64 | MEDIUM | CPU provider always available; test on Pi |
| Embedding incompatibility (SFace 128-d vs ArcFace 512-d) | HIGH | Strict version tracking; migration tool; dual DB support |
| PrivateFrame JSON schema drift | LOW | Versioned schema; migration scripts |
| Performance regression on Pi | MEDIUM | Benchmarks; adaptive detection; configurable model sizes |

## Next Steps

1. Create InsightFace adapter with ONNX Runtime
2. Implement SCRFD detector wrapper
3. Implement ArcFace recognition backend
4. Add 5-point alignment & quality assessment
5. Build anonymization pipeline
6. Enhance model registry
7. Write benchmarks
8. Integrate dashboard
9. Complete license audit
10. Run full test suite
## API-Referenz (gleicher Server, kein zweiter Webserver)

Alle Routen unter `/api/v1`, Auth wie Bestand (`require_auth`).

| Methode | Pfad | Zweck |
|---|---|---|
| GET | `/api/v1/face/backends` | aktive Backends + Verfügbarkeit (yunet/sface/scrfd/arcface) + InsightFace-Status |
| POST | `/api/v1/face/backends` | Backend-Wechsel (`detector_backend`, `recognition_backend`, `liveness_backend`); `scrfd`/`arcface` ohne lizenzierte Modelle → 409 |
| GET | `/api/v1/insightface/status` | Provider, Model-Pack, vorhandene Modelle, `init_error` |
| GET | `/api/v1/model-registry` | Manifest-Liste (optional `?backend=`) |
| POST | `/api/v1/model-registry/register` | Manifest registrieren; kommerzielle Modelle nur mit `license_consent=true`, sonst 403 |
| POST | `/api/v1/model-registry/{name}/activate` | Modell aktivieren |
| GET | `/api/v1/model-registry/license` | Lizenz-Compliance-Status |
| POST | `/api/v1/anonymize/jobs` | Batch-Job starten (Video-Pfad, Methode gaussian/mosaic); läuft im Hintergrund-Thread mit Fortschritt |
| GET | `/api/v1/anonymize/jobs` | alle Jobs (nur Zähler/Status, keine Gesichtsdaten) |
| GET | `/api/v1/anonymize/jobs/{id}` | Job-Status + Fortschritt + Fehler |
| DELETE | `/api/v1/anonymize/jobs/{id}` | Job abbrechen |
| POST | `/api/v1/anonymize/analysis/edit` | `redact`-Flags in Analyse-JSON setzen (editierbare Masken) |
| POST | `/api/v1/anonymize/render` | Rendern aus Analyse-JSON **ohne erneute Inferenz** |
| GET | `/status` (erweitert) | zusätzlich `detector_backend`, `recognition_backend`, `liveness_backend`, `insightface{enabled,available,provider}` |

Dashboard (`web/src/app.tsx`, gleiche Sektionen-Navigation): neue Bereiche
**Backends** (Auswahl Detektor/Recognition, Modellstatus, Fehlermeldungen),
**Anonymize** (Job-Formular, Fortschrittsbalken, Abbruch, Fehler),
**Licenses** (`model-registry/license`). Kein zweiter Server.

## Qualitätsprüfung / Ablehnungsgründe (B)

Registrierungs-Qualität nutzt weiter `recog/quality.py`
(`score` aus Schärfe/Helligkeit/Pose, `GOOD_THRESHOLD=0.55`,
Labels excellent/good/fair/poor). Explizite Ablehnung: `register()` verlangt
≥3 Samples (`_MIN_REG_SAMPLES`) und verweigert unsichere Namen
(`sanitize_name` → Pfad-Traversal-Schutz). 5-Punkt-Landmarks liefert SCRFD
(`FaceDetection.kps`); ohne SCRFD fällt Alignment auf Center-Crop zurück.
RGB-Liveness nach InsightFace ist als `liveness_backend`-Schalter vorgesehen,
aber ohne lizenziertes Modell nicht aktiv — Bestand (Blink+Head-Turn)
bleibt Standard.
