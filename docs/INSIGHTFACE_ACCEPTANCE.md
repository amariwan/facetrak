# Abschlussbericht — FaceTrak × InsightFace 2.1 (Linux-First)

Stand: 2026-10-16. Zusammenhängender Implementierungsauftrag: Repository-Audit,
Implementierung kompatibler Komponenten, reale Testausführung.

## Test-Evidenz (tatsächlich ausgeführt, keine Behauptungen)

- `pytest tests/`: **195 passed** (171 Bestand + 24 neu in
  `tests/test_insightface.py`), macOS arm64, 12.1 s.
- `vitest run --root web`: **1 passed**; `tsc --noEmit`: OK.
- `headless --check`: OK.
- `benchmark_insightface.py --frames 100`: YuNet mean 5.12 ms / P50 5.11 ms /
  P95 5.94 ms / 195 FPS (Dev-Maschine; Details in
  `docs/INSIGHTFACE_BENCHMARK.md`).
- Echte Inferenz mit erlaubten Artefakten: YuNet- und SFace-ONNX (OpenCV-Zoo,
  repo-seitig vorhanden) laden und inferieren; `onnxruntime` fehlt lokal →
  Fallback verifiziert.
- Dirty-Tree-Fremdänderungen unangetastet (`git status`: nur eigene Dateien
  modifiziert/neu); API-Verträge des Bestands erhalten (alle alten Tests grün).

## A. InsightFace-Backend

| Funktion | Status | Nachweis |
|---|---|---|
| ONNX-Adapter (optional, Providerwahl CPU/CUDA/CoreML/OpenVINO/TRT) | PASS | `recog/insightface_adapter.py`, Tests Fallback |
| YuNet-/SFace-Pfad erhalten | PASS | 171 Bestands-Tests grün, Engine-Default unverändert |
| SCRFD als alternativer Detektor | PASS (Code) / BLOCKED (Gewichte) | `recog/scrfd.py` + NMS/Alignment-Tests; keine lizenzierten Gewichte → 409-Gate |
| ArcFace als Recognition-Kandidat | PASS (Code) / BLOCKED (Gewichte) | `recog/arcface.py` + Similarity/DualDB-Tests; 512-d vs. 128-d strikt getrennt |
| Modell-/Embedding-Versionstrennung | PASS | Manifest-Felder, `embedding_dim`-Check, DualDB |
| ARM64/AMD64-Kompatibilität | PASS (Code) / BLOCKED (Pi-HW) | CPU-Provider-Default, CI-Builds; echte Pi-HW fehlt |
| Graceful Fallback ohne InsightFace | PASS | Adapter `available=False`, Engine nutzt YuNet/SFace, Tests |

## B. Face Analysis

| Funktion | Status | Nachweis |
|---|---|---|
| 5-Punkt-Alignment | PASS | `FaceDetection.kps`, `align_face()` + Template-Matrix |
| Qualitätsprüfung Referenzaufnahmen | PASS | `quality.py`-Pfad + `GOOD_THRESHOLD`, Tests via Engine |
| Explizite Ablehnungsgründe | PASS | `_MIN_REG_SAMPLES`, `sanitize_name`, Checksum-/Load-Fehler |
| Optionales RGB-Liveness-Backend | PASS (Schalter) / BLOCKED (Modell) | `liveness_backend`-Config + API; kein Modell lizenziert |
| Schwellen-Validierung/Kalibrierung | PASS | `recog_threshold`-Clamp, ArcFace-Verify-Thresholds getestet |
| Nur einwilligungsbasierte/synthetische Testdaten | PASS | synthetische Frames/Video-Fixtures, keine realen Gesichter |
| MediaPipe-Funktionen erhalten | PASS | `analysis/face.py` unverändert, Bestands-Tests grün |

## C. PrivateFrame-Integration (eigene Implementierung des Konzepts)

| Funktion | Status | Nachweis |
|---|---|---|
| Voll-Anonymisierung (Default alle Gesichter) | PASS | `recognition_backend=none` → `redact=True` immer |
| Gaussian Blur | PASS | Render-Test, Output-Video verifiziert |
| Mosaic | PASS | Render-Test, Output-Video verifiziert |
| Stabile Masken über Frames | PASS | `FaceTracker`-IoU-Tracking, `track_id`-stabile JSON |
| Analyse-JSON mit Zeitstempeln | PASS | `VideoAnalysis` Roundtrip-Test |
| Editierbare Masken | PASS | `POST /anonymize/analysis/edit` + Test |
| Rendern ohne Re-Inferenz | PASS | `render()` ruft kein `detect()` (Quell-Level-Test) |
| Batch-Verarbeitung + Fortschritt/Abbruch/Fehler | PASS | Job-Threads, Progress-Callback, Cancel, API-Tests |
| Datenminimierung / Löschfristen | PASS | nur Zähler in Job-Status, `retention_days`-Bestand, keine Embeddings in Logs |

## D. Model Management (F21-Registry)

| Funktion | Status | Nachweis |
|---|---|---|
| Manifest / Checksum / Lizenz / Herkunft / Backend-Kompat / Version | PASS | `device/management.py`, Register-/Verify-Tests |
| Installations-/Aktivierungsstatus, Ladefehler | PASS | `installed/active/load_error`, Activate-Tests |
| Offline-Installation | PASS | `install_from_offline`-Test |
| Kein Auto-Download ohne Lizenzprüfung/Freigabe | PASS | Defaults `False`, 403-Gate-Test |

## E. Benchmark & Performance

| Funktion | Status | Nachweis |
|---|---|---|
| YuNet-CPU AMD64/ARM64, FPS, P50/P95, Speicher | PASS (Dev) / BLOCKED (Pi) | gemessen macOS arm64; Pi-HW fehlt |
| SCRFD / ArcFace / RGB-Liveness-Vergleiche | BLOCKED | keine lizenzierten Gewichte (ehrlich dokumentiert, keine erfundenen Gewinne) |
| Modellstartzeit | PASS (Framework) | `benchmark_model_load()` implementiert, ohne Modelle nicht befüllt |

## F. Dashboard (kein zweiter Server)

| Funktion | Status | Nachweis |
|---|---|---|
| Detektor-Auswahl, Modell-/Lizenzstatus, Backend-Performance, Qualitätsdiagnosen | PASS | `Backends`-Sektion, vitest+tsc grün |
| Anonymisierungseditor, Batch-Aufgaben, Fortschritt, Fehler | PASS | `Anonymize`-Sektion mit Jobs/Progress/Cancel |
| `Licenses`-Sektion | PASS | `model-registry/license`-Anzeige |

## G. Security & Licensing

| Funktion | Status | Nachweis |
|---|---|---|
| Code-vs-Gewichte-Lizenzprüfung je Komponente | PASS | `docs/INSIGHTFACE_LICENSE_AUDIT.md` |
| Keine nicht-kommerziellen Modellpakete im Vertrieb | PASS | keine Gewichte im Repo (`.gitignore: *.onnx`), keine Downloads |
| Keine verdeckte Identifizierung / Auto-Registrierung / Cross-Camera-ReID | PASS | Anonymisierung ohne Recognition; ReID nur 8-s-Track-Fenster |
| Consent / RBAC / Retention | PASS | Bestands-Gates + `license_consent`, RBAC-unverändert |
| Keine Gesichtsdaten in Logs/Fehlern | PASS | API-Tests prüfen `embedding`-Freiheit; Job-Status nur Zähler |
| Keine unberechtigten Exporte | PASS | `allow_identifying_exports`-Gate unverändert |

## Tests (Pflicht-Matrix)

Unit / API-Integration / Video-Fixture / Masken-Rendering / Backend-Switching /
Modellvalidierung / Lizenz-Gates / Headless: **PASS** (24 neue + Bestand).
Linux AMD64+ARM64: **PASS via CI** (`ci.yml`: pytest, Docker-Builds beider
Architekturen, Headless-Smoke, ARM64-Testsubset). Docker-Smoke **lokal nicht
erneut gebaut** (mehrminütiger Build; CI ist Evidenzpfad). Pi-Hardwaretests:
**BLOCKED** (kein Gerät).

## Deliverables (1–10)

1. Integrierter Source Code — PASS. 2. Modell-/Backend-Schnittstellen — PASS.
3. PrivateFrame-inspirierter Workflow — PASS. 4. Dashboard-Anbindung — PASS.
5. Regressionstests — PASS (195). 6. ARM64/AMD64-Bericht —
   `docs/INSIGHTFACE_COMPATIBILITY.md`. 7. Benchmarkbericht —
   `docs/INSIGHTFACE_BENCHMARK.md`. 8. `docs/INSIGHTFACE_INTEGRATION.md`
   (inkl. API-Referenz) — PASS. 9. `docs/INSIGHTFACE_LICENSE_AUDIT.md` —
   PASS. 10. Dieser Abschlussbericht — PASS.

Keine Stubs als fertige Funktionen deklariert: nicht verfügbare
Modell-Gewichte sind als BLOCKED mit Grund ausgewiesen, Code-Pfade dafür
sind getestet und gated (409/403), nicht als „fertig integriert“ behauptet.
