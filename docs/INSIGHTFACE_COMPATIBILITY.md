# ARM64 / AMD64 Kompatibilitätsbericht (FaceTrak 2.1)

Stand: 2026-10-16. macOS ist nur Entwicklungs-/Testplattform.

## Lokal verifiziert (macOS arm64, Dev-Maschine)

- `pytest tests/`: **195 passed** (171 Bestand + 24 neu `test_insightface.py`).
- `pytest web` (vitest): **1 passed**; `tsc --noEmit`: OK.
- `headless --check`: OK (Kamera nicht gestartet, Engine bootet).
- Echte Inferenz mit erlaubten Artefakten:
  - YuNet (`face_detection_yunet.onnx`, OpenCV-Zoo): lädt, Inferenz läuft
    (0 Detektionen auf synthetischer Cartoon-Grafik — Pipeline-Nachweis,
    kein Genauigkeitsnachweis).
  - SFace (`face_recognition_sface.onnx`): Recognizer lädt OK.
  - `onnxruntime` **nicht installiert** → InsightFace-Adapter meldet
    `available=False` + `init_error`, Engine nutzt YuNet/SFace weiter
    (graceful Fallback verifiziert, kein Crash).
- Neue Module importieren fehlerfrei:
  `insightface_adapter`, `scrfd`, `arcface`, `video.anonymize`,
  `device.management` (F21-Registry).

## Linux AMD64 (CI)

- `ci.yml`: pytest auf Ubuntu, Docker-Build `linux/amd64`, Headless-Smoke
  im Image (`headless --check`).
- Neue Tests sind plattformneutral (keine macOS-APIs, keine Hardware);
  `onnxruntime` ist optionales Extra und fehlt in CI → Fallback-Pfade
  werden dort ebenfalls exercising (Adapter-unavailable-Tests).

## Linux ARM64 (CI)

- `ci.yml`: Docker-Build `linux/arm64` (QEMU), Headless-Smoke +
  `test_platform.py` + `test_headless.py` im ARM64-Image.
- `onnxruntime` stellt `manylinux_aarch64`-Wheels bereit; CPU-Provider
  auf Pi 4/5 damit verfügbar. Verifikation auf echter Pi-Hardware steht
  aus (siehe BLOCKED).

## Raspberry Pi 4/5 — BLOCKED (keine Hardware verfügbar)

- Nicht durchführbar ohne Gerät: GPIO/Picamera2-Pfad, Hailo/IMX500,
  Thermik-/Leistungsmessung auf Pi, Echte-Kamera-Pipeline.
- Code-Seitig vorbereitet: `discover_hardware()` erkennt Pi-Board,
  CPU-Provider-Default, `detect_width: 320`-Empfehlung, adaptive
  Detektion (bounded 1..4), optionale Modelle hinter Flags.

## Abhängigkeits-Matrix

| Paket | AMD64 | ARM64 (Linux) | macOS arm64 | Status |
|---|---|---|---|---|
| opencv-contrib-python | ✓ | ✓ | ✓ (verifiziert) | Pflicht |
| mediapipe ≥1.1.1 (aarch64-Wheel) | ✓ | ✓ (CI) | ✓ | Pflicht |
| onnxruntime (Extra `insightface`) | Wheel ✓ | Wheel ✓ | Wheel ✓ | optional, Fallback ohne |
| picamera2 / RPi.GPIO | n/a | Linux-only | n/a | optional, Linux-Guards |
| React-Dashboard (vite build) | ✓ | ✓ | ✓ (vitest+tsc lokal) | statisch, kein 2. Server |
