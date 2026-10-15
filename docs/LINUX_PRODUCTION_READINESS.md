# FaceTrak — Linux Production Readiness (2026-10-15)

Linux-first-Umbau auf Basis der fünf Commits aus Phase 1
(`ef8ea90`…`f1f52a3`). macOS war ausschließlich Dev-/Testmaschine;
keine Messung unten ist ein Linux-Hardware-Nachweis.

## 1. Implementierte Anpassungen

- **Portable Core vs. Adapter getrennt** (`facetrak/platform/`):
  OS/Pi-Erkennung, Headless-Check, V4L2/AVFoundation-Auswahl,
  systemd-/Docker-Signalhandling (SIGTERM/SIGINT, exakt einmal),
  stdlib-only Systemmetriken (CPU-Temp, Load, Mem, RSS,
  Pi-Throttle-State), Soft-RAM-Guard (warnt + GC, killt nie).
- **Produktionspfad GUI-frei nachgewiesen**: `viz/__init__.py` lädt
  kein Tkinter mehr (vorher zog `SimulationWindow` Tkinter in jede
  Engine-Instanz); Tests prüfen statisch (kein `import tkinter`
  außerhalb `viz/`) und per Laufzeit (kein `viz.ui/simulation` in
  `sys.modules` nach Import von rest/mcp/headless).
- **MediaPipe entkoppelt**: `analysis/__init__` + Engine laden
  `FaceAnalyzer` nur wenn importierbar; sonst läuft alles außer
  Emotions-/Gaze-/Pose-/Gesten-Analyse. Auf Pi **Pflicht**, nicht
  Luxus (siehe §4).
- **Kamera Linux-first**: USB-Backend V4L2 statt AVFoundation-Probing
  (auch im MCP-Probe), RTSP-Backoff + Health unverändert, keine
  Doppel-Öffnung (Engine hält die einzige Verbindung).
- **Pi-Optimierungen**: adaptive Detektionsfrequenz (1..4 Frames,
  Ziel-Latenz konfigurierbar, selbst-neutralisierend auf schneller HW),
  `detect_width` 320-Doku, Bounded Queues (RTSP maxsize 2, Drop-oldest),
  Lazy Model Loading (unverändert beibehalten), saubere Freigabe
  (Kamera/GPIO/Serial in `stop()` + Shutdown-Manager).
- **Datenverzeichnisse**: `FACETRAK_DATA`/`data_dir` für Snapshots,
  Recordings, CSV-Exporte (Docker-Volume `/data`, systemd
  `/var/lib/facetrak`).
- **Deployment**: `Dockerfile` (non-root, kein privileged, Healthcheck,
  Headless-Entrypoint), `docker-compose.yml` (127.0.0.1-Bind,
  Token via `.env`, optionales Mosquitto-Profil),
  `deploy/facetrak.service` (Restart, Hardening, DeviceAllow auskommentiert
  dokumentiert), `.dockerignore`, `just headless/headless-check/docker-*`.
- **API**: `GET /system` (Plattform + Metriken + Ressourcen + go2rtc),
  Dashboard-Hardwarekarte. Security-Defaults unverändert
  (localhost, Bearer, Envelope, Redaction).
- **Notifier**: kein `osascript`-Versuch mehr außerhalb macOS.

## 2. Zielplattformen

| Plattform | Status |
|---|---|
| Raspberry Pi OS 64-bit (Pi 4/5) | Primärziel; Install-Weg definiert, HW-Test BLOCKED |
| Ubuntu/Debian x86_64 | Docker + bare metal definiert; Container-Build in CI |
| Ubuntu ARM64 | CI-Job konfiguriert; Geräte-Test BLOCKED |
| Headless-Server | `headless --check` + systemd + Compose vorhanden |
| macOS | Nur Dev/Test/Simulation, keine Produktionsfreigabe |

## 3. Installationsanleitungen

- `docs/install-raspberrypi.md` (Pi OS, piwheels, systemd, Token, Tuning)
- `docs/install-debian.md` (Ubuntu/Debian bare metal + Docker, Kamera-Passthrough)
- `config.example.json` (Pi-Tuning, aus `DEFAULT_CONFIG` generiert), `.env.example`

## 4. ARM64-Kompatibilität (PyPI-Evidenz, 2026-10-15)

- `opencv-contrib-python` 4.13.0.92: **manylinux_aarch64-Wheels vorhanden** ✓
- `numpy` 2.4.6: aarch64-Wheels ✓ (Upstream-Standard)
- `mediapipe` 1.1.1 (Release 2026-10-14): **manylinux_2_28_aarch64-Wheel
  vorhanden** ✓ (39 MB, ABI `py3-none`, glibc ≥ 2.28 — bookworm/bullseye
  ok). Floor in `pyproject.toml` auf `>=1.1.1` angehoben, `uv.lock`
  synchronisiert.
- API-Kompatibilität (gegen installiertes 1.1.1 verifiziert):
  FaceLandmarker/PoseLandmarker/HandLandmarker (Tasks API) laden und
  inferieren; legacy-`mp.solutions` (in 1.x entfernt) aus
  `pose.py`/`gestures.py` auf Tasks-API migriert, öffentliche
  Methoden-Signaturen unverändert. Fallback bei fehlendem MediaPipe
  bleibt dokumentiert erhalten (Graceful-Disable + Warnung).
- Privacy: MediaPipe Tasks sendet Nutzungs-/Perf-Metriken an Google
  (Upstream-Hinweis) — Einwilligungspflicht beachten; voll offline =
  MediaPipe weglassen, Fallback greift.
- `picamera2`/`RPi.GPIO`: Pi-only, korrekt optional markiert, guarded Imports.
- Kein Core ML, kein AVFoundation, kein osascript im Produktionspfad
  (per Test bewiesen).

## 5. CI- und Validierungsergebnisse (echte Läufe, 2026-10-15)

| # | Check | Umgebung | Ergebnis |
|---|---|---|---|
| 1 | pytest voll (121 Tests) | macOS arm64, mp 1.1.1, venv | PASS |
| 2 | Tasks-Backend-Regression (face/pose/hand, echte Inferenz) | macOS arm64, mp 1.1.1 | PASS (4/4) |
| 3 | `headless --check` | macOS + Container (beide Arch) | PASS, Exit 0 |
| 4 | pytest Subset (114 Tests, ohne Deploy-/Tasks-Tests) | linux/arm64-Container (QEMU) | PASS |
| 5 | Tasks-Inferenz pose/hand | linux/arm64-Container (Final-Image mit libegl1+libgles2) | PASS (4/4 `test_tasks_backend.py`) |
| 6 | Tasks-Inferenz pose/hand | linux/amd64-Container (Final-Image) | PASS (4/4 `test_tasks_backend.py`) |
| 7 | `systemd-analyze verify deploy/facetrak.service` | ubuntu:24.04-Container | PASS |
| 8 | Image-Build + Smoke (Importe cv2/mp, UID 10001, `--check`) | linux/amd64 + linux/arm64 Final-Images | PASS |
| 9 | API live (`/health`, `/system` 200, Linux/x86_64/Py3.12-Felder) | linux/amd64-Container | PASS |
| 10 | SIGTERM (`docker stop`) | linux/amd64-Container | PASS (sauberer Stopp) |
| 11 | GitHub-Actions-Läufe (x86_64, ARM, Docker, systemd) | — | UNVERIFIED (kein Push beauftragt; Workflow liegt vor) |
| 12 | Pi-Hardware (GPIO, CSI, Temp/Throttle, 24 h, Kamera) | — | BLOCKED (kein Pi verfügbar) |

- Lokal (MacBook, dev-only): `ruff` auf allen neuen/geänderten Dateien
  sauber; API-Smoke `/system` ok.
- CI wurde hier nicht ausgeführt (kein Push beauftragt) → CI-Status: UNVERIFIED.

## 6. systemd- und Docker-Nachweise

- systemd-Unit **echt verifiziert**: `systemd-analyze verify
  deploy/facetrak.service` in `ubuntu:24.04`-Container → PASS
  (einzige Meldung ohne Stub-Pfad war der erwartete ExecStart-Hinweis;
  mit angelegtem Interpreter-Pfad: clean PASS). CI-Job `systemd`
  wiederholt das. Echter `systemctl`-Start/Restart: BLOCKED (kein Linux-Host).
- Dockerfile/Compose statisch geprüft: non-root, no-new-privileges,
  kein privileged, Healthchecks, localhost-Bind, Secrets via Env-File.
- Docker-Images: amd64-Build + arm64-Build, jeweils Smoke
  (`headless --check`), Container-Checks (non-root-UID, `/health`,
  SIGTERM-Shutdown) — Ergebnisse in der Matrix unten.
- Shutdown-Logik (Handler, exakt-einmal, Trigger) per Unit-Test PASS.

## 7. Performance

- Mac-Skeleton-Benchmark (Phase 1): ~199 fps, p95 ~5,4 ms — **kein
  Pi-Wert**, nur Regressionsschwelle für die Pipeline-Hülle.
- Pi-Budget ableiten aus Config: 320 px + adaptiv (Ziel 50 ms) +
  YuNet (~350 KB) + SFace alle 5 Frames → erwartet ~10–15 fps auf Pi 5,
  zu bestätigen. `GET /metrics` + `/system` liefern fps/Latenz/Drops/
  RSS/Temp/Throttle für Dauerläufe.

## 8. Raspberry-Pi-Hardwareteststatus — alles BLOCKED

| Test | Status |
|---|---|
| USB-Kamera (V4L2) öffnen/lesen | BLOCKED — keine Kamera am MacBook |
| Pi Camera (picamera2/libcamera) | BLOCKED — kein Pi |
| RTSP-Reconnect am Gerät | BLOCKED (Logik per Mock PASS) |
| GPIO/PIR + Servo an Arduino | BLOCKED (Sim-Tests PASS) |
| CPU/RAM/Dauerbetrieb 24 h | BLOCKED — Prozedur: `/metrics`+`/system` samplen |
| systemd-Start/Restart/Stop | BLOCKED |
| Docker auf Pi (ARM-Image, Device-Passthrough) | BLOCKED |
| Update/Rollback per Symlink-Verfahren | BLOCKED |

Update/Rollback-Konzept: neuer Tree nach `/opt/facetrak.new`,
dort `headless --check` + pytest, Symlink-Flip, `systemctl restart`;
Rollback = Symlink zurück + Restart. DB ist SQLite-Datei unter
`/var/lib/facetrak` — vor Updates kopieren. Logrotation: journald
(`SystemMaxUse=` in `journald.conf`) + Beispiel in §9 der
Pi-Doku-Notizen unten.

## 9. Bekannte Einschränkungen

- MediaPipe-Features auf Pi deaktiviert (by design, geloggt).
- `uv.lock` enthält macOS-resolved Pins; auf Pi löst `pip` neu auf
  (getestete Mindestversionen in `pyproject.toml`).
- SQLite ohne WAL: ein Schreiber (Engine) + Leser (API) ok; kein
  Multi-Writer.
- Fremde Dirty-Tree-Änderungen (8 Dateien, z. B. `facedb.py`,
  `recorder.py`) nicht von dieser Phase, unberührt im Tree.
- ONNX-Runtime/CoreML: weiter nur Protokoll-Seam, kein Bedarf gemessen.

## 10. Noch erforderliche echte Hardwaretests

In dieser Reihenfolge auf Pi 5: 1) Install-Doc Schritt für Schritt,
2) `headless --check`, 3) USB-Kamera 10 min + `/metrics`-Fps,
4) 24-h-Dauerlauf mit Temp/Throttle/RSS-Log, 5) Kamera abziehen
(Reconnect), 6) systemd-Reboot-Test, 7) Docker-ARM-Image mit
`/dev/video0`, 8) Servo-Bewegung + Reacquire, 9) MQTT gegen Broker.
Erst danach gilt eine Pi-Freigabe.
