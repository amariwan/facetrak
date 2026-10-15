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
- `mediapipe` 0.10.35: **kein manylinux_aarch64-Wheel**
  (nur macosx_arm64, manylinux_x86_64, win) ✗ → auf Pi auto-degradiert,
  kein Install-Fehler, keine Crashs. Volle Analyse nur auf x86_64/macOS.
- `picamera2`/`RPi.GPIO`: Pi-only, korrekt optional markiert, guarded Imports.
- Kein Core ML, kein AVFoundation, kein osascript im Produktionspfad
  (per Test bewiesen).

## 5. CI-Ergebnisse

- Erweitert: Matrix macOS-14/py3.13 (dev), Ubuntu-22.04/py3.11+3.12,
  **ubuntu-24.04-arm/py3.12**, dazu `headless --check`, Docker-Build +
  Container-Smoke. ARM-Job läuft nur soweit GitHub-Runner verfügbar.
- Lokal (MacBook, dev-only): **117/117 Tests PASS** (96 Bestand + 21 neu),
  `ruff` auf allen neuen/geänderten Dateien sauber (Rest-Findings in
  fremden Dirty-Zeilen bewusst unberührt), API-Smoke `/system` ok,
  `headless --check` Exit 0.
- CI wurde hier nicht ausgeführt (kein Push beauftragt) → CI-Status: UNVERIFIED.

## 6. systemd- und Docker-Nachweise

- systemd-Unit statisch geprüft (Test): Restart, SIGTERM, Hardening,
  Env-File, `headless`-Exec. Echter `systemctl`-Lauf: BLOCKED (kein Linux).
- Dockerfile/Compose statisch geprüft: non-root, no-new-privileges,
  kein privileged, Healthchecks, localhost-Bind, Secrets via Env-File.
  Echter `docker build`/Run: BLOCKED (MacBook-Docker ≠ Pi-Runtime;
  CI-Job baut das Image).
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
