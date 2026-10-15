# FaceTrak Next — Abschlussbericht (2026-10-15)

Single delivery phase. Verifiziert auf macOS Apple Silicon, Python 3.13.

## Ergebnis-Tabelle (Pflichtformat)

| Bereich | Ergebnis |
|---|---|
| Implementierte Funktionen | Vollständige Liste unten (§1) |
| Wiederverwendete Projekte | Keine vendored; Konzepte/Patterns aus go2rtc (MIT), Frigate (MIT), double-take (MIT), CompreFace (MIT), tracking-screen (MIT). boxmot (AGPL) und InsightFace bewusst nicht integriert, Details in `docs/oss-licenses.md` |
| Verwendete Skills | `using-superpowers`, `brainstorming` (opt-out, 1-Satz-Bauentscheidung), `agentic-engineering` (Laya-Entscheid `laya`, Prozess-Loop), `code-review` (Review-Findings behoben), `opencv` v4.9.0 + `python-opencv` (projektlokal, für Zonen/Masken/Kamera-Disziplin). Kein `find-skills`-Install (bereits abgedeckt, Supply-Chain-Risiko), Begründung in `docs/engineering/skills-selection.md` |
| Tests | 96 bestanden / 0 fehlgeschlagen (47 Baseline + 49 neu). CI-Matrix macOS/Linux × 3.11/3.13 definiert, hier nur macOS+3.13 ausgeführt |
| Performance | Skeleton-Benchmark: 198,6 fps, step p50 5,0 ms / p95 5,4 ms (Stub-Detektor 4 ms + 2 Masken + 2 Zonen, 200 Frames). Echte YuNet/MediaPipe-Latenzen gerätespezifisch, nicht gemessen (keine Kamera vorhanden) — keine erfundenen Gewinne |
| Plattformen | macOS Apple Silicon verifiziert (Tests, API-Smoke, Benchmark). Linux/Raspberry Pi nur via CI-Matrix + Mocks (kein Hardwaretest) |
| Breaking Changes | Keine stillschweigenden. Bewusst: REST bindet default `127.0.0.1` (vorher `0.0.0.0`); `/blur/{name}` akzeptiert zusätzlich JSON-Body; MCP destruktive Tools haben optionalen `confirm`-Param (additiv); frische `config.json` nutzt Servo-Baud 115200 (bestehende behalten 9600); `facetrak.db`/Modelle aus Git entfernt (Daten bleiben auf Platte) |
| Offene Risiken | Ohne Token ist API für lokale Prozesse offen (by design, localhost-only); MCP ohne Scopes (nur vertrauenswürdige Clients); Tracker-Qualität (ID-Switches) nicht quantifiziert ohne Videodaten; `ui._poll` lädt config pro Frame (ungenutzt lassen oder fixen); SQLite ohne WAL bei konkurrierenden Writern |
| Nicht abgeschlossen | Next.js-Dashboard (durch build-freies Dashboard ersetzt, ADR-1); CoreML/ONNX-Backend (Protokoll-Seam gelegt, kein messbarer Bedarf); boxmot-Tracker; Pi-Hardwaretest; e2e-UI-Tests |

## §1 Implementierte Funktionen

**A. Secure Platform Foundation** — `127.0.0.1`-Default, Bearer-Auth
(env `FACETRAK_API_TOKEN` > `api.auth_token`), Envelope-Fehlerformat,
Pydantic-Validierung, Secret-Redaction (URLs/MQTT/ONVIF), atomare
Config-Saves, `FACETRAK_CONFIG`/`FACETRAK_DB`-Overrides, Retention
(`retention_days`, `/retention/purge`, `DELETE /data` mit Consent-Gate
für Faces), `hide_estimates`-Schalter, MCP-Confirm-Gate
(`FACETRAK_MCP_REQUIRE_CONFIRM`), sensible Pfade git-ignoriert
(`facetrak.db`, Exporte, Modelle aus Tracking entfernt).
**B. Modulare Engine** — `core/interfaces.py` (Detector/Tracker/
InferenceBackend-Protokolle), YuNet-`ready`-Guard (kein Crash mehr bei
fehlgeschlagenem `create`), `detect_every`-Konfiguration,
Frame-Drop-Zähler, FPS/Latenz-Perf-Stats (`/metrics`), Threshold-Warnung.
**C. Kamera** — Health-Tracking (frames/failures/reconnects/last-frame-age)
für USB/RTSP, RTSP-Logs redigiert, `/cameras`-Health-API, go2rtc als
optionale externe Integration (URLs + Health-Probe).
**D. Zonen & Events** — `facetrak/events/`: Polygon-Zonen, Privacy-Masks
(pre-detection Blackout), Enter/Exit-Events → SQLite + In-Memory-Ring +
Subscriber + MQTT, `/events /zones /masks`-API mit Validierung.
**E. Dashboard** — `dashboard.html` (6 KB, Dark Mode, responsiv,
Live-MJPEG, Status/Perf/Faces/Events, Token-Feld, Fehlerzustände).
Tkinter-GUI unverändert. Kein zweites Vision-System.
**F. MQTT** — optionaler Publisher, anonymes Standard-Schema, Filter,
Reconnect-Backoff, `paho-mqtt` nur als optionales Extra + Lazy-Import.
**G. Servo** — `max_speed_dps`-Cap, Dead-Zone-Hold, Reacquire-to-Center
nach `reacquire_frames`, Kalibrierungs-Offsets (applies at send),
Winkel-Clamps, voll ohne Arduino testbar (9 Tests).
**H. DX** — `just test/lint/bench/purge`, GitHub-Actions-CI,
`scripts/benchmark.py`, pytest/ruff-Config, Setup-/Betriebsdoku,
MCP-Doku aktualisiert.

## §2 Verifikation (Evidenz, keine Simulation)

- `.venv/bin/python -m pytest tests/ -q` → **96 passed** (Baseline vor
  Start: 47 passed).
- `ruff check` auf allen neuen Dateien: sauber; geänderte Dateien:
  nur vorbestehende Findings (E702-Semikolons etc., bewusst nicht
  angefasst), Import-Sortierung in angefassten Blöcken behoben.
- API-Smoke via TestClient: `/health`, `/` (6270 Bytes HTML),
  `/metrics`-Keys ok.
- `scripts/benchmark.py --frames 200` → 198,6 fps, p95 5,37 ms.
- Secret-Scan: keine Tokens/Keys im Tree (CI-Job wiederholt das).

## Bekannte Einschränkungen

Siehe Offene Risiken + Nicht abgeschlossen oben. Gesichtserkennung
gilt nur für registrierte Personen mit Einwilligung; Alters-/Geschlechts-/
Emotions-/Aufmerksamkeitswerte sind Schätzungen und per
`privacy.hide_estimates` ausblendbar.
