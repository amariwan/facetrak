# Pi-Feldtest-Protokoll (auszuführen sobald Hardware verfügbar)

Ziel: BLOCKED-Punkte aus `INSIGHTFACE_ACCEPTANCE.md` messbar schließen.
Jeweils 1 h Dauerlauf + 5 min Burst, Lüfter-/Gehäuse-Setup notieren.

## Setup

- [ ] Raspberry Pi 4 (4/8 GB) + Pi 5, Raspberry Pi OS / Debian, Kernel notieren
- [ ] `detect_width: 320` in `config.json` (Pi-Empfehlung)
- [ ] Kamera: USB/V4L2 oder Picamera2 — Modell + Auflösung + FPS notieren
- [ ] Branch/Commit notieren; `pip list` (opencv, mediapipe, onnxruntime?) sichern

## Messreihen (je Gerät)

| # | Konfiguration | Dauer | FPS | P50/P95 step_ms | RSS max | Temp max | Governor-State |
|---|---|---|---|---|---|---|---|
| 1 | YuNet, `detect_width` 320, adaptiv an | 60 min | | | | | |
| 2 | YuNet, `detect_width` 480, adaptiv an | 15 min | | | | | |
| 3 | YuNet, adaptiv aus (`detect_every: 1`) | 5 min Burst | | | | | |
| 4 | SCRFD 500M (nur mit lizenzierten Gewichten + onnxruntime) | 15 min | | | | | |
| 5 | SFace-ReID an (3 registrierte Personen) | 15 min | | | | | |

Befehle:

```bash
just headless-check
.venv/bin/python scripts/benchmark.py --frames 200
.venv/bin/python scripts/benchmark_insightface.py --frames 100 --output /tmp/bench-pi.json
```

## Abnahme-Kriterien

- Kein Thermal-Throttle unter 60 min Dauerlast bei Reihe 1 (sonst
  `thermal`-Schwellen / `detect_target_ms` nachschärfen).
- P95 step_ms < 100 ms bei Reihe 1; sonst `detect_width` 320 Pflicht.
- Headless-Smoke + `test_platform.py` + `test_headless.py` grün auf Gerät.
- Anonymize-Batch eines 5-min-Clips ohne Abbruch (Fortschritt monoton).

## Ergebnisse hier eintragen

| Datum | Gerät | Commit | Reihe | Ergebnis (PASS/FAIL + Werte) |
|---|---|---|---|---|
| | | | | |
