# Benchmarkbericht (FaceTrak 2.1) — nur gemessene Werte

Stand: 2026-10-16. Keine erfundenen Leistungsgewinne. Alles nicht Messbare
ist als BLOCKED markiert.

## Gemessen: YuNet, macOS arm64 (Dev-Maschine, 100 synthetische Frames)

`scripts/benchmark_insightface.py --frames 100` (`/tmp/bench_macos.json`):

| Metrik | YuNet |
|---|---|
| Frames | 100 |
| Mean | 5.12 ms |
| Median / P50 | 5.11 ms |
| P95 | 5.94 ms |
| P99 | 6.20 ms |
| Min / Max | 3.92 / 6.29 ms |
| Std | 0.48 ms |
| FPS (1/mean) | 195.3 |
| Detektionen/Frame (Synthetik) | 0.0 |

Einordnung: reiner Pipeline-Durchsatz auf schneller Dev-Hardware, **kein**
Pi-Wert, **kein** Genauigkeitsvergleich (synthetische Frames ohne echte
Gesichter). Speicherverbrauch/Modellstartzeit wurden in diesem Lauf nicht
separat instrumentiert — `scripts/benchmark.py` (Bestand) deckt den
reproduzierbaren Rahmen ab; `benchmark_insightface.py` erweitert ihn um
P50/P95/FPS pro Backend, sobald lizenzierte Modelle vorliegen.

## Nicht messbar (BLOCKED — Begründung)

| Vergleich | Status | Grund |
|---|---|---|
| YuNet vs. SCRFD (Latenz/FPS/P50/P95) | BLOCKED | keine lizenzierten SCRFD-Gewichte, kein onnxruntime lokal |
| SFace vs. ArcFace (Embedding-Qualität) | BLOCKED | keine lizenzierten ArcFace-Gewichte |
| MediaPipe-Liveness vs. RGB-Liveness | BLOCKED | kein kompatibles RGB-Liveness-Modell lizenziert |
| CPU-Inferenz AMD64/ARM64 (Pi) | BLOCKED | keine Pi-Hardware; CI baut nur Images, misst keine Latenzen |
| Speicher / Modellstartzeit SCRFD/ArcFace | BLOCKED | wie oben |

## Reproduktion

```bash
.venv/bin/python scripts/benchmark_insightface.py --frames 100 --output /tmp/bench.json
.venv/bin/python scripts/benchmark.py --frames 200   # Bestand, No-Camera
```

Sobald lizenzierte InsightFace-Gewichte + onnxruntime vorliegen, derselbe
Befehl mit `"insightface": {"enabled": true}` in `config.json` — das
Benchmark-Skript erkennt den Adapter automatisch und berichtet zusätzlich
SCRFD/ArcFace-Zeilen sowie Modell-Ladezeiten.
