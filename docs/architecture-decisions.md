# Architecture Decision Records — FaceTrak Next

## ADR-1: No Next.js dashboard; single-file dashboard served by FastAPI

Context: the order preferred Next.js/React/TS/shadcn/Tailwind "sofern
nach Architekturprüfung sinnvoll".
Decision: serve `servers/dashboard.html` (dependency-free, ~6 KB, dark
mode, responsive, MJPEG preview, token field) directly from FastAPI.
Reasons: (1) no second vision system — preview reuses `engine._frame`;
(2) no Node/build toolchain for a local single-user console;
(3) Tkinter GUI stays the rich client; the web UI is a status/control
surface. A Next.js frontend can be added later against the same REST
API without backend changes.

## ADR-2: go2rtc stays an external optional service

Vendoring or requiring go2rtc would add a Go binary + second stream
router for a problem the engine already solves (single-open camera +
MJPEG preview). `integrations/go2rtc.py` only constructs stream URLs
and probes health when `go2rtc.enabled=true`.

## ADR-3: MQTT anonymous by default

`integrations/mqtt.py` publishes `{source, kind, ts, data}` with
`track_id`/`zone` only; `name` fields are stripped unless
`mqtt.include_names=true` is explicitly set (consent-gated). Reconnect
uses the client's backoff (1–30 s).

## ADR-4: Servo default baud 115200

Firmware `facetracker.ino` speaks 115200; config default was 9600, so
servos were silent on fresh installs. New configs default to 115200;
existing `config.json` files keep their stored value (deep-merge only
adds missing keys) → non-breaking.

## ADR-5: Threshold clamp now logs

`recog_threshold > 0.5` was silently reset to 0.36. Behavior kept, but
a warning is logged so misconfiguration is visible.

## ADR-6: Atomic config saves; CWD-relative paths kept

`config.save` uses tmp+rename. `FACETRAK_CONFIG` / `FACETRAK_DB` env
overrides allow isolated tests and multi-instance operation, but the
default CWD-relative layout is unchanged for backward compatibility.

## ADR-7: boxmot/InsightFace deferred behind protocols

AGPL (boxmot) and weight-license complexity (InsightFace) vs. no
measured tracking/accuracy deficit → `core/interfaces.py` defines
`Detector`/`Tracker`/`InferenceBackend` protocols now; adapters later,
only with proven need.

## Target architecture (as built)

```text
FaceTrak Next
├── Dashboard (dashboard.html, optional) + Tkinter GUI (existing)
├── FastAPI (auth, envelope, zones/masks/events/metrics/preview) / MCP (confirm-gated) 
├── Core Engine
│   ├── Camera (health-tracked) + Vision pipeline (masks pre-detect)
│   ├── Tracking Engine (protocol-ready) + Model Runtime (lazy)
│   ├── Event Engine (zones → SQLite + MQTT + subscribers)
│   ├── Privacy Manager (masks, hide_estimates, retention/erasure)
│   └── Plugin Interfaces (core/interfaces.py)
├── Integrations (go2rtc URL/health, MQTT optional, servo serial)
└── Infrastructure (SQLite + retention, config atomic, CI, bench)
```
