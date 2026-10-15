# FaceTrak — macOS Setup & Local Operation

## Requirements

- macOS Apple Silicon, Python 3.11+ (3.13 tested), camera permission
  for Terminal/IDE
- No cloud, no account, no additional services for default operation

## Install (reproducible)

```bash
cd facetrak
uv venv .venv 2>/dev/null || python3 -m venv .venv
.venv/bin/pip install -e .        # or: uv pip install -e .
.venv/bin/python -m pytest tests/ -q   # expect 96 passed
```

First start downloads models automatically into the project directory
(YuNet ~350 KB, SFace ~37 MB, MediaPipe task files, age/gender Caffe
files). Downloads are verified by loader success; failures disable the
feature with a warning instead of crashing.

## Run

```bash
just run                 # Tkinter GUI (needs display)
just api-serve            # REST + dashboard on http://127.0.0.1:8765
just mcp-serve            # MCP over stdio
just sim                  # pan-tilt simulation, no hardware
just bench                # perf benchmark, no camera
```

## Secure remote access (opt-in)

1. Bind LAN: `"api": {"host": "0.0.0.0"}` in `config.json` (default is
   `127.0.0.1` — changing it is a conscious act).
2. Set a token: `export FACETRAK_API_TOKEN=$(openssl rand -hex 32)`
   (env wins over `api.auth_token`; never commit the token).
3. Dashboard asks for the token and sends it as `Authorization: Bearer`.

Without a token, every endpoint except `/health` is reachable by any
local process — acceptable for single-user localhost, not for shared
machines.

## Privacy & retention

- Face recognition only matches **registered** persons; unknown faces
  stay "unknown". Registration needs physical presence + liveness.
- Age/gender/emotion/gaze are **estimates**: set
  `privacy.hide_estimates=true` to hide them from API/MCP/dashboard.
- `retention_days` (default 30): `POST /retention/purge` or automatic
  purge on demand; `DELETE /data?confirm=true` erases all rows;
  `&include_faces=true` additionally deletes `faces/data/*.npy`, but
  only with `privacy.allow_identifying_exports=true`.
- `facetrak.db`, exports, snapshots, recordings and model binaries are
  git-ignored and never committed.

## Camera & streaming

- USB/RTSP/ONVIF/Pi sources via `config.json → cameras[]`; RTSP
  passwords stay server-side (`/cameras` and logs show redacted URLs).
- Health: `GET /cameras` shows frames/failures/reconnects/last-frame
  age per camera; RTSP reconnects with 1–30 s backoff automatically.
- Multi-client viewing: point browsers at `/preview.mjpg` or put
  go2rtc in front (`go2rtc.enabled=true`, `url`); never open the same
  camera twice — the engine holds the single connection.

## Events & automation

- Zones: `POST /zones {"name","polygon":[[x,y]…]}` (normalized 0..1);
  enter/exit lands in SQLite (`GET /events`) and, if enabled, MQTT
  (`facetrak/zone_enter`, anonymized by default).
- Privacy masks: `POST /masks` — blacked out **before** detection.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Servo silent | baud mismatch: set `servo.baud=115200` (firmware default); check `servo.port` |
| `Camera not running` (503) | no camera found; `just sim` / `just bench` work without one |
| 401 unauthorized | `FACETRAK_API_TOKEN` set server-side but missing in dashboard token field |
| Slow frames on Pi | `detect_width: 320`, disable depth/pose/gestures |
| GUI won't start headless | use `just api-serve` + dashboard instead (Tk needs display) |
