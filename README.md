# FaceTrak

Real-time face detection, recognition, and pan-tilt tracking with OpenCV, MediaPipe, and YOLO.

[![Python](https://img.shields.io/badge/python-≥3.11-blue)](pyproject.toml) [![Version](https://img.shields.io/badge/version-0.2.0-blue)](pyproject.toml)

## Features

- **Face detection** via OpenCV YuNet (multi-face, 5-point landmarks)
- **Face recognition** with SFace deep embeddings — aligned 128-d vectors, top-k cosine matching with ambiguity rejection
- **Multi-face tracking** — stable IDs via IoU matching, identity smoothing by majority vote, per-face dwell time
- **Age & gender estimation** — Caffe-based models for demographic analysis
- **Expression & emotion analysis** — happy/sad/angry/surprised/neutral, smile, mouth/brow activity, eye openness, blink counting (MediaPipe blendshapes)
- **Head pose & attention** — yaw/pitch/roll via solvePnP; gaze direction (up/down/left/right/centre)
- **Liveness detection** — blink counting + head-turn challenge to prevent spoofing
- **Gesture recognition** — thumbs up/down, peace, fist, open, point, ok (MediaPipe Hands)
- **Pose estimation** — 33-landmark full-body pose with joint angles (MediaPipe Pose)
- **Object detection** — YOLOv8 (nano/small/medium auto-selection) for scene awareness
- **Camera abstraction** — USB webcams, RTSP IP cameras, ONVIF PTZ, Raspberry Pi Camera Module
- **Digital zoom** — ROI crop tracking driven by face bounding-box size
- **Monocular depth estimation** — MiDaS ONNX via OpenCV DNN (no PyTorch required)
- **Motion sensing** — PIR motion sensor (Raspberry Pi GPIO) with debounced events
- **Audio monitoring** — real-time microphone event detection (clap, voice, loud, silence)
- **Presence & crowd analytics** — SQLite-backed logs of who appeared/left; peak/average crowd counts with CSV export
- **Emotion timeline** — time-series emotion samples per tracked face with CSV export
- **Face heatmap** — position accumulation with exponential decay and Jet colormap overlay
- **Privacy blur** — pixelate unknown faces or per-person blur toggle
- **Pan-tilt servo tracking** — serial communication with Arduino to physically follow faces
- **Video recording** — MP4 capture of the tracking session
- **macOS notifications** — alert when a known person is detected
- **Tkinter GUI** — live camera view, face telemetry dashboard, emotion pie chart, controls
- **MCP server** — 20+ tools for LLM agent integration (Claude Desktop, etc.)
- **REST API** — FastAPI server with endpoints for status, faces, presence, crowd, exports
- **3D simulation** — animated pan-tilt model for testing without hardware

## Quick Start

```bash
# Install
just setup

# Or manually:
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run the GUI
just run

# Run the MCP server (for LLM integration)
just mcp-serve

# Run the REST API server
just api-serve

# Standalone pan-tilt simulation (no camera)
just sim
```

## Requirements

- Python ≥ 3.11
- macOS (for GUI and notifications)
- Arduino + servos (optional, for physical tracking)
- Raspberry Pi (optional, for PIR sensor, Pi Camera)
- Webcam, RTSP camera, or ONVIF PTZ camera

Model files are auto-downloaded on first run (~350 KB YuNet, ~37 MB SFace, ~5 MB FaceLandmarker, ~70 MB MediaPipe models).

Optional extras: `pip install -e ".[full]"` for all hardware (Linux) and AI extras.

## Commands

| `just` command | Description |
|---|---|
| `just run` | Launch the Tkinter tracker app |
| `just mcp-serve` | Start the MCP server (stdio) |
| `just api-serve` | Start the HTTP REST API server (port 8765) |
| `just setup` | Install package + dev dependencies |
| `just sim` | Run pan-tilt simulation (no camera) |
| `just list-faces` | Show registered face names |
| `just forget NAME` | Delete a registered person |
| `just clear-faces` | Delete all registered faces |
| `just sync` | Update dependency versions in pyproject.toml |
| `just clean` | Clear caches and temp files |
| `just update` | Upgrade the package via pip |
| `just install dep=PKG` | Install a single dependency |

## Usage

### GUI Mode

```
facetrak
```

Command bar: Start/Stop camera, Record video, Blur, Heatmap, Servo toggle, Register person, List faces, Simulation window, camera selector.

Right dashboard: face telemetry (name, pose, servo angles, dwell time, blinks, age/gender), attention gauge, emotion pie chart, face management list with per-person blur toggles.

Status bar: camera name, face position, servo angles, head pose, emotion, face count, recording state, known face count.

Keyboard shortcuts: `<Space>` register, `<r>` record, `<b>` blur, `<h>` heatmap, `<Esc>` stop.

### MCP Mode

```
facetrak-mcp
```

Connect any MCP-compatible client — 20+ tools including face database management, camera control, servo control, config, presence history, crowd stats, and face analysis. See [docs/MCP_API.md](docs/MCP_API.md).

### REST API Mode

```
facetrak-api
```

FastAPI server on port 8765 by default. Endpoints: `/status`, `/faces`, `/face/analysis`, `/presence`, `/crowd`, `/emotions`, `/camera/start`, `/camera/stop`, `/snapshot`, `/export/crowd`, `/export/emotions`. Browse interactive docs at `http://localhost:8765/docs`.

### Registration

Face data is stored as `.npy` files in `faces/data/`. To register a person:

1. Ensure the person is visible to the camera
2. Click **Register** in the GUI, or use `register_person` via MCP
3. Look at the camera for ~3 seconds (samples are filtered by sharpness & brightness + optional liveness check)
4. The system captures up to 20 quality-filtered SFace embeddings (128-d) and stores them

## Configuration

Settings are persisted in `config.json`:

| Key | Default | Description |
|---|---|---|
| `camera` | `0` | Legacy camera index |
| `cameras` | `[{"type":"usb","index":0}]` | Multi-camera sources (usb/rtsp/onvif/pi) |
| `detect_width` | `480` | Detection resolution (width) |
| `recog_threshold` | `0.36` | SFace cosine similarity threshold |
| `blur_unknown` | `false` | Privacy blur for unrecognized faces |
| `liveness.enabled` | `true` | Anti-spoofing liveness check |
| `liveness.min_blinks` | `2` | Blinks required for liveness |
| `liveness.min_yaw` | `12` | Yaw spread required (degrees) |
| `zoom.enabled` | `false` | Digital zoom tracking |
| `zoom.target_ratio` | `0.30` | Target face-to-frame height ratio |
| `sensors.audio` | `false` | Microphone monitoring |
| `sensors.pir_pin` | `null` | PIR sensor GPIO pin |
| `sensors.depth` | `false` | Depth estimation |
| `servo.port` | `""` | Serial port for Arduino |
| `servo.baud` | `9600` | Serial baud rate |
| `servo.pan_min/max` | `0` / `180` | Pan angle limits |
| `servo.tilt_min/max` | `0` / `180` | Tilt angle limits |
| `servo.dead_zone` | `15` | Dead zone in pixels |
| `servo.smooth` | `0.12` | Smoothing factor (0-1) |
| `servo.max_step` | `3.0` | Max angle change per frame |
| `servo.invert_pan/tilt` | `false` | Invert axis direction |

## Arduino Setup

Flash `facetracker.ino` to your Arduino with two servos on pins 9 (pan) and 10 (tilt). Connect the serial port and set `servo.port` in `config.json`.

Protocol: `P<pan>T<tilt>\n` → responds `OK <pan> <tilt>\n`.


See [docs/DEVELOPER.md](docs/DEVELOPER.md) for detailed architecture documentation and data flow.

## License

MIT
