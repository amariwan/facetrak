# FaceTrak

Real-time face detection, recognition, and pan-tilt tracking with OpenCV and MediaPipe.

[![Python](https://img.shields.io/badge/python-≥3.11-blue)](pyproject.toml)

## Features

- **Face detection** via Google MediaPipe BlazeFace (short-range model)
- **Face recognition** using HOG feature encoding with eye-alignment and CLAHE
- **Head pose estimation** — yaw, pitch, roll via MediaPipe landmarks + solvePnP
- **Pan-tilt servo tracking** — serial communication with Arduino to physically follow faces
- **Video recording** — MP4 capture of the tracking session
- **Privacy blur** — pixelate unknown faces on demand
- **macOS notifications** — alert when a known person is detected
- **Tkinter GUI** — live camera view, controls, status overlay
- **MCP server** — expose all functionality to LLM agents (Claude Desktop, etc.)
- **2D simulation** — animated pan-tilt model for testing without hardware

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

# Standalone pan-tilt simulation (no camera)
just sim
```

## Requirements

- Python ≥ 3.11
- macOS (for GUI and notifications)
- Arduino + servos (optional, for physical tracking)
- Webcam (built-in or USB)

Model files are auto-downloaded from Google on first run (~100 KB detector, ~5 MB landmarker).

## Commands

| `just` command | Description |
|---|---|
| `just run` | Launch the Tkinter tracker app |
| `just mcp-serve` | Start the MCP server (stdio) |
| `just setup` | Install package in editable mode |
| `just sim` | Run pan-tilt simulation (no camera) |
| `just list-faces` | Show registered face names |
| `just forget NAME` | Delete a registered person |
| `just clear-faces` | Delete all registered faces |
| `just sync` | Update dependency versions in pyproject.toml |
| `just clean` | Clear caches and temp files |
| `just update` | Upgrade the package via pip |

## Usage

### GUI Mode

```
facetrak
```

Toolbar controls: Start/Stop camera, Record video, Blur toggle, Servo toggle, Register person, List faces, Simulation window, camera selector.

The status bar shows: camera name, face position, servo angles (pan/tilt), head pose (yaw/pitch/roll), recording state, and known face count.

### MCP Mode

```
facetrak-mcp
```

Connect any MCP-compatible client (Claude Desktop, etc.) to this stdio server. 20 tools available — see [docs/MCP_API.md](docs/MCP_API.md).

### Registration

Face data is stored as `.npy` files in `faces/data/`. To register a person:

1. Ensure the person is visible to the camera
2. Click **Register** in the GUI, or use `register_person` via MCP
3. Look at the camera for ~3 seconds (samples are filtered by sharpness & brightness)
4. The system captures 128×128 HOG feature vectors and stores them

## Configuration

Settings are persisted in `config.json`:

| Key | Default | Description |
|---|---|---|
| `camera` | `0` | Active camera index |
| `detect_width` | `480` | Detection resolution (width) |
| `recog_threshold` | `0.55` | Recognition similarity threshold |
| `blur_unknown` | `false` | Privacy blur for unrecognized faces |
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

## Architecture

```
facetrak/
├── __main__.py    CLI entry point (Tkinter GUI)
├── config.py      JSON config load/save
├── engine.py      Core loop: camera, detection, tracking, recognition
├── facedb.py      HOG-based face recognition database
├── mcp_server.py  MCP server (20 tools for LLM integration)
├── notifier.py    macOS desktop notifications
├── pose.py        3D head pose estimation (yaw/pitch/roll)
├── recorder.py    Video recording (MP4)
├── servo.py       Serial pan-tilt controller
├── simulation.py  2D animated pan-tilt simulation
└── ui.py          Tkinter GUI (video display, controls)
```

See [docs/DEVELOPER.md](docs/DEVELOPER.md) for detailed architecture documentation.

## License

MIT
