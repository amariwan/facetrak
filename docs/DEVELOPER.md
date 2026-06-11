# Developer Documentation

## Architecture Overview

```
facetrak/
├── __main__.py    CLI entry → MainWindow (Tkinter)
├── config.py      JSON config with defaults merging
├── engine.py      FaceEngine — orchestrates all subsystems
├── facedb.py      HOG-based face recognition (enrollment & prediction)
├── mcp_server.py  MCP server (FastMCP) — 20 tools over stdio
├── notifier.py    macOS desktop notifications via osascript
├── pose.py        3DoF head pose from MediaPipe landmarks + solvePnP
├── recorder.py    MP4 video recording via cv2.VideoWriter
├── servo.py       Serial pan-tilt controller + angle interpolation
├── simulation.py  2D animated pan-tilt visualization (Tkinter Canvas)
└── ui.py          Tkinter GUI (video display, toolbar, status bar)
```

The `FaceEngine` (engine.py) is the central coordinator. It owns the camera capture, face detector, face database, servo controller, pose estimator, video recorder, and notifier. Both the GUI (`MainWindow`) and the MCP server (`mcp_server.py`) drive the engine.

## Module Reference

### `config.py` — Configuration Manager

Loads and persists `config.json` with automatic defaults merging. Nested keys use dot notation for updates (e.g. `"servo.smooth"`).

Key functions:
- `load() → dict` — read config, merge defaults, save
- `save(cfg)` — write config to disk
- `label(cfg, idx) → str` — human-readable camera name
- `source(cfg, idx)` — camera source (int for USB, str for RTSP)

### `engine.py` — FaceEngine

**Lifecycle:** `start()` → `step()` loop → `stop()`

- **Detection:** Downscales frame to `detect_width`, runs MediaPipe BlazeFace, scales detections back to original resolution
- **Recognition:** Finds highest-confidence face, runs `FaceDatabase.predict()` on the crop
- **Servo tracking:** Converts face offset (from frame center) to proportional pan/tilt via `PanTiltController.update()`
- **Pose estimation:** Runs `HeadPoseEstimator.estimate()` on each frame's best face
- **Privacy blur:** Applies Gaussian blur to unrecognized face regions
- **Sample collection:** Buffers face crops filtered by Laplacian variance (>30) and brightness (40-220)
- **Registration:** Encodes buffered samples and stores via `FaceDatabase.register()`

### `facedb.py` — Face Database

Uses HOG (Histogram of Oriented Gradients) features via `cv2.HOGDescriptor` for face encoding.

- **Eye alignment:** Haar cascade eye detection → rotation correction via `cv2.warpAffine`
- **Contrast enhancement:** CLAHE (clipLimit=2.0) before encoding
- **Encoding:** HOG features on 64×128 grayscale, L2-normalized
- **Storage:** `.npy` files in `faces/data/` — each file is a matrix of feature vectors (one row per sample)
- **Prediction:** Cosine similarity (dot product of normalized vectors). Uses best-to-second ratio check (threshold 1.12×) plus absolute threshold (default 0.55)

### `servo.py` — Pan-Tilt Controller

Serial communication with Arduino. Angle calculation:

- **Proportional control:** `offset / (frame_dim / 2) * max_angle`
  - Pan: ±60° from center, Tilt: ±45° from center
- **Smoothing:** `current += (target - current) * smooth`, clamped by `max_step`
- **Dead zone:** No movement if offset < threshold
- **Serial format:** `P<pan:03d>T<tilt:03d>\n`

### `pose.py` — Head Pose Estimation

- Uses MediaPipe FaceLandmarker (6 landmarks: nose tip, chin, eye corners, mouth corners)
- Standard 3D face model with solvePnP (ITERATIVE method)
- Focal length: `img_w * 0.8`, principal point at image center
- Returns: yaw, pitch, roll in degrees

### `recorder.py` — Video Recorder

- Wraps `cv2.VideoWriter` with `mp4v` codec at 20 FPS
- Filename: `recording_YYYYMMDD_HHMMSS.mp4`

### `notifier.py` — macOS Notifications

- Uses `osascript` for native macOS notifications
- Deduplication: one notification per recognized person per session
- Sound: `"Tink"`

### `simulation.py` — 2D Pan-Tilt Simulation

Tkinter Canvas rendering with:
- Ground ellipse, rotating base, pole, tilt joint, camera head, lens, light beam
- Compass arc (pan reference) and tilt arc
- Live updates at 25 FPS via `after()` loop
- Standalone demo with sine-wave animation (`just sim`)

### `ui.py` — Tkinter GUI

`MainWindow` class:
- **Toolbar:** Start/Stop, Record, Blur, Servo, Register, List Faces, Simulation, camera combo
- **Video display:** `Label` widget updated at ~33 FPS (30ms polling)
- **Status bar:** Camera name, face position, pan/tilt, yaw/pitch/roll, REC indicator, known face count

### `mcp_server.py` — MCP Server

Built on `mcp.server.fastmcp.FastMCP`. Runs over stdio.

- Background polling thread at 30ms when camera is active
- 20 tools organized into 4 groups: Face Database, Camera, Config, Servo
- See [MCP_API.md](MCP_API.md) for complete tool reference

## Arduino Firmware

`facetracker.ino` controls two servos (pan = pin 9, tilt = pin 10) at 115200 baud.

- **Protocol:** `P<pan>T<tilt>\n` → `OK <pan> <tilt>\n` or `ERR\n`
- **Smooth stepping:** Max 3° per 15ms toward target
- **Angle clamping:** 0-180° with configurable reversal
- **Non-blocking:** Serial reads in the main loop without `delay()`

## Data Flow

```
Camera → [downscale] → MediaPipe BlazeFace → [scale up] → best face
                                                              │
                    ┌─────────────────────────────────────────┤
                    │              │              │           │
                    ▼              ▼              ▼           ▼
              FaceDatabase  PanTiltController  HeadPose   Recorder
              .predict()    .update(dx,dy)    .estimate() .write()
                    │              │
                    ▼              ▼
               overlay text    serial → Arduino → servos
```

## Recognition Pipeline

```
face crop → eye alignment (Haar cascade) → CLAHE → resize 64×128
→ HOGDescriptor.compute() → L2 normalize → dot product with enrolled vectors
→ best-sim > threshold & best/second ratio > 1.12 → match
```

## Extending

To add a new tool to the MCP server:

```python
@server.tool(description="Your tool description")
def your_tool(param: str) -> str:
    # implementation
    return "result"
```

To add a new feature to the engine:

1. Add the subsystem as a module (e.g. `new_feature.py`)
2. Instantiate it in `FaceEngine.__init__()`
3. Call it in `FaceEngine.step()` or expose a method
4. Wire it in `MainWindow` (GUI) and/or `mcp_server.py` (MCP)
