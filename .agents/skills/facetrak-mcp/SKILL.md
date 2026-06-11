# FaceTrak MCP — LLM Interface for Face Tracking

Use this skill when the user mentions face tracking, face recognition, face scanning, servo tracking, face detection with camera, or anything related to the FaceTrak project. This skill tells an LLM how to use the FaceTrak MCP server (stdio-based) to query and control a real-time face detection, recognition, and pan-tilt tracking system.

## Architecture

FaceScan exposes 20 tools via MCP (Model Context Protocol) over stdio. The server wraps:
- `facedb` — HOG-based face recognition with `.npy` encoding storage
- `engine` — MediaPipe face detection + OpenCV camera loop
- `servo` — Serial pan/tilt controller (Arduino)
- `pose` — Head pose estimation via FaceLandmarks + solvePnP
- `recorder` — Video recording to MP4
- `config` — JSON config persistence

Start the server:
```
just mcp-serve
```
or: `facetrak-mcp` CLI command or `python -m facetrak.mcp_server`

In Claude Desktop config:
```json
{
  "mcpServers": {
    "facetrak": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "facetrak.mcp_server"]
    }
  }
}
```

## Tools Overview

### Face Database (no camera needed)
| Tool | When to use |
|------|-------------|
| `list_faces` | User asks "who is registered", "show me known faces", "list people" |
| `get_face_info(name)` | User asks about a specific person's encoding/details |
| `forget_person(name)` | User says "delete X", "remove Y", "forget Z" |
| `register_person(name)` | User says "add new person", "register X", "learn this face" — starts camera, captures samples, registers |

### Camera Control
| Tool | When to use |
|------|-------------|
| `camera_list` | User asks "which cameras", "what sources", "find cameras", or to discover available USB/IP cameras |
| `camera_add(name, source)` | User wants to add an IP camera (RTSP URL) or name a USB camera |
| `camera_switch(index)` | User wants to switch to a different camera by index from `camera_list` |
| `start_camera` | User wants to start tracking or see the camera feed |
| `stop_camera` | User wants to stop tracking or release the camera |
| `get_status` | User asks "what's happening", "is it running?", current pan/tilt/pose, active camera |
| `toggle_recording` | User wants to start/stop recording video |
| `toggle_blur` | User wants to enable/disable privacy blur on unknown faces |

### Configuration
| Tool | When to use |
|------|-------------|
| `get_config` | User asks "show settings", "what's configured" |
| `update_config(key, value)` | User wants to change a setting (e.g. detection width, recognition threshold, servo limits) |
| `reset_config` | User wants factory defaults |

### Servo Control
| Tool | When to use |
|------|-------------|
| `servo_list_ports` | User asks "what ports are available", "find my Arduino" |
| `servo_connect(port)` | User wants to connect to a servo controller |
| `servo_disconnect` | User wants to disconnect |
| `servo_set_enabled(enabled)` | User wants to turn tracking on/off |
| `servo_set_angle(pan, tilt)` | User wants to set a specific angle (0-180) |

## Important Notes

- **Face DB is persistent** — stored as `.npy` files in `faces/data/`. Tools like `list_faces`, `forget_person`, `get_face_info` work immediately without camera.
- **Camera is on-demand** — `start_camera` opens the camera and starts background polling. `register_person` temporarily starts/stops camera if needed.
- **Multi-camera** — `camera_list` detects all USB cameras + shows configured ones. `camera_add` adds RTSP URLs or IP cameras. `camera_switch` changes live source.
- **Registration** — `register_person(name)` opens camera, watches for 2 seconds collecting face crops, then saves. If no face is visible, it fails.
- **Config keys** — flat keys like `detect_width`, `recog_threshold`, `camera`, `blur_unknown`, or dotted servo keys like `servo.port`, `servo.pan_min`, `servo.baud`.
- **Servo angles** — clamped to configured min/max (default 0-180). Send `P090T045\n` style protocol over serial.
- **Head pose** — yaw/pitch/roll available via `get_status` when camera is running.

## Typical Workflows

**"Who do you know?"** → `list_faces` → print the list

**"Register Alice"** → `register_person("Alice")` → wait 2s → confirm

**"Start tracking"** → `start_camera` → `servo_list_ports` → `servo_connect("/dev/cu.usbmodem...")` → `servo_set_enabled(true)` → `get_status`

**"Switch to IP camera"** → `camera_add("Garage", "rtsp://192.168.1.100:554/stream")` → `camera_switch(1)` → `get_status`

**"What cameras are available?"** → `camera_list` → shows all USB + configured cameras

**"What's the current status?"** → `get_status` → shows pan/tilt, pose, recording state, face position

**"Delete Bob"** → `forget_person("Bob")` → confirm

**"Change detection resolution to 640"** → `update_config("detect_width", "640")`

**"Record video"** → `start_camera` → `toggle_recording` → ...later... → `toggle_recording`

**"Turn on privacy blur"** → `start_camera` → `toggle_blur` → on

**"Move servo to center"** → `servo_set_angle(90, 90)`

## Skill Metadata

- Skill name: facetrak-mcp
- Trigger on: face tracking, face recognition, face detection, face scan, servo tracking, camera tracking, person registration, "who is this", "recognize face"
