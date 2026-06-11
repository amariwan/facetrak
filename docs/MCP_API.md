# MCP Server API Reference

The `facetrak-mcp` command starts an MCP (Model Context Protocol) server over stdio, exposing 20 tools for face detection, recognition, and tracking.

**Server name:** `FaceTrak`

Run with:
```bash
just mcp-serve
# or
python -m facetrak.mcp_server
```

## Face Database Tools

### `list_faces`

List all registered people.

**Returns:** Formatted list of names, or `"No faces registered."`.

### `forget_person(name: str)`

Delete a registered person.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Name of the person to forget |

### `register_person(name: str)`

Register a new person from the camera feed. Captures face samples over ~3 seconds.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Name to register |

**Returns:** Success or failure message. Requires at least 3 sharp face samples.

### `get_face_info(name: str)`

Show encoding details about a registered person.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Registered person name |

**Returns:** Name and encoding vector shape.

## Camera Tools

### `camera_list`

List all configured camera sources and auto-detect USB cameras.

**Returns:** Camera list with index, name, source, active marker, and detected USB cameras with resolution.

### `camera_add(name: str, source: str)`

Add a camera source to `config.json`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Human-readable label (e.g. `"IP Cam"`) |
| `source` | `string` | Source: USB index (`"0"`) or RTSP URL (`"rtsp://..."`) |

### `camera_switch(index: int)`

Switch the active camera by its config index.

| Parameter | Type | Description |
|-----------|------|-------------|
| `index` | `integer` | Camera index from `camera_list` |

**Returns:** Success/failure message. Preserves recording state across switch.

### `start_camera`

Start the camera and begin face tracking. Launches a background polling thread at ~33 FPS.

### `stop_camera`

Stop the camera and face tracking.

### `get_status`

Get the current tracking status.

**Returns:** Multi-line status including camera state, source label, known face count, last face position, servo angles, head pose, recording state, and blur state.

### `toggle_recording`

Start or stop video recording (MP4, `mp4v` codec, 20 FPS).

### `toggle_blur`

Toggle privacy blur for unrecognized faces (Gaussian blur over the face region).

## Configuration Tools

### `get_config`

Show all current configuration values from `config.json`.

### `update_config(key: str, value: str)`

Update a configuration value. Supports dot-notation for nested keys.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `string` | Config key (e.g. `"detect_width"`, `"servo.smooth"`) |
| `value` | `string` | New value (auto-cast to int, float, or bool) |

**Examples:**
- `update_config("detect_width", "640")`
- `update_config("servo.smooth", "0.2")`
- `update_config("blur_unknown", "true")`

### `reset_config`

Reset configuration to factory defaults.

## Servo Tools

### `servo_list_ports`

List available serial ports for Arduino servo connection.

### `servo_connect(port: str)`

Connect to the servo controller on the given serial port.

| Parameter | Type | Description |
|-----------|------|-------------|
| `port` | `string` | Serial port (e.g. `"/dev/cu.usbmodem101"`) |

### `servo_disconnect`

Disconnect from the servo controller.

### `servo_set_enabled(enabled: bool)`

Enable or disable servo tracking.

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | `boolean` | Whether servos should track faces |

### `servo_set_angle(pan: int, tilt: int)`

Directly set pan/tilt angles (bypasses auto-tracking).

| Parameter | Type | Description |
|-----------|------|-------------|
| `pan` | `integer` | Pan angle (0-180, clamped to config limits) |
| `tilt` | `integer` | Tilt angle (0-180, clamped to config limits) |

## Arduino Serial Protocol

The servo controller communicates with an Arduino via serial at 115200 baud.

**Command:** `P<pan>T<tilt>\n`
- `pan`/`tilt` are zero-padded 3-digit integers (e.g. `P090T045\n`)

**Response:** `OK <pan> <tilt>\n` or `ERR\n`

The Arduino firmware steps toward the target at a maximum of 3° per 15 ms for smooth motion.
