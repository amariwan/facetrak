import threading
import time
from mcp.server.fastmcp import FastMCP
import cv2
from facetrak import config
from facetrak.facedb import FaceDatabase
from facetrak.servo import PanTiltController
from facetrak.engine import FaceEngine

server = FastMCP(
    "FaceTrak",
    instructions="Face detection, recognition, and tracking via MCP",
)

_engine: FaceEngine | None = None
_poll_thread: threading.Thread | None = None
_db = FaceDatabase()


def _get_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine


def _poll_loop():
    e = _get_engine()
    while e.running:
        e.step()
        time.sleep(0.03)


# ── Face Database Tools ──────────────────────────────────


@server.tool(description="List all registered people")
def list_faces() -> str:
    _db.load()
    names = _db.known_names
    if not names:
        return "No faces registered."
    return f"Registered faces ({len(names)}):\n" + "\n".join(
        f"  {i+1}. {n}" for i, n in enumerate(names)
    )


@server.tool(description="Forget (delete) a registered person")
def forget_person(name: str) -> str:
    from pathlib import Path
    import glob
    pattern = str(Path("faces/data") / f"{name.replace(' ', '_')}.npy")
    found = glob.glob(pattern)
    if not found:
        return f"Person '{name}' not found."
    for f in found:
        Path(f).unlink()
    _db.load()
    return f"Forgot '{name}'."


@server.tool(description="Register a new person from camera. Name is required.")
def register_person(name: str) -> str:
    e = _get_engine()
    started_here = not e.running
    if started_here:
        if not e.start():
            return "Cannot start camera."
        time.sleep(1)
    e.capture_samples()
    e.set_overlay(f"Registering {name}...")
    deadline = time.time() + 3
    while time.time() < deadline:
        if started_here:
            e.step()  # no UI poll loop is running for us
        time.sleep(0.03)
    ok = e.register(name)
    e.set_overlay("")
    if started_here:
        e.stop()
    if ok:
        return f"Registered '{name}' successfully."
    return f"Failed to register '{name}' — no face samples captured."


@server.tool(description="Show details about a registered person")
def get_face_info(name: str) -> str:
    _db.load()
    names = _db.known_names
    if name not in names:
        return f"Person '{name}' not found."
    idx = names.index(name)
    return f"Name: {names[idx]}\nEncoding: {_db.encodings[idx].shape} vector"


# ── Camera Tools ────────────────────────────────────────


@server.tool(description="List available camera sources")
def camera_list() -> str:
    cfg = config.load()
    cams = cfg.get("cameras", [])
    result = []
    for i, c in enumerate(cams):
        marker = " ← active" if i == cfg.get("camera", 0) else ""
        result.append(f"  [{i}] {c['name']} ({c['source']}){marker}")
    result.append("")
    result.append("Auto-detect USB cameras:")
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result.append(f"  USB {idx}: {w}x{h}")
            cap.release()
        else:
            cap.release()
    if len(result) == 1:
        result.append("  (none found)")
    return "\n".join(result)


@server.tool(description="Add a camera to the config (e.g. name='IP Cam', source='rtsp://...')")
def camera_add(name: str, source: str) -> str:
    cfg = config.load()
    cams = cfg.setdefault("cameras", [])
    cams.append({"name": name, "source": source})
    config.save(cfg)
    return f"Added camera '{name}' ({source})."


@server.tool(description="Switch to a camera by its config index (see camera_list)")
def camera_switch(index: int) -> str:
    global _poll_thread
    e = _get_engine()
    cfg = config.load()
    cams = cfg.get("cameras", [])
    if index < 0 or index >= len(cams):
        return f"Camera index {index} out of range (0-{len(cams)-1})."
    label = config.label(cfg, index)
    if not e.running:
        cfg["camera"] = index
        config.save(cfg)
        return f"Switched to [{index}] {label} (start camera to use)."
    was_rec = e.recorder.recording
    ok = e.switch_camera(index)
    if not ok:
        e.start(cfg.get("camera", 0))
        return f"Failed to switch to [{index}] {label}."
    if was_rec and not e.recorder.recording:
        e.toggle_record()
    return f"Switched to [{index}] {label}."


@server.tool(description="Start the camera and face tracking")
def start_camera() -> str:
    global _poll_thread
    e = _get_engine()
    if e.running:
        return "Camera already running."
    if not e.start():
        return "Failed to start camera."
    _poll_thread = threading.Thread(target=_poll_loop, daemon=True)
    _poll_thread.start()
    return "Camera started."


@server.tool(description="Stop the camera and face tracking")
def stop_camera() -> str:
    global _poll_thread
    e = _get_engine()
    if not e.running:
        return "Camera not running."
    e.stop()
    _poll_thread = None
    return "Camera stopped."


@server.tool(description="Get current tracking status")
def get_status() -> str:
    e = _get_engine()
    _db.load()
    cfg = config.load()
    cam_label = config.label(cfg, e.current_cam_idx)
    if e.running:
        cx, cy = e.last_face_center
        fw, fh = e.last_face_size
        return (
            f"Camera: running\n"
            f"Source: {cam_label}\n"
            f"Known faces: {len(_db.known_names)}\n"
            f"Last face: ({cx}, {cy}) {fw}x{fh}\n"
            f"Servo: P={e.current_pan:.0f} T={e.current_tilt:.0f}\n"
            f"Pose: Y={e.current_yaw:.1f} P={e.current_pitch:.1f} R={e.current_roll:.1f}\n"
            f"Recording: {'yes' if e.recorder.recording else 'no'}\n"
            f"Privacy blur: {'on' if e.blur_enabled else 'off'}"
        )
    return (
        f"Camera: stopped\n"
        f"Active source: {cam_label}\n"
        f"Known faces: {len(_db.known_names)}\n"
        f"Servo connected: {'yes' if e.servo.connected else 'no'}"
    )


@server.tool(description="List all faces currently visible, with track ID, "
                         "identity, dwell time and blink count")
def get_live_faces() -> str:
    e = _get_engine()
    if not e.running:
        return "Camera not running."
    faces = e.live_faces()
    if not faces:
        return "No faces in view."
    lines = []
    for f in faces:
        lines.append(
            f"  #{f['id']} {f['name']} (sim {f['similarity']}) — "
            f"in view {f['dwell_s']}s, {f['blinks']} blinks, bbox {f['bbox']}")
    return f"{len(faces)} face(s) in view:\n" + "\n".join(lines)


@server.tool(description="Get expression/pose analysis of the primary face: "
                         "emotion, smile, eye openness, attention, head pose")
def get_face_analysis() -> str:
    e = _get_engine()
    if not e.running:
        return "Camera not running."
    m = e.metrics
    return (
        f"Emotion: {m.emotion}\n"
        f"Smile: {m.smile:.2f}\n"
        f"Mouth open: {m.mouth_open:.2f}\n"
        f"Brow raise: {m.brow_raise:.2f}\n"
        f"Eyes open (L/R): {m.eye_left:.2f}/{m.eye_right:.2f}\n"
        f"Attentive (facing camera): {'yes' if m.attentive else 'no'}\n"
        f"Head pose: yaw={m.yaw:.1f} pitch={m.pitch:.1f} roll={m.roll:.1f}"
    )


@server.tool(description="Show the presence history (who appeared/left, "
                         "when, and for how long)")
def presence_history(limit: int = 30) -> str:
    from facetrak.events import PresenceLog
    events = PresenceLog.tail(limit)
    if not events:
        return "No presence events recorded yet."
    lines = []
    for ev in events:
        if ev.get("event") == "left":
            lines.append(f"  {ev['ts']}  {ev['name']} left "
                         f"after {ev.get('duration_s', '?')}s")
        else:
            lines.append(f"  {ev['ts']}  {ev['name']} appeared")
    return "Presence history:\n" + "\n".join(lines)


@server.tool(description="Save a snapshot of the current camera frame to disk")
def take_snapshot() -> str:
    e = _get_engine()
    if not e.running:
        return "Camera not running."
    path = e.snapshot()
    return f"Snapshot saved: {path}" if path else "No frame available yet."


@server.tool(description="Start or stop video recording")
def toggle_recording() -> str:
    e = _get_engine()
    if not e.running:
        return "Camera not running."
    e.toggle_record()
    return "Recording started." if e.recorder.recording else "Recording stopped."


@server.tool(description="Toggle privacy blur for unknown faces")
def toggle_blur() -> str:
    e = _get_engine()
    state = e.toggle_blur()
    return f"Privacy blur {'on' if state else 'off'}."


# ── Config Tools ────────────────────────────────────────


@server.tool(description="Show current configuration")
def get_config() -> str:
    cfg = config.load()
    return "\n".join(f"  {k}: {v}" for k, v in sorted(cfg.items()))


@server.tool(description="Update a configuration value (e.g. detect_width, recog_threshold)")
def update_config(key: str, value: str) -> str:
    cfg = config.load()
    if key not in cfg and not any(
        key.startswith(k + ".") for k in cfg
    ):
        return f"Unknown key '{key}'."
    try:
        old = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            old = old[p]
        old_val = old[parts[-1]]
        if isinstance(old_val, bool):
            new_val = value.lower() in ("true", "1", "yes")
        elif isinstance(old_val, int):
            new_val = int(value)
        elif isinstance(old_val, float):
            new_val = float(value)
        else:
            new_val = value
        cfg = config.load()
        target = cfg
        for p in parts[:-1]:
            target = target[p]
        target[parts[-1]] = new_val
        config.save(cfg)
        return f"Updated '{key}': {old_val} → {new_val}"
    except (ValueError, KeyError, TypeError) as e:
        return f"Failed: {e}"


@server.tool(description="Reset configuration to defaults")
def reset_config() -> str:
    config.save(config.DEFAULT_CONFIG)
    return "Config reset to defaults."


# ── Servo Tools ─────────────────────────────────────────


@server.tool(description="List available serial ports for servo connection")
def servo_list_ports() -> str:
    ports = PanTiltController.list_ports()
    if not ports:
        return "No serial ports found."
    return "Available ports:\n" + "\n".join(f"  {p}" for p in ports)


@server.tool(description="Connect to a servo controller on given port")
def servo_connect(port: str) -> str:
    e = _get_engine()
    if e.servo.connected:
        e.servo.disconnect()
    cfg = config.load()
    ok = e.servo.connect(port, cfg["servo"]["baud"])
    if ok:
        return f"Connected to {port}."
    return f"Failed to connect to {port}."


@server.tool(description="Disconnect from servo controller")
def servo_disconnect() -> str:
    e = _get_engine()
    if not e.servo.connected:
        return "Not connected."
    e.servo.disconnect()
    return "Disconnected."


@server.tool(description="Enable or disable servo tracking")
def servo_set_enabled(enabled: bool) -> str:
    e = _get_engine()
    e.servo_enabled = enabled
    e.servo.enabled = enabled
    return f"Servo tracking {'enabled' if enabled else 'disabled'}."


@server.tool(description="Directly set pan/tilt angles (0-180)")
def servo_set_angle(pan: int, tilt: int) -> str:
    e = _get_engine()
    if not e.servo.connected:
        return "Servo not connected."
    actual_pan, actual_tilt = e.servo.move_to(float(pan), float(tilt))
    return f"Pan: {actual_pan:.0f}, Tilt: {actual_tilt:.0f}"


def run():
    server.run(transport="stdio")
