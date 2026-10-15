# Open-Source Evaluation — FaceTrak Next

All licenses checked against the upstream repository state known at
delivery time (2026-10-15). No third-party code was vendored; all
integrations are optional and interface-based.

| Project | Function in FaceTrak | License (upstream) | Verdict |
|---|---|---|---|
| AlexxIT/go2rtc | Optional external streaming (WebRTC/MJPEG/HLS URLs, health check) | MIT | **Reuse as external service** — `facetrak/integrations/go2rtc.py` only builds URLs + health probe. Not bundled, disabled by default. |
| blakeblackshear/frigate | Zone + event concepts (polygons, enter/exit, per-zone analysis flag) | MIT | **Concepts reimplemented** (`facetrak/events/zones.py`, ~150 lines, OpenCV `pointPolygonTest`). No Frigate code reused — full NVR out of scope by design. |
| mikel-brostrom/boxmot | Tracking algorithms | AGPL-3.0 | **Not integrated** — AGPL copyleft conflicts with this repo's permissive posture, and the IoU+majority-vote tracker has no measured ID-switch problem yet. `core/interfaces.py:Tracker` protocol keeps the door open; a future adapter must isolate AGPL code as an optional process. |
| exadel-inc/CompreFace | API security / service management patterns | MIT (server) | **Patterns adopted, not deployed** — Bearer-token auth, localhost default, unified error envelope in `servers/rest.py`. Running CompreFace itself would duplicate the existing SFace pipeline. |
| deepinsight/insightface | Alternative inference backends | MIT (code; model weights separate) | **Evaluated, deferred** — heavier dep chain (onnxruntime, model zoo) with no measured accuracy gap on registered-face task. `Detector`/`InferenceBackend` protocols + lazy `load()`/`release()` prepare the seam. Weight licenses must be re-checked per model before any future use. |
| jakowenko/double-take | MQTT + integration patterns | MIT | **Patterns adopted** — anonymized event schema, topic-per-kind, reconnect/backoff in `integrations/mqtt.py` (stdlib-only, `paho-mqtt` lazy optional extra). No double-take code reused. |
| Quackieduckie/tracking-screen | Servo tracking / motion control | MIT (verify before reuse) | **Ideas adopted, self-implemented** — velocity cap (`max_speed_dps`), dead-zone hold, reacquire-to-center, calibration offsets in `hardware/servo.py`. No code copied. |

## Dependency posture

- Runtime defaults unchanged (`opencv-contrib-python`, `mediapipe`,
  `numpy`, `pillow`, `pyserial`, `mcp`, `fastapi`, `uvicorn`).
- New optional extra: `mqtt = ["paho-mqtt>=1.6"]` — imported lazily
  only when `mqtt.enabled=true`; tests inject fakes, zero new install
  required for default operation.
- Dev group (`pytest`, `httpx`, `ruff`) is test/CI-only.
- Removed from git tracking (kept on disk / auto-downloaded):
  `facetrak.db` (PII), `age/gender .caffemodel/.prototxt` (~90 MB
  binaries, auto-download with graceful disable), model `.onnx/.task`
  were already ignored.
