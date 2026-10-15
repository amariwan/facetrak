# OSS License Audit — FaceTrak 1.0

FaceTrak license: MIT (repo root LICENSE). Rule: no GPL/AGPL code
incorporated without explicit compatibility assessment. Model weights
assessed separately from code licenses.

## New code in this delivery

All modules under `facetrak/vision/`, `facetrak/device/`,
`facetrak/video/`, `facetrak/automation/`, `facetrak/services/`,
`facetrak/servers/routers_v1.py`, `web/src/*` (dashboard),
`scripts/install.sh` are **hand-written for this repo (MIT)**.
No third-party code copied. External systems referenced by protocol
only (MQTT, HTTP, go2rtc HTTP API).

## Dependencies (pyproject.toml — unchanged, reviewed)

| Package | License | Notes |
|---|---|---|
| opencv-contrib-python | Apache-2.0 | YuNet/SFace via OpenCV Zoo models |
| mediapipe (Tasks API) | Apache-2.0 | Face/pose/hand landmarkers; `.task` bundles redistributable under Apache-2.0 |
| numpy, pillow | BSD | — |
| pyserial | BSD-3 | servo serial |
| mcp, fastapi, uvicorn | Apache-2.0 / MIT / BSD | servers |
| ultralytics (optional) | AGPL-3.0 | **Flagged**: optional `yolo` extra only; never imported in default path. Distribution with AGPL component requires compliance review before enabling. Status: NOT integrated into new code paths. |
| picamera2 / RPi.GPIO (optional, Linux) | BSD-2 / MIT-style | Import-guarded; absent hardware => graceful fallback |
| onvif-zeep, paho-mqtt, sounddevice | MIT/Apache | optional extras |

## Reference repositories reviewed (no code copied)

| Repo | License | Integration approach | Status |
|---|---|---|---|
| AlexxIT/go2rtc | MIT | Existing HTTP integration reused (`integrations/go2rtc.py`); WebRTC not reimplemented (F22) | Approved |
| blakeblackshear/frigate | MIT* (docs/config concepts) | Architecture reference only (multi-cam, go2rtc, MQTT) | Reference only |
| mikel-brostrom/boxmot | AGPL-3.0 | **NOT incorporated** — greedy IoU + constant-velocity tracker hand-written instead (F01/F05/F06) to avoid AGPL | Rejected for copy |
| exadel-inc/CompreFace | Apache-2.0 (check subcomponents) | Concepts reference (model registry metadata) | Reference only |
| deepinsight/insightface | MIT (weights: non-commercial research for some) | **NOT used** — SFace retained; weights license risk avoided | Rejected |
| jakowenko/double-take | MIT | MQTT/HA discovery pattern reference (F31) | Reference only |
| hailo-ai/hailo-rpi5-examples | MIT (HailoRT proprietary EULA separate) | Adapter designed against example API; `hailo_platform` import-guarded, CPU fallback (F11) | Approved w/ runtime-EULA note |
| raspberrypi/picamera2, imx500-models | BSD-3 / Pi terms | `picamera2` import-guarded (F12); no model blobs vendored | Approved |
| home-assistant/core, node-red/node-red | Apache-2.0 | Protocol-level interop (MQTT discovery, HTTP) — no code reuse (F31/F32) | Approved |

## Model assets

YuNet/SFace ONNX (OpenCV Zoo, Apache-2.0). MediaPipe `.task` files
(Apache-2.0). No InsightFace/Hailo/IMX500 weights bundled. Open-vocab
(F03): no weights bundled — adapter reports availability honestly.

## Approval

No GPL/AGPL code in the default dependency path. The single AGPL
touchpoint (optional ultralytics extra, pre-existing) is isolated and
documented. No new license obligations introduced.
