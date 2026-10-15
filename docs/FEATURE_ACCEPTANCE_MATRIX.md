# Feature Acceptance Matrix — FaceTrak 1.0

Tests: `tests/test_facetrak1.py` (T), `tests/test_api_v1.py` (A),
`web/src/main.test.tsx` (W). Full suite: **171 passed** (macOS dev,
2026-10-15). Linux/Docker/systemd: CI jobs (pending run on push).
Pi/Hailo/IMX500 hardware: BLOCKED (no device in this environment).

| ID | Feature | Baseline | Implementation | Config | Tests | Runtime | HW | Docs | Status |
|---|---|---|---|---|---|---|---|---|---|
| F01 | Multi-object tracking | Face-only IoU tracker | `vision/tracking.py` MultiObjectTracker (classes, conf, pluggable algo) | `tracking.*` | T stable-ids/aging, conf-negative | pytest pass | n/a (algo hw-independent) | arch + report | PASS |
| F02 | Segmentation | none | `vision/perception.py` SegmentationService + CPU fallback + overlay | `segmentation.*` | T fallback/disabled | pytest pass | n/a | report | PASS |
| F03 | Open-vocab detection | none | OpenVocabularyDetector: validation, capability, budget, graceful unsupported | `openvocab.*` | T validation/graceful | pytest pass | n/a (no weights; honest empty) | report | PASS (software) |
| F04 | Depth-aware tracking | stereo/depth estimator existed | DepthTracker relative-depth signal + match bonus; units labeled estimate-not-meters | `depth_tracking.*` | T relative/bonus | pytest pass | n/a | report | PASS |
| F05 | Occlusion recovery | basic track aging | prediction-through-occlusion, max_age, id-switch/false-reassoc metrics | `tracking.max_age` | T aging, metrics | pytest pass | n/a | report | PASS |
| F06 | Tracking prediction | none | constant-velocity predict/update in MOT (iou_kalman) | `tracking.prediction/algorithm` | T velocity-learned | pytest pass | n/a | report | PASS |
| F07 | Multi-camera handover | none | `vision/handover.py`: relations+windows; person links need session token, else refused | `handover.*` | T occupancy + refusal | pytest pass | n/a (logic only) | report | PASS |
| F08 | Model registry | hardcoded YuNet/SFace | `vision/models.py` ModelRegistry: full metadata, sha256 validation, activation | API `/api/v1/models` | T validate/activate; A crud | pytest+API pass | n/a | report | PASS |
| F09 | Backend switching | OpenCV only | BackendManager: 6 runtimes, detect/switch/run w/ CPU fallback, stable contract | `inference.backend` | T switch/contract; A | pytest+API pass | accel runtimes absent → fallback reported | report | PASS |
| F10 | Benchmark lab | scripts/benchmark.py ad-hoc | BenchmarkLab: warmup, p50/95/99, mem/cpu, JSON+CSV, mode labeled | `benchmark.*` | T modes/csv; A run | pytest+API pass | synthetic vs hardware labeled | PERF report | PASS |
| F11 | Hailo HAT+ | none | `device/accelerators.py` HailoAdapter: discovery, compat, metrics, fallback | `hailo.*` | T graceful | pytest pass (fallback path) | BLOCKED (no HAT+) | report | SOFTWARE PASS / HW BLOCKED |
| F12 | IMX500 AI camera | picamera2 driver only | IMX500Adapter: discovery, capabilities, lifecycle, fallback | `imx500.*` | T graceful | pytest pass (fallback) | BLOCKED (no AI camera) | report | SOFTWARE PASS / HW BLOCKED |
| F13 | Auto discovery | partial (USB scan in MCP) | `discover_hardware()`: USB/V4L2, board, GPIO, serial, runtimes + `/hardware/discovery` | — | T keys; A smoke | pytest+API pass | Pi-specific entries need Pi | report | PASS (host) / Pi fields BLOCKED |
| F14 | Servo calibration | offset-only calibrate | ServoCalibrator: bounded proposals, per-step confirm, profile persist, reset, estop, sim | `servo_cal.*` | T confirm/bounds/estop | pytest pass | movement on HW BLOCKED | report | SOFTWARE PASS / HW BLOCKED |
| F15 | Predictive servo | smoothing/vel-cap in PanTilt | PredictiveServo: vel/accel limits, smoothing, dead zone, loss handling + prediction | `servo_predict.*` | T bounds/recenter | pytest pass | HW BLOCKED | report | SOFTWARE PASS / HW BLOCKED |
| F16 | Multi-axis | pan/tilt hardcoded | MultiAxisController: N-axis profiles, pan/tilt preserved | `axes.*` | T move/unknown; A | pytest+API pass | HW BLOCKED | report | SOFTWARE PASS / HW BLOCKED |
| F17 | Watchdog | none | HardwareWatchdog: stale detection, bounded retries, failed state + API | `watchdog.*` | T recovery/failed; A | pytest+API pass | n/a | report | PASS |
| F18 | Thermal governor | none | ThermalGovernor: sysfs read, warn/throttle, staged reversible policy, faults surfaced | `thermal.*` | T policy/fault | pytest pass | sensor absent on mac (unknown state reported, not hidden) | report | PASS (software) |
| F19 | Low-power mode | adaptive detect_every | LowPowerMode: eco/deep profiles, idle detection, infer scaling | `power.*` | T profiles; A | pytest+API pass | n/a | report | PASS |
| F20 | GPIO bridge | PIR sensor direct | GPIOEventBridge: allowlist, debounce, sim, schema'd events | `gpio.*` | T debounce/allowlist; A emit | pytest+API pass | HW BLOCKED | report | SOFTWARE PASS / HW BLOCKED |
| F21 | Multi-camera grid | single-view dashboard | Dashboard Cameras section + `/cameras` API (per-cam status) | `cameras` | A smoke; W render | API+vitest pass | multi-USB needs HW | report | PASS |
| F22 | WebRTC | go2rtc groundwork | `/api/v1/webrtc` capability + playback guidance; go2rtc reused, MJPEG fallback; cleanup via existing recorder lifecycle | `go2rtc.*` | A smoke | API pass | stream E2E BLOCKED w/o HW | report | SOFTWARE PASS / HW BLOCKED |
| F23 | Camera profiles | none | CameraProfileManager: eco/balanced/quality, validated switch+rollback | `camera_profiles` | T switch/rollback; A | pytest+API pass | n/a | report | PASS |
| F24 | Smart recording | manual toggle | SmartRecorder: disabled by default, event triggers, mask hook | `recording.*` | T default-off; A state | pytest+API pass | n/a | report | PASS |
| F25 | Pre/post buffer | none | EventBuffer: bounded ring, pre/post windows, trigger | `event_buffer.*` | T bounded | pytest pass | n/a | report | PASS |
| F26 | Virtual camera | simulation viz only | VirtualCamera: synthetic/file/replay/mock, repeatable | — | T deterministic | pytest pass | n/a (this IS the test double) | report | PASS |
| F27 | Privacy zone editor | API-only masks/zones | PrivacyEditor backend validation (kinds, normalized polygons) + API; dashboard panels | `privacy_editor` | T validation; A | pytest+API pass | visual editor in dashboard panels (data-level) | report | PASS |
| F28 | Camera health | basic health struct | CameraHealth: fps/age/reconnects/failures/queue/latency + `/camera/health` | — | T tick; A | pytest+API pass | n/a | report | PASS |
| F29 | Failover | camera switch only | StreamFailover: hysteresis fail/recover + API | `failover.*` | T hysteresis; A | pytest+API pass | E2E needs 2 sources | report | PASS |
| F30 | Storage manager | none | StorageManager: quotas, retention, pressure cleanup, metrics + API | `storage.*` | T enforce; A | pytest+API pass | n/a | report | PASS |
| F31 | HA discovery | MQTT publish only | hass_discovery(): 3 discovery payloads, technical states, no biometrics + API | `mqtt.*` | T no-biometrics; A | pytest+API pass | broker E2E needs HW env | report | SOFTWARE PASS / E2E BLOCKED |
| F32 | Node-RED | none | NODERED_CONTRACT (mqtt+http topics) + `/nodered/contract` | — | A smoke | API pass | flow E2E external | report | PASS |
| F33 | Rules engine | none | RulesEngine: conditions/filters/windows/debounce/rate/dry-run/validation + storm guards | `rules.*` | T validation/debounce; A crud | pytest+API pass | n/a | report | PASS |
| F34 | Gesture automation | gesture detect only | GestureAutomation: enable gate, approved actions, conf/cooldown, kill-switch | `gesture_actions.*` | T cooldown/kill; A | pytest+API pass | camera E2E needs HW | report | PASS |
| F35 | Webhooks | none | WebhookEngine: HMAC sign, retries cfg, allowlist, SSRF blocks, delivery log | `webhooks.*` | T ssrf; A queue | pytest+API pass | delivery E2E external | report | PASS |
| F36 | MCP expansion | 20 tools | +8 tools (device/camera/models/backends/bench/rules/health/logs), read-only | — | import-tested via suite | pytest pass | n/a | report | PASS |
| F37 | Event replay | none | EventReplay: deterministic ts, dry-run isolation, hardware_actuated=false | — | T never-actuates; A | pytest+API pass | n/a | report | PASS |
| F38 | Schema registry | none | SchemaRegistry: 4 versions, compat check, migrations + API | — | T versions/migrate; A | pytest+API pass | n/a | report | PASS |
| F39 | Notifications | osascript only | LocalNotifier: 5 kinds, bounded payloads, no PII by construction + API | `notifications.*` | T rejects; A | pytest+API pass | n/a | report | PASS |
| F40 | Rule simulator | none | RuleSimulator: synthetic inputs, deterministic, traces + API | — | T dry; A | pytest+API pass | n/a | report | PASS |
| F41 | Web dashboard | legacy single HTML | `web/` React+TS+Vite+Tailwind-ready: 12 sections, responsive, a11y roles, dark mode, loading/error/offline states | vite proxy | W component; build+tsc pass | vite build OK | camera-live needs backend+cam | report | PASS |
| F42 | Live profiler | perf dict | LiveProfiler: fps/infer pxx/drops/system + `/profiler` | — | T snapshot; A | pytest+API pass | n/a | PERF report | PASS |
| F43 | Device mgmt | none | DeviceRegistry: enroll→approve→heartbeat, capabilities/version, no autodiscovery | devices.json | T enroll/approve; A | pytest+API pass | multi-node E2E external | report | PASS |
| F44 | Plugin marketplace | none | PluginRegistry: manifests, compat fields, sha256, enable/disable/rollback, remote needs approval | — | T approval/rollback; A | pytest+API pass | n/a | report | PASS |
| F45 | Config profiles | config.json only | ConfigProfiles: named, scrubbed secrets, import/export, diff, atomic file write | profiles/ | T scrub/diff; A save/load | pytest+API pass | n/a | report | PASS |
| F46 | One-command installer | justfile/manual | `scripts/install.sh`: idempotent, debian/ubuntu/Pi OS, headless, --full/--dry-run/--no-system, no-root path | — | bash -n + doc | syntax OK | Pi-OS run BLOCKED (no Pi) | deploy docs | SOFTWARE PASS / HW BLOCKED |
| F47 | OTA updates | none | UpdateManager: signed stage, version check, backup gate, approve, health rollback, audit + API | — | T gates; A staged-gated | pytest+API pass | artifact E2E external | report | PASS |
| F48 | Backup/restore | none | BackupManager: config/rules/metadata, faces excluded by default, optional Fernet encrypt, verify + API | backups/ | T excludes/verify; A | pytest+API pass | n/a | report | PASS |
| F49 | Diagnostics | none | diagnostics_bundle(): redacted cfg, platform, system, versions; no secrets/images + API | — | T redacted; A | pytest+API pass | n/a | report | PASS |
| F50 | RBAC | token-or-open | RBAC: admin/operator/viewer, 9 permissions, per-action checks, permissive default when disabled | `rbac.*` | T gates; A check | pytest+API pass | n/a | report | PASS |

Overall: software implementation PASS for all 50 on macOS dev evidence;
hardware acceptance BLOCKED where physical devices are required
(F11, F12, F13-Pi fields, F14–F16 movement, F20 GPIO, F22/F31 E2E, F46 Pi run).
OVERALL PRODUCTION READINESS: BLOCKED pending hardware verification.
