# FaceTrak 1.0 Platform Architecture

Modular monolith. One config system (`core/config.py`), one device
state model (`FaceEngine` + discovery), no duplicate camera readers or
inference pipelines. Versioned contracts: REST `/api/v1/*`, event
schemas (`events.v1`, `gpio.v1`, `presence.v1`, `zone.v1`).

```
FaceTrak 1.0
├── Applications: web/ (React+TS+Vite SPA), Tkinter GUI, headless CLI, MCP
├── API Gateway: servers/rest.py (+routers_v1) — auth, RBAC, rate-safe bounds
├── Vision Platform: vision/ — tracking, perception, handover, registry, backends, bench
├── Device Platform: device/ — hailo, imx500, discovery, calibration, predictive,
│                     multiaxis, watchdog, thermal, power, gpio
├── Event Platform: automation/ — schemas, rules, replay, simulator, webhooks,
│                   hass, nodered-contract, gestures, notifications
├── Platform Services: services/ — rbac, profiler, devices, plugins, profiles,
│                     updater, backups, diagnostics
└── Infrastructure: SQLite, go2rtc (protocol), Docker, systemd, metrics
```

## Key decisions (ADRs, summarized)

1. **Hand-written IoU+Kalman tracker** over BoxMOT (AGPL) — license-safe,
   deterministic, testable without weights.
2. **Protocol interop** for HA/Node-RED/go2rtc — no embedded runtimes.
3. **Capability + availability states** on every optional subsystem —
   software-complete without hardware; hardware acceptance separate.
4. **Replay/simulation never actuate** — enforced in code, tested.
5. **Recording OFF by default**; biometrics excluded from MQTT/webhooks/
   backups by default; RBAC-gated sensitive actions.

## Interfaces

- REST: existing endpoints + `/api/v1/*` (see routers_v1.py).
- MCP: existing tools + F36 expansion (device_status, camera_diagnostics,
  model_registry, backend_status, quick_benchmark, rules_list,
  platform_health, logs_metrics) — read-only defaults.
- Config: `config.json` + dot-notation; new sections in DEFAULT_CONFIG.
- Dashboard: `web/` proxies `/api`, `/status`, `/metrics` to :8765.
