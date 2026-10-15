# Performance Report — FaceTrak 1.0 (baseline)

Date: 2026-10-15. Device: MacBook Pro (Darwin arm64, dev-only baseline —
NOT a production claim). Linux/Pi numbers require hardware runs.

## Measured

- `scripts/benchmark.py --frames 50`: 50 frames / 0.28 s = **181.5 fps**,
  step_ms mean 5.51 / p50 5.28 / p95 5.64, detect_ms(last) 4.18,
  drops 0 (simulation pipeline, no camera).
- BenchmarkLab `cpu-matmul-128` (hardware mode, this device):
  avg/p50/p95/p99 = 0.01 ms, iters 20. psutil absent in venv → mem/cpu
  reported 0.0 (honest zero, not measured).
- Full pytest suite: 171 passed in ~11.5 s.
- API v1 smoke: 12 endpoints 200 OK (TestClient, no network).
- Dashboard: vite build 194 ms, 145.6 kB JS (47 kB gzip).

## Configurations compared

Minimal CPU (eco profile: detect_width 320, every 3) vs balanced vs
quality — profile switching verified functionally (T); comparative
FPS measurements on target hardware are PENDING (need Pi/Linux runs).

## Regressions / notes

- No baseline regression: pre-existing suite still passes.
- No acceleration claimed: Hailo/IMX500 adapters report fallback;
  comparative accelerated numbers require hardware.
- Thermal: no sensor on this host → governor reports `unknown` with
  visible fault string (not hidden).
