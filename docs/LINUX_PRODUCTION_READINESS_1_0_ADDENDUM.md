# Linux Production Readiness — FaceTrak 1.0 Addendum

Companion to `docs/LINUX_PRODUCTION_READINESS.md` (prior phases).
This addendum covers only the 1.0 delivery delta.

## Verified in this delivery (evidence in PERFORMANCE_REPORT.md)

- Full suite: 171 passed (macOS dev + mocked HW), ~11.5 s.
- Docker linux/arm64 (local, 2026-10-15): image `facetrak:local-arm64`
  built OK; headless `--check` OK; all 1.0 packages + v1 router import
  OK; 62 passed (`test_facetrak1` + `test_platform` + `test_headless`)
  inside the container.
- New-module ruff: clean. Repo baseline: 120 pre-existing (CI non-blocking).
- API v1: 6 integration tests + 12-endpoint smoke (TestClient).
- Dashboard: vitest 1 passed, `tsc --noEmit` clean, `vite build` OK.
- Headless check OK; benchmark smoke 181.5 fps (dev sim pipeline).
- Installer `scripts/install.sh`: `bash -n` syntax OK.

## Platform matrix delta

| Platform | Evidence | Status |
|---|---|---|
| macOS dev | unit/integration/sim | PASS |
| Linux AMD64/ARM64 | CI jobs (test, docker, systemd, frontend, api-smoke) | ARM64 verified locally (image + 62 tests in container); AMD64 + CI PENDING (run on push) |
| Raspberry Pi 4/5, IMX500, Hailo | adapters fallback-tested only | SOFTWARE PASS / HW BLOCKED |

## Pre-production checklist (real hardware)

1. Run `scripts/install.sh` on Pi OS/Ubuntu/Debian; record output.
2. Hardware acceptance per FEATURE_ACCEPTANCE_MATRIX rows
   F11–F16, F20, F22 (+ F31 broker E2E, F43 multi-node, F46 Pi run).
3. Serve `web/dist` behind the API host; set `FACETRAK_API_TOKEN`;
   enable RBAC users; review mqtt/webhook allowlists, quotas, retention.
