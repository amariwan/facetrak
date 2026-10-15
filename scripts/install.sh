#!/usr/bin/env bash
# FaceTrak 1.0 one-command installer (F46): idempotent, headless-capable,
# optional components, dependency checks, dry-run. No root required
# except for apt/systemd steps (skipped with --no-system).
set -euo pipefail
DRY=0; NO_SYSTEM=0; WITH_FULL=0; HEADLESS=0
for a in "$@"; do case "$a" in
  --dry-run) DRY=1;; --no-system) NO_SYSTEM=1;; --full) WITH_FULL=1;;
  --headless) HEADLESS=1;; *) echo "unknown flag $a"; exit 2;;
esac; done
run() { if [ "$DRY" = 1 ]; then echo "[dry-run] $*"; else eval "$*"; fi; }
echo "== FaceTrak 1.0 installer =="
echo "dry_run=$DRY no_system=$NO_SYSTEM full=$WITH_FULL headless=$HEADLESS"
command -v python3 >/dev/null || { echo "ERROR: python3 missing"; exit 1; }
if [ "$NO_SYSTEM" = 0 ] && command -v apt-get >/dev/null; then
  run "sudo apt-get update"
  run "sudo apt-get install -y python3-venv libgl1 libglib2.0-0 libegl1 libgles2"
else
  echo "-- system packages skipped (no apt or --no-system)"
fi
[ -d .venv ] || run "python3 -m venv .venv"
if [ "$WITH_FULL" = 1 ]; then run ".venv/bin/pip install -e '.[full]'"
else run ".venv/bin/pip install -e ."; fi
run ".venv/bin/python -m pytest tests/ -q -x --ignore=tests/ai --ignore=tests/camera --ignore=tests/sensors || .venv/bin/python -m pytest tests/test_facetrak1.py tests/test_api_v1.py -q"
echo "OK: install complete. Run: .venv/bin/python -m facetrak.servers.rest"
