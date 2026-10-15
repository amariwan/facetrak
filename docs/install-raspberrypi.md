# FaceTrak on Raspberry Pi OS (primary production target)

Tested logic on macOS with mocks; Pi hardware runs are BLOCKED until
executed on device (see `docs/LINUX_PRODUCTION_READINESS.md`).

## 1. OS + system packages (Pi 4/5, 64-bit Raspberry Pi OS)

```bash
sudo apt update && sudo apt install -y \
  python3 python3-venv python3-pip v4l-utils \
  libgl1 libglib2.0-0 libgomp1 libcamera-apps
sudo usermod -aG video,dialout,gpio $USER   # re-login afterwards
```

Camera stack: USB cameras work via V4L2 out of the box. For the Pi
Camera Module, enable it in `raspi-config` → Interface Options and use
a `{"type": "pi", ...}` entry in `cameras[]` (needs `picamera2` +
`libcamera`, Pi only).

## 2. Install (headless, no desktop required)

```bash
sudo mkdir -p /opt/facetrak /var/lib/facetrak /etc/facetrak
sudo chown $USER /opt/facetrak /var/lib/facetrak
git clone <repo> /opt/facetrak && cd /opt/facetrak
python3 -m venv .venv && .venv/bin/pip install --upgrade pip
# Pi wheels first for numpy etc. (falls back to PyPI automatically):
.venv/bin/pip install --extra-index-url https://www.piwheels.org/simple -e .
.venv/bin/python -m pytest tests/ -q   # mocked suite, runs on the Pi too
```

ARM64 wheel status (verified via PyPI metadata 2026-10-15):
`opencv-contrib-python` 4.13.0.92 and `numpy` ship
`manylinux_aarch64` wheels. **`mediapipe` 0.10.35 ships NO
manylinux_aarch64 wheel** — FaceTrak detects this at import and
auto-disables face analysis (emotion/gaze/pose/gestures). Detection,
tracking, recognition, zones, servo and MQTT run fully without it.

## 3. Configure

```bash
cp config.example.json /var/lib/facetrak/config.json  # or start once to generate
# edit: cameras[], detect_width: 320, detect_adaptive: true,
#       retention_days, mqtt, api host/token
openssl rand -hex 32 | sudo tee /etc/facetrak/env  # as FACETRAK_API_TOKEN=...
sudo chmod 600 /etc/facetrak/env && sudo chown root:facetrak /etc/facetrak/env
```

Pi performance defaults: `detect_width` 320, `detect_adaptive` true,
`detect_target_ms` 50–66, disable `depth/pose/gestures`, set
`resources.max_memory_mb` (e.g. 1200 on 4 GB models).

## 4. systemd (autostart, restart, logs)

```bash
sudo cp deploy/facetrak.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now facetrak
journalctl -u facetrak -f            # logs (journald, persisted)
systemctl status facetrak            # restart counter, uptime
```

Updates/rollback: deploy new tree to `/opt/facetrak.new`, run
`headless --check` + pytest there, flip a symlink, `systemctl restart`;
rollback = symlink back + restart. Details + logrotate in the
readiness report.

## 5. Log rotation + updates

journald persists service logs; bound the size once:

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nSystemMaxUse=300M\n' | \
  sudo tee /etc/systemd/journald.conf.d/facetrak.conf
sudo systemctl restart systemd-journald
```

Update procedure (rollback-capable):

```bash
git clone <repo> /opt/facetrak.new && cd /opt/facetrak.new
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m facetrak.servers.headless --check
cp /var/lib/facetrak/facetrak.db /var/lib/facetrak/facetrak.db.bak
sudo ln -sfn /opt/facetrak.new /opt/facetrak && sudo systemctl restart facetrak
# rollback: sudo ln -sfn /opt/facetrak.old /opt/facetrak && sudo systemctl restart facetrak
```
