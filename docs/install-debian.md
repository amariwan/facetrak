# FaceTrak on Ubuntu / Debian (x86_64 + ARM64)

## Bare metal

```bash
sudo apt update && sudo apt install -y \
  python3 python3-venv python3-pip v4l-utils libgl1 libglib2.0-0 libgomp1
git clone <repo> /opt/facetrak && cd /opt/facetrak
python3 -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install -e . && .venv/bin/pip install pytest httpx
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m facetrak.servers.headless --check
```

Then configure `/var/lib/facetrak/config.json` + token and install
`deploy/facetrak.service` exactly as in `docs/install-raspberrypi.md`
§3–4 (same unit file works on Ubuntu/Debian).

MediaPipe ships `manylinux_x86_64` wheels, so on x86_64 servers all
analysis features work. On ARM64 servers the Pi limitations apply
(no MediaPipe wheel → auto-degraded; see Pi doc §2).

## Docker (recommended for servers)

```bash
cp .env.example .env 2>/dev/null; echo "FACETRAK_API_TOKEN=$(openssl rand -hex 32)" >> .env
just docker-up                        # builds + starts, port 127.0.0.1:8765
curl http://127.0.0.1:8765/health
docker compose --profile mqtt up -d   # optional local broker
```

USB camera passthrough: uncomment the `devices:` lines in
`docker-compose.yml` (host `/dev/video0` etc.). The Pi Camera Module
needs libcamera in the container and is **not** passed through by
default — run bare metal for Pi Camera. Never use `--privileged`;
device nodes are allow-listed explicitly.

Multi-arch: `docker buildx build --platform linux/amd64,linux/arm64`.
Note: `mediapipe` inside an ARM64 image installs nothing (no wheel) —
the container still boots; analysis features stay disabled by design.
