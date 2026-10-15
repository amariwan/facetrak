"""Static deployment checks: Dockerfile, Compose, systemd (no daemon needed)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text()


def test_dockerfile_hardened():
    df = _read("Dockerfile")
    assert "USER facetrak" in df
    assert "HEALTHCHECK" in df and "/health" in df
    assert "privileged" not in df.lower()
    assert ":latest" not in df
    assert "headless" in df  # GUI-free entrypoint
    assert "v4l-utils" in df or "libgl1" in df


def test_compose_safe_defaults():
    compose = _read("docker-compose.yml")
    assert "privileged: true" not in compose
    assert "no-new-privileges" in compose
    assert "127.0.0.1:8765" in compose  # not LAN-exposed by default
    assert "facetrak-data:/data" in compose
    assert ".env" in compose  # token via env file, never baked in
    assert "unless-stopped" in compose


def test_systemd_unit_sane():
    unit = _read("deploy/facetrak.service")
    assert "Type=simple" in unit
    assert "Restart=on-failure" in unit
    assert "KillSignal=SIGTERM" in unit
    assert "NoNewPrivileges=true" in unit
    assert "EnvironmentFile=-/etc/facetrak/env" in unit
    assert "headless" in unit
    assert "PrivateTmp=true" in unit
