"""Headless runner, data_dir routing, adaptive detection (mocked)."""

from pathlib import Path

import numpy as np

from facetrak.core import config as cfgmod
from facetrak.core.paths import data_dir
from facetrak.servers import headless
from facetrak.storage import db as dbmod


def test_headless_check_boot_without_camera(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    with caplog.at_level("INFO"):
        assert headless.run(check_only=True) == 0
    assert "check ok" in caplog.text


def test_data_dir_env_override(tmp_path, monkeypatch):
    target = tmp_path / "vol"
    monkeypatch.setenv("FACETRAK_DATA", str(target))
    assert data_dir({}) == target
    assert target.exists()


def test_data_dir_config_and_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("FACETRAK_DATA", raising=False)
    cfg = {"data_dir": str(tmp_path / "cfgvol")}
    assert data_dir(cfg) == tmp_path / "cfgvol"
    assert data_dir({}) == Path(".")


def test_snapshot_routes_to_data_dir(tmp_path, monkeypatch):
    from facetrak.core.engine import FaceEngine

    monkeypatch.setattr(cfgmod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    monkeypatch.delenv("FACETRAK_DATA", raising=False)  # env wins over cfg
    e = FaceEngine()
    e.cfg["data_dir"] = str(tmp_path / "shots")
    e._frame = np.zeros((8, 8, 3), dtype=np.uint8)
    path = e.snapshot()
    assert path is not None
    assert Path(path).parent == tmp_path / "shots"


def test_adaptive_detection_ramps_under_load(tmp_path, monkeypatch):
    from facetrak.core.engine import FaceEngine

    monkeypatch.setattr(cfgmod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    e = FaceEngine()
    assert e._detect_every == 1
    e._detect_adaptive = True
    for _ in range(31):  # pass the 30-frame adaptation gate
        e._adapt_detection(200.0)
    assert e._detect_every > 1
    for _ in range(60):
        e._adapt_detection(1.0)
    assert e._detect_every == 1  # recovers when fast


def test_adaptive_detection_disabled_stays_put(tmp_path, monkeypatch):
    from facetrak.core.engine import FaceEngine

    monkeypatch.setattr(cfgmod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    e = FaceEngine()
    e._detect_adaptive = False
    for _ in range(60):
        e._adapt_detection(500.0)
    assert e._detect_every == 1


def test_system_endpoint_shape(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from facetrak.servers import rest

    monkeypatch.setattr(cfgmod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    monkeypatch.delenv("FACETRAK_API_TOKEN", raising=False)
    from facetrak.core.engine import FaceEngine

    rest._engine = FaceEngine()
    try:
        with TestClient(rest.app) as c:
            body = c.get("/system").json()
        assert "platform" in body and "system" in body
        assert "resources" in body and "go2rtc" in body
        assert body["platform"]["machine"]
    finally:
        rest._engine = None
