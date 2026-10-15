"""Security helpers: token auth, URL redaction, config secrets."""

import os

from facetrak.core import config as cfgmod
from facetrak.core import security


def test_verify_token_env_wins(monkeypatch):
    monkeypatch.setenv("FACETRAK_API_TOKEN", "env-token")
    cfg = {"api": {"auth_token": "cfg-token"}}
    assert security.get_api_token(cfg) == "env-token"
    assert security.verify_token(cfg, "env-token")
    assert not security.verify_token(cfg, "cfg-token")
    assert not security.verify_token(cfg, None)


def test_verify_token_from_config(monkeypatch):
    monkeypatch.delenv("FACETRAK_API_TOKEN", raising=False)
    cfg = {"api": {"auth_token": "s3cret"}}
    assert security.verify_token(cfg, "s3cret")
    assert not security.verify_token(cfg, "wrong")
    # timing-safe comparison must not raise on different lengths
    assert not security.verify_token(cfg, "x" * 100)


def test_no_token_allows_localhost(monkeypatch):
    monkeypatch.delenv("FACETRAK_API_TOKEN", raising=False)
    assert security.verify_token({"api": {"auth_token": ""}}, None)


def test_redact_url_strips_credentials():
    assert (
        security.redact_url("rtsp://admin:p%40ss@192.168.1.10:554/stream")
        == "rtsp://192.168.1.10:554/stream"
    )
    plain = "rtsp://192.168.1.10:554/stream"
    assert security.redact_url(plain) == plain


def test_config_redact_masks_secrets():
    cfg = {
        "cameras": [
            {"name": "door", "source": "rtsp://u:p@host/s"},
            {"name": "onvif", "source": 0, "password": "pw"},
        ],
        "mqtt": {"password": "mqpw"},
    }
    out = cfgmod.redact(cfg)
    assert out["cameras"][0]["source"] == "rtsp://host/s"
    assert out["cameras"][1]["password"] == "***"
    assert out["mqtt"]["password"] == "***"
    # original untouched
    assert cfg["cameras"][0]["source"] == "rtsp://u:p@host/s"


def test_config_save_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "config.json"
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", target)
    cfg = dict(cfgmod.DEFAULT_CONFIG)
    cfg["detect_width"] = 320
    cfgmod.save(cfg)
    assert target.exists()
    assert cfgmod.load()["detect_width"] == 320
    leftovers = list(tmp_path.glob(".config.*.tmp"))
    assert leftovers == []


def test_config_source_none_guard():
    cfg = {"cameras": [{"name": "x", "source": None}]}
    assert cfgmod.source(cfg, 0) == 0
    cfg2 = {"cameras": [{"name": "y"}]}
    assert cfgmod.source(cfg2, 0) == 0
    assert cfgmod.source({"cameras": []}, 3) == 3


def test_env_confirm_flag_name():
    assert security.ENV_MCP_CONFIRM == "FACETRAK_MCP_REQUIRE_CONFIRM"
    assert security.ENV_TOKEN == "FACETRAK_API_TOKEN"
    assert os.environ.get("FACETRAK_MCP_REQUIRE_CONFIRM", "") in ("", "1")
