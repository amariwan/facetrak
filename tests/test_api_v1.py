"""API v1 integration: router mounts, auth passthrough, key flows."""

from fastapi.testclient import TestClient


def _client():
    from facetrak.servers.rest import app

    return TestClient(app)


def test_v1_endpoints_smoke():
    c = _client()
    for path in [
        "/api/v1/tracking",
        "/api/v1/models",
        "/api/v1/backends",
        "/api/v1/rules",
        "/api/v1/schemas",
        "/api/v1/profiler",
        "/api/v1/devices",
        "/api/v1/plugins",
        "/api/v1/hardware/discovery",
        "/api/v1/hardware/thermal",
        "/api/v1/nodered/contract",
        "/api/v1/diagnostics",
    ]:
        r = c.get(path)
        assert r.status_code == 200, (path, r.status_code, r.text[:200])


def test_v1_rules_crud_and_simulate():
    c = _client()
    r = c.post(
        "/api/v1/rules", json={"name": "t1", "on": "zone_enter", "action": "log"}
    )
    assert r.status_code in (200, 400)  # 400 if re-run duplicate
    r = c.post("/api/v1/rules/simulate", json=[{"kind": "zone_enter"}])
    assert r.status_code == 200 and r.json()["dry_run"] is True
    c.delete("/api/v1/rules/t1")


def test_v1_model_register_activate():
    c = _client()
    body = {
        "model_id": "test-det-1",
        "name": "T",
        "task": "detection",
        "framework": "onnx",
    }
    r = c.post("/api/v1/models", json=body)
    assert r.status_code == 200
    r = c.post("/api/v1/models/test-det-1/activate")
    assert r.status_code == 200 and r.json()["ok"]


def test_v1_privacy_validation():
    c = _client()
    r = c.post("/api/v1/privacy/zones", json={"kind": "exclusion", "polygon": [[0, 0]]})
    assert r.status_code == 400
    r = c.post(
        "/api/v1/privacy/zones",
        json={"kind": "exclusion", "polygon": [[0, 0], [1, 0], [0, 1]]},
    )
    assert r.status_code == 200


def test_v1_webhook_replay_power():
    c = _client()
    r = c.post("/api/v1/events/replay", json=[{"kind": "zone_enter"}])
    assert r.json()["hardware_actuated"] is False
    r = c.post("/api/v1/power/eco")
    assert r.json()["ok"]
    r = c.get("/api/v1/power")
    assert r.json()["profile"] == "eco"
    c.post("/api/v1/power/off")


def test_v1_updates_gated():
    c = _client()
    r = c.post(
        "/api/v1/updates/stage",
        json={"version": "0.1", "sha256": "x", "signature_ok": False},
    )
    assert r.json()["ok"] is False
