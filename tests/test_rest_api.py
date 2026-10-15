"""REST API: auth, envelopes, zones, masks, retention, redaction."""

import pytest
from fastapi.testclient import TestClient

from facetrak.core import config as cfgmod
from facetrak.core.engine import FaceEngine
from facetrak.servers import rest
from facetrak.storage import db as dbmod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    monkeypatch.delenv("FACETRAK_API_TOKEN", raising=False)
    e = FaceEngine()
    rest._engine = e
    rest._auth_warned = True  # silence first-run warning in tests
    with TestClient(rest.app) as c:
        yield c
    rest._engine = None


def _auth(client, token="s3cret"):
    client.headers.update({"Authorization": f"Bearer {token}"})


def test_health_needs_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_status_open_without_token(client):
    assert client.get("/status").status_code == 200


def test_auth_enforced_when_token_set(client):
    rest._engine.cfg["api"] = {"host": "127.0.0.1", "auth_token": "s3cret"}
    r = client.get("/status")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"
    client.get("/status", headers={"Authorization": "Bearer wrong"})
    assert client.get("/status").status_code == 401
    _auth(client)
    assert client.get("/status").status_code == 200


def test_error_envelope_shape(client):
    r = client.get("/faces")  # camera not running
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "unavailable"


def test_zone_crud_validation(client):
    good = {"name": "door", "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}
    assert client.post("/zones", json=good).status_code == 200
    assert client.post("/zones", json=good).status_code == 400  # dup
    bad = {"name": "x", "polygon": [[0, 0], [1, 1]]}
    assert client.post("/zones", json=bad).status_code == 400
    assert client.get("/zones").json()[0]["name"] == "door"
    assert client.delete("/zones/door").status_code == 200
    assert client.delete("/zones/door").status_code == 404


def test_mask_crud(client):
    m = {"name": "win", "polygon": [[0, 0], [1, 0], [1, 1]]}
    assert client.post("/masks", json=m).status_code == 200
    assert len(client.get("/masks").json()) == 1
    assert client.delete("/masks/0").status_code == 200
    assert client.delete("/masks/0").status_code == 404


def test_events_and_retention(client):
    dbmod.log_event("zone_enter", zone="door", track_id=1)
    ev = client.get("/events").json()
    assert ev["stored"] and ev["stored"][0]["zone"] == "door"
    assert client.get("/events?kind=zone_exit").json()["stored"] == []
    r = client.post("/retention/purge").json()
    assert r["ok"] and r["deleted"]["events"] == 0


def test_delete_data_needs_confirm(client):
    assert client.delete("/data").status_code == 400
    assert client.delete("/data?confirm=true").status_code == 200
    # face erasure gated by consent flag
    r = client.delete("/data?confirm=true&include_faces=true")
    assert r.status_code == 403


def test_cameras_redacted(client):
    rest._engine.cfg["cameras"] = [{"name": "d", "source": "rtsp://u:p@host/s"}]
    cams = client.get("/cameras").json()
    assert cams[0]["source"] == "rtsp://host/s"


def test_hide_estimates(client):
    rest._engine.cfg["privacy"] = {
        "hide_estimates": True,
        "allow_identifying_exports": False,
    }
    assert client.get("/face/analysis").status_code == 403
    assert client.get("/emotions").status_code == 403


def test_blur_name_validated(client):
    assert client.post("/blur/" + "n" * 200).status_code == 400
    assert client.post("/blur/amy", json={"blur": True}).status_code == 200
