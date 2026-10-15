"""Camera health tracking and storage retention/erasure."""

import numpy as np

from facetrak.camera.base import CameraHealth
from facetrak.camera.rtsp import RTSPCamera
from facetrak.camera.usb import USBCamera
from facetrak.storage import db as dbmod


def test_health_transitions():
    h = CameraHealth()
    assert not h.connected
    h.ok()
    assert h.connected and h.frames == 1
    for _ in range(4):
        h.fail("x")
    assert h.connected  # tolerates bursts < 5
    h.fail("x")
    assert not h.connected
    h.reconnected()
    assert h.connected and h.reconnects == 1
    d = h.to_dict()
    assert d["frames"] == 1 and d["failures"] == 5
    assert d["last_frame_age_s"] is not None


def test_rtsp_empty_queue_is_not_a_crash():
    import queue

    cam = RTSPCamera.__new__(RTSPCamera)  # no thread started
    cam._queue = queue.Queue(maxsize=2)
    ok, frame = RTSPCamera.read(cam)
    assert (ok, frame) == (False, None)
    assert cam.health.to_dict()["frames"] == 0


def test_usb_read_failure_tracked(monkeypatch):
    cam = USBCamera.__new__(USBCamera)
    cam._cap = None

    class DeadCap:
        def read(self):
            return False, None

    cam._cap = DeadCap()
    ok, frame = cam.read()
    assert (ok, frame) == (False, None)
    assert cam.health.to_dict()["failures"] == 1


def test_usb_read_success_tracked():
    cam = USBCamera.__new__(USBCamera)

    class GoodCap:
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    cam._cap = GoodCap()
    ok, frame = cam.read()
    assert ok and frame is not None
    assert cam.health.to_dict()["frames"] == 1


def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    dbmod.init()


def test_event_log_and_query(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    dbmod.log_event("zone_enter", zone="door", track_id=1, name="amy")
    rows = dbmod.query_events()
    assert len(rows) == 1
    assert rows[0]["kind"] == "zone_enter" and rows[0]["zone"] == "door"
    assert dbmod.query_events(kind="nope") == []


def test_purge_and_delete_all(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)
    dbmod.log_presence("appeared", "amy", 1)
    dbmod.log_event("zone_enter", zone="door")
    assert dbmod.purge_older_than(30) == {
        "presence": 0,
        "emotion_log": 0,
        "crowd_samples": 0,
        "events": 0,
    }
    assert dbmod.purge_older_than(0)["presence"] == 0  # 0 = keep
    out = dbmod.delete_all()
    assert out["presence"] == 1 and out["events"] == 1
    assert dbmod.query_presence() == [] and dbmod.query_events() == []
