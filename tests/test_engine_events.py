"""Engine event fan-out, zone wiring, perf counters (mock camera)."""

from types import SimpleNamespace

import numpy as np
import pytest

from facetrak.core import config as cfgmod
from facetrak.core.engine import FaceEngine
from facetrak.storage import db as dbmod


class MockCam:
    def __init__(self, n=5):
        self.frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(n)]
        self.i = 0
        from facetrak.camera.base import CameraHealth

        self.health = CameraHealth()

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        f = self.frames[self.i]
        self.i += 1
        self.health.ok()
        return True, f

    def release(self):
        pass

    @property
    def resolution(self):
        return 320, 240


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "t.db")
    e = FaceEngine()
    e.cfg["zones"] = [{"name": "left", "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}]
    e.zones = __import__(
        "facetrak.events", fromlist=["ZoneManager"]
    ).ZoneManager.from_config(e.cfg)
    e.cam = MockCam()
    e.running = True
    e.detector = None  # no model: reuse empty detections
    e.tracker.update([])  # warm
    return e


def _track(tid, cx, cy, name=None):
    return SimpleNamespace(
        track_id=tid, name=name, det=SimpleNamespace(center=(cx, cy))
    )


def test_process_zones_emits_and_persists(engine):
    seen = []
    engine.on_event(seen.append)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    engine._process_zones([_track(1, 40, 120, "amy")], frame)
    assert len(seen) == 1 and seen[0]["kind"] == "zone_enter"
    stored = dbmod.query_events()
    assert stored and stored[0]["zone"] == "left"
    # exit on leaving
    engine._process_zones([_track(1, 300, 120, "amy")], frame)
    assert seen[-1]["kind"] == "zone_exit"


def test_step_counts_drops_and_frames(engine):
    for _ in range(5):
        engine.step()
    assert engine.perf["frames"] == 5
    assert engine.step() is None  # camera exhausted
    assert engine.perf["drops"] == 1
    stats = engine.perf_stats()
    assert stats["frames"] == 5 and stats["drops"] == 1
    assert stats["detector_ready"] is False


def test_recent_events_ring_capped(engine):
    engine.cfg["events"] = {"max_events": 3}
    for i in range(5):
        engine._emit_event({"kind": "zone_enter", "track_id": i})
    assert len(engine.recent_events()) == 3
    assert engine.recent_events(limit=1)[0]["track_id"] == 4


def test_purge_and_erase(engine, tmp_path):
    dbmod.log_presence("appeared", "amy", 1)
    assert engine.purge_retention()["presence"] == 0
    out = engine.delete_runtime_data()
    assert out["presence"] == 1
    (tmp_path / "faces").mkdir(exist_ok=True)
