"""Detection zones, privacy masks, enter/exit events."""

import numpy as np
import pytest

from facetrak.events import PrivacyMask, Zone, ZoneManager, load_masks, valid_polygon


def test_valid_polygon():
    assert valid_polygon([(0, 0), (1, 0), (1, 1)])
    assert not valid_polygon([(0, 0), (1, 0)])  # too few
    assert not valid_polygon([(0, 0), (2, 0), (1, 1)])  # out of range
    assert not valid_polygon("nope")
    assert not valid_polygon([(0, 0), ("a", 0), (1, 1)])


def test_zone_contains():
    z = Zone("door", [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)])
    assert z.contains(0.25, 0.25)
    assert not z.contains(0.75, 0.75)


def test_zone_requires_name_and_polygon():
    with pytest.raises(ValueError):
        Zone("", [(0, 0), (1, 0), (1, 1)])
    with pytest.raises(ValueError):
        Zone("x", [(0, 0), (1, 0)])


def test_enter_exit_events():
    zm = ZoneManager([Zone("gate", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)])])
    evs = zm.update(7, 0.8, 0.5, name="amy")
    assert evs == []  # outside
    evs = zm.update(7, 0.2, 0.5, name="amy")
    assert len(evs) == 1 and evs[0]["kind"] == "zone_enter"
    assert evs[0]["zone"] == "gate" and evs[0]["track_id"] == 7
    evs = zm.update(7, 0.2, 0.5)  # still inside: no repeat
    assert evs == []
    evs = zm.update(7, 0.9, 0.5)
    assert len(evs) == 1 and evs[0]["kind"] == "zone_exit"
    # per-track isolation
    assert zm.update(9, 0.2, 0.5)[0]["kind"] == "zone_enter"


def test_disabled_zone_ignored():
    zm = ZoneManager([Zone("off", [(0, 0), (1, 0), (1, 1)], enabled=False)])
    assert zm.update(1, 0.3, 0.2) == []


def test_from_config_skips_invalid(caplog):
    cfg = {
        "zones": [
            {"name": "good", "polygon": [[0, 0], [1, 0], [1, 1]]},
            {"name": "bad", "polygon": [[0, 0], [1, 0]]},
            {"name": "", "polygon": [[0, 0], [1, 0], [1, 1]]},
            {"nope": True},
        ]
    }
    zm = ZoneManager.from_config(cfg)
    assert [z.name for z in zm.zones] == ["good"]
    assert zm.to_config()[0]["enabled"] is True


def test_privacy_mask_blacks_region():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    m = PrivacyMask([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)], "corner")
    out = m.apply(frame.copy())
    assert out[10, 10].tolist() == [0, 0, 0]
    assert out[90, 90].tolist() == [255, 255, 255]


def test_load_masks_skips_invalid():
    cfg = {
        "privacy_masks": [
            {"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]]},
            {"name": "b", "polygon": [[0, 0]]},
        ]
    }
    assert [m.name for m in load_masks(cfg)] == ["a"]
