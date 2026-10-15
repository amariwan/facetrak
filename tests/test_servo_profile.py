"""Servo motion profiles: limits, dead-zone, reacquisition, calibration."""

from facetrak.hardware.servo import PanTiltController


def _servo(**kw):
    cfg = {
        "servo": {
            "port": "",
            "baud": 115200,
            "pan_min": 0,
            "pan_max": 180,
            "tilt_min": 0,
            "tilt_max": 180,
            "dead_zone": 15,
            "smooth": 0.5,
            "max_step": 10.0,
            "max_speed_dps": 90.0,
            "reacquire_frames": 5,
            "pan_offset": 0.0,
            "tilt_offset": 0.0,
            "invert_pan": False,
            "invert_tilt": False,
        }
    }
    cfg["servo"].update(kw)
    s = PanTiltController(cfg)
    s.enabled = True
    return s


def test_no_hardware_needed():
    s = _servo()
    assert not s.connected
    pan, tilt = s.update(100, 50, 640, 480)  # must not raise
    assert 0 <= pan <= 180 and 0 <= tilt <= 180


def test_dead_zone_holds_position():
    s = _servo()
    before = (s.pan, s.tilt)
    assert s.update(5, 5, 640, 480) == before


def test_angle_limits_clamped():
    s = _servo()
    pan, _ = s.update(100000, 0, 640, 480)
    assert pan <= 180.0
    pan, tilt = s.move_to(-50, 999)
    assert (pan, tilt) == (0.0, 180.0)


def test_velocity_cap_per_frame():
    s = _servo(max_speed_dps=1.0, max_step=100.0, smooth=1.0)
    pan0 = s.pan
    pan1, _ = s.update(300, 0, 640, 480)
    assert abs(pan1 - pan0) <= 1.0 + 1e-6  # dps cap dominates


def test_reacquire_returns_to_center():
    s = _servo(reacquire_frames=3)
    s.pan, s.tilt = 140.0, 140.0
    assert s.note_target_lost() is None
    assert s.note_target_lost() is None
    moved = s.note_target_lost()
    assert moved is not None
    assert moved[0] < 140.0  # easing toward 90


def test_target_resets_lost_counter():
    s = _servo(reacquire_frames=3)
    s.note_target_lost()
    s.update(100, 0, 640, 480)
    assert s._lost_frames == 0


def test_calibrate_clamped():
    s = _servo()
    out = s.calibrate(3.0, -50.0)
    assert out == {"pan_offset": 3.0, "tilt_offset": -10.0}
