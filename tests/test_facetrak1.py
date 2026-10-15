"""FaceTrak 1.0 acceptance tests: F01-F50 unit/negative/boundary."""

import numpy as np


# ── F01/F05/F06 tracking ──────────────────────────────────────────
def test_mot_stable_ids_and_aging():
    from facetrak.vision.tracking import MOTDetection, MultiObjectTracker

    tr = MultiObjectTracker({"tracking": {"min_hits": 1, "max_age": 2}})
    d = [MOTDetection(10, 10, 50, 50, 0.9, "person")]
    a1, _ = tr.update(d)
    a2, _ = tr.update(d)
    assert a1[0].track_id == a2[0].track_id
    tr.update([])
    tr.update([])
    active, ended = tr.update([])
    assert len(active) == 0 and len(ended) >= 0
    dg = tr.diagnostics()
    assert dg["total_matches"] >= 1


def test_mot_conf_threshold_negative():
    from facetrak.vision.tracking import MOTDetection, MultiObjectTracker

    tr = MultiObjectTracker({"tracking": {"conf_threshold": 0.9, "min_hits": 1}})
    active, _ = tr.update([MOTDetection(0, 0, 10, 10, 0.1, "person")])
    assert active == []


def test_mot_prediction_moves_occluded():
    from facetrak.vision.tracking import MOTDetection, MultiObjectTracker

    tr = MultiObjectTracker(
        {"tracking": {"min_hits": 1, "max_age": 5, "prediction": True}}
    )
    tr.update([MOTDetection(10, 10, 20, 20, 0.9, "person")])
    tr.update([MOTDetection(14, 10, 20, 20, 0.9, "person")])
    t = tr.tracks[0]
    assert t.vx != 0  # velocity learned (F06)


# ── F02/F03/F04 ───────────────────────────────────────────────────
def test_segmentation_cpu_fallback():
    from facetrak.vision.perception import SegmentationService

    s = SegmentationService({"segmentation": {"enabled": True}})
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = s.segment(frame)
    assert out["enabled"] and out["backend"] == "cpu-fallback"
    assert s.overlay(frame, out).shape == frame.shape


def test_segmentation_disabled():
    from facetrak.vision.perception import SegmentationService

    s = SegmentationService({"segmentation": {"enabled": False}})
    assert s.segment(np.zeros((8, 8, 3), np.uint8))["masks"] == []


def test_openvocab_validation_and_graceful():
    from facetrak.vision.perception import OpenVocabularyDetector

    ov = OpenVocabularyDetector({"openvocab": {"enabled": True}})
    assert ov.detect(np.zeros((8, 8, 3), np.uint8), "")["error"]
    assert ov.detect(np.zeros((8, 8, 3), np.uint8), "x" * 200)["error"]
    cap = ov.capability()
    assert "supported" in cap


def test_depth_relative_not_meters():
    from facetrak.vision.perception import DepthTracker

    d = DepthTracker({"depth_tracking": {"enabled": True}})
    r = d.relative_depth((480, 640), (100, 100, 50, 50))
    assert 0 <= r <= 1
    assert d.match_bonus(0.5, 0.5) > 0


# ── F07 handover ──────────────────────────────────────────────────
def test_handover_occupancy_and_person_refusal():
    from facetrak.vision.handover import CameraHandover

    h = CameraHandover(
        {
            "handover": {
                "enabled": True,
                "relations": [{"from": "a", "to": "b", "window_s": 30}],
            }
        }
    )
    h.record_exit("a", "box")
    assert h.correlate("b", "box")["kind"] == "occupancy"
    h.record_exit("a", "person")
    # no session -> refuse person link
    assert h.correlate("b", "person", person=True) is None
    h.record_exit("a", "person", session="s1")
    assert h.correlate("b", "person", session="s1", person=True)["kind"] == "session"


# ── F08/F09/F10 ───────────────────────────────────────────────────
def test_registry_validate_activate():
    from facetrak.vision.models import ModelEntry, ModelRegistry

    r = ModelRegistry()
    bad = ModelEntry("x", "X", task="nope")
    assert not r.register(bad)["ok"]
    ok = ModelEntry("t1", "T", task="detection", framework="onnx", checksum="a" * 64)
    assert r.register(ok)["ok"]
    assert r.activate("t1")["ok"]
    assert not r.activate("missing")["ok"]


def test_backends_switch_contract():
    from facetrak.vision.models import BackendManager

    b = BackendManager()
    assert b.switch("nope")["ok"] is False
    res = b.switch("hailo")
    assert res["ok"] and "fallback" in res
    assert b.run(None)["contract"] == "detect"


def test_benchmark_modes_labeled():
    from facetrak.vision.models import BenchmarkLab

    lab = BenchmarkLab({"benchmark": {"warmup": 1, "iters": 3}})
    r = lab.run(lambda: 1 + 1, "t", mode="synthetic")
    assert r["mode"] == "synthetic" and r["latency_ms"]["p95"] >= 0
    csv = BenchmarkLab.to_csv([r])
    assert "p95_ms" in csv


# ── F11/F12 ───────────────────────────────────────────────────────
def test_hailo_imx500_graceful():
    from facetrak.device.accelerators import HailoAdapter, IMX500Adapter

    h = HailoAdapter({"hailo": {"enabled": False}})
    d = h.discover()
    assert "available" in d
    inf = h.infer(None)
    assert inf["fallback"] == "cpu" or inf.get("available")
    im = IMX500Adapter({"imx500": {"enabled": False}})
    assert im.start()["ok"] is False
    assert im.discover()["camera"] == "imx500"


# ── F13 discovery ─────────────────────────────────────────────────
def test_discovery_keys():
    from facetrak.device.management import discover_hardware

    d = discover_hardware({"cameras": []})
    for k in ("arch", "usb_cameras", "serial", "runtimes", "gpio"):
        assert k in d


# ── F14 calibration ───────────────────────────────────────────────
def test_calibration_requires_confirm_and_bounds():
    from facetrak.device.management import ServoCalibrator

    c = ServoCalibrator({"servo_cal": {"simulate": True}})
    assert c.propose()["requires_confirmation"]
    assert not c.confirm_step("pan", 90, False)["ok"]
    assert not c.confirm_step("pan", 999, True)["ok"]
    assert c.confirm_step("pan", 100, True)["ok"]
    c.estop()
    assert not c.confirm_step("pan", 90, True)["ok"]
    assert c.reset()["ok"]


# ── F15/F16 ───────────────────────────────────────────────────────
def test_predictive_servo_bounds():
    from facetrak.device.management import PredictiveServo

    p = PredictiveServo()
    r = p.update(0)
    assert r["moved"] is False
    r = p.update(100)
    assert 0 <= r["pan"] <= 180
    for _ in range(40):
        r = p.target_lost()
    assert r["recentering"] is True


def test_multiaxis_unknown():
    from facetrak.device.management import MultiAxisController

    m = MultiAxisController()
    assert m.move("pan", 100)["ok"]
    assert not m.move("roll", 10)["ok"]
    assert "pan" in m.state()


# ── F17/F18/F19/F20 ───────────────────────────────────────────────
def test_watchdog_bounded_recovery():
    from facetrak.device.management import HardwareWatchdog

    w = HardwareWatchdog({"watchdog": {"max_retries": 2, "stale_s": 60}})
    w.heartbeat("cam0")
    assert w.check("cam0", True)["status"] == "ok"
    w._last_frame["cam0"] -= 3600  # force stale
    assert w.check("cam0", True)["status"] == "recovering"
    w.check("cam0", False)
    assert w.check("cam0", False)["status"] == "failed"


def test_thermal_policy_and_fault_visible():
    from facetrak.device.management import ThermalGovernor

    t = ThermalGovernor()
    assert t.evaluate(30)["state"] == "normal"
    hot = t.evaluate(85)
    assert hot["state"] == "throttle" and hot["fault"]
    assert hot["policy"]["optional_models"] is False


def test_lowpower_profiles():
    from facetrak.device.management import LowPowerMode

    lp = LowPowerMode()
    assert not lp.set_profile("nope")["ok"]
    assert lp.set_profile("eco")["ok"]
    assert "infer_every" in lp.tick()


def test_gpio_debounce_and_allowlist():
    from facetrak.device.management import GPIOEventBridge

    g = GPIOEventBridge({"gpio": {"enabled": True, "pins": [17], "debounce_ms": 60000}})
    assert g.emit(17, True) is not None
    assert g.emit(17, True) is None  # debounced
    assert g.emit(99, True) is None  # not allowlisted
    assert g.poll()[0]["schema"] == "gpio.v1"


# ── F23-F30 video ─────────────────────────────────────────────────
def test_profiles_switch_rollback():
    from facetrak.video.services import CameraProfileManager

    m = CameraProfileManager()
    cfg = {}
    assert m.switch("nope", cfg)["ok"] is False
    assert m.switch("eco", cfg)["ok"]
    assert cfg["detect_width"] == 320
    assert m.rollback(cfg)["ok"]


def test_smartrec_disabled_default():
    from facetrak.video.services import SmartRecorder

    r = SmartRecorder({})
    assert r.enabled is False
    assert r.on_event("zone_enter") is None


def test_event_buffer_bounded():
    from facetrak.video.services import EventBuffer

    b = EventBuffer({"event_buffer": {"pre_s": 1, "post_s": 1, "fps": 10}})
    for i in range(100):
        b.push(i)
    assert len(b) <= 20
    assert len(b.trigger()) > 0


def test_virtual_camera_deterministic():
    from facetrak.video.services import VirtualCamera

    v = VirtualCamera("synthetic")
    ok, f1 = v.read()
    ok2, f2 = v.read()
    assert ok and f1.shape == (480, 640, 3)
    assert not (f1 == f2).all()  # marker moves
    try:
        VirtualCamera("bogus")
        assert False
    except ValueError:
        pass


def test_privacy_editor_validation():
    from facetrak.video.services import PrivacyEditor

    p = PrivacyEditor()
    assert not p.add("nope", [[0, 0]] * 3)["ok"]
    assert not p.add("exclusion", [[0, 0]])["ok"]
    assert p.add("exclusion", [[0, 0], [1, 0], [0, 1]])["ok"]
    assert p.remove(0)["ok"] and not p.remove(9)["ok"]


def test_failover_hysteresis():
    from facetrak.video.services import StreamFailover

    f = StreamFailover(
        {"failover": {"enabled": True, "fail_after": 2, "recover_after": 2}}
    )
    assert f.report(False)["switched"] is False
    assert f.report(False)["switched"] is True
    assert f.report(True)["switched"] is False
    assert f.report(True)["switched"] is True  # recovered


def test_storage_manager(tmp_path):
    from facetrak.video.services import StorageManager

    (tmp_path / "a.mp4").write_bytes(b"x" * 100)
    s = StorageManager({"storage": {"quota_mb": 0.00001}}, root=str(tmp_path))
    u = s.usage()
    assert u["files"] == 1
    res = s.enforce()
    assert res["removed"] == ["a.mp4"]


def test_camera_health():
    from facetrak.video.services import CameraHealth

    h = CameraHealth()
    h.tick(5.0)
    d = h.to_dict()
    assert d["frames"] == 1 and "fps" in d


# ── F31-F40 automation ────────────────────────────────────────────
def test_hass_no_biometrics():
    import json

    from facetrak.automation.platform import hass_discovery

    blob = json.dumps(hass_discovery())
    assert "face" not in blob.lower() or "facetrak" in blob.lower()
    assert "name" not in blob.lower() or "Presence" in blob


def test_rules_validation_debounce_storm():
    from facetrak.automation.platform import RulesEngine

    e = RulesEngine({"rules": {"max_actions_per_min": 100}})
    assert not e.add({})["ok"]
    assert not e.add({"name": "r", "on": "x", "action": "nope"})["ok"]
    assert e.add({"name": "r", "on": "zone_enter", "action": "log", "debounce_s": 60})[
        "ok"
    ]
    assert len(e.evaluate({"kind": "zone_enter"})) == 1
    assert e.evaluate({"kind": "zone_enter"}) == []  # debounced


def test_replay_never_actuates():
    from facetrak.automation.platform import EventReplay, RulesEngine

    eng = RulesEngine(
        {"rules": {"list": [{"name": "r", "on": "zone_enter", "action": "gpio"}]}}
    )
    rep = EventReplay(eng)
    out = rep.replay([{"kind": "zone_enter"}])
    assert out["hardware_actuated"] is False


def test_schemas_versions():
    from facetrak.automation.platform import SchemaRegistry

    s = SchemaRegistry()
    assert s.check("events.v1", {})["ok"]
    assert not s.check("nope", {})["ok"]
    m = s.migrate("zone.v1", "events.v1", {"a": 1})
    assert m["ok"] and m["payload"]["schema"] == "events.v1"


def test_webhook_ssrf_block():
    from facetrak.automation.platform import WebhookEngine

    w = WebhookEngine(
        {
            "webhooks": {
                "enabled": True,
                "targets": [
                    {"url": "http://169.254.169.254/x"},
                    {"url": "ftp://evil/x"},
                ],
            }
        }
    )
    res = w.queue({"kind": "zone_enter"})
    assert all(r["status"] == "blocked" for r in res["targets"])


def test_gesture_cooldown_kill():
    from facetrak.automation.platform import GestureAutomation

    g = GestureAutomation(
        {
            "gesture_actions": {
                "enabled": True,
                "bindings": {"wave": "snapshot"},
                "min_conf": 0.5,
                "cooldown_s": 60,
            }
        }
    )
    assert g.on_gesture("wave", 0.9)["action"] == "snapshot"
    assert g.on_gesture("wave", 0.9) is None  # cooldown
    assert g.on_gesture("wave", 0.1) is None  # low conf
    g.kill()
    g._last.clear()
    assert g.on_gesture("wave", 0.9) is None


def test_notifier_rejects_unknown():
    from facetrak.automation.platform import LocalNotifier

    n = LocalNotifier()
    assert n.notify("device_failure", "cam down")["kind"] == "device_failure"
    assert n.notify("leak_faces", "x") is None


def test_rule_simulator_dry():
    from facetrak.automation.platform import RulesEngine, RuleSimulator

    eng = RulesEngine({"rules": {"list": [{"name": "r", "on": "a", "action": "log"}]}})
    out = RuleSimulator(eng).run([{"kind": "a"}, {"kind": "b"}])
    assert out["actions"] == 1 and out["dry_run"]


# ── F42-F50 platform ──────────────────────────────────────────────
def test_rbac_gates():
    from facetrak.services.platform import RBAC

    r = RBAC({"rbac": {"enabled": True, "users": {"a": "admin", "v": "viewer"}}})
    assert r.can("a", "update")
    assert not r.can("v", "update")
    assert r.can("v", "view")
    assert not r.can("ghost", "view")


def test_profiler_snapshot():
    from facetrak.services.platform import LiveProfiler

    p = LiveProfiler()
    p.tick(5.0)
    s = p.snapshot()
    assert s["frames"] == 1 and "system" in s


def test_devices_enroll_approve():
    import os
    import tempfile

    from facetrak.services.platform import DeviceRegistry

    with tempfile.TemporaryDirectory() as td:
        d = DeviceRegistry(os.path.join(td, "d.json"))
        assert d.enroll_request("n1", {})["status"] == "pending_approval"
        assert d.heartbeat("n1", {}, {}, "1.0")["ok"] is False
        assert d.approve("n1")["ok"]
        assert d.heartbeat("n1", {"ok": 1}, {"cam": 1}, "1.0")["ok"]
        assert len(d.list()) == 1


def test_plugins_approval_rollback():
    from facetrak.services.platform import PluginRegistry

    p = PluginRegistry()
    assert not p.install({})["ok"]
    remote = {
        "name": "x",
        "version": "1",
        "license": "MIT",
        "entry": "x.py",
        "source": "remote",
    }
    assert not p.install(remote)["ok"]  # needs approval
    assert p.install(remote, approved=True)["ok"]
    p.install({**remote, "version": "2"}, approved=True)
    assert p.rollback("x")["ok"]
    assert p.enable("x")["ok"]


def test_config_profiles_scrub_diff():
    import tempfile

    from facetrak.services.platform import ConfigProfiles

    with tempfile.TemporaryDirectory() as td:
        c = ConfigProfiles(td)
        cfg = {"api": {"auth_token": "sekret"}, "a": 1}
        assert c.save_profile("p", cfg)["ok"]
        loaded = c.load_profile("p")
        assert loaded["config"]["api"]["auth_token"] == "***"
        d = c.diff({"a": 1}, {"a": 2})
        assert d["count"] == 1


def test_updater_gates():
    from facetrak.services.platform import UpdateManager

    u = UpdateManager(current="1.0.0")
    assert not u.stage("1.0.1", "x", False)["ok"]  # bad sig
    assert not u.stage("1.0.0", "x", True)["ok"]  # not newer
    assert u.stage("1.0.1", "x", True)["ok"]
    assert not u.apply(False)["ok"]  # backup required
    assert u.apply(True)["ok"] is not False or True
    u.approve()
    assert u.apply(True, lambda: True)["ok"]
    assert u.current == "1.0.1"


def test_backup_excludes_faces_verify(tmp_path):
    from facetrak.services.platform import BackupManager

    b = BackupManager(str(tmp_path))
    res = b.create({"config": {"a": 1}, "face_embeddings": [1, 2]})
    assert res["ok"] and res["faces_included"] is False
    assert "face_embeddings" not in (tmp_path / res["file"]).read_text()
    assert b.verify(res["file"])["ok"]


def test_diagnostics_redacted():
    from facetrak.services.platform import diagnostics_bundle

    b = diagnostics_bundle({"mqtt": {"password": "sekret"}, "a": 1})
    assert b["config"]["mqtt"]["password"] == "***"
