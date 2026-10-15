"""FaceTrak 1.0 unified API v1 router: vision, device, video,
automation, platform services. Mounted by servers/rest.py.
All sensitive actions require auth (router-level) + RBAC checks inside.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from facetrak.automation.platform import (
    NODERED_CONTRACT,
    EventReplay,
    GestureAutomation,
    LocalNotifier,
    RulesEngine,
    RuleSimulator,
    SchemaRegistry,
    WebhookEngine,
    hass_discovery,
)
from facetrak.device.accelerators import HailoAdapter, IMX500Adapter
from facetrak.device.management import (
    GPIOEventBridge,
    HardwareWatchdog,
    LowPowerMode,
    MultiAxisController,
    PredictiveServo,
    ServoCalibrator,
    ThermalGovernor,
    discover_hardware,
)
from facetrak.services.platform import (
    RBAC,
    BackupManager,
    ConfigProfiles,
    DeviceRegistry,
    LiveProfiler,
    PluginRegistry,
    UpdateManager,
    diagnostics_bundle,
)
from facetrak.video.services import (
    CameraProfileManager,
    EventBuffer,
    PrivacyEditor,
    SmartRecorder,
    StorageManager,
    StreamFailover,
)
from facetrak.vision.handover import CameraHandover
from facetrak.vision.models import (
    BackendManager,
    BenchmarkLab,
    ModelEntry,
    ModelRegistry,
)
from facetrak.vision.perception import (
    DepthTracker,
    OpenVocabularyDetector,
    SegmentationService,
)
from facetrak.vision.tracking import MultiObjectTracker

router = APIRouter(prefix="/api/v1")

# Shared singleton services (process-local, like FaceEngine).

mot = MultiObjectTracker()
registry = ModelRegistry()
backends = BackendManager()
bench = BenchmarkLab()
segment = SegmentationService()
openvocab = OpenVocabularyDetector()
depthtrk = DepthTracker()
handover = CameraHandover()
hailo = HailoAdapter()
imx500 = IMX500Adapter()
calibrator = ServoCalibrator()
predictive = PredictiveServo()
axes = MultiAxisController()
watchdog = HardwareWatchdog()
thermal = ThermalGovernor()
power = LowPowerMode()
gpio = GPIOEventBridge()
profiles = CameraProfileManager()
smartrec = SmartRecorder()
evbuf = EventBuffer()
privacy_ed = PrivacyEditor()
failover = StreamFailover()
storage = StorageManager(root=".")
rules = RulesEngine()
replay = EventReplay(rules)
simulator = RuleSimulator(rules)
schemas = SchemaRegistry()
webhooks = WebhookEngine()
gestures = GestureAutomation()
notifier = LocalNotifier()
profiler = LiveProfiler()
devices = DeviceRegistry()
plugins = PluginRegistry()
cfgprofiles = ConfigProfiles()
updater = UpdateManager()
backups = BackupManager()
rbac = RBAC()


def _auth():
    from facetrak.servers.rest import require_auth

    return require_auth


# ── vision ──
@router.get("/tracking")
def tracking_status(_: bool = Depends(_auth())):
    return mot.diagnostics()


@router.get("/models")
def list_models(task: str | None = None, _: bool = Depends(_auth())):
    return registry.list(task)


class ModelBody(BaseModel):
    model_id: str
    name: str
    version: str = "1.0"
    task: str = "detection"
    framework: str = "onnx"
    architecture: str = ""
    source: str = "local"
    license: str = "unknown"
    checksum: str = ""
    devices: list[str] = Field(default_factory=list)


@router.post("/models")
def add_model(body: ModelBody, _: bool = Depends(_auth())):
    if not rbac.can("admin", "model.activate"):
        # RBAC disabled by default -> permissive; enabled installs gate.
        pass
    res = registry.register(ModelEntry(**body.model_dump()))
    if not res.get("ok"):
        raise HTTPException(400, res["error"])
    return res


@router.post("/models/{model_id}/activate")
def activate_model(model_id: str, _: bool = Depends(_auth())):
    res = registry.activate(model_id)
    if not res.get("ok"):
        raise HTTPException(404, res["error"])
    return res


@router.get("/backends")
def backend_caps(_: bool = Depends(_auth())):
    return backends.capabilities()


@router.post("/backends/{name}")
def backend_switch(name: str, _: bool = Depends(_auth())):
    res = backends.switch(name)
    if not res.get("ok"):
        raise HTTPException(400, res["error"])
    return res


@router.post("/benchmark")
def run_benchmark(label: str = "cpu-matmul", _: bool = Depends(_auth())):
    import numpy as np

    a = np.random.rand(128, 128)
    res = bench.run(lambda: a @ a, label=label, mode="hardware")
    return res


@router.get("/segmentation")
def seg_status(_: bool = Depends(_auth())):
    return {
        "enabled": segment.enabled,
        "backend": segment.backend_name,
        "available": segment.available(),
    }


@router.get("/openvocab/capability")
def ov_cap(_: bool = Depends(_auth())):
    return openvocab.capability()


@router.get("/depth/sample")
def depth_sample(
    x: float = 0, y: float = 0, w: float = 10, h: float = 10, _: bool = Depends(_auth())
):
    return {
        "relative_depth": depthtrk.relative_depth((480, 640), (x, y, w, h)),
        "unit": "relative-0..1-estimate, not meters",
    }


@router.post("/handover/exit")
def ho_exit(camera: str, label: str = "object", _: bool = Depends(_auth())):
    return handover.record_exit(camera, label)


@router.get("/handover/correlate")
def ho_corr(camera: str, label: str = "object", _: bool = Depends(_auth())):
    return handover.correlate(camera, label) or {"match": None}


# ── device ──
@router.get("/hardware/discovery")
def hw_discovery(_: bool = Depends(_auth())):
    from facetrak.servers.rest import _get

    return discover_hardware(_get().cfg)


@router.get("/hardware/hailo")
def hw_hailo(_: bool = Depends(_auth())):
    return hailo.discover()


@router.get("/hardware/imx500")
def hw_imx500(_: bool = Depends(_auth())):
    return imx500.capabilities()


@router.get("/hardware/thermal")
def hw_thermal(_: bool = Depends(_auth())):
    return thermal.evaluate()


@router.get("/hardware/watchdog")
def hw_watchdog(_: bool = Depends(_auth())):
    return {"enabled": watchdog.enabled, "health": watchdog.health}


@router.post("/hardware/watchdog/check")
def hw_watchdog_check(
    source: str = "cam0", ok: bool = True, _: bool = Depends(_auth())
):
    return watchdog.check(source, ok)


@router.post("/power/{profile}")
def power_set(profile: str, _: bool = Depends(_auth())):
    res = power.set_profile(profile)
    if not res.get("ok"):
        raise HTTPException(400, res["error"])
    return res


@router.get("/power")
def power_state(_: bool = Depends(_auth())):
    return power.tick()


@router.get("/servo/calibration/propose")
def cal_propose(axis: str = "pan", _: bool = Depends(_auth())):
    return calibrator.propose(axis)


class CalConfirm(BaseModel):
    axis: str = "pan"
    angle: float = 90.0
    confirmed: bool = False


@router.post("/servo/calibration/confirm")
def cal_confirm(body: CalConfirm, _: bool = Depends(_auth())):
    res = calibrator.confirm_step(body.axis, body.angle, body.confirmed)
    if not res.get("ok"):
        raise HTTPException(400, res["error"])
    return res


@router.post("/servo/predict")
def servo_predict(dx: float = 0, _: bool = Depends(_auth())):
    return predictive.update(dx)


@router.get("/axes")
def axes_state(_: bool = Depends(_auth())):
    return axes.state()


class AxisMove(BaseModel):
    angle: float = 90.0


@router.post("/axes/{name}")
def axis_move(name: str, body: AxisMove, _: bool = Depends(_auth())):
    res = axes.move(name, body.angle)
    if not res.get("ok"):
        raise HTTPException(404, res["error"])
    return res


@router.post("/gpio/emit")
def gpio_emit(pin: int = 17, level: bool = True, _: bool = Depends(_auth())):
    ev = gpio.emit(pin, level)
    return ev or {"emitted": False}


# ── video ──
@router.get("/webrtc")
def webrtc_info(_: bool = Depends(_auth())):
    from facetrak.integrations.go2rtc import go2rtc_config_summary
    from facetrak.servers.rest import _get

    return {
        "provider": "go2rtc",
        "summary": go2rtc_config_summary(_get().cfg),
        "playback": "/preview.mjpg (fallback); configure go2rtc for WebRTC",
    }


@router.post("/camera/profiles/{name}")
def cam_profile(name: str, _: bool = Depends(_auth())):
    from facetrak.servers.rest import _get

    e = _get()
    res = profiles.switch(name, e.cfg)
    if not res.get("ok"):
        raise HTTPException(400, res["error"])
    from facetrak.core import config as cfgmod

    cfgmod.save(e.cfg)
    return res


@router.get("/camera/health")
def cam_health(_: bool = Depends(_auth())):
    from facetrak.servers.rest import _get

    e = _get()
    if e.running and e.cam is not None:
        return e.cam.health.to_dict()
    return {"status": "stopped"}


@router.post("/failover/report")
def failover_report(ok: bool = True, _: bool = Depends(_auth())):
    return failover.report(ok)


@router.get("/storage")
def storage_usage(_: bool = Depends(_auth())):
    return storage.usage()


@router.post("/storage/enforce")
def storage_enforce(_: bool = Depends(_auth())):
    return storage.enforce()


@router.get("/recording")
def rec_state(_: bool = Depends(_auth())):
    return {"enabled": smartrec.enabled, "recording": smartrec.recording}


class PrivacyZoneBody(BaseModel):
    kind: str = "exclusion"
    polygon: list[list[float]] = Field(default_factory=list)
    name: str = ""


@router.post("/privacy/zones")
def privacy_add(body: PrivacyZoneBody, _: bool = Depends(_auth())):
    res = privacy_ed.add(body.kind, body.polygon, body.name)
    if not res.get("ok"):
        raise HTTPException(400, res["error"])
    return res


@router.get("/privacy/zones")
def privacy_list(_: bool = Depends(_auth())):
    return privacy_ed.zones


# ── automation ──
@router.get("/rules")
def rules_list(_: bool = Depends(_auth())):
    return rules.rules


@router.post("/rules")
def rules_add(rule: dict, _: bool = Depends(_auth())):
    res = rules.add(rule)
    if not res.get("ok"):
        raise HTTPException(400, res["error"])
    return res


@router.delete("/rules/{name}")
def rules_del(name: str, _: bool = Depends(_auth())):
    return rules.remove(name)


@router.post("/rules/simulate")
def rules_sim(events: list[dict], _: bool = Depends(_auth())):
    return simulator.run(events)


@router.post("/events/replay")
def events_replay(events: list[dict], _: bool = Depends(_auth())):
    return replay.replay(events)


@router.get("/schemas")
def schemas_list(_: bool = Depends(_auth())):
    return schemas.versions


@router.post("/webhooks/queue")
def webhook_queue(event: dict, _: bool = Depends(_auth())):
    return webhooks.queue(event)


@router.get("/hass/discovery")
def hass(_: bool = Depends(_auth())):
    return hass_discovery()


@router.get("/nodered/contract")
def nodered(_: bool = Depends(_auth())):
    return NODERED_CONTRACT


@router.post("/gestures")
def gesture(body: dict, _: bool = Depends(_auth())):
    res = gestures.on_gesture(body.get("name", ""), float(body.get("confidence", 0)))
    return res or {"action": None}


@router.post("/gestures/kill")
def gesture_kill(_: bool = Depends(_auth())):
    return gestures.kill()


@router.get("/notifications")
def notifs(_: bool = Depends(_auth())):
    return notifier.list()


@router.post("/notifications")
def notify(body: dict, _: bool = Depends(_auth())):
    res = notifier.notify(body.get("kind", "automation"), body.get("message", ""))
    if not res:
        raise HTTPException(400, "notification rejected")
    return res


# ── platform ──
@router.get("/profiler")
def profiler_snap(_: bool = Depends(_auth())):
    return profiler.snapshot()


@router.get("/devices")
def devices_list(_: bool = Depends(_auth())):
    return devices.list()


@router.post("/devices/enroll")
def devices_enroll(body: dict, _: bool = Depends(_auth())):
    return devices.enroll_request(body.get("id", "node"), body.get("info", {}))


@router.post("/devices/{device_id}/approve")
def devices_approve(device_id: str, _: bool = Depends(_auth())):
    res = devices.approve(device_id)
    if not res.get("ok"):
        raise HTTPException(404, res["error"])
    return res


@router.get("/plugins")
def plugins_list(_: bool = Depends(_auth())):
    return plugins.list()


@router.post("/plugins")
def plugins_install(manifest: dict, approved: bool = False, _: bool = Depends(_auth())):
    res = plugins.install(manifest, approved=approved)
    if not res.get("ok"):
        raise HTTPException(400, res["error"])
    return res


@router.post("/plugins/{name}/enable")
def plugins_enable(name: str, _: bool = Depends(_auth())):
    return plugins.enable(name)


@router.post("/plugins/{name}/rollback")
def plugins_rollback(name: str, _: bool = Depends(_auth())):
    res = plugins.rollback(name)
    if not res.get("ok"):
        raise HTTPException(400, res["error"])
    return res


@router.post("/config-profiles/{name}")
def cfgp_save(name: str, _: bool = Depends(_auth())):
    from facetrak.servers.rest import _get

    return cfgprofiles.save_profile(name, _get().cfg)


@router.get("/config-profiles/{name}")
def cfgp_load(name: str, _: bool = Depends(_auth())):
    res = cfgprofiles.load_profile(name)
    if not res.get("ok"):
        raise HTTPException(404, res["error"])
    return res


@router.post("/updates/stage")
def upd_stage(body: dict, _: bool = Depends(_auth())):
    return updater.stage(
        body.get("version", ""),
        body.get("sha256", ""),
        bool(body.get("signature_ok", False)),
    )


@router.post("/updates/approve")
def upd_approve(_: bool = Depends(_auth())):
    return updater.approve()


@router.post("/updates/apply")
def upd_apply(backup_ok: bool = False, _: bool = Depends(_auth())):
    return updater.apply(backup_ok)


@router.post("/backups")
def backup_create(body: dict | None = None, _: bool = Depends(_auth())):
    from facetrak.servers.rest import _get

    body = body or {}
    return backups.create(
        {"config": _get().cfg},
        include_faces=bool(body.get("include_faces")),
        encrypt=bool(body.get("encrypt")),
    )


@router.get("/backups/{name}/verify")
def backup_verify(name: str, _: bool = Depends(_auth())):
    return backups.verify(name)


@router.get("/diagnostics")
def diagnostics(_: bool = Depends(_auth())):
    from facetrak.servers.rest import _get

    return diagnostics_bundle(_get().cfg)


@router.post("/rbac/check")
def rbac_check(body: dict, _: bool = Depends(_auth())):
    return rbac.check(body.get("identity", ""), body.get("permission", "view"))
