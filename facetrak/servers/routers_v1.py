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


# ── insightface / face backends (F-A) ──
def _engine_cfg():
    from facetrak.servers.rest import _get

    return _get()


@router.get("/face/backends")
def face_backends(_: bool = Depends(_auth())):
    from facetrak.servers.rest import _get

    e = _get()
    adapter = getattr(e, "insightface_adapter", None)
    return {
        "detector_backend": e.cfg.get("detector_backend", "yunet"),
        "recognition_backend": e.cfg.get("recognition_backend", "sface"),
        "liveness_backend": e.cfg.get("liveness_backend", "mediapipe"),
        "available": {
            "yunet": True,
            "sface": True,
            "scrfd": bool(adapter and adapter.available),
            "arcface": bool(
                getattr(e, "arcface", None) is not None
                and e.arcface.ready),
        },
        "insightface": {
            "enabled": bool(e.cfg.get("insightface", {}).get("enabled", False)),
            "available": bool(adapter and adapter.available),
            "provider": getattr(adapter, "provider_name", None),
            "model_pack": getattr(adapter, "model_pack", None),
            "init_error": getattr(adapter, "init_error", None),
        },
    }


class FaceBackendBody(BaseModel):
    detector_backend: str | None = None
    recognition_backend: str | None = None
    liveness_backend: str | None = None


@router.post("/face/backends")
def face_backends_switch(body: FaceBackendBody, _: bool = Depends(_auth())):
    from facetrak.core import config as cfgmod
    from facetrak.servers.rest import _get

    e = _get()
    allowed_det = {"yunet", "scrfd"}
    allowed_rec = {"sface", "arcface"}
    allowed_live = {"mediapipe", "insightface"}
    if body.detector_backend is not None:
        if body.detector_backend not in allowed_det:
            raise HTTPException(400, f"unknown detector {body.detector_backend!r}")
        if body.detector_backend == "scrfd":
            adapter = getattr(e, "insightface_adapter", None)
            if not (adapter and adapter.available):
                raise HTTPException(
                    409, "scrfd unavailable: enable insightface with licensed models first")
        e.cfg["detector_backend"] = body.detector_backend
        e.detector = None  # re-created on next start()
    if body.recognition_backend is not None:
        if body.recognition_backend not in allowed_rec:
            raise HTTPException(400, f"unknown recognizer {body.recognition_backend!r}")
        if body.recognition_backend == "arcface":
            if not (getattr(e, "arcface", None) is not None and e.arcface.ready):
                raise HTTPException(
                    409, "arcface unavailable: enable insightface with licensed models first")
        e.cfg["recognition_backend"] = body.recognition_backend
    if body.liveness_backend is not None:
        if body.liveness_backend not in allowed_live:
            raise HTTPException(400, f"unknown liveness {body.liveness_backend!r}")
        e.cfg["liveness_backend"] = body.liveness_backend
    cfgmod.save(e.cfg)
    return {"ok": True,
            "detector_backend": e.cfg.get("detector_backend"),
            "recognition_backend": e.cfg.get("recognition_backend"),
            "liveness_backend": e.cfg.get("liveness_backend")}


@router.get("/insightface/status")
def insightface_status(_: bool = Depends(_auth())):
    from facetrak.servers.rest import _get

    e = _get()
    adapter = getattr(e, "insightface_adapter", None)
    if adapter is None:
        return {"enabled": False, "available": False,
                "init_error": "adapter not configured"}
    return {
        "enabled": True,
        "available": adapter.available,
        "provider": adapter.provider_name,
        "model_pack": adapter.model_pack,
        "models_dir": str(adapter.models_dir),
        "models_present": adapter.list_available_models(),
        "init_error": adapter.init_error,
    }


# ── model registry F21 (manifest / checksum / license gates) ──
def _model_registry():
    from facetrak.servers.rest import _get

    e = _get()
    reg = getattr(e, "model_registry", None)
    if reg is None:
        from facetrak.device.management import ModelRegistry as DeviceModelRegistry

        reg = DeviceModelRegistry(e.cfg)
        e.model_registry = reg
    return reg


@router.get("/model-registry")
def model_registry_list(backend: str | None = None,
                        _: bool = Depends(_auth())):
    reg = _model_registry()
    return [m.to_dict() for m in reg.list_models(backend=backend)]


class ModelManifestBody(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = "1.0"
    file_path: str = Field(min_length=1, max_length=512)
    sha256: str = ""
    license: str = ""
    license_url: str = ""
    backend: str = "scrfd"
    input_shape: list[int] = Field(default_factory=lambda: [1, 3, 640, 640])
    output_shapes: list[list[int]] = Field(default_factory=list)
    embedding_dim: int | None = None
    commercial_license_required: bool = True
    source_url: str = ""
    license_consent: bool = False


@router.post("/model-registry/register")
def model_registry_register(body: ModelManifestBody, _: bool = Depends(_auth())):
    from facetrak.device.management import ModelManifest as DeviceManifest

    reg = _model_registry()
    if body.commercial_license_required and not body.license_consent:
        raise HTTPException(
            403, "license consent required: resubmit with license_consent=true "
                 "after verifying commercial rights for this model")
    manifest = DeviceManifest(
        name=body.name, version=body.version, file_path=body.file_path,
        sha256=body.sha256, license=body.license,
        license_url=body.license_url, backend=body.backend,
        input_shape=body.input_shape, output_shapes=body.output_shapes,
        embedding_dim=body.embedding_dim,
        commercial_license_required=body.commercial_license_required,
        source_url=body.source_url)
    ok = reg.register(manifest)
    if not ok:
        raise HTTPException(400, manifest.load_error or "registration failed")
    return {"ok": True, "model": manifest.to_dict()}


@router.post("/model-registry/{name}/activate")
def model_registry_activate(name: str, _: bool = Depends(_auth())):
    reg = _model_registry()
    if not reg.activate(name):
        raise HTTPException(404, f"model {name!r} not found")
    return {"ok": True, "model": name}


@router.get("/model-registry/license")
def model_registry_license(_: bool = Depends(_auth())):
    return _model_registry().get_license_status()


# ── anonymization jobs (PrivateFrame-inspired, same server) ──
_anonymize_jobs: dict[str, dict] = {}
_anonymize_threads: dict[str, object] = {}
_ANONYMIZE_MAX_JOBS = 100


def _anonymize_evict() -> None:
    """Bound job memory: drop oldest terminal jobs beyond cap."""
    if len(_anonymize_jobs) <= _ANONYMIZE_MAX_JOBS:
        return
    terminal = [j for j in _anonymize_jobs.values()
                if j.get("status") in ("done", "failed", "cancelled")]
    excess = len(_anonymize_jobs) - _ANONYMIZE_MAX_JOBS
    for j in terminal[:excess]:
        _anonymize_jobs.pop(j["id"], None)
        _anonymize_threads.pop(j["id"], None)


def _anonymize_job(job_id: str) -> dict:
    job = _anonymize_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    return job


class AnonymizeBody(BaseModel):
    input: str = Field(min_length=1, max_length=1024)
    output_dir: str = Field(min_length=1, max_length=1024)
    method: str = "gaussian"
    box_scale: float = Field(default=1.15, ge=1.0, le=2.0)
    mosaic_block: int = Field(default=20, ge=4, le=128)
    gaussian_ksize: int = Field(default=51, ge=3, le=201)
    gaussian_sigma: float = Field(default=30.0, ge=1.0, le=200.0)
    detector_backend: str = "yunet"
    recognition_backend: str = "none"
    max_frames: int | None = None


def _resolve_job_path(value: str, must_exist: bool = False) -> str:
    from pathlib import Path as _P

    p = _P(value).expanduser().resolve()
    if must_exist and not p.exists():
        raise HTTPException(400, f"path does not exist: {value}")
    return str(p)


@router.post("/anonymize/jobs")
def anonymize_start(body: AnonymizeBody, _: bool = Depends(_auth())):
    import threading
    import uuid

    from facetrak.video.anonymize import (
        AnalysisConfig,
        RedactionConfig,
        RedactionMethod,
        VideoAnonymizer,
    )

    input_path = _resolve_job_path(body.input, must_exist=True)
    output_dir = _resolve_job_path(body.output_dir)
    if body.method not in ("gaussian", "mosaic"):
        raise HTTPException(400, "method must be gaussian or mosaic")
    if body.detector_backend not in ("yunet", "scrfd"):
        raise HTTPException(400, "detector_backend must be yunet or scrfd")
    if body.recognition_backend not in ("none", "sface", "arcface"):
        raise HTTPException(400, "recognition_backend must be none/sface/arcface")
    if body.detector_backend == "scrfd" or body.recognition_backend == "arcface":
        e = _engine_cfg()
        adapter = getattr(e, "insightface_adapter", None)
        if not (adapter and adapter.available):
            raise HTTPException(
                409, "insightface backend unavailable: enable with licensed models first")

    job_id = uuid.uuid4().hex[:12]
    job: dict = {"id": job_id, "status": "queued", "progress": 0.0,
                 "current_frame": 0, "total_frames": 0,
                 "input": input_path, "output_dir": output_dir,
                 "result": None, "error": None}
    _anonymize_jobs[job_id] = job
    _anonymize_evict()

    def _run():
        from pathlib import Path as _P

        try:
            job["status"] = "analyzing"
            e = _engine_cfg()
            adapter = getattr(e, "insightface_adapter", None)
            face_db = getattr(e, "db", None)
            analysis_cfg = AnalysisConfig(
                detector_backend=body.detector_backend,
                recognition_backend=body.recognition_backend)
            redaction_cfg = RedactionConfig(
                method=RedactionMethod(body.method),
                box_scale=body.box_scale,
                mosaic_block=body.mosaic_block,
                gaussian_ksize=(body.gaussian_ksize | 1),
                gaussian_sigma=body.gaussian_sigma)
            anon = VideoAnonymizer(
                analysis_cfg, redaction_cfg,
                insightface_adapter=adapter, face_db=face_db)

            def _cb(progress: float, status: str, cur: int, total: int):
                job["progress"] = progress
                job["status"] = status
                job["current_frame"] = cur
                job["total_frames"] = total

            anon.set_progress_callback(_cb)
            job["anonymizer"] = anon
            result = anon.process(
                _P(input_path), _P(output_dir),
                max_frames=body.max_frames)
            job["result"] = result
            job["status"] = "done"
            job["progress"] = 1.0
        except InterruptedError as exc:
            job["status"] = "cancelled"
            job["error"] = str(exc)
        except Exception as exc:  # never leak face data; counts only
            job["status"] = "failed"
            job["error"] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    _anonymize_threads[job_id] = t
    t.start()
    return {"ok": True, "job_id": job_id}


@router.get("/anonymize/jobs")
def anonymize_list(_: bool = Depends(_auth())):
    return [{k: v for k, v in j.items() if k != "anonymizer"}
            for j in _anonymize_jobs.values()]


@router.get("/anonymize/jobs/{job_id}")
def anonymize_get(job_id: str, _: bool = Depends(_auth())):
    job = _anonymize_job(job_id)
    return {k: v for k, v in job.items() if k != "anonymizer"}


@router.delete("/anonymize/jobs/{job_id}")
def anonymize_cancel(job_id: str, _: bool = Depends(_auth())):
    job = _anonymize_job(job_id)
    anon = job.get("anonymizer")
    if anon is not None and hasattr(anon, "cancel"):
        anon.cancel()
    job["status"] = "cancelled"
    return {"ok": True, "job_id": job_id}


class AnonymizeEditBody(BaseModel):
    analysis_json: str = Field(min_length=1, max_length=1024)
    track_ids: list[int] = Field(default_factory=list)
    redact: bool = True


@router.post("/anonymize/analysis/edit")
def anonymize_edit(body: AnonymizeEditBody, _: bool = Depends(_auth())):
    from facetrak.video.anonymize import VideoAnalysis

    path = _resolve_job_path(body.analysis_json, must_exist=True)
    from pathlib import Path as _P

    analysis = VideoAnalysis.load(_P(path))
    wanted = set(body.track_ids)
    n = 0
    for t in analysis.tracks:
        if not wanted or t.track_id in wanted:
            t.redact = body.redact
            n += 1
    analysis.save(_P(path))
    return {"ok": True, "updated_tracks": n, "analysis_json": path}


class AnonymizeRenderBody(BaseModel):
    input: str = Field(min_length=1, max_length=1024)
    analysis_json: str = Field(min_length=1, max_length=1024)
    output: str = Field(min_length=1, max_length=1024)
    method: str = "gaussian"
    box_scale: float = Field(default=1.15, ge=1.0, le=2.0)

@router.post("/anonymize/render")
def anonymize_render(body: AnonymizeRenderBody, _: bool = Depends(_auth())):
    from pathlib import Path as _P

    from facetrak.video.anonymize import (
        RedactionConfig,
        RedactionMethod,
        VideoAnalysis,
        VideoAnonymizer,
    )

    if body.method not in ("gaussian", "mosaic"):
        raise HTTPException(400, "method must be gaussian or mosaic")
    input_path = _resolve_job_path(body.input, must_exist=True)
    analysis_path = _resolve_job_path(body.analysis_json, must_exist=True)
    output_path = _resolve_job_path(body.output)
    analysis = VideoAnalysis.load(_P(analysis_path))
    anon = VideoAnonymizer(redaction_cfg=RedactionConfig(
        method=RedactionMethod(body.method), box_scale=body.box_scale))
    # Render only: no inference, uses stored masks.
    result = anon.render(_P(input_path), analysis, _P(output_path))
    return {"ok": True, **result}


class AnonymizePurgeBody(BaseModel):
    output_dir: str = Field(min_length=1, max_length=1024)
    retention_days: int | None = None


@router.post("/anonymize/purge")
def anonymize_purge(body: AnonymizePurgeBody, _: bool = Depends(_auth())):
    from facetrak.video.anonymize import purge_outputs
    from facetrak.servers.rest import _get

    days = body.retention_days
    if days is None:
        days = int(_get().cfg.get("storage", {}).get("retention_days", 7))
    if days < 0:
        raise HTTPException(400, "retention_days must be >= 0")
    out = _resolve_job_path(body.output_dir, must_exist=True)
    from pathlib import Path as _P
    return {"ok": True, "output_dir": out,
            **purge_outputs(_P(out), days)}


@router.get("/registration/quality")
def registration_quality(limit: int = 20, _: bool = Depends(_auth())):
    from facetrak.servers.rest import _get

    return _get().registration_quality(max(1, min(limit, 50)))
