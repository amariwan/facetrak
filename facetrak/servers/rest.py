import logging
import threading
import time
from contextlib import asynccontextmanager

import cv2
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from facetrak.core import config as cfgmod
from facetrak.core.engine import FaceEngine
from facetrak.core.security import get_api_token, verify_token
from facetrak.events import PrivacyMask, Zone
from facetrak.storage import db

logger = logging.getLogger(__name__)

_engine: FaceEngine | None = None
_auth_warned = False

bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    db.init()
    yield


def _get() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine


def _cfg() -> dict:
    return _get().cfg


def _err(code: str, message: str, status: int):
    return JSONResponse({"error": {"code": code, "message": message}},
                        status_code=status)


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    global _auth_warned
    e = _get()
    expected = get_api_token(e.cfg)
    if not expected:
        if not _auth_warned:
            logger.warning(
                "API auth disabled (no token). Set FACETRAK_API_TOKEN or "
                "api.auth_token. Binding stays localhost-only by default.")
            _auth_warned = True
        return True
    presented = creds.credentials if creds else None
    if not verify_token(e.cfg, presented):
        raise HTTPException(401, "Invalid or missing API token")
    return True


app = FastAPI(title="FaceTrak", version="2.0", lifespan=_lifespan)


@app.exception_handler(HTTPException)
async def _http_errors(_: Request, exc: HTTPException):
    code = {400: "bad_request", 401: "unauthorized", 403: "forbidden",
            404: "not_found", 503: "unavailable"}.get(exc.status_code, "error")
    return _err(code, str(exc.detail), exc.status_code)


_MAX_LIMIT = 1000


def _clamp_limit(limit: int) -> int:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return 50
    return max(1, min(_MAX_LIMIT, limit))


def _hide_estimates() -> bool:
    return bool(_cfg().get("privacy", {}).get("hide_estimates", False))


def _public_faces(faces: list[dict]) -> list[dict]:
    if not _hide_estimates():
        return faces
    out = []
    for f in faces:
        f = dict(f)
        f.pop("age", None)
        f.pop("gender", None)
        out.append(f)
    return out


# ── health / status ──

@app.get("/health")
def health():
    return {"ok": True, "version": "2.0"}


@app.get("/status")
def status(_: bool = Depends(require_auth)):
    e = _get()
    return {
        "running":       e.running,
        "camera":        e.current_cam_idx,
        "faces":         len(e.tracker.active),
        "recording":     e.recorder.recording,
        "blur_unknown":  e.blur_enabled,
        "servo_enabled": e.servo_enabled,
        "heatmap":       e.heatmap_enabled,
        "known_faces":   e.db.known_names,
        "zones":         [z.name for z in e.zones.zones],
        "masks":         len(e.masks),
        "mqtt":          {"enabled": e.mqtt.enabled,
                          "connected": e.mqtt.connected},
    }


@app.get("/metrics")
def metrics(_: bool = Depends(require_auth)):
    return _get().perf_stats()


@app.get("/cameras")
def cameras(_: bool = Depends(require_auth)):
    e = _get()
    red = cfgmod.redact(e.cfg)
    out = []
    for i, c in enumerate(red.get("cameras", [])):
        entry = {"index": i, "name": c.get("name"),
                 "source": c.get("source", c.get("url")),
                 "type": c.get("type", "usb"),
                 "active": i == e.current_cam_idx}
        if e.running and i < len(e.cameras):
            entry["health"] = e.cameras[i].health.to_dict()
        out.append(entry)
    return out


# ── faces / analysis (estimates, can be hidden) ──

@app.get("/faces")
def faces(_: bool = Depends(require_auth)):
    e = _get()
    if not e.running:
        raise HTTPException(503, "Camera not running")
    return _public_faces(e.live_faces())


@app.get("/face/analysis")
def analysis(_: bool = Depends(require_auth)):
    if _hide_estimates():
        raise HTTPException(403, "Estimates hidden by privacy settings")
    e = _get()
    m = e.metrics
    return {
        "emotion":    m.emotion,
        "smile":      m.smile,
        "mouth_open": m.mouth_open,
        "brow_raise": m.brow_raise,
        "eye_left":   m.eye_left,
        "eye_right":  m.eye_right,
        "attentive":  m.attentive,
        "gaze_h":     m.gaze_h,
        "gaze_v":     m.gaze_v,
        "gaze":       m.gaze_label,
        "yaw":        m.yaw,
        "pitch":      m.pitch,
        "roll":       m.roll,
    }


@app.get("/presence")
def presence(limit: int = 50, _: bool = Depends(require_auth)):
    return db.query_presence(_clamp_limit(limit))


@app.get("/crowd")
def crowd(_: bool = Depends(require_auth)):
    e = _get()
    summary = db.crowd_summary()
    return {"current": e.crowd.current, **summary}


@app.get("/emotions")
def emotions(limit: int = 200, _: bool = Depends(require_auth)):
    if _hide_estimates():
        raise HTTPException(403, "Estimates hidden by privacy settings")
    return db.query_emotions(_clamp_limit(limit))


# ── events / zones / masks ──

@app.get("/events")
def events(limit: int = 100, kind: str | None = None,
           _: bool = Depends(require_auth)):
    recent = _get().recent_events(_clamp_limit(limit))
    if kind:
        recent = [ev for ev in recent if ev.get("kind") == kind]
    stored = db.query_events(_clamp_limit(limit), kind=kind)
    return {"live": recent, "stored": stored}


@app.get("/zones")
def list_zones(_: bool = Depends(require_auth)):
    return _get().zones.to_config()


class ZoneBody(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    polygon: list[list[float]]
    enabled: bool = True
    analyze: bool = True


@app.post("/zones")
def add_zone(body: ZoneBody, _: bool = Depends(require_auth)):
    e = _get()
    try:
        zone = Zone(body.name, body.polygon, body.enabled, body.analyze)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if any(z.name == zone.name for z in e.zones.zones):
        raise HTTPException(400, f"Zone '{zone.name}' already exists")
    e.zones.zones.append(zone)
    e.cfg["zones"] = e.zones.to_config()
    cfgmod.save(e.cfg)
    return {"ok": True, "zone": zone.to_dict()}


@app.delete("/zones/{name}")
def delete_zone(name: str, _: bool = Depends(require_auth)):
    e = _get()
    before = len(e.zones.zones)
    e.zones.zones = [z for z in e.zones.zones if z.name != name]
    if len(e.zones.zones) == before:
        raise HTTPException(404, f"Zone '{name}' not found")
    e.cfg["zones"] = e.zones.to_config()
    cfgmod.save(e.cfg)
    return {"ok": True}


@app.get("/masks")
def list_masks(_: bool = Depends(require_auth)):
    return [m.to_dict() for m in _get().masks]


class MaskBody(BaseModel):
    polygon: list[list[float]]
    name: str = ""


@app.post("/masks")
def add_mask(body: MaskBody, _: bool = Depends(require_auth)):
    e = _get()
    try:
        mask = PrivacyMask(body.polygon, body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    e.masks.append(mask)
    e.cfg["privacy_masks"] = [m.to_dict() for m in e.masks]
    cfgmod.save(e.cfg)
    return {"ok": True}


@app.delete("/masks/{idx}")
def delete_mask(idx: int, _: bool = Depends(require_auth)):
    e = _get()
    if not 0 <= idx < len(e.masks):
        raise HTTPException(404, "Mask not found")
    del e.masks[idx]
    e.cfg["privacy_masks"] = [m.to_dict() for m in e.masks]
    cfgmod.save(e.cfg)
    return {"ok": True}


# ── camera control ──

@app.post("/camera/start")
def camera_start(_: bool = Depends(require_auth)):
    e = _get()
    if e.running:
        return {"ok": True, "message": "already running"}
    if not e.start():
        raise HTTPException(500, "Failed to start camera")
    return {"ok": True}


@app.post("/camera/stop")
def camera_stop(_: bool = Depends(require_auth)):
    e = _get()
    e.stop()
    return {"ok": True}


@app.post("/snapshot")
def snapshot(_: bool = Depends(require_auth)):
    e = _get()
    if not e.running:
        raise HTTPException(503, "Camera not running")
    path = e.snapshot()
    if not path:
        raise HTTPException(404, "No frame available")
    return FileResponse(path, media_type="image/png", filename="snapshot.png")


class BlurBody(BaseModel):
    blur: bool = True


@app.post("/blur/{name}")
def set_blur(name: str, body: BlurBody | None = None,
             blur: bool = True, _: bool = Depends(require_auth)):
    if body is not None:
        blur = body.blur
    if not name or len(name) > 128:
        raise HTTPException(400, "Invalid name")
    e = _get()
    e.set_blur_person(name, blur)
    return {"ok": True, "name": name, "blur": blur}


@app.get("/export/crowd")
def export_crowd(_: bool = Depends(require_auth)):
    e = _get()
    path = e.export_crowd_csv()
    if not path:
        raise HTTPException(404, "No crowd data")
    return FileResponse(path, media_type="text/csv", filename="crowd_stats.csv")


@app.get("/export/emotions")
def export_emotions(_: bool = Depends(require_auth)):
    if _hide_estimates():
        raise HTTPException(403, "Estimates hidden by privacy settings")
    from facetrak.storage.stats import EmotionTimeline
    path = EmotionTimeline.export_csv()
    if not path:
        raise HTTPException(404, "No emotion data")
    return FileResponse(path, media_type="text/csv", filename="emotion_log.csv")


# ── retention / erasure ──

@app.post("/retention/purge")
def retention_purge(_: bool = Depends(require_auth)):
    return {"ok": True, "deleted": _get().purge_retention()}


@app.delete("/data")
def delete_data(confirm: bool = False, include_faces: bool = False,
                _: bool = Depends(require_auth)):
    if not confirm:
        raise HTTPException(400, "Pass confirm=true to erase runtime data")
    if include_faces and not _cfg().get(
            "privacy", {}).get("allow_identifying_exports", False):
        raise HTTPException(
            403, "Face erasure needs privacy.allow_identifying_exports=true")
    return {"ok": True, "deleted": _get().delete_runtime_data(include_faces)}


# ── servo ──

class ServoCalBody(BaseModel):
    pan_offset: float = Field(ge=-10.0, le=10.0)
    tilt_offset: float = Field(ge=-10.0, le=10.0)


@app.post("/servo/calibrate")
def servo_calibrate(body: ServoCalBody, _: bool = Depends(require_auth)):
    e = _get()
    result = e.servo.calibrate(body.pan_offset, body.tilt_offset)
    e.cfg.setdefault("servo", {})["pan_offset"] = result["pan_offset"]
    e.cfg["servo"]["tilt_offset"] = result["tilt_offset"]
    cfgmod.save(e.cfg)
    return {"ok": True, **result}


# ── live preview + dashboard ──

def _mjpeg():
    e = _get()
    while True:
        frame = e.current_frame
        if frame is None:
            time.sleep(0.1)
            continue
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.get("/preview.mjpg")
def preview(_: bool = Depends(require_auth)):
    if not _get().running:
        raise HTTPException(503, "Camera not running")
    return StreamingResponse(
        _mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    import pathlib
    html = pathlib.Path(__file__).with_name("dashboard.html").read_text()
    return HTMLResponse(html)


def _api_host_port() -> tuple[str, int]:
    try:
        e = _get()
        api = e.cfg.get("api", {})
        return str(api.get("host", "127.0.0.1")), \
            int(e.cfg.get("api_port", 8765))
    except Exception:
        return "127.0.0.1", 8765


def start_api_thread(engine: FaceEngine, port: int | None = None):
    global _engine
    _engine = engine
    host, default_port = _api_host_port()
    cfg = uvicorn.Config(app, host=host, port=port or default_port,
                         log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    logger.info("REST API listening on http://%s:%d", host,
                port or default_port)
    return t


def run(port: int | None = None):
    host, default_port = _api_host_port()
    uvicorn.run(app, host=host, port=port or default_port)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(port)
