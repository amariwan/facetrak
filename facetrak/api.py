"""FastAPI REST server for FaceTrak.

Runs in a daemon thread alongside the main UI or MCP server. Start with:
    from facetrak.api import start_api_thread
    start_api_thread(engine, port=8765)

Or standalone:
    python -m facetrak.api
"""
import logging
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import db
from .engine import FaceEngine

logger = logging.getLogger(__name__)

_engine: FaceEngine | None = None


def _get() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine


@asynccontextmanager
async def _lifespan(app: FastAPI):
    db.init()
    yield


app = FastAPI(title="FaceTrak", version="1.0", lifespan=_lifespan)


@app.get("/status")
def status():
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
    }


@app.get("/faces")
def faces():
    e = _get()
    if not e.running:
        raise HTTPException(503, "Camera not running")
    return e.live_faces()


@app.get("/face/analysis")
def analysis():
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
def presence(limit: int = 50):
    return db.query_presence(limit)


@app.get("/crowd")
def crowd():
    e = _get()
    summary = db.crowd_summary()
    return {"current": e.crowd.current, **summary}


@app.get("/emotions")
def emotions(limit: int = 200):
    return db.query_emotions(limit)


@app.post("/camera/start")
def camera_start():
    e = _get()
    if e.running:
        return {"ok": True, "message": "already running"}
    if not e.start():
        raise HTTPException(500, "Failed to start camera")
    return {"ok": True}


@app.post("/camera/stop")
def camera_stop():
    e = _get()
    e.stop()
    return {"ok": True}


@app.post("/snapshot")
def snapshot():
    e = _get()
    if not e.running:
        raise HTTPException(503, "Camera not running")
    path = e.snapshot()
    if not path:
        raise HTTPException(404, "No frame available")
    return FileResponse(path, media_type="image/png", filename="snapshot.png")


@app.post("/blur/{name}")
def set_blur(name: str, blur: bool = True):
    e = _get()
    e.set_blur_person(name, blur)
    return {"ok": True, "name": name, "blur": blur}


@app.get("/export/crowd")
def export_crowd():
    e = _get()
    path = e.export_crowd_csv()
    if not path:
        raise HTTPException(404, "No crowd data")
    return FileResponse(path, media_type="text/csv", filename="crowd_stats.csv")


@app.get("/export/emotions")
def export_emotions():
    from .stats import EmotionTimeline
    path = EmotionTimeline.export_csv()
    if not path:
        raise HTTPException(404, "No emotion data")
    return FileResponse(path, media_type="text/csv", filename="emotion_log.csv")


def start_api_thread(engine: FaceEngine, port: int = 8765):
    global _engine
    _engine = engine
    cfg = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    logger.info("REST API listening on http://0.0.0.0:%d", port)
    return t


def run(port: int = 8765):
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run(port)
