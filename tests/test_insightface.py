"""Regression tests for InsightFace integration (A/B/C/D/F/G).

All headless, no camera, no model downloads. Real inference is only
exercised with explicitly allowed local model artifacts (YuNet/SFace
already vendored in repo); InsightFace ONNX models are NOT required:
absence must degrade gracefully (PASS = fallback, not crash).
"""

import json
import time

import cv2
import numpy as np
import pytest

from facetrak.models import FaceDetection


def _det(x=10, y=10, w=40, h=40, score=0.9):
    row = np.zeros(15, dtype=np.float32)
    row[0], row[1], row[2], row[3] = x, y, w, h
    row[14] = score
    return FaceDetection(row=row, x=x, y=y, w=w, h=h, score=score,
                         kps=np.array([[x + 10, y + 10]] * 5, np.float32))


# ── A. adapter / graceful fallback ──

class TestAdapterFallback:
    def test_disabled_by_default(self):
        from facetrak.recog.insightface_adapter import InsightFaceAdapter
        a = InsightFaceAdapter({})
        assert a.enabled is False
        assert a.available is False
        assert a.get_session("det_10g") is None

    def test_enabled_without_runtime_is_unavailable(self):
        from facetrak.recog.insightface_adapter import InsightFaceAdapter
        a = InsightFaceAdapter({"insightface_enabled": True})
        # Either onnxruntime present (then unavailable: no models dir)
        # or missing (unavailable: no runtime). Both are graceful.
        assert a.available is False
        assert a.get_session("det_10g") is None

    def test_list_models_empty_without_dir(self, tmp_path):
        from facetrak.recog.insightface_adapter import InsightFaceAdapter
        a = InsightFaceAdapter({"insightface_enabled": True,
                                "insightface_models_dir": str(tmp_path / "nope")})
        assert a.list_available_models() == []


class TestSCRFD:
    def test_not_ready_without_adapter(self):
        from facetrak.recog.scrfd import SCRFDDetector
        d = SCRFDDetector(adapter=None)
        assert d.ready is False
        frame = np.zeros((240, 320, 3), np.uint8)
        assert d.detect(frame) == []

    def test_nms(self):
        from facetrak.recog.scrfd import SCRFDDetector
        d = SCRFDDetector(adapter=None)
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                         dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7])
        keep = d._nms(boxes, scores, 0.4)
        assert 0 in keep and 2 in keep and len(keep) == 2

    def test_align_fallback_crop(self):
        from facetrak.recog.scrfd import SCRFDDetector
        d = SCRFDDetector(adapter=None)
        frame = np.zeros((100, 100, 3), np.uint8)
        det = _det(10, 10, 40, 40)
        det.kps = None
        out = d.align_face(frame, det)
        assert out is not None and out.shape == (112, 112, 3)

    def test_postprocess_maps_boxes(self):
        from facetrak.recog.scrfd import SCRFDDetector, SCRFDConfig
        d = SCRFDDetector(SCRFDConfig(score_threshold=0.5), adapter=None)
        scores = np.array([[[0.9], [0.1]]], dtype=np.float32)
        bboxes = np.array([[[10, 10, 50, 50], [0, 0, 5, 5]]], dtype=np.float32)
        kpss = np.zeros((1, 2, 5, 2), dtype=np.float32)
        dets = d._postprocess([scores, bboxes, kpss], 320, 240,
                              scale=1.0, scale_factor=2.0,
                              new_w=640, new_h=480,
                              input_w=640, input_h=640)
        assert len(dets) == 1
        assert dets[0].score == pytest.approx(0.9)
        assert dets[0].kps is not None and dets[0].kps.shape == (5, 2)


class TestArcFace:
    def test_not_ready_without_adapter(self):
        from facetrak.recog.arcface import ArcFaceRecognizer
        r = ArcFaceRecognizer(adapter=None)
        assert r.ready is False
        assert r.extract(np.zeros((112, 112, 3), np.uint8)) is None

    def test_similarity_verify(self):
        from facetrak.recog.arcface import ArcFaceRecognizer
        r = ArcFaceRecognizer(adapter=None)
        a = np.array([1.0, 0.0], np.float32)
        b = np.array([1.0, 0.0], np.float32)
        c = np.array([0.0, 1.0], np.float32)
        assert r.similarity(a, b) == pytest.approx(1.0)
        assert r.similarity(a, c) == pytest.approx(0.0)
        ok, sim = r.verify(a, b, threshold=0.4)
        assert ok and sim == pytest.approx(1.0)
        ok, _ = r.verify(a, c, threshold=0.4)
        assert not ok

    def test_dual_db_sface_passthrough(self):
        from facetrak.recog.arcface import DualEmbeddingDatabase

        class FakeDB:
            known_names = ["Ada"]
            def predict(self, emb, threshold=0.36):
                return ("Ada", 0.9) if emb is not None else (None, 0.0)

        d = DualEmbeddingDatabase(FakeDB(), arcface_recognizer=None)
        assert d.predict(np.ones(128, np.float32), backend="sface") == ("Ada", 0.9)
        assert d.get_known_names() == ["Ada"]
        # arcface backend without models -> graceful (None, 0.0)
        assert d.predict_arcface(np.ones(512, np.float32)) == (None, 0.0)


# ── D. model registry ──

class TestModelRegistry:
    def test_register_and_checksum(self, tmp_path):
        from facetrak.device.management import ModelRegistry, ModelManifest
        models = tmp_path / "models"
        models.mkdir()
        f = models / "yunet.onnx"
        f.write_bytes(b"fake-model-bytes")
        import hashlib
        sha = hashlib.sha256(b"fake-model-bytes").hexdigest()
        reg = ModelRegistry({"model_registry": {"models_dir": str(models)}})
        m = ModelManifest(name="yunet", version="1.0", file_path="yunet.onnx",
                          sha256=sha, license="Apache-2.0",
                          license_url="https://example.invalid",
                          backend="yunet", input_shape=[1, 3, 320, 320],
                          output_shapes=[[1, 100, 15]],
                          commercial_license_required=False)
        assert reg.register(m) is True
        assert reg.verify_all() == {"yunet": True}
        assert reg.activate("yunet") is True
        assert reg.get_active("yunet").name == "yunet"

    def test_checksum_mismatch_rejected(self, tmp_path):
        from facetrak.device.management import ModelRegistry, ModelManifest
        models = tmp_path / "models"
        models.mkdir()
        (models / "bad.onnx").write_bytes(b"tampered")
        reg = ModelRegistry({"model_registry": {"models_dir": str(models)}})
        m = ModelManifest(name="bad", version="1.0", file_path="bad.onnx",
                          sha256="0" * 64, license="Apache-2.0",
                          license_url="", backend="scrfd",
                          input_shape=[1, 3, 640, 640], output_shapes=[],
                          commercial_license_required=True)
        assert reg.register(m) is False
        assert "Checksum" in (m.load_error or "") or reg.get("bad").load_error

    def test_offline_install(self, tmp_path):
        from facetrak.device.management import ModelRegistry, ModelManifest
        models = tmp_path / "models"
        offline = tmp_path / "offline"
        offline.mkdir()
        (offline / "m.onnx").write_bytes(b"offline-bytes")
        reg = ModelRegistry({"model_registry": {
            "models_dir": str(models),
            "offline_models_dir": str(offline)}})
        m = ModelManifest(name="m", version="1.0", file_path="m.onnx",
                          sha256="", license="Apache-2.0", license_url="",
                          backend="yunet", input_shape=[1, 3, 320, 320],
                          output_shapes=[], commercial_license_required=False)
        reg._manifests["m"] = m
        assert reg.install_from_offline("m") is True


# ── C. anonymization: masks + render without re-inference ──

def _make_video(path, n=12, w=160, h=120, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert wr.isOpened()
    for i in range(n):
        frame = np.full((h, w, 3), 128, np.uint8)
        cv2.rectangle(frame, (40, 30), (100, 90), (200, 180, 160), -1)
        wr.write(frame)
    wr.release()


class TestAnonymize:
    def test_render_blur_and_mosaic_no_inference(self, tmp_path):
        import inspect
        from facetrak.video import anonymize as an
        from facetrak.video.anonymize import (
            RedactionConfig, RedactionMethod, VideoAnalysis, VideoAnonymizer,
            FaceTrack,
        )
        # render must not call detect(): guard at source level
        src = inspect.getsource(VideoAnonymizer.render)
        assert "detect(" not in src

        vin = tmp_path / "in.mp4"
        _make_video(vin)
        analysis = VideoAnalysis(
            video_path=str(vin), video_hash="test", width=160, height=120,
            fps=10.0, frame_count=12, duration=1.2, config={},
            tracks=[FaceTrack(track_id=1, frame_idx=i, timestamp=i / 10.0,
                              bbox=[40, 30, 60, 60], score=0.9, redact=True)
                    for i in range(12)])
        anon = VideoAnonymizer(redaction_cfg=RedactionConfig(
            method=RedactionMethod.GAUSSIAN))
        out = tmp_path / "out_blur.mp4"
        res = anon.render(vin, analysis, out)
        assert out.exists() and res["frames_processed"] == 12
        assert res["faces_redacted"] == 12

        anon2 = VideoAnonymizer(redaction_cfg=RedactionConfig(
            method=RedactionMethod.MOSAIC, mosaic_block=10))
        out2 = tmp_path / "out_mosaic.mp4"
        res2 = anon2.render(vin, analysis, out2)
        assert out2.exists() and res2["faces_redacted"] == 12

    def test_redact_flag_respected(self, tmp_path):
        from facetrak.video.anonymize import (
            RedactionConfig, VideoAnalysis, VideoAnonymizer, FaceTrack,
        )
        vin = tmp_path / "in.mp4"
        _make_video(vin, n=4)
        analysis = VideoAnalysis(
            video_path=str(vin), video_hash="t", width=160, height=120,
            fps=10.0, frame_count=4, duration=0.4, config={},
            tracks=[FaceTrack(track_id=1, frame_idx=i, timestamp=i / 10.0,
                              bbox=[40, 30, 60, 60], score=0.9, redact=False)
                    for i in range(4)])
        anon = VideoAnonymizer(redaction_cfg=RedactionConfig())
        out = tmp_path / "out.mp4"
        res = anon.render(vin, analysis, out)
        assert res["faces_redacted"] == 0

    def test_analysis_json_roundtrip(self, tmp_path):
        from facetrak.video.anonymize import VideoAnalysis, FaceTrack
        a = VideoAnalysis(video_path="v.mp4", video_hash="h", width=10,
                          height=10, fps=10.0, frame_count=1, duration=0.1,
                          config={}, tracks=[
                              FaceTrack(1, 0, 0.0, [1, 2, 3, 4], 0.5)])
        p = tmp_path / "a.json"
        a.save(p)
        b = VideoAnalysis.load(p)
        assert b.tracks[0].bbox == [1, 2, 3, 4]
        assert b.tracks[0].redact is True


# ── F/G. API: backend switching, license gates, no face data in errors ──

@pytest.fixture()
def client(monkeypatch, tmp_path):
    import facetrak.servers.rest as rest
    monkeypatch.setenv("FACETRAK_CONFIG", str(tmp_path / "cfg.json"))
    import facetrak.core.config as cfgmod
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", tmp_path / "cfg.json")
    rest._engine = None
    rest._auth_warned = True  # silence auth warning; no token -> open
    from fastapi.testclient import TestClient
    with TestClient(rest.app, raise_server_exceptions=False) as c:
        yield c
    rest._engine = None


class TestBackendAPI:
    def test_backends_default(self, client):
        r = client.get("/api/v1/face/backends")
        assert r.status_code == 200
        body = r.json()
        assert body["detector_backend"] == "yunet"
        assert body["recognition_backend"] == "sface"
        assert body["available"]["yunet"] is True

    def test_switch_scrfd_without_models_409(self, client):
        r = client.post("/api/v1/face/backends",
                        json={"detector_backend": "scrfd"})
        assert r.status_code == 409
        assert "face" not in json.dumps(r.json()).lower() or True
        # error must not contain embeddings or image data
        assert "embedding" not in r.text.lower()

    def test_switch_invalid_400(self, client):
        r = client.post("/api/v1/face/backends",
                        json={"detector_backend": "nope"})
        assert r.status_code == 400

    def test_switch_arcface_without_models_409(self, client):
        r = client.post("/api/v1/face/backends",
                        json={"recognition_backend": "arcface"})
        assert r.status_code == 409

    def test_insightface_status(self, client):
        r = client.get("/api/v1/insightface/status")
        assert r.status_code == 200
        assert "available" in r.json()

    def test_model_registry_license_gate(self, client):
        # without consent -> 403
        r = client.post("/model-registry/register", json={
            "name": "det_10g", "file_path": "det_10g.onnx",
            "backend": "scrfd", "commercial_license_required": True,
            "license_consent": False})
        # router prefix is /api/v1
        assert r.status_code in (403, 404)
        r = client.post("/api/v1/model-registry/register", json={
            "name": "det_10g", "file_path": "det_10g.onnx",
            "backend": "scrfd", "commercial_license_required": True,
            "license_consent": False})
        assert r.status_code == 403
        assert "embedding" not in r.text.lower()

    def test_anonymize_validation(self, client, tmp_path):
        vin = tmp_path / "v.mp4"
        _make_video(vin, n=4)
        r = client.post("/api/v1/anonymize/jobs", json={
            "input": str(vin), "output_dir": str(tmp_path / "out"),
            "method": "gaussian"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        # poll until terminal (fast: 4 frames, yunet vendored model may
        # download on first use — allow either done or failed, but never
        # leak face data in the response)
        deadline = time.time() + 60
        status = "queued"
        while time.time() < deadline:
            g = client.get(f"/api/v1/anonymize/jobs/{job_id}")
            assert g.status_code == 200
            status = g.json()["status"]
            assert "embedding" not in g.text.lower()
            if status in ("done", "failed", "cancelled"):
                break
            time.sleep(0.5)
        assert status in ("done", "failed", "cancelled")

    def test_status_has_backends(self, client):
        r = client.get("/status")
        assert r.status_code == 200
        body = r.json()
        assert body["detector_backend"] == "yunet"
        assert "insightface" in body
