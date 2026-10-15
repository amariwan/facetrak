"""Linux-first platform layer: detection, backends, metrics, shutdown."""

import sys
from pathlib import Path

import pytest

from facetrak.platform import IS_LINUX, IS_MAC, cv_backend, platform_summary
from facetrak.platform.metrics import (
    cpu_temp_c,
    load_avg,
    mem_info,
    self_rss_mb,
    system_status,
)
from facetrak.platform.resources import ResourceGuard
from facetrak.platform.shutdown import ShutdownManager

ROOT = Path(__file__).resolve().parent.parent


def test_platform_summary_shape():
    s = platform_summary()
    assert s["is_linux"] == IS_LINUX
    assert s["is_mac"] == IS_MAC
    assert isinstance(s["headless"], bool)
    assert s["machine"]  # e.g. arm64 / x86_64


def test_is_headless_respects_env(monkeypatch):
    import facetrak.platform as plat

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    if sys.platform.startswith("linux"):
        assert plat.is_headless()
        monkeypatch.setenv("DISPLAY", ":0")
        assert not plat.is_headless()
    else:
        assert plat.is_headless() is False


def test_cv_backend_matches_platform():
    import cv2

    b = cv_backend()
    if sys.platform.startswith("linux"):
        assert b == cv2.CAP_V4L2
    elif sys.platform == "darwin":
        assert b == cv2.CAP_AVFOUNDATION
    else:
        assert b is None


def test_usb_open_uses_platform_backend(monkeypatch):
    import cv2

    import facetrak.camera.usb as usbmod

    seen = {}

    class FakeCap:
        def __init__(self, *a):
            seen["args"] = a

        def set(self, *a):
            pass

        def get(self, *a):
            return 640

    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
    usbmod.USBCamera(0)
    import facetrak.platform as plat

    if plat.cv_backend() is not None:
        assert seen["args"] == (0, plat.cv_backend())
    else:
        assert seen["args"] == (0,)


def test_metrics_parse_fixtures(tmp_path):
    mem = tmp_path / "meminfo"
    mem.write_text("MemTotal:        8192000 kB\nMemAvailable:    4096000 kB\n")
    info = mem_info(str(mem))
    assert info["memtotal"] == 8192000
    assert mem_info(str(tmp_path / "missing")) is None

    temp = tmp_path / "temp"
    temp.write_text("51234\n")
    assert cpu_temp_c(str(temp)) == pytest.approx(51.234)
    assert cpu_temp_c(str(tmp_path / "missing")) is None


def test_metrics_never_raise_on_this_host():
    assert load_avg() is None or len(load_avg()) == 3
    rss = self_rss_mb()
    assert rss is None or rss > 0
    status = system_status()
    assert "ts" in status and "self_rss_mb" in status


def test_resource_guard_disabled_by_default():
    g = ResourceGuard(0)
    assert not g.enabled
    assert g.check()["over"] is False


def test_resource_guard_warns_once_and_gc(tmp_path, caplog):
    g = ResourceGuard(0.0001)  # impossibly small → always over
    with caplog.at_level("WARNING", logger="facetrak.platform.resources"):
        first = g.check()
    assert first["over"] is True
    assert "exceeds budget" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="facetrak.platform.resources"):
        g.check()
    assert "exceeds budget" not in caplog.text  # no spam


def test_shutdown_manager_install_and_trigger():
    fired = []
    mgr = ShutdownManager()
    mgr.install(lambda: fired.append(True))
    import signal

    # Simulate delivery without killing the test process:
    mgr.trigger()
    assert mgr.wait(timeout=1)
    # Direct handler path via installed SIGTERM handler is process-global;
    # at least verify install did not raise and trigger works.
    assert signal.getsignal(signal.SIGTERM) is not None


def test_no_tkinter_in_production_imports():
    # Static: no tkinter import outside viz/ + __main__ (GUI-only code).
    # (On macOS dev machines tkinter may still appear transitively via
    # mediapipe→matplotlib; on headless Linux it does not.)
    import re

    offenders = []
    for path in list(ROOT.glob("facetrak/**/*.py")):
        if "/viz/" in path.as_posix() or path.name == "__main__.py":
            continue
        text = path.read_text()
        if re.search(r"^\s*(import tkinter|from tkinter)", text, re.MULTILINE):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_gui_package_not_loaded_by_headless():
    # overlay/heatmap are pure cv2/numpy and may load; the tkinter GUI
    # modules (ui, simulation, components) must not.
    import subprocess

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import facetrak.servers.rest, facetrak.servers.mcp, "
            "facetrak.servers.headless; "
            "import sys; "
            "print(sorted(m for m in sys.modules if m in "
            "('facetrak.viz.ui', 'facetrak.viz.simulation', "
            "'facetrak.viz.components')))",
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
