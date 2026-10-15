"""Headless production entrypoint: camera + REST API, no GUI.

Designed for systemd and Docker (PID 1): installs SIGTERM/SIGINT
handlers, releases camera/GPIO on exit, returns nonzero when the
camera cannot start. `python -m facetrak.servers.headless --check`
boots everything except the camera for smoke tests.
"""

import argparse
import logging
import sys

from facetrak.core.engine import FaceEngine
from facetrak.platform.shutdown import ShutdownManager

logger = logging.getLogger(__name__)


def build_engine() -> FaceEngine:
    return FaceEngine()


def run(check_only: bool = False) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    engine = build_engine()
    if check_only:
        from facetrak.storage import db

        db.init()
        stats = engine.perf_stats()
        logger.info("check ok (camera not started): %s", stats)
        return 0
    if not engine.start():
        logger.error("Camera failed to start, exiting")
        return 2
    from facetrak.servers import rest

    rest.start_api_thread(engine)
    if engine.mqtt.enabled:
        engine.mqtt.start()
    shutdown = ShutdownManager()

    def _on_stop():
        engine.stop()

    shutdown.install(_on_stop)
    try:
        # Pacing: USB read() blocks per frame, but RTSP read() is
        # non-blocking — the small wait prevents a busy spin there.
        while not shutdown.wait(timeout=0.005):
            if not engine.running:
                break
            engine.step()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    purged = engine.purge_retention()
    logger.info("Shutdown complete, retention purged: %s", purged)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FaceTrak headless server")
    ap.add_argument(
        "--check",
        action="store_true",
        help="boot without camera, then exit 0 (smoke test)",
    )
    args = ap.parse_args(argv)
    return run(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
