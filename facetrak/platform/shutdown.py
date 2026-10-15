"""Graceful shutdown: SIGTERM/SIGINT → stop callback exactly once.

Works under systemd, Docker (PID 1 with proper signal proxy), and
interactive Ctrl-C. Windows/macOS degrade to KeyboardInterrupt paths.
"""

import logging
import signal
import threading

logger = logging.getLogger(__name__)


class ShutdownManager:
    def __init__(self):
        self._event = threading.Event()
        self._fired = False
        self._lock = threading.Lock()

    def install(self, on_stop) -> None:
        def _handler(signum, _frame):
            with self._lock:
                if self._fired:
                    return
                self._fired = True
            logger.info("Received signal %d, shutting down", signum)
            try:
                on_stop()
            finally:
                self._event.set()

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except (OSError, ValueError, RuntimeError) as exc:
            # Non-main thread or unsupported platform: caller falls back
            # to KeyboardInterrupt handling.
            logger.debug("Signal install skipped: %s", exc)

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def trigger(self):
        """Manual shutdown (tests, API-driven stop)."""
        with self._lock:
            self._fired = True
        self._event.set()
