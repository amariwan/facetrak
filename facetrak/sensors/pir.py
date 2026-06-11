"""PIR (passive infrared) motion sensor via Raspberry Pi GPIO.

Monitors a GPIO pin connected to an HC-SR501 or similar PIR sensor.
On non-Pi hardware the sensor self-disables gracefully — the engine
continues without it.

Install: pip install RPi.GPIO   (only on Pi)
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

_DEBOUNCE_S = 0.5   # ignore re-triggers within this window


@dataclass
class MotionEvent:
    pin: int
    timestamp: float = field(default_factory=time.time)
    active: bool = True   # True = motion started, False = motion ended


class PIRSensor:
    """Monitors a GPIO pin for PIR motion events.

    Calls the optional `on_motion` callback from the GPIO interrupt thread.
    Safe to call poll() from the main loop instead.

    Usage (Pi):
        pir = PIRSensor(pin=17)
        pir.start()
        event = pir.poll()   # non-blocking
        pir.stop()

    Usage (non-Pi / simulation):
        pir = PIRSensor(pin=17)
        pir.start()          # logs warning, stays disabled
        assert not pir.enabled
    """

    def __init__(self, pin: int = 17,
                 on_motion: Callable[[MotionEvent], None] | None = None,
                 debounce_s: float = _DEBOUNCE_S):
        self._pin       = pin
        self._callback  = on_motion
        self._debounce  = debounce_s
        self._gpio      = None
        self._enabled   = False
        self._last_time = 0.0
        self._events: list[MotionEvent] = []
        self._lock = threading.Lock()

    def start(self) -> bool:
        try:
            import RPi.GPIO as GPIO  # type: ignore
        except ImportError:
            logger.warning(
                "RPi.GPIO not available — PIR sensor disabled. "
                "Install with: pip install RPi.GPIO  (Raspberry Pi only)"
            )
            return False

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.add_event_detect(
                self._pin,
                GPIO.BOTH,
                callback=self._on_gpio,
                bouncetime=int(self._debounce * 1000),
            )
            self._gpio = GPIO
            self._enabled = True
            logger.info("PIRSensor started on GPIO pin %d", self._pin)
            return True
        except Exception as exc:
            logger.error("PIRSensor GPIO setup failed: %s", exc)
            return False

    def _on_gpio(self, channel: int) -> None:
        now = time.time()
        if now - self._last_time < self._debounce:
            return
        self._last_time = now
        active = bool(self._gpio.input(channel))
        event = MotionEvent(pin=channel, timestamp=now, active=active)
        with self._lock:
            self._events.append(event)
        if self._callback:
            try:
                self._callback(event)
            except Exception as exc:
                logger.warning("PIR callback error: %s", exc)

    def poll(self) -> MotionEvent | None:
        """Non-blocking: return oldest pending event or None."""
        with self._lock:
            return self._events.pop(0) if self._events else None

    def poll_all(self) -> list[MotionEvent]:
        with self._lock:
            events, self._events = self._events, []
        return events

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stop(self) -> None:
        if self._gpio and self._enabled:
            try:
                self._gpio.remove_event_detect(self._pin)
                self._gpio.cleanup(self._pin)
            except Exception:
                pass
        self._enabled = False
