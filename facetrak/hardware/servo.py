import logging
from typing import Optional

import numpy as np
import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)


class PanTiltController:
    def __init__(self, cfg: dict):
        s = cfg["servo"]
        self.port = s["port"]
        self.baud = s["baud"]
        self.ser: Optional[serial.Serial] = None
        self.enabled = False

        self.pan = 90.0
        self.tilt = 90.0
        self.pan_target = 90.0
        self.tilt_target = 90.0

        self.pan_min = s["pan_min"]
        self.pan_max = s["pan_max"]
        self.tilt_min = s["tilt_min"]
        self.tilt_max = s["tilt_max"]
        self.dead_zone = s["dead_zone"]
        self.smooth = s["smooth"]
        self.max_step = s["max_step"]
        self.max_speed_dps = float(s.get("max_speed_dps", 90.0))
        self.reacquire_frames = int(s.get("reacquire_frames", 30))
        self.pan_offset = float(s.get("pan_offset", 0.0))
        self.tilt_offset = float(s.get("tilt_offset", 0.0))
        self.invert_pan = s["invert_pan"]
        self.invert_tilt = s["invert_tilt"]
        self._last_t: float | None = None
        self._lost_frames = 0

        if self.port:
            self.connect(self.port, self.baud)

    def connect(self, port: str, baud: int = 9600) -> bool:
        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            self.port = port
            self.baud = baud
            return True
        except Exception as exc:
            logger.debug("Servo connect failed on %s: %s", port, exc)
            self.ser = None
            return False

    def disconnect(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    @property
    def connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def update(self, dx: int, dy: int, fw: int, fh: int) -> tuple[float, float]:
        import time
        if not self.enabled:
            return self.pan, self.tilt
        if abs(dx) < self.dead_zone and abs(dy) < self.dead_zone:
            return self.pan, self.tilt
        if fw <= 0 or fh <= 0:
            return self.pan, self.tilt

        self._lost_frames = 0
        po = (dx / (fw / 2)) * 60
        to = (dy / (fh / 2)) * 45
        if self.invert_pan:
            po = -po
        if self.invert_tilt:
            to = -to

        self.pan_target = float(np.clip(90 + po, self.pan_min, self.pan_max))
        self.tilt_target = float(np.clip(90 + to, self.tilt_min, self.tilt_max))

        now = time.monotonic()
        dt = now - self._last_t if self._last_t else 1 / 30
        self._last_t = now
        # Velocity cap: degrees allowed this frame by max_speed_dps,
        # combined with the existing smoothing/max_step profile.
        vel_cap = max(self.max_speed_dps * min(dt, 0.5), 1e-6)

        self.pan += float(np.clip((self.pan_target - self.pan) * self.smooth,
                                  -min(self.max_step, vel_cap),
                                  min(self.max_step, vel_cap)))
        self.tilt += float(np.clip((self.tilt_target - self.tilt) * self.smooth,
                                   -min(self.max_step, vel_cap),
                                   min(self.max_step, vel_cap)))
        self.pan = float(np.clip(self.pan, self.pan_min, self.pan_max))
        self.tilt = float(np.clip(self.tilt, self.tilt_min, self.tilt_max))

        self._send()
        return self.pan, self.tilt

    def note_target_lost(self) -> tuple[float, float] | None:
        """Call once per frame without a servo target.

        After `reacquire_frames` target-less frames, eases back to center
        so the camera re-acquires new faces instead of staring at a wall.
        Returns the new (pan, tilt) when recentering, else None.
        """
        if not self.enabled:
            return None
        self._lost_frames += 1
        if self._lost_frames < self.reacquire_frames:
            return None
        self.pan_target, self.tilt_target = 90.0, 90.0
        self.pan += (90.0 - self.pan) * self.smooth
        self.tilt += (90.0 - self.tilt) * self.smooth
        self._send()
        return self.pan, self.tilt

    def calibrate(self, pan_offset: float, tilt_offset: float) -> dict:
        """Store static calibration offsets (degrees, applied at send)."""
        self.pan_offset = float(np.clip(pan_offset, -10.0, 10.0))
        self.tilt_offset = float(np.clip(tilt_offset, -10.0, 10.0))
        return {"pan_offset": self.pan_offset,
                "tilt_offset": self.tilt_offset}

    def move_to(self, pan: float, tilt: float) -> tuple[float, float]:
        self.pan = float(np.clip(pan, self.pan_min, self.pan_max))
        self.tilt = float(np.clip(tilt, self.tilt_min, self.tilt_max))
        self.pan_target = self.pan
        self.tilt_target = self.tilt
        self._send()
        return self.pan, self.tilt

    def _send(self):
        if not self.connected:
            return
        try:
            pan = float(np.clip(self.pan + self.pan_offset,
                                self.pan_min, self.pan_max))
            tilt = float(np.clip(self.tilt + self.tilt_offset,
                                 self.tilt_min, self.tilt_max))
            self.ser.write(
                f"P{int(pan):03d}T{int(tilt):03d}\n".encode())
        except (serial.SerialException, OSError):
            self.disconnect()

    @staticmethod
    def list_ports() -> list[str]:
        return [p.device for p in serial.tools.list_ports.comports()]
