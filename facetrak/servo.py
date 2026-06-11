import serial
import serial.tools.list_ports
import numpy as np
from typing import Optional


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
        self.invert_pan = s["invert_pan"]
        self.invert_tilt = s["invert_tilt"]

        if self.port:
            self.connect(self.port, self.baud)

    def connect(self, port: str, baud: int = 9600) -> bool:
        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            self.port = port
            self.baud = baud
            return True
        except Exception:
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
        if not self.enabled:
            return self.pan, self.tilt
        if abs(dx) < self.dead_zone and abs(dy) < self.dead_zone:
            return self.pan, self.tilt

        po = (dx / (fw / 2)) * 60
        to = (dy / (fh / 2)) * 45
        if self.invert_pan:
            po = -po
        if self.invert_tilt:
            to = -to

        self.pan_target = np.clip(90 + po, self.pan_min, self.pan_max)
        self.tilt_target = np.clip(90 + to, self.tilt_min, self.tilt_max)

        dp = np.clip(self.pan_target - self.pan, -self.max_step, self.max_step)
        dt = np.clip(self.tilt_target - self.tilt, -self.max_step, self.max_step)
        self.pan += dp * self.smooth + dp * (1 - self.smooth)
        self.tilt += dt * self.smooth + dt * (1 - self.smooth)
        self.pan = np.clip(self.pan, self.pan_min, self.pan_max)
        self.tilt = np.clip(self.tilt, self.tilt_min, self.tilt_max)

        if self.connected:
            try:
                self.ser.write(
                    f"P{int(self.pan):03d}T{int(self.tilt):03d}\n".encode())
            except Exception:
                pass
        return self.pan, self.tilt

    @staticmethod
    def list_ports() -> list[str]:
        return [p.device for p in serial.tools.list_ports.comports()]
