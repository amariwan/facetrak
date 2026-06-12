from dataclasses import dataclass, field

BLINK_REQUIRED = 2
HEAD_TURN_DEG = 12.0
_YAW_WINDOW = 20


@dataclass
class LivenessChecker:
    blinks: int = 0
    _max_spread: float = 0.0
    _yaw_buf: list = field(default_factory=list)
    _was_closed: bool = False

    def update(self, eyes_closed: bool, yaw: float):
        if eyes_closed and not self._was_closed:
            self.blinks += 1
        self._was_closed = eyes_closed

        self._yaw_buf.append(yaw)
        if len(self._yaw_buf) > _YAW_WINDOW:
            self._yaw_buf.pop(0)
        spread = max(self._yaw_buf) - min(self._yaw_buf)
        self._max_spread = max(self._max_spread, spread)

    @property
    def passed(self) -> bool:
        return (self.blinks >= BLINK_REQUIRED
                and self._max_spread >= HEAD_TURN_DEG)

    @property
    def status_line(self) -> str:
        b = ("✓" if self.blinks >= BLINK_REQUIRED
             else f"{self.blinks}/{BLINK_REQUIRED}")
        t = ("✓" if self._max_spread >= HEAD_TURN_DEG else "move head")
        return f"Blinks: {b}   Head: {t}"

    def reset(self):
        self.blinks = 0
        self._max_spread = 0.0
        self._yaw_buf.clear()
        self._was_closed = False
