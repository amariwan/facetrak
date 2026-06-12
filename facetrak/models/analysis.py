from dataclasses import dataclass, field


@dataclass
class FaceMetrics:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    smile: float = 0.0
    mouth_open: float = 0.0
    brow_raise: float = 0.0
    eye_left: float = 1.0
    eye_right: float = 1.0
    emotion: str = "neutral"
    attentive: bool = False
    blendshapes: dict[str, float] = field(default_factory=dict)
    gaze_h: float = 0.0
    gaze_v: float = 0.0
    gaze_label: str = "centre"

    @property
    def eyes_closed(self) -> bool:
        return self.eye_left < 0.5 and self.eye_right < 0.5
