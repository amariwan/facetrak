from dataclasses import dataclass
import numpy as np


@dataclass(eq=False)
class FaceDetection:
    row: np.ndarray
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def area(self) -> int:
        return self.w * self.h
