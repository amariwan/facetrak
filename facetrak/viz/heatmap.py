import cv2
import numpy as np


class FaceHeatmap:
    def __init__(self, decay: float = 0.992, radius: int = 40):
        self._map: np.ndarray | None = None
        self._decay = decay
        self._radius = radius

    def update(self, centers: list[tuple[int, int]],
               shape: tuple[int, int]):
        h, w = shape
        if self._map is None or self._map.shape != (h, w):
            self._map = np.zeros((h, w), np.float32)
        self._map *= self._decay
        for cx, cy in centers:
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(self._map, (cx, cy), self._radius, 1.0, -1)

    def overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        if self._map is None or not self._map.any() or self._map.max() < 1e-6:
            return frame
        norm = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_TURBO)
        mask = norm > 5
        result = frame.copy()
        blended = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
        result[mask] = blended[mask]
        return result

    def reset(self):
        self._map = None
