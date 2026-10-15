"""Import-light viz package: overlay/heatmap are pure cv2/numpy and
safe for headless production; the Tkinter GUI (ui, simulation,
components) must be imported explicitly and never via this package."""
from .heatmap import FaceHeatmap
from .overlay import draw_hud, draw_overlay, draw_track

__all__ = [
    "draw_overlay", "draw_track", "draw_hud",
    "FaceHeatmap",
]
