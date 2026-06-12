"""Canvas-based button with rounded corners, hover and pressed states.

tk.Button cannot reliably style backgrounds on macOS (Aqua ignores bg),
so all FaceTrak buttons render through this widget instead.
"""
import tkinter as tk
import tkinter.font as tkfont
from typing import Callable

from .. import theme as T

# variant -> (fill, fill_hover, fill_press, text, border)
_VARIANTS = {
    "default": (T.BG_RAISED,  T.BG_OVERLAY, T.BG_INPUT,   T.TEXT_PRIMARY,   T.BORDER_SUBTLE),
    "ghost":   (T.BG_SURFACE, T.BG_RAISED,  T.BG_INPUT,   T.TEXT_SECONDARY, ""),
    "primary": (T.ACCENT_DIM, T.ACCENT,     T.ACCENT_DIM, T.TEXT_PRIMARY,   ""),
    "success": (T.BG_RAISED,  T.BG_OVERLAY, T.BG_INPUT,   T.SUCCESS,        T.BORDER_SUBTLE),
    "danger":  (T.BG_RAISED,  T.BG_OVERLAY, T.BG_INPUT,   T.DANGER,         T.BORDER_SUBTLE),
}


class Button(tk.Canvas):
    def __init__(self, parent, text: str, command: Callable | None = None,
                 variant: str = "default", width: int = 0, height: int = 28,
                 font=None, bg: str | None = None, **kw):
        self._font = font or T.FONT_LABEL
        self._text = text
        self._command = command
        self._variant = variant
        self._enabled = True
        self._hover = False
        self._pressed = False

        # Auto-width from text if not given
        if width <= 0:
            try:
                width = tkfont.Font(font=self._font).measure(text) + 26
            except tk.TclError:
                width = len(text) * 7 + 26

        super().__init__(parent, width=width, height=height,
                         bg=bg or parent.cget("bg"),
                         highlightthickness=0, cursor="hand2", **kw)
        self._bw, self._bh = width, height

        self.bind("<Enter>",           self._on_enter)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._render()

    # ── Public ────────────────────────────────────────────────────────────────

    def set_text(self, text: str):
        self._text = text
        self._render()

    def set_variant(self, variant: str):
        self._variant = variant
        self._render()

    def set_state(self, text: str | None = None, variant: str | None = None,
                  enabled: bool | None = None):
        if text is not None:
            self._text = text
        if variant is not None:
            self._variant = variant
        if enabled is not None:
            self._enabled = enabled
            self.configure(cursor="hand2" if enabled else "arrow")
        self._render()

    def set_enabled(self, enabled: bool):
        self.set_state(enabled=enabled)

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_enter(self, _):
        self._hover = True
        self._render()

    def _on_leave(self, _):
        self._hover = False
        self._pressed = False
        self._render()

    def _on_press(self, _):
        if not self._enabled:
            return
        self._pressed = True
        self._render()

    def _on_release(self, e):
        if not self._enabled:
            return
        was = self._pressed
        self._pressed = False
        self._render()
        inside = 0 <= e.x <= self._bw and 0 <= e.y <= self._bh
        if was and inside and self._command:
            self._command()

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self):
        self.delete("all")
        fill, hover, press, text_col, border = _VARIANTS.get(
            self._variant, _VARIANTS["default"])

        if not self._enabled:
            bg, fg = T.BG_SURFACE, T.TEXT_MUTED
        elif self._pressed:
            bg, fg = press, text_col
        elif self._hover:
            bg, fg = hover, text_col
        else:
            bg, fg = fill, text_col

        self._rounded(1, 1, self._bw - 1, self._bh - 1, T.RADIUS,
                      fill=bg, outline=border or bg, width=1)
        self.create_text(self._bw // 2, self._bh // 2, text=self._text,
                         fill=fg, font=self._font)

    def _rounded(self, x1, y1, x2, y2, r, **kw):
        pts = [
            x1 + r, y1,  x2 - r, y1,  x2, y1,  x2, y1 + r,
            x2, y2 - r,  x2, y2,  x2 - r, y2,  x1 + r, y2,
            x1, y2,  x1, y2 - r,  x1, y1 + r,  x1, y1,
        ]
        return self.create_polygon(pts, smooth=True, **kw)
