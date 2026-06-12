"""Top command bar component."""
import tkinter as tk
from tkinter import ttk
from typing import Callable

from .. import theme as T
from .button import Button


def _sep(parent) -> ttk.Separator:
    return ttk.Separator(parent, orient=tk.VERTICAL)


class CommandBar(tk.Frame):
    """Single-row toolbar with grouped actions."""

    HEIGHT = 48

    def __init__(self, parent, callbacks: dict, **kw):
        super().__init__(parent, bg=T.BG_SURFACE, height=self.HEIGHT, **kw)
        self.grid_propagate(False)
        self._cb = callbacks
        self._build()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_running(self, running: bool):
        if running:
            self._start_btn.set_state(text=f"{T.ICON_STOP}  Stop", variant="danger")
        else:
            self._start_btn.set_state(text=f"{T.ICON_PLAY}  Start", variant="success")
            self._rec_btn.set_state(text=f"{T.ICON_REC}  Record",
                                    variant="default", enabled=False)

    def set_recording(self, recording: bool):
        if recording:
            self._rec_btn.set_state(text=f"{T.ICON_REC}  Stop Rec",
                                    variant="danger", enabled=True)
        else:
            self._rec_btn.set_state(text=f"{T.ICON_REC}  Record",
                                    variant="default", enabled=True)

    def set_cam_values(self, values: list, active_label: str):
        self._cam_cb["values"] = values
        if active_label:
            self._cam_var.set(active_label)

    @property
    def target_var(self) -> tk.StringVar:
        return self._srv_var

    @property
    def cam_cb(self) -> ttk.Combobox:
        return self._cam_cb

    @property
    def cam_var(self) -> tk.StringVar:
        return self._cam_var

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        # Bottom border
        tk.Frame(self, bg=T.BORDER_SUBTLE, height=1).place(
            relx=0, rely=1.0, anchor="sw", relwidth=1.0
        )

        pad = dict(padx=4, pady=0)
        col = 0

        # ── Logo ──────────────────────────────────────────────────────────────
        logo = tk.Frame(self, bg=T.BG_SURFACE)
        logo.grid(row=0, column=col, padx=(16, 8), pady=0); col += 1

        logo_dot = tk.Canvas(logo, width=8, height=8, bg=T.BG_SURFACE,
                              highlightthickness=0)
        logo_dot.pack(side=tk.LEFT, padx=(0, 6))
        logo_dot.create_oval(1, 1, 7, 7, fill=T.ACCENT, outline="")

        tk.Label(logo, text="FACETRAK", fg=T.TEXT_PRIMARY, bg=T.BG_SURFACE,
                 font=T.FONT_APP).pack(side=tk.LEFT)
        tk.Label(logo, text="  v2", fg=T.TEXT_MUTED, bg=T.BG_SURFACE,
                 font=T.FONT_MICRO).pack(side=tk.LEFT, pady=(4, 0))

        _sep(self).grid(row=0, column=col, sticky=tk.NS, padx=8, pady=12); col += 1

        # ── Primary actions ───────────────────────────────────────────────────
        self._start_btn = Button(self, f"{T.ICON_PLAY}  Start",
                                  self._cb["toggle_start"], variant="success",
                                  width=92, bg=T.BG_SURFACE)
        self._start_btn.grid(row=0, column=col, **pad); col += 1

        self._rec_btn = Button(self, f"{T.ICON_REC}  Record",
                                self._cb["toggle_rec"], variant="default",
                                width=100, bg=T.BG_SURFACE)
        self._rec_btn.set_enabled(False)
        self._rec_btn.grid(row=0, column=col, **pad); col += 1

        _sep(self).grid(row=0, column=col, sticky=tk.NS, padx=8, pady=12); col += 1

        # ── Toggles ───────────────────────────────────────────────────────────
        for label, key in [
            (f"{T.ICON_BLUR} Blur",    "toggle_blur"),
            (f"{T.ICON_HEAT} Heatmap", "toggle_heatmap"),
            (f"{T.ICON_SERVO} Servo",  "toggle_servo"),
        ]:
            cb = self._make_check(self, label, self._cb[key])
            cb.grid(row=0, column=col, **pad); col += 1

        _sep(self).grid(row=0, column=col, sticky=tk.NS, padx=8, pady=12); col += 1

        # ── DB actions ────────────────────────────────────────────────────────
        Button(self, f"{T.ICON_REG} Register", self._cb["register_dialog"],
               variant="primary", bg=T.BG_SURFACE
               ).grid(row=0, column=col, **pad); col += 1
        Button(self, f"{T.ICON_DIR} Directory", self._cb["list_faces"],
               variant="default", bg=T.BG_SURFACE
               ).grid(row=0, column=col, **pad); col += 1

        _sep(self).grid(row=0, column=col, sticky=tk.NS, padx=8, pady=12); col += 1

        # ── Target dropdown ───────────────────────────────────────────────────
        tk.Label(self, text="TARGET", fg=T.TEXT_MUTED, bg=T.BG_SURFACE,
                 font=T.FONT_MICRO).grid(row=0, column=col, padx=(4, 2)); col += 1

        from facetrak.core.engine import (
            SERVO_TARGET_LARGEST, SERVO_TARGET_KNOWN, SERVO_TARGET_UNKNOWN
        )
        self._srv_var = tk.StringVar(value=SERVO_TARGET_LARGEST)
        srv = self._styled_combo(self._srv_var,
                                  [SERVO_TARGET_LARGEST, SERVO_TARGET_KNOWN,
                                   SERVO_TARGET_UNKNOWN], width=11)
        srv.grid(row=0, column=col, **pad); col += 1
        srv.bind("<<ComboboxSelected>>",
                 lambda _: self._cb["set_servo_target"](self._srv_var.get()))

        # ── Source dropdown ───────────────────────────────────────────────────
        tk.Label(self, text="SOURCE", fg=T.TEXT_MUTED, bg=T.BG_SURFACE,
                 font=T.FONT_MICRO).grid(row=0, column=col, padx=(8, 2)); col += 1

        self._cam_var = tk.StringVar()
        self._cam_cb = self._styled_combo(self._cam_var, [], width=16)
        self._cam_cb.grid(row=0, column=col, **pad); col += 1
        self._cam_cb.bind("<<ComboboxSelected>>", self._cb["on_cam_select"])

        # ── Spacer ────────────────────────────────────────────────────────────
        self.columnconfigure(col, weight=1); col += 1

        # ── Simulation ────────────────────────────────────────────────────────
        Button(self, f"{T.ICON_SIM} Simulation", self._cb["open_sim"],
               variant="ghost", bg=T.BG_SURFACE
               ).grid(row=0, column=col, padx=(0, 16))

    def _make_check(self, parent: tk.Frame, text: str, cmd: Callable) -> ttk.Checkbutton:
        style = ttk.Style()
        style.configure("Bar.TCheckbutton",
                         background=T.BG_SURFACE,
                         foreground=T.TEXT_SECONDARY,
                         focuscolor="none")
        style.map("Bar.TCheckbutton",
                  background=[("active", T.BG_SURFACE)],
                  foreground=[("active", T.TEXT_PRIMARY)])
        cb = ttk.Checkbutton(parent, text=text, style="Bar.TCheckbutton",
                              command=cmd)
        cb.state(["!alternate"])
        return cb

    def _styled_combo(self, var: tk.StringVar, values: list,
                       width: int = 12) -> ttk.Combobox:
        style = ttk.Style()
        style.configure("Bar.TCombobox",
                         background=T.BG_OVERLAY,
                         foreground=T.TEXT_PRIMARY,
                         fieldbackground=T.BG_OVERLAY,
                         arrowcolor=T.TEXT_SECONDARY,
                         selectbackground=T.ACCENT,
                         selectforeground=T.TEXT_PRIMARY)
        style.map("Bar.TCombobox",
                  fieldbackground=[("readonly", T.BG_OVERLAY)],
                  foreground=[("readonly", T.TEXT_PRIMARY)])
        return ttk.Combobox(self, textvariable=var, values=values,
                             state="readonly", width=width,
                             style="Bar.TCombobox")
