"""Top command bar component."""

import tkinter as tk
from tkinter import ttk

from .. import theme as T
from .button import Button


class CommandBar(tk.Frame):
    """Two-level toolbar with adaptive action groups."""

    HEIGHT = 48

    def __init__(self, parent, callbacks: dict, **kw):
        super().__init__(parent, bg=T.BG_SURFACE, height=self.HEIGHT, **kw)
        self._cb = callbacks
        self._build()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_running(self, running: bool):
        self._rec_btn.set_enabled(running)
        self._register_btn.set_enabled(running)
        self._state_label.config(
            text="Camera running" if running else "Camera stopped",
            fg=T.SUCCESS if running else T.TEXT_SECONDARY,
        )
        if running:
            self._start_btn.set_state(text="■  Stop camera", variant="danger")
        else:
            self._start_btn.set_state(text="▶  Start camera", variant="primary")
            self._rec_btn.set_state(
                text=f"{T.ICON_REC}  Record", variant="default", enabled=False
            )

    def set_recording(self, recording: bool):
        if recording:
            self._rec_btn.set_state(
                text=f"{T.ICON_REC}  Stop Rec", variant="danger", enabled=True
            )
        else:
            self._rec_btn.set_state(
                text=f"{T.ICON_REC}  Record", variant="default", enabled=True
            )

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

    def set_toggles(self, **states):
        for key, value in states.items():
            if key in self._toggles:
                self._toggles[key].set(value)

    def _build(self):
        self.columnconfigure(0, weight=1)
        header = tk.Frame(self, bg=T.BG_SURFACE)
        header.grid(row=0, column=0, sticky=tk.EW, padx=20, pady=(16, 12))
        header.columnconfigure(1, weight=1)
        brand = tk.Frame(header, bg=T.BG_SURFACE)
        brand.grid(row=0, column=0, sticky=tk.W, padx=(0, 32))
        tk.Label(
            brand, text="FaceTrak", font=T.FONT_APP, bg=T.BG_SURFACE, fg=T.TEXT_PRIMARY
        ).pack(anchor=tk.W)
        self._state_label = tk.Label(
            brand,
            text="Camera stopped",
            font=T.FONT_MICRO,
            bg=T.BG_SURFACE,
            fg=T.TEXT_SECONDARY,
        )
        self._state_label.pack(anchor=tk.W, pady=(2, 0))

        source = tk.Frame(header, bg=T.BG_SURFACE)
        source.grid(row=0, column=1, sticky=tk.W)
        tk.Label(
            source,
            text="Camera source",
            font=T.FONT_MICRO,
            bg=T.BG_SURFACE,
            fg=T.TEXT_SECONDARY,
        ).pack(anchor=tk.W)
        self._cam_var = tk.StringVar()
        self._cam_cb = self._styled_combo(source, self._cam_var, [], width=22)
        self._cam_cb.pack(pady=(4, 0))
        self._cam_cb.bind("<<ComboboxSelected>>", self._cb["on_cam_select"])

        self._start_btn = Button(
            header,
            "▶  Start camera",
            self._cb["toggle_start"],
            variant="primary",
            width=155,
            height=42,
        )
        self._start_btn.grid(row=0, column=2, padx=(16, 8))
        self._rec_btn = Button(
            header, "●  Record", self._cb["toggle_rec"], width=120, height=42
        )
        self._rec_btn.set_enabled(False)
        self._rec_btn.grid(row=0, column=3)

        tk.Frame(self, bg=T.BORDER_SUBTLE, height=1).grid(
            row=1, column=0, sticky=tk.EW, padx=20
        )
        tools = tk.Frame(self, bg=T.BG_SURFACE)
        tools.grid(row=2, column=0, sticky=tk.EW, padx=20, pady=12)
        tools.columnconfigure(1, weight=1)
        view = tk.Frame(tools, bg=T.BG_SURFACE)
        view.grid(row=0, column=0, sticky=tk.W)
        self._toggles = {}
        for label, key in [
            ("Blur unknown", "blur"),
            ("Heatmap", "heatmap"),
            ("Servo", "servo"),
            ("Mirror", "mirror"),
        ]:
            var = tk.BooleanVar(value=False)
            self._toggles[key] = var
            self._make_check(view, label, self._cb["toggle_" + key], var).pack(
                side=tk.LEFT, padx=(0, 14)
            )

        from facetrak.core.engine import (
            SERVO_TARGET_KNOWN,
            SERVO_TARGET_LARGEST,
            SERVO_TARGET_UNKNOWN,
        )

        target = tk.Frame(view, bg=T.BG_SURFACE)
        target.pack(side=tk.LEFT)
        tk.Label(
            target,
            text="Track",
            font=T.FONT_MICRO,
            bg=T.BG_SURFACE,
            fg=T.TEXT_SECONDARY,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self._srv_var = tk.StringVar(value=SERVO_TARGET_LARGEST)
        combo = self._styled_combo(
            target,
            self._srv_var,
            [SERVO_TARGET_LARGEST, SERVO_TARGET_KNOWN, SERVO_TARGET_UNKNOWN],
            width=10,
        )
        combo.pack(side=tk.LEFT)
        combo.bind(
            "<<ComboboxSelected>>",
            lambda _: self._cb["set_servo_target"](self._srv_var.get()),
        )

        actions = tk.Frame(tools, bg=T.BG_SURFACE)
        actions.grid(row=0, column=1, sticky=tk.E)
        self._register_btn = Button(actions, "Add person", self._cb["register_dialog"])
        self._register_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._register_btn.set_enabled(False)
        Button(actions, "Directory", self._cb["list_faces"], variant="ghost").pack(
            side=tk.LEFT, padx=(0, 8)
        )
        Button(actions, "Simulation", self._cb["open_sim"], variant="ghost").pack(
            side=tk.LEFT
        )

        # Keep every action reachable when the window is narrower than both groups.
        def reflow(event):
            needed = view.winfo_reqwidth() + actions.winfo_reqwidth() + 24
            if event.width < needed:
                actions.grid_configure(
                    row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 0)
                )
            else:
                actions.grid_configure(
                    row=0, column=1, columnspan=1, sticky=tk.E, pady=0
                )

        tools.bind("<Configure>", reflow)

    def _make_check(self, parent, text, cmd, variable):
        style = ttk.Style()
        style.configure(
            "Bar.TCheckbutton",
            background=T.BG_SURFACE,
            foreground=T.TEXT_SECONDARY,
            font=T.FONT_LABEL,
            focuscolor=T.BORDER_FOCUS,
            padding=(0, 6),
        )
        style.map(
            "Bar.TCheckbutton",
            background=[("active", T.BG_SURFACE)],
            foreground=[("active", T.TEXT_PRIMARY)],
        )
        return ttk.Checkbutton(
            parent, text=text, style="Bar.TCheckbutton", variable=variable, command=cmd
        )

    def _styled_combo(self, parent, var, values, width=12):
        style = ttk.Style()
        style.configure(
            "Bar.TCombobox",
            background=T.BG_OVERLAY,
            foreground=T.TEXT_PRIMARY,
            fieldbackground=T.BG_OVERLAY,
            arrowcolor=T.TEXT_SECONDARY,
            padding=6,
            selectbackground=T.ACCENT_DIM,
            selectforeground=T.TEXT_PRIMARY,
        )
        style.map(
            "Bar.TCombobox",
            fieldbackground=[("readonly", T.BG_OVERLAY)],
            foreground=[("readonly", T.TEXT_PRIMARY)],
        )
        return ttk.Combobox(
            parent,
            textvariable=var,
            values=values,
            state="readonly",
            width=width,
            font=T.FONT_LABEL,
            style="Bar.TCombobox",
        )
