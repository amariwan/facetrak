"""FaceTrak HUD — structured dashboard with card-based layout.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  ◆ OMEN    [▶] [●] [Register] [Servo] [Cam]     Sim   │  ← Command bar
  ├─────────────────────────────┬───────────────────────────┤
  │                             │ ┌─ FACE TELEMETRY ──────┐│
  │                             │ │ Name   Pose   Servo   ││
  │      VIDEO FEED             │ │ Dwell  Blinks Age/Gdr ││
  │      (primary zone)         │ └───────────────────────┘│
  │                             │ ┌─ ATTENTION ───────────┐│
  │                             │ │ ████████░░ 72%       ││
  │                             │ └───────────────────────┘│
  │                             │ ┌─ EMOTION SIGNATURE ───┐│
  │                             │ │    (pie chart)        ││
  │                             │ └───────────────────────┘│
  │                             │ ┌─ FACE MANAGEMENT ─────┐│
  │                             │ │ Alice  ●  Bob  ○     ││
  │                             │ │ [Toggle Blur]         ││
  │                             │ └───────────────────────┘│
  ├─────────────────────────────┴───────────────────────────┤
  │ 📷 cam1  ⊞ 320,240  ◈ 5/12/3  ◎ 90°  ◆ 3  ● REC       │ ← Telemetry
  └─────────────────────────────────────────────────────────┘
"""
import math
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

import cv2
from PIL import Image, ImageTk

from facetrak import config
from facetrak.engine import (FaceEngine, SERVO_TARGET_LARGEST,
                              SERVO_TARGET_KNOWN, SERVO_TARGET_UNKNOWN)
from facetrak.simulation import SimulationWindow

_POLL_MS = 30
_MAX_W   = 800
_SPARK_W = 200
_SPARK_H = 85

# ── OMEN palette ─────────────────────────────────────────
_VOID   = "#0A0E17"
_SURF   = "#111827"
_CARD   = "#1A1F2E"
_BORD   = "#1E293B"
_CYAN   = "#00E5FF"
_MAG    = "#FF0088"
_GRN    = "#39FF9E"
_AMB    = "#FFB347"
_RED    = "#FF3355"
_TXT    = "#C8D6E5"
_TXT2   = "#5A6A7E"

_EMO_COLORS = {"happy": _GRN, "sad": "#5B9BFF", "angry": _RED,
               "surprised": _AMB, "neutral": _TXT}


class _Card(ttk.LabelFrame):
    """A labeled card panel with fixed styling."""
    def __init__(self, parent, title, **kw):
        super().__init__(parent, text=f"  {title}  ", **kw)
        self.configure(labelwidget=ttk.Label(
            self, text=f"  {title}  ",
            foreground=_CYAN, background=_CARD,
            font=("Helvetica", 9, "bold")))


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.eng  = FaceEngine()
        self._poll_id = None
        self._reg_name: str | None = None
        self._reg_progress: ttk.Progressbar | None = None
        self._scan_y = 0
        self._scan_dir = 1

        self._setup_theme()
        root.title("FaceTrak  |  OMEN Interface")
        root.configure(bg=_VOID)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.bind("<space>",  lambda _: self._register_dialog())
        root.bind("<r>",      lambda _: self._toggle_rec())
        root.bind("<R>",      lambda _: self._toggle_rec())
        root.bind("<b>",      lambda _: self._toggle_blur())
        root.bind("<B>",      lambda _: self._toggle_blur())
        root.bind("<h>",      lambda _: self._toggle_heatmap())
        root.bind("<H>",      lambda _: self._toggle_heatmap())
        root.bind("<Escape>", lambda _: self._stop() if self.eng.running else None)

        self._build_layout()
        self._animate_hud()

    # ── theme ─────────────────────────────────────────────

    def _setup_theme(self):
        style = ttk.Style()
        style.theme_use("clam")
        for s in ("TFrame", "TLabel", "TButton", "TCheckbutton",
                   "TCombobox", "TProgressbar", "TSeparator",
                   "TLabelframe"):
            style.configure(s, background=_VOID)
        style.configure("TLabel", foreground=_TXT, background=_VOID)
        style.configure("Card.TLabelframe", background=_CARD,
                        foreground=_CYAN, bordercolor=_BORD,
                        lightcolor=_BORD, darkcolor=_BORD)
        style.configure("Card.TLabelframe.Label", background=_CARD,
                        foreground=_CYAN)
        style.configure("TButton", background=_CARD, foreground=_CYAN,
                        bordercolor=_BORD, focuscolor="none",
                        lightcolor=_CARD, darkcolor=_CARD)
        style.map("TButton",
                  background=[("active", _CYAN)],
                  foreground=[("active", _VOID)])
        style.configure("TCheckbutton", background=_SURF, foreground=_TXT,
                        focuscolor="none")
        style.map("TCheckbutton",
                  background=[("active", _SURF)],
                  foreground=[("active", _TXT)])
        style.configure("TCombobox", background=_CARD, foreground=_TXT,
                        fieldbackground=_CARD, arrowcolor=_CYAN,
                        bordercolor=_BORD, selectbackground=_CYAN,
                        selectforeground=_VOID)
        style.map("TCombobox",
                  fieldbackground=[("readonly", _CARD)],
                  foreground=[("readonly", _TXT)])
        style.configure("TProgressbar", background=_GRN, troughcolor=_CARD,
                        bordercolor=_BORD, lightcolor=_GRN, darkcolor=_GRN)
        style.configure("TSeparator", background=_BORD)
        style.configure("TScale", background=_VOID, troughcolor=_CARD,
                        bordercolor=_BORD, slidercolor=_CYAN)

    # ── layout ───────────────────────────────────────────

    def _make_chk(self, parent, text, command):
        cb = ttk.Checkbutton(parent, text=text, command=command)
        cb.state(["!alternate"])
        return cb

    def _build_layout(self):
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._build_command_bar()    # row 0
        self._build_content_area()   # row 1
        self._build_telemetry_bar()  # row 2

    # ── row 0: command bar ───────────────────────────────

    def _build_command_bar(self):
        bar = tk.Frame(self.root, bg=_SURF, height=42)
        bar.grid(row=0, column=0, sticky=tk.EW)
        bar.grid_propagate(False)
        bar.columnconfigure(10, weight=1)  # spacer on the right

        pad = {"padx": 6, "pady": 0}

        # Brand
        tk.Label(bar, text="◆ OMEN", fg=_CYAN, bg=_SURF,
                 font=("Helvetica", 14, "bold")).grid(row=0, column=0, padx=(12, 8))

        ttk.Separator(bar, orient=tk.VERTICAL).grid(
            row=0, column=1, sticky=tk.NS, padx=4, pady=8)

        # Group: camera control
        col = 2
        self._start_btn = tk.Button(bar, text="▶ START", fg=_GRN, bg=_CARD,
                                    activeforeground=_VOID, activebackground=_GRN,
                                    font=("Helvetica", 9, "bold"),
                                    relief=tk.FLAT, padx=14,
                                    cursor="hand2", command=self._toggle_start)
        self._start_btn.grid(row=0, column=col, **pad); col += 1

        self._rec_btn = tk.Button(bar, text="● REC", fg=_TXT2, bg=_CARD,
                                  activeforeground=_VOID, activebackground=_RED,
                                  font=("Helvetica", 9, "bold"),
                                  relief=tk.FLAT, padx=10, state=tk.DISABLED,
                                  cursor="hand2", command=self._toggle_rec)
        self._rec_btn.grid(row=0, column=col, **pad); col += 1

        ttk.Separator(bar, orient=tk.VERTICAL).grid(
            row=0, column=col, sticky=tk.NS, padx=4, pady=8); col += 1

        # Group: toggles
        for txt, cmd in [("Blur [B]", self._toggle_blur),
                          ("Heat [H]", self._toggle_heatmap),
                          ("Servo", self._toggle_servo)]:
            self._make_chk(bar, txt, cmd).grid(row=0, column=col, **pad)
            col += 1

        ttk.Separator(bar, orient=tk.VERTICAL).grid(
            row=0, column=col, sticky=tk.NS, padx=4, pady=8); col += 1

        # Group: register
        tk.Button(bar, text="⟐ REGISTER", fg=_MAG, bg=_CARD,
                  activeforeground=_VOID, activebackground=_MAG,
                  font=("Helvetica", 9, "bold"), relief=tk.FLAT, padx=8,
                  cursor="hand2", command=self._register_dialog
                  ).grid(row=0, column=col, **pad); col += 1
        tk.Button(bar, text="👤 FACES", fg=_AMB, bg=_CARD,
                  activeforeground=_VOID, activebackground=_AMB,
                  font=("Helvetica", 9, "bold"), relief=tk.FLAT, padx=8,
                  cursor="hand2", command=self._list_faces
                  ).grid(row=0, column=col, **pad); col += 1

        ttk.Separator(bar, orient=tk.VERTICAL).grid(
            row=0, column=col, sticky=tk.NS, padx=4, pady=8); col += 1

        # Group: servo target
        tk.Label(bar, text="TRACK", fg=_TXT2, bg=_SURF,
                 font=("Helvetica", 8)).grid(row=0, column=col, padx=(6, 2))
        col += 1
        self._srv_var = tk.StringVar(value=SERVO_TARGET_LARGEST)
        srv_cb = ttk.Combobox(bar, textvariable=self._srv_var,
                               values=[SERVO_TARGET_LARGEST,
                                       SERVO_TARGET_KNOWN,
                                       SERVO_TARGET_UNKNOWN],
                               state="readonly", width=8)
        srv_cb.grid(row=0, column=col, **pad); col += 1
        srv_cb.bind("<<ComboboxSelected>>",
                    lambda _: self.eng.set_servo_target(self._srv_var.get()))

        # Group: camera selector (right-aligned via column 10 spacer)
        tk.Label(bar, text="CAM", fg=_TXT2, bg=_SURF,
                 font=("Helvetica", 8)).grid(row=0, column=col, padx=(6, 2))
        col += 1
        self._cam_var = tk.StringVar()
        self._cam_cb = ttk.Combobox(bar, textvariable=self._cam_var,
                                     state="readonly", width=16)
        self._cam_cb.grid(row=0, column=col, **pad); col += 1
        self._cam_cb.bind("<<ComboboxSelected>>", self._on_cam_select)

        # Sim button (far right)
        col = 12
        tk.Button(bar, text="⛭ SIM", fg=_TXT2, bg=_CARD,
                  activeforeground=_TXT, activebackground=_BORD,
                  font=("Helvetica", 8), relief=tk.FLAT, padx=6,
                  cursor="hand2", command=self._open_sim
                  ).grid(row=0, column=col, padx=(0, 8))

    # ── row 1: content (video + dashboard panel) ────────

    def _build_content_area(self):
        main = tk.Frame(self.root, bg=_VOID)
        main.grid(row=1, column=0, sticky=tk.NSEW)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # Left: video
        video_frame = tk.Frame(main, bg=_VOID)
        video_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(10, 4), pady=6)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self._video_canvas = tk.Canvas(video_frame, bg=_VOID,
                                        highlightthickness=0)
        self._video_canvas.grid(row=0, column=0, sticky=tk.NSEW)

        self._video_label = tk.Label(self._video_canvas, bg=_VOID)
        self._video_canvas.create_window(0, 0, window=self._video_label,
                                          anchor=tk.NW, tags="vid")

        # Face strip below video
        self._face_strip = tk.Frame(video_frame, bg=_VOID, height=64)
        self._face_strip.grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))
        self._face_strip.grid_propagate(False)
        self._face_thumbs: list[tk.Label] = []

        # Right: dashboard panel
        right = tk.Frame(main, bg=_SURF, width=280)
        right.grid(row=0, column=1, sticky=tk.NS, padx=(4, 10), pady=6)
        right.grid_propagate(False)

        self._build_dashboard(right)

    def _build_dashboard(self, parent):
        parent.columnconfigure(0, weight=1)

        # ── Card 1: Face Telemetry ──
        card = _Card(parent, "FACE TELEMETRY")
        card.grid(row=0, column=0, sticky=tk.EW, padx=8, pady=(8, 4))
        card.columnconfigure(1, weight=1)

        fields = [
            ("Name",   "name"),
            ("Pose",   "pose"),
            ("Servo",  "servo"),
            ("Dwell",  "dwell"),
            ("Blinks", "blinks"),
            ("Age",    "age"),
        ]
        self._tel = {}
        for i, (label, key) in enumerate(fields):
            tk.Label(card, text=label, fg=_TXT2, bg=_CARD,
                     font=("Helvetica", 8), anchor=tk.W, width=6
                     ).grid(row=i, column=0, sticky=tk.W, padx=(8, 2), pady=1)
            val = tk.Label(card, text="—", fg=_TXT, bg=_CARD,
                           font=("Helvetica", 9, "bold"), anchor=tk.W)
            val.grid(row=i, column=1, sticky=tk.EW, padx=(0, 8), pady=1)
            self._tel[key] = val

        # ── Card 2: Attention Gauge ──
        card2 = _Card(parent, "ATTENTION")
        card2.grid(row=1, column=0, sticky=tk.EW, padx=8, pady=4)
        self._attn_canvas = tk.Canvas(card2, width=248, height=36,
                                       bg=_VOID, highlightthickness=0)
        self._attn_canvas.pack(padx=8, pady=4)

        # ── Card 3: Emotion Signature ──
        card3 = _Card(parent, "EMOTION SIGNATURE")
        card3.grid(row=2, column=0, sticky=tk.EW, padx=8, pady=4)
        self._emo_canvas = tk.Canvas(card3, width=248, height=110,
                                      bg=_VOID, highlightthickness=0)
        self._emo_canvas.pack(padx=8, pady=4)

        # ── Card 4: Face Management ──
        card4 = _Card(parent, "FACE MANAGEMENT")
        card4.grid(row=3, column=0, sticky=tk.NSEW, padx=8, pady=(4, 8))
        card4.columnconfigure(0, weight=1)
        card4.rowconfigure(0, weight=1)

        list_frame = tk.Frame(card4, bg=_CARD)
        list_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=6, pady=4)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        scroll = tk.Scrollbar(list_frame, bg=_CARD, troughcolor=_VOID)
        self._blur_list = tk.Listbox(list_frame, height=5,
                                      bg=_CARD, fg=_TXT,
                                      selectbackground=_CYAN,
                                      selectforeground=_VOID,
                                      borderwidth=0, highlightthickness=0,
                                      activestyle="none",
                                      font=("Helvetica", 9),
                                      yscrollcommand=scroll.set)
        scroll.config(command=self._blur_list.yview)
        self._blur_list.grid(row=0, column=0, sticky=tk.NSEW)
        scroll.grid(row=0, column=1, sticky=tk.NS)

        btn = tk.Button(card4, text="⏎ TOGGLE BLUR", fg=_MAG, bg=_CARD,
                        activeforeground=_VOID, activebackground=_MAG,
                        font=("Helvetica", 9, "bold"),
                        relief=tk.FLAT, cursor="hand2",
                        command=self._toggle_person_blur)
        btn.grid(row=1, column=0, sticky=tk.EW, padx=6, pady=(0, 4))

        # spacer to push cards up
        parent.rowconfigure(4, weight=1)

    # ── row 2: telemetry bar ─────────────────────────────

    def _build_telemetry_bar(self):
        bar = tk.Frame(self.root, bg=_VOID, height=26)
        bar.grid(row=2, column=0, sticky=tk.EW)
        bar.grid_propagate(False)

        tk.Frame(bar, bg=_BORD, height=1).pack(fill=tk.X)

        inner = tk.Frame(bar, bg=_VOID)
        inner.pack(fill=tk.X, padx=10, pady=3)

        self._tele_labels = {}
        items = [
            ("cam",     "📷", _CYAN, 14),
            ("face",    "⊞", _GRN, 12),
            ("pose",    "◈", _AMB, 10),
            ("servo",   "◎", _MAG, 10),
            ("emotion", "◉", _TXT, 10),
            ("crowd",   "◆", _GRN, 6),
            ("known",   "⬡", _CYAN, 6),
        ]
        for key, icon, color, width in items:
            tk.Label(inner, text=icon, fg=color, bg=_VOID,
                     font=("Helvetica", 9)).pack(side=tk.LEFT)
            lbl = tk.Label(inner, text="—", fg=_TXT2, bg=_VOID,
                           font=("Helvetica", 9), width=width, anchor=tk.W)
            lbl.pack(side=tk.LEFT, padx=(2, 8))
            self._tele_labels[key] = lbl

        self._rec_dot = tk.Canvas(inner, width=12, height=12,
                                   bg=_VOID, highlightthickness=0)
        self._rec_dot.pack(side=tk.RIGHT, padx=4)

    # ── camera ───────────────────────────────────────────

    def _populate_cameras(self):
        cfg = config.load()
        cams = cfg.get("cameras", [])
        labels = [config.label(cfg, i) for i in range(len(cams))]
        self._cam_cb["values"] = labels
        active = cfg.get("camera", 0)
        if 0 <= active < len(labels):
            self._cam_var.set(labels[active])

    def _on_cam_select(self, _=None):
        idx = self._cam_cb.current()
        if idx < 0:
            return
        if not self.eng.running:
            cfg = config.load()
            cfg["camera"] = idx
            config.save(cfg)
            self.eng.current_cam_idx = idx
            return
        if not self.eng.switch_camera(idx):
            messagebox.showerror("Error", "Failed to switch camera.")
            self._populate_cameras()

    # ── start / stop ─────────────────────────────────────

    def _toggle_start(self):
        if self.eng.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        if not self.eng.start():
            messagebox.showerror("Error", "Cannot open camera.")
            return
        self._start_btn.config(text="■ STOP", fg=_RED)
        self._rec_btn.config(state=tk.NORMAL, fg=_TXT2)
        self._populate_cameras()
        self._cam_cb.state(["!disabled"])
        self._refresh_blur_list()
        self._poll()

    def _stop(self):
        if self._poll_id:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        self.eng.stop()
        self._start_btn.config(text="▶ START", fg=_GRN)
        self._rec_btn.config(state=tk.DISABLED, fg=_TXT2, text="● REC")
        self._video_label.config(image="")
        self._clear_telemetry()
        for lbl in self._face_thumbs:
            lbl.destroy()
        self._face_thumbs.clear()

    # ── poll loop ─────────────────────────────────────────

    def _poll(self):
        if not self.eng.running:
            return
        frame = self.eng.step()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if w > _MAX_W:
                s = _MAX_W / w
                rgb = cv2.resize(rgb, (_MAX_W, int(h * s)))
            img = Image.fromarray(rgb)
            self._tk_img = ImageTk.PhotoImage(img)
            self._video_label.config(image=self._tk_img)
            self._update_telemetry()
            self._update_face_telemetry()
            self._draw_attn_gauge()
            self._draw_emotion_wheel()
            self._update_face_strip()

            rec_on = self.eng.recorder.recording
            self._rec_btn.config(text="■ REC" if rec_on else "● REC",
                                  fg=_RED if rec_on else _TXT2)

        if self._reg_name and self.eng._capturing:
            from facetrak.engine import _MAX_SAMPLES, _MIN_REG_SAMPLES
            n = len(self.eng._samples_buffer)
            if self._reg_progress:
                self._reg_progress["value"] = int(100 * n / _MAX_SAMPLES)
            if n >= _MAX_SAMPLES or (n >= _MIN_REG_SAMPLES
                                      and self.eng.liveness.passed):
                name = self._reg_name
                self._reg_name = None
                self.root.after(0, lambda: self._do_register(name))
                return

        self._poll_id = self.root.after(_POLL_MS, self._poll)

    def _clear_telemetry(self):
        for lbl in self._tele_labels.values():
            lbl.config(text="—")

    def _update_telemetry(self):
        e = self.eng
        cfg = config.load()
        m = e.metrics
        self._tele_labels["cam"].config(text=config.label(cfg, e.current_cam_idx))
        self._tele_labels["face"].config(text=f"{e.last_face_center[0]},{e.last_face_center[1]}")
        self._tele_labels["pose"].config(text=f"{m.yaw:.0f}/{m.pitch:.0f}/{m.roll:.0f}")
        self._tele_labels["servo"].config(text=f"{e.current_pan:.0f}° {e.current_tilt:.0f}°")
        self._tele_labels["emotion"].config(text=m.emotion or "—")
        self._tele_labels["crowd"].config(text=str(len(e.tracker.active)))
        self._tele_labels["known"].config(text=str(len(e.db.known_names)))

    # ── face telemetry card ──────────────────────────────

    def _update_face_telemetry(self):
        e = self.eng
        m = e.metrics
        primary = e.tracker.largest()
        if primary:
            name = primary.name or "Unknown"
            self._tel["name"].config(text=name, fg=_GRN if primary.name else _AMB)
            self._tel["dwell"].config(text=f"{primary.dwell:.0f}s")
            self._tel["blinks"].config(text=str(primary.blink_count))
            self._tel["age"].config(text=f"{primary.gender}/{primary.age}")
        else:
            self._tel["name"].config(text="—", fg=_TXT)
            self._tel["dwell"].config(text="—")
            self._tel["blinks"].config(text="—")
            self._tel["age"].config(text="—")
        self._tel["pose"].config(
            text=f"Y:{m.yaw:.0f}°  P:{m.pitch:.0f}°  R:{m.roll:.0f}°")
        self._tel["servo"].config(
            text=f"Pan:{e.current_pan:.0f}°  Tilt:{e.current_tilt:.0f}°")

    # ── face thumbnails ──────────────────────────────────

    def _update_face_strip(self):
        for lbl in self._face_thumbs:
            lbl.destroy()
        self._face_thumbs.clear()

        frame = self.eng._frame
        if frame is None:
            return

        for t in self.eng.tracker.active[:6]:
            d = t.det
            x1 = max(0, d.x); y1 = max(0, d.y)
            x2 = min(frame.shape[1], d.x + d.w)
            y2 = min(frame.shape[0], d.y + d.h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            thumb = Image.fromarray(crop_rgb).resize((46, 46), Image.LANCZOS)

            known = t.name is not None
            border = _GRN if known else _AMB

            framed = Image.new("RGB", (50, 50), _VOID)
            framed.paste(thumb, (2, 2))
            photo = ImageTk.PhotoImage(framed)

            lbl = tk.Label(self._face_strip, image=photo, bg=_VOID,
                           highlightbackground=border,
                           highlightcolor=border,
                           highlightthickness=1)
            lbl.photo = photo
            lbl.pack(side=tk.LEFT, padx=2)
            self._face_thumbs.append(lbl)

            if known:
                tk.Label(self._face_strip, text=t.name[:6],
                         fg=_GRN, bg=_VOID,
                         font=("Helvetica", 7)).pack(side=tk.LEFT)

    # ── attention gauge ──────────────────────────────────

    def _draw_attn_gauge(self):
        c = self._attn_canvas
        c.delete("all")
        w = 248
        m = self.eng.metrics

        c.create_rectangle(8, 10, w - 8, 28, fill=_CARD, outline=_BORD, width=1)

        engaged = 0.3
        if m.attentive:
            engaged = 0.7 + (m.smile or 0) * 0.3
        engaged = max(0.05, min(1.0, engaged))

        fw = max(4, int((w - 18) * engaged))
        color = _GRN if engaged > 0.6 else (_AMB if engaged > 0.3 else _RED)
        c.create_rectangle(9, 11, 9 + fw, 27, fill=color, outline="")

        label = "ATTENTIVE" if m.attentive else "DISTRACTED"
        c.create_text(w // 2, 8, text=label, fill=_TXT,
                      font=("Helvetica", 8), anchor=tk.S)

    # ── emotion wheel ────────────────────────────────────

    def _draw_emotion_wheel(self):
        c = self._emo_canvas
        c.delete("all")
        counts = self.eng.timeline.recent_emotion_counts(60)
        if not counts:
            c.create_text(124, 55, text="no data yet", fill=_TXT2,
                          font=("Helvetica", 9))
            return

        total = sum(counts.values())
        cx, cy = 110, 56
        r = 34
        start = 0
        for emotion, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            extent = 360 * cnt / total
            color = _EMO_COLORS.get(emotion, _TXT2)
            c.create_arc(cx - r, cy - r, cx + r, cy + r,
                         start=start, extent=extent,
                         fill=color, outline=_VOID, width=1)
            mid = math.radians(start + extent / 2)
            lr = r + 14
            lx = cx + lr * math.cos(mid)
            ly = cy + lr * math.sin(mid)
            c.create_text(lx, ly, text=f"{emotion[:3]}",
                          fill=color, font=("Helvetica", 7))
            start += extent

    # ── HUD overlays ─────────────────────────────────────

    def _animate_hud(self):
        vw = self._video_canvas.winfo_width()
        vh = self._video_canvas.winfo_height()
        self._video_canvas.delete("hud")

        if vw > 10 and vh > 10:
            inset = 6
            bl = 14
            bw = 2
            for x, y, xd, yd in [(inset, inset, 1, 1),
                                  (vw - inset, inset, -1, 1),
                                  (inset, vh - inset, 1, -1),
                                  (vw - inset, vh - inset, -1, -1)]:
                self._video_canvas.create_line(
                    x - xd * bl, y, x + xd * bl, y,
                    fill=_CYAN, width=bw, tags="hud")
                self._video_canvas.create_line(
                    x, y - yd * bl, x, y + yd * bl,
                    fill=_CYAN, width=bw, tags="hud")

            if self.eng.running:
                self._scan_y += self._scan_dir * 1.2
                if self._scan_y > vh - 10 or self._scan_y < 10:
                    self._scan_dir *= -1
                self._video_canvas.create_line(
                    10, self._scan_y, vw - 10, self._scan_y,
                    fill=_CYAN, width=1, tags="hud", stipple="gray50")

                self._video_canvas.create_text(
                    12, vh - 8, text="LIVE", fill=_GRN,
                    font=("Helvetica", 8, "bold"),
                    anchor=tk.SW, tags="hud")

        # Recording dot animation
        rec_on = self.eng.running and self.eng.recorder.recording
        self._rec_dot.delete("all")
        if rec_on:
            import time
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 4)
            r = int(4 + pulse * 2)
            cx = cy = 6
            self._rec_dot.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=_RED, outline=_RED, width=1)
        else:
            self._rec_dot.create_oval(
                2, 2, 10, 10, fill=_CARD, outline=_BORD, width=1)

        self.root.after(50, self._animate_hud)

    # ── controls ─────────────────────────────────────────

    def _toggle_rec(self):
        self.eng.toggle_record()

    def _toggle_blur(self):
        self.eng.toggle_blur()

    def _toggle_heatmap(self):
        self.eng.toggle_heatmap()

    def _toggle_servo(self):
        self.eng.toggle_servo()

    # ── registration ─────────────────────────────────────

    def _register_dialog(self):
        if not self.eng.running:
            messagebox.showinfo("Info", "Start camera first.")
            return
        name = simpledialog.askstring("Register", "Name:")
        if not name:
            return
        self._reg_name = name
        self.eng.set_overlay("Look at camera — blink twice and turn head")
        self.eng.capture_samples()

        win = tk.Toplevel(self.root, bg=_SURF)
        win.title("Registering…")
        win.geometry("340x130")
        win.resizable(False, False)
        tk.Label(win, text=f"Scanning \"{name}\"",
                 fg=_CYAN, bg=_SURF,
                 font=("Helvetica", 13, "bold")).pack(pady=(16, 4))
        self._reg_progress = ttk.Progressbar(win, length=280, mode="determinate")
        self._reg_progress.pack(pady=4)
        tk.Label(win, text="blink twice · turn head slowly · hold still",
                 fg=_TXT2, bg=_SURF,
                 font=("Helvetica", 9)).pack()
        self.root.after(5000, lambda: self._close_reg_win(win))

    def _close_reg_win(self, win):
        if self._reg_name is None and win.winfo_exists():
            win.destroy()
            self._reg_progress = None

    def _do_register(self, name: str):
        ok = self.eng.register(name)
        self.eng.set_overlay("")
        self._refresh_blur_list()
        if self._reg_progress:
            try:
                self._reg_progress.master.destroy()
            except tk.TclError:
                pass
            self._reg_progress = None
        if ok:
            messagebox.showinfo("Done", f"Registered '{name}'.")
        else:
            messagebox.showerror("Error", "Not enough samples — try again.")

    # ── blur list ────────────────────────────────────────

    def _refresh_blur_list(self):
        self._blur_list.delete(0, tk.END)
        for name in self.eng.db.known_names:
            self._blur_list.insert(tk.END, f"  {name}")
            if name in self.eng.blur_persons:
                self._blur_list.itemconfig(tk.END, bg=_CARD, fg=_MAG)
            else:
                self._blur_list.itemconfig(tk.END, bg=_CARD, fg=_TXT)

    def _toggle_person_blur(self):
        sel = self._blur_list.curselection()
        names = self.eng.db.known_names
        for i in sel:
            name = names[i]
            self.eng.set_blur_person(name, name not in self.eng.blur_persons)
        self._refresh_blur_list()

    def _list_faces(self):
        names = self.eng.db.known_names
        if not names:
            messagebox.showinfo("Known Faces", "(none)")
            return
        text = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names))
        blured = ", ".join(sorted(self.eng.blur_persons))
        if blured:
            text += f"\n\nBlurred: {blured}"
        messagebox.showinfo("Known Faces", text)

    def _open_sim(self):
        self._sim_win = SimulationWindow(
            self.root,
            lambda: (self.eng.current_pan, self.eng.current_tilt))

    def _on_close(self):
        self._stop()
        self.root.destroy()
