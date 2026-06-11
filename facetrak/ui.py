import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

import cv2
from PIL import Image, ImageTk

from facetrak import config
from facetrak.engine import (FaceEngine, SERVO_TARGET_LARGEST,
                              SERVO_TARGET_KNOWN, SERVO_TARGET_UNKNOWN)
from facetrak.simulation import SimulationWindow

_POLL_MS = 30
_MAX_W   = 900
_SPARK_W = 200
_SPARK_H = 80

# ── Dark theme color palette ──────────────────────────────
_BG      = "#0F1117"
_SURFACE = "#1A1D29"
_CARD    = "#242736"
_BORDER  = "#2A2D35"
_TXT     = "#E2E8F0"
_TXT2    = "#94A3B8"
_TXT3    = "#64748B"
_GRN     = "#22C55E"
_BLU     = "#2563EB"
_RED     = "#EF4444"
_AMB     = "#F59E0B"
_PUR     = "#A78BFA"
_TEAL    = "#2DD4BF"

# Emotion → color
_EMO_COLORS = {"happy": "#a6e3a1", "sad": "#89b4fa", "angry": "#f38ba8",
               "surprised": "#fab387", "neutral": "#cdd6f4"}


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.eng  = FaceEngine()
        self._poll_id = None
        self._reg_name: str | None = None
        self._reg_progress: ttk.Progressbar | None = None

        self._setup_theme()
        root.title("FaceTrak")
        root.configure(bg=_BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.bind("<space>",  lambda _: self._register_dialog())
        root.bind("<r>",      lambda _: self._toggle_rec())
        root.bind("<R>",      lambda _: self._toggle_rec())
        root.bind("<b>",      lambda _: self._toggle_blur())
        root.bind("<B>",      lambda _: self._toggle_blur())
        root.bind("<h>",      lambda _: self._toggle_heatmap())
        root.bind("<H>",      lambda _: self._toggle_heatmap())
        root.bind("<Escape>", lambda _: self._stop() if self.eng.running else None)

        self._build_toolbar()
        self._build_main()
        self._build_statusbar()

    # ── theme ─────────────────────────────────────────────

    def _setup_theme(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=_BG)
        style.configure("Toolbar.TFrame", background=_SURFACE)
        style.configure("Sidebar.TFrame", background=_SURFACE)
        style.configure("Status.TFrame", background=_CARD)

        style.configure("TLabel", background=_BG, foreground=_TXT)
        style.configure("Toolbar.TLabel", background=_SURFACE, foreground=_TXT)
        style.configure("Sidebar.TLabel", background=_SURFACE, foreground=_TXT)
        style.configure("Status.TLabel", background=_CARD, foreground=_TXT2)

        style.configure("TButton", background=_CARD, foreground=_TXT,
                        bordercolor=_BORDER, focuscolor="none",
                        lightcolor=_CARD, darkcolor=_CARD)
        style.map("TButton",
                  background=[("active", _BLU), ("pressed", "#1D4ED8")],
                  foreground=[("active", "#FFF"), ("pressed", "#FFF")])

        style.configure("Accent.TButton", background=_BLU, foreground="#FFF",
                        bordercolor=_BLU, focuscolor="none",
                        lightcolor=_BLU, darkcolor=_BLU)
        style.map("Accent.TButton",
                  background=[("active", "#1D4ED8"), ("pressed", "#1E40AF")],
                  foreground=[("active", "#FFF"), ("pressed", "#FFF")])

        style.configure("TMenubutton", background=_CARD, foreground=_TXT,
                        bordercolor=_BORDER, focuscolor="none",
                        lightcolor=_CARD, darkcolor=_CARD)
        style.map("TMenubutton",
                  background=[("active", _SURFACE)],
                  foreground=[("active", _TXT)])

        style.configure("TCheckbutton", background=_SURFACE, foreground=_TXT,
                        focuscolor="none")
        style.map("TCheckbutton",
                  background=[("active", _SURFACE)],
                  foreground=[("active", _TXT)])

        style.configure("TCombobox", background=_CARD, foreground=_TXT,
                        fieldbackground=_CARD, arrowcolor=_TXT,
                        bordercolor=_BORDER, selectbackground=_BLU,
                        selectforeground="#FFF")
        style.map("TCombobox",
                  fieldbackground=[("readonly", _CARD)],
                  foreground=[("readonly", _TXT)])

        style.configure("TProgressbar", background=_GRN, troughcolor=_CARD,
                        bordercolor=_BORDER, lightcolor=_GRN, darkcolor=_GRN)

        style.configure("TSeparator", background=_BORDER)

        style.configure("Sunken.TLabel", background=_CARD, foreground=_TXT2,
                        relief=tk.SUNKEN, padding=4)

    # ── layout ───────────────────────────────────────────

    def _make_btn(self, parent, text, command, accent=False):
        style_name = "Accent.TButton" if accent else "TButton"
        return ttk.Button(parent, text=text, command=command, style=style_name)

    def _make_chk(self, parent, text, command):
        cb = ttk.Checkbutton(parent, text=text, command=command)
        cb.state(["!alternate"])
        return cb

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=6)
        bar.pack(fill=tk.X)

        # Group 1 — camera control
        self.b_start = self._make_btn(bar, "▶ Start", self._toggle_start, accent=True)
        self.b_start.pack(side=tk.LEFT, padx=2)

        self.b_rec = self._make_btn(bar, "● Rec [R]", self._toggle_rec)
        self.b_rec.pack(side=tk.LEFT, padx=2)
        self.b_rec.state(["disabled"])

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        # Group 2 — overlays
        self.cb_blur  = self._make_chk(bar, "Blur [B]", self._toggle_blur)
        self.cb_blur.pack(side=tk.LEFT, padx=2)
        self.cb_heat  = self._make_chk(bar, "Heatmap [H]", self._toggle_heatmap)
        self.cb_heat.pack(side=tk.LEFT, padx=2)
        self.cb_servo = self._make_chk(bar, "Servo", self._toggle_servo)
        self.cb_servo.pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        # Group 3 — registration
        self._make_btn(bar, "Register [Space]", self._register_dialog).pack(
            side=tk.LEFT, padx=2)
        self._make_btn(bar, "Faces", self._list_faces).pack(
            side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        # Group 4 — servo target
        ttk.Label(bar, text="Track:", style="Toolbar.TLabel").pack(side=tk.LEFT)
        self.servo_var = tk.StringVar(value=SERVO_TARGET_LARGEST)
        srv_combo = ttk.Combobox(bar, textvariable=self.servo_var,
                                 values=[SERVO_TARGET_LARGEST,
                                         SERVO_TARGET_KNOWN,
                                         SERVO_TARGET_UNKNOWN],
                                 state="readonly", width=9)
        srv_combo.pack(side=tk.LEFT, padx=2)
        srv_combo.bind("<<ComboboxSelected>>",
                       lambda _: self.eng.set_servo_target(self.servo_var.get()))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        # Group 5 — camera
        ttk.Label(bar, text="Cam:", style="Toolbar.TLabel").pack(side=tk.LEFT)
        self.cam_var   = tk.StringVar()
        self.cam_combo = ttk.Combobox(bar, textvariable=self.cam_var,
                                      state="readonly", width=22)
        self.cam_combo.pack(side=tk.LEFT, padx=2)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_cam_select)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        # Group 6 — sim
        self._make_btn(bar, "Sim", self._open_sim).pack(side=tk.LEFT, padx=2)

    def _build_main(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        self.l_video = tk.Label(frame, bg=_BG)
        self.l_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.Frame(frame, style="Sidebar.TFrame", width=_SPARK_W + 40, padding=8)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        ttk.Label(side, text="Emotion Trend", font=("", 10, "bold"),
                  style="Sidebar.TLabel").pack(anchor=tk.W)

        self.spark_canvas = tk.Canvas(side, width=_SPARK_W, height=_SPARK_H,
                                      bg=_CARD, highlightthickness=0)
        self.spark_canvas.pack(pady=(4, 12))

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        blur_header = ttk.Frame(side, style="Sidebar.TFrame")
        blur_header.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(blur_header, text="Per-Person Blur",
                  font=("", 10, "bold"),
                  style="Sidebar.TLabel").pack(side=tk.LEFT)

        list_frame = ttk.Frame(side, style="Sidebar.TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.blur_list = tk.Listbox(list_frame, height=8,
                                    bg=_CARD, fg=_TXT,
                                    selectbackground=_BLU,
                                    selectforeground="#FFF",
                                    borderwidth=0,
                                    highlightthickness=1,
                                    highlightbackground=_BORDER,
                                    highlightcolor=_BORDER,
                                    activestyle="none",
                                    yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.blur_list.yview)
        self.blur_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._make_btn(side, "Toggle Blur", self._toggle_person_blur).pack(
            fill=tk.X, pady=(4, 0))

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, style="Status.TFrame", padding=6)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.l_cam = ttk.Label(bar, style="Status.TLabel")
        self.l_cam.pack(side=tk.LEFT, padx=(0, 12))

        self.l_face = ttk.Label(bar, style="Status.TLabel")
        self.l_face.pack(side=tk.LEFT, padx=(0, 12))

        self.l_pose = ttk.Label(bar, style="Status.TLabel")
        self.l_pose.pack(side=tk.LEFT, padx=(0, 12))

        self.l_servo = ttk.Label(bar, style="Status.TLabel")
        self.l_servo.pack(side=tk.LEFT, padx=(0, 12))

        self.l_emotion = ttk.Label(bar, style="Status.TLabel")
        self.l_emotion.pack(side=tk.LEFT, padx=(0, 12))

        self.l_faces = ttk.Label(bar, style="Status.TLabel")
        self.l_faces.pack(side=tk.LEFT, padx=(0, 12))

        self.l_known = ttk.Label(bar, style="Status.TLabel")
        self.l_known.pack(side=tk.LEFT, padx=(0, 12))

        self.l_rec = ttk.Label(bar, style="Status.TLabel")
        self.l_rec.pack(side=tk.RIGHT)

    # ── camera ───────────────────────────────────────────

    def _populate_cameras(self):
        cfg    = config.load()
        cams   = cfg.get("cameras", [])
        labels = [config.label(cfg, i) for i in range(len(cams))]
        self.cam_combo["values"] = labels
        active = cfg.get("camera", 0)
        if 0 <= active < len(labels):
            self.cam_var.set(labels[active])

    def _on_cam_select(self, _=None):
        idx = self.cam_combo.current()
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
        self.b_start.config(text="■ Stop")
        self.b_rec.state(["!disabled"])
        self._populate_cameras()
        self.cam_combo.state(["!disabled"])
        self._refresh_blur_list()
        self._poll()

    def _stop(self):
        if self._poll_id:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        self.eng.stop()
        self.b_start.config(text="▶ Start")
        self.b_rec.state(["disabled"])
        self.b_rec.config(text="● Rec [R]")
        self.l_video.config(image="")
        self._clear_status()

    # ── poll loop ─────────────────────────────────────────

    def _poll(self):
        if not self.eng.running:
            return
        frame = self.eng.step()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if w > _MAX_W:
                scale = _MAX_W / w
                rgb   = cv2.resize(rgb, (_MAX_W, int(h * scale)))
            img = Image.fromarray(rgb)
            self._tk_img = ImageTk.PhotoImage(img)
            self.l_video.config(image=self._tk_img)
            self._update_status()
            self._draw_sparkline()
            rec_on = self.eng.recorder.recording
            self.b_rec.config(text="■ Rec [R]" if rec_on else "● Rec [R]")
        if self._reg_name and self.eng._capturing:
            from facetrak.engine import _MAX_SAMPLES, _MIN_REG_SAMPLES
            n = len(self.eng._samples_buffer)
            if self._reg_progress and n < _MAX_SAMPLES:
                self._reg_progress["value"] = int(100 * n / _MAX_SAMPLES)
            if n >= _MAX_SAMPLES or (n >= _MIN_REG_SAMPLES
                                      and self.eng.liveness.passed):
                name = self._reg_name
                self._reg_name = None
                self.root.after(0, lambda: self._do_register(name))
                return
        self._poll_id = self.root.after(_POLL_MS, self._poll)

    def _clear_status(self):
        for lbl in (self.l_cam, self.l_face, self.l_pose, self.l_servo,
                     self.l_emotion, self.l_faces, self.l_known, self.l_rec):
            lbl.config(text="")

    def _update_status(self):
        e   = self.eng
        cx, cy = e.last_face_center
        fw, fh = e.last_face_size
        cfg = config.load()
        cam = config.label(cfg, e.current_cam_idx)
        m   = e.metrics
        self.l_cam.config(text=f"📷 {cam}")
        self.l_face.config(text=f"Face: {cx},{cy} {fw}×{fh}")
        self.l_pose.config(text=f"Pose: {m.yaw:.0f}°/{m.pitch:.0f}°/{m.roll:.0f}°")
        self.l_servo.config(text=f"Servo: {e.current_pan:.0f}°/{e.current_tilt:.0f}°")
        self.l_emotion.config(text=f"Emotion: {m.emotion or '—'}")
        self.l_faces.config(text=f"Faces: {len(e.tracker.active)}")
        self.l_known.config(text=f"Known: {len(e.db.known_names)}")
        rec_on = e.recorder.recording
        self.l_rec.config(text="🔴 REC" if rec_on else "", foreground=_RED if rec_on else _TXT3)

    # ── sparkline ────────────────────────────────────────

    def _draw_sparkline(self):
        counts = self.eng.timeline.recent_emotion_counts(60)
        c = self.spark_canvas
        c.delete("all")
        if not counts:
            return
        emotions = sorted(counts.items(), key=lambda x: -x[1])
        total = sum(v for _, v in emotions)
        x = 2
        bar_w = max(4, (_SPARK_W - 4) // len(emotions))
        for emotion, cnt in emotions:
            bar_h = int((_SPARK_H - 8) * cnt / total)
            col   = _EMO_COLORS.get(emotion, "#6c7086")
            c.create_rectangle(x, _SPARK_H - bar_h - 4, x + bar_w - 2,
                                _SPARK_H - 4, fill=col, outline="")
            pct = int(100 * cnt / total)
            c.create_text(x + bar_w // 2, _SPARK_H - bar_h - 10,
                          text=f"{emotion[:3]} {pct}%",
                          font=("Consolas", 7), fill=col)
            x += bar_w

    # ── controls ─────────────────────────────────────────

    def _toggle_rec(self):
        self.eng.toggle_record()

    def _toggle_blur(self):
        state = self.eng.toggle_blur()
        self.cb_blur.state(["selected" if state else "!selected"])

    def _toggle_heatmap(self):
        state = self.eng.toggle_heatmap()
        self.cb_heat.state(["selected" if state else "!selected"])

    def _toggle_servo(self):
        state = self.eng.toggle_servo()
        self.cb_servo.state(["selected" if state else "!selected"])

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
        win = tk.Toplevel(self.root, bg=_SURFACE)
        win.title("Registering…")
        win.geometry("320x120")
        win.resizable(False, False)
        ttk.Label(win, text=f"Capturing samples for \"{name}\"…",
                  font=("", 11)).pack(pady=(16, 8))
        self._reg_progress = ttk.Progressbar(win, length=260, mode="determinate")
        self._reg_progress.pack(pady=4)
        ttk.Label(win, text="Look at camera, blink twice, turn head",
                  foreground=_TXT2).pack(pady=4)
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
            w = self._reg_progress.master if hasattr(self._reg_progress, "master") else None
            if w:
                try:
                    w.destroy()
                except tk.TclError:
                    pass
            self._reg_progress = None
        if ok:
            messagebox.showinfo("Done", f"Registered '{name}'.")
        else:
            messagebox.showerror("Error", "Not enough samples — try again.")

    # ── blur list ────────────────────────────────────────

    def _refresh_blur_list(self):
        self.blur_list.delete(0, tk.END)
        for name in self.eng.db.known_names:
            self.blur_list.insert(tk.END, name)
            if name in self.eng.blur_persons:
                self.blur_list.itemconfig(tk.END, bg="#3B1F25", fg=_RED)

    def _toggle_person_blur(self):
        sel   = self.blur_list.curselection()
        names = self.eng.db.known_names
        for i in sel:
            name      = names[i]
            currently = name in self.eng.blur_persons
            self.eng.set_blur_person(name, not currently)
        self._refresh_blur_list()

    # ── dialogs ──────────────────────────────────────────

    def _list_faces(self):
        names = self.eng.db.known_names
        if not names:
            messagebox.showinfo("Known Faces", "(none)")
            return
        text = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names))
        messagebox.showinfo("Known Faces", text)

    def _open_sim(self):
        self._sim_win = SimulationWindow(
            self.root,
            lambda: (self.eng.current_pan, self.eng.current_tilt))

    def _on_close(self):
        self._stop()
        self.root.destroy()
