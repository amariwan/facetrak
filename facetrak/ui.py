"""Tkinter main window.

Features:
  - Hotkeys: Space=Register, R=Record, B=Blur, H=Heatmap, Esc=Stop
  - Registration wizard: live progress bar, liveness status, auto-complete
  - Per-person blur: select name from list, toggle blur
  - Emotion sparkline: bar chart of recent emotion counts
  - Servo target selector: Largest / Known / Unknown
"""
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
_SPARK_W = 160
_SPARK_H = 60


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.eng  = FaceEngine()
        self._poll_id = None
        self._reg_name: str | None = None

        root.title("FaceTrak")
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
        self._build_bottom()

    # ── layout ───────────────────────────────────────────

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=4)
        bar.pack(fill=tk.X)

        self.b_start = ttk.Button(bar, text="Start", command=self._toggle_start)
        self.b_start.pack(side=tk.LEFT, padx=2)

        self.b_rec = ttk.Button(bar, text="● Rec [R]", command=self._toggle_rec)
        self.b_rec.pack(side=tk.LEFT, padx=2)
        self.b_rec.state(["disabled"])

        self.cb_blur = ttk.Checkbutton(bar, text="Blur [B]",
                                       command=self._toggle_blur)
        self.cb_blur.state(["!alternate"])
        self.cb_blur.pack(side=tk.LEFT, padx=6)

        self.cb_heat = ttk.Checkbutton(bar, text="Heatmap [H]",
                                       command=self._toggle_heatmap)
        self.cb_heat.state(["!alternate"])
        self.cb_heat.pack(side=tk.LEFT, padx=2)

        self.cb_servo = ttk.Checkbutton(bar, text="Servo",
                                        command=self._toggle_servo)
        self.cb_servo.state(["!alternate"])
        self.cb_servo.pack(side=tk.LEFT, padx=2)

        ttk.Button(bar, text="Register [Space]",
                   command=self._register_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Faces",
                   command=self._list_faces).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Sim",
                   command=self._open_sim).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(bar, text="Servo:").pack(side=tk.LEFT)
        self.servo_var = tk.StringVar(value=SERVO_TARGET_LARGEST)
        srv_combo = ttk.Combobox(bar, textvariable=self.servo_var,
                                 values=[SERVO_TARGET_LARGEST,
                                         SERVO_TARGET_KNOWN,
                                         SERVO_TARGET_UNKNOWN],
                                 state="readonly", width=10)
        srv_combo.pack(side=tk.LEFT, padx=2)
        srv_combo.bind("<<ComboboxSelected>>",
                       lambda _: self.eng.set_servo_target(self.servo_var.get()))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        self.cam_var   = tk.StringVar()
        self.cam_combo = ttk.Combobox(bar, textvariable=self.cam_var,
                                      state="readonly", width=22)
        self.cam_combo.pack(side=tk.LEFT, padx=2)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_cam_select)

    def _build_main(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        self.l_video = tk.Label(frame, bg="#111")
        self.l_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel: emotion sparkline + per-person blur manager
        side = ttk.Frame(frame, width=_SPARK_W + 20, padding=6)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        ttk.Label(side, text="Emotion trend").pack()
        self.spark_canvas = tk.Canvas(side, width=_SPARK_W, height=_SPARK_H,
                                      bg="#1e1e2e", highlightthickness=0)
        self.spark_canvas.pack(pady=(0, 10))

        ttk.Label(side, text="Per-person blur:").pack(anchor=tk.W)
        self.blur_list = tk.Listbox(side, height=8, selectmode=tk.MULTIPLE)
        self.blur_list.pack(fill=tk.X)
        ttk.Button(side, text="Toggle blur",
                   command=self._toggle_person_blur).pack(fill=tk.X, pady=2)

    def _build_bottom(self):
        self.l_status = ttk.Label(self.root, relief=tk.SUNKEN, anchor=tk.W)
        self.l_status.pack(fill=tk.X, side=tk.BOTTOM)

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
        self.b_start.config(text="Stop")
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
        self.b_start.config(text="Start")
        self.b_rec.state(["disabled"])
        self.b_rec.config(text="● Rec [R]")
        self.l_video.config(image="")

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
        # Registration auto-complete check
        if self._reg_name and self.eng._capturing:
            from facetrak.engine import _MAX_SAMPLES, _MIN_REG_SAMPLES
            n = len(self.eng._samples_buffer)
            if n >= _MAX_SAMPLES or (n >= _MIN_REG_SAMPLES
                                     and self.eng.liveness.passed):
                name = self._reg_name
                self._reg_name = None
                self.root.after(0, lambda: self._do_register(name))
                return
        self._poll_id = self.root.after(_POLL_MS, self._poll)

    def _update_status(self):
        e   = self.eng
        cx, cy = e.last_face_center
        fw, fh = e.last_face_size
        cfg = config.load()
        cam = config.label(cfg, e.current_cam_idx)
        m   = e.metrics
        self.l_status.config(text=(
            f"{cam}  |  Face:({cx},{cy}) {fw}×{fh}  |  "
            f"S:{e.current_pan:.0f}/{e.current_tilt:.0f}  |  "
            f"Pose:{m.yaw:.0f}/{m.pitch:.0f}/{m.roll:.0f}  |  "
            f"Gaze:{m.gaze_label}  Emotion:{m.emotion}  |  "
            f"Faces:{len(e.tracker.active)}  |  "
            f"{'REC' if e.recorder.recording else '   '}  |  "
            f"K:{len(e.db.known_names)}"
        ))

    def _draw_sparkline(self):
        counts = self.eng.timeline.recent_emotion_counts(60)
        c = self.spark_canvas
        c.delete("all")
        if not counts:
            return
        emotions = sorted(counts.items(), key=lambda x: -x[1])
        total = sum(v for _, v in emotions)
        colors = {"happy": "#a6e3a1", "sad": "#89b4fa", "angry": "#f38ba8",
                  "surprised": "#fab387", "neutral": "#cdd6f4"}
        x = 2
        bar_w = max(4, (_SPARK_W - 4) // len(emotions))
        for emotion, cnt in emotions:
            bar_h = int((_SPARK_H - 4) * cnt / total)
            col   = colors.get(emotion, "#6c7086")
            c.create_rectangle(x, _SPARK_H - bar_h - 2, x + bar_w - 2,
                                _SPARK_H - 2, fill=col, outline="")
            c.create_text(x + bar_w // 2, _SPARK_H - bar_h - 6,
                          text=emotion[:3], font=("Consolas", 7), fill=col)
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

    def _do_register(self, name: str):
        ok = self.eng.register(name)
        self.eng.set_overlay("")
        self._refresh_blur_list()
        if ok:
            messagebox.showinfo("Done", f"Registered '{name}'.")
        else:
            messagebox.showerror("Error", "Not enough samples — try again.")

    def _refresh_blur_list(self):
        self.blur_list.delete(0, tk.END)
        for name in self.eng.db.known_names:
            self.blur_list.insert(tk.END, name)
            if name in self.eng.blur_persons:
                self.blur_list.itemconfig(tk.END, bg="#f38ba8")

    def _toggle_person_blur(self):
        sel   = self.blur_list.curselection()
        names = self.eng.db.known_names
        for i in sel:
            name      = names[i]
            currently = name in self.eng.blur_persons
            self.eng.set_blur_person(name, not currently)
        self._refresh_blur_list()

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
