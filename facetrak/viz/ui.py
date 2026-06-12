"""Main window — orchestrates all UI components."""
import math
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from facetrak.core import config
from facetrak.core.engine import FaceEngine
from .components import CommandBar, Sidebar, StatusBar, VideoPanel
from . import theme as T
from .simulation import SimulationWindow

_POLL_MS = 30


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.eng  = FaceEngine()
        self._poll_id  = None
        self._anim_id  = None
        self._reg_name: str | None = None
        self._reg_progress = None

        self._setup_theme()

        root.title("FaceTrak")
        root.configure(bg=T.BG_ROOT)
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
        self._animate()

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _setup_theme(self):
        s = ttk.Style()
        s.theme_use("clam")
        for w in ("TFrame", "TLabel", "TButton", "TCheckbutton",
                  "TCombobox", "TProgressbar", "TSeparator"):
            s.configure(w, background=T.BG_ROOT)
        s.configure("TLabel",     foreground=T.TEXT_PRIMARY, background=T.BG_ROOT)
        s.configure("TSeparator", background=T.BORDER_SUBTLE)
        s.configure("TProgressbar", background=T.ACCENT,
                    troughcolor=T.BG_RAISED, borderwidth=0,
                    lightcolor=T.ACCENT, darkcolor=T.ACCENT)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        callbacks = {
            "toggle_start":    self._toggle_start,
            "toggle_rec":      self._toggle_rec,
            "toggle_blur":     self._toggle_blur,
            "toggle_heatmap":  self._toggle_heatmap,
            "toggle_servo":    self._toggle_servo,
            "register_dialog": self._register_dialog,
            "list_faces":      self._list_faces,
            "set_servo_target": self.eng.set_servo_target,
            "on_cam_select":   self._on_cam_select,
            "open_sim":        self._open_sim,
        }

        self._cmd_bar = CommandBar(self.root, callbacks)
        self._cmd_bar.grid(row=0, column=0, sticky=tk.EW)

        # Content area
        main = tk.Frame(self.root, bg=T.BG_ROOT)
        main.grid(row=1, column=0, sticky=tk.NSEW, padx=14, pady=14)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self._video = VideoPanel(main)
        self._video.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 14))

        self._sidebar = Sidebar(main, on_blur_toggle=self._toggle_person_blur)
        self._sidebar.grid(row=0, column=1, sticky=tk.NS)

        self._status = StatusBar(self.root)
        self._status.grid(row=2, column=0, sticky=tk.EW, padx=14, pady=(0, 8))

    # ── Camera control ────────────────────────────────────────────────────────

    def _populate_cameras(self):
        cfg = config.load()
        cams = cfg.get("cameras", [])
        labels = [config.label(cfg, i) for i in range(len(cams))]
        active = cfg.get("camera", 0)
        active_label = labels[active] if 0 <= active < len(labels) else ""
        self._cmd_bar.set_cam_values(labels, active_label)

    def _on_cam_select(self, _=None):
        idx = self._cmd_bar.cam_cb.current()
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

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def _toggle_start(self):
        if self.eng.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        if not self.eng.start():
            messagebox.showerror("Error", "Cannot open camera.")
            return
        self._cmd_bar.set_running(True)
        self._populate_cameras()
        self._refresh_blur_list()
        self._poll()

    def _stop(self):
        if self._poll_id:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        self.eng.stop()
        self._cmd_bar.set_running(False)
        self._video.clear()
        self._status.clear()

    # ── Poll loop ─────────────────────────────────────────────────────────────

    def _poll(self):
        if not self.eng.running:
            return

        frame = self.eng.step()
        if frame is not None:
            self._video.show_frame(frame)
            self._video.update_strip(self.eng.tracker.active, self.eng._frame)

            m   = self.eng.metrics
            pri = self.eng.tracker.largest()
            cfg = config.load()

            self._sidebar.update_telemetry(pri, m, self.eng)
            self._sidebar.update_attention(m)
            self._sidebar.update_emotion(self.eng.timeline)
            self._status.update(self.eng, m, cfg)

            rec = self.eng.recorder.recording
            self._cmd_bar.set_recording(rec)
            self._status.set_recording(rec)

        # Registration progress
        if self._reg_name and self.eng._capturing:
            from facetrak.core.engine import _MAX_SAMPLES, _MIN_REG_SAMPLES
            n = len(self.eng._samples_buffer)
            if self._reg_progress:
                self._reg_progress["value"] = int(100 * n / _MAX_SAMPLES)
            if n >= _MAX_SAMPLES or (n >= _MIN_REG_SAMPLES and self.eng.liveness.passed):
                name = self._reg_name
                self._reg_name = None
                self.root.after(0, lambda: self._do_register(name))
                return

        self._poll_id = self.root.after(_POLL_MS, self._poll)

    # ── Animate ───────────────────────────────────────────────────────────────

    def _animate(self):
        """Pulsing rec dot."""
        if self.eng.running and self.eng.recorder.recording:
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 4)
            r = int(3 + pulse * 1.5)
            dot = self._status._rec_dot
            dot.delete("dot")
            dot.create_oval(6 - r, 6 - r, 6 + r, 6 + r,
                            fill=T.DANGER, outline="", tags="dot")
        self._anim_id = self.root.after(100, self._animate)

    # ── Toggles ───────────────────────────────────────────────────────────────

    def _toggle_rec(self):      self.eng.toggle_record()
    def _toggle_blur(self):     self.eng.toggle_blur()
    def _toggle_heatmap(self):  self.eng.toggle_heatmap()
    def _toggle_servo(self):    self.eng.toggle_servo()

    # ── Registration ─────────────────────────────────────────────────────────

    def _register_dialog(self):
        if not self.eng.running:
            messagebox.showinfo("Notice", "Start the camera first.")
            return
        name = simpledialog.askstring("Register Subject", "Enter name:")
        if not name:
            return
        self._reg_name = name
        self.eng.set_overlay("Blink twice and turn head slowly.")
        self.eng.capture_samples()
        self._show_reg_progress(name)

    def _show_reg_progress(self, name: str):
        win = tk.Toplevel(self.root, bg=T.BG_SURFACE)
        win.title("Scanning")
        win.geometry("340x130")
        win.resizable(False, False)

        # Header
        tk.Frame(win, bg=T.ACCENT, height=3).pack(fill=tk.X)

        tk.Label(win, text=f"Scanning  {name}",
                 fg=T.TEXT_PRIMARY, bg=T.BG_SURFACE,
                 font=T.FONT_TITLE).pack(pady=(18, 4))
        tk.Label(win, text="Blink twice and slowly turn your head",
                 fg=T.TEXT_SECONDARY, bg=T.BG_SURFACE,
                 font=T.FONT_MICRO).pack()

        bar = ttk.Progressbar(win, length=300, mode="determinate")
        bar.pack(pady=(12, 0))
        self._reg_progress = bar

        self.root.after(6000, lambda: self._close_reg_win(win))

    def _close_reg_win(self, win: tk.Toplevel):
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
            messagebox.showinfo("Success", f"'{name}' registered.")
        else:
            messagebox.showerror("Error", "Insufficient samples — try again.")

    # ── Privacy list ──────────────────────────────────────────────────────────

    def _refresh_blur_list(self):
        lb = self._sidebar.listbox
        lb.delete(0, tk.END)
        for name in self.eng.db.known_names:
            lb.insert(tk.END, f"  {name}")
            if name in self.eng.blur_persons:
                lb.itemconfig(tk.END, bg=T.BG_ROOT, fg=T.TEXT_MUTED)
            else:
                lb.itemconfig(tk.END, bg=T.BG_RAISED, fg=T.TEXT_PRIMARY)

    def _toggle_person_blur(self):
        lb = self._sidebar.listbox
        sel = lb.curselection()
        names = self.eng.db.known_names
        for i in sel:
            name = names[i]
            self.eng.set_blur_person(name, name not in self.eng.blur_persons)
        self._refresh_blur_list()

    def _list_faces(self):
        names = self.eng.db.known_names
        if not names:
            messagebox.showinfo("Database", "No subjects registered.")
            return
        text = "\n".join(f"  {T.ICON_DOT}  {n}" for n in names)
        blurred = ", ".join(sorted(self.eng.blur_persons))
        if blurred:
            text += f"\n\nBlurred:\n  {blurred}"
        messagebox.showinfo("Subject Directory", text)

    # ── Simulation ────────────────────────────────────────────────────────────

    def _open_sim(self):
        self._sim_win = SimulationWindow(
            self.root,
            lambda: (self.eng.current_pan, self.eng.current_tilt),
        )

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop()
        if self._anim_id:
            self.root.after_cancel(self._anim_id)
        self.root.destroy()
