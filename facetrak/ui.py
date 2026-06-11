import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
from facetrak import config
from facetrak.engine import FaceEngine
from facetrak.simulation import SimulationWindow


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.eng = FaceEngine()
        self._poll_id = None

        root.title("FaceTrak")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_toolbar()
        self._build_main()
        self._build_status()

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=4)
        bar.pack(fill=tk.X)

        self.b_start = ttk.Button(bar, text="Start", command=self._toggle_start)
        self.b_start.pack(side=tk.LEFT, padx=2)

        self.b_rec = ttk.Button(bar, text="● Record", command=self._toggle_rec)
        self.b_rec.pack(side=tk.LEFT, padx=2)
        self.b_rec.state(["disabled"])

        self.cb_blur = ttk.Checkbutton(
            bar, text="Blur", command=self._toggle_blur)
        self.cb_blur.state(["!alternate"])
        self.cb_blur.pack(side=tk.LEFT, padx=8)

        self.cb_servo = ttk.Checkbutton(
            bar, text="Servo", command=self._toggle_servo)
        self.cb_servo.state(["!alternate"])
        self.cb_servo.pack(side=tk.LEFT, padx=2)

        ttk.Button(bar, text="Register", command=self._register_dialog
                   ).pack(side=tk.LEFT, padx=2)

        ttk.Button(bar, text="List Faces", command=self._list_faces
                   ).pack(side=tk.LEFT, padx=2)

        ttk.Button(bar, text="Simulation", command=self._open_sim
                   ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(bar, textvariable=self.cam_var,
                                      state="readonly", width=22)
        self.cam_combo.pack(side=tk.LEFT, padx=2)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_cam_select)

        self.l_port = ttk.Label(bar, text="")
        self.l_port.pack(side=tk.RIGHT, padx=4)

    def _build_main(self):
        self.l_video = tk.Label(self.root, bg="#111")
        self.l_video.pack()

    def _build_status(self):
        self.l_status = ttk.Label(self.root, relief=tk.SUNKEN, anchor=tk.W)
        self.l_status.pack(fill=tk.X, side=tk.BOTTOM)

    def _populate_cameras(self):
        cfg = config.load()
        cams = cfg.get("cameras", [])
        labels = [config.label(cfg, i) for i in range(len(cams))]
        self.cam_combo["values"] = labels
        active = cfg.get("camera", 0)
        if 0 <= active < len(labels):
            self.cam_var.set(labels[active])

    def _on_cam_select(self, _event=None):
        idx = self.cam_combo.current()
        if idx < 0:
            return
        if not self.eng.running:
            cfg = config.load()
            cfg["camera"] = idx
            config.save(cfg)
            self.eng.current_cam_idx = idx
            return
        was_rec = self.eng.recorder.recording
        ok = self.eng.switch_camera(idx)
        if not ok:
            self.eng.start(cfg.get("camera", 0))
            messagebox.showerror("Error", f"Failed to switch camera.")

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
        self.cb_blur.state(["!disabled"])
        self._populate_cameras()
        self.cam_combo.state(["!disabled"])
        self._poll()

    def _stop(self):
        if self._poll_id:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        self.eng.stop()
        self.b_start.config(text="Start")
        self.b_rec.state(["disabled"])
        self.b_rec.config(text="● Record")
        self.l_video.config(image="")

    def _poll(self):
        if not self.eng.running:
            return
        frame = self.eng.step()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            max_w = 900
            if w > max_w:
                scale = max_w / w
                h, w = int(h * scale), max_w
                rgb = cv2.resize(rgb, (w, h))
            img = Image.fromarray(rgb)
            self._tk_img = ImageTk.PhotoImage(img)
            self.l_video.config(image=self._tk_img)
            cx, cy = self.eng.last_face_center
            fw, fh = self.eng.last_face_size
            cfg = config.load()
            cam_name = config.label(cfg, self.eng.current_cam_idx)
            status = (
                f"{cam_name}  |  "
                f"Face: ({cx}, {cy}) {fw}x{fh}  |  "
                f"S:{self.eng.current_pan:.0f}/{self.eng.current_tilt:.0f}  |  "
                f"P:{self.eng.current_yaw:.0f}/{self.eng.current_pitch:.0f}/{self.eng.current_roll:.0f}  |  "
                f"{'REC' if self.eng.recorder.recording else '   '}  |  "
                f"K:{len(self.eng.db.known_names)} known"
            )
            self.l_status.config(text=status)
            rec_on = self.eng.recorder.recording
            self.b_rec.config(text="■ Record" if rec_on else "● Record")
        self._poll_id = self.root.after(30, self._poll)

    def _toggle_rec(self):
        self.eng.toggle_record()

    def _toggle_blur(self):
        state = self.eng.toggle_blur()
        self.cb_blur.state(["selected" if state else "!selected"])

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
        self.eng.set_overlay("Look at camera... registering")
        self.eng.capture_samples()
        self.root.after(3000, lambda: self._do_register(name))

    def _do_register(self, name: str):
        ok = self.eng.register(name)
        self.eng.set_overlay("")
        if ok:
            messagebox.showinfo("Done", f"Registered '{name}'.")
        else:
            messagebox.showerror("Error", "No samples captured.")

    def _list_faces(self):
        names = self.eng.db.known_names
        if not names:
            messagebox.showinfo("Known Faces", "(none)")
            return
        text = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names))
        messagebox.showinfo("Known Faces", text)

    def _open_sim(self):
        self._sim_win = SimulationWindow(
            self.root, lambda: (self.eng.current_pan, self.eng.current_tilt))

    def _on_close(self):
        self._stop()
        self.root.destroy()
