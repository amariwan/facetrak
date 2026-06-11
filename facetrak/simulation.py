import tkinter as tk
import math


class SimulationWindow:
    def __init__(self, root: tk.Tk, get_angles):
        self.win = tk.Toplevel(root)
        self.win.title("Pan-Tilt Simulation")
        self.win.geometry("380x420")
        self.win.resizable(False, False)

        self.get_angles = get_angles
        self.cvs = tk.Canvas(self.win, width=360, height=360, bg="#1e1e2e",
                             highlightthickness=0)
        self.cvs.pack(pady=(10, 0))

        self.l_info = tk.Label(self.win, font=("Consolas", 11),
                               fg="#cdd6f4", bg="#1e1e2e")
        self.l_info.pack(fill=tk.X, padx=10, pady=(4, 10))

        self.cx, self.cy = 180, 200
        self._anim_id = None
        self._poll()

    def _poll(self):
        pan, tilt = self.get_angles()
        self._draw(pan, tilt)
        self.l_info.config(
            text=f"Pan: {pan:>6.1f}°    Tilt: {tilt:>6.1f}°")
        self._anim_id = self.win.after(40, self._poll)

    def close(self):
        if self._anim_id:
            self.win.after_cancel(self._anim_id)
        self.win.destroy()

    def _draw(self, pan: float, tilt: float):
        c = self.cvs
        c.delete("all")

        bg = "#1e1e2e"
        base_col = "#45475a"
        pole_col = "#585b70"
        head_col = "#89b4fa"
        lens_col = "#a6e3a1"
        accent = "#f5c2e7"

        cx, cy = self.cx, self.cy
        pr = math.radians(pan)
        tr = math.radians(90 - tilt)

        # ── ground ellipse ──
        c.create_oval(cx - 70, cy + 10, cx + 70, cy + 30,
                      fill=base_col, outline="#6c7086", width=2)

        # ── pan base (rotates with pan) ──
        bx = cx + 30 * math.sin(pr)
        b_scale = 0.5 + 0.5 * abs(math.cos(pr))
        base_w = 50 * b_scale
        base_h = 16
        c.create_oval(cx - base_w, cy - base_h, cx + base_w, cy + base_h,
                      fill="#585b70", outline="#6c7086", width=2)

        # ── pole ──
        pole_len = 100
        pole_top = cy - base_h - pole_len
        pole_x = cx + 8 * math.sin(pr)
        pole_w = 10
        c.create_rectangle(pole_x - pole_w // 2, pole_top,
                           pole_x + pole_w // 2, cy - base_h,
                           fill=pole_col, outline="#6c7086", width=1)

        # ── tilt joint ──
        jr = 10
        jx = pole_x
        jy = pole_top
        c.create_oval(jx - jr, jy - jr, jx + jr, jy + jr,
                      fill="#6c7086", outline="#9399b2", width=2)

        # ── camera head (tilts) ──
        head_len = 60
        head_w = 22
        hx = jx + head_len * math.cos(tr) * math.sin(pr)
        hy = jy - head_len * math.sin(tr)

        # head body as rotated rectangle — approximate with polygon
        hw2 = head_w / 2
        hl2 = head_len / 2
        pts = []
        for lx, ly in [(-hl2, -hw2), (hl2, -hw2), (hl2, hw2), (-hl2, hw2)]:
            rx = lx * math.cos(tr) - ly * math.sin(tr)
            ry = lx * math.sin(tr) + ly * math.cos(tr)
            pts.append(jx + rx)
            pts.append(jy - ry - 10)

        c.create_polygon(*pts, fill=head_col, outline="#b4befe", width=2)

        # ── lens (front of camera) ──
        lens_x = jx + (head_len - 8) * math.cos(tr) * math.sin(pr)
        lens_y = jy - (head_len - 8) * math.sin(tr)
        lr = 8
        c.create_oval(lens_x - lr, lens_y - lr, lens_x + lr, lens_y + lr,
                      fill=lens_col, outline="#a6e3a1", width=1)

        # ── light beam (forward direction hint) ──
        beam_len = 40
        bx2 = lens_x + beam_len * math.cos(tr) * math.sin(pr)
        by2 = lens_y - beam_len * math.sin(tr)
        c.create_line(lens_x, lens_y, bx2, by2,
                      fill="#f5e0dc", width=1, dash=(4, 4))

        # ── angle arc labels ──
        c.create_text(60, 30, text=f"{pan:7.1f}°", anchor="w",
                      font=("Consolas", 13), fill="#fab387")
        c.create_text(60, 55, text="pan", anchor="w",
                      font=("Consolas", 9), fill="#6c7086")

        c.create_text(300, 30, text=f"{tilt:7.1f}°", anchor="e",
                      font=("Consolas", 13), fill="#89b4fa")
        c.create_text(300, 55, text="tilt", anchor="e",
                      font=("Consolas", 9), fill="#6c7086")

        # ── compass arc (pan reference) ──
        c.create_arc(cx - 80, cy - 40, cx + 80, cy + 40,
                     start=90, extent=-pan - 90,
                     outline="#fab387", width=2, style="arc")
        c.create_line(cx, cy, cx + 60 * math.sin(pr), cy - 60 * math.cos(pr),
                      fill="#fab387", width=1.5, dash=(3, 3))

        # ── tilt arc ──
        arc_r = 50
        c.create_arc(jx - arc_r, jy - arc_r, jx + arc_r, jy + arc_r,
                     start=0, extent=-tilt + 90,
                     outline="#89b4fa", width=2, style="arc",
                     )

        # ── title ──
        c.create_text(180, 340, text="PAN / TILT",
                      font=("Consolas", 8, "bold"), fill="#6c7086")


def demo():
    root = tk.Tk()
    root.withdraw()
    import math, time
    angles = [90.0, 90.0]
    def get():
        t = time.time()
        angles[0] = 90 + 45 * math.sin(t * 0.5)
        angles[1] = 70 + 40 * math.sin(t * 0.7 + 1)
        return angles[0], angles[1]
    sw = SimulationWindow(root, get)
    root.mainloop()
