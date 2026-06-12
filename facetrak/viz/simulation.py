"""3D pan-tilt simulation window — themed to match the main workspace."""
import math
import time
import tkinter as tk

from . import theme as T

_W, _H = 460, 520


class _Vector3:
    @staticmethod
    def rotate_x(x, y, z, angle_deg):
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        return x, y * c - z * s, y * s + z * c

    @staticmethod
    def rotate_y(x, y, z, angle_deg):
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        return x * c + z * s, y, -x * s + z * c


class SimulationWindow:
    def __init__(self, root: tk.Tk, get_angles_callback):
        self.win = tk.Toplevel(root)
        self.win.title("Pan-Tilt Simulation")
        self.win.geometry(f"{_W}x{_H}")
        self.win.configure(bg=T.BG_ROOT)
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.get_angles = get_angles_callback

        self.cvs = tk.Canvas(self.win, width=_W, height=_H,
                              bg=T.BG_ROOT, highlightthickness=0)
        self.cvs.pack(fill=tk.BOTH, expand=True)

        self.cx, self.cy = _W // 2, 310

        self._anim_id = None
        self._poll()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            pan, tilt = self.get_angles()
            self._draw_scene(pan, tilt)
        except Exception:
            import traceback
            traceback.print_exc()
        self._anim_id = self.win.after(30, self._poll)

    def close(self):
        if self._anim_id:
            self.win.after_cancel(self._anim_id)
        self.win.destroy()

    # ── Projection ────────────────────────────────────────────────────────────

    def _project(self, x, y, z):
        x, y, z = _Vector3.rotate_x(x, y, z, 15)
        return self.cx + x, self.cy - y

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, pan, tilt):
        c = self.cvs

        # Header strip
        c.create_rectangle(0, 0, _W, 56, fill=T.BG_SURFACE, outline="")
        c.create_rectangle(0, 56, _W, 57, fill=T.BORDER_SUBTLE, outline="")
        c.create_rectangle(0, 0, 3, 56, fill=T.ACCENT, outline="")

        c.create_text(20, 22, text="PTZ TRACKER", anchor="w",
                       font=T.FONT_TITLE, fill=T.TEXT_PRIMARY)
        c.create_text(20, 40, text="System Online  ·  Tracking Active",
                       anchor="w", font=T.FONT_MICRO, fill=T.TEXT_SECONDARY)

        # Live dot
        pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 3)
        r = 3 + pulse * 1.5
        c.create_oval(_W - 28 - r, 28 - r, _W - 28 + r, 28 + r,
                       fill=T.SUCCESS, outline="")

        # Angle readouts
        self._angle_block(20, 80, "PAN  ·  Y-AXIS", pan, T.WARNING, anchor="w")
        self._angle_block(_W - 20, 80, "TILT  ·  X-AXIS", tilt, T.INFO, anchor="e")

        # Crosshair
        c.create_line(self.cx - 10, self.cy, self.cx + 10, self.cy,
                       fill=T.BORDER_MUTED)
        c.create_line(self.cx, self.cy - 10, self.cx, self.cy + 10,
                       fill=T.BORDER_MUTED)

    def _angle_block(self, x, y, label, value, color, anchor="w"):
        c = self.cvs
        c.create_text(x, y, text=label, anchor=anchor,
                       font=T.FONT_MICRO, fill=T.TEXT_MUTED)
        c.create_text(x, y + 24, text=f"{value:05.1f}°", anchor=anchor,
                       font=T.font("mono", 22), fill=color)

    # ── Scene ─────────────────────────────────────────────────────────────────

    def _draw_scene(self, pan: float, tilt: float):
        self.cvs.delete("all")
        c = self.cvs
        self._draw_hud(pan, tilt)

        pole_height = 90
        self._draw_ellipse(0, 0, 0, radius_x=70, radius_z=70,
                            color=T.BG_SURFACE, outline=T.BORDER_MUTED)

        cam_length, cam_width, cam_height = 80, 30, 30

        vertices = [
            ( cam_width/2,  cam_height/2,  cam_length/2),
            (-cam_width/2,  cam_height/2,  cam_length/2),
            (-cam_width/2, -cam_height/2,  cam_length/2),
            ( cam_width/2, -cam_height/2,  cam_length/2),
            ( cam_width/2,  cam_height/2, -cam_length/2),
            (-cam_width/2,  cam_height/2, -cam_length/2),
            (-cam_width/2, -cam_height/2, -cam_length/2),
            ( cam_width/2, -cam_height/2, -cam_length/2),
        ]

        transformed = []
        for vx, vy, vz in vertices:
            rx, ry, rz = _Vector3.rotate_x(vx, vy, vz, tilt - 90)
            rx, ry, rz = _Vector3.rotate_y(rx, ry, rz, -pan)
            transformed.append((rx, ry + pole_height, rz))

        proj = [self._project(x, y, z) for x, y, z in transformed]

        # Pole
        p_base = self._project(0, 0, 0)
        p_top  = self._project(0, pole_height, 0)
        c.create_line(*p_base, *p_top, fill=T.BG_OVERLAY, width=12,
                       capstyle=tk.ROUND)
        c.create_line(*p_base, *p_top, fill=T.TEXT_MUTED, width=4,
                       capstyle=tk.ROUND)

        front = proj[0:4]
        back  = proj[4:8]

        # Wireframe edges
        for i in range(4):
            c.create_line(front[i][0], front[i][1], back[i][0], back[i][1],
                           fill=T.BG_OVERLAY, width=2)

        c.create_polygon(*[xy for pt in back for xy in pt],
                          fill=T.BG_ROOT, outline=T.BG_OVERLAY, width=2)
        c.create_polygon(*[xy for pt in front for xy in pt],
                          fill=T.BG_RAISED, outline=T.ACCENT, width=2)

        # Lens
        lx = sum(p[0] for p in front) / 4
        ly = sum(p[1] for p in front) / 4
        c.create_oval(lx - 8, ly - 8, lx + 8, ly + 8,
                       fill=T.BG_ROOT, outline=T.SUCCESS, width=2)
        c.create_oval(lx - 3, ly - 3, lx + 3, ly + 3,
                       fill=T.SUCCESS, outline="")

        # Sight beam
        bx, by, bz = _Vector3.rotate_x(0, 0, cam_length/2 + 60, tilt - 90)
        bx, by, bz = _Vector3.rotate_y(bx, by, bz, -pan)
        px, py = self._project(bx, by + pole_height, bz)
        c.create_line(lx, ly, px, py, fill=T.SUCCESS, width=1, dash=(2, 4))

        # Pan arc on the base
        self._draw_arc(0, 0, 0, 50, pan, T.WARNING)

    def _draw_ellipse(self, x, y, z, radius_x, radius_z, color, outline):
        pts = []
        for a in range(0, 360, 10):
            rad = math.radians(a)
            pts.append(self._project(x + math.cos(rad) * radius_x, y,
                                      z + math.sin(rad) * radius_z))
        self.cvs.create_polygon(*[xy for p in pts for xy in p],
                                 fill=color, outline=outline, width=2,
                                 smooth=True)

    def _draw_arc(self, x, y, z, radius, pan_angle, color):
        pts = []
        if pan_angle >= 0:
            angles = range(0, int(pan_angle), 5)
        else:
            angles = range(360 + int(pan_angle), 360, 5)
        for a in angles:
            rad = math.radians(a)
            pts.append(self._project(x - math.sin(rad) * radius, y,
                                      z + math.cos(rad) * radius))
        if len(pts) > 1:
            self.cvs.create_line(*[xy for p in pts for xy in p],
                                  fill=color, width=2, dash=(4, 4))


def demo():
    root = tk.Tk()
    root.withdraw()

    def get_simulated_angles():
        t = time.time()
        return math.sin(t * 0.4) * 120, 90 + math.cos(t * 0.6) * 45

    SimulationWindow(root, get_simulated_angles)
    root.mainloop()


if __name__ == '__main__':
    demo()
