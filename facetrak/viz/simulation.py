import tkinter as tk
import math
import time


class _Colors:
    BG = "#1e1e2e"
    SURFACE = "#313244"
    OVERLAY = "#45475a"
    TEXT = "#cdd6f4"
    SUBTEXT = "#a6adc8"
    PAN_ACCENT = "#fab387"
    TILT_ACCENT = "#89b4fa"
    LENS = "#a6e3a1"
    WIRE = "#585b70"


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
        self.win.title("Pan-Tilt 3D Simulation")
        self.win.geometry("450x500")
        self.win.configure(bg=_Colors.BG)
        self.win.resizable(False, False)

        self.get_angles = get_angles_callback

        self.cvs = tk.Canvas(
            self.win, width=450, height=500, bg=_Colors.BG,
            highlightthickness=0
        )
        self.cvs.pack(fill=tk.BOTH, expand=True)

        self.cx, self.cy = 225, 300

        self._anim_id = None
        self._poll()

    def _project(self, x, y, z):
        x, y, z = _Vector3.rotate_x(x, y, z, 15)
        scale = 1.0
        return self.cx + x * scale, self.cy - y * scale

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

    def _draw_hud(self, pan, tilt):
        c = self.cvs
        c.create_text(30, 30, text="PTZ TRACKER", anchor="w",
                      font=("Helvetica", 14, "bold"), fill=_Colors.TEXT)
        c.create_text(30, 50, text="System Online • Tracking Active", anchor="w",
                      font=("Helvetica", 9), fill=_Colors.SUBTEXT)
        c.create_text(30, 90, text="PAN (Y-AXIS)", anchor="w",
                      font=("Helvetica", 8, "bold"), fill=_Colors.OVERLAY)
        c.create_text(30, 110, text=f"{pan:05.1f}°", anchor="w",
                      font=("Consolas", 22), fill=_Colors.PAN_ACCENT)
        c.create_text(420, 90, text="TILT (X-AXIS)", anchor="e",
                      font=("Helvetica", 8, "bold"), fill=_Colors.OVERLAY)
        c.create_text(420, 110, text=f"{tilt:05.1f}°", anchor="e",
                      font=("Consolas", 22), fill=_Colors.TILT_ACCENT)
        c.create_line(self.cx - 10, self.cy, self.cx + 10, self.cy, fill=_Colors.SURFACE)
        c.create_line(self.cx, self.cy - 10, self.cx, self.cy + 10, fill=_Colors.SURFACE)

    def _draw_scene(self, pan: float, tilt: float):
        self.cvs.delete("all")
        c = self.cvs
        self._draw_hud(pan, tilt)

        pole_height = 90
        self._draw_ellipse(0, 0, 0, radius_x=70, radius_z=70, color=_Colors.SURFACE, outline=_Colors.WIRE)

        cam_length = 80
        cam_width = 30
        cam_height = 30

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

        transformed_verts = []
        for vx, vy, vz in vertices:
            rx, ry, rz = _Vector3.rotate_x(vx, vy, vz, tilt - 90)
            rx, ry, rz = _Vector3.rotate_y(rx, ry, rz, -pan)
            ry += pole_height
            transformed_verts.append((rx, ry, rz))

        proj_verts = [self._project(x, y, z) for x, y, z in transformed_verts]

        p_base = self._project(0, 0, 0)
        p_top = self._project(0, pole_height, 0)
        c.create_line(p_base[0], p_base[1], p_top[0], p_top[1],
                      fill=_Colors.OVERLAY, width=12, capstyle=tk.ROUND)
        c.create_line(p_base[0], p_base[1], p_top[0], p_top[1],
                      fill=_Colors.SUBTEXT, width=4, capstyle=tk.ROUND)

        front_face = [proj_verts[0], proj_verts[1], proj_verts[2], proj_verts[3]]
        back_face = [proj_verts[4], proj_verts[5], proj_verts[6], proj_verts[7]]

        for i in range(4):
            c.create_line(front_face[i][0], front_face[i][1],
                          back_face[i][0], back_face[i][1], fill=_Colors.OVERLAY, width=2)

        c.create_polygon(*[coord for pt in back_face for coord in pt],
                         fill=_Colors.BG, outline=_Colors.OVERLAY, width=2)
        c.create_polygon(*[coord for pt in front_face for coord in pt],
                         fill=_Colors.SURFACE, outline=_Colors.TILT_ACCENT, width=2)

        lx = sum(p[0] for p in front_face) / 4
        ly = sum(p[1] for p in front_face) / 4

        c.create_oval(lx - 8, ly - 8, lx + 8, ly + 8,
                      fill=_Colors.BG, outline=_Colors.LENS, width=2)
        c.create_oval(lx - 3, ly - 3, lx + 3, ly + 3,
                      fill=_Colors.LENS, outline="")

        beam_end_local = (0, 0, cam_length/2 + 60)
        bx, by, bz = _Vector3.rotate_x(*beam_end_local, tilt - 90)
        bx, by, bz = _Vector3.rotate_y(bx, by, bz, -pan)
        bx, by, bz = bx, by + pole_height, bz
        px, py = self._project(bx, by, bz)
        c.create_line(lx, ly, px, py, fill=_Colors.LENS, width=1, dash=(2, 4))

        self._draw_arc(0, 0, 0, 50, pan, _Colors.PAN_ACCENT)

    def _draw_ellipse(self, x, y, z, radius_x, radius_z, color, outline):
        pts = []
        for a in range(0, 360, 10):
            rad = math.radians(a)
            px, py, pz = x + math.cos(rad) * radius_x, y, z + math.sin(rad) * radius_z
            pts.append(self._project(px, py, pz))
        self.cvs.create_polygon(*[c for p in pts for c in p], fill=color, outline=outline, width=2, smooth=True)

    def _draw_arc(self, x, y, z, radius, pan_angle, color):
        pts = []
        if pan_angle >= 0:
            angles = range(0, int(pan_angle), 5)
        else:
            angles = range(360 + int(pan_angle), 360, 5)
        for a in angles:
            rad = math.radians(a)
            px, py, pz = x - math.sin(rad) * radius, y, z + math.cos(rad) * radius
            pts.append(self._project(px, py, pz))
        if len(pts) > 1:
            self.cvs.create_line(*[c for p in pts for c in p], fill=color, width=2, dash=(4, 4))


def demo():
    root = tk.Tk()
    root.withdraw()
    angles = [0.0, 90.0]

    def get_simulated_angles():
        t = time.time()
        angles[0] = math.sin(t * 0.4) * 120
        angles[1] = 90 + math.cos(t * 0.6) * 45
        return angles[0], angles[1]

    sw = SimulationWindow(root, get_simulated_angles)
    root.mainloop()


if __name__ == '__main__':
    demo()
