"""Display-backed GUI regressions; no camera, models or persistent data needed."""

import gc
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from facetrak.viz import theme as T  # noqa: E402
from facetrak.viz.components import (  # noqa: E402
    CommandBar,
    Sidebar,
    StatusBar,
    VideoPanel,
)
from facetrak.viz.components.button import Button  # noqa: E402


@pytest.fixture(scope="session")
def gui_root():
    # A single interpreter must stay alive until session teardown on macOS.
    # Otherwise later API worker-thread GC can abort in Tk_AsyncDelete.
    try:
        window = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"GUI display unavailable: {exc}")
    T.configure_fonts()
    yield window
    window.destroy()
    gc.collect()


@pytest.fixture
def root(gui_root):
    gui_root.geometry("1180x820")
    yield gui_root
    for child in gui_root.winfo_children():
        child.destroy()
    # MainWindow installs root bindings; remove callbacks between cases.
    for sequence in gui_root.bind():
        gui_root.unbind(sequence)
    gc.collect()


def callbacks():
    return {
        key: Mock()
        for key in (
            "toggle_start",
            "toggle_rec",
            "toggle_blur",
            "toggle_heatmap",
            "toggle_servo",
            "toggle_mirror",
            "register_dialog",
            "list_faces",
            "set_servo_target",
            "on_cam_select",
            "open_sim",
        )
    }


def test_video_fits_viewport_and_restores_empty_state(root):
    panel = VideoPanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    root.update()
    for geometry in ("1180x820", "960x620"):
        root.geometry(geometry)
        root.update()
        for shape in ((480, 640, 3), (640, 360, 3)):
            panel.show_frame(np.zeros(shape, dtype=np.uint8))
            root.update()
            width, height = panel._tk_img.width(), panel._tk_img.height()
            assert width <= panel._canvas.winfo_width()
            assert height <= panel._canvas.winfo_height()
            assert abs(width / height - shape[1] / shape[0]) < 0.01
            assert not panel._placeholder.winfo_ismapped()
    panel.clear()
    root.update()
    assert panel._placeholder.winfo_ismapped()
    assert panel._canvas.itemcget(panel._image_id, "image") == ""


def test_buttons_support_keyboard_and_disabled_state(root):
    command = Mock()
    button = Button(root, "Test action", command)
    button.pack()
    root.update()
    button.focus_force()
    root.update()
    button.event_generate("<KeyPress-Return>")
    button.event_generate("<KeyPress-space>")
    root.update()
    assert command.call_count == 2
    button.set_enabled(False)
    button.event_generate("<KeyPress-Return>")
    root.update()
    assert command.call_count == 2
    assert button.cget("takefocus") == "0"


def test_toolbar_actions_fit_and_states_follow_engine(root):
    bar = CommandBar(root, callbacks())
    bar.pack(fill=tk.X)
    for geometry in ("1180x820", "960x620"):
        root.geometry(geometry)
        root.update()
        for widget in (bar._start_btn, bar._rec_btn, bar._register_btn, bar.cam_cb):
            assert widget.winfo_rootx() >= root.winfo_rootx()
            assert widget.winfo_rootx() + widget.winfo_width() <= (
                root.winfo_rootx() + root.winfo_width()
            )
            assert widget.winfo_rooty() + widget.winfo_height() <= (
                bar.winfo_rooty() + bar.winfo_height()
            )
    assert not bar._rec_btn._enabled
    assert not bar._register_btn._enabled
    bar.set_running(True)
    assert bar._rec_btn._enabled and bar._register_btn._enabled
    bar.set_toggles(blur=True, mirror=True)
    assert bar._toggles["blur"].get() and bar._toggles["mirror"].get()
    bar.set_running(False)
    assert not bar._rec_btn._enabled and not bar._register_btn._enabled


def test_privacy_controls_scroll_into_view_on_focus(root):
    root.geometry("960x620")
    sidebar = Sidebar(root, on_blur_toggle=Mock())
    sidebar.pack(side=tk.RIGHT, fill=tk.Y)
    root.update()
    card = sidebar.priv_card
    for item in card.find_all():
        bounds = card.bbox(item)
        assert bounds[3] <= card._ph
    card.listbox.focus_force()
    root.update()
    assert sidebar._viewport.yview()[0] > 0
    top = sidebar._viewport.winfo_rooty()
    assert card.listbox.winfo_rooty() >= top
    assert card.listbox.winfo_rooty() + card.listbox.winfo_height() <= (
        top + sidebar._viewport.winfo_height()
    )


def test_shortcuts_do_not_intercept_focused_controls():
    from facetrak.viz.ui import MainWindow

    window = MainWindow.__new__(MainWindow)
    window.root = Mock()
    action = Mock()
    entry = Mock()
    entry.winfo_class.return_value = "TEntry"
    window._shortcut(SimpleNamespace(widget=entry, keysym="r"), action)
    action.assert_not_called()
    canvas = Mock()
    canvas.winfo_class.return_value = "Canvas"
    window._shortcut(SimpleNamespace(widget=canvas, keysym="space"), action)
    action.assert_not_called()
    assert (
        window._shortcut(SimpleNamespace(widget=canvas, keysym="r"), action) == "break"
    )
    action.assert_called_once()


def test_status_bar_fits_small_window(root):
    root.geometry("960x620")
    bar = StatusBar(root)
    bar.pack(side=tk.BOTTOM, fill=tk.X)
    root.update()
    for widget in [*bar._pills.values(), bar._rec_label]:
        assert widget.winfo_rootx() + widget.winfo_width() <= (
            root.winfo_rootx() + root.winfo_width()
        )
    bar.set_recording(True)
    assert bar._rec_label.cget("text") == "Recording"
    bar.set_recording(False)
    assert bar._rec_label.cget("text") == "Not recording"


def test_workspace_initializes_without_starting_camera(root):
    from facetrak.viz.ui import MainWindow

    engine = Mock()
    engine.running = False
    engine.cfg = {"mirror": True}
    engine.blur_enabled = True
    engine.heatmap_enabled = False
    engine.servo_enabled = False
    engine.servo_target = "known"
    engine.db.known_names = ["Test person"]
    engine.blur_persons = set()
    cfg = {"camera": 0, "cameras": [{"name": "Test camera", "source": 0}]}
    with (
        patch("facetrak.viz.ui.FaceEngine", return_value=engine),
        patch("facetrak.viz.ui.config.load", return_value=cfg),
    ):
        window = MainWindow(root)
        root.update()
        engine.start.assert_not_called()
        assert window._cmd_bar.cam_var.get() == "Test camera (0)"
        assert window._cmd_bar.target_var.get() == "known"
        assert window._cmd_bar._toggles["blur"].get()
        assert window._cmd_bar._toggles["mirror"].get()
        assert window._sidebar.listbox.size() == 1
        root.after_cancel(window._anim_id)


def test_text_contrast_on_panels():
    def luminance(color):
        rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [
            v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb
        ]
        return sum(a * b for a, b in zip(linear, (0.2126, 0.7152, 0.0722)))

    for foreground in (
        T.TEXT_PRIMARY,
        T.TEXT_SECONDARY,
        T.TEXT_MUTED,
        T.DANGER,
        T.SUCCESS,
        T.WARNING,
    ):
        for background in (T.BG_ROOT, T.BG_SURFACE, T.BG_RAISED):
            dark, light = sorted((luminance(foreground), luminance(background)))
            assert (light + 0.05) / (dark + 0.05) >= 4.5
