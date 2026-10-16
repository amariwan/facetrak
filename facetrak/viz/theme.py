"""Design tokens for FaceTrak — camera workspace theme."""

# ── Surfaces ────────────────────────────────────────────────────────────────
BG_ROOT = "#10151C"
BG_SURFACE = "#18212C"
BG_RAISED = "#222E3B"
BG_OVERLAY = "#2C3B4C"
BG_INPUT = "#222E3B"

# ── Borders ─────────────────────────────────────────────────────────────────
BORDER_SUBTLE = "#344456"
BORDER_MUTED = "#465B70"
BORDER_FOCUS = "#93C5FD"

# ── Text ────────────────────────────────────────────────────────────────────
TEXT_PRIMARY = "#F1F5F9"
TEXT_SECONDARY = "#BDC9D6"
TEXT_MUTED = "#9AABBD"
TEXT_INVERSE = "#10151C"

# ── Accent ──────────────────────────────────────────────────────────────────
ACCENT = "#3B82F6"  # Electric blue
ACCENT_BRIGHT = "#60A5FA"
ACCENT_DIM = "#1D4ED8"

# ── Semantic ────────────────────────────────────────────────────────────────
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER = "#F87171"
INFO = "#38BDF8"

# ── Emotion map ─────────────────────────────────────────────────────────────
EMO_COLORS = {
    "happy": SUCCESS,
    "sad": ACCENT,
    "angry": DANGER,
    "surprised": WARNING,
    "neutral": TEXT_MUTED,
    "fear": "#A855F7",
    "disgust": "#84CC16",
}


# ── Typography ──────────────────────────────────────────────────────────────
def font(name: str, size: int, *style) -> tuple:
    """Use Tk's platform-selected system families, including on Linux/Pi."""
    import tkinter.font as tkfont

    default = "TkFixedFont" if name == "mono" else "TkDefaultFont"
    family = tkfont.nametofont(default).actual("family")
    return (family, size) + style


# Resolved lazily: importing the theme must not create a Tk root (headless API).
FONT_DISPLAY = ("Helvetica", 18, "bold")
FONT_LABEL = ("Helvetica", 11)
FONT_LABEL_B = ("Helvetica", 11, "bold")
FONT_MONO = ("Courier", 11)
FONT_MONO_B = ("Courier", 11, "bold")
FONT_MONO_SM = ("Courier", 10)
FONT_MICRO = ("Helvetica", 10)
FONT_TITLE = ("Helvetica", 12, "bold")
FONT_APP = ("Helvetica", 18, "bold")


def configure_fonts():
    """Resolve native fonts after the application's root exists."""
    global FONT_DISPLAY, FONT_LABEL, FONT_LABEL_B, FONT_MONO, FONT_MONO_B
    global FONT_MONO_SM, FONT_MICRO, FONT_TITLE, FONT_APP
    FONT_DISPLAY = font("display", 18, "bold")
    FONT_LABEL = font("text", 11)
    FONT_LABEL_B = font("text", 11, "bold")
    FONT_MONO = font("mono", 11)
    FONT_MONO_B = font("mono", 11, "bold")
    FONT_MONO_SM = font("mono", 10)
    FONT_MICRO = font("text", 10)
    FONT_TITLE = font("display", 12, "bold")
    FONT_APP = FONT_DISPLAY


# ── Geometry ─────────────────────────────────────────────────────────────────
RADIUS = 6
SPACING = 8
PADDING = 12

# ── Icons (Unicode) ──────────────────────────────────────────────────────────
ICON_PLAY = "▶"
ICON_STOP = "■"
ICON_REC = "●"
ICON_CAM = "⊙"
ICON_BLUR = "◈"
ICON_HEAT = "⟁"
ICON_SERVO = "⊕"
ICON_REG = "⊞"
ICON_DIR = "⊟"
ICON_SIM = "⊡"
ICON_MIRROR = "⇔"
ICON_DOT = "●"
