"""Design tokens for FaceTrak — Tactical Dark theme."""

# ── Surfaces ────────────────────────────────────────────────────────────────
BG_ROOT    = "#0D0D0F"
BG_SURFACE = "#16161A"
BG_RAISED  = "#1E1E24"
BG_OVERLAY = "#26262E"
BG_INPUT   = "#1E1E24"

# ── Borders ─────────────────────────────────────────────────────────────────
BORDER_SUBTLE = "#2A2A34"
BORDER_MUTED  = "#38383F"
BORDER_FOCUS  = "#4A4A5A"

# ── Text ────────────────────────────────────────────────────────────────────
TEXT_PRIMARY  = "#EBEBF0"
TEXT_SECONDARY = "#8E8E99"
TEXT_MUTED    = "#55555F"
TEXT_INVERSE  = "#0D0D0F"

# ── Accent ──────────────────────────────────────────────────────────────────
ACCENT        = "#3B82F6"   # Electric blue
ACCENT_BRIGHT = "#60A5FA"
ACCENT_DIM    = "#1D4ED8"

# ── Semantic ────────────────────────────────────────────────────────────────
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER  = "#EF4444"
INFO    = "#38BDF8"

# ── Emotion map ─────────────────────────────────────────────────────────────
EMO_COLORS = {
    "happy":     SUCCESS,
    "sad":       ACCENT,
    "angry":     DANGER,
    "surprised": WARNING,
    "neutral":   TEXT_MUTED,
    "fear":      "#A855F7",
    "disgust":   "#84CC16",
}

# ── Typography ──────────────────────────────────────────────────────────────
FONT_DISPLAY = ("SF Pro Display", 13, "bold")
FONT_LABEL   = ("SF Pro Text",    9)
FONT_LABEL_B = ("SF Pro Text",    9,  "bold")
FONT_MONO    = ("SF Mono",        9)
FONT_MONO_B  = ("SF Mono",        9,  "bold")
FONT_MICRO   = ("SF Pro Text",    8)
FONT_TITLE   = ("SF Pro Display", 10, "bold")

# Fallback stacks (macOS → Linux → Windows)
def font(name: str, size: int, *style) -> tuple:
    stacks = {
        "display": ["SF Pro Display", "Helvetica Neue", "Segoe UI", "sans-serif"],
        "text":    ["SF Pro Text",    "Helvetica Neue", "Segoe UI", "sans-serif"],
        "mono":    ["SF Mono",        "JetBrains Mono", "Consolas", "Courier New"],
    }
    family = stacks.get(name, stacks["text"])[0]
    return (family, size) + style


FONT_DISPLAY = font("display", 13, "bold")
FONT_LABEL   = font("text",    9)
FONT_LABEL_B = font("text",    9, "bold")
FONT_MONO    = font("mono",    9)
FONT_MONO_SM = font("mono",    8)
FONT_MICRO   = font("text",    8)
FONT_TITLE   = font("display", 10, "bold")
FONT_APP     = font("display", 11, "bold")

# ── Geometry ─────────────────────────────────────────────────────────────────
RADIUS   = 6
SPACING  = 8
PADDING  = 12

# ── Icons (Unicode) ──────────────────────────────────────────────────────────
ICON_PLAY   = "▶"
ICON_STOP   = "■"
ICON_REC    = "●"
ICON_CAM    = "⊙"
ICON_BLUR   = "◈"
ICON_HEAT   = "⟁"
ICON_SERVO  = "⊕"
ICON_REG    = "⊞"
ICON_DIR    = "⊟"
ICON_SIM    = "⊡"
ICON_DOT    = "●"
