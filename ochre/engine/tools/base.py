# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Tool protocol.

Tools live in the engine and know nothing about Qt. They receive a ToolEvent
-- plain data, no widget types -- so the whole tool layer is testable headless
and a tool can be driven from a script exactly as it is from a mouse.

Two things are worth stating because they shape every tool:

Tools never write to a Surface directly. Everything goes through
`Surface.apply_masked`, with the selection fused into the tool's own coverage
by one multiply. That is why selection clipping is automatic and why no tool
can forget to honour it.

Modifier state is re-read on every motion event rather than captured at
mouse-down. That is what lets pressing or releasing Shift mid-drag update a
constrained line live, instead of locking in whatever was held at the start.
"""

from ..geometry import Rect

# Modifier bits. Deliberately not Qt's, so the engine stays Qt-free; the UI
# translates at the boundary.
MOD_NONE = 0
MOD_SHIFT = 1
MOD_CTRL = 2
MOD_ALT = 4


class ToolEvent:
    """One input event in document coordinates."""

    __slots__ = ("x", "y", "pressure", "modifiers", "button")

    def __init__(self, x, y, pressure=1.0, modifiers=MOD_NONE, button=1):
        self.x = float(x)
        self.y = float(y)
        self.pressure = float(pressure)
        self.modifiers = int(modifiers)
        self.button = int(button)

    @property
    def ix(self):
        return int(round(self.x))

    @property
    def iy(self):
        return int(round(self.y))

    def has(self, mod):
        return bool(self.modifiers & mod)

    def __repr__(self):
        return "ToolEvent(%.1f, %.1f, p=%.2f, mods=%d)" % (
            self.x, self.y, self.pressure, self.modifiers)


class ToolContext:
    """What a tool is allowed to reach: the document, and the current choices."""

    def __init__(self, document, layer=None, frame=None, selection=None,
                 primary=(0, 0, 0, 255), secondary=(255, 255, 255, 255),
                 settings=None, history=None):
        self.doc = document
        self.layer = layer
        self.frame = frame if frame is not None else document.frame
        self.selection = selection
        self.primary = primary
        self.secondary = secondary
        self.settings = settings
        self.history = history
        self.primary_index = None     # set when the document is index-locked
        self.secondary_index = None

    @property
    def cell(self):
        if self.layer is None:
            return None
        return self.doc.cell(self.layer, self.frame)

    def colour_for(self, event):
        """Primary on the left button, secondary on the right."""
        return self.secondary if event.button == 2 else self.primary

    def index_for(self, event):
        return self.secondary_index if event.button == 2 else self.primary_index

    def coverage(self, other, rect):
        """Fuse the selection into a tool's coverage. One multiply."""
        if self.selection is None:
            return other
        return self.selection.combine_mask(other, rect)


class Tool:
    """Base class. Subclasses override the three lifecycle hooks."""

    name = "tool"
    label = "Tool"
    target_plane = "rgba"       # INI-overridable; a height brush sets "height"
    wants_stroke = True         # False for tools that do not paint pixels
    cursor = "crosshair"

    def __init__(self, **options):
        self.options = dict(options)
        self.active = False
        self._session = None

    # ---- options ---------------------------------------------------------

    def option(self, key, default=None):
        return self.options.get(key, default)

    def set_option(self, key, value):
        self.options[key] = value
        return self

    # ---- lifecycle -------------------------------------------------------

    def begin(self, ctx, event):
        self.active = True
        return self.on_begin(ctx, event)

    def motion(self, ctx, event):
        if not self.active:
            return None
        return self.on_motion(ctx, event)

    def end(self, ctx, event):
        if not self.active:
            return None
        self.active = False
        return self.on_end(ctx, event)

    def cancel(self, ctx):
        """Escape mid-drag. Must leave no trace in history."""
        if not self.active:
            return None
        self.active = False
        return self.on_cancel(ctx)

    # ---- hooks -----------------------------------------------------------

    def on_begin(self, ctx, event):
        return None

    def on_motion(self, ctx, event):
        return None

    def on_end(self, ctx, event):
        return None

    def on_cancel(self, ctx):
        return None

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.name)


# ---- shared geometry helpers -------------------------------------------

def constrain_angle(x0, y0, x1, y1, step_degrees=15.0):
    """Snap a line to the nearest multiple of step_degrees.

    Re-applied on every motion event rather than latched at mouse-down, so
    toggling Shift updates the preview live.
    """
    import math
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return x1, y1
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx))
    step = round(angle / step_degrees) * step_degrees
    rad = math.radians(step)
    return x0 + length * math.cos(rad), y0 + length * math.sin(rad)


def constrain_square(x0, y0, x1, y1):
    """Make a drag square while preserving the quadrant it was made in."""
    dx, dy = x1 - x0, y1 - y0
    size = max(abs(dx), abs(dy))
    return (x0 + (size if dx >= 0 else -size),
            y0 + (size if dy >= 0 else -size))


def from_centre(x0, y0, x1, y1):
    """Treat the anchor as the centre rather than a corner."""
    dx, dy = x1 - x0, y1 - y0
    return x0 - dx, y0 - dy, x1, y1


def drag_rect(event_start, event_now, square=False, centre=False):
    """The rectangle a drag describes, honouring live modifier state."""
    x0, y0 = event_start.x, event_start.y
    x1, y1 = event_now.x, event_now.y
    if square:
        x1, y1 = constrain_square(x0, y0, x1, y1)
    if centre:
        x0, y0, x1, y1 = from_centre(x0, y0, x1, y1)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def bounds_of(rect_tuple, pad=1):
    """Integer Rect covering a float rect, padded for antialiased edges."""
    x0, y0, x1, y1 = rect_tuple
    import math
    return Rect.from_bounds(int(math.floor(x0)) - pad, int(math.floor(y0)) - pad,
                            int(math.ceil(x1)) + pad, int(math.ceil(y1)) + pad)
