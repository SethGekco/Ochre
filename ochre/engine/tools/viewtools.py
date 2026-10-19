# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Zoom and pan, as tools.

Both were already possible -- ctrl-wheel zooms, middle-drag pans -- so what
these add is the toolbar affordance, for people who reach for a magnifier
rather than remembering a chord. That makes them the only tools that change
nothing about the document, which raises the obvious question of why they
live in the engine at all when the view they move is a UI concern.

The answer is the split this project uses everywhere: the DECISION is here,
the EFFECT is in Qt. Which button zooms out, whether a drag is a
rubber-band or a click, how small a drag is too small to mean a rectangle --
that is policy, it is where the bugs live, and it is testable with no
display. Applying the result to a ViewState is three lines in the canvas.

So a view tool returns a ViewRequest instead of a dirty rect, and the canvas
applies it. Nothing here imports Qt and nothing here knows what a pixel on
screen is: positions are document coordinates, like every other tool.
"""

from .base import MOD_ALT, Tool

PAN = "pan"
ZOOM_IN = "zoom_in"
ZOOM_OUT = "zoom_out"
ZOOM_RECT = "zoom_rect"
ROTATE = "rotate"
ROTATE_RESET = "rotate_reset"


class ViewRequest:
    """What a view tool wants done, in document coordinates.

    Returned in place of a dirty rect. The canvas is the only thing that can
    carry it out, because only the canvas knows the viewport size and the
    current scale.
    """

    __slots__ = ("kind", "dx", "dy", "anchor", "rect", "angle")

    def __init__(self, kind, dx=0.0, dy=0.0, anchor=None, rect=None, angle=0.0):
        self.kind = kind
        self.angle = angle          # for ROTATE: radians to turn the view by
        self.dx = dx                # for PAN: document units to shift by
        self.dy = dy
        self.anchor = anchor        # for ZOOM_IN/OUT: doc point to pin
        self.rect = rect            # for ZOOM_RECT: doc rect to fill

    def __repr__(self):
        return "ViewRequest(%r, dx=%.2f, dy=%.2f, anchor=%r, rect=%r)" % (
            self.kind, self.dx, self.dy, self.anchor, self.rect)


class PanTool(Tool):
    """Drag the image around.

    Reports the DOCUMENT distance the grabbed point has moved, and leaves the
    canvas to turn that into scroll offset. Working in document units means
    the tool needs no notion of scale, and the point grabbed on mouse-down
    stays under the cursor at any zoom -- which is the whole feel of panning
    and the thing that goes wrong when the maths is done in widget pixels
    against a stale transform.
    """

    name = "pan"
    label = "Pan"
    wants_stroke = False
    affects = "view"
    cursor = "pointing"

    def on_begin(self, ctx, event):
        self._grab = (event.x, event.y)
        return None

    def on_motion(self, ctx, event):
        grab = getattr(self, "_grab", None)
        if grab is None:
            return None
        # Deliberately measured against the grab point every time rather than
        # accumulated between motions. Panning moves the very coordinates
        # these events are reported in, so summing deltas drifts; anchoring
        # to a fixed document point cannot.
        return ViewRequest(PAN, dx=event.x - grab[0], dy=event.y - grab[1])

    def on_end(self, ctx, event):
        result = self.on_motion(ctx, event)
        self._grab = None
        return result

    def on_cancel(self, ctx):
        self._grab = None
        return None


class ZoomTool(Tool):
    """Click to zoom in, right-click or alt-click to zoom out, drag a box.

    A bare click steps the zoom ladder around the point clicked, so the pixel
    under the cursor stays put. A drag asks for a rectangle to be filled,
    which is the only way to get to an arbitrary zoom without repeated
    clicking -- and it is why the tool has to distinguish a drag from a click
    at all. The threshold is in document units and INI-tunable, because a
    few pixels of hand tremor at 800% is a large document distance.
    """

    name = "zoom"
    label = "Zoom"
    wants_stroke = False
    affects = "view"
    cursor = "pointing"
    default_min_drag = 4.0

    def on_begin(self, ctx, event):
        self._start = (event.x, event.y)
        self._out = self._means_out(event)
        self._band = None
        return None

    def on_motion(self, ctx, event):
        start = getattr(self, "_start", None)
        if start is None:
            return None
        self._band = self._rect(start, event)
        return None             # the band is a preview; nothing to apply yet

    def on_end(self, ctx, event):
        start = getattr(self, "_start", None)
        self._start = None
        if start is None:
            return None

        rect = self._rect(start, event)
        floor = float(self.option("min_drag", self.default_min_drag))
        if rect is not None and rect[2] >= floor and rect[3] >= floor:
            self._band = None
            return ViewRequest(ZOOM_RECT, rect=rect)

        self._band = None
        # Re-read at mouse-UP as well as mouse-down, so pressing alt part-way
        # through a click still means "out" -- the same live-modifier rule
        # the shape tools follow.
        out = self._out or self._means_out(event)
        return ViewRequest(ZOOM_OUT if out else ZOOM_IN,
                           anchor=(event.x, event.y))

    @staticmethod
    def _means_out(event):
        """Right button or alt. NOT event.has(2) -- has() tests MODIFIERS,
        and MOD_CTRL happens to be 2, so asking it about a button number
        silently turns ctrl-click into zoom-out and right-click into nothing.
        """
        return event.button == 2 or event.has(MOD_ALT)

    def on_cancel(self, ctx):
        self._start = None
        self._band = None
        return None

    @property
    def band(self):
        """The rubber-band rectangle to draw, or None. Document units."""
        return self._band

    @staticmethod
    def _rect(start, event):
        x0, y0 = start
        x = min(x0, event.x)
        y = min(y0, event.y)
        w = abs(event.x - x0)
        h = abs(event.y - y0)
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)



class RotateViewTool(Tool):
    """Turn the canvas on screen, like turning paper on a desk -- Photoshop's
    Rotate View (R). Changes nothing in the document. Shift snaps to 15
    degrees; a click without dragging resets to upright.

    Measured, like the pan tool, as the angle the GRABBED document point has
    moved about the canvas centre, rather than accumulated between events:
    rotating the view changes the very coordinates the events arrive in.
    """

    name = "rotate_view"
    label = "Rotate View"
    wants_stroke = False
    affects = "view"
    cursor = "pointing"

    def on_begin(self, ctx, event):
        self._grab = (event.x, event.y)
        self._moved = False
        self._centre = (ctx.doc.width / 2.0, ctx.doc.height / 2.0) if ctx is not None \
            and getattr(ctx, "doc", None) is not None else (0.0, 0.0)
        self._total = 0.0
        return None

    def on_motion(self, ctx, event):
        import math
        if getattr(self, "_grab", None) is None:
            return None
        cx, cy = self._centre
        a0 = math.atan2(self._grab[1] - cy, self._grab[0] - cx)
        a1 = math.atan2(event.y - cy, event.x - cx)
        delta = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
        if abs(delta) > 1e-4:
            self._moved = True
        if event.has(1):
            step = math.radians(15)
            target = round((self._total + delta) / step) * step
            delta = target - self._total
        self._total += delta
        return ViewRequest(ROTATE, angle=delta)

    def on_end(self, ctx, event):
        result = self.on_motion(ctx, event)
        moved = getattr(self, "_moved", False)
        self._grab = None
        if not moved:
            return ViewRequest(ROTATE_RESET)
        return result

    def on_cancel(self, ctx):
        self._grab = None
        return None
