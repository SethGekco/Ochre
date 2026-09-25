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


class ViewRequest:
    """What a view tool wants done, in document coordinates.

    Returned in place of a dirty rect. The canvas is the only thing that can
    carry it out, because only the canvas knows the viewport size and the
    current scale.
    """

    __slots__ = ("kind", "dx", "dy", "anchor", "rect")

    def __init__(self, kind, dx=0.0, dy=0.0, anchor=None, rect=None):
        self.kind = kind
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
