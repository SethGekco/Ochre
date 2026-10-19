# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Zoom and pan, as pure arithmetic.

No Qt here at all, deliberately: zoom and pan bugs are miserable to chase
through a widget, and this way the whole coordinate system can be tested
exhaustively without a display, a QApplication, or an event loop.

Zoom is an exact rational, not a float. Paint.NET does the same and the
reason is concrete: repeatedly zooming in and back out with floats
accumulates error until "100%" is 99.9999% and a pixel grid no longer lands
on pixel boundaries. With Fraction, 100% is exactly 1 forever, and the round
trip through any sequence of steps is exact.
"""

import math
from fractions import Fraction

# Paint.NET's ladder. Values below 1 are exact reciprocals so that zooming
# out and back in returns precisely where it started.
ZOOM_STEPS = tuple(Fraction(n, d) for n, d in (
    (1, 100), (1, 50), (1, 32), (1, 24), (1, 16), (1, 12), (1, 8), (1, 6),
    (1, 5), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (1, 1),
    (3, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1), (12, 1),
    (16, 1), (24, 1), (32, 1), (50, 1), (64, 1),
))

MIN_ZOOM = ZOOM_STEPS[0]
MAX_ZOOM = ZOOM_STEPS[-1]

# At or above this, drawing switches to nearest-neighbour so pixels stay
# crisp; below it, smooth interpolation avoids aliasing when minifying.
NEAREST_AT = Fraction(1, 1)


class ViewState:
    """Maps between document pixels and widget pixels."""

    def __init__(self, doc_w, doc_h, zoom=Fraction(1, 1)):
        self.doc_w = int(doc_w)
        self.doc_h = int(doc_h)
        self._zoom = Fraction(zoom)
        # Widget-space position of document (0, 0) BEFORE rotation. Panning
        # moves this. With rotation, the view turns about the canvas centre.
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.rotation = 0.0           # radians, clockwise on screen

    # ---- zoom ------------------------------------------------------------

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        self._zoom = clamp_zoom(Fraction(value))

    @property
    def scale(self):
        """Zoom as a float, for the arithmetic that has to be floating point."""
        return float(self._zoom)

    @property
    def percent(self):
        return float(self._zoom * 100)

    @property
    def use_nearest(self):
        """True when pixels should be drawn hard rather than interpolated."""
        return self._zoom >= NEAREST_AT

    def zoom_at(self, new_zoom, anchor):
        """Change zoom while pinning the document point under `anchor`.

        This is what makes ctrl-wheel feel right: the pixel under the cursor
        stays under the cursor.
        """
        ax, ay = anchor
        dx, dy = self.widget_to_doc(ax, ay)
        self.zoom = new_zoom
        self._pin(dx, dy, ax, ay)
        return self._zoom

    def _pin(self, dx, dy, wx, wy):
        """Pan so document point (dx, dy) lands on widget point (wx, wy)."""
        cx, cy = self.doc_to_widget(dx, dy)
        self.offset_x += wx - cx
        self.offset_y += wy - cy

    def rotate_by(self, radians, anchor=None):
        """Turn the view, keeping the document point under `anchor` still.

        Rotation is about the anchor (the viewport centre by default), not
        the canvas centre: zoomed into a corner, turning the view about the
        middle of the image would swing what you are looking at off screen.
        """
        if anchor is not None:
            dx, dy = self.widget_to_doc(*anchor)
        self.rotation = (self.rotation + float(radians)) % (2.0 * math.pi)
        if abs(self.rotation - 2.0 * math.pi) < 1e-9:
            self.rotation = 0.0
        if anchor is not None:
            self._pin(dx, dy, anchor[0], anchor[1])

    def set_rotation(self, radians, anchor=None):
        return self.rotate_by(float(radians) - self.rotation, anchor)

    @property
    def rotated(self):
        return abs(self.rotation) > 1e-9

    def zoom_in(self, anchor=None):
        return self._step(+1, anchor)

    def zoom_out(self, anchor=None):
        return self._step(-1, anchor)

    def _step(self, direction, anchor):
        steps = ZOOM_STEPS
        if direction > 0:
            nxt = next((z for z in steps if z > self._zoom), steps[-1])
        else:
            nxt = next((z for z in reversed(steps) if z < self._zoom), steps[0])
        if anchor is None:
            self.zoom = nxt
            return self._zoom
        return self.zoom_at(nxt, anchor)

    # ---- fitting and centring -------------------------------------------

    def fit(self, viewport_w, viewport_h, margin=0):
        """Scale so the whole document is visible, then centre it."""
        avail_w = max(1, int(viewport_w) - 2 * margin)
        avail_h = max(1, int(viewport_h) - 2 * margin)
        if self.rotated:
            # A turned canvas needs room for its rotated bounding box.
            c, s_ = abs(math.cos(self.rotation)), abs(math.sin(self.rotation))
            bw = max(1, int(math.ceil(self.doc_w * c + self.doc_h * s_)))
            bh = max(1, int(math.ceil(self.doc_w * s_ + self.doc_h * c)))
        else:
            bw, bh = max(1, self.doc_w), max(1, self.doc_h)
        ratio = min(Fraction(avail_w, bw), Fraction(avail_h, bh))
        self.zoom = ratio if ratio > MIN_ZOOM else MIN_ZOOM
        self.center(viewport_w, viewport_h)
        return self._zoom

    def fit_rect(self, rect, viewport_w, viewport_h):
        """Scale and scroll so a DOCUMENT rect fills the viewport.

        What the zoom tool's rubber band asks for. Zoom is still clamped to
        the legal range, so dragging a one-pixel box asks for the maximum
        rather than an absurdity -- and the centring is done afterwards from
        the clamped scale, so the rect stays centred even when the requested
        zoom was not granted.
        """
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return self._zoom
        ratio = min(Fraction(max(1, int(viewport_w)), 1) / Fraction(w).limit_denominator(1000),
                    Fraction(max(1, int(viewport_h)), 1) / Fraction(h).limit_denominator(1000))
        self.zoom = ratio
        self._pin(x + w / 2.0, y + h / 2.0, viewport_w / 2.0, viewport_h / 2.0)
        return self._zoom

    def center(self, viewport_w, viewport_h):
        self.offset_x = (viewport_w - self.doc_w * self.scale) / 2.0
        self.offset_y = (viewport_h - self.doc_h * self.scale) / 2.0

    def pan_by(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy

    # ---- coordinate mapping ---------------------------------------------

    def doc_to_widget(self, x, y):
        s = self.scale
        if not self.rotated:
            return (x * s + self.offset_x, y * s + self.offset_y)
        pcx, pcy = self.doc_w / 2.0, self.doc_h / 2.0
        wcx, wcy = self.offset_x + s * pcx, self.offset_y + s * pcy
        ux, uy = (x - pcx) * s, (y - pcy) * s
        c, sn = math.cos(self.rotation), math.sin(self.rotation)
        return (wcx + c * ux - sn * uy, wcy + sn * ux + c * uy)

    def widget_to_doc(self, x, y):
        s = self.scale
        if not self.rotated:
            return ((x - self.offset_x) / s, (y - self.offset_y) / s)
        pcx, pcy = self.doc_w / 2.0, self.doc_h / 2.0
        wcx, wcy = self.offset_x + s * pcx, self.offset_y + s * pcy
        ux, uy = x - wcx, y - wcy
        c, sn = math.cos(self.rotation), math.sin(self.rotation)
        return (pcx + (c * ux + sn * uy) / s, pcy + (-sn * ux + c * uy) / s)

    def affine(self):
        """(m11, m12, m21, m22, dx, dy): doc -> widget as a QTransform takes it."""
        s = self.scale
        c, sn = math.cos(self.rotation), math.sin(self.rotation)
        pcx, pcy = self.doc_w / 2.0, self.doc_h / 2.0
        wcx, wcy = self.offset_x + s * pcx, self.offset_y + s * pcy
        a, b, cc, d = s * c, s * sn, -s * sn, s * c
        dx = wcx - (a * pcx + cc * pcy)
        dy = wcy - (b * pcx + d * pcy)
        return (a, b, cc, d, dx, dy)

    def doc_to_widget_rect(self, rect):
        if not self.rotated:
            x0, y0 = self.doc_to_widget(rect.x, rect.y)
            x1, y1 = self.doc_to_widget(rect.x1, rect.y1)
            return (x0, y0, x1 - x0, y1 - y0)
        pts = [self.doc_to_widget(x, y) for x, y in ((rect.x, rect.y), (rect.x1, rect.y),
                                                     (rect.x1, rect.y1), (rect.x, rect.y1))]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def widget_to_doc_rect(self, x, y, w, h):
        """Document Rect covering a widget rectangle.

        Rounded OUTWARD, and that matters: this decides which document pixels
        a repaint must cover, so rounding inward would leave a sliver of
        stale pixels at the edge of every update.
        """
        from ..engine.geometry import Rect
        pts = [self.widget_to_doc(px, py) for px, py in ((x, y), (x + w, y),
                                                         (x + w, y + h), (x, y + h))]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return Rect.from_bounds(int(math.floor(min(xs))), int(math.floor(min(ys))),
                                int(math.ceil(max(xs))), int(math.ceil(max(ys))))

    def visible_doc_rect(self, viewport_w, viewport_h):
        """The document region on screen, clipped to the canvas.

        The interactive path composites THIS and never the whole document --
        a full-canvas composite of a large multi-layer document runs into
        tens of seconds, so viewport clipping is load-bearing rather than
        merely tidy.
        """
        rect = self.widget_to_doc_rect(0, 0, viewport_w, viewport_h)
        return rect.clipped_to(self.doc_w, self.doc_h)

    # ---- misc ------------------------------------------------------------

    def canvas_size(self):
        """Size of the drawn document in widget pixels."""
        return (self.doc_w * self.scale, self.doc_h * self.scale)

    def clamp_pan(self, viewport_w, viewport_h, slack=0.25):
        """Keep the canvas from being dragged entirely out of view.

        `slack` is the fraction of the viewport the canvas may hang past the
        edge, so a user can still reach a corner without losing the document.
        """
        cw, ch = self.canvas_size()
        min_x = -cw + viewport_w * slack
        max_x = viewport_w - viewport_w * slack
        min_y = -ch + viewport_h * slack
        max_y = viewport_h - viewport_h * slack
        if cw <= viewport_w:
            self.offset_x = (viewport_w - cw) / 2.0
        else:
            self.offset_x = min(max(self.offset_x, min_x), max_x)
        if ch <= viewport_h:
            self.offset_y = (viewport_h - ch) / 2.0
        else:
            self.offset_y = min(max(self.offset_y, min_y), max_y)

    def copy(self):
        out = ViewState(self.doc_w, self.doc_h, self._zoom)
        out.offset_x, out.offset_y = self.offset_x, self.offset_y
        out.rotation = self.rotation
        return out

    def __repr__(self):
        return "ViewState(%dx%d, zoom=%s, offset=(%.1f, %.1f))" % (
            self.doc_w, self.doc_h, self._zoom, self.offset_x, self.offset_y)


def clamp_zoom(value):
    return max(MIN_ZOOM, min(MAX_ZOOM, Fraction(value)))


def format_zoom(zoom):
    """Human-readable percentage. Exact values print without decimals."""
    pct = Fraction(zoom) * 100
    if pct.denominator == 1:
        return "%d%%" % pct.numerator
    return "%.2f%%" % float(pct)
