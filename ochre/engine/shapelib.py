# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The shape catalog: every shape the Shapes tool can draw.

Each entry turns a drag rectangle into one or more closed RINGS of points.
That single representation drives everything -- the pixels, the dirty rect
and the chooser's icons -- so a shape cannot look one way in the menu and
another on the canvas.

Two rules make the rings rasterise correctly:

  * Rings combine EVEN-ODD. A gear is an outer ring plus its hole; a thought
    bubble is a body plus two disjoint dots. Neither needs a union.
  * Rings never overlap partially. Overlapping rings would XOR into holes,
    and their outlines would draw seams across the interior. Every shape
    whose natural construction overlaps -- a speech bubble is a body plus a
    tail -- is instead built as ONE ring with the tail spliced into the
    body's edge, so its outline is a single unbroken stroke.

Shapes inscribe in the drag rectangle, so a wide drag gives a wide shape,
like every other shape tool here. Curves are sampled densely enough that
no facet is visible at the size drawn.

Rasterisation goes through Pillow's polygon filler at up to 4x supersampling,
not the pure-numpy rasteriser the selection uses: that one evaluates every
edge against every sample in Python, which is fine for a pentagon and far
too slow for a 300-vertex cloud being rubber-banded at frame rate.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from .geometry import Rect

# Categories in the order a chooser lists them, matching Paint.NET's layout.
BASIC, POLY, ARROWS, CALLOUTS, SYMBOLS = (
    "Basic", "Polygons and Stars", "Arrows", "Callouts", "Symbols")


# ---- helpers ----------------------------------------------------------------

def _map(unit_points, x0, y0, x1, y1):
    w, h = x1 - x0, y1 - y0
    return [(x0 + u * w, y0 + v * h) for u, v in unit_points]


def _segments(length_px, lo=12, hi=360):
    return int(min(hi, max(lo, length_px / 2.5)))


def _ellipse_ring(cx, cy, rx, ry, start=0.0, end=2 * math.pi, n=None, close=True):
    if n is None:
        n = _segments(math.pi * (abs(rx) + abs(ry)))
    if close:
        return [(cx + rx * math.cos(start + (end - start) * i / n),
                 cy + ry * math.sin(start + (end - start) * i / n)) for i in range(n)]
    return [(cx + rx * math.cos(start + (end - start) * i / n),
             cy + ry * math.sin(start + (end - start) * i / n)) for i in range(n + 1)]


def _arc(cx, cy, r, a0, a1):
    n = max(3, _segments(abs(a1 - a0) * r, lo=3, hi=64))
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
             cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def _rounded_ring(x0, y0, x1, y1, r, tail=None):
    """A rounded rectangle, clockwise from the top-left straight edge.

    `tail` = (base_left_x, base_right_x, tip_x, tip_y) splices a speech-bubble
    tail into the BOTTOM edge, so body and tail are one ring with one outline.
    """
    w, h = x1 - x0, y1 - y0
    r = max(0.0, min(r, w / 2.0, h / 2.0))
    pts = []
    pts += _arc(x1 - r, y0 + r, r, -math.pi / 2, 0)            # top-right
    pts += _arc(x1 - r, y1 - r, r, 0, math.pi / 2)              # bottom-right
    if tail is not None:
        bl, br, tx, ty = tail
        pts += [(br, y1), (tx, ty), (bl, y1)]                    # right-to-left
    pts += _arc(x0 + r, y1 - r, r, math.pi / 2, math.pi)        # bottom-left
    pts += _arc(x0 + r, y0 + r, r, math.pi, 1.5 * math.pi)      # top-left
    return pts


def _star(x0, y0, x1, y1, points, inner):
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    out = []
    for i in range(points * 2):
        a = -math.pi / 2 + math.pi * i / points
        s = 1.0 if i % 2 == 0 else inner
        out.append((cx + math.cos(a) * rx * s, cy + math.sin(a) * ry * s))
    return out


def _regular(x0, y0, x1, y1, sides):
    # Same geometry as shapes.polygon_points, so nothing moves for anyone
    # who already drew a hexagon.
    from .shapes import polygon_points
    return polygon_points((x0, y0, x1, y1), sides, 0)


# ---- the shapes ---------------------------------------------------------------
# Each takes the drag rect and keyword params, and returns a list of rings.

def _rectangle(x0, y0, x1, y1, **_):
    return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]


def _rounded_rectangle(x0, y0, x1, y1, radius=0, **_):
    r = float(radius) if radius else 0.18 * min(x1 - x0, y1 - y0)
    return [_rounded_ring(x0, y0, x1, y1, r)]


def _ellipse(x0, y0, x1, y1, **_):
    return [_ellipse_ring((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                          (x1 - x0) / 2.0, (y1 - y0) / 2.0)]


def _unit(points):
    def shape(x0, y0, x1, y1, **_):
        return [_map(points, x0, y0, x1, y1)]
    return shape


def _star_n(points, inner):
    def shape(x0, y0, x1, y1, **_):
        return [_star(x0, y0, x1, y1, points, inner)]
    return shape


def _regular_n(sides):
    def shape(x0, y0, x1, y1, **_):
        return [_regular(x0, y0, x1, y1, sides)]
    return shape


def _callout_rounded(x0, y0, x1, y1, radius=0, **_):
    w, h = x1 - x0, y1 - y0
    body_bottom = y0 + 0.8 * h
    # The corner radius is capped so the tail's base always lands on the
    # straight part of the bottom edge, never inside a corner arc.
    r = float(radius) if radius else 0.18 * min(w, 0.8 * h)
    r = min(r, 0.18 * w)
    tail = (x0 + 0.2 * w, x0 + 0.38 * w, x0 + 0.08 * w, y1)
    return [_rounded_ring(x0, y0, x1, body_bottom, r, tail)]


def _callout_oval(x0, y0, x1, y1, **_):
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + 0.5 * w, y0 + 0.42 * h
    rx, ry = 0.5 * w, 0.42 * h
    # Splice the tail into the lower-left of the rim: walk the ellipse the
    # long way round from where the tail leaves to where it returns.
    a_out, a_back = math.radians(128), math.radians(104)
    rim = _ellipse_ring(cx, cy, rx, ry, a_out, a_back + 2 * math.pi, close=False)
    return [rim + [(x0 + 0.1 * w, y1)]]


def _callout_cloud(x0, y0, x1, y1, **_):
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + 0.5 * w, y0 + 0.4 * h
    rx, ry = 0.5 * w, 0.4 * h
    bumps = 9
    n = _segments(math.pi * (rx + ry) * 1.3, lo=90, hi=540)
    body = []
    for i in range(n):
        a = 2 * math.pi * i / n
        # |sin| gives rounded bumps with sharp cusps between them -- the
        # scalloped edge people read as a cloud.
        s = 0.86 + 0.14 * abs(math.sin(bumps * a / 2.0))
        body.append((cx + rx * s * math.cos(a), cy + ry * s * math.sin(a)))
    dot1 = _ellipse_ring(x0 + 0.2 * w, y0 + 0.86 * h, 0.07 * w, 0.06 * h)
    dot2 = _ellipse_ring(x0 + 0.08 * w, y0 + 0.96 * h, 0.04 * w, 0.035 * h)
    return [body, dot1, dot2]


def _gear(x0, y0, x1, y1, teeth=8, **_):
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    n = max(4, int(teeth))
    p = math.pi / n
    outer, root = 1.0, 0.76
    ring = []
    for i in range(n):
        base = 2 * math.pi * i / n - math.pi / 2
        for r, a in ((root, base - 0.78 * p), (outer, base - 0.42 * p),
                     (outer, base + 0.42 * p), (root, base + 0.78 * p),
                     (root, base + p)):
            ring.append((cx + rx * r * math.cos(a), cy + ry * r * math.sin(a)))
    hole = _ellipse_ring(cx, cy, rx * 0.34, ry * 0.34)
    return [ring, hole]


def _heart(x0, y0, x1, y1, **_):
    n = _segments(math.pi * ((x1 - x0) + (y1 - y0)) / 2.0, lo=64, hi=400)
    raw = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        raw.append((x, y))
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    lx, hx, ly, hy = min(xs), max(xs), min(ys), max(ys)
    return [[(x0 + (x - lx) / (hx - lx) * (x1 - x0),
              y0 + (y - ly) / (hy - ly) * (y1 - y0)) for x, y in raw]]


# id -> (label, category, builder). Order within a category is display order.
CATALOG = {}


def _add(ident, label, category, builder):
    CATALOG[ident] = (label, category, builder)


_add("rectangle", "Rectangle", BASIC, _rectangle)
_add("rounded_rectangle", "Rounded Rectangle", BASIC, _rounded_rectangle)
_add("ellipse", "Ellipse", BASIC, _ellipse)
_add("diamond", "Diamond", BASIC, _unit([(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]))
_add("trapezoid", "Trapezoid", BASIC, _unit([(0.25, 0), (0.75, 0), (1, 1), (0, 1)]))
_add("parallelogram", "Parallelogram", BASIC,
     _unit([(0.25, 0), (1, 0), (0.75, 1), (0, 1)]))
_add("triangle", "Triangle", BASIC, _unit([(0.5, 0), (1, 1), (0, 1)]))
_add("right_triangle", "Right Triangle", BASIC, _unit([(0, 0), (1, 1), (0, 1)]))

_add("pentagon", "Pentagon", POLY, _regular_n(5))
_add("hexagon", "Hexagon", POLY, _regular_n(6))
_add("star3", "3-Point Star", POLY, _star_n(3, 0.25))
_add("star4", "4-Point Star", POLY, _star_n(4, 0.3))
_add("star5", "5-Point Star", POLY, _star_n(5, 0.382))
_add("star6", "6-Point Star", POLY, _star_n(6, 0.577))
_add("heptagon", "Heptagon", POLY, _regular_n(7))
_add("octagon", "Octagon", POLY, _regular_n(8))
_add("star8", "8-Point Star", POLY, _star_n(8, 0.55))

_add("arrow", "Arrow", ARROWS,
     _unit([(0, 0.3), (0.6, 0.3), (0.6, 0), (1, 0.5), (0.6, 1), (0.6, 0.7), (0, 0.7)]))
_add("notched_arrow", "Notched Arrow", ARROWS,
     _unit([(0, 0.3), (0.6, 0.3), (0.6, 0), (1, 0.5), (0.6, 1), (0.6, 0.7),
            (0, 0.7), (0.16, 0.5)]))
_add("pentagon_arrow", "Pentagon Arrow", ARROWS,
     _unit([(0, 0), (0.68, 0), (1, 0.5), (0.68, 1), (0, 1)]))
_add("chevron", "Chevron", ARROWS,
     _unit([(0, 0), (0.68, 0), (1, 0.5), (0.68, 1), (0, 1), (0.32, 0.5)]))

_add("callout_rect", "Rectangle Callout", CALLOUTS,
     _unit([(0, 0), (1, 0), (1, 0.8), (0.38, 0.8), (0.08, 1), (0.2, 0.8), (0, 0.8)]))
_add("callout_rounded", "Rounded Callout", CALLOUTS, _callout_rounded)
_add("callout_oval", "Oval Callout", CALLOUTS, _callout_oval)
_add("callout_cloud", "Cloud Callout", CALLOUTS, _callout_cloud)

_add("lightning", "Lightning Bolt", SYMBOLS,
     _unit([(0.4, 0), (0.75, 0), (0.55, 0.35), (0.9, 0.35), (0.3, 1),
            (0.45, 0.55), (0.1, 0.55)]))
_add("check", "Check Mark", SYMBOLS,
     _unit([(0, 0.55), (0.15, 0.4), (0.38, 0.62), (0.85, 0.05), (1, 0.2), (0.38, 0.95)]))
_add("cross", "Cross", SYMBOLS,
     _unit([(0.15, 0), (0.5, 0.35), (0.85, 0), (1, 0.15), (0.65, 0.5), (1, 0.85),
            (0.85, 1), (0.5, 0.65), (0.15, 1), (0, 0.85), (0.35, 0.5), (0, 0.15)]))
_add("gear", "Gear", SYMBOLS, _gear)
_add("heart", "Heart", SYMBOLS, _heart)

CATEGORIES = (BASIC, POLY, ARROWS, CALLOUTS, SYMBOLS)


def grouped():
    """((category, (id, ...)), ...) in display order."""
    return tuple((cat, tuple(i for i, (_l, c, _b) in CATALOG.items() if c == cat))
                 for cat in CATEGORIES)


def label(ident):
    return CATALOG[ident][0] if ident in CATALOG else ident


def rings(ident, rect, **params):
    """The closed rings for a shape in a float rect (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
        return []
    return CATALOG[ident][2](float(x0), float(y0), float(x1), float(y1), **params)


# ---- rasterising ---------------------------------------------------------------

def _supersample_factor(w, h, antialias):
    if not antialias:
        return 1
    area = w * h
    if area <= 512 * 512:
        return 4
    if area <= 1024 * 1024:
        return 3
    if area <= 2048 * 2048:
        return 2
    return 1


def ring_bounds(ring_list, width, height, pad):
    xs = [x for r in ring_list for x, _ in r]
    ys = [y for r in ring_list for _, y in r]
    if not xs:
        return None
    return Rect.from_bounds(int(math.floor(min(xs))) - pad, int(math.floor(min(ys))) - pad,
                            int(math.ceil(max(xs))) + pad, int(math.ceil(max(ys))) + pad
                            ).clipped_to(width, height)


def rasterize_rings(width, height, ring_list, style="both", stroke=1.0, antialias=True):
    """Full-canvas (height, width) uint8 coverage for a set of rings.

    `style` is "fill", "outline" or "both". Fill combines rings even-odd;
    the outline is every ring's stroke, maxed together.
    """
    out = np.zeros((height, width), dtype=np.uint8)
    ring_list = [r for r in ring_list if len(r) >= 3]
    if not ring_list:
        return out
    pad = int(math.ceil(stroke / 2.0)) + 2
    box = ring_bounds(ring_list, width, height, pad)
    if box is None:
        return out
    ss = _supersample_factor(box.w, box.h, antialias)
    W, H = box.w * ss, box.h * ss

    def local(ring):
        return [((x - box.x) * ss, (y - box.y) * ss) for x, y in ring]

    acc = np.zeros((H, W), dtype=bool)
    if style in ("fill", "both"):
        for ring in ring_list:
            img = Image.new("1", (W, H), 0)
            ImageDraw.Draw(img).polygon(local(ring), fill=1)
            acc ^= np.asarray(img, dtype=bool)
    if style in ("outline", "both"):
        width_px = max(1, int(round(stroke * ss)))
        edge = Image.new("1", (W, H), 0)
        draw = ImageDraw.Draw(edge)
        for ring in ring_list:
            pts = local(ring)
            # Repeat the first two points so the closing corner gets a
            # proper join instead of a notch.
            draw.line(pts + pts[:2], fill=1, width=width_px, joint="curve")
            if width_px > 2:
                rad = width_px / 2.0
                for x, y in pts:
                    draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=1)
        acc |= np.asarray(edge, dtype=bool)

    if ss == 1:
        block = acc.astype(np.uint8) * 255
    else:
        block = acc.reshape(box.h, ss, box.w, ss).mean(axis=(1, 3))
        block = np.rint(block * 255.0).astype(np.uint8)
    out[box.slice()] = block
    return out


def coverage(width, height, ident, rect, style="both", stroke=1.0, antialias=True,
             **params):
    return rasterize_rings(width, height, rings(ident, rect, **params), style,
                           stroke, antialias)
