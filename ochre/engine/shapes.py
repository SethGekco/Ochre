# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Shape coverage: outlines and fills, as antialiased masks.

Nothing here rasterises anything new. A shape's coverage is exactly what the
selection rasterisers already produce, so this module composes them:

    fill     = coverage(shape)
    outline  = coverage(shape grown by w/2) - coverage(shape shrunk by w/2)

That subtraction is the whole trick, and it is worth preferring over a
dedicated stroke rasteriser for three reasons. It inherits the symmetric
supersampling those rasterisers were fixed to use, so a circle's outline is
as symmetric as its fill. It produces genuine antialiased coverage rather
than a hard edge. And there is one rasteriser per shape to be correct rather
than two.

Everything returns a (H, W) uint8 coverage plane, which is the same currency
a brush dab and a selection mask trade in -- so a shape clips against a
selection through exactly the same multiply, with no shape-specific path.
"""

import math

import numpy as np

from .geometry import Rect
from .selection import rasterize_ellipse, rasterize_polygon, rasterize_rect

OUTLINE = "outline"
FILL = "fill"
BOTH = "both"
STYLES = (OUTLINE, FILL, BOTH)


def _inset(rect, amount):
    """Shrink a float rect by `amount` on each side, collapsing rather than
    inverting when it runs out of room."""
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    nx0, nx1 = x0 + amount, x1 - amount
    ny0, ny1 = y0 + amount, y1 - amount
    if nx1 <= nx0:
        nx0 = nx1 = cx
    if ny1 <= ny0:
        ny0 = ny1 = cy
    return (nx0, ny0, nx1, ny1)


def _subtract(outer, inner):
    """outer minus inner, multiplicatively, so soft edges stay soft."""
    from .arith import mul255
    return mul255(outer, (255 - inner).astype(np.uint8))


def rect_coverage(width, height, rect, style=BOTH, stroke=1.0, antialias=True):
    """Coverage for an axis-aligned rectangle."""
    outer = rasterize_rect(width, height, _inset(rect, -stroke / 2.0), antialias)
    if style == FILL:
        return rasterize_rect(width, height, rect, antialias)
    inner = rasterize_rect(width, height, _inset(rect, stroke / 2.0), antialias)
    if style == BOTH:
        return outer
    return _subtract(outer, inner)


def ellipse_coverage(width, height, rect, style=BOTH, stroke=1.0, antialias=True):
    """Coverage for an ellipse inscribed in rect."""
    outer = rasterize_ellipse(width, height, _inset(rect, -stroke / 2.0), antialias)
    if style == FILL:
        return rasterize_ellipse(width, height, rect, antialias)
    inner = rasterize_ellipse(width, height, _inset(rect, stroke / 2.0), antialias)
    if style == BOTH:
        return outer
    return _subtract(outer, inner)


def polygon_coverage(width, height, points, style=BOTH, stroke=1.0,
                     antialias=True, closed=True):
    """Coverage for a polygon or polyline.

    An open polyline has no interior, so `fill` is meaningless there and the
    outline is drawn as a chain of thick segments instead.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        return np.zeros((height, width), dtype=np.uint8)
    if not closed or len(pts) < 3:
        return polyline_coverage(width, height, pts, stroke, antialias)
    if style == FILL:
        return rasterize_polygon(width, height, pts, antialias)
    body = rasterize_polygon(width, height, pts, antialias)
    edge = polyline_coverage(width, height, pts + [pts[0]], stroke, antialias)
    if style == BOTH:
        return np.maximum(body, edge)
    return edge


def line_coverage(width, height, x0, y0, x1, y1, stroke=1.0, antialias=True):
    """A single thick line segment, as a rotated rectangle."""
    return polyline_coverage(width, height, [(x0, y0), (x1, y1)], stroke, antialias)


def polyline_coverage(width, height, points, stroke=1.0, antialias=True):
    """A chain of thick segments with round joins.

    Each segment becomes a quad and each interior vertex a disc. Taking the
    per-piece maximum rather than summing matters: overlapping coverage that
    added would saturate to a visible bright seam at every joint.
    """
    half = max(0.5, float(stroke) / 2.0)
    out = np.zeros((height, width), dtype=np.uint8)
    pts = [(float(x), float(y)) for x, y in points]

    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        nx, ny = -dy / length * half, dx / length * half
        quad = [(ax + nx, ay + ny), (bx + nx, by + ny),
                (bx - nx, by - ny), (ax - nx, ay - ny)]
        out = np.maximum(out, rasterize_polygon(width, height, quad, antialias))

    # Round the joins and the caps, so a bent polyline has no notch.
    if half > 0.6:
        for (px, py) in pts:
            disc = (px - half, py - half, px + half, py + half)
            out = np.maximum(out, rasterize_ellipse(width, height, disc, antialias))
    return out


def coverage(width, height, kind, geometry, style=BOTH, stroke=1.0,
             antialias=True):
    """Dispatch by shape name. `geometry` is a rect tuple or a point list."""
    if kind == "rectangle":
        return rect_coverage(width, height, geometry, style, stroke, antialias)
    if kind == "ellipse":
        return ellipse_coverage(width, height, geometry, style, stroke, antialias)
    if kind == "line":
        x0, y0, x1, y1 = geometry
        return line_coverage(width, height, x0, y0, x1, y1, stroke, antialias)
    if kind == "freeform":
        return polygon_coverage(width, height, geometry, style, stroke,
                                antialias, closed=(style != OUTLINE))
    raise ValueError("unknown shape %r" % (kind,))


def bounds_for(width, height, kind, geometry, stroke=1.0):
    """The dirty rect a shape will touch, padded for stroke and antialiasing."""
    pad = int(math.ceil(stroke / 2.0)) + 2
    if kind in ("rectangle", "ellipse", "line"):
        x0, y0, x1, y1 = geometry
    else:
        if not geometry:
            return None
        xs = [p[0] for p in geometry]
        ys = [p[1] for p in geometry]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return Rect.from_bounds(int(math.floor(min(x0, x1))) - pad,
                            int(math.floor(min(y0, y1))) - pad,
                            int(math.ceil(max(x0, x1))) + pad,
                            int(math.ceil(max(y0, y1))) + pad
                            ).clipped_to(width, height)
