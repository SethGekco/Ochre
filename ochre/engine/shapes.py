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


# Regular shapes, as (sides, star points skipped). `skip` of 0 means a plain
# polygon; 2 gives the classic five-pointed star, 3 a sharper six-pointed one.
# Expressed this way because a star is not a special case of anything -- it is
# the same ring of vertices visited in a different order, so one generator
# covers both and there is no second code path to get wrong.
POLYGONS = {
    "triangle": (3, 0),
    "diamond": (4, 0),
    "pentagon": (5, 0),
    "hexagon": (6, 0),
    "heptagon": (7, 0),
    "octagon": (8, 0),
    "star5": (5, 2),
    "star6": (6, 2),
    "star8": (8, 3),
}


def polygon_points(rect, sides, skip=0, rotation=None):
    """Vertices of a regular polygon or star inscribed in `rect`.

    Inscribed in the drag rectangle rather than in a circle, so dragging a
    wide box gives a wide pentagon. That is what every other shape tool in
    the editor does and what a user dragging a box expects; a shape that
    refused to be anything but circular would be the odd one out.

    `rotation` defaults to putting a vertex at the top, which is the
    orientation people picture for a triangle, a pentagon or a star. An even
    polygon gets a half-step extra so it sits flat rather than on a point.
    """
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0
    sides = max(3, int(sides))

    if rotation is None:
        rotation = -math.pi / 2.0
        # Even polygons look settled sitting on an edge rather than a point.
        # Four sides is the exception: a flat-topped quadrilateral is just a
        # rectangle, which the tool already offers, so the four-sided entry
        # is the one standing on its point -- which is what "diamond" means.
        if skip == 0 and sides % 2 == 0 and sides != 4:
            rotation += math.pi / sides

    if skip:
        # A star is the outer ring with an inner ring interleaved. The inner
        # radius is derived from the skip rather than picked by eye, so the
        # limbs stay straight -- choosing it by hand produces points that
        # visibly bend where the edges meet.
        inner = math.cos(math.pi * skip / sides) / math.cos(math.pi * (skip - 1) / sides)
        inner = min(max(inner, 0.05), 0.95)
        points = []
        for i in range(sides * 2):
            angle = rotation + math.pi * i / sides
            scale = 1.0 if i % 2 == 0 else inner
            points.append((cx + math.cos(angle) * rx * scale,
                           cy + math.sin(angle) * ry * scale))
        return points

    return [(cx + math.cos(rotation + 2.0 * math.pi * i / sides) * rx,
             cy + math.sin(rotation + 2.0 * math.pi * i / sides) * ry)
            for i in range(sides)]


def coverage(width, height, kind, geometry, style=BOTH, stroke=1.0,
             antialias=True, **params):
    """Dispatch by shape name. `geometry` is a rect tuple or a point list.

    Rectangle and ellipse keep their analytic rasterisers, which the tests
    pin for exact symmetry. Every other named shape comes from the catalog
    in shapelib, which is the single definition the chooser's icons are
    drawn from too.
    """
    from . import shapelib
    if kind in shapelib.CATALOG and kind not in ("rectangle", "ellipse"):
        return shapelib.coverage(width, height, kind, geometry, style, stroke,
                                 antialias, **params)
    if kind == "rectangle":
        return rect_coverage(width, height, geometry, style, stroke, antialias)
    if kind == "ellipse":
        return ellipse_coverage(width, height, geometry, style, stroke, antialias)
    if kind == "line":
        x0, y0, x1, y1 = geometry
        return line_coverage(width, height, x0, y0, x1, y1, stroke, antialias)
    if kind in ("freeform", "polygon"):
        return polygon_coverage(width, height, geometry, style, stroke,
                                antialias,
                                closed=(kind == "polygon" or style != OUTLINE))
    if kind in POLYGONS:
        sides, skip = POLYGONS[kind]
        return polygon_coverage(width, height,
                                polygon_points(geometry, sides, skip),
                                style, stroke, antialias, closed=True)
    raise ValueError("unknown shape %r" % (kind,))


def bounds_for(width, height, kind, geometry, stroke=1.0):
    """The dirty rect a shape will touch, padded for stroke and antialiasing."""
    from . import shapelib
    pad = int(math.ceil(stroke / 2.0)) + 2
    if kind in ("rectangle", "ellipse", "line") or kind in POLYGONS \
            or kind in shapelib.CATALOG:
        # A polygon is inscribed in the drag rect, so the rect already bounds
        # it -- no need to generate the vertices just to measure them.
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
