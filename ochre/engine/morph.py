# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Mask morphology: grow, shrink, border, remove holes, round corners.

Pure numpy, because the engine depends on nothing else. The core is a
Euclidean distance transform LIMITED TO A RADIUS, which is all grow and
shrink ever need and is far cheaper than a full transform:

  1. along each column, the exact 1-D distance to the nearest set pixel,
     from two cumulative passes (vectorised over every column at once);
  2. across each row, the minimum of dx^2 + column_distance^2 over the
     window |dx| <= r -- 2r+1 shifted np.minimum calls.

That is exact Euclidean distance wherever it is <= r, which is what makes a
grown selection round rather than the octagon a repeated 3x3 max produces.
Edges come out antialiased: a pixel half a pixel past the radius is half
covered.
"""

import numpy as np

_BIG = np.float32(1e12)


def _col_dist(mask):
    """Per-column distance (float32) from each pixel to the nearest True."""
    h, w = mask.shape
    idx = np.arange(h, dtype=np.float32)[:, None]
    inf = np.float32(1e9)
    # Nearest set pixel at or above.
    above = np.where(mask, idx, -inf)
    above = np.maximum.accumulate(above, axis=0)
    # Nearest set pixel at or below.
    below = np.where(mask, idx, inf)
    below = np.minimum.accumulate(below[::-1], axis=0)[::-1]
    return np.minimum(idx - above, below - idx)


def distance_sq(mask, radius):
    """Squared distance to the nearest True pixel, exact up to `radius`.

    Beyond the radius the value is only known to be larger; callers compare
    against radius**2 and never look further.
    """
    mask = np.asarray(mask, dtype=bool)
    r = int(np.ceil(radius))
    if not mask.any():
        return np.full(mask.shape, _BIG, dtype=np.float32)
    g = _col_dist(mask)
    g2 = np.where(g > r + 1, _BIG, g * g).astype(np.float32)
    best = g2.copy()
    h, w = mask.shape
    for dx in range(1, r + 1):
        d2 = np.float32(dx * dx)
        if dx < w:
            np.minimum(best[:, dx:], g2[:, :-dx] + d2, out=best[:, dx:])
            np.minimum(best[:, :-dx], g2[:, dx:] + d2, out=best[:, :-dx])
    return best


def _soft(d2, radius):
    """Coverage 0..255 for 'within radius', with a one-pixel antialiased rim."""
    d = np.sqrt(d2.astype(np.float64))
    return np.clip(np.rint((radius + 0.5 - d) * 255.0), 0, 255).astype(np.uint8)


def grow(mask, radius):
    """Dilate a coverage mask (uint8, 0..255) by a Euclidean radius."""
    m = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return m.copy()
    solid = m >= 128
    return np.maximum(m, _soft(distance_sq(solid, radius), radius))


def shrink(mask, radius):
    """Erode by a Euclidean radius: grow the complement, then invert."""
    m = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return m.copy()
    outside = grow((255 - m).astype(np.uint8), radius)
    return (255 - outside).astype(np.uint8)


def border(mask, radius):
    """A band of `radius` either side of the selection's edge."""
    outer = grow(mask, radius)
    inner = shrink(mask, radius)
    return np.clip(outer.astype(np.int32) - inner.astype(np.int32), 0, 255).astype(np.uint8)


def remove_holes(mask):
    """Fill every unselected region that does not touch the canvas edge."""
    from .accel import flood_fill
    m = np.asarray(mask, dtype=np.uint8)
    outside = m < 128
    if not outside.any():
        return m.copy()
    # Pad with a ring of "outside" so a single flood from the corner reaches
    # every region connected to the edge.
    padded = np.pad(outside, 1, constant_values=True).astype(np.uint8) * 255
    reach = flood_fill(padded, 0, 0, 0)[1:-1, 1:-1]
    holes = outside & ~reach
    out = m.copy()
    out[holes] = 255
    return out


def round_corners(mask, radius):
    """Round convex corners: a morphological opening with a disc.

    Shrinking by r and growing back by r leaves straight edges where they
    were and turns every outward corner into a quarter circle of radius r,
    which is what GIMP's "Rounded Rectangle" does to a selection.
    """
    m = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return m.copy()
    # Grown back one pixel further than it was shrunk, then clipped to the
    # original: an opening is a subset by definition, and the extra pixel
    # stops the half-pixel antialiasing convention from eating the last row
    # of every straight edge.
    return np.minimum(m, grow(shrink(m, radius), radius + 1))


def sharpen(mask):
    """Remove partial coverage: every pixel fully in or fully out."""
    return np.where(np.asarray(mask) >= 128, 255, 0).astype(np.uint8)


def component_at(mask, x, y):
    """The connected region of `mask` (>=128) containing (x, y), as bool."""
    from .accel import flood_fill
    m = np.asarray(mask) >= 128
    h, w = m.shape
    if not (0 <= x < w and 0 <= y < h) or not m[y, x]:
        return np.zeros_like(m)
    return flood_fill(m.astype(np.uint8) * 255, int(x), int(y), 0)


def largest_component(mask):
    """The biggest connected region of `mask`, as bool."""
    m = np.asarray(mask) >= 128
    remaining = m.copy()
    best = np.zeros_like(m)
    best_n = 0
    # Seed from the densest points first; stop once the rest cannot beat it.
    while remaining.any():
        ys, xs = np.nonzero(remaining)
        comp = component_at(remaining.astype(np.uint8) * 255, int(xs[0]), int(ys[0]))
        n = int(comp.sum())
        if n > best_n:
            best, best_n = comp, n
        remaining &= ~comp
        if int(remaining.sum()) <= best_n:
            break
    return best
