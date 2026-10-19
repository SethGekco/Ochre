# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Intelligent scissors: the cheapest path between two points along edges.

Pixels on strong edges are cheap to travel through, flat areas expensive, so
the shortest path between two clicked anchors hugs the outline of whatever
lies between them. That is GIMP's Scissors Select and Photoshop's Magnetic
Lasso, minus the auto-placed anchors.

The shortest path is found by vectorised relaxation rather than a heap-based
Dijkstra: every iteration relaxes all pixels in all eight directions at once
with numpy, and stops when nothing improves. Restricting the search to a
window around the two anchors -- paths that wander far from the straight
line are not what anyone clicked for -- keeps a typical segment to a few
tens of milliseconds.
"""

import math

import numpy as np

_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def edge_cost(rgba):
    """Per-pixel travel cost: near 0 on strong edges, 1 on flat colour."""
    a = rgba[..., 3].astype(np.float64) / 255.0
    lum = (0.2126 * rgba[..., 0] + 0.7152 * rgba[..., 1] + 0.0722 * rgba[..., 2]) * a
    gy, gx = np.gradient(lum.astype(np.float64))
    mag = np.hypot(gx, gy)
    peak = np.percentile(mag, 99) if mag.size else 1.0
    norm = np.clip(mag / max(peak, 1e-6), 0.0, 1.0)
    return (1.0 - norm) * 0.95 + 0.05


def _shift(arr, dy, dx, fill):
    out = np.full_like(arr, fill)
    h, w = arr.shape
    ys = slice(max(dy, 0), h + min(dy, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = arr[yd, xd]
    return out


def path(cost, start, end, margin=None, max_iter=None):
    """Pixel path [(x, y), ...] from start to end, following low cost."""
    h, w = cost.shape
    sx, sy = int(round(start[0])), int(round(start[1]))
    ex, ey = int(round(end[0])), int(round(end[1]))
    sx, sy = min(max(sx, 0), w - 1), min(max(sy, 0), h - 1)
    ex, ey = min(max(ex, 0), w - 1), min(max(ey, 0), h - 1)
    if (sx, sy) == (ex, ey):
        return [(sx, sy)]
    span = math.hypot(ex - sx, ey - sy)
    m = int(margin if margin is not None else max(8, span * 0.35))
    x0, x1 = max(0, min(sx, ex) - m), min(w, max(sx, ex) + m + 1)
    y0, y1 = max(0, min(sy, ey) - m), min(h, max(sy, ey) + m + 1)
    c = cost[y0:y1, x0:x1]
    wh, ww = c.shape
    inf = np.float64(1e18)
    dist = np.full((wh, ww), inf)
    dist[sy - y0, sx - x0] = 0.0
    pred = np.full((wh, ww), -1, dtype=np.int8)
    limit = max_iter or (wh + ww) * 4
    for _ in range(limit):
        changed = False
        for k, (dy, dx) in enumerate(_DIRS):
            step = math.sqrt(2.0) if dy and dx else 1.0
            # Arriving at a pixel FROM its (dy,dx) neighbour costs the
            # destination pixel's cost times the step length.
            cand = _shift(dist, dy, dx, inf) + c * step
            better = cand < dist - 1e-12
            if better.any():
                dist[better] = cand[better]
                pred[better] = k
                changed = True
        if not changed:
            break
    out = []
    cx, cy = ex - x0, ey - y0
    guard = wh * ww
    while guard > 0:
        out.append((cx + x0, cy + y0))
        k = pred[cy, cx]
        if k < 0:
            break
        dy, dx = _DIRS[k]
        cx, cy = cx - dx, cy - dy
        guard -= 1
    out.reverse()
    return out
