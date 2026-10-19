# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Perspective Warp (Photoshop's Edit > Perspective Warp): several quads
laid over the planes of a picture -- the two visible walls of a building,
say -- sharing their edges, then pulled into a new perspective.

Each quad carries its own homography from where it was laid out to where it
was pulled. Every output pixel takes the transform of the pulled quad it
falls in, or, outside them all, of the nearest one, so the whole layer moves
and nothing is left behind. Shared corners move together, so neighbouring
planes stay joined along their common edge.
"""

import numpy as np

from . import warp


def _inside(quad, xs, ys):
    """Boolean mask: points inside a convex quad (either winding)."""
    q = np.asarray(quad, np.float64)
    signs = []
    for i in range(4):
        ax, ay = q[i]
        bx, by = q[(i + 1) % 4]
        signs.append((bx - ax) * (ys - ay) - (by - ay) * (xs - ax))
    s = np.stack(signs)
    return np.all(s >= -1e-9, axis=0) | np.all(s <= 1e-9, axis=0)


def source_coords(layouts, warps, width, height):
    """(sx, sy): for every output pixel, where to sample the original."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    xs += 0.5
    ys += 0.5
    owner = np.full((height, width), -1, np.int64)
    for i, q in enumerate(warps):
        owner[(owner < 0) & _inside(q, xs, ys)] = i
    if (owner < 0).any():
        cents = np.asarray([np.mean(q, axis=0) for q in warps])
        d = np.stack([(xs - c[0]) ** 2 + (ys - c[1]) ** 2 for c in cents])
        nearest = np.argmin(d, axis=0)
        owner = np.where(owner < 0, nearest, owner)
    sx = np.empty_like(xs)
    sy = np.empty_like(ys)
    for i, (lay, wq) in enumerate(zip(layouts, warps)):
        sel = owner == i
        if not sel.any():
            continue
        inv = warp.homography(wq, lay)              # pulled -> laid out (inverse map)
        pts = warp.apply(inv, np.stack([xs[sel], ys[sel]], axis=1))
        sx[sel], sy[sel] = pts[:, 0], pts[:, 1]
    return sx, sy


def render(src, layouts, warps):
    """The layer, every quad pulled from its layout to its warp."""
    h, w = src.shape[:2]
    if not layouts:
        return np.array(src, copy=True)
    sx, sy = source_coords(layouts, warps, w, h)
    return warp.warp_field(src, sx, sy, warp.BILINEAR)


def render_index(src, layouts, warps, transparent=0):
    """The same warp for an index plane: sampled nearest, never blended;
    where nothing maps, the layer's transparent index."""
    h, w = src.shape[:2]
    if not layouts:
        return np.array(src, copy=True)
    sx, sy = source_coords(layouts, warps, w, h)
    idx = warp.sample_index(src, sx, sy)
    return np.where(idx >= 0, idx, transparent).astype(np.uint8)


def quad_grid(quad, n=3):
    """Segments of an n x n grid inside a quad (bilinear), for drawing."""
    q = np.asarray(quad, np.float64)
    out = []
    for k in range(1, n):
        t = k / float(n)
        a = q[0] + (q[1] - q[0]) * t
        b = q[3] + (q[2] - q[3]) * t
        out.append((a[0], a[1], b[0], b[1]))
        a = q[0] + (q[3] - q[0]) * t
        b = q[1] + (q[2] - q[1]) * t
        out.append((a[0], a[1], b[0], b[1]))
    return out
