# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Vanishing Point: editing in perspective (Photoshop's Filter > Vanishing
Point).

A PLANE is four image points -- the corners of a rectangle as seen in
perspective (a wall, a floor, the side of a box), clockwise from top-left.
A homography maps plane coordinates (u, v), measured in pixels at the
plane's average scale, to the image and back. With it:

  clone_dab   the Vanishing Point stamp: the brush is a circle IN THE
              PLANE, so on screen it shrinks and squashes as it recedes,
              and the source is sampled at a fixed plane-space offset, so a
              tile copied from the near end of a floor lands foreshortened
              at the far end.
  paste_matrix  an image laid onto the plane (Paste in Perspective).
  grid        lines for drawing the plane's grid.

Plain numpy; the planes live on the Document as marks (doc.planes) so they
are undoable, follow resize/crop/rotate and save in .ochre.
"""

import math

import numpy as np

from . import warp


def edges(corners):
    """Average plane width and height, in image pixels."""
    c = np.asarray(corners, np.float64)
    w = (np.hypot(*(c[1] - c[0])) + np.hypot(*(c[2] - c[3]))) / 2.0
    h = (np.hypot(*(c[3] - c[0])) + np.hypot(*(c[2] - c[1]))) / 2.0
    return max(w, 1.0), max(h, 1.0)


def matrices(corners):
    """(H, H_inv, W, H): plane (u, v) in [0, W] x [0, H] -> image, and back."""
    w, h = edges(corners)
    hm = warp.homography([(0, 0), (w, 0), (w, h), (0, h)], corners)
    return hm, np.linalg.inv(hm), w, h


def valid(corners):
    """Four distinct points making a convex quad in a consistent winding --
    the only shapes a real rectangle can project to."""
    if corners is None or len(corners) != 4:
        return False
    c = np.asarray(corners, np.float64)
    cross = []
    for i in range(4):
        a, b, d = c[i], c[(i + 1) % 4], c[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0]))
    cross = np.asarray(cross)
    return bool((np.all(cross > 1e-6) or np.all(cross < -1e-6)))


def ordered(points):
    """Four clicked points in plane order: top-left, top-right, bottom-right,
    bottom-left (sorted around their centre), whatever order they came in."""
    p = np.asarray(points, np.float64)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]                      # clockwise on screen (y down)
    start = int(np.argmin(p[:, 0] + p[:, 1]))   # nearest the top-left
    p = np.roll(p, -start, axis=0)
    return [tuple(map(float, q)) for q in p]


def to_plane(corners, pts):
    _h, inv, _w, _hh = matrices(corners)
    return warp.apply(inv, pts)


def to_image(corners, pts):
    hm, _inv, _w, _hh = matrices(corners)
    return warp.apply(hm, pts)


def inside(corners, x, y):
    u, v = to_plane(corners, [(x, y)])[0]
    w, h = edges(corners)
    return 0 <= u <= w and 0 <= v <= h


def local_scale(corners, x, y):
    """Image pixels per plane unit near (x, y): how much the plane is
    shrunk there by distance."""
    (u, v), = to_plane(corners, [(x, y)])
    pts = to_image(corners, [(u, v), (u + 1, v), (u, v + 1)])
    a, b = pts[1] - pts[0], pts[2] - pts[0]
    return math.sqrt(max(abs(a[0] * b[1] - a[1] * b[0]), 1e-9))


def clone_dab(src, corners, dest_uv, offset_uv, radius, hardness, width, height):
    # src may be an (h, w) INDEX plane: then it is sampled nearest, never
    # blended -- an indexed layer's truth is its indices.
    """One perspective clone dab.

    src       (h, w, 4) uint8 to sample from (a snapshot)
    dest_uv   the brush centre in plane units
    offset_uv source minus destination, in plane units
    radius    the brush radius in plane units

    Returns (x0, y0, patch, coverage) -- coverage uint8, zero outside the
    brush and outside the plane -- or None when nothing is touched.
    """
    hm, inv, pw, ph = matrices(corners)
    ring = [(dest_uv[0] + radius * math.cos(t), dest_uv[1] + radius * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    img = warp.apply(hm, ring)
    x0 = max(0, int(math.floor(img[:, 0].min())) - 1)
    y0 = max(0, int(math.floor(img[:, 1].min())) - 1)
    x1 = min(width, int(math.ceil(img[:, 0].max())) + 2)
    y1 = min(height, int(math.ceil(img[:, 1].max())) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    gx, gy = warp._grid(x0, y0, x1, y1)
    den = inv[2, 0] * gx + inv[2, 1] * gy + inv[2, 2]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    u = (inv[0, 0] * gx + inv[0, 1] * gy + inv[0, 2]) / den
    v = (inv[1, 0] * gx + inv[1, 1] * gy + inv[1, 2]) / den
    d = np.hypot(u - dest_uv[0], v - dest_uv[1]) / max(radius, 1e-6)
    hard = min(max(float(hardness), 0.0), 0.999)
    cov = np.clip((1.0 - d) / (1.0 - hard), 0.0, 1.0)
    cov *= (u >= 0) & (u <= pw) & (v >= 0) & (v <= ph)
    su, sv = u + offset_uv[0], v + offset_uv[1]
    den2 = hm[2, 0] * su + hm[2, 1] * sv + hm[2, 2]
    den2 = np.where(np.abs(den2) < 1e-12, 1e-12, den2)
    sx = (hm[0, 0] * su + hm[0, 1] * sv + hm[0, 2]) / den2
    sy = (hm[1, 0] * su + hm[1, 1] * sv + hm[1, 2]) / den2
    sh, sw = src.shape[:2]
    if src.ndim == 2:
        idx = warp.sample_index(src, sx, sy)
        cov *= idx >= 0
        patch = np.clip(idx, 0, 255).astype(np.uint8)
    else:
        patch = warp._sample_rgba(src, sx, sy, warp.BILINEAR)
    # Only what exists can be cloned: a source sample off the canvas (alpha
    # from outside) must not paint transparency into the destination.
    cov *= (sx >= 0) & (sy >= 0) & (sx <= sw) & (sy <= sh)
    cov8 = np.clip(np.rint(cov * 255), 0, 255).astype(np.uint8)
    if not cov8.any():
        return None
    return x0, y0, patch, cov8


def paste_matrix(corners, src_w, src_h):
    """Matrix laying a src_w x src_h image onto the plane: as large as fits
    with its aspect kept (in plane units), centred. Source -> image."""
    hm, _inv, pw, ph = matrices(corners)
    s = min(pw / float(src_w), ph / float(src_h))
    tu = (pw - src_w * s) / 2.0
    tv = (ph - src_h * s) / 2.0
    fit = np.array([[s, 0, tu], [0, s, tv], [0, 0, 1]], np.float64)
    return hm @ fit


def grid(corners, n=8):
    """Line segments (x0, y0, x1, y1) of an n x n grid on the plane."""
    _hm, _inv, pw, ph = matrices(corners)
    out = []
    for i in range(n + 1):
        t = i / float(n)
        a, b = to_image(corners, [(t * pw, 0), (t * pw, ph)])
        out.append((a[0], a[1], b[0], b[1]))
        a, b = to_image(corners, [(0, t * ph), (pw, t * ph)])
        out.append((a[0], a[1], b[0], b[1]))
    return out
