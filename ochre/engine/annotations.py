# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Marks on a document that are not pixels: guides, samplers, counts, notes, slices.

Each lives on the Document as a plain list of tuples, so a snapshot for undo
is a shallow list copy and the file format is one INI line per mark:

  doc.guides    [("h" | "v", position), ...]    Photoshop / GIMP guides
  doc.samplers  [(x, y), ...]                   Photoshop's Color Sampler
  doc.counts    [(x, y), ...]                   Photoshop's Count tool
  doc.notes     [(x, y, text), ...]             Photoshop's Note tool
  doc.slices    [(name, (x, y, w, h)), ...]     slices, for Export Slices

Positions are in document coordinates on pixel CORNERS for guides (a guide
at 100 runs between pixel 99 and pixel 100), and pixel coordinates for the
point marks.
"""

import math

import numpy as np

H, V = "h", "v"
MAX_SAMPLERS = 10        # Photoshop's limit; a readout of more is unreadable


def snap(x, y, guides=(), grid=None, bounds=None, tolerance=4.0):
    """Snap a point to the nearest guide, grid line or canvas edge.

    Each axis snaps independently, to whichever candidate is closest within
    `tolerance` document pixels, so a point near a vertical guide and a
    horizontal grid line snaps to their crossing. grid is (spacing_x,
    spacing_y[, offset_x, offset_y]) or None; bounds is (width, height).
    Returns (x, y, snapped_x, snapped_y).
    """
    xs, ys = [], []
    for orient, pos in guides:
        (ys if orient == H else xs).append(float(pos))
    if grid:
        gx, gy = float(grid[0]), float(grid[1])
        ox = float(grid[2]) if len(grid) > 2 else 0.0
        oy = float(grid[3]) if len(grid) > 3 else 0.0
        if gx > 0:
            xs.append(ox + round((x - ox) / gx) * gx)
        if gy > 0:
            ys.append(oy + round((y - oy) / gy) * gy)
    if bounds:
        xs.extend((0.0, float(bounds[0])))
        ys.extend((0.0, float(bounds[1])))

    def nearest(v, cands):
        best, out = tolerance, None
        for c in cands:
            d = abs(c - v)
            if d <= best:
                best, out = d, c
        return out
    sx, sy = nearest(x, xs), nearest(y, ys)
    return (x if sx is None else sx, y if sy is None else sy,
            sx is not None, sy is not None)


def slices_from_guides(guides, width, height):
    """Cut the canvas at every guide: Photoshop's Slices From Guides."""
    xs = sorted({0.0, float(width)} | {float(p) for o, p in guides if o == V and 0 < p < width})
    ys = sorted({0.0, float(height)} | {float(p) for o, p in guides if o == H and 0 < p < height})
    out = []
    for r, (y0, y1) in enumerate(zip(ys, ys[1:])):
        for c, (x0, x1) in enumerate(zip(xs, xs[1:])):
            x0i, y0i = int(round(x0)), int(round(y0))
            w, h = int(round(x1)) - x0i, int(round(y1)) - y0i
            if w > 0 and h > 0:
                out.append(("slice_%02d_%02d" % (r + 1, c + 1), (x0i, y0i, w, h)))
    return out


def sample(rgba, x, y, size=1):
    """Average colour of a size x size square centred on a pixel.

    Alpha-weighted, like every average here: averaging straight colour
    across differing alpha is what makes a sampler read dark at an edge.
    Returns (r, g, b, a) ints, or None off the canvas.
    """
    h, w = rgba.shape[:2]
    cx, cy = int(math.floor(x)), int(math.floor(y))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    r = max(0, int(size) // 2)
    block = rgba[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
    block = block.reshape(-1, 4).astype(np.float64)
    a = block[:, 3]
    total = a.sum()
    if total <= 0:
        return (0, 0, 0, 0)
    rgb = (block[:, :3] * a[:, None]).sum(axis=0) / total
    return (int(round(rgb[0])), int(round(rgb[1])), int(round(rgb[2])),
            int(round(a.mean())))


def hsv(rgb):
    """(h degrees, s %, v %) for an (r, g, b) triple, for the Info readout."""
    r, g, b = (c / 255.0 for c in rgb[:3])
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0:
        hue = 0.0
    elif mx == r:
        hue = (60 * ((g - b) / d)) % 360
    elif mx == g:
        hue = 60 * ((b - r) / d) + 120
    else:
        hue = 60 * ((r - g) / d) + 240
    sat = 0.0 if mx == 0 else d / mx
    return (int(round(hue)) % 360, int(round(sat * 100)), int(round(mx * 100)))


def measure(x0, y0, x1, y1):
    """Distance, angle (degrees, counter-clockwise from +x as GIMP reports)
    and the horizontal / vertical components of a measured line."""
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(-dy, dx))
    return {"distance": dist, "angle": angle, "dx": dx, "dy": dy}


def straighten_angle(x0, y0, x1, y1):
    """The rotation (degrees, clockwise positive) that makes a measured line
    level, or plumb if it is closer to vertical: GIMP's Straighten."""
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
    # Bring into (-45, 45] from the nearest axis.
    k = round(ang / 90.0)
    return -(ang - 90.0 * k)


# ---- serialisation ------------------------------------------------------------

def encode_note(x, y, text):
    return "%g,%g,%s" % (x, y, str(text).replace("\\", "\\\\").replace("\n", "\\n"))


def decode_note(value):
    parts = str(value).split(",", 2)
    if len(parts) < 2:
        return None
    text = parts[2] if len(parts) > 2 else ""
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            out.append("\n" if text[i + 1] == "n" else text[i + 1])
            i += 2
            continue
        out.append(c)
        i += 1
    return (float(parts[0]), float(parts[1]), "".join(out))
