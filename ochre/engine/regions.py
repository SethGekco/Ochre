# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Artboards and frames: a group whose picture is cut to a region.

  artboard  Photoshop's Artboards -- several pages or screens laid out on
            one canvas, each a group cut to its rectangle, with its own
            background colour, exported one file per artboard.
  frame     Photoshop's Frame tool -- a rectangle or ellipse placeholder;
            whatever is placed inside it is cropped to its shape, and an
            image placed into an empty frame is scaled to fill it.

Both are the same mechanism: a group's `region` attribute, a small dict

    {"type": "artboard" | "frame", "shape": "rect" | "ellipse",
     "rect": [x, y, w, h], "bg": [r, g, b, a] or None}

which the compositor reads (the group composites in isolation, then is
multiplied by the region's coverage) and which follows the image through
resize, crop, flip and rotate like the other marks.
"""

import numpy as np

ARTBOARD = "artboard"
FRAME = "frame"
SHAPES = ("rect", "ellipse")


def make(kind, rect, shape="rect", background=None):
    x, y, w, h = (int(round(v)) for v in rect)
    return {"type": kind, "shape": shape if shape in SHAPES else "rect",
            "rect": [x, y, max(1, w), max(1, h)],
            "bg": None if background is None else [int(v) for v in background]}


def coverage(region, r):
    """Float coverage (r.h, r.w), 0..1, of the region over document rect r.
    A rectangle is exact per pixel; an ellipse is antialiased."""
    x, y, w, h = region["rect"]
    xs = np.arange(r.x, r.x + r.w, dtype=np.float64) + 0.5
    ys = np.arange(r.y, r.y + r.h, dtype=np.float64) + 0.5
    if region.get("shape") == "ellipse":
        rx, ry = w / 2.0, h / 2.0
        cx, cy = x + rx, y + ry
        f = np.sqrt(((xs[None, :] - cx) / rx) ** 2 + ((ys[:, None] - cy) / ry) ** 2)
        return np.clip(0.5 - (f - 1.0) * min(rx, ry), 0.0, 1.0)
    cx = np.clip(np.minimum(xs + 0.5, x + w) - np.maximum(xs - 0.5, x), 0, 1)
    cy = np.clip(np.minimum(ys + 0.5, y + h) - np.maximum(ys - 0.5, y), 0, 1)
    return cy[:, None] * cx[None, :]


def contains(region, px, py):
    x, y, w, h = region["rect"]
    if region.get("shape") == "ellipse":
        rx, ry = w / 2.0, h / 2.0
        return ((px - x - rx) / rx) ** 2 + ((py - y - ry) / ry) ** 2 <= 1.0
    return x <= px < x + w and y <= py < y + h


def translated(region, dx, dy):
    out = dict(region)
    x, y, w, h = region["rect"]
    out["rect"] = [int(round(x + dx)), int(round(y + dy)), w, h]
    return out


def mapped(region, fn):
    """The region after a point mapping (resize, crop, flip, rotate): the
    bounding box of its mapped corners."""
    x, y, w, h = region["rect"]
    pts = [fn(p) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = int(round(min(xs))), int(round(min(ys)))
    out = dict(region)
    out["rect"] = [x0, y0, max(1, int(round(max(xs))) - x0), max(1, int(round(max(ys))) - y0)]
    return out


def cover_matrix(src_w, src_h, region):
    """Matrix placing a src_w x src_h image so it FILLS the region's box,
    centred (the frame crops the overflow) -- Photoshop's placement into a
    frame. Row-major 3x3, source -> document."""
    x, y, w, h = region["rect"]
    # One pixel of overscan on each side: an image edge landing exactly on
    # the frame's edge would be resampled half-transparent there, leaving a
    # pale fringe. Overscanned, the frame crops a clean edge.
    s = max((w + 2) / float(src_w), (h + 2) / float(src_h))
    tx = x + (w - src_w * s) / 2.0
    ty = y + (h - src_h * s) / 2.0
    return [s, 0.0, tx, 0.0, s, ty, 0.0, 0.0, 1.0]


def groups(doc, kind=None):
    """Groups carrying a region (of one kind), bottom-first."""
    return [n for n in doc.root.walk() if n.is_group and getattr(n, "region", None)
            and (kind is None or n.region.get("type") == kind)]


def at(doc, px, py, kind=None):
    """The topmost region group containing the point, or None."""
    hits = [g for g in groups(doc, kind) if g.visible and contains(g.region, px, py)]
    return hits[-1] if hits else None


def frame_of(layer):
    """The frame group a layer is in (or is), or None."""
    node = layer
    while node is not None:
        reg = getattr(node, "region", None)
        if node.is_group and reg and reg.get("type") == FRAME:
            return node
        node = node.parent
    return None


def encode(region):
    x, y, w, h = region["rect"]
    bg = region.get("bg")
    # '|' not ';' -- the INI reader treats ';' as a comment marker.
    return "%s|%s|%d,%d,%d,%d|%s" % (region["type"], region.get("shape", "rect"), x, y, w, h,
                                     "-" if bg is None else ",".join(str(int(v)) for v in bg))


def decode(text):
    try:
        kind, shape, rect, bg = text.split("|")
        nums = [int(v) for v in rect.split(",")]
        colour = None if bg.strip() == "-" else [int(v) for v in bg.split(",")]
        if kind not in (ARTBOARD, FRAME) or len(nums) != 4:
            return None
        return make(kind, nums, shape, colour)
    except (ValueError, AttributeError):
        return None


def safe_name(name, fallback):
    keep = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or ""))
    return keep.strip().replace(" ", "_") or fallback

