# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Render > Picture Frame: a decorative border drawn over the picture.

Vines, flowers and leaves laid round the edge of the canvas, or a ruled or
moulded frame. Photoshop's dialog lists 47 frames, each its own pattern of
the same three ingredients -- a vine in one color, flowers from 22 shapes
in another, leaves from 23 in a third -- so the frames are recipes and the
shapes a shared catalogue. The Advanced tab's Number of Lines, Thickness,
Angle, Fade and Invert apply to every frame they make sense for.

Ported from Kooka Pictura (crates/pictura-filters/src/render/picture_frame.rs
and picture_frame/{frames,geometry,moulding,scale,shapes}.rs, GPL-3.0),
which Kooka ported from perfecto25/photorust (core/src/filters/frame.rs,
GPL-3.0). All artwork is procedural parametric geometry; no Adobe asset is
traced. The recipes, shapes, seeded arrangement generator and scan
converter (non-zero winding, four sub-scanlines a row, exact coverage
across) follow Kooka's step for step; Kooka works in 32-bit floats and this
port in 64-bit, so an edge pixel can round one step differently.

The frame is drawn on a clear sheet first and laid over the picture after,
so Fade thins it as a whole.
"""

import math

import numpy as np

from .filters import _Filter
from .props import BoolProperty, ChoiceProperty, ColorProperty, IntProperty, PropertyCollection

PI = math.pi
TAU = 2.0 * math.pi

FRAME_NAMES = (
    "Happy Vine", "Pretty Vine", "Smoke Signals", "Party", "Big Curls", "Tilde", "Romance",
    "Curly Dance", "Wisps", "Spring Weed", "Eyelash", "Magic Smoke", "Curly Vine", "Chainmail",
    "Fun Event", "Bush", "Simple Lace", "Pulse", "Root", "Snakes", "Mustache",
    "Circle Sprinkle", "Aligned Flowers", "Little Flowers", "Snowflakes", "Flurry",
    "Check Marks", "Focused Lines", "Focused Parallel Lines", "Focused Vibration",
    "Zen Garden", "Spikes", "Anemone", "Pinwheel", "Spacing", "Line Box", "Rounded Corners",
    "Inverse Rounded Corners 1", "Inverse Rounded Corners 2", "Dual Rounded Corners 1",
    "Dual Rounded Corners 2", "Art Frame", "Rounded Art Frame", "Inverse Rounded Art Frame 1",
    "Inverse Rounded Art Frame 2", "Dual Rounded Art Frame 1", "Dual Rounded Art Frame 2")

FLOWER_NAMES = (
    "Small Circle", "Circle", "Star Flower", "Small Flower", "Orbit", "Pinwheel", "Petals",
    "Sunflower Gate", "Rose", "Grass Circle", "Sun", "Broken Line Circle", "Magical Door",
    "Twinkle", "Shiny Star", "Star Cloud", "Round Plus", "Plus Mark", "Heart", "Star",
    "Snow Flake", "Cat's Footprint")

LEAF_NAMES = (
    "Circle", "Square", "Drop 1", "Leaf 1", "Drop 2", "Drop 3", "Leaf 2", "Puff", "Leaf 3",
    "Ginkgo leaf", "Triangle", "Leaf 4", "Star 1", "Oval", "Trapezoid", "Lollipop 1",
    "Leaf 5", "Star 2", "Lollipop 2", "Cross", "Heart", "T", "Star 3")


def _key(label):
    return "".join(c if c.isalnum() else "_" for c in label.lower().replace("'", "")).strip("_")


FRAMES = tuple(_key(n) for n in FRAME_NAMES)
FLOWERS = ("none",) + tuple(_key(n) for n in FLOWER_NAMES)
LEAVES = ("none",) + tuple(_key(n) for n in LEAF_NAMES)


def uses(frame):
    """Which of (vine, flower, leaf) frame 1..47 draws; Photoshop greys out
    the controls for the rest."""
    if frame in (5, 11, 12, 13, 16):
        return True, True, True
    if 1 <= frame <= 17:
        return True, True, False
    if frame in (18, 19):
        return True, True, True
    if frame in (20, 21):
        return True, True, False
    if frame in (22, 23):
        return False, True, False
    if 24 <= frame <= 26:
        return False, True, True
    if 27 <= frame <= 32:
        return True, True, False
    if frame == 33:
        return False, True, False
    if frame == 34:
        return True, True, False
    if frame == 35:
        return False, True, False
    # The ruled and moulded frames are drawn in the vine color alone.
    return True, False, False


def uses_lines(frame):
    """Whether Number of Lines sets frame's line density."""
    return frame in (14, 28, 29, 30)


def _rnd(x):
    """Rust's round() for the non-negative values it meets here."""
    return math.floor(x + 0.5)


# ==== rasterising ======================================================================
#
# A contour is an (n, 2) float array; a batch of equal-length contours an
# (k, n, 2) one. Solid contours wind one way, holes the other, and the
# non-zero rule unions the solids and takes the holes back out.

SUBSAMPLES = 4


def _area(c):
    """Twice the signed area, per contour of a batch."""
    x, y = c[..., 0], c[..., 1]
    return (x * np.roll(y, -1, -1) - np.roll(x, -1, -1) * y).sum(-1)


def solid(c):
    c = np.asarray(c, np.float64)
    if c.ndim == 2:
        return c[::-1].copy() if _area(c) < 0 else c
    flip = _area(c) < 0
    c = c.copy()
    c[flip] = c[flip, ::-1]
    return c


def hole(c):
    c = np.asarray(c, np.float64)
    return c[::-1].copy() if _area(c) > 0 else c


def source_over(dst, rgb, alpha, src_a=None):
    """Kooka's straight-alpha source-over of colour(s) `rgb` (N, 3) at
    coverage `alpha` (N,) onto `dst` (N, 4) uint8; returns the new pixels,
    rounded as Kooka's u8 casts round."""
    sa = np.clip(alpha, 0.0, 1.0) * (1.0 if src_a is None else src_a / 255.0)
    da = dst[:, 3] / 255.0
    out_a = sa + da * (1.0 - sa)
    safe = np.where(out_a > 0, out_a, 1.0)
    mix = (np.asarray(rgb, np.float64) / 255.0 * sa[:, None]
           + dst[:, :3] / 255.0 * (da * (1.0 - sa))[:, None]) / safe[:, None]
    out = dst.copy()
    live = sa > 0
    out[live, :3] = np.floor(np.clip(mix[live], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out[live, 3] = np.floor(out_a[live] * 255.0 + 0.5).astype(np.uint8)
    return out


def fill(sheet, contours, colour, alpha):
    """Fill `contours` under the non-zero rule with `colour`, `alpha` of the
    way, anti-aliased: four sub-scanlines a pixel row, exact coverage across."""
    if alpha <= 0:
        return
    h, w = sheet.shape[:2]
    edges = []
    for c in contours:
        c = np.asarray(c, np.float64)
        if c.shape[-2] < 3:
            continue
        a = c.reshape(-1, c.shape[-2], 2)
        edges.append(np.concatenate([a, np.roll(a, -1, 1)], -1).reshape(-1, 4))
    if not edges:
        return
    e = np.concatenate(edges)
    # A point that is not a number would throw the winding count off.
    e = e[np.isfinite(e).all(1)]
    if not len(e):
        return
    x0, y0, x1, y1 = e.T
    # Sub-scanline K sits at y = (K + 0.5) / 4; an edge crosses it when it
    # lies in [min y, max y) -- half-open, so a vertex on it counts once.
    lo = np.clip(np.ceil(np.minimum(y0, y1) * SUBSAMPLES - 0.5), 0, h * SUBSAMPLES).astype(np.int64)
    hi = np.clip(np.ceil(np.maximum(y0, y1) * SUBSAMPLES - 0.5), 0, h * SUBSAMPLES).astype(np.int64)
    cnt = np.maximum(hi - lo, 0)
    total = int(cnt.sum())
    if not total:
        return
    which = np.repeat(np.arange(len(e)), cnt)
    k = lo[which] + np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    sy = (k + 0.5) / SUBSAMPLES
    ex0, ey0, ex1, ey1 = x0[which], y0[which], x1[which], y1[which]
    xs = ex0 + (sy - ey0) / (ey1 - ey0) * (ex1 - ex0)
    wind = np.where(ey0 <= sy, 1, -1)
    order = np.lexsort((xs, k))
    k, xs, wind = k[order], xs[order], wind[order]
    run = np.cumsum(wind)
    first = np.concatenate([[True], k[1:] != k[:-1]])
    group = np.cumsum(first) - 1
    local = run - (run - wind)[first][group]
    on = (k[:-1] == k[1:]) & (local[:-1] != 0)
    left = np.clip(xs[:-1][on], 0, w)
    right = np.clip(xs[1:][on], 0, w)
    rows = k[:-1][on] // SUBSAMPLES
    keep = right > left
    left, right, rows = left[keep], right[keep], rows[keep]
    if not len(rows):
        return
    weight = 1.0 / SUBSAMPLES
    # Each span adds its overlap with every pixel: as ramps that start at
    # its ends, summed by two running totals.
    stride = w + 2
    band = max(1, (1 << 21) // stride)
    flat = sheet.reshape(-1, 4)
    for r0 in range(int(rows.min()), int(rows.max()) + 1, band):
        a, b = np.searchsorted(rows, [r0, r0 + band])
        if a == b:
            continue
        rr = rows[a:b] - r0
        pts = np.concatenate([left[a:b], right[a:b]])
        sign = np.concatenate([np.full(b - a, weight), np.full(b - a, -weight)])
        idx = np.concatenate([rr, rr]) * stride + np.ceil(pts).astype(np.int64)
        n = band * stride
        slope = np.cumsum(np.bincount(idx, sign, n).reshape(band, stride), 1)
        offset = np.cumsum(np.bincount(idx, sign * pts, n).reshape(band, stride), 1)
        ks = np.arange(stride, dtype=np.float64)
        ramp = ks * slope - offset
        cover = np.minimum(ramp[:, 1:w + 1] - ramp[:, :w], 1.0) * alpha
        ys, xs_ = np.nonzero(cover > 1e-9)
        ys = ys + r0
        live = ys < h
        ys, xs_ = ys[live], xs_[live]
        if not len(ys):
            continue
        at = ys * w + xs_
        flat[at] = source_over(flat[at], np.broadcast_to(colour, (len(at), 3)),
                               cover[ys - r0, xs_])


# ==== geometry =========================================================================

def ellipse(cx, cy, rx, ry, turn):
    n = min(max(int(max(rx, ry) * 1.5), 10), 96)
    t = np.arange(n) / n * TAU
    x, y = rx * np.cos(t), ry * np.sin(t)
    s, c = math.sin(turn), math.cos(turn)
    return solid(np.stack([cx + x * c - y * s, cy + x * s + y * c], -1))


def circle(cx, cy, r):
    return ellipse(cx, cy, r, r, 0.0)


def circles(points, r):
    """circle() round each of `points`, as one batch."""
    n = min(max(int(r * 1.5), 10), 96)
    t = np.arange(n) / n * TAU
    ring = np.stack([r * np.cos(t), r * np.sin(t)], -1)
    return solid(np.asarray(points, np.float64)[:, None, :] + ring[None])


def rect(x0, y0, x1, y1):
    return solid([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def polar(n, radius):
    """A closed curve round the origin, radius(angle) at each of n steps
    (radius takes an array of angles)."""
    t = np.arange(n) / n * TAU
    r = radius(t)
    return solid(np.stack([r * np.cos(t), r * np.sin(t)], -1))


def star(points, outer, inner):
    i = np.arange(points * 2)
    r = np.where(i % 2 == 0, outer, inner)
    t = i / (points * 2.0) * TAU - PI / 2
    return solid(np.stack([r * np.cos(t), r * np.sin(t)], -1))


def petal(angle, r0, r1, width, bulge):
    n = 16
    t = np.arange(n + 1) / n
    along = r0 + t * (r1 - r0)
    half = width * 0.5 * np.maximum(np.sin(PI * t ** bulge), 0.0)
    a = np.concatenate([along, along[n - 1:0:-1]])
    hh = np.concatenate([half, -half[n - 1:0:-1]])
    s, c = math.sin(angle), math.cos(angle)
    return solid(np.stack([a * c - hh * s, a * s + hh * c], -1))


def arc_band(cx, cy, inner, outer, a0, a1):
    n = min(max(int(abs(a1 - a0) * outer * 0.8), 6), 64)
    t = a0 + (a1 - a0) * np.arange(n + 1) / n
    out = np.stack([cx + outer * np.cos(t), cy + outer * np.sin(t)], -1)
    inn = np.stack([cx + inner * np.cos(t[::-1]), cy + inner * np.sin(t[::-1])], -1)
    return solid(np.concatenate([out, inn]))


def dashed_ring(inner, outer, dashes, share):
    step = TAU / dashes
    out = []
    for i in range(dashes):
        a = i * step - PI / 2
        out.append(arc_band(0.0, 0.0, inner, outer, a - step * share / 2, a + step * share / 2))
    return out


def stroke(points, width):
    """A line of `width` along `points`, round joins and ends, as outlines
    that union under the non-zero rule."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    half = width * 0.5
    out = []
    if len(p) >= 2:
        d = np.diff(p, axis=0)
        length = np.hypot(d[:, 0], d[:, 1])
        ok = length > 1e-4
        if ok.any():
            a, b, d, length = p[:-1][ok], p[1:][ok], d[ok], length[ok]
            nrm = np.stack([-d[:, 1] / length * half, d[:, 0] / length * half], -1)
            out.append(solid(np.stack([a + nrm, b + nrm, b - nrm, a - nrm], 1)))
    # Round joins, which also round the two ends, once a corner would show.
    if half > 0.6 and len(p):
        out.append(circles(p, half))
    return out


def bezier(a, b, c, n):
    t = np.arange(n + 1) / n
    u = 1 - t
    return [(float(x), float(y)) for x, y in zip(
        u * u * a[0] + 2 * u * t * b[0] + t * t * c[0],
        u * u * a[1] + 2 * u * t * b[1] + t * t * c[1])]


def curl(x, y, heading, radius, turns, sense):
    """A spiral curling in from (x, y), setting off along `heading`, turning
    `turns` times as its radius shrinks; sense 1 clockwise, -1 against."""
    n = min(max(int(turns * radius * 2.0), 12), 120)
    side = heading + sense * PI / 2
    cx, cy = x + radius * math.cos(side), y + radius * math.sin(side)
    start = side + PI
    t = np.arange(n + 1) / n
    r = radius * (1 - 0.75 * t)
    a = start + sense * t * turns * TAU
    return [(float(px), float(py)) for px, py in zip(cx + r * np.cos(a), cy + r * np.sin(a))]


def place(contours, x, y, scale, turn):
    """Scale, turn and move every contour."""
    s, c = math.sin(turn), math.cos(turn)
    out = []
    for k in contours:
        px, py = k[..., 0] * scale, k[..., 1] * scale
        out.append(np.stack([x + px * c - py * s, y + px * s + py * c], -1))
    return out


class Track:
    """The rounded rectangle motifs are laid along, clockwise from the top left."""

    def __init__(self, x0, y0, x1, y1, radius):
        self.x0, self.y0, self.x1, self.y1, self.radius = x0, y0, x1, y1, radius

    def sides(self):
        return (max(self.x1 - self.x0 - 2 * self.radius, 0.0),
                max(self.y1 - self.y0 - 2 * self.radius, 0.0))

    def length(self):
        w, h = self.sides()
        return 2 * (w + h) + TAU * self.radius

    def at(self, s):
        """Where s along the track is, and which way it runs there."""
        w, h = self.sides()
        r = self.radius
        quarter = PI / 2 * r
        total = self.length()
        s = s % total if total > 0 else 0.0
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        pieces = (((x0 + r, y0), (1.0, 0.0), w, (x1 - r, y0 + r), -PI / 2),
                  ((x1, y0 + r), (0.0, 1.0), h, (x1 - r, y1 - r), 0.0),
                  ((x1 - r, y1), (-1.0, 0.0), w, (x0 + r, y1 - r), PI / 2),
                  ((x0, y1 - r), (0.0, -1.0), h, (x0 + r, y0 + r), PI))
        for start, d, length, centre, a0 in pieces:
            if s <= length:
                return (start[0] + d[0] * s, start[1] + d[1] * s), d
            s -= length
            if s <= quarter and r > 0:
                a = a0 + s / r
                return ((centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)),
                        (-math.sin(a), math.cos(a)))
            s -= quarter
        return (x0 + r, y0), (1.0, 0.0)

    def spread(self, spacing):
        """Positions spread evenly round the track, about `spacing` apart,
        the first half a step in so the corners come out alike."""
        n = int(max(_rnd(self.length() / max(spacing, 1.0)), 4.0))
        step = self.length() / n
        return [(i + 0.5) * step for i in range(n)]


_M64 = (1 << 64) - 1


class Rng:
    """Kooka's seeded xorshift: the same arrangement gives the same frame."""

    def __init__(self, seed, frame):
        self.s = 0x9E3779B97F4A7C15 ^ ((((seed << 32) | frame) * 0xBF58476D1CE4E5B9) & _M64)

    def next(self):
        s = self.s
        s ^= (s << 13) & _M64
        s ^= s >> 7
        s ^= (s << 17) & _M64
        self.s = s
        return (s >> 40) / float(1 << 24)

    def range(self, lo, hi):
        return lo + (hi - lo) * self.next()


# ==== shapes ===========================================================================

def _around(n, f):
    return [f(i / n * TAU - PI / 2) for i in range(n)]


def _spike(a, inner, swing):
    """A point from a ring of radius `inner` out to the unit circle, its tip
    swung `swing` radians round."""
    return solid([(inner * math.cos(a - 0.3), inner * math.sin(a - 0.3)),
                  (inner * math.cos(a + 0.3), inner * math.sin(a + 0.3)),
                  (math.cos(a + swing), math.sin(a + swing))])


def _astroid(t, squash, floor, scale=1.0):
    c, s = np.abs(np.cos(t)), np.abs(np.sin(t))
    r = (c ** (2.0 / 3.0) / squash ** (2.0 / 3.0) + s ** (2.0 / 3.0)) ** -1.5
    return np.maximum(r, floor) * scale


def heart():
    t = np.arange(64) / 64.0 * TAU
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return solid(np.stack([x / 17.0, -y / 17.0 - 0.1], -1))


def flower_shape(index):
    """Flower 1..22 about the origin, radius 1; 0 (None) is empty."""
    if index == 1:
        return [circle(0, 0, 0.35)]
    if index == 2:
        return [circle(0, 0, 1.0)]
    if index == 3:
        return [circle(0, 0, 0.3)] + _around(5, lambda a: petal(a, 0.42, 1.0, 0.62, 0.75))
    if index == 4:
        return [circle(0, 0, 0.2)] + _around(8, lambda a: petal(a, 0.3, 1.0, 0.26, 1.0))
    if index == 5:
        return [circle(0, 0, 0.6)] + dashed_ring(0.78, 0.95, 8, 0.7)
    if index == 6:
        # A disc with a hole and eight swirling points.
        return [circle(0, 0, 0.6), hole(circle(0, 0, 0.28))] + _around(8, lambda a: _spike(a, 0.55, 0.45))
    if index == 7:
        return _around(4, lambda a: petal(a, 0.12, 1.0, 0.84, 0.6))
    if index == 8:
        out = [circle(0, 0, 0.2)]
        for i in range(8):
            a = i / 8.0 * TAU
            out.append(circle(0.42 * math.cos(a), 0.42 * math.sin(a), 0.08))
            out.append(petal(a, 0.55, 1.0, 0.14, 1.0))
            out.append(petal(a + PI / 8, 0.55, 0.78, 0.12, 1.0))
        return out
    if index == 9:
        # A rose: five rounded lobes, the folds of the petals cut out of it
        # in arcs spiralling in.
        out = [polar(80, lambda t: 0.8 + 0.14 * np.abs(np.cos(2.5 * t)))]
        for r, a0, span in ((0.64, 0.3, 2.4), (0.64, 3.4, 2.2), (0.46, 1.6, 2.6),
                            (0.46, 4.6, 2.0), (0.27, 0.8, 3.4)):
            out.append(hole(arc_band(0, 0, r - 0.04, r + 0.04, a0, a0 + span)))
        return out
    if index == 10:
        return [circle(0, 0, 0.2), hole(circle(0, 0, 0.1))] + _around(
            16, lambda a: petal(a, 0.16, 1.0, 0.11, 1.0))
    if index == 11:
        return [circle(0, 0, 0.42)] + _around(8, lambda a: _spike(a, 0.52, 0.0))
    if index == 12:
        return [circle(0, 0, 0.5)] + dashed_ring(0.68, 0.95, 10, 0.62)
    if index == 13:
        # A cross with its arms curling back at the ends.
        out = [rect(-0.08, -1.0, 0.08, 1.0), rect(-1.0, -0.08, 1.0, 0.08)]
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            out.append(arc_band(0.45 * sx, 0.45 * sy, 0.2, 0.3, 0.0, TAU * 0.7))
        return out
    if index == 14:
        # Four points, the upright ones longer: an astroid.
        return [polar(96, lambda t: _astroid(t, 0.7, 0.06))]
    if index == 15:
        out = [circle(0, 0, 0.12)]
        for i in range(8):
            a = i / 8.0 * TAU - PI / 2
            out.append(petal(a, 0.0, 1.0 if i % 2 == 0 else 0.75, 0.14, 1.0))
        return out
    if index == 16:
        return [polar(96, lambda t: 0.86 + 0.14 * np.abs(np.cos(4 * t))),
                hole(circle(0, 0, 0.72)), star(5, 0.55, 0.22)]
    if index == 17:
        return _around(4, lambda a: petal(a, 0.0, 1.0, 0.8, 1.6))
    if index == 18:
        return [rect(-0.25, -1.0, 0.25, 1.0), rect(-1.0, -0.25, 1.0, 0.25)]
    if index == 19:
        return [heart()]
    if index == 20:
        return [star(5, 1.0, 0.4)]
    if index == 21:
        out = [polar(6, lambda t: np.full(len(t), 0.2))]
        for i in range(6):
            a = i / 6.0 * TAU - PI / 2
            c, s = math.cos(a), math.sin(a)
            out += stroke([(0, 0), (0.95 * c, 0.95 * s)], 0.1)
            for at, reach in ((0.45, 0.28), (0.7, 0.22)):
                bx, by = at * c, at * s
                for side in (-0.8, 0.8):
                    b = a + side
                    out += stroke([(bx, by), (bx + reach * math.cos(b), by + reach * math.sin(b))],
                                  0.08)
        return out
    if index == 22:
        return [ellipse(0, 0.42, 0.45, 0.38, 0), ellipse(-0.62, -0.12, 0.15, 0.21, -0.4),
                ellipse(-0.24, -0.55, 0.16, 0.22, -0.12), ellipse(0.24, -0.55, 0.16, 0.22, 0.12),
                ellipse(0.62, -0.12, 0.15, 0.21, 0.4)]
    return []


def profile(half):
    """A closed curve from the top point round to the bottom and back,
    half(t) the half-width at each height, t from 0 at the top to 1."""
    n = 40
    t = np.arange(n + 1) / n
    # fmax also turns the NaN a fractional power of sin(pi)'s rounding error
    # would give into the zero it should be.
    hw = np.fmax(half(t), 0.0)
    y = -1.0 + 2.0 * t
    back = np.arange(n - 1, 0, -1)
    return solid(np.concatenate([np.stack([hw, y], -1), np.stack([-hw[back], y[back]], -1)]))


def _sinp(t):
    """sin(pi t), never below 0 (so fractional powers stay real)."""
    return np.maximum(np.sin(PI * t), 0.0)


def _mirrored(half):
    pts = list(half) + [(-x, y) for x, y in reversed(half) if x != 0.0]
    return solid(pts)


def leaf_shape(index):
    """Leaf 1..23 as Photoshop's list draws it: hanging from its stalk at
    (0, -1) down to y = 1; 0 (None) is empty."""
    if index == 1:
        return [circle(0, 0, 0.95)]
    if index == 2:
        return [rect(-0.9, -0.9, 0.9, 0.9)]
    if index == 3:
        return [profile(lambda t: 0.8 * np.sin(PI * t) * np.sin(PI * t / 2))]
    if index == 4:
        return [profile(lambda t: 0.5 * _sinp(t) ** 1.4)]
    if index == 5:
        return [profile(lambda t: 0.7 * np.sin(PI * t) * np.sin(PI * (1 - t) / 2))]
    if index == 6:
        return [profile(lambda t: 0.62 * np.sin(PI * t) * np.maximum(np.sin(PI * t / 2), 0) ** 1.3)]
    if index == 7:
        return [_mirrored([(0.0, -1.0), (0.3, -0.35), (0.75, 0.55), (0.25, 0.35), (0.0, 1.0)])]
    if index == 8:
        return [circle(0, -0.25, 0.55), circle(-0.55, -0.15, 0.42), circle(0.55, -0.15, 0.42),
                circle(0, 0.55, 0.3)]
    if index == 9:
        return [_mirrored([(0.0, -1.0), (0.8, -0.3), (0.22, -0.35), (0.16, 0.2), (0.0, 1.0)])]
    if index == 10:
        half = [(0.0, -0.9)]
        for i in range(1, 11):
            t = i / 10.0
            half.append((0.9 * t * t, -0.9 + 1.4 * t))
        for i in range(1, 9):
            t = i / 8.0
            half.append((0.9 * (1 - t), 0.5 + 0.3 * math.sin(PI * t / 2)))
        half.pop()
        half.append((0.0, 0.8))
        return [_mirrored(half)]
    if index == 11:
        return [solid([(0.0, -0.8), (0.85, 0.8), (-0.85, 0.8)])]
    if index == 12:
        return [_mirrored([(0.0, -1.0), (0.85, -0.05), (0.3, 0.1), (0.0, 0.9)])]
    if index == 13:
        return [polar(96, lambda t: _astroid(t, 0.55, 0.05))]
    if index == 14:
        return [ellipse(0, 0, 0.42, 0.95, 0)]
    if index == 15:
        return [solid([(-0.35, -0.6), (0.35, -0.6), (0.95, 0.6), (-0.95, 0.6)])]
    if index == 16:
        return [rect(-0.08, -1.0, 0.08, 0.3), circle(0, 0.55, 0.45)]
    if index == 17:
        return [profile(lambda t: 0.75 * _sinp(t) ** 0.7 * (1 - 0.3 * (1 - t)))]
    if index == 18:
        return [polar(96, lambda t: _astroid(t, 1.0, 0.08, 0.95))]
    if index == 19:
        return [rect(-0.12, -1.0, 0.12, -0.2),
                solid([(-0.12, -0.35), (0.12, -0.35), (0.42, 0.25), (-0.42, 0.25)]),
                circle(0, 0.45, 0.5)]
    if index == 20:
        return [rect(-0.2, -0.9, 0.2, 0.9), rect(-0.9, -0.2, 0.9, 0.2)]
    if index == 21:
        return [heart()]
    if index == 22:
        return [rect(-0.18, -1.0, 0.18, 0.55), rect(-0.9, 0.5, 0.9, 0.9)]
    if index == 23:
        return [star(5, 1.0, 0.4)]
    return []


# ==== scale ============================================================================

# Margin per step as a share of the shorter side, and the base it starts from
# (Margin 14 puts the pinwheels' centres 7% of the way in, as Photoshop's
# preview does).
MARGIN_BASE, MARGIN_PER_STEP = 0.015, 0.004
# A motif's reach, a flower's radius and a leaf's length at 100 on their
# sliders, as shares of the shorter side.
UNIT_AT_FULL, FLOWER_AT_FULL, LEAF_AT_FULL = 0.36, 0.065, 0.1
# A vine's width as a share of a motif's reach at the default Thickness.
VINE_SHARE, THICKNESS_DEFAULT = 0.045, 29.0


class Scale:
    def __init__(self, o, width, height):
        side = float(min(width, height))
        self.unit = side * min(max(o["size"], 1), 100) / 100.0 * UNIT_AT_FULL
        self.thickness = min(max(o["thickness"], 1), 200) / THICKNESS_DEFAULT
        self.inset = side * (MARGIN_BASE + MARGIN_PER_STEP * min(o["margin"], 100))
        self.flower = side * min(max(o["flower_size"], 1), 100) / 100.0 * FLOWER_AT_FULL
        self.leaf = side * min(max(o["leaf_size"], 1), 100) / 100.0 * LEAF_AT_FULL
        self.vine = max(self.unit * VINE_SHARE * self.thickness, 0.8)


# ==== moulding =========================================================================

# How wide a ruled or moulded frame is, and how far its corner shapes reach,
# at Size 100 as a share of the shorter side.
BAND_AT_FULL, CORNER_PER_BAND = 0.25, 2.4
_CORNERS = ("square", "rounded", "inverse", "inverse_small", "dual", "dual_small")


def _depth(corner, u, v, reach):
    """How deep a point is inside the frame's outline, (u, v) its distance in
    from the nearer vertical and horizontal edges: worked out on one corner
    and folded onto the other three."""
    straight = np.minimum(u, v)
    if corner == "square":
        return straight
    if corner == "rounded":
        return np.where((u < reach) & (v < reach),
                        reach - np.hypot(reach - u, reach - v), straight)
    if corner == "inverse":
        return np.minimum(straight, np.hypot(u, v) - reach)
    if corner == "inverse_small":
        return np.minimum(straight, np.hypot(u, v) - reach * 0.55)
    # Two scallops, one off each edge, meeting in a point on the diagonal.
    r = reach * 0.75 if corner == "dual" else reach * 0.45
    c = r * 0.93
    return np.minimum(np.minimum(straight, np.hypot(u - c, v) - r), np.hypot(u, v - c) - r)


def moulding(sheet, frame, colour, scale, invert):
    """The ruled frames (36 to 41) -- one heavy line and one fine following
    the outline -- and the moulded ones (42 to 47), a bevelled band lit from
    the top left; both from a distance field, so every line and step of the
    moulding stays parallel round the most elaborate corner."""
    h, w = sheet.shape[:2]
    corner = _CORNERS[(frame - 36) % 6]
    art = frame >= 42
    reach = scale.unit / UNIT_AT_FULL * BAND_AT_FULL * CORNER_PER_BAND
    band = min(scale.unit / UNIT_AT_FULL * BAND_AT_FULL * scale.thickness, reach * 1.5)
    inset = scale.inset
    fw, fh = float(w), float(h)

    def deep(x, y):
        return _depth(corner, np.minimum(x, fw - x) - inset, np.minimum(y, fh - y) - inset, reach)

    heavy, fine = max(band * 0.28, 1.0), max(band * 0.1, 1.0)
    reach_in = int(inset + band + reach + 2.0)
    step = max(1, (1 << 20) // w)
    for r0 in range(0, h, step):
        ys, xs = np.mgrid[r0:min(h, r0 + step), 0:w]
        # Only the border can be touched; the middle is skipped outright.
        near = (ys < reach_in) | (ys >= h - reach_in) | (xs < reach_in) | (xs >= w - reach_in)
        if near.any():
            _moulding_rows(sheet, ys[near], xs[near], deep, art, band, heavy, fine, colour,
                           invert)


def _moulding_rows(sheet, ys, xs, deep, art, band, heavy, fine, colour, invert):
    w = sheet.shape[1]
    px, py = xs + 0.5, ys + 0.5
    d = deep(px, py)
    live = (d >= -1.0) & (d <= band + 1.0)
    px, py, d, ys, xs = px[live], py[live], d[live], ys[live], xs[live]
    if art:
        # The profile, outside in: a dark outer edge, a rounded bead lit by
        # its slope, a pale flat and a dark lip; inverted, it runs the other
        # way across the band.
        cover = np.clip(d + 0.5, 0, 1) * np.clip(band - d + 0.5, 0, 1)
        t = np.clip(d / band, 0, 1)
        if invert:
            t = 1 - t
        slope = np.cos(PI * (t - 0.1) / 0.6)
        gx = deep(px + 0.5, py) - deep(px - 0.5, py)
        gy = deep(px, py + 0.5) - deep(px, py - 0.5)
        g = np.maximum(np.hypot(gx, gy), 1e-3)
        # The light is at the top left, facing down and right.
        facing = -(gx / g * -0.7 + gy / g * -0.7) * slope
        tone = np.where(t < 0.1, 0.55, np.where(t < 0.7, 1.0 + 0.45 * facing,
                                                 np.where(t < 0.86, 1.2, 0.6)))
    else:
        def line(at, width):
            return np.clip(width * 0.5 - np.abs(d - at) + 0.5, 0, 1)
        # Inverted, the fine line goes outside and the heavy one in.
        outer, inner = (fine, heavy) if invert else (heavy, fine)
        cover = np.maximum(line(outer * 0.5, outer), line(outer + band * 0.3 + inner * 0.5, inner))
        tone = np.ones_like(d)
    on = cover > 0
    lit = np.clip(np.floor(np.asarray(colour, np.float64)[None, :] * tone[on, None] + 0.5), 0, 255)
    flat = sheet.reshape(-1, 4)
    at = ys[on] * w + xs[on]
    flat[at] = source_over(flat[at], lit, cover[on])


# ==== frames ===========================================================================

class _Layers:
    """What a motif frame draws, by color, so each goes down in one pass:
    vines under leaves under flowers."""

    def __init__(self, turn):
        self.vine, self.leaves, self.flowers, self.soft = [], [], [], []
        self.turn = turn

    def flower(self, shape, x, y, radius, turn):
        self.flowers += place(shape, x, y, radius, turn + self.turn)

    def soft_flower(self, shape, x, y, radius, turn):
        self.soft += place(shape, x, y, radius, turn + self.turn)

    def leaf(self, shape, x, y, length, angle):
        """A leaf with its stalk at (x, y), reaching `length` towards angle."""
        half = length * 0.5
        angle = angle + self.turn
        self.leaves += place(shape, x + half * math.cos(angle), y + half * math.sin(angle),
                             half, angle - PI / 2)

    def line(self, points, width):
        # Never finer than a pixel: thinner, a stem breaks up into dots.
        self.vine += stroke(points, max(width, 1.0))


def draw_frame(sheet, o):
    """Draw frame o["frame"] (1..47) on the clear `sheet`."""
    frame = min(max(o["frame"], 1), 47)
    h, w = sheet.shape[:2]
    scale = Scale(o, w, h)
    if frame >= 36:
        moulding(sheet, frame, o["vine"], scale, o["invert"])
        return
    flower, leaf = flower_shape(o["flower"]), leaf_shape(o["leaf"])
    rng = Rng(o["arrangement"], frame)
    layers = _Layers(min(o["angle"], 360) * PI / 180.0)
    w, h = float(w), float(h)
    inset = scale.inset

    def track(radius):
        return Track(inset, inset, w - inset, h - inset,
                     min(radius, max(min(w, h) * 0.5 - inset, 0.0)))

    centre = (w / 2, h / 2)
    u, f, l, v = scale.unit, scale.flower, scale.leaf, scale.vine
    # Invert flips the inward normal of the clockwise track outward: whatever
    # leant in towards the picture leans out towards the edge.
    flip = -1.0 if o["invert"] else 1.0

    def inward(t):
        return -t[1] * flip, t[0] * flip

    density = min(max(o["lines"], 1), 30) / 15.0
    lean = layers.turn

    def wavy(path, wave, amplitude, phase):
        """A vine waving `amplitude` either side of the track, a full wave
        every `wave` along it."""
        n = max(_rnd(path.length() / wave), 4.0)
        wave = path.length() / n
        steps = max(int(path.length() / 1.5), 1)
        pts = []
        for i in range(steps + 1):
            s = i / steps * path.length()
            p, t = path.at(s)
            nx, ny = inward(t)
            off = amplitude * math.sin(TAU * s / wave + phase)
            pts.append((p[0] + nx * off, p[1] + ny * off))
        return pts

    def crests(path, wave, phase):
        """The same wave's crests, and whether each swings out."""
        n = int(max(_rnd(path.length() / wave), 4.0))
        wave = path.length() / n
        quarter = (0.25 - phase / TAU) % 1.0
        return [((k * 0.5 + quarter) * wave, k % 2 == 0) for k in range(n * 2)]

    def sign_of(flag):
        return 1.0 if flag else -1.0

    def coin():
        return 1.0 if rng.next() < 0.5 else -1.0

    if frame == 1:
        # Happy Vine: a thin vine waving along the edge, a tendril curling
        # off every swing, flowers on the crests.
        path = track(u * 0.6)
        wave, amp = u * 1.6, u * 0.14
        layers.line(wavy(path, wave, amp, 0.0), v * 0.7)
        for s, out in crests(path, wave, 0.0):
            p, t = path.at(s)
            nx, ny = inward(t)
            sign = sign_of(out)
            x, y = p[0] + nx * amp * sign, p[1] + ny * amp * sign
            heading = math.atan2(t[1], t[0]) + sign * 0.9
            layers.line(curl(x, y, heading, u * 0.09, 1.2, -sign), v * 0.5)
            layers.flower(flower, x, y, f * rng.range(0.6, 1.2), rng.range(0.0, TAU))
            if rng.next() < 0.5:
                off = rng.range(0.25, 0.4) * u * sign
                layers.flower(flower, p[0] + nx * off, p[1] + ny * off, f * 0.45, 0.0)
    elif frame == 2:
        # Pretty Vine: a faint vine under loose clusters of flowers, some
        # solid, some see-through, of every size.
        path = track(u * 0.6)
        layers.line(wavy(path, u * 2.0, u * 0.08, 0.0), v * 0.45)
        for s in path.spread(u * 0.8):
            for _ in range(2 + int(rng.next() * 3.0)):
                p, t = path.at(s + rng.range(-0.35, 0.35) * u)
                nx, ny = inward(t)
                off = rng.range(-0.25, 0.25) * u
                x, y = p[0] + nx * off, p[1] + ny * off
                size = f * rng.range(0.4, 1.3)
                if rng.next() < 0.45:
                    layers.soft_flower(flower, x, y, size, rng.range(0.0, TAU))
                else:
                    layers.flower(flower, x, y, size, rng.range(0.0, TAU))
    elif frame == 3:
        # Smoke Signals: flowers along the edge, each sending up a wisp that
        # winds away and ends in a curl.
        path = track(u * 0.5)
        for s in path.spread(u * 1.1):
            p, t = path.at(s)
            nx, ny = inward(t)
            rise = u * rng.range(0.6, 0.9)
            top = (p[0] + nx * rise + t[0] * rise * 0.3, p[1] + ny * rise + t[1] * rise * 0.3)
            bend = (p[0] + nx * rise * 0.5 - t[0] * rise * 0.4,
                    p[1] + ny * rise * 0.5 - t[1] * rise * 0.4)
            wisp = bezier(p, bend, top, 14)
            heading = math.atan2(top[1] - bend[1], top[0] - bend[0])
            wisp += curl(top[0], top[1], heading, u * 0.08, 1.1, 1.0)[1:]
            layers.line(wisp, v * 0.5)
            layers.flower(flower, p[0], p[1], f * rng.range(1.0, 1.4), rng.range(0.0, TAU))
            layers.flower(flower, bend[0], bend[1], f * 0.45, 0.0)
    elif frame == 4:
        # Party: bunches of flowers with curled streamers flying off them.
        path = track(u * 0.5)
        for s in path.spread(u * 1.4):
            p, t = path.at(s)
            nx, ny = inward(t)
            for _ in range(3):
                a = rng.range(0.0, TAU)
                reach = u * rng.range(0.2, 0.4)
                end = (p[0] + reach * math.cos(a), p[1] + reach * math.sin(a))
                streamer = [p, end] + curl(end[0], end[1], a, u * 0.07, 1.0, coin())[1:]
                layers.line(streamer, v * 0.45)
            for _ in range(3 + int(rng.next() * 3.0)):
                ox, oy = rng.range(-0.3, 0.3) * u, rng.range(-0.25, 0.25) * u
                layers.flower(flower, p[0] + t[0] * ox + nx * oy, p[1] + t[1] * ox + ny * oy,
                              f * rng.range(0.5, 1.2), rng.range(0.0, TAU))
    elif frame == 5:
        # Big Curls: a vine swinging wide, looping at every crest, leaves down
        # its length and a flower in each loop.
        path = track(u * 0.6)
        wave, amp = u * 1.8, u * 0.2
        layers.line(wavy(path, wave, amp, 0.0), v)
        for s, out in crests(path, wave, 0.0):
            p, t = path.at(s)
            nx, ny = inward(t)
            sign = sign_of(out)
            x, y = p[0] + nx * amp * sign, p[1] + ny * amp * sign
            loop = curl(x, y, math.atan2(t[1], t[0]), u * 0.16, 0.9, -sign)
            mid = loop[len(loop) // 2]
            layers.line(loop, v * 0.8)
            layers.flower(flower, mid[0], mid[1], f, rng.range(0.0, TAU))
        for k, s in enumerate(path.spread(u * 0.45)):
            p, t = path.at(s)
            nx, ny = inward(t)
            off = amp * math.sin(TAU * s / wave)
            side = 1.0 if k % 2 == 0 else -1.0
            layers.leaf(leaf, p[0] + nx * off, p[1] + ny * off, l,
                        math.atan2(t[1], t[0]) + side * 1.0)
    elif frame == 6:
        # Tilde: little "~" strokes one after another, a flower on each end.
        path = track(u * 0.5)
        for s in path.spread(u * 1.2):
            p, t = path.at(s)
            nx, ny = inward(t)
            length, amp = u * 0.8, u * 0.1
            pts = []
            for i in range(25):
                q = i / 24.0
                a = (q - 0.5) * length
                off = amp * math.sin(TAU * q)
                pts.append((p[0] + t[0] * a + nx * off, p[1] + t[1] * a + ny * off))
            layers.line(pts, v * 0.6)
            layers.flower(flower, pts[-1][0], pts[-1][1], f, 0.0)
    elif frame == 7:
        # Romance: flowers in pairs and threes, each hanging from a curl like
        # a question mark.
        path = track(u * 0.5)
        for s in path.spread(u * 0.9):
            p, t = path.at(s + rng.range(-0.2, 0.2) * u)
            nx, ny = inward(t)
            off = rng.range(-0.2, 0.2) * u
            base = (p[0] + nx * off, p[1] + ny * off)
            reach = u * rng.range(0.25, 0.4)
            top = (base[0] + nx * reach, base[1] + ny * reach)
            stem = [base, top] + curl(top[0], top[1], math.atan2(ny, nx), u * 0.08, 0.8,
                                      coin())[1:]
            layers.line(stem, v * 0.5)
            layers.flower(flower, base[0], base[1], f * rng.range(0.9, 1.3), rng.range(0.0, TAU))
            if rng.next() < 0.6:
                a = rng.range(0.0, TAU)
                layers.flower(flower, base[0] + math.cos(a) * f * 2.0,
                              base[1] + math.sin(a) * f * 2.0, f * 0.6, 0.0)
    elif frame == 8:
        # Curly Dance: a vine of arches, looping at the foot of each, a
        # flower on every arch.
        path = track(u * 0.6)
        n = max(_rnd(path.length() / (u * 1.1)), 4.0)
        arch = path.length() / n
        height = u * 0.25
        steps = max(int(path.length() / 1.5), 1)
        pts = []
        for i in range(steps + 1):
            s = i / steps * path.length()
            p, t = path.at(s)
            nx, ny = inward(t)
            off = -height * abs(math.sin(PI * s / arch)) + height * 0.5
            pts.append((p[0] + nx * off, p[1] + ny * off))
        layers.line(pts, v * 0.6)
        for k in range(int(n)):
            p, t = path.at(k * arch)
            nx, ny = inward(t)
            foot = (p[0] + nx * height * 0.5, p[1] + ny * height * 0.5)
            layers.line(curl(foot[0], foot[1], math.atan2(t[1], t[0]), u * 0.08, 1.0, 1.0),
                        v * 0.5)
            p, t = path.at((k + 0.5) * arch)
            nx, ny = inward(t)
            layers.flower(flower, p[0] - nx * height * 0.5, p[1] - ny * height * 0.5,
                          f * rng.range(1.0, 1.4), rng.range(0.0, TAU))
    elif frame == 9:
        # Wisps: two fine strands twisting round each other, small flowers
        # where they cross.
        path = track(u * 0.6)
        wave, amp = u * 1.4, u * 0.12
        layers.line(wavy(path, wave, amp, 0.0), v * 0.4)
        layers.line(wavy(path, wave, amp, PI), v * 0.4)
        for k, s in enumerate(path.spread(wave * 0.5)):
            p, _ = path.at(s - wave * 0.25)
            layers.flower(flower, p[0], p[1], f * (0.9 if k % 2 == 0 else 0.55), 0.0)
    elif frame == 10:
        # Spring Weed: tufts of fine blades springing up off the edge, some
        # tipped with a flower.
        path = track(u * 0.4)
        for s in path.spread(u * 0.6):
            p, t = path.at(s + rng.range(-0.15, 0.15) * u)
            nx, ny = inward(t)
            blades = 3 + int(rng.next() * 3.0)
            for b in range(blades):
                tilt = (b / (blades - 1.0) - 0.5) * 1.4 + rng.range(-0.2, 0.2)
                c, sn = math.cos(tilt), math.sin(tilt)
                d = (nx * c - ny * sn, nx * sn + ny * c)
                reach = u * rng.range(0.2, 0.42)
                tip = (p[0] + d[0] * reach, p[1] + d[1] * reach)
                sg = math.copysign(1.0, tilt)
                bend = (p[0] + d[0] * reach * 0.5 + t[0] * reach * 0.2 * sg,
                        p[1] + d[1] * reach * 0.5 + t[1] * reach * 0.2 * sg)
                layers.line(bezier(p, bend, tip, 10), v * 0.45)
                if rng.next() < 0.45:
                    layers.flower(flower, tip[0], tip[1], f * rng.range(0.5, 0.9),
                                  rng.range(0.0, TAU))
    elif frame == 11:
        # Eyelash: curved sprays of leaves, each rooted in a knot of flowers,
        # curving one way and then the other.
        path = track(u * 0.5)
        for k, s in enumerate(path.spread(u * 1.3)):
            p, t = path.at(s)
            nx, ny = inward(t)
            sign = 1.0 if k % 2 == 0 else -1.0
            reach = u * 0.55
            tip = (p[0] + t[0] * reach * sign + nx * reach * 0.2,
                   p[1] + t[1] * reach * sign + ny * reach * 0.2)
            ctrl = (p[0] + t[0] * reach * 0.5 * sign + nx * reach * 0.55,
                    p[1] + t[1] * reach * 0.5 * sign + ny * reach * 0.55)
            stem = bezier(p, ctrl, tip, 16)
            layers.line(stem, v * 0.5)
            for i in range(3, len(stem), 3):
                a, b = stem[i - 1], stem[i]
                layers.leaf(leaf, b[0], b[1], l * 0.8,
                            math.atan2(b[1] - a[1], b[0] - a[0]) - sign * 0.8)
            for _ in range(3):
                ox, oy = rng.range(-0.12, 0.12) * u, rng.range(-0.1, 0.1) * u
                layers.flower(flower, p[0] + ox, p[1] + oy, f * rng.range(0.7, 1.1),
                              rng.range(0.0, TAU))
    elif frame == 12:
        # Magic Smoke: short wavering stems, each with a pair of leaves and a
        # few flowers at its top.
        path = track(u * 0.5)
        for s in path.spread(u * 1.1):
            p, t = path.at(s)
            nx, ny = inward(t)
            reach = u * 0.4
            tilt = rng.range(-0.5, 0.5)
            top = (p[0] + nx * reach + t[0] * reach * tilt, p[1] + ny * reach + t[1] * reach * tilt)
            ctrl = (p[0] + nx * reach * 0.5 - t[0] * reach * 0.3,
                    p[1] + ny * reach * 0.5 - t[1] * reach * 0.3)
            stem = bezier(p, ctrl, top, 12)
            layers.line(stem, v * 0.5)
            mid = stem[len(stem) // 2]
            up = math.atan2(ny, nx)
            layers.leaf(leaf, mid[0], mid[1], l, up + 0.9)
            layers.leaf(leaf, mid[0], mid[1], l, up - 0.9)
            for _ in range(2 + int(rng.next() * 2.0)):
                ox, oy = rng.range(-0.1, 0.1) * u, rng.range(-0.1, 0.1) * u
                layers.flower(flower, top[0] + ox, top[1] + oy, f * rng.range(0.7, 1.1),
                              rng.range(0.0, TAU))
    elif frame == 13:
        # Curly Vine: a leafy vine, thick with leaves both sides and curling
        # tendrils, a flower here and there.
        path = track(u * 0.6)
        wave, amp = u * 1.5, u * 0.1
        layers.line(wavy(path, wave, amp, 0.0), v)
        for k, s in enumerate(path.spread(l * 0.4)):
            p, t = path.at(s)
            nx, ny = inward(t)
            off = amp * math.sin(TAU * s / wave)
            x, y = p[0] + nx * off, p[1] + ny * off
            along = math.atan2(t[1], t[0])
            side = 1.0 if k % 2 == 0 else -1.0
            layers.leaf(leaf, x, y, l * rng.range(0.7, 1.2), along + side * rng.range(0.5, 1.3))
            if k % 5 == 2:
                layers.line(curl(x, y, along - side * 1.0, u * 0.08, 1.2, side), v * 0.5)
        for s in path.spread(u * 1.2):
            p, _ = path.at(s + rng.range(-0.3, 0.3) * u)
            layers.flower(flower, p[0], p[1], f * rng.range(0.7, 1.0), rng.range(0.0, TAU))
    elif frame == 14:
        # Chainmail: two rows of flowers along the band's edges, laced
        # together by parallel slanting lines.
        path = track(u * 0.3)
        half = u * 0.22
        slant = 0.5 + lean
        for s in path.spread(u * 0.2 / density):
            p, t = path.at(s)
            nx, ny = inward(t)
            shift = half * min(max(math.tan(slant), -3.0), 3.0)
            a = (p[0] - nx * half - t[0] * shift, p[1] - ny * half - t[1] * shift)
            b = (p[0] + nx * half + t[0] * shift, p[1] + ny * half + t[1] * shift)
            layers.line([a, b], v * 0.35)
        for side in (-1.0, 1.0):
            for s in path.spread(u * 0.3):
                p, t = path.at(s + rng.range(-0.08, 0.08) * u)
                nx, ny = inward(t)
                off = side * half * 1.1
                layers.flower(flower, p[0] + nx * off, p[1] + ny * off, f * rng.range(0.55, 1.0),
                              rng.range(0.0, TAU))
    elif frame == 15:
        # Fun Event: bunches of flowers on curling stems, a pair of small ones
        # between.
        path = track(u * 0.5)
        for s in path.spread(u * 1.2):
            p, t = path.at(s)
            nx, ny = inward(t)
            for k in range(3):
                a = math.atan2(ny, nx) + (k - 1.0) * 0.8 + rng.range(-0.2, 0.2)
                reach = u * rng.range(0.2, 0.35)
                tip = (p[0] + reach * math.cos(a), p[1] + reach * math.sin(a))
                stem = [p, tip] + curl(tip[0], tip[1], a, u * 0.05, 0.9,
                                       1.0 if k % 2 == 0 else -1.0)[1:]
                layers.line(stem, v * 0.45)
                layers.flower(flower, tip[0], tip[1], f * rng.range(0.7, 1.0), rng.range(0.0, TAU))
            layers.flower(flower, p[0], p[1], f * 1.2, 0.0)
            q, _ = path.at(s + u * 0.6)
            layers.flower(flower, q[0] - f, q[1], f * 0.5, 0.0)
            layers.flower(flower, q[0] + f, q[1] + f * 0.5, f * 0.5, 0.0)
    elif frame == 16:
        # Bush: clumps of leaves bursting up off a grassy line, flowers in
        # among them, big clumps and small by turns.
        path = track(u * 0.5)
        steps = max(int(path.length() / 2.0), 1)
        layers.line([path.at(i / steps * path.length())[0] for i in range(steps + 1)], v * 0.5)
        for k, s in enumerate(path.spread(u * 1.0)):
            p, t = path.at(s)
            nx, ny = inward(t)
            big = k % 2 == 0
            up = math.atan2(ny, nx)
            for _ in range(18 if big else 9):
                a = up + rng.range(-1.4, 1.4)
                ox = rng.range(-0.3, 0.3) * u
                length = l * rng.range(1.0, 2.2 if big else 1.5)
                layers.leaf(leaf, p[0] + t[0] * ox, p[1] + t[1] * ox, length, a)
            for _ in range(3 if big else 1):
                ox, oy = rng.range(-0.2, 0.2) * u, rng.range(0.05, 0.3) * u
                layers.flower(flower, p[0] + t[0] * ox + nx * oy, p[1] + t[1] * ox + ny * oy,
                              f * rng.range(0.6, 1.0), rng.range(0.0, TAU))
    elif frame == 17:
        # Simple Lace: a fine waving line, flowers on the crests, large and
        # small by turns.
        path = track(u * 0.6)
        wave, amp = u * 1.4, u * 0.1
        layers.line(wavy(path, wave, amp, 0.0), v * 0.45)
        for k, (s, out) in enumerate(crests(path, wave, 0.0)):
            p, t = path.at(s)
            nx, ny = inward(t)
            sign = sign_of(out)
            layers.flower(flower, p[0] + nx * amp * sign, p[1] + ny * amp * sign,
                          f * (1.2 if k % 2 == 0 else 0.8), 0.0)
    elif frame == 18:
        # Pulse: a vine waving in and out, a flower at every outer swing and a
        # pair of leaves where it crosses the line.
        path = track(u * 0.6)
        n = max(_rnd(path.length() / (u * 1.8)), 4.0)
        wave = path.length() / n
        amplitude = u * 0.28
        steps = max(int(path.length() / 2.0), 1)
        pts = []
        for i in range(steps + 1):
            s = i / steps * path.length()
            p, t = path.at(s)
            nx, ny = inward(t)
            off = amplitude * math.sin(TAU * s / wave)
            pts.append((p[0] + nx * off, p[1] + ny * off))
        layers.line(pts, v)
        for k in range(int(n)):
            p, t = path.at((k + 0.75) * wave)
            nx, ny = inward(t)
            layers.flower(flower, p[0] - nx * amplitude, p[1] - ny * amplitude, f,
                          rng.range(0.0, TAU))
            p, t = path.at((k + 0.5) * wave)
            along = math.atan2(t[1], t[0])
            layers.leaf(leaf, p[0], p[1], l, along - 0.9)
            layers.leaf(leaf, p[0], p[1], l, along + PI + 0.9)
    elif frame == 19:
        # Root: a garland -- a gently swaying vine thick with leaves on both
        # sides, flowers tucked in among them.
        path = track(u * 0.8)
        steps = max(int(path.length() / 2.0), 1)

        def sway(s):
            return u * 0.12 * math.sin(s / (u * 1.4))
        pts = []
        for i in range(steps + 1):
            s = i / steps * path.length()
            p, t = path.at(s)
            nx, ny = inward(t)
            pts.append((p[0] + nx * sway(s), p[1] + ny * sway(s)))
        layers.line(pts, v * 1.3)
        for k, s in enumerate(path.spread(l * 0.45)):
            p, t = path.at(s)
            nx, ny = inward(t)
            x, y = p[0] + nx * sway(s), p[1] + ny * sway(s)
            along = math.atan2(t[1], t[0])
            side = 1.0 if k % 2 == 0 else -1.0
            spread = rng.range(0.6, 1.1)
            layers.leaf(leaf, x, y, l * rng.range(0.8, 1.15), along + side * spread)
        for s in path.spread(u * 1.1):
            p, t = path.at(s + rng.range(-0.3, 0.3) * u)
            nx, ny = inward(t)
            off = rng.range(-0.25, 0.25) * u
            layers.flower(flower, p[0] + nx * off, p[1] + ny * off, f * rng.range(0.7, 1.1),
                          rng.range(0.0, TAU))
    elif frame == 20:
        # Snakes: curling S-shaped stems, each with a flower at its head.
        path = track(u * 0.5)
        for s in path.spread(u * 1.5):
            p, t = path.at(s)
            nx, ny = inward(t)
            reach = u * rng.range(0.45, 0.7)
            a = math.atan2(t[1], t[0]) + rng.range(-0.8, 0.8)
            dx, dy = math.cos(a) * reach, math.sin(a) * reach
            head, tail = (p[0] + dx, p[1] + dy), (p[0] - dx, p[1] - dy)
            bend = reach * 0.9
            first = bezier(tail, (p[0] - dx * 0.5 + nx * bend, p[1] - dy * 0.5 + ny * bend), p, 12)
            second = bezier(p, (p[0] + dx * 0.5 - nx * bend, p[1] + dy * 0.5 - ny * bend), head, 12)
            layers.line(first + second[1:], v)
            layers.flower(flower, head[0], head[1], f * rng.range(0.9, 1.6), rng.range(0.0, TAU))
    elif frame == 21:
        # Mustache: flowers joined dot to dot by fine lines, zigzagging
        # across the line.
        path = track(u * 0.5)
        chain = []
        for k, s in enumerate(path.spread(u * 0.7)):
            p, t = path.at(s)
            nx, ny = inward(t)
            off = (1.0 if k % 2 == 0 else -1.0) * u * rng.range(0.15, 0.4)
            chain.append((p[0] + nx * off, p[1] + ny * off))
        if chain:
            chain.append(chain[0])
        layers.line(chain, max(v * 0.6, 0.8))
        for x, y in chain[:max(len(chain) - 1, 0)]:
            layers.flower(flower, x, y, f * rng.range(0.5, 1.1), rng.range(0.0, TAU))
    elif frame == 22:
        # Circle Sprinkle: flowers of every size scattered along the band.
        path = track(u * 0.5)
        for s in path.spread(u * 0.45):
            p, t = path.at(s + rng.range(-0.2, 0.2) * u)
            nx, ny = inward(t)
            off = rng.range(-0.45, 0.45) * u
            layers.flower(flower, p[0] + nx * off, p[1] + ny * off, f * rng.range(0.4, 1.7),
                          rng.range(0.0, TAU))
    elif frame == 23:
        # Aligned Flowers: a row, large and small by turns, a small one set
        # off to the side now and then.
        path = track(u * 0.3)
        for k, s in enumerate(path.spread(u * 0.9)):
            p, t = path.at(s)
            layers.flower(flower, p[0], p[1], f * (1.3 if k % 2 == 0 else 0.8), 0.0)
            if k % 3 == 0:
                nx, ny = inward(t)
                off = u * 0.35 * coin()
                layers.flower(flower, p[0] + nx * off, p[1] + ny * off, f * 0.45, 0.0)
    elif frame == 24:
        # Little Flowers: single flowers each with a pair of leaves, soft.
        path = track(u * 0.5)
        for s in path.spread(u * 1.3):
            p, t = path.at(s)
            along = math.atan2(t[1], t[0])
            # Long enough to reach out from under the flower.
            layers.leaf(leaf, p[0], p[1], l * 1.8, along + PI / 2 + 0.7)
            layers.leaf(leaf, p[0], p[1], l * 1.8, along + PI / 2 - 0.7)
            layers.flower(flower, p[0], p[1], f * rng.range(1.0, 1.5), rng.range(0.0, TAU))
    elif frame == 25:
        # Snowflakes: a drift of small flowers and leaves along the band.
        path = track(u * 0.5)
        for s in path.spread(u * 0.22):
            p, t = path.at(s)
            nx, ny = inward(t)
            off = rng.range(-0.5, 0.5) * u
            x, y = p[0] + nx * off, p[1] + ny * off
            if rng.next() < 0.3:
                layers.leaf(leaf, x, y, l * rng.range(0.5, 0.8), rng.range(0.0, TAU))
            else:
                layers.flower(flower, x, y, f * rng.range(0.35, 0.8), rng.range(0.0, TAU))
    elif frame == 26:
        # Flurry: tight clusters of flowers with a leaf or two.
        path = track(u * 0.5)
        for s in path.spread(u * 1.0):
            p, t = path.at(s)
            nx, ny = inward(t)
            for _ in range(2):
                a = rng.range(0.0, TAU)
                layers.leaf(leaf, p[0], p[1], l * rng.range(0.6, 0.9), a)
            for _ in range(3 + int(rng.next() * 3.0)):
                ox, oy = rng.range(-0.3, 0.3) * u, rng.range(-0.25, 0.25) * u
                layers.flower(flower, p[0] + t[0] * ox + nx * oy, p[1] + t[1] * ox + ny * oy,
                              f * rng.range(0.6, 1.1), rng.range(0.0, TAU))
    elif frame == 27:
        # Check Marks: ticks in the vine color, a flower on the end of every
        # other one.
        path = track(u * 0.5)
        for k, s in enumerate(path.spread(u * 0.8)):
            p, t = path.at(s)
            nx, ny = inward(t)
            arm = u * 0.35
            a = (p[0] - t[0] * arm * 0.3 - nx * arm * 0.6, p[1] - t[1] * arm * 0.3 - ny * arm * 0.6)
            b = (p[0] + nx * arm * 0.4, p[1] + ny * arm * 0.4)
            c = (p[0] + t[0] * arm - nx * arm * 0.9, p[1] + t[1] * arm - ny * arm * 0.9)
            layers.line([a, b, c], max(v * 0.7, 0.8))
            if k % 2 == 1:
                layers.flower(flower, c[0], c[1], f * 1.2, 0.0)
    elif frame == 28:
        # Focused Lines: short lines all aimed at the middle of the picture,
        # each ending in a flower on the inside.
        path = track(u * 0.5)
        for k, s in enumerate(path.spread(u * 0.7 / density)):
            p, _ = path.at(s)
            dx, dy = (centre[0] - p[0]) * flip, (centre[1] - p[1]) * flip
            d = max(math.hypot(dx, dy), 1.0)
            length = u * (0.55 if k % 2 == 0 else 0.3) * rng.range(0.8, 1.2)
            end = (p[0] + dx / d * length, p[1] + dy / d * length)
            start = (p[0] - dx / d * length * 0.3, p[1] - dy / d * length * 0.3)
            layers.line([start, end], max(v * 0.6, 0.8))
            layers.flower(flower, end[0], end[1], f * (1.2 if k % 2 == 0 else 0.8), 0.0)
    elif frame == 29:
        # Focused Parallel Lines: the same, every line on one slant (Angle
        # leans it on from 45 degrees).
        path = track(u * 0.5)
        slant = PI / 4 + lean
        dx, dy = math.cos(slant) * flip, math.sin(slant) * flip
        for k, s in enumerate(path.spread(u * 0.75 / density)):
            p, _ = path.at(s)
            length = u * (0.5 if k % 2 == 0 else 0.3)
            start = (p[0] - dx * length * 0.5, p[1] - dy * length * 0.5)
            end = (p[0] + dx * length * 0.5, p[1] + dy * length * 0.5)
            layers.line([start, end], max(v * 0.6, 0.8))
            layers.flower(flower, start[0], start[1], f * (1.4 if k % 2 == 0 else 0.8), 0.0)
    elif frame == 30:
        # Focused Vibration: lines standing straight off the edge, their
        # lengths jumping about like a trace, each tipped with a flower.
        path = track(0.0)
        for s in path.spread(u * 0.45 / density):
            p, t = path.at(s)
            nx, ny = inward(t)
            length = u * rng.range(0.1, 0.6)
            end = (p[0] + nx * length, p[1] + ny * length)
            layers.line([p, end], max(v * 0.5, 0.8))
            layers.flower(flower, end[0], end[1], f * 0.8, 0.0)
    elif frame == 31:
        # Zen Garden: each flower set in rings, like a stone in raked sand.
        path = track(u * 0.5)
        for s in path.spread(u * 1.1):
            p, t = path.at(s)
            nx, ny = inward(t)
            off = rng.range(-0.2, 0.2) * u
            x, y = p[0] + nx * off, p[1] + ny * off
            for ring in range(1, 4):
                r = u * 0.16 * ring * rng.range(0.9, 1.1)
                n = min(max(int(r * 0.8), 16), 96)
                a = np.arange(n + 1) / n * TAU
                layers.line(np.stack([x + r * np.cos(a), y + r * np.sin(a)], -1),
                            max(v * 0.4, 0.7))
            layers.flower(flower, x, y, f, rng.range(0.0, TAU))
    elif frame == 32:
        # Spikes: small flowers along the line, a burst of fine spikes round each.
        path = track(u * 0.5)
        for s in path.spread(u * 1.0):
            p, _ = path.at(s)
            turn = rng.range(0.0, TAU)
            for i in range(6):
                a = turn + i / 6.0 * TAU
                reach = u * (0.4 if i % 2 == 0 else 0.25)
                layers.line([p, (p[0] + reach * math.cos(a), p[1] + reach * math.sin(a))],
                            max(v * 0.35, 0.7))
            layers.flower(flower, p[0], p[1], f * 0.8, 0.0)
    elif frame == 33:
        # Anemone: flowers crowded all the way round in clumps of two or
        # three, jostling in and out of line, on a well-rounded track.
        path = track(u * 1.2)
        for s in path.spread(f * 3.4):
            p, t = path.at(s + rng.range(-0.4, 0.4) * f)
            nx, ny = inward(t)
            off = rng.range(-0.9, 0.9) * f
            for _ in range(2 + int(rng.next() * 2.0)):
                ax, ay = rng.range(-0.8, 0.8) * f, rng.range(-0.6, 0.6) * f
                layers.flower(flower, p[0] + nx * (off + ay) + t[0] * ax,
                              p[1] + ny * (off + ay) + t[1] * ax, f * rng.range(1.1, 1.45),
                              rng.range(0.0, TAU))
    elif frame == 34:
        _pinwheel(layers, rng, track(0.0), u, f, v, flower)
    else:
        # Spacing: flowers in a repeating run of large, medium and small.
        path = track(u * 0.3)
        for k, s in enumerate(path.spread(u * 0.6)):
            p, _ = path.at(s)
            layers.flower(flower, p[0], p[1], f * (1.3, 0.85, 0.45)[k % 3], 0.0)

    vine, flowers, leaves = uses(frame)
    if vine:
        fill(sheet, layers.vine, o["vine"], 1.0)
    if leaves and o["leaf"] != 0:
        # Little Flowers and Snowflakes are soft, as Photoshop draws them.
        fill(sheet, layers.leaves, o["leaf_color"], 0.7 if frame in (24, 25) else 1.0)
    if flowers and o["flower"] != 0:
        fill(sheet, layers.soft, o["flower_color"], 0.45)
        fill(sheet, layers.flowers, o["flower_color"], 0.8 if frame in (24, 25) else 1.0)


def _pinwheel(layers, rng, path, u, f, v, flower):
    """Pinwheel: sprays of curved stems tipped with flowers, swirling about
    a centre; symmetrical, a tight bunch at each corner and a double spray
    in the middle of the long sides."""

    def spray(x, y, reach, arms, turn, swirl):
        for i in range(arms):
            a = turn + i / float(arms) * TAU
            tip = (x + reach * math.cos(a), y + reach * math.sin(a))
            bend = a + swirl
            ctrl = (x + reach * 0.55 * math.cos(bend), y + reach * 0.55 * math.sin(bend))
            layers.line(bezier((x, y), ctrl, tip, 12), v * 0.8)
            layers.flower(flower, tip[0], tip[1], f, 0.0)

    for x, y in ((path.x0, path.y0), (path.x1, path.y0), (path.x0, path.y1), (path.x1, path.y1)):
        for k in range(9):
            a = k / 9.0 * TAU + rng.range(-0.2, 0.2)
            r = u * 0.45 * rng.range(0.4, 1.0)
            tip = (x + r * math.cos(a), y + r * math.sin(a))
            layers.line([(x, y), tip], v * 0.8)
            layers.flower(flower, tip[0], tip[1], f * 1.2, 0.0)
    # Along each side, mirrored about its middle: a long side carries a
    # corner, a large spray, a small one, the double in the middle and back
    # again; the short sides have half as many gaps, whatever their length.
    long_is_width = path.x1 - path.x0 >= path.y1 - path.y0
    long_length = max(path.x1 - path.x0, path.y1 - path.y0)
    gaps_long = max(int(_rnd(long_length / (u * 3.0))) if u > 0 else 0, 2)

    def side(a, b, long):
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = gaps_long if long else max(gaps_long // 2, 1)
        for i in range(1, n):
            t = i / float(n)
            x, y = a[0] + dx * t, a[1] + dy * t
            middle = long and n % 2 == 0 and i == n // 2
            small = not middle and long and i % 2 == 0
            reach = u * (0.55 if small else 1.0)
            turn = rng.range(0.0, TAU)
            # One way on the left half, the other on the right: a mirror image.
            swirl = 1.1 if t < 0.5 else -1.1
            spray(x, y, reach, 6, turn, swirl)
            if middle:
                spray(x, y, reach, 6, turn + PI / 6, -swirl)

    side((path.x0, path.y0), (path.x1, path.y0), long_is_width)
    side((path.x0, path.y1), (path.x1, path.y1), long_is_width)
    side((path.x0, path.y0), (path.x0, path.y1), not long_is_width)
    side((path.x1, path.y0), (path.x1, path.y1), not long_is_width)


def picture_frame(src, o):
    """src (H, W, 4) uint8 with the frame `o` (a dict as PictureFrame's
    properties, the frame/flower/leaf as numbers) drawn over it."""
    h, w = src.shape[:2]
    if not h or not w:
        return src.copy()
    sheet = np.zeros((h, w, 4), np.uint8)
    draw_frame(sheet, o)
    ys, xs = np.nonzero(sheet[..., 3])
    out = src.copy()
    if not len(ys):
        return out
    # Fade: how far along its side from the nearer corner to the middle a
    # pixel is, 0 at a corner and 1 half way, measured along the nearer edge.
    fade = min(o["fade"], 100) / 100.0
    px, py = xs + 0.5, ys + 0.5
    ex, ey = np.minimum(px, w - px), np.minimum(py, h - py)
    along = np.where(ey < ex, ex / (w / 2.0), ey / (h / 2.0))
    alpha = 1.0 - fade * np.clip(along, 0.0, 1.0)
    top = sheet[ys, xs]
    flat = out.reshape(-1, 4)
    at = ys * w + xs
    flat[at] = source_over(flat[at], top[:, :3], alpha, top[:, 3].astype(np.float64))
    return out


def _labels(keys, names, none=False):
    out = dict(zip(keys[1:] if none else keys, names))
    if none:
        out["none"] = "None"
    return out


class PictureFrame(_Filter):
    """A decorative border -- vines, flowers and leaves round the edge, or a
    ruled or moulded frame -- drawn over the picture."""
    name, label, category = "picture_frame", "Picture Frame", "Render"
    moves_alpha = True            # the frame is opaque where it lies

    def declare(self):
        return PropertyCollection([
            ChoiceProperty("frame", "pinwheel", list(FRAMES), labels=_labels(FRAMES, FRAME_NAMES)),
            ColorProperty("vine_color", (60, 84, 34, 255), label="Vines"),
            IntProperty("margin", 14, 0, 100),
            IntProperty("size", 20, 1, 100),
            IntProperty("arrangement", 1, 1, 20),
            ChoiceProperty("flower", "circle", list(FLOWERS),
                           labels=_labels(FLOWERS, FLOWER_NAMES, True)),
            ColorProperty("flower_color", (0, 88, 218, 255)),
            IntProperty("flower_size", 20, 1, 100),
            ChoiceProperty("leaf", "leaf_1", list(LEAVES), labels=_labels(LEAVES, LEAF_NAMES, True)),
            ColorProperty("leaf_color", (104, 170, 60, 255)),
            IntProperty("leaf_size", 20, 1, 100),
            IntProperty("lines", 15, 1, 30, label="Number of Lines"),
            IntProperty("thickness", 29, 1, 200),
            IntProperty("angle", 10, 0, 360),
            IntProperty("fade", 0, 0, 100),
            BoolProperty("invert", False)])

    def compute(self, t, src):
        o = {k: t[k] for k in ("margin", "size", "arrangement", "flower_size", "leaf_size",
                               "lines", "thickness", "angle", "fade", "invert")}
        o["frame"] = FRAMES.index(t["frame"]) + 1 if t["frame"] in FRAMES else 34
        o["flower"] = FLOWERS.index(t["flower"]) if t["flower"] in FLOWERS else 2
        o["leaf"] = LEAVES.index(t["leaf"]) if t["leaf"] in LEAVES else 4
        for k, key in (("vine", "vine_color"), ("flower_color", "flower_color"),
                       ("leaf_color", "leaf_color")):
            o[k] = np.array(t[key][:3], np.float64)
        return picture_frame(src, o)


FRAME = (PictureFrame,)
