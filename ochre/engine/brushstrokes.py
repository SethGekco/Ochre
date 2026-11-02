# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop's Filter Gallery > Brush Strokes family, all eight: Accented
Edges, Angled Strokes, Crosshatch, Dark Strokes, Ink Outlines, Spatter,
Sprayed Strokes and Sumi-e.

The common move is LAYING STROKES: each pixel is dragged a little along a
direction by streaked noise, then the picture is median-filtered along
that direction, by brightness, so it settles into runs of one color about
a stroke long (lay_strokes). The Artistic family reuses it.

Algorithms, constants and slider ranges are ported from Kooka Pictura's
photorust engine (crates/pictura-filters/src/photorust/brush_strokes,
GPL-3.0), whose sliders follow CS6's. Alpha is never touched.
"""

import numpy as np

from . import fields as F
from .filters import _Filter
from .props import ChoiceProperty, IntProperty, PropertyCollection
from .sketch import _direction, _seed


def _props(*items):
    return PropertyCollection(list(items))


def _rgb(src):
    return src[..., :3].astype(np.float32)


def _u8(rgb):
    """Rounded to bytes, as Kooka's pixmap is between stages."""
    return np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8)


def lay_strokes(rgb, angle, length, salt, seed=0):
    """Drag each pixel along `angle` by streaked noise, then settle the
    picture with a brightness median along the same line: strokes about
    `length` pixels long. rgb is uint8 (H, W, 3); so is the result."""
    h, w = rgb.shape[:2]
    direction = F.along(angle)
    drag = F.streaked_noise(h, w, length + 3.0, salt, direction, seed)
    drag = F.unit_spread(F.blur(drag, 1.6))
    offset = np.clip(drag, -2, 2) * length * 0.15 * 0.5
    xs, ys = F.grid(h, w)
    dragged = F.displace_nearest(rgb, xs + direction[0] * offset, ys + direction[1] * offset)
    return F.median_along(dragged, direction, max(length * 0.7, 1.0))


def bristle_lines(h, w, angle, length, salt, width=0.5, seed=0):
    lines = F.streaked_noise(h, w, length * 1.5, salt, F.along(angle), seed)
    if width > 0:
        lines = F.blur(lines, width)
    return F.unit_spread(lines)


class _Strokes(_Filter):
    category = "Brush Strokes"


class AccentedEdges(_Strokes):
    """The edges traced in chalk (high Edge Brightness) or ink (low) over a
    lightly softened picture."""
    name, label = "accented_edges", "Accented Edges"

    def declare(self):
        return _props(IntProperty("edge_width", 2, 1, 14), IntProperty("edge_brightness", 38, 0, 50),
                      IntProperty("smoothness", 5, 1, 15))

    def compute(self, t, src):
        width = float(t["edge_width"])
        smooth = t["smoothness"] * 0.45
        base = _u8(F.blur_rgb(src, smooth * 0.35)).astype(np.float32)
        edges = F.sobel(F.blur(F.luma(src), smooth))
        lower = max(1 - (width - 1) * 0.04, 0.3)
        lo, full = 10.0 * lower, 45.0 * lower
        edges = F.smooth((edges - lo) / (full - lo))
        edges = F.blur(F.widest_nearby(edges, F.rnd(width * 0.3)), 0.6)
        tint = (t["edge_brightness"] - 25.0) / 25.0
        accent = (edges * abs(tint))[..., None]
        gray = F.luma(base)[..., None]
        if tint >= 0:
            v = gray + (base - gray) * (1 + 0.9 * accent)
            v = v * (1 + 1.6 * accent) + 25.0 * accent
        else:
            v = base * (1 - accent)
        return F.put_rgb(src, v)


class AngledStrokes(_Strokes):
    """Lights painted in strokes one way, darks the other."""
    name, label = "angled_strokes", "Angled Strokes"

    def declare(self):
        return _props(IntProperty("balance", 50, 0, 100, label="Direction Balance"),
                      IntProperty("stroke_length", 15, 3, 50), IntProperty("sharpness", 3, 0, 10))

    def compute(self, t, src):
        h, w = src.shape[:2]
        length = float(t["stroke_length"])
        rgb = src[..., :3]
        rising = lay_strokes(rgb, 45.0, length, 21).astype(np.float32)
        falling = lay_strokes(rgb, -45.0, length, 22).astype(np.float32)
        up_lines = bristle_lines(h, w, 45.0, length, 23)
        down_lines = bristle_lines(h, w, -45.0, length, 24)
        judged = F.blur(F.luma(src) / 255.0, 2.0)
        fall = F.smooth((judged - t["balance"] / 100.0) / 0.08 * 0.5 + 0.5)
        n = up_lines * (1 - fall) + down_lines * fall
        dark = 1 - judged
        n = n + dark * np.abs(n) * 0.5
        line = n * 7.0 * dark * dark
        out = falling * fall[..., None] + rising * (1 - fall[..., None]) + line[..., None]
        out = _u8(out)
        return F.put_rgb(src, _sharpen_luma(out, src[..., 3], t["sharpness"] * 0.15))


def _sharpen_luma(rgb, alpha, gain, scale=1.2, start=6.0, most=40.0):
    """Lift each pixel by how far its brightness stands off a blur of it,
    past a threshold: the Sharpness slider of the stroke filters."""
    if gain <= 0:
        return rgb.astype(np.float32)
    soft = F.blur_rgb(np.dstack([rgb, alpha]), scale)
    soft = _u8(soft).astype(np.float32)
    s = rgb.astype(np.float32)
    diff = F.luma(s) - F.luma(soft)
    over = np.maximum(np.abs(diff) - start, 0) * np.sign(diff)
    lift = np.clip(over * gain, -most, most)
    return s + lift[..., None]


def _hash64(a, b, salt):
    """Kooka's 64-bit stroke hash: 0..1 from two whole numbers."""
    a = np.atleast_1d(np.asarray(a, np.int64)).astype(np.uint64)
    b = np.atleast_1d(np.asarray(b, np.int64)).astype(np.uint64)
    v = (a * np.uint64(0x9E3779B97F4A7C15)) ^ (b * np.uint64(0xC2B2AE3D27D4EB4F)) \
        ^ np.uint64((salt * 0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF)
    v ^= v >> np.uint64(31)
    v = v * np.uint64(0xBF58476D1CE4E5B9)
    v ^= v >> np.uint64(29)
    return ((v >> np.uint64(11)).astype(np.float64) / float(1 << 53)).astype(np.float32)


def pen_lines(h, w, rising, length, salt):
    """Short straight pencil dashes on the diagonals, -1 (shade) or +1 (pale)."""
    span = max(length * 1.6, 3.0)
    xs, ys = F.grid(h, w)
    diagonal, run = (xs + ys, xs - ys) if rising else (xs - ys, xs + ys)
    offset = _hash64(diagonal, -1, salt) * span
    segment = np.floor((run + offset) / span).astype(np.int64)
    pick = _hash64(diagonal, segment, salt)
    side = np.where(_hash64(diagonal, segment + 7919, salt) < 0.5, -1.0, 1.0)
    return np.where(pick < 0.22, side * (0.6 + 0.4 * pick / 0.22), 0.0).astype(np.float32)


class Crosshatch(_Strokes):
    """Pencil hatching laid over the picture where it has detail, crossing
    both diagonals; Strength repeats the pass."""
    name, label = "crosshatch", "Crosshatch"

    def declare(self):
        return _props(IntProperty("stroke_length", 9, 3, 50), IntProperty("sharpness", 6, 0, 20),
                      IntProperty("strength", 1, 1, 3))

    def compute(self, t, src):
        h, w = src.shape[:2]
        length, sharp = float(t["stroke_length"]), float(t["sharpness"])
        rgb = src[..., :3].copy()
        alpha = src[..., 3]
        for p in range(int(t["strength"])):
            salt = 40 + p * 4
            r = lay_strokes(rgb, 45.0, length, salt).astype(np.float32)
            f = lay_strokes(rgb, -45.0, length, salt + 1).astype(np.float32)
            up = pen_lines(h, w, True, length, salt + 2)
            down = pen_lines(h, w, False, length, salt + 3)
            fringe = F.rnd(sharp / 20.0)
            detail = _busyness(rgb, alpha)
            mean = (r + f) * 0.5
            lightness = F.luma(mean) / 255.0
            busy = 0.05 + 0.95 * detail
            amp = 45.0 * busy * (0.35 + 0.65 * lightness) * (1 + sharp * 0.05) * (1 + p * 0.3)
            out = np.empty_like(mean)
            for c in range(3):
                shift = (c - 1) * fringe
                u = F._shift_clamped(up, shift, 0)
                d = F._shift_clamped(down, shift, 0)
                pale = np.maximum(u, 0) + np.maximum(d, 0)
                shade = -(np.minimum(u, 0) + np.minimum(d, 0))
                out[..., c] = mean[..., c] + (pale * (1 - lightness) - shade * lightness) * amp
            rgb = _u8(out)
            gain = sharp * 0.08
            if gain > 0:
                threshold = max(5.0 * (1 - sharp / 20.0), 2.0)
                soft = _u8(F.blur_rgb(np.dstack([rgb, alpha]), 1.0)).astype(np.float32)
                s = rgb.astype(np.float32)
                diff = s - soft
                over = np.maximum(np.abs(diff) - threshold, 0) * np.sign(diff)
                rgb = _u8(s + over * gain)
        return F.put_rgb(src, rgb)


def _busyness(rgb, alpha):
    """How much fine detail is round each pixel, 0..1."""
    soft = _u8(F.blur_rgb(np.dstack([rgb, alpha]), 1.5))
    detail = np.abs(F.luma(rgb) - F.luma(soft))
    return np.minimum(F.blur(detail, 4.0) / 10.0, 1.0)


class DarkStrokes(_Strokes):
    """Darks in short tight strokes, lights in long white ones."""
    name, label = "dark_strokes", "Dark Strokes"

    def declare(self):
        return _props(IntProperty("balance", 5, 0, 10), IntProperty("black_intensity", 6, 0, 10),
                      IntProperty("white_intensity", 5, 0, 10))

    def compute(self, t, src):
        balance, black, white = (float(t["balance"]), float(t["black_intensity"]),
                                 float(t["white_intensity"]))
        rgb = src[..., :3]
        lights = lay_strokes(rgb, 45.0, 14.0, 61).astype(np.float32)
        darks = lay_strokes(rgb, -45.0, 5.0, 62).astype(np.float32)
        judged = F.blur(F.luma(src) / 255.0, 2.0)
        black_from = 0.12 + black * 0.037 + balance * 0.028
        white_from = 0.85 - white * 0.03
        lift = white / 10.0 * 0.85
        contrast = 1 + (black + white) / 20.0 * 0.35
        pivot = (black_from + white_from) * 0.5 * 255.0
        lit = F.smooth((judged - 0.5) / 0.08 * 0.5 + 0.5)[..., None]
        paint = np.clip((lights * lit + darks * (1 - lit) - pivot) * contrast + pivot, 0, 255)
        level = F.luma(paint) / 255.0
        below = np.clip(1 - level / max(white_from, 0.01), 0, 1)
        deepen = np.maximum(level, 0.01) ** (black * 0.4 * below)
        paint = paint * deepen[..., None]
        level = level * deepen
        kept = F.smooth((level - black_from) / 0.08 * 0.5 + 0.5)
        whiten = np.clip((level - white_from) / max(1 - white_from, 0.01), 0, 1) * lift
        v = paint * kept[..., None]
        return F.put_rgb(src, v + (255 - v) * whiten[..., None])


def _dark_mask(tone, start):
    return 1 - F.smooth((tone / 255.0 - start) / 0.10 * 0.5 + 0.5)


class InkOutlines(_Strokes):
    """A pen-and-ink drawing: dark shapes filled solid, their rims and the
    main edges inked, chalky marks on the light side, over stroked color."""
    name, label = "ink_outlines", "Ink Outlines"

    def declare(self):
        return _props(IntProperty("stroke_length", 10, 1, 50), IntProperty("dark_intensity", 25, 0, 50),
                      IntProperty("light_intensity", 25, 0, 50))

    def compute(self, t, src):
        length, dark, light = (float(t["stroke_length"]), float(t["dark_intensity"]),
                               float(t["light_intensity"]))
        stroke = 4.0 + length * 0.5
        tone = F.luma(src)
        judged = F.blur(tone, 2.0)
        broad = F.blur(tone, 10.0)
        rgb = src[..., :3]
        rising = lay_strokes(rgb, 45.0, stroke, 81).astype(np.float32)
        falling = lay_strokes(rgb, -45.0, stroke, 82).astype(np.float32)
        detail = tone - F.blur(tone, 1.3)
        up = F.streak_field(detail, F.along(45.0), stroke)
        down = F.streak_field(detail, F.along(-45.0), stroke)
        outline = F.sobel(F.blur(tone, 0.8))
        ink_gain = dark * 0.04
        outline_gain = min(0.55 + dark * 0.009, 1.0)
        rim_gain = min(0.6 + light * 0.011, 1.0)
        chalk_gain = light * 0.02
        black_from = 0.08 + dark * 0.008
        contrast = 1 + (dark + light) / 100.0 * 1.1
        shape = np.minimum(_dark_mask(judged, black_from), _dark_mask(broad, black_from + 0.06))
        shape = F.fill_holes(F.close_gaps(shape, 26.0))
        within = _dark_mask(F.blur(judged, 26.0), black_from + 0.06)
        eroded = -F.widest_nearby(-shape, 2)
        ring = np.maximum(shape - eroded, 0)
        lit = F.smooth((judged / 255.0 - 0.5) / 0.08 * 0.5 + 0.5)

        def mark(v):
            return F.smooth((v - 1.0) / 4.0)
        drawn_edge = F.smooth((outline - 6.0) / 20.0)
        side = judged - broad
        dark_side = np.where(side < 0, drawn_edge, 0.0)
        light_side = np.where(side > 0, drawn_edge, 0.0)
        inked = np.minimum(np.maximum(mark(-down) * ink_gain, dark_side * outline_gain), 1.0)
        outside = 1 - within
        chalked = np.minimum(np.maximum(np.maximum(mark(up) * chalk_gain * outside,
                                                   light_side * rim_gain * outside),
                                        ring * rim_gain), 1.0)
        paint = rising * lit[..., None] + falling * (1 - lit[..., None])
        paint = np.clip((paint - 140.0) * contrast + 140.0, 0, 255) * (1 - 0.95 * within)[..., None]
        v = paint * ((1 - shape) * (1 - inked))[..., None]
        return F.put_rgb(src, v + (255 - v) * chalked[..., None])


class Spatter(_Strokes):
    """An airbrush's spatter: every pixel thrown a little way at random,
    less across strong edges, then gathered by a small median."""
    name, label = "spatter", "Spatter"

    def declare(self):
        return _props(IntProperty("spray_radius", 10, 0, 25), IntProperty("smoothness", 5, 1, 15),
                      _seed())

    def compute(self, t, src):
        radius, smoothness = float(t["spray_radius"]), float(t["smoothness"])
        if radius == 0:
            return src.copy()
        h, w = src.shape[:2]
        edges = F.sobel(F.blur(F.luma(src), 2.0))
        coherence = 0.4 + smoothness * 0.15
        xs, ys = F.grid(h, w)
        aside = F.unit_spread(F.blur(F.noise(xs, ys, t["seed"]) * 2 - 1, coherence))
        down = F.unit_spread(F.blur(F.noise(xs, ys + h + 977, t["seed"]) * 2 - 1, coherence))
        edge = np.clip((edges - 10.0) / 50.0, 0, 1)
        reach = radius * 0.45 * (1 - 0.3 * edge)
        rgb = F.displace_nearest(src[..., :3], xs + np.clip(aside, -2, 2) * reach,
                                 ys + np.clip(down, -2, 2) * reach)
        rgb = F.median_square(rgb, F.rnd(1.0 + smoothness * 0.08))
        return F.put_rgb(src, rgb)


class SprayedStrokes(_Strokes):
    """The picture sprayed along a direction: pixels thrown along the
    stroke, then settled into runs."""
    name, label = "sprayed_strokes", "Sprayed Strokes"

    def declare(self):
        return _props(IntProperty("stroke_length", 12, 0, 20), IntProperty("spray_radius", 7, 0, 25),
                      _direction(label="Direction"), _seed())

    def compute(self, t, src):
        length, radius = float(t["stroke_length"]), float(t["spray_radius"])
        if length == 0 and radius == 0:
            return src.copy()
        h, w = src.shape[:2]
        direction = F.along(t["direction"])
        throw = F.streaked_noise(h, w, max(length, 5.0), 91, direction, t["seed"])
        throw = F.unit_spread(F.blur(throw, 0.5))
        offset = np.clip(throw, -2, 2) * radius * 0.4
        xs, ys = F.grid(h, w)
        rgb = F.displace_nearest(src[..., :3], xs + direction[0] * offset,
                                 ys + direction[1] * offset)
        if length > 0:
            rgb = F.median_along(rgb, direction, max(length * 0.8, 1.0))
        return F.put_rgb(src, rgb)


SUMI_PAPER = np.array([245.0, 242.0, 231.0], np.float32)
SUMI_INK = np.array([28.0, 30.0, 36.0], np.float32)


class SumiE(_Strokes):
    """Japanese ink-wash: the picture flattened into a few washes of ink on
    warm paper, bleeding at the edges, strong colors kept a little."""
    name, label = "sumi_e", "Sumi-e"

    def declare(self):
        return _props(IntProperty("stroke_width", 8, 3, 15), IntProperty("stroke_pressure", 5, 0, 15),
                      IntProperty("contrast", 20, 0, 40))

    def compute(self, t, src):
        h, w = src.shape[:2]
        width, pressure, contrast = (float(t["stroke_width"]), float(t["stroke_pressure"]),
                                     float(t["contrast"]))
        washed = _u8(F.blur_rgb(src, 0.6 + width * 0.22))
        washed = F.median_square(washed, F.rnd(width * 0.6))
        paper_from = min(0.62 + contrast * 0.006, 0.95)
        ink_full = min(0.15 + contrast * 0.004, paper_from - 0.08)
        hard = 1 + contrast * 0.03
        load = 0.95 + pressure * 0.045
        tone = F.luma(washed) / 255.0
        d = np.clip((paper_from - tone) / (paper_from - ink_full), 0, 1)
        d = np.clip(d ** (1.0 / hard) * load, 0, 1)
        ink = np.minimum(np.maximum(np.floor(d * 4.0 - 0.3 + 0.5), 0) / 4.0, 1.0)
        xs, ys = F.grid(h, w)

        def grain(salt):
            return F.unit_spread(F.blur(F.noise(xs, ys + salt) * 2 - 1, 1.6))
        aside, down = grain(0), grain(h + 613)
        ink = F.displace_nearest(ink, xs + np.clip(aside, -2, 2) * 2.2,
                                 ys + np.clip(down, -2, 2) * 2.2)
        ink = F.blur(ink, 0.7 + width * 0.14)
        rim = F.sobel(ink)
        wf = washed.astype(np.float32)
        most, least = wf.max(axis=-1), wf.min(axis=-1)
        saturation = np.where(most > 1, (most - least) / np.maximum(most, 1), 0.0)
        keep = (F.smooth((saturation - 0.45) / 0.45) * 0.6)[..., None]
        body = np.clip(ink + rim * 0.22, 0, 1)[..., None]
        fibre = ((F.noise(xs, ys + 7919) * 2 - 1) * 3.5)[..., None]
        pigment = SUMI_INK * (1 - keep) + wf * 0.85 * keep
        return F.put_rgb(src, SUMI_PAPER * (1 - body) + pigment * body + fibre)


BRUSH_STROKES = (AccentedEdges, AngledStrokes, Crosshatch, DarkStrokes, InkOutlines, Spatter,
                 SprayedStrokes, SumiE)
