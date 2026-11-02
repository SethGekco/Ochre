# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop's Filter Gallery > Artistic family: Colored Pencil, Cutout,
Dry Brush, Film Grain, Fresco, Neon Glow, Paint Daubs, Palette Knife,
Plastic Wrap, Poster Edges, Rough Pastels, Smudge Stick, Sponge,
Underpainting and Watercolor. (Ochre's own Neon, Oil Painting, Pencil and
Ink Sketch, Cartoon and the rest share the Artistic menu with them.)

Most are a few recurring moves: paint laid in dabs (each pixel takes the
colour of whichever diagonal neighbour's blurred paint is calmest and
closest in tone), raised into relief, posterized with a canvas tooth;
medians that cut a picture into flat patches; motion-blurred strokes.

Algorithms, constants and slider ranges are ported from Kooka Pictura's
photorust engine (crates/pictura-filters/src/photorust/artistic,
GPL-3.0), whose sliders follow CS6's. Kooka works on bytes between stages
and so does this port (_u8), which is part of each filter's look. Window
operations clamp at the image edge rather than shrinking the window; that
is the only place the result can differ from Kooka's.
"""

import math

import numpy as np

from . import fields as F
from .filters import _Filter, _box
from .props import BoolProperty, ChoiceProperty, ColorProperty, IntProperty, PropertyCollection
from .sketch import BLACK, WHITE, _central, _light, _seed, _texture


def _props(*items):
    return PropertyCollection(list(items))


def _u8(a):
    return np.clip(np.floor(np.asarray(a, np.float32) + 0.5), 0, 255).astype(np.uint8)


def _f(a):
    return np.asarray(a, np.float32)


def _gauss(rgb, alpha, sigma):
    """Kooka's gaussian_blur on a picture: premultiplied, back to bytes."""
    if sigma <= 0:
        return rgb
    return _u8(F.blur_rgb(np.dstack([rgb, alpha]), sigma))


def _box_pm(rgb, alpha, r):
    """Box blur (2r+1 square), premultiplied, back to bytes."""
    if r <= 0:
        return rgb
    a = alpha.astype(np.float32)[..., None] / 255.0
    pm = np.concatenate([rgb.astype(np.float32) * a, a], -1)
    pm = _box(_box(pm, r, 0), r, 1)
    out = np.where(pm[..., 3:] > 1e-4, pm[..., :3] / np.maximum(pm[..., 3:], 1e-4), rgb)
    return _u8(out)


def _disc_pm(rgb, alpha, r):
    """Mean over a disc of radius r, premultiplied; the disc shrinks at the
    border (only pixels on the canvas count), as Kooka's does."""
    if r <= 0:
        return rgb
    h, w = rgb.shape[:2]
    a = alpha.astype(np.float64)[..., None] / 255.0
    pm = np.concatenate([rgb.astype(np.float64) * a, a, np.ones((h, w, 1))], -1)
    # Row sums over each half-width the disc uses, by cumulative sums.
    c = np.concatenate([np.zeros((h, 1, 5)), np.cumsum(pm, axis=1)], axis=1)
    xs = np.arange(w)
    total = np.zeros((h, w, 5))
    spans = {}
    for dy in range(-r, r + 1):
        half = int(math.sqrt(r * r - dy * dy))
        if half not in spans:
            x0 = np.clip(xs - half, 0, w)
            x1 = np.clip(xs + half + 1, 0, w)
            spans[half] = c[:, x1] - c[:, x0]
        rows = np.arange(h) + dy
        ok = (rows >= 0) & (rows < h)
        total[ok] += spans[half][rows[ok]]
    n = np.maximum(total[..., 4:5], 1)
    mean = total[..., :4] / n
    out = np.where(mean[..., 3:] > 1e-6, mean[..., :3] / np.maximum(mean[..., 3:], 1e-6), rgb)
    return np.clip(np.floor(out), 0, 255).astype(np.uint8)


def _median_of(rgb, across, down):
    """Per-channel median over a (2 across + 1) x (2 down + 1) window."""
    across, down = int(across), int(down)
    if across <= 0 and down <= 0:
        return rgb
    h, w = rgb.shape[:2]
    offs = [(dx, dy) for dy in range(-down, down + 1) for dx in range(-across, across + 1)]
    k = len(offs)
    p = np.pad(rgb, ((down, down), (across, across), (0, 0)), mode="edge")
    out = np.empty_like(rgb)
    rows = max(1, int(2_000_000 // max(1, w * k * 3)))
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        n = y1 - y0
        stack = np.stack([p[down + y0 + dy:down + y0 + dy + n, across + dx:across + dx + w]
                          for dx, dy in offs])
        out[y0:y1] = np.partition(stack, k // 2, axis=0)[k // 2]
    return out


def _motion(rgb, angle, distance):
    """Mean along a direction, `distance` pixels long; samples off the
    canvas are left out. Back to bytes."""
    steps = int(math.floor(max(distance, 0.0) + 0.5))
    if steps < 1:
        return rgb
    direction = F.along(angle)
    out = np.stack([F.line_mean(rgb[..., c].astype(np.float32), direction, steps // 2)
                    for c in range(3)], -1)
    return _u8(out)


def _darkest_along(rgb, angle, length):
    steps = max(int(math.floor(length + 0.5)), 1)
    offs = F._offsets(F.along(angle), steps // 2)
    m = max([1] + [max(abs(a), abs(b)) for a, b in offs])
    h, w = rgb.shape[:2]
    p = np.pad(rgb, ((m, m), (m, m), (0, 0)), constant_values=255)
    out = rgb.copy()
    for ox, oy in offs:
        out = np.minimum(out, p[m + oy:m + oy + h, m + ox:m + ox + w])
    return out


def speck_gain(scale):
    return 2.0 * math.sqrt(math.pi) * scale / 52.0


def _specks(h, w, scale, salt, seed):
    """Per-channel speckle, blurred to `scale`: 0..255 floats about 127.5."""
    xs, ys = F.grid(h, w)
    ch = [np.floor((F.speck(xs, ys, c + salt, seed) * 0.5 + 0.5) * 255.0) for c in range(3)]
    return np.floor(F.blur(np.stack(ch, -1), scale) + 0.5)


def _lay_tooth(rgb, levels, scale, salt, seed):
    h, w = rgb.shape[:2]
    tooth = _specks(h, w, scale, salt, seed)
    return _u8(_f(rgb) + (tooth - 127.5) * levels * speck_gain(scale))


def _weave(h, w, seed):
    """The canvas's tooth, -1..1: value noise on a 3-pixel grid."""
    xs, ys = F.grid(h, w)
    cx, cy = xs // 3, ys // 3

    def across(d):
        t = d / 3.0
        return t * t * (3 - 2 * t)
    sx, sy = across(xs % 3), across(ys % 3)
    top = F.noise(cx, cy, seed) * (1 - sx) + F.noise(cx + 1, cy, seed) * sx
    bottom = F.noise(cx, cy + 1, seed) * (1 - sx) + F.noise(cx + 1, cy + 1, seed) * sx
    return (top * (1 - sy) + bottom * sy) * 2 - 1


def _finish(rgb, levels, tooth, seed):
    """Posterize to `levels` steps and press in the canvas tooth."""
    steps = max(int(levels), 2)
    step = np.minimum(rgb.astype(np.int32) * steps // 256, steps - 1)
    mixed = step * 255.0 / (steps - 1)
    grain = _weave(rgb.shape[0], rgb.shape[1], seed) * tooth if tooth > 0 else 0.0
    return _u8(mixed + (grain[..., None] if tooth > 0 else 0.0))


def _lightness(rgb, inverted=False):
    lum = np.floor(F.luma(rgb)).astype(np.int32)
    return 255 - lum if inverted else lum


def _roughness(rgb, reach):
    """How far brightness strays from its neighbourhood mean, averaged."""
    lum = _lightness(rgb).astype(np.float32)
    mean = np.floor(_box(_box(lum, reach, 0), reach, 1) + 0.5)
    strays = np.abs(lum - mean)
    return np.floor(_box(_box(strays, reach, 0), reach, 1) + 0.5).astype(np.int64)


def _lay_the_paint(rgb, load, calm, reach):
    h, w = rgb.shape[:2]
    mine = F.luma(rgb)
    load_l = F.luma(load)
    xs, ys = F.grid(h, w)
    best = np.full((h, w), np.iinfo(np.int64).max)
    bx, by = xs.copy(), ys.copy()
    for dx, dy in ((-reach, -reach), (reach, -reach), (-reach, reach), (reach, reach)):
        sx, sy = np.clip(xs + dx, 0, w - 1), np.clip(ys + dy, 0, h - 1)
        apart = np.minimum(np.abs(mine - load_l[sy, sx]).astype(np.int64), 511)
        score = (calm[sy, sx] << 9) | apart
        better = score < best
        best = np.where(better, score, best)
        bx, by = np.where(better, sx, bx), np.where(better, sy, by)
    return load[by, bx]


def _paint_in_dabs(rgb, alpha, reach, rounded):
    if reach <= 0:
        return rgb
    load = _disc_pm(rgb, alpha, reach) if rounded else _box_pm(rgb, alpha, reach)
    return _lay_the_paint(rgb, load, _roughness(rgb, reach), reach)


def _raise(rgb, alpha, scale, amount):
    if amount <= 0:
        return rgb
    flat = _gauss(rgb, alpha, scale)
    lift = np.clip((_f(rgb) - flat) * amount, -14.0, 14.0)
    return _u8(_f(rgb) + lift)


def _put_back(rgb, over, under, gain):
    return _u8(_f(rgb) + (_f(over) - _f(under)) * gain)


def _sharpen(rgb, alpha, gain):
    if gain <= 0:
        return rgb
    return _put_back(rgb, rgb, _gauss(rgb, alpha, 2.0), gain)


class _Artistic(_Filter):
    category = "Artistic"


class ColoredPencil(_Artistic):
    """Crosshatched colored pencil on paper: where the picture has detail
    it is drawn in, elsewhere the paper (Background color, dimmed by Paper
    Brightness) shows between the hatching."""
    name, label = "colored_pencil", "Colored Pencil"

    def declare(self):
        return _props(IntProperty("pencil_width", 4, 1, 24), IntProperty("stroke_pressure", 8, 0, 15),
                      IntProperty("paper_brightness", 25, 0, 50),
                      ColorProperty("background", WHITE), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        width, seed = int(t["pencil_width"]), t["seed"]
        rgb, alpha = src[..., :3], src[..., 3]
        sheet = t["paper_brightness"] / 50.0
        paper = np.asarray(t["background"][:3], np.float32) * sheet
        gain = t["stroke_pressure"] / 8.0
        press = min(gain, 1.0)
        gap = min(max(0.35 + (1 - gain) * 0.2, 0.0), 0.97) if gain > 0 else 0.97
        soft = _gauss(rgb, alpha, 2.0)
        detail = np.minimum(np.abs(F.luma(rgb) - F.luma(soft)), 255).astype(np.uint8).astype(np.float32)
        drawn = np.floor(F.blur(detail, min(width * 1.3, 8.0)) + 0.5)
        drawn = np.floor(F.smooth((drawn * gain - 1.5) / 2.5) * 255.0)
        drawn = F.widest_nearby(drawn, max(F.rnd(min(width * 2.5, 8.0)), 1))
        drawn = np.floor(F.blur(drawn, min(width * 0.3, 4.0)) + 0.5) / 255.0
        outline = F.widest_nearby(detail, max(F.rnd(width * 0.5), 1))
        wash = _f(_gauss(rgb, alpha, 0.8))
        xs, ys = F.grid(h, w)
        spacing = max(width, 1) * 1.3

        def stroke(across, along, salt):
            stripe = np.floor(across / spacing)
            jitter = F.noise(stripe.astype(np.int64), salt, seed) * spacing
            q = (across + jitter) / spacing
            frac = np.abs(q - np.trunc(q))
            ridge = (1 - np.abs(2 * frac - 1)) ** 2.5
            dash = F.noise(np.floor(along / (spacing * 3)).astype(np.int64),
                           stripe.astype(np.int64) + 31, seed)
            return ridge * (0.35 + 0.65 * dash)
        sx, dx = (xs + ys).astype(np.float32), (xs - ys).astype(np.float32)
        hatch = np.maximum(stroke(sx, dx, 0), stroke(dx, sx, 1) * 0.45)
        stroke_k = 0.12 + 0.88 * drawn
        coverage = (press * stroke_k * (1 - gap * (1 - hatch)))[..., None]
        ink = (np.maximum(hatch - 0.6, 0) / 0.4 * 12.0 * gain)[..., None]
        pressed = np.minimum(outline / 255.0 * gain, 1.0)
        lead = wash * (1 - 0.9 * pressed)[..., None]
        return F.put_rgb(src, paper + (lead - paper) * coverage - ink)


def _pieces(key):
    """Connected pieces (4-connected) of equal keys: a label per pixel."""
    h, w = key.shape
    lab = np.arange(h * w, dtype=np.int64).reshape(h, w)
    same_x = key[:, :-1] == key[:, 1:]
    same_y = key[:-1] == key[1:]
    while True:
        m = lab.copy()
        m[:, :-1] = np.where(same_x, np.minimum(m[:, :-1], lab[:, 1:]), m[:, :-1])
        m[:, 1:] = np.where(same_x, np.minimum(m[:, 1:], lab[:, :-1]), m[:, 1:])
        m[:-1] = np.where(same_y, np.minimum(m[:-1], lab[1:]), m[:-1])
        m[1:] = np.where(same_y, np.minimum(m[1:], lab[:-1]), m[1:])
        flat = m.ravel()
        while True:                         # pointer jumping
            nxt = flat[flat]
            if np.array_equal(nxt, flat):
                break
            flat = nxt
        m = flat.reshape(h, w)
        if np.array_equal(m, lab):
            return lab
        lab = m


class Cutout(_Artistic):
    """The picture rebuilt from roughly cut pieces of colored paper: the
    median flattens it, a few levels per channel cut it into bands, and
    each connected piece takes the mean color of the picture under it."""
    name, label = "cutout", "Cutout"

    def declare(self):
        return _props(IntProperty("levels", 4, 2, 8), IntProperty("edge_simplicity", 4, 0, 10),
                      IntProperty("edge_fidelity", 2, 1, 3))

    def compute(self, t, src):
        h, w = src.shape[:2]
        levels = int(t["levels"])
        rgb = src[..., :3]
        reach = 1 + int(t["edge_simplicity"])
        flat = _median_of(rgb, reach, reach)
        bands = np.minimum(flat.astype(np.int64) * levels // 256, levels - 1)
        key = (bands[..., 0] << 16) | (bands[..., 1] << 8) | bands[..., 2]
        vote = 3 - int(t["edge_fidelity"])
        if vote > 0:
            key = _tidy(key, vote)
        lab = _pieces(key).ravel()
        n = h * w
        counts = np.maximum(np.bincount(lab, minlength=n), 1)
        flat_rgb = rgb.reshape(-1, 3).astype(np.int64)
        colours = np.stack([np.bincount(lab, weights=flat_rgb[:, c], minlength=n)
                            for c in range(3)], -1)
        colours = np.floor(colours / counts[:, None])
        return F.put_rgb(src, colours[lab].reshape(h, w, 3))


def _tidy(key, radius):
    """Each pixel takes the commonest key in its window (its own on a tie,
    else the first in reading order); the window shrinks at the border."""
    h, w = key.shape
    p = np.pad(key, radius, constant_values=-1)
    offs = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    samples = [p[radius + dy:radius + dy + h, radius + dx:radius + dx + w] for dx, dy in offs]
    counts = []
    for s in samples:
        c = np.zeros((h, w), np.int32)
        for o in samples:
            c += (o == s)
        counts.append(np.where(s >= 0, c, 0))
    best = np.max(counts, axis=0)
    own = np.zeros((h, w), np.int32)
    for s in samples:
        own += (s == key)
    won = key.copy()
    chosen = own == best
    for s, c in zip(samples, counts):
        take = ~chosen & (c == best)
        won = np.where(take, s, won)
        chosen |= take
    return won


def _brush_props():
    return (IntProperty("brush_size", 2, 0, 10), IntProperty("brush_detail", 8, 0, 10),
            IntProperty("texture", 1, 1, 3), _seed())


class DryBrush(_Artistic):
    """Paint dragged on with a dry brush: dabs, a little relief, then a
    few flat levels over the canvas tooth."""
    name, label = "dry_brush", "Dry Brush"

    def declare(self):
        return _props(*_brush_props())

    def compute(self, t, src):
        rgb, alpha = src[..., :3], src[..., 3]
        reach = int(t["brush_size"])
        rgb = _paint_in_dabs(rgb, alpha, reach, False)
        rgb = _raise(rgb, alpha, float(max(reach, 1)), (t["texture"] - 1) * 0.5)
        return F.put_rgb(src, _finish(rgb, 4 + t["brush_detail"] * 6, 1.2, t["seed"]))


class Fresco(_Artistic):
    """Coarse round dabs sunk into wet plaster: darker, contrastier paint."""
    name, label = "fresco", "Fresco"

    def declare(self):
        return _props(*_brush_props())

    def compute(self, t, src):
        rgb, alpha = src[..., :3], src[..., 3]
        reach = int(t["brush_size"]) * 2
        rgb = _paint_in_dabs(rgb, alpha, reach, True)
        lut = np.floor(255.0 * np.minimum(np.arange(256) / 255.0 * 1.1, 1.0) ** 2.0 + 0.5)
        rgb = lut.astype(np.uint8)[rgb]
        rgb = _raise(rgb, alpha, float(max(reach, 1)), (t["texture"] - 1) * 0.5)
        return F.put_rgb(src, _finish(rgb, 4 + t["brush_detail"] * 6, 1.2, t["seed"]))


class FilmGrain(_Artistic):
    """Even film grain in the shadows and midtones, a smoother, brighter
    pattern in the highlights."""
    name, label = "film_grain", "Film Grain"

    def declare(self):
        return _props(IntProperty("grain", 4, 0, 20), IntProperty("highlight_area", 0, 0, 20),
                      IntProperty("intensity", 10, 0, 10), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        seed = t["seed"]
        coarse = t["grain"] / 20.0 * 140.0
        line = 255.0 * (1 - t["highlight_area"] / 20.0)
        pull = t["intensity"] / 10.0 * 0.5
        xs, ys = F.grid(h, w)
        silver = F.speck(xs, ys, 0, seed)
        out = np.empty((h, w, 3), np.float32)
        for c in range(3):
            v = src[..., c].astype(np.float32)
            lit = F.smooth((v - line) / 48.0)
            amount = coarse * (1 - 0.7 * lit)
            deposit = silver + lit * 0.6 * (F.speck(xs, ys, c + 1, seed) - silver)
            value = v + deposit * amount
            out[..., c] = value + (255 - value) * lit * pull
        return F.put_rgb(src, out)


class NeonGlow(_Artistic):
    """A glow of the Glow Color round the lights (negative Glow Size:
    round the darks), over the picture mapped between the foreground and
    background colors."""
    name, label = "neon_glow", "Neon Glow"

    def declare(self):
        return _props(IntProperty("glow_size", 5, -24, 24), IntProperty("glow_brightness", 15, 0, 50),
                      ColorProperty("glow_color", (0, 0, 255, 255)),
                      ColorProperty("foreground", BLACK), ColorProperty("background", WHITE))

    def compute(self, t, src):
        size = int(t["glow_size"])
        strength = t["glow_brightness"] / 25.0
        lamp = _lightness(src[..., :3], size < 0).astype(np.float32)
        lamp = np.floor(F.blur(lamp, abs(size)) + 0.5) / 255.0
        own = F.luma(src) / 255.0
        light = lamp * lamp * strength
        mix = np.clip(1 - np.abs(1 - (own + light)), 0, 1)
        tint = np.clip(mix * light, 0, 1)[..., None]
        between = F.swatches(t["background"], t["foreground"], mix)
        glow = np.asarray(t["glow_color"][:3], np.float32)
        return F.put_rgb(src, between + (glow - between) * tint)


DAUB_BRUSHES = ("simple", "light_rough", "dark_rough", "wide_sharp", "wide_blurry", "sparkle")
DAUB_LABELS = {"simple": "Simple", "light_rough": "Light Rough", "dark_rough": "Dark Rough",
               "wide_sharp": "Wide Sharp", "wide_blurry": "Wide Blurry", "sparkle": "Sparkle"}


class PaintDaubs(_Artistic):
    """Paint in daubs: a median of the brush's size and shape, then the
    brush type's finish -- sharpened, blurred, roughened or sparkling."""
    name, label = "paint_daubs", "Paint Daubs"

    def declare(self):
        return _props(IntProperty("brush_size", 8, 1, 50), IntProperty("sharpness", 7, 0, 40),
                      ChoiceProperty("brush_type", "simple", list(DAUB_BRUSHES), labels=DAUB_LABELS),
                      _seed())

    def compute(self, t, src):
        rgb, alpha = src[..., :3], src[..., 3]
        brush, sharp, seed = t["brush_type"], float(t["sharpness"]), t["seed"]
        reach = max(float(F.rnd(t["brush_size"] * 0.45)), 1.0)
        if brush in ("wide_sharp", "wide_blurry"):
            across, down = int(reach * 2), int(max(reach / 2, 1))
        else:
            across = down = int(reach)
        rough = brush in ("light_rough", "dark_rough")
        if rough:
            rgb = _lay_tooth(rgb, 15.0, 2.5, 0, seed)
        rgb = _median_of(rgb, across, down)
        edge, halo = sharp * 0.2, sharp * 0.5
        if brush == "simple":
            rgb = _sharpen(rgb, alpha, edge)
        elif brush == "wide_sharp":
            rgb = _sharpen(rgb, alpha, edge * 2)
        elif brush == "wide_blurry":
            rgb = _sharpen(_gauss(rgb, alpha, down * 2 * 0.3), alpha, edge * 0.5)
        elif rough and halo > 0:
            soft = _gauss(rgb, alpha, 3.0)
            diff = F.luma(rgb) - F.luma(soft)
            diff = np.maximum(diff, 0) if brush == "light_rough" else np.minimum(diff, 0)
            rgb = _u8(_f(rgb) + (diff * halo)[..., None])
        elif brush == "sparkle":
            rgb = _sparkle(rgb, alpha, sharp, seed)
        if rough:
            rgb = _lay_tooth(rgb, 14.0, 2.0, 3, seed)
        return F.put_rgb(src, rgb)


def _sparkle(rgb, alpha, sharpness, seed):
    h, w = rgb.shape[:2]
    field = _gauss(rgb, alpha, 0.7)
    wobble = _specks(h, w, 12.0, 7, seed)[..., 0]
    band = np.floor((F.luma(field) + (wobble - 127.5) * speck_gain(12.0)) / 6.0)
    right = F._shift_clamped(band, 1, 0)
    below = F._shift_clamped(band, 0, 1)
    lines = ((band != right) | (band != below)).astype(np.float32)
    glow = np.floor(F.blur(lines * 255.0, 2.0) + 0.5) / 255.0
    light = 0.65 + sharpness * 0.05
    lum = F.luma(rgb) / 255.0
    k = np.minimum((lines * lum * lum + glow * 0.6 * (0.3 + 0.7 * lum)) * light, 1.0)[..., None]
    out = _u8(_f(rgb) + (255 - _f(rgb)) * k)
    out = _put_back(out, out, _gauss(out, alpha, 1.5), sharpness * 0.1)
    x = (np.arange(256) / 255.0) ** 0.85
    kk = F.smooth((x - 0.3) / 0.7)
    lut = np.floor((x + 0.1 * kk * (1 - x)) * 255.0 + 0.5).astype(np.uint8)
    return lut[out]


def _kuwahara(rgb, reach):
    """Each pixel takes the mean of whichever of its four overlapping
    quarters varies least in brightness (windows shrink at the border)."""
    h, w = rgb.shape[:2]
    half = max(-(-reach // 2), 1)
    f = rgb.astype(np.float64)
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    planes = np.dstack([f, lum, lum * lum])
    sums = np.zeros((h + 1, w + 1, 5))
    sums[1:, 1:] = planes.cumsum(0).cumsum(1)
    xs, ys = F.grid(h, w)
    best = np.full((h, w), np.inf)
    out = np.zeros((h, w, 3))
    for dx, dy in ((-half, -half), (half, -half), (-half, half), (half, half)):
        cx, cy = xs + dx, ys + dy
        x0, x1 = np.clip(cx - half, 0, w), np.clip(cx + half + 1, 0, w)
        y0, y1 = np.clip(cy - half, 0, h), np.clip(cy + half + 1, 0, h)
        n = np.maximum((x1 - x0) * (y1 - y0), 1)[..., None]
        q = (sums[y1, x1] - sums[y0, x1] - sums[y1, x0] + sums[y0, x0]) / n
        spread = q[..., 4] - q[..., 3] ** 2
        better = spread < best
        best = np.where(better, spread, best)
        out = np.where(better[..., None], q[..., :3], out)
    return _u8(out)


class PaletteKnife(_Artistic):
    """Thin flat paint laid with a knife: a Kuwahara filter twice, a few
    rounded levels, softened."""
    name, label = "palette_knife", "Palette Knife"

    def declare(self):
        return _props(IntProperty("stroke_size", 25, 1, 50), IntProperty("stroke_detail", 3, 1, 3),
                      IntProperty("softness", 0, 0, 10))

    def compute(self, t, src):
        rgb, alpha = src[..., :3], src[..., 3]
        reach = max(int(t["stroke_size"]) // 3, 1)
        for _ in range(2):
            rgb = _kuwahara(rgb, reach)
        steps = 1 + int(t["stroke_detail"]) * 3
        v = np.arange(256) / 255.0
        rounded = np.floor(v * steps + 0.5) / steps
        lut = np.floor((rounded * 0.5 + v * 0.5) * 255.0 + 0.5).astype(np.uint8)
        rgb = _gauss(lut[rgb], alpha, t["softness"] * 0.5)
        return F.put_rgb(src, rgb)


class PlasticWrap(_Artistic):
    """The picture shrink-wrapped in glossy plastic: highlights where the
    local relief faces the light."""
    name, label = "plastic_wrap", "Plastic Wrap"

    def declare(self):
        return _props(IntProperty("highlight_strength", 15, 0, 20), IntProperty("detail", 9, 1, 15),
                      IntProperty("smoothness", 7, 1, 15))

    def compute(self, t, src):
        hl = float(t["highlight_strength"])
        b = F.luma(src) / 255.0
        relief = F.blur(b - F.blur(b, 8.0), 1.0 + t["smoothness"] * 0.7)
        relief = np.tanh(relief * (0.5 + t["detail"] * 0.08))
        light = np.array([-1.0, -1.0, 1.2])
        light /= np.linalg.norm(light)
        half = np.array([light[0], light[1], light[2] + 1.0])
        half /= np.linalg.norm(half)
        gloss = (hl / 20.0) ** 1.6 * 5.0
        dull = 1 - 0.25 * hl / 20.0
        dx, dy = F.gradient(relief)
        nx, ny = -dx * 250.0, -dy * 250.0
        norm = np.sqrt(nx * nx + ny * ny + 1)
        facing = np.maximum((nx * half[0] + ny * half[1] + half[2]) / norm, 0)
        shine = np.minimum(facing ** 50.0 * gloss, 1.0)[..., None]
        under = _f(src[..., :3]) * dull
        return F.put_rgb(src, under + (255 - under) * shine)


class PosterEdges(_Artistic):
    """Posterized color with black ink along the edges."""
    name, label = "poster_edges", "Poster Edges"

    def declare(self):
        return _props(IntProperty("edge_thickness", 2, 0, 10), IntProperty("edge_intensity", 1, 0, 10),
                      IntProperty("posterization", 2, 0, 10))

    def compute(self, t, src):
        rgb, alpha = src[..., :3], src[..., 3]
        thick, inten, post = (float(t["edge_thickness"]), float(t["edge_intensity"]),
                              float(t["posterization"]))
        settled = _f(_gauss(rgb, alpha, 2.0))
        brightness = F.luma(_gauss(rgb, alpha, 0.5))
        around = F.blur(brightness, 1.5 + thick * 0.5)
        steps = 3.0 + post + post * post * 0.25 - 1.0
        ink_gain = (12.0 + inten * 4.0) / 255.0
        level = F.luma(settled)
        band = np.minimum(np.floor(level / 255.0 * steps + 0.64), steps) * 255.0 / steps
        scale = (band / np.maximum(level, 1.0))[..., None]
        bare = (1 - np.clip((around - brightness - 2.0) * ink_gain, 0, 1))[..., None]
        return F.put_rgb(src, np.minimum(settled * scale, 255) * bare)


class RoughPastels(_Artistic):
    """Chalk pastel strokes on a textured ground."""
    name, label = "rough_pastels", "Rough Pastels"

    def declare(self):
        return _props(IntProperty("stroke_length", 6, 0, 40), IntProperty("stroke_detail", 4, 1, 20),
                      _texture(), IntProperty("scaling", 100, 50, 200),
                      IntProperty("relief", 20, 0, 50), _light(label="Light Direction"),
                      BoolProperty("invert", False))

    def compute(self, t, src):
        h, w = src.shape[:2]
        rgb, alpha = src[..., :3], src[..., 3]
        length, detail = float(t["stroke_length"]), float(t["stroke_detail"])
        stroke = 6.0 + length * 0.6
        direction = F.along(45.0)
        drag = F.streaked_noise(h, w, stroke * 2 + 3, 11, direction)
        offset = np.clip(drag, -2, 2) * stroke * 0.12
        xs, ys = F.grid(h, w)
        dragged = F.displace_nearest(rgb, xs + direction[0] * offset, ys + direction[1] * offset)
        smear = 2.0 + length * 0.1
        strokes = _f(_motion(dragged, 45.0, smear))
        sharp = _f(_motion(rgb, 45.0, smear))
        soft = _f(_motion(_gauss(rgb, alpha, 2.0), 45.0, smear))
        carry = 0.5 + detail * 0.15
        grain = F.streaked_noise(h, w, stroke * 1.5 + 3, 12, direction)
        chalk = strokes + (sharp - soft) * carry
        lightness = F.luma(chalk) / 255.0
        tooth = (grain * 22.0 * (1.2 - lightness))[..., None]
        v = np.floor(np.clip(chalk + tooth, 0, 255) + 0.5)
        pale = np.floor(255.0 * (v / 255.0) ** 0.8 + 0.5)
        out = F.relief(pale, t["texture"], t["scaling"], t["relief"], t["light"], t["invert"])
        return F.put_rgb(src, out)


class SmudgeStick(_Artistic):
    """Short diagonal smudges that soften the darks; the lights brighten
    and lose detail."""
    name, label = "smudge_stick", "Smudge Stick"

    def declare(self):
        return _props(IntProperty("stroke_length", 2, 0, 10), IntProperty("highlight_area", 0, 0, 20),
                      IntProperty("intensity", 10, 0, 10), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        rgb = src[..., :3]
        length, highlight = float(t["stroke_length"]), float(t["highlight_area"])
        intensity = t["intensity"] / 10.0
        stroke = 8.0 + length * 2.0
        patch = F.rnd(1.0 + length * 0.35)
        patches = _median_of(rgb, patch, patch)
        streak = _f(_motion(patches, 45.0, stroke))
        d = rgb.astype(np.int32) - patches.astype(np.int32)
        detail = np.clip(128 + np.trunc(d / 2.0).astype(np.int32), 0, 255).astype(np.uint8)
        detail = _f(_motion(detail, 45.0, stroke * 0.5))
        grain = F.streaked_noise(h, w, stroke * 0.5 + 3, 13, F.along(45.0), t["seed"])
        darkest = _f(_motion(_darkest_along(patches, 45.0, stroke), 45.0, stroke * 0.5))
        start = (0.9 - highlight * 0.02) * 255.0
        soft = 0.15 * 255.0
        contrast = 1 + intensity * 0.2
        lift = intensity * 0.9
        picture = _f(rgb)
        lp, ld = F.luma(picture), F.luma(darkest)
        darkness = 1 - np.minimum(lp, ld) / 255.0
        own = 1 - lp / 255.0
        smudged = (np.maximum(darkness, 0) ** 0.4 * 0.5)[..., None]
        own_k = (np.maximum(own, 0) ** 0.4)[..., None]
        smear = streak + (np.minimum(darkest, streak) - streak) * 0.8
        under = picture + (_f(patches) - picture) * own_k
        streaks = (detail - 128.0) * 2.0 * 0.8 + (grain * 7.0 * own)[..., None]
        paint = (under + (smear - under) * smudged + streaks - 128.0) * contrast + 128.0
        below = np.clip(1 - F.luma(paint) / max(start, 1.0), 0, 1)
        paint = paint * (1 - 0.4 * 4 * below * (1 - below))[..., None]
        lit = (F.smooth((F.luma(paint) - start) / soft) * lift)[..., None]
        return F.put_rgb(src, paint + (255 - paint) * lit)


class Sponge(_Artistic):
    """Dabbed on with a sponge: contrasting blotches over softened color."""
    name, label = "sponge", "Sponge"

    def declare(self):
        return _props(IntProperty("brush_size", 2, 0, 10), IntProperty("definition", 12, 0, 25),
                      IntProperty("smoothness", 5, 1, 15), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        rgb, alpha = src[..., :3], src[..., 3]
        size, definition, smoothness = (float(t["brush_size"]), float(t["definition"]),
                                        float(t["smoothness"]))
        settle = F.rnd(1.0 + size * 0.2)
        settled = _f(_gauss(_median_of(rgb, settle, settle), alpha, smoothness * 0.12))
        scale = 2.0 + size * 0.6
        blotches = _specks(h, w, scale, 31, t["seed"])[..., 0]
        cut = max(3.0 - (smoothness - 1) * 0.12, 0.5)
        cell = np.tanh((blotches - 127.5) * speck_gain(scale) * cut) * definition * 1.4
        return F.put_rgb(src, settled + cell[..., None])


class Underpainting(_Artistic):
    """The picture painted on a textured ground, the paint pushed about by
    the texture and the ground showing through in patches."""
    name, label = "underpainting", "Underpainting"

    def declare(self):
        return _props(IntProperty("brush_size", 6, 0, 40), IntProperty("texture_coverage", 16, 0, 40),
                      _texture(), IntProperty("scaling", 100, 50, 200),
                      IntProperty("relief", 4, 0, 50), _light("top", label="Light Direction"),
                      BoolProperty("invert", False), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        rgb, alpha = src[..., :3], src[..., 3]
        size, coverage, seed = float(t["brush_size"]), float(t["texture_coverage"]), t["seed"]
        patch = F.rnd(size * 0.6)
        paint = _median_of(rgb, patch, patch) if patch > 0 else rgb
        paint = _f(_gauss(paint, alpha, size * 0.8))
        surface = F.height_map(t["texture"], h, w, t["scaling"], seed)
        if t["invert"]:
            surface = 1 - surface
        surface = F.blur(surface, 0.7)
        lx, ly = F.towards(t["light"])
        dx, dy = _central(surface)
        push = (dx * lx + dy * ly) * coverage * 3.5 * F.patchiness(h, w, 0.7, seed)
        xs, ys = F.grid(h, w)
        fx = np.clip(xs + lx * push, 0, w - 1)
        fy = np.clip(ys + ly * push, 0, h - 1)
        x0, y0 = np.floor(fx).astype(np.intp), np.floor(fy).astype(np.intp)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]
        top = paint[y0, x0] * (1 - tx) + paint[y0, x1] * tx
        bottom = paint[y1, x0] * (1 - tx) + paint[y1, x1] * tx
        laid = np.floor(top * (1 - ty) + bottom * ty + 0.5)
        relief = t["relief"] + F.rnd(coverage)
        out = F.relief(laid, t["texture"], t["scaling"], relief, t["light"], t["invert"], seed=seed,
                       dark_bias=0.4, bevel=0.7, crisp=0.75, occlusion=70.0, patchy=0.7)
        return F.put_rgb(src, out)


def _surface_blur(rgb, reach, threshold):
    """Mean of the window values within `threshold` of the centre, per
    channel (Kooka's surface_blur); the window shrinks at the border."""
    h, w = rgb.shape[:2]
    f = rgb.astype(np.float32)
    p = np.pad(f, ((reach, reach), (reach, reach), (0, 0)), constant_values=-1e6)
    total = np.zeros_like(f)
    count = np.zeros_like(f)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            s = p[reach + dy:reach + dy + h, reach + dx:reach + dx + w]
            ok = np.abs(s - f) <= threshold
            total += np.where(ok, s, 0)
            count += ok
    return np.floor(total / np.maximum(count, 1)).astype(np.uint8)


class Watercolor(_Artistic):
    """Watercolor: color pooled into washes, edges darkened where the
    paint gathers, the shadows deepened."""
    name, label = "watercolor", "Watercolor"

    def declare(self):
        return _props(IntProperty("brush_detail", 9, 1, 14), IntProperty("shadow_intensity", 1, 0, 10),
                      IntProperty("texture", 1, 1, 3), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        rgb, alpha = src[..., :3], src[..., 3]
        detail, shadow, texture, seed = (float(t["brush_detail"]), float(t["shadow_intensity"]),
                                         float(t["texture"]), t["seed"])
        grain = _specks(h, w, 2.0, 61, seed)[..., 0]
        grit = _specks(h, w, 0.7, 62, seed)[..., 0]
        grained = _u8(_f(rgb) + ((grain - 127.5) * 2.0 * speck_gain(2.0))[..., None])
        reach = int(max(F.rnd(3.6 - (detail - 1) * 0.2), 1))
        wash = _surface_blur(_paint_in_dabs(grained, alpha, reach, True), 2, 16)
        steps = max(8.0 + (detail - 1) - 1, 1.0)
        level = F.luma(wash)
        pool = np.floor(level / 255.0 * steps + 0.5) / steps * 255.0
        mix = 0.35 * (1 - level / 255.0)
        scale = (1 + (pool / np.maximum(level, 1.0) - 1) * mix)[..., None]
        wash = _u8(_f(wash) * scale)
        tone = F.blur(F.luma(wash), 0.8)
        around = F.blur(tone, 3.0)
        start = 0.12 + shadow * 0.085
        depth = 255.0 * (np.arange(256) / 255.0) ** 1.15
        line = 1 - np.minimum(F.sobel(tone) * 0.0025, 1.0)
        paint = depth[wash] * line[..., None]
        pale = around / 255.0
        lift = (tone - around) * 7.0 * pale * pale
        grey = F.luma(paint)
        lifted = np.maximum(grey + lift, 0)
        ratio = (lifted / np.maximum(grey, 1.0))[..., None]
        paint = np.clip(lifted[..., None] + (paint * ratio - lifted[..., None]) * 1.25, 0, 255)
        g = np.minimum(lifted, 255.0)
        kept = F.smooth((g / 255.0 - (start - 0.1)) / 0.2)[..., None]
        speck = ((grit - 127.5) * (texture - 1) * 9.0 * speck_gain(0.7) * (1 - g / 255.0))[..., None]
        return F.put_rgb(src, paint * kept + speck)


ARTISTIC = (ColoredPencil, Cutout, DryBrush, FilmGrain, Fresco, NeonGlow, PaintDaubs,
            PaletteKnife, PlasticWrap, PosterEdges, RoughPastels, SmudgeStick, Sponge,
            Underpainting, Watercolor)
