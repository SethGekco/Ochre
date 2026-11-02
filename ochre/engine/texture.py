# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop's Filter Gallery > Texture family: Craquelure, Grain, Mosaic
Tiles, Patchwork, Stained Glass and Texturizer. (Ochre's own Canvas
Texture, GIMP's Apply Canvas, stays under Artistic.)

Each builds a height field -- cracks, tiles, squares -- and lights it from
the top left, or lays a grain over the picture. Algorithms, constants and
slider ranges are ported from Kooka Pictura's photorust engine
(crates/pictura-filters/src/photorust/texture, GPL-3.0), whose sliders
follow CS6's. Alpha is never touched.
"""

import math

import numpy as np

from . import fields as F
from .filters import _Filter
from .props import BoolProperty, ChoiceProperty, ColorProperty, IntProperty, PropertyCollection
from .sketch import BLACK, WHITE, _light, _seed, _texture

INF = np.float32(np.inf)


def _props(*items):
    return PropertyCollection(list(items))


def _round(a):
    """Halves away from zero, as Rust rounds."""
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def _safe_div(num, den, where):
    """num / den where `where`, else infinity -- without dividing by zero."""
    return np.where(where, num / np.where(where, den, 1.0), INF)


def _cast_shadow(surface, direction, reach, fall=0.0):
    """How far the surface rises above each pixel along the light, up to
    `reach` pixels, less `fall` per pixel: the shadow a ridge throws."""
    h, w = surface.shape
    lx, ly = direction
    xs, ys = F.grid(h, w)
    shadow = np.zeros((h, w), np.float32)
    for k in range(1, reach + 1):
        sx = xs + F.rnd(lx * k)
        sy = ys + F.rnd(ly * k)
        ok = (sx >= 0) & (sy >= 0) & (sx < w) & (sy < h)
        above = surface[np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)] - surface - fall * k
        shadow = np.where(ok, np.maximum(shadow, above), shadow)
    return shadow


def _shade(rgb, shade):
    return rgb * (1 + shade[..., None] * F.SHADE_BY_COLOUR) + (shade * F.SHADE_BY_LIGHT)[..., None]


class _Texture(_Filter):
    category = "Texture"


class Craquelure(_Texture):
    """The picture painted on cracked plaster: long courses of cracks in the
    lights, crumpled contour cracks and pockets in the darks, all in relief."""
    name, label = "craquelure", "Craquelure"

    def declare(self):
        return _props(IntProperty("crack_spacing", 10, 2, 100), IntProperty("crack_depth", 6, 0, 10),
                      IntProperty("crack_brightness", 9, 0, 10))

    def compute(self, t, src):
        h, w = src.shape[:2]
        spacing, depth, brightness = (float(t["crack_spacing"]), float(t["crack_depth"]),
                                      float(t["crack_brightness"]))
        cell = max(spacing * 0.85, 2.0)
        gap = 0.38 + 0.1 * min(spacing / 60.0, 1.0)

        def opened(gate):
            return np.clip((gate - gap) / 0.12, 0, 1)
        lum = F.luma(src) / 255.0
        tone = F.blur(lum, cell * 0.3)

        def ramp(v, start, full):
            return np.clip((v - start) / (full - start), 0, 1)
        xs, ys = F.grid(h, w)
        xf, yf = xs.astype(np.float32), ys.astype(np.float32)
        vn = F.value_noise
        crumple = cell * 1.3
        grain, jag = cell * 0.15, cell * 0.07
        u = xf + (vn(xf, yf, 3.0, 71) - 0.5) * 2 * 1.2 + (vn(xf, yf, grain, 76) - 0.5) * 2 * jag
        v = yf + (vn(xf, yf, 3.0, 72) - 0.5) * 2 * 1.2 + (vn(xf, yf, grain, 77) - 0.5) * 2 * jag
        noise = (0.55 * vn(u, v, crumple, 73) + 0.3 * vn(u, v, crumple * 0.45, 74)
                 + 0.15 * vn(u, v, crumple * 0.2, 75))

        def courses_at(size, seed, extra):
            s = seed * 16
            wobble = min(1.3, size * 0.08)
            uu = (xf + (vn(xf, yf, size, 61 + s) - 0.5) * 2 * 0.18 * size
                  + (vn(xf, yf, size * 0.5, 68 + s) - 0.5) * 2 * 0.12 * size
                  + (vn(xf, yf, 4.0, 62 + s) - 0.5) * 2 * wobble)
            vv = (yf + (vn(xf, yf, size, 63 + s) - 0.5) * 2 * 0.18 * size
                  + (vn(xf, yf, size * 0.5, 69 + s) - 0.5) * 2 * 0.12 * size
                  + (vn(xf, yf, 4.0, 64 + s) - 0.5) * 2 * wobble)
            sk = seed * 7919
            course = size * 0.8
            stretch = np.floor(uu / (size * 2.6)).astype(np.int64) + sk

            def boundary(k):
                return (k + (F.lattice(k, stretch, 85) - 0.5) * 2 * 0.42) * course
            k = np.floor(vv / course).astype(np.int64)
            k = np.where(vv < boundary(k), k - 1, np.where(vv >= boundary(k + 1), k + 1, k))
            dash = size * 0.7
            across = np.full((h, w), INF)
            for b, d in ((k, vv - boundary(k)), (k + 1, boundary(k + 1) - vv)):
                gate = vn(uu + b * 97.0, np.full((h, w), float(sk), np.float32), dash, 81)
                o = opened(gate - extra)
                across = np.minimum(across, _safe_div(np.abs(d), o, o > 0))
            slab = size * 1.5
            column = np.floor(uu / slab).astype(np.int64)
            down = np.full((h, w), INF)
            for dj in (-1, 0, 1):
                j = column + dj
                closed = F.lattice(j, k + sk, 82) > 0.55 - extra
                at = (j + 0.15 + 0.7 * F.lattice(j, k + sk, 65)) * slab
                down = np.where(closed, down, np.minimum(down, np.abs(uu - at)))
            return np.minimum(across, down)
        light = ramp(tone, 0.25, 0.55)
        dark = 1 - ramp(tone, 0.3, 0.72)
        both = np.minimum(courses_at(cell, 0, 0.0), courses_at(cell * 0.6, 1, 0.22) * 1.8)
        courses = _safe_div(both, light, light > 0)
        gx, gy = F.gradient(noise)
        slope = np.maximum(np.sqrt(gx * gx + gy * gy), 1e-4)
        level = noise / 0.1
        contour = np.abs(level - np.round(level)) * 0.1 / slope
        along = opened(vn(xf, yf, cell * 0.7, 83) + 0.12)
        crumpled = _safe_div(contour, along * dark, (along > 0) & (dark > 0))
        crack = np.where(dark > 0, np.minimum(courses, crumpled), courses)
        deep = 0.45 + 1.1 * vn(xf, yf, cell * 0.6, 91)
        groove = deep * (1 - F.smooth((crack - 0.8) / 1.1))
        pocket_size = max(cell * 0.22, 2.5)
        n = 0.65 * vn(xf, yf, pocket_size, 92) + 0.35 * vn(xf, yf, pocket_size * 0.5, 93)
        pocket = F.smooth((n - 0.66) / 0.08) * 0.8 * dark
        surface = F.blur(1 - np.maximum(groove, pocket) + lum * 0.9, 0.5)
        lx, ly = F.towards("top_left")
        shade_gain = depth / 10.0 * 1.6
        dx, dy = F.gradient(surface)
        shade = -(dx * lx + dy * ly) * shade_gain * (1 - (1 - 0.28) * lum)
        shadow = _cast_shadow(surface, (lx, ly), 4, 0.22)
        shadow = 1 - 0.55 * np.clip(shadow * shade_gain / 1.6, 0, 1)
        floor = 0.15 + brightness * 0.095
        opening = 1 - F.smooth(crack / 1.0)
        tone_k = (1 + (floor - 1) * opening) * shadow
        rgb = _shade(src[..., :3].astype(np.float32), shade) * tone_k[..., None]
        return F.put_rgb(src, rgb)


GRAIN_TYPES = ("regular", "soft", "sprinkles", "clumped", "contrasty", "enlarged", "stippled",
               "horizontal", "vertical", "speckle")
# clump, strength, contrast, boost
_COLOUR_GRAIN = {"regular": (0.0, 1.0, 1.0, 1.0), "soft": (0.7, 0.6, 1.0, 1.0),
                 "clumped": (2.0, 0.5, 1.6, 1.4), "contrasty": (1.6, 0.4, 2.0, 1.65),
                 "enlarged": (2.4, 0.55, 1.3, 1.15)}


def _grain_noise(xs, ys, salt, seed):
    return F.noise(xs + salt * 7919, ys - salt * 104729, seed) * 2 - 1


class Grain(_Texture):
    """Ten kinds of grain: colour noise of several clump sizes, sprinkles,
    stipple, horizontal or vertical streaks, or speckle."""
    name, label = "grain", "Grain"

    def declare(self):
        return _props(IntProperty("intensity", 40, 0, 100), IntProperty("contrast", 50, 0, 100),
                      ChoiceProperty("grain_type", "regular", list(GRAIN_TYPES),
                                     labels={k: k.title() for k in GRAIN_TYPES}),
                      ColorProperty("foreground", BLACK), ColorProperty("background", WHITE),
                      _seed())

    def compute(self, t, src):
        intensity, contrast, kind, seed = (float(t["intensity"]), float(t["contrast"]),
                                           t["grain_type"], t["seed"])
        if intensity == 0:
            return src.copy()
        h, w = src.shape[:2]
        spread = intensity * 0.8
        share = intensity / 100.0
        colour = _COLOUR_GRAIN.get(kind)
        push = colour[2] if colour else 1.0
        boost = 1.25 if kind in ("horizontal", "vertical") else (colour[3] if colour else 1.0)
        gain = boost * 2.0 ** ((contrast - 50.0) / 40.0 * push)

        def stretch(v):
            return np.clip((v - 128.0) * gain + 128.0, 0, 255)
        xs, ys = F.grid(h, w)
        px = src[..., :3].astype(np.float32)
        lum = F.luma(px)
        ink = np.asarray(t["foreground"][:3], np.float32)
        paper = np.asarray(t["background"][:3], np.float32)
        if colour:
            clump, strength = colour[0], colour[1]
            out = np.empty_like(px)
            for c in range(3):
                field = 0.55 * _grain_noise(xs, ys, 10, seed) \
                    + 0.45 * _grain_noise(xs, ys, 11 + c, seed)
                if clump > 0:
                    field = F.blur(field, clump)
                out[..., c] = stretch(px[..., c] + F.unit_spread(field) * strength * spread)
        elif kind == "sprinkles":
            hit = (_grain_noise(xs, ys, 31, seed) + 1) * 0.5 < share * 0.7
            lifted = stretch(px + (255 - px) * share * 0.6)
            out = np.where(hit[..., None], paper, lifted)
        elif kind == "speckle":
            around = F.blur(lum, 2.0)
            on_mesh = (xs % 4 == 0) | (ys % 4 == 0)
            cut = share * (95.0 + np.where(on_mesh, 125.0, 0.0)
                           + _grain_noise(xs, ys, 41, seed) * 55.0)
            darker = np.maximum(around - lum, 0) * 2.5
            lifted = lum + (255 - lum) * share * 0.7
            rich = 1 + share * 0.5
            scale = np.minimum(lifted / np.maximum(lum, 1.0), 3.0)
            v = stretch(lifted[..., None] + (px - lum[..., None]) * (scale * rich)[..., None])
            out = np.where((lum - darker < cut)[..., None], ink, v)
        elif kind == "stippled":
            shaken = stretch(lum) + _grain_noise(xs, ys, 51, seed) * spread * 2.2
            out = np.where((shaken < 128)[..., None], ink, paper)
        else:                                       # horizontal / vertical streaks
            run, band = (xs, ys) if kind == "horizontal" else (ys, xs)
            band = band.astype(np.float32)
            row = F.value_noise(np.zeros_like(band), band, 1.8, 61, seed) * 2 - 1
            fade = F.value_noise(run.astype(np.float32), band * 37.0, 240.0, 62, seed)
            fade = 0.3 + 0.7 * fade
            s = row * fade + _grain_noise(xs, ys, 63, seed) * 0.5
            s = np.where(s < 0, s, s * 0.35)
            out = stretch(px + (s * spread * 2.2)[..., None])
        return F.put_rgb(src, out)


class MosaicTiles(_Texture):
    """Small irregular tiles with grout between, a few joints missing, lit
    from the top left."""
    name, label = "mosaic_tiles", "Mosaic Tiles"

    def declare(self):
        return _props(IntProperty("tile_size", 12, 2, 100), IntProperty("grout_width", 3, 1, 15),
                      IntProperty("lighten_grout", 1, 0, 10), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        size, grout, lighten, seed = (float(t["tile_size"]), float(t["grout_width"]),
                                      float(t["lighten_grout"]), t["seed"])
        jag = 2.4 + size * 0.07
        grain = max(size * 0.2, 6.0)
        half_grout = min(max(grout * 0.55 - 0.75, 0.5), size * 0.3)
        bevel = max(min(2.0, size * 0.06), 0.7)
        lum = F.luma(src) / 255.0
        xs, ys = F.grid(h, w)
        xf, yf = xs.astype(np.float32), ys.astype(np.float32)
        u = xf + (F.value_noise(xf, yf, grain, 101, seed) - 0.5) * 2 * jag
        v = yf + (F.value_noise(xf, yf, grain, 102, seed) - 0.5) * 2 * jag
        col, row = _round(u / size), _round(v / size)
        cell_x, cell_y = np.floor(u / size).astype(np.int64), np.floor(v / size).astype(np.int64)
        across = np.where(F.lattice(col.astype(np.int64), cell_y, 103, seed) < 0.2, INF,
                          np.abs(u - col * size))
        down = np.where(F.lattice(cell_x, row.astype(np.int64), 104, seed) < 0.2, INF,
                        np.abs(v - row * size))
        inside = np.minimum(across, down) - half_grout
        surface = F.blur(F.smooth(inside / bevel) * 0.55 + lum * 1.6, 0.4)
        lx, ly = F.towards("top_left")
        dx, dy = F.gradient(surface)
        shade = -(dx * lx + dy * ly) * 1.3
        shadow = 1 - 0.45 * np.clip(_cast_shadow(surface, (lx, ly), 3), 0, 1)
        tile = F.smooth(inside + 0.5)[..., None]
        grey = 90.0 + lighten * 14.0
        cover = 0.25 + lighten * 0.05
        own = src[..., :3].astype(np.float32)
        grouted = own + (grey - own) * cover
        base = grouted + (own - grouted) * tile
        return F.put_rgb(src, _shade(base, shade) * shadow[..., None])


class Patchwork(_Texture):
    """The picture as raised squares, each its own average colour, set at
    heights by brightness and lit from above."""
    name, label = "patchwork", "Patchwork"

    def declare(self):
        return _props(IntProperty("square_size", 5, 0, 10), IntProperty("relief", 8, 0, 25),
                      _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        n = int(t["square_size"]) + 5
        relief = math.sqrt(t["relief"] / 8.0)
        cols, rows = -(-w // n), -(-h // n)
        pad = np.pad(src[..., :3].astype(np.float64), ((0, rows * n - h), (0, cols * n - w), (0, 0)))
        cnt = np.pad(np.ones((h, w)), ((0, rows * n - h), (0, cols * n - w)))
        sums = pad.reshape(rows, n, cols, n, 3).sum(axis=(1, 3))
        counts = cnt.reshape(rows, n, cols, n).sum(axis=(1, 3))
        mean = sums / np.maximum(counts, 1)[..., None]
        lum = (0.299 * mean[..., 0] + 0.587 * mean[..., 1] + 0.114 * mean[..., 2]) / 255.0
        gain = np.where(lum > 0, np.maximum(lum, 1e-9) ** 0.65 / np.maximum(lum, 1e-9), 1.0)
        cy, cx = np.mgrid[0:rows, 0:cols]
        vary = (F.lattice(cx, cy, 111, t["seed"]) - 0.5) * 2 * 0.02 * 255.0 * min(relief, 1.5)
        face = np.clip(mean * gain[..., None] + vary[..., None], 0, 255).astype(np.float32)
        size = float(n)
        rnd = min(max(size * 0.32, 1.5), size * 0.5)

        def surface(x, y):
            x = np.clip(x, 0.0, w - 0.001)
            y = np.clip(y, 0.0, h - 0.001)
            ix, iy = (x / size).astype(np.intp), (y / size).astype(np.intp)
            base = lum[iy, ix] * 10.0

            def quarter(edge, r):
                q = 1 - np.minimum(edge / r, 1.0)
                return np.sqrt(np.maximum(1 - q * q, 0))
            uu, vv = x - ix * size, y - iy * size
            down = quarter(np.minimum(vv, size - vv), rnd)
            across = quarter(np.minimum(uu, size - uu), rnd * 0.35)
            rim = down * (1 - 0.65 * (1 - across))
            tilt = np.modf(y / size)[0] * 2.0
            return (base + rim * 1.2 + tilt) * relief
        lx, ly = -0.2, -0.98
        length = math.hypot(lx, ly)
        lx, ly = lx / length, ly / length
        elev = math.radians(28.0)
        flat, rise = math.cos(elev), math.tan(elev)
        light = (lx * flat, ly * flat, math.sin(elev))
        xs, ys = F.grid(h, w)
        fx, fy = xs + 0.5, ys + 0.5
        dx = surface(fx + 0.5, fy) - surface(fx - 0.5, fy)
        dy = surface(fx, fy + 0.5) - surface(fx, fy - 0.5)
        norm = np.sqrt(dx * dx + dy * dy + 1)
        facing = np.maximum((-dx * light[0] - dy * light[1] + light[2]) / norm, 0)
        here = surface(fx, fy)
        over = np.zeros((h, w))
        for i in range(1, 29):
            step = i * 0.5
            over = np.maximum(over, surface(fx + lx * step, fy + ly * step) - here - step * rise)
        lit = 1 - np.clip(over / 1.1, 0, 1)
        light_here = 0.38 + 0.62 * facing * lit
        slope = 2.0 * relief / size
        light_flat = 0.38 + 0.62 * ((-slope * light[1] + light[2]) / math.sqrt(slope * slope + 1))
        k = (light_here / light_flat)[..., None]
        colour = face[ys // n, xs // n]
        return F.put_rgb(src, colour * k)


class StainedGlass(_Texture):
    """Single-colour panes of a jittered hexagonal Voronoi, leaded in the
    foreground colour, lit from the middle by Light Intensity."""
    name, label = "stained_glass", "Stained Glass"

    def declare(self):
        return _props(IntProperty("cell_size", 10, 2, 50), IntProperty("border_thickness", 4, 1, 20),
                      IntProperty("light_intensity", 5, 0, 10),
                      ColorProperty("foreground", BLACK, label="Border color"), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        seed = t["seed"]
        step = 2.2 * t["cell_size"]
        row_step = step * math.sqrt(3) / 2
        half_lead = 0.5 * 0.62 * t["border_thickness"]
        cols = int(math.ceil(w / step)) + 5
        rows = int(math.ceil(h / row_step)) + 5
        xs, ys = F.grid(h, w)
        px, py = xs + 0.5, ys + 0.5
        i0 = np.floor(px / step).astype(np.int64)
        j0 = np.floor(py / row_step).astype(np.int64)

        def seed_at(i, j):
            shift = 0.5 * (j & 1)
            return ((i + shift + 0.6 * (F.lattice(i, j, 41, seed) - 0.5)) * step,
                    (j + 0.6 * (F.lattice(i, j, 42, seed) - 0.5)) * row_step)
        cands = []
        for dj in range(-2, 3):
            for di in range(-2, 3):
                i, j = i0 + di, j0 + dj
                ok = (i >= -2) & (i <= cols - 3) & (j >= -2) & (j <= rows - 3)
                sx, sy = seed_at(i, j)
                d = np.where(ok, (sx - px) ** 2 + (sy - py) ** 2, np.inf)
                cands.append((sx, sy, d, (j + 2) * cols + (i + 2)))
        best = np.full((h, w), np.inf)
        owner = np.zeros((h, w), np.int64)
        bx = np.zeros((h, w))
        by = np.zeros((h, w))
        for sx, sy, d, index in cands:
            better = d < best
            best = np.where(better, d, best)
            owner = np.where(better, index, owner)
            bx = np.where(better, sx, bx)
            by = np.where(better, sy, by)
        edge = np.minimum(np.minimum(px, w - px), np.minimum(py, h - py)).astype(np.float64)
        for sx, sy, d, index in cands:
            apart = np.sqrt((sx - bx) ** 2 + (sy - by) ** 2)
            use = np.isfinite(d) & (apart > 0)
            gap = np.where(use, (np.where(use, d, 0) - best) / (2 * np.where(use, apart, 1)), np.inf)
            edge = np.minimum(edge, gap)
        flat = owner.ravel()
        count = cols * rows
        rgb = src[..., :3].reshape(-1, 3).astype(np.float64)
        counts = np.maximum(np.bincount(flat, minlength=count), 1)
        panes = np.stack([np.bincount(flat, weights=rgb[:, c], minlength=count)
                          for c in range(3)], -1) / counts[:, None]
        pane = panes[flat].reshape(h, w, 3)
        reach = 0.26 * math.sqrt(w * h)
        intensity = min(t["light_intensity"], 10) / 10.0
        peak = min(1.75 * intensity * intensity, 1.0)
        glow = peak * np.exp(-((px - w / 2.0) ** 2 + (py - h / 2.0) ** 2) / (reach * reach))
        lit = pane + (255 - pane) * glow[..., None]
        glass = np.clip(edge - half_lead + 0.5, 0, 1)[..., None]
        lead = np.asarray(t["foreground"][:3], np.float64)
        return F.put_rgb(src, lead + (lit - lead) * glass)


_TEXTURIZER = {"brick": dict(gain=135.0, crisp=2.2, occlusion=60.0), "burlap": dict(gain=13.0),
               "canvas": dict(gain=20.0), "sandstone": dict(gain=15.0)}


class Texturizer(_Texture):
    """A surface -- Brick, Burlap, Canvas or Sandstone -- pressed into the
    picture and lit, more strongly in the darks."""
    name, label = "texturizer", "Texturizer"

    def declare(self):
        return _props(_texture(), IntProperty("scaling", 100, 50, 200),
                      IntProperty("relief", 4, 0, 50), _light(label="Light Direction"),
                      BoolProperty("invert", False))

    def compute(self, t, src):
        finish = dict(_TEXTURIZER.get(t["texture"], {}))
        rgb = F.relief(src[..., :3].astype(np.float32), t["texture"], t["scaling"], t["relief"],
                       t["light"], t["invert"], dark_bias=0.6, **finish)
        return F.put_rgb(src, rgb)


TEXTURE = (Craquelure, Grain, MosaicTiles, Patchwork, StainedGlass, Texturizer)
