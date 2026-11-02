# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop's Filter Gallery > Sketch family, all fourteen.

What binds the family together is that it paints in TWO SWATCHES, not in
the picture's own colors: every one of these but Chrome and Water Paper
throws the hue away, works on brightness alone, and maps the answer
between a foreground and a background color (black and white by default).

Algorithms, constants and slider ranges are ported from Kooka Pictura's
photorust engine (crates/pictura-filters/src/photorust/sketch, GPL-3.0),
whose sliders follow CS6's. The shared machinery is in fields.py. Alpha is
never touched: a sketch keeps the layer's shape.
"""

import math

import numpy as np

from . import fields as F
from .filters import _Filter
from .props import (BoolProperty, ChoiceProperty, ColorProperty, IntProperty,
                    PropertyCollection)

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)


def _props(*items):
    return PropertyCollection(list(items))


def _swatches():
    return (ColorProperty("foreground", BLACK), ColorProperty("background", WHITE))


def _seed():
    return IntProperty("seed", 1, 0, 999)


def _light(default="bottom", name="light", label="Light"):
    return ChoiceProperty(name, default, list(F.LIGHTS), labels=F.LIGHT_LABELS, label=label)


def _direction(name="direction", label="Stroke Direction"):
    return ChoiceProperty(name, "right_diagonal", list(F.DIRECTIONS),
                          labels=F.DIRECTION_LABELS, label=label)


def _texture(default="canvas"):
    return ChoiceProperty("texture", default, list(F.TEXTURES), labels=F.TEXTURE_LABELS)


def _tone(src):
    """Brightness 0..1 of an RGBA picture."""
    return F.luma(src) / 255.0


class _Sketch(_Filter):
    category = "Sketch"


class BasRelief(_Sketch):
    """The picture carved in shallow relief and lit from one side: the
    brightness, softened by Smoothness, is read as a surface and its slope
    toward the light drives the two swatches."""
    name, label = "bas_relief", "Bas Relief"

    def declare(self):
        return _props(IntProperty("detail", 6, 1, 15), IntProperty("smoothness", 3, 1, 15),
                      _light(), *_swatches())

    def compute(self, t, src):
        h, w = src.shape[:2]
        surface = F.blur(F.luma(src), t["smoothness"] * 0.38)
        lx, ly = F.towards(t["light"])
        xs, ys = F.grid(h, w)

        def at(ox, oy):
            return F.displace_nearest(surface, xs + ox, ys + oy)
        slope = (at(-lx, -ly) - at(lx, ly)) / 255.0
        lit = np.clip(0.5 + slope * t["detail"] * 0.85, 0, 1)
        return F.put_rgb(src, F.swatches(t["background"], t["foreground"], lit))


class ChalkCharcoal(_Sketch):
    """Shadows in charcoal strokes one way, highlights in chalk strokes the
    other, mid-gray paper between."""
    name, label = "chalk_charcoal", "Chalk & Charcoal"

    def declare(self):
        return _props(IntProperty("charcoal_area", 6, 0, 20), IntProperty("chalk_area", 6, 0, 20),
                      IntProperty("pressure", 1, 0, 5, label="Stroke Pressure"),
                      *_swatches(), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        tone = F.blur(_tone(src), 1.4)

        def marks(angle, salt):
            f = F.streaked_noise(h, w, 26.0, salt, F.along(angle), t["seed"])
            return F.unit_spread(F.blur(f, 1.1))
        soft = max(0.38 - t["pressure"] * 0.072, 0.015)

        def stick(edge, swing, below):
            against = tone + np.clip(swing, -2, 2) * 0.06
            over = edge - against if below else against - edge
            return F.smooth(over / soft * 0.5 + 0.5)
        dark = stick(0.185 + t["charcoal_area"] * 0.019, marks(-45.0, 131), True)
        light = stick(0.955 - t["chalk_area"] * 0.0125, marks(45.0, 132), False)
        laid = dark + light
        over = np.maximum(laid, 1.0)
        dark, light = dark / over, light / over
        bare = 1 - dark - light
        ink = np.asarray(t["foreground"][:3], np.float32)
        paper = np.asarray(t["background"][:3], np.float32)
        rgb = 128.0 * bare[..., None] + ink * dark[..., None] + paper * light[..., None]
        return F.put_rgb(src, rgb)


class Charcoal(_Sketch):
    """Posterized, smudged charcoal: dark masses plus the main edges, with a
    diagonal grain where the stick is half laid."""
    name, label = "charcoal", "Charcoal"

    def declare(self):
        return _props(IntProperty("thickness", 1, 1, 7), IntProperty("detail", 3, 0, 5),
                      IntProperty("balance", 50, 0, 100, label="Light/Dark Balance"),
                      *_swatches(), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        thickness, detail = float(t["thickness"]), float(t["detail"])
        tone = F.blur(F.luma(src), max(2.6 - detail * 0.44, 0.3))
        edge_from = max(180.0 - detail * 15.0, 3.0)
        edges = F.smooth((F.sobel(tone) - edge_from) / 30.0)
        edges = F.widest_nearby(edges, int(round(thickness * 0.25)))
        edges = F.blur(edges, 0.7) * 0.45
        grain = F.streaked_noise(h, w, 10.0, 141, F.along(45.0), t["seed"])
        grain = F.unit_spread(F.blur(grain, 0.8))
        mass_from = 0.10 + t["balance"] / 100.0 * 0.50
        bias = thickness / 7.0 * 0.12
        mass = F.smooth((mass_from - tone / 255.0) / 0.22 * 0.5 + 0.5)
        owed = np.clip(np.maximum(mass, edges), 0, 1)
        open_ = np.sqrt(np.clip(4 * owed * (1 - owed), 0, 1))
        coal = np.clip(owed + bias * open_ + grain * 0.24 * open_, 0, 1)
        return F.put_rgb(src, F.swatches(t["foreground"], t["background"], coal))


class Chrome(_Sketch):
    """A polished-metal surface: the softened brightness folded through a
    few cycles of a cosine, so every tone becomes a band of reflection."""
    name, label = "chrome", "Chrome"

    def declare(self):
        return _props(IntProperty("detail", 4, 0, 10), IntProperty("smoothness", 7, 0, 10))

    def compute(self, t, src):
        detail, smoothness = float(t["detail"]), float(t["smoothness"])
        tone = F.luma(src)
        melt = 1.2 + smoothness * 1.6
        melted = F.blur(tone, melt)
        broad = F.blur(tone, melt * 3.0)
        keep = 0.2 + detail / 10.0 * 0.8
        surface = broad + (melted - broad) * keep
        flat = np.sort(surface, axis=None)
        n = flat.size - 1
        low, high = flat[int(n * 0.005)], flat[int(n * 0.995)]
        span = max(float(high - low), 1.0)
        pulled = np.clip((surface - low) / span, 0, 1) * 255.0
        surface = surface + (pulled - surface) * 0.5
        h, w = surface.shape
        xs, ys = F.grid(h, w)
        lit = (F.displace_nearest(surface, xs - 1, ys - 1)
               - F.displace_nearest(surface, xs + 1, ys + 1)) / 255.0
        cycles = 3.5 + detail * 0.7
        level = np.clip(surface / 255.0, 0, 1)
        phase = level * cycles + np.clip(lit, -1, 1) * 0.10
        wave = 0.5 - 0.5 * np.cos(phase * 2 * math.pi)
        polished = 1 - (1 - wave) ** 3.5
        v = polished * 255.0
        return F.put_rgb(src, np.repeat(v[..., None], 3, axis=-1))


class ConteCrayon(_Sketch):
    """Dense crayon: the tones squeezed between foreground and background
    levels, then the paper's texture pressed into it."""
    name, label = "conte_crayon", "Conte Crayon"

    def declare(self):
        return _props(IntProperty("foreground_level", 8, 1, 15),
                      IntProperty("background_level", 7, 1, 15),
                      _texture(), IntProperty("scaling", 100, 50, 200, label="Texture Scaling"),
                      IntProperty("relief", 4, 0, 50, label="Texture Relief"),
                      _light(name="texture_light", label="Texture Light"),
                      BoolProperty("invert", False, label="Invert Texture"),
                      *_swatches(), _seed())

    def compute(self, t, src):
        fore = (t["foreground_level"] - 1) / 14.0
        back = (t["background_level"] - 1) / 14.0
        tones = F.blur(_tone(src), 1.1)
        black, white = 0.28 * fore, 1 - 0.45 * back
        span = max(white - black, 0.05)
        laid = np.clip((white - tones) / span, 0, 1)
        rgb = F.swatches(t["foreground"], t["background"], laid)
        rgb = F.relief(rgb, t["texture"], t["scaling"], t["relief"], t["texture_light"],
                       t["invert"], seed=t["seed"], crisp=0.35, occlusion=1.3)
        return F.put_rgb(src, rgb)


class GraphicPen(_Sketch):
    """Fine straight ink strokes, all one way: each pixel is ink or paper,
    by its darkness against a streaked threshold."""
    name, label = "graphic_pen", "Graphic Pen"

    def declare(self):
        return _props(IntProperty("stroke_length", 6, 1, 15),
                      IntProperty("balance", 50, 0, 100, label="Light/Dark Balance"),
                      _direction(), *_swatches())

    def compute(self, t, src):
        h, w = src.shape[:2]
        streak = max(1.0 + t["stroke_length"] * 1.6, 1.0)
        hatch = F.streaked_noise(h, w, streak, 161, F.along(t["direction"]))
        shift = (t["balance"] / 100.0 - 0.5) * 0.9
        coverage = np.clip((1 - _tone(src) + shift) * 1.5, 0, 1)
        inked = (coverage > 0.5 + hatch * 0.22).astype(np.float32)
        return F.put_rgb(src, F.swatches(t["foreground"], t["background"], inked))


HALFTONE_PATTERNS = ("dot", "line", "circle")


class HalftonePattern(_Sketch):
    """A halftone screen of dots, lines or rings, in the two swatches."""
    name, label = "halftone_pattern", "Halftone Pattern"

    def declare(self):
        return _props(IntProperty("size", 5, 1, 12), IntProperty("contrast", 5, 0, 50),
                      ChoiceProperty("pattern", "dot", list(HALFTONE_PATTERNS),
                                     labels={"dot": "Dot", "line": "Line", "circle": "Circle"},
                                     label="Pattern Type"),
                      *_swatches())

    def compute(self, t, src):
        h, w = src.shape[:2]
        size = float(t["size"])
        gain = 2.0 ** (t["contrast"] / 5.0)
        pattern = t["pattern"]
        cell = size if pattern == "dot" else max(size, 3.0)
        rung = 0.25 if pattern == "dot" else 0.5 / cell
        xs, ys = F.grid(h, w)
        xs, ys = xs.astype(np.float32), ys.astype(np.float32)
        if pattern == "dot":
            screen = 0.5 - 0.5 * np.cos(math.pi * xs / cell) * np.cos(math.pi * ys / cell)
        elif pattern == "line":
            fy = np.modf((ys + 0.5) / cell)[0] - 0.5
            screen = np.abs(fy) * 2
        else:
            dist = np.hypot(xs - w / 2.0, ys - h / 2.0) / cell
            screen = np.abs(np.modf(dist)[0] - 0.5) * 2
        screen = rung + np.clip(screen, 0, 1) * (1 - 2 * rung)
        coverage = np.clip((1 - _tone(src) - screen) * gain + 0.5, 0, 1)
        return F.put_rgb(src, F.swatches(t["foreground"], t["background"], coverage))


class NotePaper(_Sketch):
    """Embossed handmade paper: the darks cut out of the sheet as a lower
    layer, the fibres raised, all lit from the top left."""
    name, label = "note_paper", "Note Paper"

    def declare(self):
        return _props(IntProperty("balance", 25, 0, 50, label="Image Balance"),
                      IntProperty("graininess", 10, 0, 20), IntProperty("relief", 11, 0, 25),
                      *_swatches(), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        tone = F.blur(_tone(src), 1.0)
        paper = np.clip((tone - t["balance"] / 50.0) / 0.04 + 0.5, 0, 1)
        surface = F.blur(paper, 0.9)
        if t["graininess"] > 0:
            xs, ys = F.grid(h, w)
            fibre = F.blur(F.noise(xs, ys + 7919, t["seed"]) - 0.5, 0.8)
            p = np.pad(fibre, ((0, 0), (1, 1)), mode="edge")
            drawn = F.unit_spread((p[:, :-2] + fibre + p[:, 2:]) / 3.0)
            surface = surface + drawn * t["graininess"] * 0.007
        steep = t["relief"] * 0.18
        lx, ly, lz = -0.3, -1.0, 1.2
        n = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx / n, ly / n, lz / n
        gx, gy = _central(surface)
        nx, ny = -gx * steep, -gy * steep
        tilt = np.sqrt(nx * nx + ny * ny + 1)
        lit = np.maximum((nx * lx + ny * ly + lz) / tilt / lz, 0) / tilt ** 3.0
        fg = np.asarray(t["foreground"][:3], np.float32)
        bg = np.asarray(t["background"][:3], np.float32)
        hole = bg + (fg - bg) * 0.33
        base = hole + (bg - hole) * paper[..., None]
        lit = lit[..., None]
        rgb = np.where(lit <= 1, base * lit, base + (255 - base) * (lit - 1))
        return F.put_rgb(src, rgb)


def _central(field):
    """Half the difference of the two neighbors, clamped at the border."""
    p = np.pad(field, 1, mode="edge")
    return (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5, (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5


class Photocopy(_Sketch):
    """Toner wherever a pixel is darker than its surroundings, paper
    elsewhere: edges and dark masses, as a copier gives them."""
    name, label = "photocopy", "Photocopy"

    def declare(self):
        return _props(IntProperty("detail", 5, 0, 24), IntProperty("darkness", 20, 1, 50),
                      *_swatches())

    def compute(self, t, src):
        tone = F.luma(src)
        around = F.blur(tone, 0.6 + t["detail"] * 1.2)
        below = np.maximum(around - tone - 1.5, 0)
        ink = np.minimum(below * t["darkness"] * 1.1 / 255.0, 1)
        return F.put_rgb(src, F.swatches(t["foreground"], t["background"], ink))


class Plaster(_Sketch):
    """The darks raised as glossy molded plaster, lit from one side, over a
    ramp between the swatches."""
    name, label = "plaster", "Plaster"

    def declare(self):
        return _props(IntProperty("balance", 25, 0, 50, label="Image Balance"),
                      IntProperty("smoothness", 2, 0, 15), _light(), *_swatches())

    def compute(self, t, src):
        h, w = src.shape[:2]
        smoothness = float(t["smoothness"])
        tone = F.blur(_tone(src), 0.6 + smoothness * 0.2)
        plaster = (tone >= t["balance"] / 50.0).astype(np.float32)
        edge = F.blur(plaster, 1.0)
        bevel = F.blur(plaster, 4.0 + smoothness * 0.08)
        height = F.smooth(2 * (1 - bevel))
        lx, ly = F.towards(t["light"])
        ax, ay = -lx, -ly
        corners = [0.0, ax * (w - 1), ay * (h - 1), ax * (w - 1) + ay * (h - 1)]
        near, far = min(corners), max(corners)
        span = max(far - near, 1.0)
        n = math.sqrt(lx * lx + ly * ly + 0.49)
        lamp = (lx / n, ly / n, 0.7 / n)
        hx, hy, hz = lamp[0], lamp[1], lamp[2] + 1
        n = math.sqrt(hx * hx + hy * hy + hz * hz)
        half = (hx / n, hy / n, hz / n)
        flat_sheen = half[2] ** 8.0
        p = np.clip((0.5 - edge) / 0.35 + 0.5, 0, 1)
        gx, gy = _central(height)
        nx, ny, nz = -gx * 14.0, -gy * 14.0, np.ones_like(gx)
        norm = np.sqrt(nx * nx + ny * ny + 1)
        nx, ny, nz = nx / norm, ny / norm, nz / norm
        diffuse = np.maximum(nx * lamp[0] + ny * lamp[1] + nz * lamp[2], 0) / lamp[2]
        shade = np.minimum(np.maximum(1 - diffuse, 0) * 0.6, 1)
        lift = np.minimum(np.maximum(diffuse - 1, 0) * 2.5 + (1 - nz) * 0.6, 1)
        sheen = np.maximum(nx * half[0] + ny * half[1] + nz * half[2], 0) ** 8.0
        gloss = np.minimum(np.maximum(sheen - flat_sheen, 0) / (1 - flat_sheen) * 1.4, 1) \
            * (1 - p * (1 - 0.35))
        echo = np.exp(-((bevel - 0.25) / 0.06) ** 2) * 0.3 * np.minimum(bevel * 4, 1) * p
        xs, ys = F.grid(h, w)
        pos = ((xs * ax + ys * ay) - near) / span
        ink = np.asarray(t["foreground"][:3], np.float32)
        paper = np.asarray(t["background"][:3], np.float32)
        ramp = paper + (ink - paper) * pos[..., None]
        v = ramp + (ink - ramp) * p[..., None]
        v = v + (ramp - v) * echo[..., None]
        v = v + (ink - v) * shade[..., None]
        v = v + (paper - v) * lift[..., None]
        v = v + (paper - v) * gloss[..., None]
        return F.put_rgb(src, v)


class Reticulation(_Sketch):
    """Film emulsion's controlled shrinking: clumped grain in the shadows,
    finer in the highlights."""
    name, label = "reticulation", "Reticulation"

    def declare(self):
        return _props(IntProperty("density", 13, 0, 50),
                      IntProperty("black_level", 10, 0, 50),
                      IntProperty("white_level", 40, 0, 50), *_swatches(), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        density, fl, bl = float(t["density"]), float(t["black_level"]), float(t["white_level"])
        black = 0.051 + fl * 0.0086
        white = 1.127 - bl * 0.0085
        bend = 0.619 + fl * 0.0542
        lift = density * 0.004
        xs, ys = F.grid(h, w)
        noise = F.noise(xs, ys + 104729, t["seed"]) - 0.5
        clump = 1.15 - density * 0.006
        grain = F.unit_spread(F.blur(noise, clump) - F.blur(noise, clump * 2))
        amount = 0.167 + density * 0.0032
        span = max(white - black, 1e-3)
        curved = np.clip((_tone(src) - black) / span, 0, 1) ** bend
        curved = lift + (1 - lift) * curved
        shaken = np.clip(curved + grain * amount, 0, 1)
        light = 0.068 + (0.938 - 0.068) * shaken
        return F.put_rgb(src, F.swatches(t["background"], t["foreground"], light))


class Stamp(_Sketch):
    """A rubber or wooden stamp: local contrast boosted, softened, cut at
    one level into the two swatches."""
    name, label = "stamp", "Stamp"

    def declare(self):
        return _props(IntProperty("balance", 25, 0, 50, label="Light/Dark Balance"),
                      IntProperty("smoothness", 5, 1, 50), *_swatches())

    def compute(self, t, src):
        tone = _tone(src)
        tone = tone + (tone - F.blur(tone, 6.0)) * 2.0
        tone = F.blur(tone, t["smoothness"] * 0.35)
        cut = 0.225 + t["balance"] * 0.0083
        light = np.clip((tone - cut) / 0.015 + 0.5, 0, 1)
        return F.put_rgb(src, F.swatches(t["background"], t["foreground"], light))


class TornEdges(_Sketch):
    """Ragged torn paper: the dark mask's edge roughened by grain."""
    name, label = "torn_edges", "Torn Edges"

    def declare(self):
        return _props(IntProperty("balance", 25, 0, 50, label="Image Balance"),
                      IntProperty("smoothness", 1, 1, 15), IntProperty("contrast", 8, 1, 25),
                      *_swatches())

    def compute(self, t, src):
        h, w = src.shape[:2]
        contrast = float(t["contrast"])
        cut = 0.06 + t["balance"] * 0.018
        mask = F.blur((_tone(src) < cut).astype(np.float32), 18.0 / t["smoothness"])
        xs, ys = F.grid(h, w)
        grain = F.unit_spread(F.blur(F.noise(xs, ys + 15485) - 0.5, 0.7))
        grain = np.clip(0.5 + grain * 0.2887, 0, 1)
        top = (contrast - 1) / 24.0
        hole = 1.2 * top ** 8
        gain = 2.0 + contrast * 0.6
        edge = np.sqrt(np.maximum(4 * mask * (1 - mask), 0))
        torn = mask + (grain - 0.5) * 0.8 * edge - grain * hole
        inked = np.clip((torn - 0.5) * gain + 0.5, 0, 1)
        return F.put_rgb(src, F.swatches(t["foreground"], t["background"], inked))


class WaterPaper(_Sketch):
    """Blotchy daubs on damp fibrous paper: color bleeds along the fibres,
    which show darker where the paint is thin."""
    name, label = "water_paper", "Water Paper"

    def declare(self):
        return _props(IntProperty("fiber_length", 15, 3, 50), IntProperty("brightness", 45, 0, 100),
                      IntProperty("contrast", 60, 0, 100), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        run = t["fiber_length"] * 0.9
        reach = int(max(1.0, round(run / 2.0)))
        down = F.streaked_noise(h, w, run, 31, (0.0, 1.0), t["seed"])
        across = F.streaked_noise(h, w, run, 32, (1.0, 0.0), t["seed"])
        wet_down = np.exp(0.9 * down)
        wet_across = np.exp(0.9 * across)
        rgb = src[..., :3].astype(np.float32)
        total = np.zeros(rgb.shape, np.float32)
        weight = np.zeros((h, w), np.float32)
        # Bleed: a weighted mean along the column and the row through each
        # pixel, the weights being how wet the fibres there are.
        wd = np.pad(wet_down, ((reach, reach), (0, 0)))
        wa = np.pad(wet_across, ((0, 0), (reach, reach)))
        rd = np.pad(rgb * wet_down[..., None], ((reach, reach), (0, 0), (0, 0)))
        ra = np.pad(rgb * wet_across[..., None], ((0, 0), (reach, reach), (0, 0)))
        for k in range(2 * reach + 1):
            total += rd[k:k + h] + ra[:, k:k + w]
            weight += wd[k:k + h] + wa[:, k:k + w]
        bled = total / weight[..., None]
        tone = F.luma(bled) / 255.0
        weave = np.maximum(down, across) * 18.0 * (1 - tone)
        lift = 2.0 ** ((t["brightness"] - 50.0) / 30.0)
        gain = 2.0 ** ((t["contrast"] - 50.0) / 30.0)
        v = np.clip((bled - weave[..., None]) / 255.0, 0, 1)
        v = v * lift if lift < 1 else v ** (1.0 / lift)
        v = (v - 0.5) * gain + 0.5
        room = 1 - 0.7
        v = np.where(v > 0.7, 1 - room * np.exp(-np.maximum(v - 0.7, 0) / room), v)
        return F.put_rgb(src, v * 255.0)


SKETCH = (BasRelief, ChalkCharcoal, Charcoal, Chrome, ConteCrayon, GraphicPen,
          HalftonePattern, NotePaper, Photocopy, Plaster, Reticulation, Stamp, TornEdges,
          WaterPaper)
