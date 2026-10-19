# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Colour adjustments: GIMP's Colors menu and Photoshop's Adjustments.

Curves, Color Balance, Exposure, Vibrance, Channel Mixer, Photo Filter,
Gradient Map, Black & White, Colorize, Color Temperature, Threshold,
Selective Replace Color, Color to Alpha -- and the automatic ones, Auto
Levels, Auto Contrast, Auto Color and Equalize.

Every one here is pointwise (a pixel's result depends on that pixel alone),
so each is also offered as an adjustment layer -- except the automatic
ones, which read statistics from the WHOLE image and would compute
something different on every tile the compositor hands them. They declare
`global_stats` and stay menu commands.

Colours are averaged and mixed in straight RGB here because nothing below
averages across pixels; alpha passes through untouched, as with every
adjustment.
"""

import math

import numpy as np

from .effects import Adjustment
from .props import (BoolProperty, ChoiceProperty, ColorProperty, CurveProperty,
                    FloatProperty, IntProperty, PropertyCollection)


def _lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _out(src, dst, rect, rgb):
    target = dst[rect.slice()]
    target[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    target[..., 3] = src[rect.slice()][..., 3]


def monotone_curve(points):
    """256-entry LUT through control points: monotone cubic (Fritsch-Carlson).

    Monotone so a curve never overshoots between points -- the classic
    defect of a plain spline, where pulling one point up makes the curve
    dip below zero beside it and the shadows posterise.
    """
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    n = len(x)
    if n == 2:
        xs = np.arange(256, dtype=np.float64)
        return np.clip(np.rint(np.interp(xs, x, y)), 0, 255).astype(np.uint8)
    d = np.diff(y) / np.diff(x)
    m = np.empty(n)
    m[0], m[-1] = d[0], d[-1]
    m[1:-1] = (d[:-1] + d[1:]) / 2.0
    for k in range(n - 1):
        if d[k] == 0:
            m[k] = m[k + 1] = 0.0
            continue
        a, b = m[k] / d[k], m[k + 1] / d[k]
        s = a * a + b * b
        if s > 9:
            t = 3.0 / math.sqrt(s)
            m[k], m[k + 1] = t * a * d[k], t * b * d[k]
    xs = np.arange(256, dtype=np.float64)
    k = np.clip(np.searchsorted(x, xs, side="right") - 1, 0, n - 2)
    h = x[k + 1] - x[k]
    t = (xs - x[k]) / h
    t2, t3 = t * t, t * t * t
    out = ((2 * t3 - 3 * t2 + 1) * y[k] + (t3 - 2 * t2 + t) * h * m[k]
           + (-2 * t3 + 3 * t2) * y[k + 1] + (t3 - t2) * h * m[k + 1])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class Curves(Adjustment):
    """Tone curves: one for all channels, then one per channel on top."""

    name = "curves"
    label = "Curves"

    def declare(self):
        return PropertyCollection([CurveProperty("rgb", label="RGB"),
                                   CurveProperty("red"), CurveProperty("green"),
                                   CurveProperty("blue")])

    def build_luts(self, token):
        master = monotone_curve(token["rgb"])
        return tuple(monotone_curve(token[c])[master] for c in ("red", "green", "blue"))


class ColorBalance(Adjustment):
    """Shift colour separately in the shadows, midtones and highlights (GIMP's
    algorithm: each range's shift weighted by how much a level belongs to it)."""

    name = "color_balance"
    label = "Color Balance"

    def declare(self):
        props = []
        for rng in ("shadows", "midtones", "highlights"):
            for axis in ("cyan_red", "magenta_green", "yellow_blue"):
                props.append(IntProperty("%s_%s" % (rng, axis), 0, -100, 100))
        props.append(BoolProperty("preserve_luminosity", True))
        return PropertyCollection(props)

    def prepare(self, token, source):
        x = np.arange(256, dtype=np.float64)
        a, b, scale = 64.0, 85.0, 1.785
        low = np.clip((x - b) / -a + 0.5, 0, 1) * scale
        mid = np.clip((x - b) / a + 0.5, 0, 1) * np.clip((x + b - 255) / -a + 0.5, 0, 1) * scale
        high = np.clip((x + b - 255) / a + 0.5, 0, 1) * scale
        weights = {"shadows": low, "midtones": mid, "highlights": high}
        self._luts = []
        for axis in ("cyan_red", "magenta_green", "yellow_blue"):
            shift = sum(weights[r] * token["%s_%s" % (r, axis)] for r in weights)
            self._luts.append(np.clip(x + shift, 0, 255))
        self._preserve = token["preserve_luminosity"]

    def render_tile(self, src, dst, rect):
        win = src[rect.slice()]
        rgb = np.stack([self._luts[c][win[..., c]] for c in range(3)], axis=-1)
        if self._preserve:
            # GIMP's way: the shifted colour's hue and saturation, the
            # original's HSL lightness. Adding a flat offset instead clips
            # near black and white and does not preserve anything.
            rgb = _with_lightness(rgb, _hsl_lightness(win[..., :3].astype(np.float64)))
        _out(src, dst, rect, rgb)


def _hsl_lightness(rgb):
    return (rgb.max(axis=-1) + rgb.min(axis=-1)) / 2.0


def _with_lightness(rgb, light):
    """rgb with its HSL hue and saturation, at HSL lightness `light` (0..255)."""
    c = np.clip(rgb, 0, 255) / 255.0
    mx, mn = c.max(axis=-1), c.min(axis=-1)
    L = (mx + mn) / 2.0
    d = mx - mn
    S = np.where(d == 0, 0.0, d / np.where(1 - np.abs(2 * L - 1) == 0, 1, 1 - np.abs(2 * L - 1)))
    h, _d = _hue(c)
    L2 = np.clip(light / 255.0, 0, 1)
    C = (1 - np.abs(2 * L2 - 1)) * np.clip(S, 0, 1)
    hp = h / 60.0
    X = C * (1 - np.abs(hp % 2 - 1))
    z = np.zeros_like(C)
    sector = np.floor(hp).astype(np.int64) % 6
    r = np.select([sector == k for k in range(6)], [C, X, z, z, X, C])
    g = np.select([sector == k for k in range(6)], [X, C, C, X, z, z])
    b = np.select([sector == k for k in range(6)], [z, z, X, C, C, X])
    m = L2 - C / 2
    return np.stack([r + m, g + m, b + m], axis=-1) * 255.0


class Exposure(Adjustment):
    """Photoshop's Exposure: stops of light, an offset, a gamma."""

    name = "exposure"
    label = "Exposure"

    def declare(self):
        return PropertyCollection([FloatProperty("exposure", 0.0, -5.0, 5.0, decimals=2),
                                   FloatProperty("offset", 0.0, -0.5, 0.5, decimals=3),
                                   FloatProperty("gamma", 1.0, 0.1, 9.99, decimals=2)])

    def build_luts(self, token):
        x = np.arange(256, dtype=np.float64) / 255.0
        # In linear light, as a camera would: decode sRGB, scale, re-encode.
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        lin = lin * (2.0 ** token["exposure"]) + token["offset"]
        lin = np.clip(lin, 0, None) ** (1.0 / max(0.01, token["gamma"]))
        enc = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
        return np.clip(np.rint(enc * 255), 0, 255).astype(np.uint8)


class Threshold(Adjustment):
    name = "threshold"
    label = "Threshold"

    def declare(self):
        return PropertyCollection([IntProperty("level", 128, 1, 255)])

    def prepare(self, token, source):
        self._level = token["level"]

    def render_tile(self, src, dst, rect):
        win = src[rect.slice()].astype(np.float64)
        v = np.where(_lum(win) >= self._level, 255.0, 0.0)
        _out(src, dst, rect, np.repeat(v[..., None], 3, axis=-1))


class Vibrance(Adjustment):
    """Raise saturation most where it is lowest, sparing what is already vivid."""

    name = "vibrance"
    label = "Vibrance"

    def declare(self):
        return PropertyCollection([IntProperty("vibrance", 0, -100, 100),
                                   IntProperty("saturation", 0, -100, 100)])

    def prepare(self, token, source):
        self._v = token["vibrance"] / 100.0
        self._s = token["saturation"] / 100.0

    def render_tile(self, src, dst, rect):
        rgb = src[rect.slice()][..., :3].astype(np.float64)
        mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
        sat = (mx - mn) / 255.0
        amount = self._s + self._v * (1.0 - sat)
        grey = _lum(rgb)[..., None]
        _out(src, dst, rect, grey + (rgb - grey) * (1.0 + amount)[..., None])


class ChannelMixer(Adjustment):
    name = "channel_mixer"
    label = "Channel Mixer"

    def declare(self):
        props = []
        for out in ("red", "green", "blue"):
            for inp in ("red", "green", "blue"):
                props.append(IntProperty("%s_from_%s" % (out, inp), 100 if out == inp else 0,
                                         -200, 200))
        props.append(BoolProperty("monochrome", False))
        return PropertyCollection(props)

    def prepare(self, token, source):
        self._m = np.array([[token["%s_from_%s" % (o, i)] / 100.0
                             for i in ("red", "green", "blue")]
                            for o in ("red", "green", "blue")])
        self._mono = token["monochrome"]

    def render_tile(self, src, dst, rect):
        rgb = src[rect.slice()][..., :3].astype(np.float64)
        if self._mono:
            v = rgb @ self._m[0]
            out = np.repeat(v[..., None], 3, axis=-1)
        else:
            out = rgb @ self._m.T
        _out(src, dst, rect, out)


class PhotoFilter(Adjustment):
    """A coloured lens filter: multiply toward a colour, keep the lightness."""

    name = "photo_filter"
    label = "Photo Filter"

    def declare(self):
        return PropertyCollection([ColorProperty("color", (236, 138, 0, 255)),
                                   IntProperty("density", 25, 1, 100),
                                   BoolProperty("preserve_luminosity", True)])

    def prepare(self, token, source):
        self._c = np.array(token["color"][:3], np.float64)
        self._d = token["density"] / 100.0
        self._keep = token["preserve_luminosity"]

    def render_tile(self, src, dst, rect):
        rgb = src[rect.slice()][..., :3].astype(np.float64)
        tinted = rgb * self._c / 255.0
        out = rgb + (tinted - rgb) * self._d
        if self._keep:
            out = out + (_lum(rgb) - _lum(out))[..., None]
        _out(src, dst, rect, out)


class GradientMap(Adjustment):
    """Each pixel's lightness picks a colour along a two-colour gradient."""

    name = "gradient_map"
    label = "Gradient Map"

    def declare(self):
        return PropertyCollection([ColorProperty("shadows", (0, 0, 0, 255)),
                                   ColorProperty("highlights", (255, 255, 255, 255)),
                                   BoolProperty("reverse", False)])

    def prepare(self, token, source):
        a = np.array(token["shadows"][:3], np.float64)
        b = np.array(token["highlights"][:3], np.float64)
        if token["reverse"]:
            a, b = b, a
        t = np.arange(256, dtype=np.float64)[:, None] / 255.0
        self._lut = a + (b - a) * t

    def render_tile(self, src, dst, rect):
        win = src[rect.slice()].astype(np.float64)
        idx = np.clip(np.rint(_lum(win)), 0, 255).astype(np.int64)
        _out(src, dst, rect, self._lut[idx])


def _hue(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    d = mx - mn
    safe = np.where(d == 0, 1, d)
    h = np.where(mx == r, ((g - b) / safe) % 6,
                 np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60.0
    return np.where(d == 0, 0.0, h), d


class BlackAndWhite(Adjustment):
    """Photoshop's Black & White: grey, but each hue chooses how light it goes."""

    name = "black_white"
    label = "Black & White (by hue)"

    HUES = ("reds", "yellows", "greens", "cyans", "blues", "magentas")
    DEFAULTS = (40, 60, 40, 60, 20, 80)       # Photoshop's default preset

    def declare(self):
        props = [IntProperty(h, d, -200, 300) for h, d in zip(self.HUES, self.DEFAULTS)]
        props += [BoolProperty("tint", False), ColorProperty("tint_color", (225, 211, 179, 255))]
        return PropertyCollection(props)

    def prepare(self, token, source):
        self._w = np.array([token[h] for h in self.HUES], np.float64) / 100.0
        self._tint = token["tint"]
        self._tc = np.array(token["tint_color"][:3], np.float64)

    def render_tile(self, src, dst, rect):
        rgb = src[rect.slice()][..., :3].astype(np.float64)
        h, d = _hue(rgb)
        pos = h / 60.0
        i0 = np.floor(pos).astype(np.int64) % 6
        frac = pos - np.floor(pos)
        w = self._w[i0] * (1 - frac) + self._w[(i0 + 1) % 6] * frac
        grey = rgb.min(axis=-1) + d * w
        if self._tint:
            out = grey[..., None] * self._tc / 255.0 * 1.0
            out = out + (grey - _lum(out))[..., None]
        else:
            out = np.repeat(grey[..., None], 3, axis=-1)
        _out(src, dst, rect, out)


class Colorize(Adjustment):
    """GIMP's Colorize: every pixel takes one hue and saturation, keeping its
    lightness -- a duotone in one step."""

    name = "colorize"
    label = "Colorize"

    def declare(self):
        return PropertyCollection([IntProperty("hue", 30, 0, 360),
                                   IntProperty("saturation", 50, 0, 100),
                                   IntProperty("lightness", 0, -100, 100)])

    def prepare(self, token, source):
        self._h = token["hue"] / 60.0
        self._s = token["saturation"] / 100.0
        self._l = token["lightness"] / 100.0

    def render_tile(self, src, dst, rect):
        rgb = src[rect.slice()][..., :3].astype(np.float64) / 255.0
        L = _lum(rgb)
        if self._l > 0:
            L = L + (1 - L) * self._l
        else:
            L = L * (1 + self._l)
        C = (1 - np.abs(2 * L - 1)) * self._s
        X = C * (1 - abs(self._h % 2 - 1))
        sector = int(self._h) % 6
        z = np.zeros_like(L)
        order = [(C, X, z), (X, C, z), (z, C, X), (z, X, C), (X, z, C), (C, z, X)][sector]
        m = L - C / 2
        out = np.stack([o + m for o in order], axis=-1) * 255.0
        _out(src, dst, rect, out)


def _kelvin_rgb(k):
    """Approximate white point of a black body (Tanner Helland's fit)."""
    t = k / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = (99.4708025861 * math.log(t) - 161.1195681661) if t <= 66 else \
        288.1221695283 * ((t - 60) ** -0.0755148492)
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return np.clip(np.array([r, g, b]), 1, 255)


class ColorTemperature(Adjustment):
    """GIMP's Color Temperature: re-light from one white point to another."""

    name = "color_temperature"
    label = "Color Temperature"

    def declare(self):
        return PropertyCollection([IntProperty("original", 6500, 1000, 12000),
                                   IntProperty("intended", 6500, 1000, 12000)])

    def build_luts(self, token):
        gain = _kelvin_rgb(token["intended"]) / _kelvin_rgb(token["original"])
        gain = gain / gain.max() * max(1.0, gain.max()) if gain.max() > 1 else gain
        x = np.arange(256, dtype=np.float64)
        return tuple(np.clip(np.rint(x * g), 0, 255).astype(np.uint8) for g in gain)


class ReplaceColor(Adjustment):
    """Photoshop's Replace Color: shift hue / saturation / lightness of the
    pixels near one colour, softly by distance."""

    name = "replace_color"
    label = "Replace Color"

    def declare(self):
        return PropertyCollection([ColorProperty("color", (255, 0, 0, 255)),
                                   IntProperty("fuzziness", 40, 0, 200),
                                   IntProperty("hue", 0, -180, 180),
                                   IntProperty("saturation", 0, -100, 100),
                                   IntProperty("lightness", 0, -100, 100)])

    def prepare(self, token, source):
        self._c = np.array(token["color"][:3], np.float64)
        self._fuzz = max(1.0, float(token["fuzziness"]))
        self._h, self._s, self._l = token["hue"], token["saturation"] / 100.0, token["lightness"] / 100.0

    def render_tile(self, src, dst, rect):
        from .effects import _rotate_hue
        rgb = src[rect.slice()][..., :3].astype(np.float64)
        dist = np.sqrt(((rgb - self._c) ** 2).sum(axis=-1))
        w = np.clip(1.0 - dist / self._fuzz, 0, 1)[..., None]
        out = _rotate_hue(rgb, self._h) if self._h else rgb.copy()
        grey = _lum(out)[..., None]
        out = grey + (out - grey) * (1 + self._s)
        out = out + (255 - out) * self._l if self._l > 0 else out * (1 + self._l)
        _out(src, dst, rect, rgb + (out - rgb) * w)


class ColorToAlpha(Adjustment):
    """GIMP's Color to Alpha: make one colour transparent, and unmix it from
    everything partly that colour -- so a white background comes off a soft
    black drawing and leaves its antialiased edges clean, not haloed."""

    name = "color_to_alpha"
    label = "Color to Alpha"

    def declare(self):
        return PropertyCollection([ColorProperty("color", (255, 255, 255, 255))])

    def prepare(self, token, source):
        self._c = np.array(token["color"][:3], np.float64) / 255.0

    def render_tile(self, src, dst, rect):
        win = src[rect.slice()].astype(np.float64) / 255.0
        rgb, a0 = win[..., :3], win[..., 3]
        c = self._c
        # Per channel, how much of the colour must be removed to reach the
        # nearest extreme; the largest of these is the new alpha.
        up = np.where(rgb > c, (rgb - c) / np.where(c < 1, 1 - c, 1), 0)
        down = np.where(rgb < c, (c - rgb) / np.where(c > 0, c, 1), 0)
        alpha = np.clip(np.maximum(up, down).max(axis=-1), 0, 1)
        safe = np.where(alpha > 1e-6, alpha, 1)[..., None]
        out = (rgb - c) / safe + c
        target = dst[rect.slice()]
        target[..., :3] = np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)
        target[..., 3] = np.clip(np.rint(alpha * a0 * 255), 0, 255).astype(np.uint8)


# ---- the automatic ones: statistics of the whole image ----------------------------

def _alpha_hist(source, ch):
    w = source[..., 3].ravel().astype(np.float64)
    return np.bincount(source[..., ch].ravel(), weights=w, minlength=256)


def _stretch(hist, clip):
    total = hist.sum()
    if total <= 0:
        return 0, 255
    c = np.cumsum(hist) / total
    # The first level the cumulative share passes `clip` (side="right", or
    # the empty bins below the darkest pixel would count), and the first at
    # which it reaches 1 - clip.
    lo = int(np.searchsorted(c, clip, side="right"))
    hi = int(np.searchsorted(c, 1 - clip - 1e-12))
    return lo, max(lo + 1, min(255, hi))


def _ramp(lo, hi):
    x = np.arange(256, dtype=np.float64)
    return np.clip(np.rint((x - lo) * 255.0 / max(1, hi - lo)), 0, 255).astype(np.uint8)


class AutoLevels(Adjustment):
    """Stretch each channel to the full range (it also removes colour casts)."""

    name = "auto_levels"
    label = "Auto Levels"
    global_stats = True

    def declare(self):
        return PropertyCollection([FloatProperty("clip", 0.1, 0.0, 5.0, decimals=2,
                                                 label="Clip (%)")])

    def build_luts(self, token):
        src = self._source
        return tuple(_ramp(*_stretch(_alpha_hist(src, c), token["clip"] / 100.0)) for c in range(3))

    def prepare(self, token, source):
        self._source = source
        super().prepare(token, source)


class AutoContrast(AutoLevels):
    """Stretch all channels together: more contrast, the colour balance kept."""

    name = "auto_contrast"
    label = "Auto Contrast"

    def build_luts(self, token):
        src = self._source
        hist = sum(_alpha_hist(src, c) for c in range(3))
        return _ramp(*_stretch(hist, token["clip"] / 100.0))


class AutoColor(AutoLevels):
    """Neutralise a colour cast by balancing the channels' midtones, then
    stretch -- GIMP's White Balance, Photoshop's Auto Color."""

    name = "auto_color"
    label = "Auto Color"

    def build_luts(self, token):
        src = self._source
        clip = token["clip"] / 100.0
        luts = []
        means = []
        for c in range(3):
            lut = _ramp(*_stretch(_alpha_hist(src, c), clip))
            h = _alpha_hist(src, c)
            mean = (lut.astype(np.float64) * h).sum() / max(h.sum(), 1)
            luts.append(lut)
            means.append(mean)
        target = float(np.mean(means))
        out = []
        for lut, mean in zip(luts, means):
            g = math.log(max(target, 1) / 255.0) / math.log(max(mean, 1) / 255.0) if 0 < mean < 255 else 1.0
            x = lut.astype(np.float64) / 255.0
            out.append(np.clip(np.rint(255 * x ** g), 0, 255).astype(np.uint8))
        return tuple(out)


class Equalize(AutoLevels):
    """Spread the levels so each is used about equally."""

    name = "equalize"
    label = "Equalize"

    def declare(self):
        return PropertyCollection()

    def build_luts(self, token):
        src = self._source
        hist = sum(_alpha_hist(src, c) for c in range(3))
        c = np.cumsum(hist)
        if c[-1] <= 0:
            return np.arange(256, dtype=np.uint8)
        nz = c[np.nonzero(hist)[0][0]]
        return np.clip(np.rint((c - nz) / max(c[-1] - nz, 1) * 255), 0, 255).astype(np.uint8)


ADJUSTMENTS = (Curves, ColorBalance, Exposure, Threshold, Vibrance, ChannelMixer,
               PhotoFilter, GradientMap, BlackAndWhite, Colorize, ColorTemperature,
               ReplaceColor, ColorToAlpha, AutoLevels, AutoContrast, AutoColor, Equalize)


# ---- Match Color: one image's colour statistics onto another ---------------------

def _srgb_to_lab(rgb):
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def _lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116.0) / 7.787)
    xyz = xyz * np.array([0.95047, 1.0, 1.08883])
    m = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    lin = np.clip(xyz @ m.T, 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(c * 255.0, 0, 255)


def match_color(target, source, amount=1.0):
    """Photoshop's Match Color, by Reinhard's colour transfer: give `target`
    the mean and spread of `source` in Lab, so one shot takes on another's
    light and cast. Both uint8 RGBA; only opaque-ish pixels are measured.
    Returns new uint8 RGBA (alpha untouched)."""
    def stats(img):
        keep = img[..., 3] >= 128
        lab = _srgb_to_lab(img[..., :3][keep].astype(np.float64))
        if lab.size == 0:
            return np.zeros(3), np.ones(3)
        return lab.mean(axis=0), lab.std(axis=0) + 1e-6
    tm, ts = stats(target)
    sm, ss = stats(source)
    lab = _srgb_to_lab(target[..., :3].astype(np.float64))
    moved = (lab - tm) * (ss / ts) + sm
    lab = lab + (moved - lab) * float(amount)
    out = target.copy()
    out[..., :3] = np.rint(_lab_to_srgb(lab)).astype(np.uint8)
    return out
