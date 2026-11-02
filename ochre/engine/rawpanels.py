# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Camera Raw Filter's panels beyond Basic: Tone Curve, Detail,
HSL / Grayscale, Split Toning and Camera Calibration (Effects -- the
vignette's and grain's extra sliders -- lives with them in photo.py).

Ported from Kooka Pictura's pictura_raw renderer (crates/pictura-adjust/
src/pictura_raw, GPL-3.0), with its slider ranges (those of its dialog,
which follow Adobe Camera Raw's). Every slider's default is a no-op, so
the filter at its defaults still changes nothing; in Camera Raw itself
Sharpening Amount starts at 25. Kooka's dialog also shows Luminance
Detail / Contrast and Color Detail / Smoothness, which its renderer does
not use; they are left out here rather than shown doing nothing.

Each panel works on floats 0..1, clamping as it goes, as Kooka's does.
"""

import numpy as np

from .filters import _box
from .props import BoolProperty, CurveProperty, FloatProperty, IntProperty

# HSL / Grayscale's eight ranges and their hue centers (degrees).
RANGES = ("reds", "oranges", "yellows", "greens", "aquas", "blues", "purples", "magentas")
HUE_CENTERS = (0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 275.0, 300.0)
IDENTITY = ((0, 0), (255, 255))


def _int(name, label, section, lo=-100, hi=100, default=0):
    return IntProperty(name, default, lo, hi, label=label, section=section)


def props():
    """The panels' properties up to Split Toning, each tagged with its
    dialog section."""
    tc, de, hs, st = "Tone Curve", "Detail", "HSL / Grayscale", "Split Toning"
    out = [_int("region_highlights", "Highlights", tc), _int("region_lights", "Lights", tc),
           _int("region_darks", "Darks", tc), _int("region_shadows", "Shadows", tc),
           _int("shadow_split", "Shadow Split", tc, 0, 100, 25),
           _int("midtone_split", "Midtone Split", tc, 0, 100, 50),
           _int("highlight_split", "Highlight Split", tc, 0, 100, 75),
           CurveProperty("curve_rgb", label="Point Curve (RGB)", section=tc),
           CurveProperty("curve_red", label="Point Curve (Red)", section=tc),
           CurveProperty("curve_green", label="Point Curve (Green)", section=tc),
           CurveProperty("curve_blue", label="Point Curve (Blue)", section=tc),
           _int("sharpen_amount", "Sharpening Amount", de, 0, 150),
           FloatProperty("sharpen_radius", 1.0, 0.5, 3.0, decimals=1, label="Sharpening Radius",
                         section=de),
           _int("sharpen_detail", "Sharpening Detail", de, 0, 100, 25),
           _int("sharpen_masking", "Sharpening Masking", de, 0, 100),
           _int("noise_luminance", "Noise Reduction: Luminance", de, 0, 100),
           _int("noise_color", "Noise Reduction: Color", de, 0, 100),
           BoolProperty("grayscale", False, label="Convert to Grayscale", section=hs)]
    for kind, word in (("hue", "Hue"), ("sat", "Saturation"), ("lum", "Luminance"),
                       ("gray", "Grayscale Mix")):
        out += [_int("%s_%s" % (kind, r), "%s: %s" % (word, r.title()), hs) for r in RANGES]
    out += [_int("highlight_hue", "Highlights Hue", st, 0, 360),
            _int("highlight_saturation", "Highlights Saturation", st, 0, 100),
            _int("balance", "Balance", st),
            _int("shadow_hue", "Shadows Hue", st, 0, 360),
            _int("shadow_saturation", "Shadows Saturation", st, 0, 100)]
    return out


def calibration_props():
    """Camera Calibration's sliders: the dialog's last tab, after Effects."""
    ca = "Camera Calibration"
    out = [_int("shadows_tint", "Shadows Tint", ca)]
    for primary in ("red", "green", "blue"):
        out += [_int(primary + "_hue", primary.title() + " Primary Hue", ca),
                _int(primary + "_saturation", primary.title() + " Primary Saturation", ca)]
    return out


def active(t):
    """Whether any panel here would change a picture."""
    return any(t[n] for n in ("region_highlights", "region_lights", "region_darks",
                              "region_shadows", "sharpen_amount", "noise_luminance",
                              "noise_color", "grayscale", "highlight_saturation",
                              "shadow_saturation", "shadows_tint", "red_hue", "red_saturation",
                              "green_hue", "green_saturation", "blue_hue", "blue_saturation")) \
        or any(t["%s_%s" % (k, r)] for k in ("hue", "sat", "lum") for r in RANGES) \
        or any(tuple(map(tuple, t[c])) != IDENTITY
               for c in ("curve_rgb", "curve_red", "curve_green", "curve_blue"))


def apply(x, t):
    """Tone Curve, Detail, HSL / Grayscale, Split Toning (Kooka's order;
    Effects follows in photo.py, then calibrate())."""
    x = np.clip(x, 0.0, 1.0)
    x = tone_curve(x, t)
    x = detail(x, t)
    x = hsl(x, t)
    return split_toning(x, t)


# ---- helpers ------------------------------------------------------------

def _luma(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def rgb_to_hsl(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    hi, lo = x.max(-1), x.min(-1)
    light = (hi + lo) / 2
    d = hi - lo
    flat = d < 1e-12
    dd = np.where(flat, 1.0, d)
    sat = np.where(flat, 0.0, np.where(light > 0.5, d / np.where(flat, 1.0, 2 - hi - lo),
                                       d / np.where(flat | (hi + lo == 0), 1.0, hi + lo)))
    hue = np.where(hi == r, (g - b) / dd, np.where(hi == g, (b - r) / dd + 2, (r - g) / dd + 4))
    return np.where(flat, 0.0, np.mod(hue * 60.0, 360.0)), sat, light


def hsl_to_rgb(h, s, l):
    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q

    def channel(t):
        t = np.mod(t, 1.0)
        return np.where(t < 1 / 6, p + (q - p) * 6 * t,
                        np.where(t < 0.5, q, np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p)))
    hh = h / 360.0
    out = np.stack([channel(hh + 1 / 3), channel(hh), channel(hh - 1 / 3)], -1)
    return np.where((s <= 0)[..., None], l[..., None], out)


def hue_weight(h, center):
    """1 within 20 degrees of the center, fading to 0 over 30 more."""
    d = np.mod(h - center, 360.0)
    d = np.minimum(d, 360.0 - d)
    return np.clip((50.0 - d) / 30.0, 0.0, 1.0)


def _monotone(xs, ys):
    """Fritsch-Carlson tangents for a monotone cubic Hermite curve."""
    n = len(xs)
    d = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)]
    m = [d[0]] + [(d[i - 1] + d[i]) / 2 for i in range(1, n - 1)] + [d[-1]]
    for i in range(n - 1):
        if d[i] == 0:
            m[i] = m[i + 1] = 0.0
        else:
            a, b = m[i] / d[i], m[i + 1] / d[i]
            s = a * a + b * b
            if s > 9:
                tau = 3 / s ** 0.5
                m[i], m[i + 1] = tau * a * d[i], tau * b * d[i]
    return m


def eval_curve(points, x):
    pts = sorted({p[0] / 255.0: p[1] / 255.0 for p in points}.items())
    if len(pts) < 2:
        return x
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    m = _monotone(xs, ys)
    x = min(max(x, 0.0), 1.0)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = 0
    while x > xs[i + 1]:
        i += 1
    h = xs[i + 1] - xs[i]
    t = (x - xs[i]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i]
            + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1])


# ---- the panels ---------------------------------------------------------

def tone_curve(x, t):
    regions = [t["region_shadows"], t["region_darks"], t["region_lights"], t["region_highlights"]]
    curves = [tuple(map(tuple, t[c])) for c in ("curve_rgb", "curve_red", "curve_green",
                                                 "curve_blue")]
    if not any(regions) and all(c == IDENTITY for c in curves):
        return x
    s = (t["shadow_split"] / 100.0, t["midtone_split"] / 100.0, t["highlight_split"] / 100.0)
    centers = (s[0] / 2, (s[0] + s[1]) / 2, (s[1] + s[2]) / 2, (s[2] + 1) / 2)
    base = np.empty(256)
    for i in range(256):
        v = y = i / 255.0
        if any(regions):
            for value, center in zip(regions, centers):
                d = abs(v - center)
                if value and d < 0.25:
                    y += value / 100.0 * 0.15 * 0.5 * (1 + np.cos(np.pi * d / 0.25))
        if curves[0] != IDENTITY:
            y = eval_curve(curves[0], y)
        base[i] = min(max(y, 0.0), 1.0)
    index = np.clip(np.floor(x * 255.0 + 0.5), 0, 255).astype(np.intp)
    out = np.empty_like(x)
    for c, curve in enumerate(curves[1:]):
        lut = base if curve == IDENTITY else \
            np.clip([eval_curve(curve, v) for v in base], 0.0, 1.0)
        out[..., c] = lut[index[..., c]]
    return out


def detail(x, t):
    amount, lum_nr, col_nr = t["sharpen_amount"], t["noise_luminance"], t["noise_color"]
    if lum_nr > 0:
        y = _luma(x)
        x = np.clip(x + ((_box(_box(y, 1, 0), 1, 1) - y) * (lum_nr / 100.0))[..., None], 0, 1)
    if amount > 0:
        y = _luma(x)
        radius = int(min(max(np.floor(t["sharpen_radius"] + 0.5), 1), 16))
        found = y - _box(_box(y, radius, 0), radius, 1)
        gain = amount / 100.0 * (0.5 + t["sharpen_detail"] / 200.0)
        edge = np.minimum(np.abs(found) * 6.0, 1.0)
        mask = 1.0 - t["sharpen_masking"] / 100.0 * (1.0 - edge)
        x = np.clip(x + (found * gain * mask)[..., None], 0, 1)
    if col_nr > 0:
        k = col_nr / 100.0 * 0.8
        y = _luma(x)[..., None]
        x = np.clip(y + (x - y) * (1 - k), 0, 1)
    return x


def hsl(x, t):
    presence = np.minimum((x.max(-1) - x.min(-1)) * 4.0, 1.0)
    h, s, l = rgb_to_hsl(x)
    if t["grayscale"]:
        gray = _luma(x)
        for r, center in zip(RANGES, HUE_CENTERS):
            if t["gray_" + r]:
                gray = gray + hue_weight(h, center) * presence * t["gray_" + r] / 100.0 * 0.5
        return np.repeat(np.clip(gray, 0, 1)[..., None], 3, -1)
    if not any(t["%s_%s" % (k, r)] for k in ("hue", "sat", "lum") for r in RANGES):
        return x
    dh = np.zeros_like(h)
    ds, dl = dh.copy(), dh.copy()
    for r, center in zip(RANGES, HUE_CENTERS):
        w = hue_weight(h, center) * presence
        dh += w * t["hue_" + r]
        ds += w * t["sat_" + r]
        dl += w * t["lum_" + r]
    out = hsl_to_rgb(np.mod(h + dh * 0.5, 360.0), np.clip(s * (1 + ds / 100.0), 0, 1),
                     np.clip(l + dl / 100.0 * 0.5, 0, 1))
    return np.where((presence > 0)[..., None], out, x)


def split_toning(x, t):
    if not (t["shadow_saturation"] or t["highlight_saturation"]):
        return x
    mid = min(max(0.5 + t["balance"] / 100.0 * 0.5, 0.05), 0.95)

    def tint(hue, sat):
        rgb = hsl_to_rgb(np.array(float(hue % 360)), np.array(1.0), np.array(0.5))
        return (rgb - 0.5) * (sat / 100.0 * 0.25)
    shadow = tint(t["shadow_hue"], t["shadow_saturation"])
    highlight = tint(t["highlight_hue"], t["highlight_saturation"])
    y = np.clip(_luma(x), 0, 1)
    sw = np.maximum((mid - y) / mid, 0)[..., None]
    hw = np.maximum((y - mid) / (1 - mid), 0)[..., None]
    return np.clip(x + sw * shadow + hw * highlight, 0, 1)


def calibrate(x, t):
    """Camera Calibration: the primaries' hue and saturation, then a
    green-magenta tint in the shadows. Runs last, after Effects."""
    hues = [t["red_hue"], t["green_hue"], t["blue_hue"]]
    sats = [t["red_saturation"], t["green_saturation"], t["blue_saturation"]]
    tint = t["shadows_tint"]
    if not (any(hues) or any(sats) or tint):
        return x
    x = np.clip(x, 0, 1)
    if any(hues) or any(sats):
        h, s, l = rgb_to_hsl(x)
        dh = np.zeros_like(h)
        ds = dh.copy()
        for hue, sat, center in zip(hues, sats, (0.0, 120.0, 240.0)):
            w = hue_weight(h, center)
            dh += w * hue
            ds += w * sat
        x = hsl_to_rgb(np.mod(h + dh * 0.5, 360.0), np.clip(s * (1 + ds / 100.0), 0, 1), l)
    if tint:
        k = tint / 100.0 * 0.2
        shadow = np.clip(1 - _luma(x), 0, 1)[..., None]
        x = np.clip(x + k * shadow * np.array([1.0, -1.0, 1.0]), 0, 1)
    return x
