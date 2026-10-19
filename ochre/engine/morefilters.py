# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The rest of Photoshop's Filter and Adjustments menus.

Adjustments: Selective Color, Color Lookup (3D LUTs, including .cube files).
Photo: HDR Toning (local tone mapping), Smart Sharpen, Shake Reduction
(Richardson-Lucy deconvolution). Blurs: Iris Blur, Field Blur. Noise:
Despeckle. Render: Difference Clouds, Fibers, Lighting Effects. Stylize:
Extrude, Tiles, Trace Contour. Distort: ZigZag, Shear, Displace. Pixelate
(here under Artistic): Facet, Mezzotint. Other: HSB/HSL.

Where Photoshop takes an interactive panel (Selective Color's colour menu,
Field Blur's pins, Shear's curve), the settings here are a fixed set of
numbers that cover the same ground; each docstring says what differs.
"""

import math
import os

import numpy as np

from .effects import Effect
from .filters import _Filter, _blur_rgba, _box, _gauss, _grid, _hash01, _lum, _pm, _unpm, \
    _value_noise, _warp
from .props import (BoolProperty, ChoiceProperty, ColorProperty, FloatProperty, IntProperty,
                    PropertyCollection, TextProperty)


def _props(*items):
    return PropertyCollection(list(items))


def _rgb(src):
    return src[..., :3].astype(np.float64) / 255.0


def _put(src, rgb):
    out = src.copy()
    out[..., :3] = np.clip(np.rint(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8)
    return out


class _Point(Effect):
    """A pointwise adjustment computed in float (not a per-channel LUT)."""

    category = "Adjustments"
    pointwise = True

    def transform(self, token, rgb):
        raise NotImplementedError

    def prepare(self, token, source):
        self._token = token

    def render_tile(self, src, dst, rect):
        win = src[rect.slice()]
        rgb = self.transform(self._token, win[..., :3].astype(np.float64) / 255.0)
        dst[rect.slice()][..., :3] = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
        dst[rect.slice()][..., 3] = win[..., 3]


# ==== Selective Color ==================================================================

RANGES = ("reds", "yellows", "greens", "cyans", "blues", "magentas", "whites", "neutrals", "blacks")


def selective_weights(rgb, which):
    """How much of each pixel belongs to a colour range (0..1), the way
    Photoshop's Selective Color divides them: hues by how far the dominant
    channels stand above the rest, whites / blacks / neutrals by level."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    mid = rgb.sum(axis=-1) - mx - mn
    if which == "reds":
        return np.where(r >= mx, mx - mid, 0.0)
    if which == "greens":
        return np.where(g >= mx, mx - mid, 0.0)
    if which == "blues":
        return np.where(b >= mx, mx - mid, 0.0)
    if which == "cyans":
        return np.where(r <= mn, mid - mn, 0.0)
    if which == "magentas":
        return np.where(g <= mn, mid - mn, 0.0)
    if which == "yellows":
        return np.where(b <= mn, mid - mn, 0.0)
    if which == "whites":
        return np.clip((mn - 0.5) * 2, 0, 1)
    if which == "blacks":
        return np.clip((0.5 - mx) * 2, 0, 1)
    return np.clip(1 - (np.abs(mx - 0.5) + np.abs(mn - 0.5)), 0, 1)     # neutrals


class SelectiveColor(_Point):
    """Photoshop's Selective Color: add or remove cyan, magenta, yellow and
    black ink in one colour range. One range per application (Photoshop's
    dialog edits all nine at once; apply again for another range)."""

    name, label = "selective_color", "Selective Color"

    def declare(self):
        return _props(
            ChoiceProperty("colors", "reds", list(RANGES)),
            IntProperty("cyan", 0, -100, 100), IntProperty("magenta", 0, -100, 100),
            IntProperty("yellow", 0, -100, 100), IntProperty("black", 0, -100, 100),
            ChoiceProperty("method", "relative", ["relative", "absolute"]))

    def transform(self, t, rgb):
        w = selective_weights(rgb, t["colors"])[..., None]
        inks = np.array([t["cyan"], t["magenta"], t["yellow"]], np.float64) / 100.0
        ink = 1.0 - rgb                                  # C, M, Y as RGB complements
        scale = ink if t["method"] == "relative" else 1.0
        ink = ink + w * inks * scale
        out = 1.0 - np.clip(ink, 0, 1)
        k = t["black"] / 100.0
        if k > 0:
            out = out * (1 - w * k * (out if t["method"] == "relative" else 1.0))
        elif k < 0:
            out = out + (1 - out) * w * (-k)
        return np.clip(out, 0, 1)


# ==== Color Lookup ======================================================================

def parse_cube(text):
    """A .cube 3D LUT (Adobe / Resolve): returns (N, N, N, 3) float indexed
    [b][g][r], with the domain, or raises ValueError."""
    size, dmin, dmax, rows = None, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head = line.split()
        key = head[0].upper()
        if key == "LUT_3D_SIZE":
            size = int(head[1])
        elif key == "DOMAIN_MIN":
            dmin = tuple(float(v) for v in head[1:4])
        elif key == "DOMAIN_MAX":
            dmax = tuple(float(v) for v in head[1:4])
        elif key in ("TITLE", "LUT_1D_SIZE", "LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE"):
            if key == "LUT_1D_SIZE":
                raise ValueError("1D LUTs are not supported")
            continue
        else:
            try:
                rows.append([float(v) for v in head[:3]])
            except ValueError:
                continue
    if not size or len(rows) != size ** 3:
        raise ValueError("not a 3D .cube LUT (size %s, %d rows)" % (size, len(rows)))
    lut = np.asarray(rows, np.float64).reshape(size, size, size, 3)     # red fastest
    return lut, np.asarray(dmin), np.asarray(dmax)


def apply_lut3d(rgb, lut, dmin=(0, 0, 0), dmax=(1, 1, 1)):
    """Trilinear lookup of rgb (..., 3) in 0..1 through lut[b][g][r]."""
    n = lut.shape[0]
    x = (rgb - np.asarray(dmin)) / np.maximum(np.asarray(dmax) - np.asarray(dmin), 1e-9)
    x = np.clip(x, 0, 1) * (n - 1)
    i0 = np.minimum(np.floor(x).astype(np.int64), n - 2)
    f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = 0
    for db, wb in ((0, 1 - fb), (1, fb)):
        for dg, wg in ((0, 1 - fg), (1, fg)):
            for dr, wr in ((0, 1 - fr), (1, fr)):
                out = out + lut[b0 + db, g0 + dg, r0 + dr] * (wb * wg * wr)
    return out


def _look(name, n=17):
    """A built-in look as a 3D LUT, so built-ins and .cube files share a path."""
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([r, gg, b], axis=-1)
    lum = (0.2126 * r + 0.7152 * gg + 0.0722 * b)[..., None]
    if name == "warm":
        out = rgb * np.array([1.08, 1.0, 0.88])
    elif name == "cool":
        out = rgb * np.array([0.9, 1.0, 1.1])
    elif name == "teal_orange":
        # Shadows toward teal, highlights and skin toward orange.
        out = rgb + (1 - lum) * np.array([-0.06, 0.03, 0.08]) + lum * np.array([0.08, 0.02, -0.08])
    elif name == "bleach_bypass":
        sat = lum + (rgb - lum) * 0.45
        out = np.clip((sat - 0.5) * 1.3 + 0.5, 0, 1)
    elif name == "film_fade":
        out = 0.06 + rgb * 0.88
        out = out + (lum - out) * 0.15
    elif name == "cross_process":
        out = np.stack([np.clip((r - 0.5) * 1.4 + 0.55, 0, 1), gg ** 0.85,
                        np.clip(b * 0.7 + 0.15, 0, 1)], axis=-1)
    elif name == "monochrome_warm":
        out = lum * np.array([1.05, 0.98, 0.85])
    else:
        out = rgb
    return np.clip(out, 0, 1)


LOOKS = ("warm", "cool", "teal_orange", "bleach_bypass", "film_fade", "cross_process",
         "monochrome_warm")


class ColorLookup(_Point):
    """Photoshop's Color Lookup: a 3D LUT -- a built-in look, or any .cube
    file (which takes precedence when given) -- blended in by amount."""

    name, label = "color_lookup", "Color Lookup"

    def declare(self):
        return _props(ChoiceProperty("look", "teal_orange", list(LOOKS)),
                      TextProperty("file", "", label="3D LUT file (.cube)"),
                      IntProperty("amount", 100, 0, 100))

    def prepare(self, token, source):
        super().prepare(token, source)
        path = (token.get("file") or "").strip()
        self._lut = None
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    self._lut = parse_cube(fh.read())
            except (OSError, ValueError):
                self._lut = None
        if self._lut is None:
            self._lut = (_look(token["look"]), np.zeros(3), np.ones(3))

    def transform(self, t, rgb):
        lut, dmin, dmax = self._lut
        out = apply_lut3d(rgb, lut, dmin, dmax)
        a = t["amount"] / 100.0
        return rgb + (out - rgb) * a


# ==== HDR Toning ========================================================================

class HDRToning(_Filter):
    """Photoshop's HDR Toning (Local Adaptation) on an ordinary image:
    log-luminance split into a base and a detail layer with an EDGE-AWARE
    filter (the guided filter standing in for Durand and Dorsey's bilateral,
    2002); the base is compressed, the detail kept or boosted, so shadows and
    highlights come together without halos at strong edges."""

    name, label, category = "hdr_toning", "HDR Toning", "Photo"

    def declare(self):
        return _props(IntProperty("radius", 40, 2, 300, label="Radius (px)"),
                      IntProperty("strength", 60, 0, 100, label="Compression"),
                      IntProperty("detail", 30, -100, 300),
                      FloatProperty("exposure", 0.0, -3.0, 3.0, decimals=2),
                      FloatProperty("gamma", 1.0, 0.3, 3.0, decimals=2),
                      IntProperty("vibrance", 10, -100, 100))

    def compute(self, t, src):
        from .photo import to_linear, to_srgb
        from .refine import guided_filter
        rgb = to_linear(_rgb(src))
        lum = np.maximum(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2], 1e-5)
        log = np.log2(lum)
        base = guided_filter(log, log, int(t["radius"]), 0.15)
        detail = log - base
        mean = float(base.mean())
        k = 1.0 - t["strength"] / 100.0 * 0.8
        new_log = mean + (base - mean) * k + detail * (1 + t["detail"] / 100.0) + t["exposure"]
        ratio = (2.0 ** new_log / lum)[..., None]
        out = to_srgb(rgb * ratio)
        out = np.clip(out, 0, 1) ** (1.0 / t["gamma"])
        if t["vibrance"]:
            grey = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2])[..., None]
            sat = (out.max(-1) - out.min(-1))[..., None]
            out = grey + (out - grey) * (1 + t["vibrance"] / 100.0 * (1 - sat))
        return _put(src, np.clip(out, 0, 1))


# ==== Sharpen and deconvolution ============================================================

def _kernel(kind, radius, angle=0.0):
    """A blur kernel (sums to 1): gaussian, lens (disc) or motion (line)."""
    r = max(1, int(math.ceil(radius * (2 if kind == "gaussian" else 1))))
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    if kind == "lens":
        k = (np.hypot(xs, ys) <= radius + 0.5).astype(np.float64)
    elif kind == "motion":
        a = math.radians(angle)
        along = xs * math.cos(a) - ys * math.sin(a)
        across = xs * math.sin(a) + ys * math.cos(a)
        k = ((np.abs(across) <= 0.5) & (np.abs(along) <= radius)).astype(np.float64)
    else:
        s = max(radius / 2.0, 0.3)
        k = np.exp(-(xs * xs + ys * ys) / (2 * s * s))
    return k / k.sum()


def _convolve(img, k):
    """2-D convolution of (h, w[, c]) with a small kernel, edges clamped, via FFT."""
    kh, kw = k.shape
    py, px = kh // 2, kw // 2
    pad = [(py, py), (px, px)] + [(0, 0)] * (img.ndim - 2)
    p = np.pad(img, pad, mode="edge")
    H, W = p.shape[:2]
    K = np.fft.rfft2(k, s=(H, W))
    if img.ndim == 3:
        out = np.fft.irfft2(np.fft.rfft2(p, axes=(0, 1)) * K[..., None], s=(H, W), axes=(0, 1))
    else:
        out = np.fft.irfft2(np.fft.rfft2(p) * K, s=(H, W))
    # The FFT puts the kernel's centre at (py, px); after shifting it back,
    # the original image starts at (py, px) of the padded frame.
    out = np.roll(out, (-py, -px), axis=(0, 1))
    return out[py:py + img.shape[0], px:px + img.shape[1]]


def richardson_lucy(img, k, iterations=15, damp=0.0):
    """Richardson-Lucy deconvolution of img (float, > 0) by kernel k."""
    est = np.array(img, copy=True)
    flip = k[::-1, ::-1]
    for _ in range(iterations):
        blurred = np.maximum(_convolve(est, k), 1e-6)
        ratio = img / blurred
        if damp > 0:
            ratio = 1 + (ratio - 1) / (1 + damp)
        est = est * _convolve(ratio, flip)
    return est


class SmartSharpen(_Filter):
    """Photoshop's Smart Sharpen: remove a Gaussian, lens or motion blur
    (a few Richardson-Lucy steps toward the unblurred image), with noise
    held back and the shadows / highlights optionally spared."""

    name, label, category = "smart_sharpen", "Smart Sharpen", "Photo"

    def declare(self):
        return _props(IntProperty("amount", 150, 0, 500, label="Amount (%)"),
                      FloatProperty("radius", 1.5, 0.3, 20.0, decimals=1),
                      IntProperty("noise", 10, 0, 100, label="Reduce noise (%)"),
                      ChoiceProperty("remove", "gaussian", ["gaussian", "lens", "motion"]),
                      FloatProperty("angle", 0.0, -180.0, 180.0, decimals=0),
                      IntProperty("shadows", 0, 0, 100, label="Fade shadows (%)"),
                      IntProperty("highlights", 0, 0, 100, label="Fade highlights (%)"))

    def compute(self, t, src):
        x = _rgb(src)
        k = _kernel(t["remove"], t["radius"], t["angle"])
        sharp = richardson_lucy(np.maximum(x, 1e-3), k, iterations=6, damp=t["noise"] / 25.0)
        detail = sharp - x
        if t["noise"]:
            detail = _gauss(detail.astype(np.float32), t["noise"] / 50.0).astype(np.float64)
        lum = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
        fade = 1 - t["shadows"] / 100.0 * np.clip(1 - lum * 3, 0, 1) \
            - t["highlights"] / 100.0 * np.clip((lum - 0.67) * 3, 0, 1)
        return _put(src, np.clip(x + detail * (t["amount"] / 100.0) * np.clip(fade, 0, 1), 0, 1))


class ShakeReduction(_Filter):
    """Photoshop's Shake Reduction for a straight camera shake: Richardson-
    Lucy deconvolution by a line kernel of the given angle and length.
    (Photoshop estimates the shake path from the image; here it is given.)"""

    name, label, category = "shake_reduction", "Shake Reduction", "Photo"

    def declare(self):
        return _props(FloatProperty("angle", 0.0, -180.0, 180.0, decimals=0),
                      FloatProperty("length", 6.0, 1.0, 60.0, decimals=1, label="Blur trace length"),
                      IntProperty("iterations", 20, 1, 100),
                      IntProperty("smoothing", 10, 0, 100))

    def compute(self, t, src):
        x = np.maximum(_rgb(src), 1e-3)
        k = _kernel("motion", t["length"] / 2.0, t["angle"])
        out = richardson_lucy(x, k, iterations=int(t["iterations"]), damp=t["smoothing"] / 20.0)
        return _put(src, np.clip(out, 0, 1))


# ==== Blur Gallery ==========================================================================

def _blend_by(sharp, blurred, m):
    m = np.clip(m, 0, 1)[..., None]
    return _unpm(_pm(sharp) * (1 - m) + _pm(blurred) * m)


class IrisBlur(_Filter):
    """Photoshop's Iris Blur: sharp inside an ellipse, blurring outside it,
    with a feathered transition."""

    name, label, category = "iris_blur", "Iris Blur", "Blurs"

    def declare(self):
        return _props(IntProperty("center_x", 50, 0, 100, label="Center X (%)"),
                      IntProperty("center_y", 50, 0, 100, label="Center Y (%)"),
                      IntProperty("radius_x", 30, 1, 100, label="Radius X (%)"),
                      IntProperty("radius_y", 30, 1, 100, label="Radius Y (%)"),
                      FloatProperty("angle", 0.0, -180.0, 180.0, decimals=0),
                      IntProperty("feather", 40, 0, 100, label="Feather (%)"),
                      IntProperty("blur", 15, 0, 200, label="Blur (px)"))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        dx, dy = xs - w * t["center_x"] / 100.0, ys - h * t["center_y"] / 100.0
        a = math.radians(t["angle"])
        u = dx * math.cos(a) + dy * math.sin(a)
        v = -dx * math.sin(a) + dy * math.cos(a)
        rx, ry = max(1.0, w * t["radius_x"] / 100.0), max(1.0, h * t["radius_y"] / 100.0)
        d = np.hypot(u / rx, v / ry)
        inner = 1 - t["feather"] / 100.0
        m = (d - inner) / max(1e-6, 1 - inner) if inner < 1 else (d >= 1).astype(np.float64)
        return _blend_by(src, _blur_rgba(src, t["blur"]), m)


class FieldBlur(_Filter):
    """Photoshop's Field Blur with two pins: the blur amount changes smoothly
    from one pin's value to the other's across the image (Photoshop allows
    any number of pins, placed on the canvas)."""

    name, label, category = "field_blur", "Field Blur", "Blurs"

    def declare(self):
        return _props(IntProperty("pin1_x", 50, 0, 100, label="Pin 1 X (%)"),
                      IntProperty("pin1_y", 20, 0, 100, label="Pin 1 Y (%)"),
                      IntProperty("blur1", 0, 0, 200, label="Pin 1 blur (px)"),
                      IntProperty("pin2_x", 50, 0, 100, label="Pin 2 X (%)"),
                      IntProperty("pin2_y", 80, 0, 100, label="Pin 2 Y (%)"),
                      IntProperty("blur2", 20, 0, 200, label="Pin 2 blur (px)"))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        p1 = np.array([w * t["pin1_x"] / 100.0, h * t["pin1_y"] / 100.0])
        p2 = np.array([w * t["pin2_x"] / 100.0, h * t["pin2_y"] / 100.0])
        d = p2 - p1
        n2 = max(float(d @ d), 1e-9)
        s = np.clip(((xs - p1[0]) * d[0] + (ys - p1[1]) * d[1]) / n2, 0, 1)
        s = s * s * (3 - 2 * s)
        amount = t["blur1"] + (t["blur2"] - t["blur1"]) * s
        top = max(t["blur1"], t["blur2"])
        if top <= 0:
            return src.copy()
        levels = np.linspace(0, top, 5)
        stack = [_pm(src) if lv < 0.5 else _pm(_blur_rgba(src, lv)) for lv in levels]
        pos = np.clip(amount / top * 4, 0, 4)
        i0 = np.minimum(np.floor(pos).astype(np.int64), 3)
        f = (pos - i0)[..., None]
        out = np.zeros(src.shape, np.float32)
        for i in range(4):
            sel = (i0 == i)[..., None]
            out += np.where(sel, stack[i] * (1 - f) + stack[i + 1] * f, 0)
        return _unpm(out)


# ==== Noise ===================================================================================

class Despeckle(_Filter):
    """Photoshop's Despeckle: smooth everywhere except at edges, so noise
    goes and outlines stay."""

    name, label, category = "despeckle", "Despeckle", "Noise"

    def declare(self):
        return _props(IntProperty("threshold", 12, 1, 100, label="Edge threshold"))

    def compute(self, t, src):
        from .filters import _rank
        x = src.astype(np.float32)
        med = np.stack([_rank(src[..., c], 1, "median") for c in range(3)], -1).astype(np.float32)
        lum = _lum(x)
        gy, gx = np.gradient(_gauss(lum[..., None], 1.0)[..., 0])
        edge = np.clip(np.hypot(gx, gy) / t["threshold"], 0, 1)[..., None]
        out = src.copy()
        out[..., :3] = np.clip(np.rint(med * (1 - edge) + x[..., :3] * edge), 0, 255)
        return out


# ==== Render ===================================================================================

class DifferenceClouds(_Filter):
    """Clouds, combined with the image by Difference: run it repeatedly for
    marbled, veined textures."""

    name, label, category = "difference_clouds", "Difference Clouds", "Render"

    def declare(self):
        return _props(IntProperty("scale", 120, 4, 1000), IntProperty("roughness", 55, 0, 100),
                      IntProperty("seed", 0, 0, 9999))

    def compute(self, t, src):
        h, w = src.shape[:2]
        n = _value_noise(h, w, t["scale"], t["seed"], 6, t["roughness"] / 100.0)
        out = src.copy()
        cl = (n * 255.0)[..., None]
        out[..., :3] = np.abs(src[..., :3].astype(np.float64) - cl).astype(np.uint8)
        return out


class Fibers(_Filter):
    """Photoshop's Fibers: woven, vertical streaks in two colours."""

    name, label, category = "fibers", "Fibers", "Render"

    def declare(self):
        return _props(IntProperty("variance", 16, 1, 64), IntProperty("strength", 4, 1, 64),
                      ColorProperty("foreground", (0, 0, 0, 255)),
                      ColorProperty("background", (255, 255, 255, 255)),
                      IntProperty("seed", 0, 0, 9999))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs = np.arange(w)
        cols = _hash01(xs, np.zeros(w, np.int64), t["seed"])
        cols = np.convolve(cols, np.ones(3) / 3, mode="same")
        stretch = max(2.0, 64.0 / t["strength"])
        n = _value_noise(h, w, t["variance"], t["seed"] + 1, 3, 0.5)
        n = _box(n.astype(np.float32)[..., None], int(stretch), 0)[..., 0]
        v = np.clip(0.6 * cols[None, :] + 0.6 * n - 0.1, 0, 1)[..., None]
        fg = np.array(t["foreground"][:3], np.float64)
        bg = np.array(t["background"][:3], np.float64)
        out = np.empty_like(src)
        out[..., :3] = np.clip(np.rint(bg + (fg - bg) * v), 0, 255)
        out[..., 3] = 255
        return out


class LightingEffects(_Filter):
    """Photoshop's Lighting Effects: one light -- spot, point or infinite --
    on the image as a surface, its relief taken from the brightness (the
    texture channel), lit with Lambert shading plus a specular highlight."""

    name, label, category = "lighting_effects", "Lighting Effects", "Render"

    def declare(self):
        return _props(ChoiceProperty("light", "spot", ["spot", "point", "infinite"]),
                      IntProperty("light_x", 30, 0, 100, label="Light X (%)"),
                      IntProperty("light_y", 30, 0, 100, label="Light Y (%)"),
                      IntProperty("height", 50, 1, 200, label="Light height (%)"),
                      IntProperty("intensity", 80, 0, 200),
                      IntProperty("spot_size", 60, 5, 200, label="Spot size (%)"),
                      ColorProperty("color", (255, 255, 240, 255)),
                      IntProperty("ambience", 20, 0, 100),
                      IntProperty("relief", 20, 0, 100, label="Texture height"),
                      IntProperty("gloss", 30, 0, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        x = _rgb(src)
        height = _gauss(_lum(x)[..., None].astype(np.float32), 1.0)[..., 0] * t["relief"] / 10.0
        gy, gx = np.gradient(height)
        normal = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        xs, ys = _grid(h, w)
        size = max(w, h)
        lx, ly = w * t["light_x"] / 100.0, h * t["light_y"] / 100.0
        lz = size * t["height"] / 100.0
        if t["light"] == "infinite":
            L = np.array([lx - w / 2.0, ly - h / 2.0, -lz])
            L = -L / np.linalg.norm(L)
            ldir = np.broadcast_to(L, normal.shape)
            fall = np.ones((h, w))
        else:
            vec = np.stack([lx - xs, ly - ys, np.full_like(xs, lz)], -1)
            dist = np.linalg.norm(vec, axis=-1, keepdims=True)
            ldir = vec / dist
            if t["light"] == "spot":
                r = np.hypot(xs - lx, ys - ly) / (size * t["spot_size"] / 100.0)
                fall = np.clip(1 - r, 0, 1) ** 1.5
            else:
                fall = 1.0 / (1 + (dist[..., 0] / (size * 0.8)) ** 2)
        diffuse = np.clip((normal * ldir).sum(-1), 0, 1)
        half = ldir + np.array([0, 0, 1.0])
        half /= np.linalg.norm(half, axis=-1, keepdims=True)
        spec = np.clip((normal * half).sum(-1), 0, 1) ** (4 + t["gloss"]) * t["gloss"] / 100.0
        colour = np.array(t["color"][:3], np.float64) / 255.0
        light = t["ambience"] / 100.0 + (t["intensity"] / 100.0) * fall * diffuse
        out = x * light[..., None] * (colour * 0.5 + 0.5) + (spec * fall)[..., None] * colour
        return _put(src, np.clip(out, 0, 1))


# ==== Stylize ===================================================================================

class Extrude(_Filter):
    """Photoshop's Extrude: the image broken into blocks or pyramids that
    stand out of the surface, each its average colour (solid front faces) or
    its own pixels, shaded as if lit from the top left."""

    name, label, category = "extrude", "Extrude", "Stylize"

    def declare(self):
        return _props(ChoiceProperty("type", "blocks", ["blocks", "pyramids"]),
                      IntProperty("size", 16, 2, 255), IntProperty("depth", 30, 1, 255),
                      BoolProperty("solid", False, label="Solid front faces"),
                      IntProperty("seed", 0, 0, 9999))

    def compute(self, t, src):
        h, w = src.shape[:2]
        s = int(t["size"])
        x = _rgb(src)
        ys, xs = np.mgrid[0:h, 0:w]
        by, bx = ys // s, xs // s
        fy, fx = (ys % s + 0.5) / s, (xs % s + 0.5) / s
        nby, nbx = (h + s - 1) // s, (w + s - 1) // s
        if t["solid"] or t["type"] == "pyramids":
            sums = np.zeros((nby, nbx, 3))
            cnt = np.zeros((nby, nbx, 1))
            np.add.at(sums, (by, bx), x)
            np.add.at(cnt, (by, bx), 1)
            base = (sums / np.maximum(cnt, 1))[by, bx]
        else:
            base = x
        if t["type"] == "pyramids":
            # Four faces, lit from the top left.
            top = (fy < fx) & (fy < 1 - fx)
            left = (fx <= fy) & (fx < 1 - fy)
            bottom = (fy >= fx) & (fy >= 1 - fx)
            shade = np.where(top, 1.25, np.where(left, 1.1, np.where(bottom, 0.65, 0.8)))
        else:
            # A block's face shrinks with its (random) height; the edges round
            # it are its sides.
            # Depth in pixels, relative to the block: a deep block shows wide sides.
            hgt = _hash01(bx, by, t["seed"]) * t["depth"] / float(s)
            m = np.minimum(0.45, 0.5 * hgt)
            side_r = fx > 1 - m
            side_b = fy > 1 - m
            shade = np.where(side_r & ~side_b, 0.7, np.where(side_b, 0.55, 1.0))
        return _put(src, np.clip(base * shade[..., None], 0, 1))


class Tiles(_Filter):
    """Photoshop's Tiles: square tiles, each nudged off its place, the gaps
    filled with a colour, the inverse image or the image unaltered."""

    name, label, category = "tiles", "Tiles", "Stylize"

    def declare(self):
        return _props(IntProperty("tiles", 10, 1, 99, label="Tiles across"),
                      IntProperty("offset", 10, 1, 99, label="Maximum offset (%)"),
                      ChoiceProperty("fill", "background", ["background", "inverse", "unaltered"]),
                      ColorProperty("color", (255, 255, 255, 255), label="Background"),
                      IntProperty("seed", 0, 0, 9999))

    def compute(self, t, src):
        h, w = src.shape[:2]
        s = max(1, int(round(w / float(t["tiles"]))))
        if t["fill"] == "inverse":
            out = src.copy()
            out[..., :3] = 255 - src[..., :3]
        elif t["fill"] == "unaltered":
            out = src.copy()
        else:
            out = np.empty_like(src)
            out[...] = t["color"]
        mo = max(1, int(s * t["offset"] / 100.0))
        for ty in range(0, h, s):
            for tx in range(0, w, s):
                ox = int(_hash01(tx, ty, t["seed"]) * (2 * mo + 1)) - mo
                oy = int(_hash01(ty, tx, t["seed"] + 7) * (2 * mo + 1)) - mo
                tile = src[ty:ty + s, tx:tx + s]
                y0, x0 = ty + oy, tx + ox
                ya, xa = max(0, y0), max(0, x0)
                yb, xb = min(h, y0 + tile.shape[0]), min(w, x0 + tile.shape[1])
                if yb > ya and xb > xa:
                    out[ya:yb, xa:xb] = tile[ya - y0:yb - y0, xa - x0:xb - x0]
        return out


class TraceContour(_Filter):
    """Photoshop's Trace Contour: thin lines where each channel's brightness
    crosses a level, on white -- a topographic map of the image."""

    name, label, category = "trace_contour", "Trace Contour", "Stylize"

    def declare(self):
        return _props(IntProperty("level", 128, 0, 255),
                      ChoiceProperty("edge", "lower", ["lower", "upper"]))

    def compute(self, t, src):
        v = src[..., :3].astype(np.int64)
        lv = t["level"]
        above = v >= lv if t["edge"] == "upper" else v < lv
        edge = np.zeros_like(above)
        edge[:-1] |= above[:-1] & ~above[1:]
        edge[:, :-1] |= above[:, :-1] & ~above[:, 1:]
        out = src.copy()
        out[..., :3] = np.where(edge, 0, 255).astype(np.uint8)
        return out


# ==== Distort ====================================================================================

class ZigZag(_Filter):
    """Photoshop's ZigZag: radial ripples -- around the centre, out from it,
    or as pond ripples."""

    name, label, category = "zigzag", "ZigZag", "Distort"

    def declare(self):
        return _props(IntProperty("amount", 10, -100, 100), IntProperty("ridges", 5, 1, 20),
                      ChoiceProperty("style", "pond", ["pond", "out", "around"],
                                     labels={"pond": "Pond ripples", "out": "Out from center",
                                             "around": "Around center"}))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        cx, cy = w / 2.0, h / 2.0
        dx, dy = xs - cx, ys - cy
        r = np.hypot(dx, dy)
        rmax = math.hypot(cx, cy)
        wave = np.sin(r / rmax * t["ridges"] * 2 * math.pi) * (t["amount"] / 100.0)
        fade = np.clip(1 - r / rmax, 0, 1)
        if t["style"] == "around":
            ang = wave * fade * 0.5
            c, s = np.cos(ang), np.sin(ang)
            return _warp(src, cx + c * dx - s * dy, cy + s * dx + c * dy)
        scale = 1 + wave * fade * (0.25 if t["style"] == "out" else 0.12)
        if t["style"] == "pond":
            ang = wave * fade * 0.25
            c, s = np.cos(ang), np.sin(ang)
            dx, dy = c * dx - s * dy, s * dx + c * dy
        return _warp(src, cx + dx * scale, cy + dy * scale)


class Shear(_Filter):
    """Photoshop's Shear: rows shifted sideways along a curve. Here the curve
    is a bend through three points (top, middle, bottom offsets) rather than
    one drawn by hand."""

    name, label, category = "shear", "Shear", "Distort"

    def declare(self):
        return _props(IntProperty("top", 0, -100, 100, label="Top offset (%)"),
                      IntProperty("middle", 30, -100, 100, label="Middle offset (%)"),
                      IntProperty("bottom", 0, -100, 100, label="Bottom offset (%)"),
                      ChoiceProperty("edges", "wrap", ["wrap", "clamp"],
                                     labels={"wrap": "Wrap around", "clamp": "Repeat edge pixels"}))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        v = ys / h
        a, b, c = t["top"], t["middle"], t["bottom"]
        # Quadratic through (0, a), (0.5, b), (1, c).
        off = a * (1 - v) * (1 - 2 * v) + b * 4 * v * (1 - v) + c * v * (2 * v - 1)
        return _warp(src, xs - off / 100.0 * w * 0.5, ys, t["edges"])


class Displace(_Filter):
    """Photoshop's Displace: move pixels by a displacement map -- an image
    file (mid grey = no move; red drives horizontal, green vertical), or,
    with no file, the image's own brightness."""

    name, label, category = "displace", "Displace", "Distort"

    def declare(self):
        return _props(IntProperty("horizontal", 10, -999, 999, label="Horizontal scale"),
                      IntProperty("vertical", 10, -999, 999, label="Vertical scale"),
                      TextProperty("file", "", label="Displacement map file"),
                      ChoiceProperty("fit", "stretch", ["stretch", "tile"]),
                      ChoiceProperty("edges", "wrap", ["wrap", "clamp"]))

    def compute(self, t, src):
        h, w = src.shape[:2]
        dmap = None
        path = (t.get("file") or "").strip()
        if path and os.path.exists(path):
            try:
                from PIL import Image
                im = Image.open(path).convert("RGB")
                if t["fit"] == "stretch":
                    dmap = np.asarray(im.resize((w, h), Image.BILINEAR), np.float64)
                else:
                    a = np.asarray(im, np.float64)
                    reps = (h // a.shape[0] + 1, w // a.shape[1] + 1, 1)
                    dmap = np.tile(a, reps)[:h, :w]
            except (OSError, ValueError):
                dmap = None
        if dmap is None:
            g = _lum(src[..., :3].astype(np.float64))
            dmap = np.stack([g, g, g], -1)
        xs, ys = _grid(h, w)
        # Photoshop's scale: 100% moves up to 128 pixels.
        mx = (dmap[..., 0] - 128.0) / 128.0 * t["horizontal"] / 100.0 * 128.0
        my = (dmap[..., 1] - 128.0) / 128.0 * t["vertical"] / 100.0 * 128.0
        return _warp(src, xs + mx, ys + my, t["edges"])


# ==== Pixelate ====================================================================================

class Facet(_Filter):
    """Photoshop's Facet: similar neighbouring colours clumped into flat
    patches, edges kept (a small Kuwahara pass, repeated)."""

    name, label, category = "facet", "Facet", "Artistic"

    def declare(self):
        return _props(IntProperty("passes", 2, 1, 5))

    def compute(self, t, src):
        from .filters import OilPainting
        out = src
        oil = OilPainting()
        for _ in range(int(t["passes"])):
            out = oil.compute({"brush": 2}, out)
        return out


class Mezzotint(_Filter):
    """Photoshop's Mezzotint: every channel reduced to full-on / full-off by a
    random screen of dots, lines or strokes."""

    name, label, category = "mezzotint", "Mezzotint", "Artistic"

    def declare(self):
        return _props(ChoiceProperty("type", "medium_dots",
                                     ["fine_dots", "medium_dots", "coarse_dots", "short_lines",
                                      "long_lines", "short_strokes"]),
                      IntProperty("seed", 0, 0, 9999))

    def compute(self, t, src):
        h, w = src.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        kind = t["type"]
        cell = {"fine_dots": 1, "medium_dots": 2, "coarse_dots": 3}.get(kind)

        def screen(seed):
            if cell:
                return _hash01(xs // cell, ys // cell, seed)
            if kind in ("short_lines", "long_lines"):
                run = 4 if kind == "short_lines" else 16
                return _hash01(xs // run, ys, seed)
            return _hash01((xs + ys) // 4, ys - xs, seed)          # short strokes, diagonal

        out = src.copy()
        for c in range(3):
            out[..., c] = np.where(src[..., c] / 255.0 > screen(t["seed"] * 3 + c), 255, 0)
        return out


# ==== Other ========================================================================================

def rgb_to_hsv(x):
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    hue = np.zeros_like(mx)
    nz = d > 1e-12
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    hue[rm] = ((g - b)[rm] / d[rm]) % 6
    hue[gm] = (b - r)[gm] / d[gm] + 2
    hue[bm] = (r - g)[bm] / d[bm] + 4
    return hue / 6.0, np.where(mx > 0, d / np.maximum(mx, 1e-12), 0), mx, mn


class HSBHSL(_Filter):
    """Photoshop's Filter > Other > HSB/HSL: put hue, saturation and
    brightness (or lightness) INTO the red, green and blue channels -- or
    back -- so each can be edited as a channel."""

    name, label, category = "hsb_hsl", "HSB / HSL", "Edge & Generic"

    def declare(self):
        return _props(ChoiceProperty("mode", "rgb_to_hsb",
                                     ["rgb_to_hsb", "rgb_to_hsl", "hsb_to_rgb", "hsl_to_rgb"],
                                     labels={"rgb_to_hsb": "RGB to HSB", "rgb_to_hsl": "RGB to HSL",
                                             "hsb_to_rgb": "HSB to RGB", "hsl_to_rgb": "HSL to RGB"}))

    def compute(self, t, src):
        x = _rgb(src)
        mode = t["mode"]
        if mode == "rgb_to_hsb":
            hue, sat, v, _mn = rgb_to_hsv(x)
            return _put(src, np.stack([hue, sat, v], -1))
        if mode == "rgb_to_hsl":
            hue, _s, mx, mn = rgb_to_hsv(x)
            light = (mx + mn) / 2
            d = mx - mn
            sat = np.where(d > 1e-12, d / np.maximum(1 - np.abs(2 * light - 1), 1e-12), 0)
            return _put(src, np.stack([hue, np.clip(sat, 0, 1), light], -1))
        hue, sat, third = x[..., 0] * 6.0, x[..., 1], x[..., 2]
        if mode == "hsb_to_rgb":
            c = third * sat
        else:
            c = (1 - np.abs(2 * third - 1)) * sat
        xx = c * (1 - np.abs(hue % 2 - 1))
        m = third - c if mode == "hsb_to_rgb" else third - c / 2
        zeros = np.zeros_like(c)
        sector = np.floor(hue).astype(np.int64) % 6
        r = np.choose(sector, [c, xx, zeros, zeros, xx, c])
        g = np.choose(sector, [xx, c, c, xx, zeros, zeros])
        b = np.choose(sector, [zeros, zeros, xx, c, c, xx])
        return _put(src, np.clip(np.stack([r + m, g + m, b + m], -1), 0, 1))


MORE = (SelectiveColor, ColorLookup, HDRToning, SmartSharpen, ShakeReduction, IrisBlur, FieldBlur,
        Despeckle, DifferenceClouds, Fibers, LightingEffects, Extrude, Tiles, TraceContour, ZigZag,
        Shear, Displace, Facet, Mezzotint, HSBHSL)
