# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photo correction filters: Photoshop's Camera Raw Filter, Lens Correction
and Adaptive Wide Angle, done with classical image processing.

  camera_raw     one panel of the Basic controls: white balance (temperature,
                 tint), exposure in stops, contrast, highlights / shadows
                 (local, by blurred brightness so detail is kept), whites /
                 blacks, texture (fine detail), clarity (midtone local
                 contrast), dehaze (the dark channel prior, He, Sun and Tang
                 2009, refined with a guided filter), vibrance, saturation,
                 post-crop vignette and grain. White balance and exposure
                 work in LINEAR light, as a camera's do.
  lens_correction  remove barrel / pincushion distortion, chromatic
                 aberration (red/cyan and blue/yellow fringes, by scaling
                 those channels radially), lens vignetting, and the
                 Transform section: vertical / horizontal perspective, angle
                 and scale.
  adaptive_wide_angle  re-project a wide shot: fisheye to rectilinear, or a
                 rectilinear wide angle to cylindrical / stereographic,
                 which un-stretches people at the edges. Photoshop also takes
                 hand-drawn constraint lines; that part is not done here.

Each is an ordinary effect: it can be applied, recorded as an action, used
as a smart filter, or stacked in the Filter Gallery (FilterStack below).
"""

import json
import math

import numpy as np

from .effects import Effect
from .props import (ChoiceProperty, FloatProperty, IntProperty, PropertyCollection,
                    TextProperty)
from .warp import warp_field


def _props(*items):
    return PropertyCollection(list(items))


class _Whole(Effect):
    """Computes the whole image once per setting; tiles copy from it."""

    def compute(self, token, src):
        raise NotImplementedError

    def prepare(self, token, source):
        key = (repr(sorted(token.items())), source.shape, source.ctypes.data,
               int(source[::17, ::13].sum()))
        if getattr(self, "_key", None) != key:
            self._result = self.compute(token, source)
            self._key = key

    def render_tile(self, src, dst, rect):
        dst[rect.slice()] = self._result[rect.slice()]


# ---- colour helpers ---------------------------------------------------------------

def to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _lum(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _blur(a, sigma):
    """Gaussian-ish blur (three box passes per axis) of a 2-D or 3-D float array."""
    if sigma < 0.5:
        return a
    r = max(1, int(round(math.sqrt(12 * sigma * sigma / 3 + 1) / 2)))
    out = a.astype(np.float64)
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0)] * out.ndim
            pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
            n = out.shape[axis]
            out = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
                   - np.take(c, np.arange(0, n), axis=axis)) / (2 * r + 1)
    return out


def _min_filter(a, r):
    """Minimum over a (2r+1) square, separably (van Herk would be faster;
    this is plain and exact)."""
    out = a
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = np.take(p, np.arange(0, n), axis=axis)
        for k in range(1, 2 * r + 1):
            acc = np.minimum(acc, np.take(p, np.arange(k, k + n), axis=axis))
        out = acc
    return out


def dehaze(rgb, amount, radius=7):
    """The dark channel prior: haze lifts the darkest channel of every patch;
    estimating that lift gives the transmission, and dividing it out
    recovers the scene. amount in -1..1; negative adds haze."""
    from .refine import guided_filter
    h, w = rgb.shape[:2]
    dark = _min_filter(rgb.min(axis=-1), radius)
    n = max(1, int(h * w * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    airlight = np.maximum(rgb.reshape(-1, 3)[idx].mean(axis=0), 1e-3)
    if amount < 0:
        return rgb + (airlight - rgb) * (-amount) * 0.6
    norm = _min_filter((rgb / airlight).min(axis=-1), radius)
    t = 1.0 - 0.95 * norm
    t = guided_filter(_lum(rgb), t, max(4, radius * 4), 1e-3)
    t = np.clip(t, 0.1, 1.0)[..., None]
    clear = (rgb - airlight) / t + airlight
    return rgb + (clear - rgb) * amount


# ---- Camera Raw ---------------------------------------------------------------------

class CameraRaw(_Whole):
    """Photoshop's Camera Raw Filter, Basic panel (plus vignette and grain)."""

    name, label, category = "camera_raw", "Camera Raw Filter", "Photo"

    def declare(self):
        return _props(
            IntProperty("temperature", 0, -100, 100), IntProperty("tint", 0, -100, 100),
            FloatProperty("exposure", 0.0, -5.0, 5.0, decimals=2, label="Exposure (stops)"),
            IntProperty("contrast", 0, -100, 100),
            IntProperty("highlights", 0, -100, 100), IntProperty("shadows", 0, -100, 100),
            IntProperty("whites", 0, -100, 100), IntProperty("blacks", 0, -100, 100),
            IntProperty("texture", 0, -100, 100), IntProperty("clarity", 0, -100, 100),
            IntProperty("dehaze", 0, -100, 100),
            IntProperty("vibrance", 0, -100, 100), IntProperty("saturation", 0, -100, 100),
            IntProperty("vignette", 0, -100, 100, label="Post-crop vignette"),
            IntProperty("grain", 0, 0, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        x = src[..., :3].astype(np.float64) / 255.0
        diag = math.hypot(w, h)

        # White balance and exposure: in linear light.
        lin = to_linear(x)
        if t["temperature"] or t["tint"]:
            k, g = t["temperature"] / 100.0, t["tint"] / 100.0
            gain = np.array([1 + 0.35 * k, 1 - 0.25 * g, 1 - 0.35 * k])
            gain /= 0.2126 * gain[0] + 0.7152 * gain[1] + 0.0722 * gain[2]
            lin = lin * gain
        if t["exposure"]:
            lin = lin * (2.0 ** t["exposure"])
        x = to_srgb(lin)

        if t["dehaze"]:
            x = dehaze(np.clip(x, 0, 1), t["dehaze"] / 100.0, radius=max(3, int(diag / 300)))

        # Highlights and shadows: by LOCAL brightness, so a dark detail in a
        # bright area is not lifted with the shadows.
        if t["highlights"] or t["shadows"]:
            local = np.clip(_blur(_lum(np.clip(x, 0, 1)), diag * 0.01), 0, 1)
            # Up to 2.5 stops into the shadows, 2 out of the highlights.
            stops = (t["shadows"] / 100.0) * 2.5 * (1 - local) ** 2 \
                + (t["highlights"] / 100.0) * 2.0 * local ** 2
            x = to_srgb(to_linear(np.clip(x, 0, None)) * (2.0 ** stops)[..., None])

        if t["whites"] or t["blacks"]:
            black = -t["blacks"] / 100.0 * 0.15
            white = 1.0 - t["whites"] / 100.0 * 0.15
            x = (x - black) / max(white - black, 1e-3)

        if t["contrast"]:
            c = t["contrast"] / 100.0
            y = np.clip(x, 0, 1)
            if c > 0:
                s = y * y * (3 - 2 * y)              # smoothstep S-curve
                x = x + (s - y) * c * 1.6
            else:
                x = 0.5 + (x - 0.5) * (1 + c * 0.7)

        if t["texture"]:
            x = x + (x - _blur(x, 1.5)) * (t["texture"] / 100.0) * 1.5
        if t["clarity"]:
            y = np.clip(x, 0, 1)
            mid = 4 * y * (1 - y)
            x = x + (x - _blur(x, diag * 0.015)) * mid * (t["clarity"] / 100.0)

        if t["vibrance"] or t["saturation"]:
            y = np.clip(x, 0, 1)
            grey = _lum(y)[..., None]
            sat = (y.max(axis=-1) - y.min(axis=-1))[..., None]
            k = t["saturation"] / 100.0 + t["vibrance"] / 100.0 * (1 - sat) ** 2 * 1.2
            x = grey + (y - grey) * (1 + k)

        if t["vignette"]:
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
            r = np.hypot((xs + 0.5 - w / 2.0) / (w / 2.0), (ys + 0.5 - h / 2.0) / (h / 2.0)) / math.sqrt(2)
            fall = np.clip((r - 0.35) / 0.65, 0, 1) ** 2
            v = t["vignette"] / 100.0
            x = x * (1 + v * fall)[..., None] if v < 0 else x + (1 - x) * (v * fall)[..., None]

        if t["grain"]:
            rng = np.random.default_rng(12345)
            noise = _blur(rng.standard_normal((h, w)), 0.6)
            x = x + (noise * (t["grain"] / 100.0) * 0.08)[..., None]

        out = src.copy()
        out[..., :3] = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
        return out


# ---- Lens Correction ----------------------------------------------------------------

class LensCorrection(_Whole):
    """Photoshop's Filter > Lens Correction, Custom tab."""

    name, label, category = "lens_correction", "Lens Correction", "Photo"

    def declare(self):
        return _props(
            IntProperty("distortion", 0, -100, 100, label="Remove distortion"),
            IntProperty("red_cyan", 0, -100, 100, label="Fix red/cyan fringe"),
            IntProperty("blue_yellow", 0, -100, 100, label="Fix blue/yellow fringe"),
            IntProperty("vignette", 0, -100, 100, label="Vignette amount"),
            IntProperty("midpoint", 50, 0, 100, label="Vignette midpoint"),
            IntProperty("vertical", 0, -100, 100, label="Vertical perspective"),
            IntProperty("horizontal", 0, -100, 100, label="Horizontal perspective"),
            FloatProperty("angle", 0.0, -180.0, 180.0, decimals=1),
            IntProperty("scale", 100, 50, 200, label="Scale (%)"),
            ChoiceProperty("edges", "transparent", ["transparent", "edge"],
                           labels={"transparent": "Transparency", "edge": "Edge extension"}))

    def _source_coords(self, t, h, w):
        """For each output pixel, where it comes from (before the per-channel
        fringe scaling): inverse of scale, rotation, perspective and the
        distortion removal, in that order."""
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        cx, cy = w / 2.0, h / 2.0
        norm = math.hypot(cx, cy)
        u = (xs + 0.5 - cx) / norm
        v = (ys + 0.5 - cy) / norm
        s = t["scale"] / 100.0
        u, v = u / s, v / s
        a = math.radians(t["angle"])
        ca, sa = math.cos(a), math.sin(a)
        u, v = ca * u + sa * v, -sa * u + ca * v
        # Perspective: tilting the image plane. A vertical correction makes
        # the top wider than the bottom (or the reverse), straightening
        # converging verticals.
        pv, ph = t["vertical"] / 100.0 * 0.6, t["horizontal"] / 100.0 * 0.6
        den = 1.0 + pv * v + ph * u
        den = np.where(np.abs(den) < 1e-6, 1e-6, den)
        u, v = u / den, v / den
        k = -t["distortion"] / 100.0 * 0.35
        r2 = u * u + v * v
        f = 1 + k * r2
        return u * f, v * f, cx, cy, norm

    def compute(self, t, src):
        h, w = src.shape[:2]
        u, v, cx, cy, norm = self._source_coords(t, h, w)
        edge = t["edges"]

        def sample(scale):
            sx, sy = cx + u * scale * norm, cy + v * scale * norm
            if edge == "edge":
                sx, sy = np.clip(sx, 0.5, w - 0.5), np.clip(sy, 0.5, h - 0.5)
            return warp_field(src, sx, sy)

        green = sample(1.0)
        out = green.copy()
        if t["red_cyan"]:
            out[..., 0] = sample(1.0 + t["red_cyan"] / 100.0 * 0.01)[..., 0]
        if t["blue_yellow"]:
            out[..., 2] = sample(1.0 + t["blue_yellow"] / 100.0 * 0.01)[..., 2]
        if t["vignette"]:
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
            r = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy) / norm
            mid = t["midpoint"] / 100.0
            fall = np.clip((r - mid * 0.7) / max(1e-3, 1 - mid * 0.7), 0, 1) ** 2
            gain = 1 + t["vignette"] / 100.0 * fall
            rgb = out[..., :3].astype(np.float64) / 255.0
            rgb = to_srgb(to_linear(rgb) * gain[..., None])
            out[..., :3] = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        return out


# ---- Adaptive Wide Angle --------------------------------------------------------------

class AdaptiveWideAngle(_Whole):
    """Re-project a wide-angle photograph (Photoshop's Adaptive Wide Angle,
    without its constraint lines)."""

    name, label, category = "adaptive_wide_angle", "Adaptive Wide Angle", "Photo"

    def declare(self):
        return _props(
            ChoiceProperty("correction", "fisheye", ["fisheye", "cylindrical", "stereographic"],
                           labels={"fisheye": "Fisheye to rectilinear",
                                   "cylindrical": "Wide angle to cylindrical",
                                   "stereographic": "Wide angle to stereographic"}),
            FloatProperty("focal", 15.0, 4.0, 100.0, decimals=1, label="Focal length (mm)"),
            FloatProperty("crop_factor", 1.0, 0.5, 10.0, decimals=2, label="Crop factor"),
            IntProperty("scale", 100, 25, 300, label="Scale (%)"))

    def compute(self, t, src):
        h, w = src.shape[:2]
        # Focal length in pixels: a full-frame sensor is 36 mm wide.
        f = t["focal"] * t["crop_factor"] / 36.0 * w
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        cx, cy = w / 2.0, h / 2.0
        s = t["scale"] / 100.0
        x = (xs + 0.5 - cx) / s
        y = (ys + 0.5 - cy) / s
        kind = t["correction"]
        if kind == "fisheye":
            # Output is rectilinear (r = f tan theta); the source is an
            # equidistant fisheye (r = f theta).
            r = np.hypot(x, y)
            theta = np.arctan(r / f)
            k = np.where(r > 1e-9, f * theta / np.maximum(r, 1e-9), 1.0)
            sx, sy = x * k, y * k
        elif kind == "cylindrical":
            # Output is cylindrical: x = f phi, y = f h. The source is
            # rectilinear: X = f tan phi, Y = f h / cos phi.
            phi = np.clip(x / f, -1.5, 1.5)
            sx = f * np.tan(phi)
            sy = y / np.cos(phi)
        else:
            # Output stereographic (r = 2 f tan(theta / 2)); source rectilinear.
            r = np.hypot(x, y)
            theta = np.clip(2 * np.arctan(r / (2 * f)), 0, 1.5)
            k = np.where(r > 1e-9, f * np.tan(theta) / np.maximum(r, 1e-9), 1.0)
            sx, sy = x * k, y * k
        return warp_field(src, cx + sx, cy + sy)


# ---- Filter Gallery -----------------------------------------------------------------

class FilterStack(Effect):
    """Several effects applied in order, as one: the Filter Gallery's stack.
    `stack` is JSON: [{"name": ..., "params": {...}, "enabled": true}, ...],
    so the whole stack is one action step and one smart filter."""

    name, label, category = "filter_stack", "Filter Gallery", "Artistic"

    def declare(self):
        return _props(TextProperty("stack", "[]"))

    @staticmethod
    def parse(text):
        try:
            items = json.loads(text) if isinstance(text, str) else list(text or [])
        except ValueError:
            return []
        return [i for i in items if isinstance(i, dict) and i.get("name")]

    def prepare(self, token, source):
        from .effects import EffectRegistry
        key = (token.get("stack"), source.shape, source.ctypes.data, int(source[::17, ::13].sum()))
        if getattr(self, "_key", None) == key:
            return
        reg = getattr(self, "registry", None) or EffectRegistry()
        img = np.array(source, copy=True)
        for item in self.parse(token.get("stack")):
            if not item.get("enabled", True) or item["name"] == self.name:
                continue
            fx = reg.create(item["name"], **dict(item.get("params") or {}))
            if fx is not None:
                img = fx.apply(img)
        self._result = img
        self._key = key

    def render_tile(self, src, dst, rect):
        dst[rect.slice()] = self._result[rect.slice()]


PHOTO = (CameraRaw, LensCorrection, AdaptiveWideAngle, FilterStack)
