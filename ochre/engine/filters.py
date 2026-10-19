# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Filters: Paint.NET's Effects menu, GIMP's Filters, Photoshop's Filter menu.

Categories follow Paint.NET's menus, which are the shallowest of the three:
Blurs, Photo, Noise, Distort, Stylize, Artistic, Render, and Edge & Generic.
Where the three editors have the same filter under different names, the
docstring says so.

Every filter here computes over the WHOLE image once per setting and the
tiles copy from that result. That costs some time on a huge image, but it
makes a halo impossible to get wrong -- a distortion can reach anywhere,
and a tile-by-tile version of Twirl or Polar Coordinates would need the
entire source anyway. The result is cached per setting, so dragging a
slider in the dialog re-renders only when the value actually changes.

Conventions shared by all of them:

  * Anything that averages pixels does it premultiplied (no dark fringes at
    transparent edges), via _pm / _unpm.
  * Geometric filters resample through warp.warp_field -- inverse mapping,
    alpha-weighted bilinear -- with the edge behaviour a filter needs.
  * Randomness is seeded and positional, so a filter gives the same result
    every time, in the preview and on OK.
"""

import math

import numpy as np

from .effects import Effect
from .props import (AngleProperty, BoolProperty, ChoiceProperty, ColorProperty,
                    FloatProperty, IntProperty, PropertyCollection)
from .warp import warp_field


# ---- shared machinery ----------------------------------------------------------------

def _pm(src):
    a = src[..., 3:4].astype(np.float32) / 255.0
    out = np.empty(src.shape, np.float32)
    out[..., :3] = src[..., :3].astype(np.float32) * a
    out[..., 3:] = src[..., 3:4].astype(np.float32)
    return out


def _unpm(work):
    a = np.clip(work[..., 3], 0, 255)
    safe = np.where(a > 1e-3, a / 255.0, 1.0)
    out = np.empty(work.shape, np.uint8)
    rgb = np.clip(work[..., :3] / safe[..., None], 0, 255)
    rgb[a <= 1e-3] = 0
    out[..., :3] = np.rint(rgb)
    out[..., 3] = np.rint(a)
    return out


def _box(a, r, axis):
    """Mean over a (2r+1) window along an axis, edges clamped."""
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def _gauss(a, radius):
    """Gaussian approximation: three box passes per axis (rel. error ~3%)."""
    radius = float(radius)
    if radius < 0.5:
        return a
    # Box width giving the requested standard deviation over three passes.
    sigma = radius / 2.0
    r = max(1, int(round(math.sqrt(12 * sigma * sigma / 3 + 1) / 2)))
    out = a
    for _ in range(3):
        out = _box(_box(out, r, 0), r, 1)
    return out


def _blur_rgba(src, radius):
    return _unpm(_gauss(_pm(src), radius))


def _lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _grid(h, w):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return xs + 0.5, ys + 0.5


def _edge(sx, sy, w, h, mode):
    """Bring sample coordinates back onto the canvas: clamp, wrap or leave."""
    if mode == "clamp":
        return np.clip(sx, 0.5, w - 0.5), np.clip(sy, 0.5, h - 0.5)
    if mode == "wrap":
        return np.mod(sx, w), np.mod(sy, h)
    if mode == "reflect":
        def refl(v, n):
            v = np.mod(v, 2 * n)
            return np.where(v > n, 2 * n - v, v)
        return refl(sx, w), refl(sy, h)
    return sx, sy                       # transparent beyond the edge


def _warp(src, sx, sy, edge="clamp"):
    h, w = src.shape[:2]
    sx, sy = _edge(sx, sy, w, h, edge)
    return warp_field(src, sx, sy)


def _hash01(xx, yy, seed):
    """Positional white noise in [0, 1): the same pixel, the same value."""
    v = (np.asarray(xx, np.uint64) * np.uint64(73856093)) ^ \
        (np.asarray(yy, np.uint64) * np.uint64(19349663)) ^ np.uint64((seed * 83492791) & 0xFFFFFFFF)
    v = (v ^ (v >> np.uint64(13))) * np.uint64(1274126177)
    v = v ^ (v >> np.uint64(16))
    return (v & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def _value_noise(h, w, scale, seed, octaves=6, roughness=0.5):
    """Fractal (fBm) value noise in [0, 1]: the basis of Clouds."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    total = np.zeros((h, w))
    amp, norm = 1.0, 0.0
    freq = 1.0 / max(1.0, scale)
    for o in range(octaves):
        x, y = xs * freq, ys * freq
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        x0i, y0i = x0.astype(np.int64) + 10000, y0.astype(np.int64) + 10000
        s = seed + 1013 * o
        v00 = _hash01(x0i, y0i, s)
        v10 = _hash01(x0i + 1, y0i, s)
        v01 = _hash01(x0i, y0i + 1, s)
        v11 = _hash01(x0i + 1, y0i + 1, s)
        n = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy
        total += n * amp
        norm += amp
        amp *= roughness
        freq *= 2.0
    return total / norm


def _shifted(a, dy, dx):
    """a moved by whole pixels, edges clamped (for small stencils)."""
    h, w = a.shape[:2]
    ys = np.clip(np.arange(h) - dy, 0, h - 1)
    xs = np.clip(np.arange(w) - dx, 0, w - 1)
    return a[ys][:, xs]


def _convolve(a, kernel):
    """Small-kernel 2-D convolution by shifted sums, edges clamped."""
    k = np.asarray(kernel, np.float64)
    kh, kw = k.shape
    out = np.zeros(a.shape, np.float64)
    for j in range(kh):
        for i in range(kw):
            if k[j, i]:
                out += k[j, i] * _shifted(a, kh // 2 - j, kw // 2 - i)
    return out


def _fft_convolve(a, kernel):
    """Large-kernel convolution of a float (h, w, c) image, edges clamped."""
    k = np.asarray(kernel, np.float64)
    kh, kw = k.shape
    ph, pw = kh // 2 + 1, kw // 2 + 1
    padded = np.pad(a, ((ph, ph), (pw, pw), (0, 0)), mode="edge")
    H, W = padded.shape[:2]
    kpad = np.zeros((H, W))
    kpad[:kh, :kw] = k
    kpad = np.roll(kpad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    fk = np.fft.rfft2(kpad)
    out = np.empty(padded.shape)
    for c in range(a.shape[2]):
        out[..., c] = np.fft.irfft2(np.fft.rfft2(padded[..., c]) * fk, s=(H, W))
    return out[ph:ph + a.shape[0], pw:pw + a.shape[1]].astype(np.float32)


def _rank(plane, r, kind):
    """Square min / max / median of a 2-D plane over (2r+1)^2, chunked."""
    h, w = plane.shape
    if kind in ("min", "max"):
        fn = np.minimum if kind == "min" else np.maximum
        out = plane
        for axis in (0, 1):
            acc = out
            for d in range(1, r + 1):
                acc = fn(acc, _shifted(out, d, 0) if axis == 0 else _shifted(out, 0, d))
                acc = fn(acc, _shifted(out, -d, 0) if axis == 0 else _shifted(out, 0, -d))
            out = acc
        return out
    p = np.pad(plane, r, mode="edge")
    out = np.empty_like(plane)
    rows = max(1, int(4_000_000 // max(1, w * (2 * r + 1) ** 2)))
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        stack = [p[y0 + dy:y1 + dy, dx:dx + w] for dy in range(2 * r + 1) for dx in range(2 * r + 1)]
        out[y0:y1] = np.median(np.stack(stack), axis=0)
    return out


class _Filter(Effect):
    """Computes the whole image once per setting; tiles copy from that."""

    category = "Effects"

    def compute(self, token, src):
        raise NotImplementedError

    def prepare(self, token, source):
        # Same settings, same buffer, same contents (a sparse checksum): reuse.
        key = (repr(sorted(token.items())), source.shape, source.ctypes.data,
               int(source[::17, ::13].sum()))
        if getattr(self, "_key", None) != key:
            self._result = self.compute(token, source)
            self._key = key

    def render_tile(self, src, dst, rect):
        dst[rect.slice()] = self._result[rect.slice()]


def _props(*items):
    return PropertyCollection(list(items))


def _seed():
    return IntProperty("seed", 0, 0, 9999)


def _centre():
    return (IntProperty("center_x", 50, 0, 100, label="Center X (%)"),
            IntProperty("center_y", 50, 0, 100, label="Center Y (%)"))


def _edge_choice(default="clamp"):
    return ChoiceProperty("edges", default, ["clamp", "wrap", "reflect", "transparent"],
                          labels={"clamp": "Clamp", "wrap": "Wrap", "reflect": "Reflect",
                                  "transparent": "Transparent"})


# ==== Blurs ==========================================================================

class MotionBlur(_Filter):
    """Blur along a line, as if the camera moved. Centered, as in Paint.NET."""
    name, label, category = "motion_blur", "Motion Blur", "Blurs"

    def declare(self):
        return _props(FloatProperty("angle", 0.0, -180.0, 180.0, decimals=1), IntProperty("distance", 10, 1, 200),
                      BoolProperty("centered", True))

    def compute(self, t, src):
        h, w = src.shape[:2]
        n = max(2, min(64, int(t["distance"])))
        a = math.radians(t["angle"])
        ux, uy = math.cos(a), -math.sin(a)
        xs, ys = _grid(h, w)
        acc = np.zeros(src.shape, np.float32)
        for i in range(n):
            f = i / (n - 1.0) - (0.5 if t["centered"] else 0.0)
            d = f * t["distance"]
            acc += _pm(_warp(src, xs - ux * d, ys - uy * d))
        return _unpm(acc / n)


class RadialBlur(_Filter):
    """Spin blur around a point: Paint.NET's Radial Blur, Photoshop's Spin."""
    name, label, category = "radial_blur", "Radial Blur", "Blurs"

    def declare(self):
        return _props(FloatProperty("angle", 10.0, 0.0, 180.0, decimals=1), *_centre(), IntProperty("quality", 3, 1, 5))

    def compute(self, t, src):
        h, w = src.shape[:2]
        cx, cy = w * t["center_x"] / 100.0, h * t["center_y"] / 100.0
        xs, ys = _grid(h, w)
        n = 4 + 6 * t["quality"]
        acc = np.zeros(src.shape, np.float32)
        for i in range(n):
            th = math.radians(t["angle"]) * (i / (n - 1.0) - 0.5)
            c, s = math.cos(th), math.sin(th)
            dx, dy = xs - cx, ys - cy
            acc += _pm(_warp(src, cx + c * dx - s * dy, cy + s * dx + c * dy))
        return _unpm(acc / n)


class ZoomBlur(_Filter):
    """Streaks toward a point, as if zooming during the exposure."""
    name, label, category = "zoom_blur", "Zoom Blur", "Blurs"

    def declare(self):
        return _props(IntProperty("amount", 20, 0, 100), *_centre())

    def compute(self, t, src):
        h, w = src.shape[:2]
        cx, cy = w * t["center_x"] / 100.0, h * t["center_y"] / 100.0
        xs, ys = _grid(h, w)
        n = 24
        acc = np.zeros(src.shape, np.float32)
        for i in range(n):
            k = 1.0 - (t["amount"] / 100.0) * 0.5 * i / (n - 1.0)
            acc += _pm(_warp(src, cx + (xs - cx) * k, cy + (ys - cy) * k))
        return _unpm(acc / n)


class BoxBlur(_Filter):
    name, label, category = "box_blur", "Box Blur", "Blurs"

    def declare(self):
        return _props(IntProperty("radius", 4, 1, 200))

    def compute(self, t, src):
        r = t["radius"]
        return _unpm(_box(_box(_pm(src), r, 0), r, 1))


class LensBlur(_Filter):
    """Disc-shaped (bokeh) blur: bright points become discs, not smudges.
    Paint.NET's Unfocus / Bokeh, Photoshop's Lens Blur (without depth)."""
    name, label, category = "lens_blur", "Lens Blur (Bokeh)", "Blurs"

    def declare(self):
        return _props(IntProperty("radius", 8, 1, 100), IntProperty("highlights", 0, 0, 100,
                                                                    label="Highlight boost"))

    def compute(self, t, src):
        r = t["radius"]
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disc = np.clip(r + 0.5 - np.hypot(xx, yy), 0, 1)
        disc /= disc.sum()
        work = _pm(src)
        boost = t["highlights"] / 100.0
        if boost:
            # Expand bright values before blurring so they dominate their disc.
            work[..., :3] = work[..., :3] * (1 + boost * 3 * (work[..., :3] / 255.0) ** 4)
        out = _fft_convolve(work, disc)
        if boost:
            out[..., :3] = np.clip(out[..., :3], 0, 255 * 4)
        return _unpm(np.clip(out, 0, None))


class SurfaceBlur(_Filter):
    """Smooths surfaces, keeps edges: an edge-aware (bilateral) blur, done as
    two 1-D passes. Photoshop's Surface Blur, Paint.NET's Surface Blur."""
    name, label, category = "surface_blur", "Surface Blur", "Blurs"

    def declare(self):
        return _props(IntProperty("radius", 6, 1, 50), IntProperty("threshold", 20, 1, 100))

    def compute(self, t, src):
        work = _pm(src).astype(np.float64)
        r, thr = t["radius"], float(t["threshold"])
        sig_s = r / 2.0
        for axis in (1, 0):
            acc = np.zeros_like(work)
            wsum = np.zeros(work.shape[:2])
            ref = _lum(work)
            for d in range(-r, r + 1):
                sh = _shifted(work, 0, d) if axis == 1 else _shifted(work, d, 0)
                diff = np.abs(_lum(sh) - ref)
                wgt = np.exp(-(d * d) / (2 * sig_s * sig_s)) * np.clip(1 - diff / (2.5 * thr), 0, 1)
                acc += sh * wgt[..., None]
                wsum += wgt
            work = acc / np.maximum(wsum, 1e-9)[..., None]
        return _unpm(work.astype(np.float32))


class TiltShift(_Filter):
    """A sharp horizontal band with blur growing above and below it, which
    makes a scene look like a miniature."""
    name, label, category = "tilt_shift", "Tilt-Shift", "Blurs"

    def declare(self):
        return _props(IntProperty("radius", 10, 1, 60), IntProperty("center_y", 50, 0, 100,
                                                                    label="Band center (%)"),
                      IntProperty("band", 20, 0, 100, label="Sharp band (%)"),
                      IntProperty("falloff", 25, 1, 100, label="Falloff (%)"))

    def compute(self, t, src):
        h = src.shape[0]
        blurred = _pm(_blur_rgba(src, t["radius"]))
        sharp = _pm(src)
        y = (np.arange(h) + 0.5) / h * 100.0
        d = np.abs(y - t["center_y"]) - t["band"] / 2.0
        k = np.clip(d / t["falloff"], 0, 1)[:, None, None].astype(np.float32)
        return _unpm(sharp + (blurred - sharp) * k)


class FragmentBlur(_Filter):
    """Paint.NET's Fragment: copies of the image offset round a circle."""
    name, label, category = "fragment", "Fragment", "Blurs"

    def declare(self):
        return _props(IntProperty("fragments", 4, 2, 50), IntProperty("distance", 8, 0, 100),
                      FloatProperty("rotation", 0.0, 0.0, 360.0, decimals=1))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        n = t["fragments"]
        acc = np.zeros(src.shape, np.float32)
        for i in range(n):
            a = math.radians(t["rotation"]) + 2 * math.pi * i / n
            acc += _pm(_warp(src, xs + math.cos(a) * t["distance"], ys + math.sin(a) * t["distance"]))
        return _unpm(acc / n)


class Pixelate(_Filter):
    """Big square cells of averaged colour: Pixelate / Mosaic."""
    name, label, category = "pixelate", "Pixelate", "Distort"

    def declare(self):
        return _props(IntProperty("cell", 8, 2, 200, label="Cell size"))

    def compute(self, t, src):
        h, w = src.shape[:2]
        c = t["cell"]
        gh, gw = -(-h // c), -(-w // c)
        work = np.pad(_pm(src), ((0, gh * c - h), (0, gw * c - w), (0, 0)), mode="edge")
        means = work.reshape(gh, c, gw, c, 4).mean(axis=(1, 3))
        out = np.repeat(np.repeat(means, c, axis=0), c, axis=1)[:h, :w]
        return _unpm(out)


# ==== Photo ============================================================================

class UnsharpMask(_Filter):
    """Sharpen by adding back what a blur removes; threshold spares noise."""
    name, label, category = "unsharp_mask", "Unsharp Mask", "Photo"

    def declare(self):
        return _props(FloatProperty("radius", 2.0, 0.1, 100.0, decimals=1),
                      IntProperty("amount", 80, 1, 500), IntProperty("threshold", 0, 0, 255))

    def compute(self, t, src):
        sharp = src.astype(np.float32)
        blur = _blur_rgba(src, t["radius"]).astype(np.float32)
        diff = sharp[..., :3] - blur[..., :3]
        keep = np.abs(diff).max(axis=-1, keepdims=True) >= t["threshold"]
        out = src.copy()
        out[..., :3] = np.clip(np.rint(sharp[..., :3] + diff * (t["amount"] / 100.0) * keep), 0, 255)
        return out


class HighPass(_Filter):
    """Keep only fine detail, around mid-grey: blend it Overlay to sharpen."""
    name, label, category = "high_pass", "High Pass", "Photo"

    def declare(self):
        return _props(FloatProperty("radius", 10.0, 0.1, 250.0, decimals=1))

    def compute(self, t, src):
        blur = _blur_rgba(src, t["radius"]).astype(np.float32)
        out = src.copy()
        out[..., :3] = np.clip(np.rint(src[..., :3].astype(np.float32) - blur[..., :3] + 128), 0, 255)
        return out


class Vignette(_Filter):
    """Darken (or lighten, or tint) toward the edges."""
    name, label, category = "vignette", "Vignette", "Photo"

    def declare(self):
        return _props(IntProperty("amount", 60, 0, 100), IntProperty("radius", 70, 10, 150),
                      IntProperty("softness", 50, 1, 100), *_centre(),
                      ColorProperty("color", (0, 0, 0, 255)))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        cx, cy = w * t["center_x"] / 100.0, h * t["center_y"] / 100.0
        d = np.hypot((xs - cx) / (w / 2.0), (ys - cy) / (h / 2.0)) * 100.0 / t["radius"]
        soft = t["softness"] / 100.0
        k = np.clip((d - (1 - soft)) / max(soft, 1e-3), 0, 1) ** 2 * t["amount"] / 100.0
        c = np.array(t["color"][:3], np.float32)
        out = src.copy()
        rgb = src[..., :3].astype(np.float32)
        out[..., :3] = np.clip(np.rint(rgb + (c - rgb) * k[..., None]), 0, 255)
        return out


class Glow(_Filter):
    """Paint.NET's Glow / GIMP's Softglow: a blurred, brightened copy screened on."""
    name, label, category = "glow", "Glow", "Photo"

    def declare(self):
        return _props(IntProperty("radius", 8, 1, 50), IntProperty("brightness", 10, -100, 100),
                      IntProperty("contrast", 10, -100, 100))

    def compute(self, t, src):
        b = _blur_rgba(src, t["radius"]).astype(np.float32)[..., :3]
        b = (b - 127.5) * (1 + t["contrast"] / 100.0) + 127.5 + t["brightness"] * 2.55
        b = np.clip(b, 0, 255) / 255.0
        a = src[..., :3].astype(np.float32) / 255.0
        out = src.copy()
        out[..., :3] = np.clip(np.rint((1 - (1 - a) * (1 - b)) * 255), 0, 255)
        return out


class SoftenPortrait(_Filter):
    """Paint.NET's Soften Portrait: soft skin, a touch of light, some warmth."""
    name, label, category = "soften_portrait", "Soften Portrait", "Photo"

    def declare(self):
        return _props(IntProperty("softness", 5, 0, 10), IntProperty("lighting", 0, -20, 20),
                      IntProperty("warmth", 10, 0, 20))

    def compute(self, t, src):
        soft = _blur_rgba(src, 1 + t["softness"] * 1.5).astype(np.float32)
        base = src.astype(np.float32)
        mix = base[..., :3] * 0.5 + soft[..., :3] * 0.5
        mix += t["lighting"] * 2.0
        warm = t["warmth"] / 100.0
        mix[..., 0] *= 1 + warm
        mix[..., 2] *= 1 - warm
        out = src.copy()
        out[..., :3] = np.clip(np.rint(mix), 0, 255)
        return out


class ShadowsHighlights(_Filter):
    """Lift shadows and recover highlights by local brightness, not by level:
    Photoshop's Shadows/Highlights, GIMP's Shadows-Highlights."""
    name, label, category = "shadows_highlights", "Shadows / Highlights", "Photo"

    def declare(self):
        return _props(IntProperty("shadows", 35, 0, 100), IntProperty("highlights", 0, 0, 100),
                      IntProperty("radius", 30, 1, 200))

    def compute(self, t, src):
        rgb = src[..., :3].astype(np.float32)
        local = _gauss(_lum(rgb)[..., None].astype(np.float32), t["radius"])[..., 0] / 255.0
        lift = (1 - local) ** 2 * t["shadows"] / 100.0
        drop = local ** 2 * t["highlights"] / 100.0
        x = rgb / 255.0
        out_rgb = x + lift[..., None] * (1 - x) * x * 2 - drop[..., None] * x * (1 - x) * 2
        out = src.copy()
        out[..., :3] = np.clip(np.rint(out_rgb * 255), 0, 255)
        return out


# ==== Noise ============================================================================

class Median(_Filter):
    """Each pixel becomes the median (or another percentile) of its square."""
    name, label, category = "median", "Median", "Noise"

    def declare(self):
        return _props(IntProperty("radius", 2, 1, 10), IntProperty("percentile", 50, 0, 100))

    def compute(self, t, src):
        r = t["radius"]
        out = np.empty_like(src)
        if t["percentile"] == 50:
            for c in range(4):
                out[..., c] = _rank(src[..., c], r, "median")
        else:
            p = np.pad(src, ((r, r), (r, r), (0, 0)), mode="edge")
            h, w = src.shape[:2]
            stack = np.stack([p[dy:dy + h, dx:dx + w] for dy in range(2 * r + 1)
                              for dx in range(2 * r + 1)])
            out = np.percentile(stack, t["percentile"], axis=0).astype(np.uint8)
        return out


class DustScratches(_Filter):
    """A median, used only where a pixel differs from it by more than the
    threshold: removes specks and scratches, leaves texture."""
    name, label, category = "dust_scratches", "Dust & Scratches", "Noise"

    def declare(self):
        return _props(IntProperty("radius", 2, 1, 10), IntProperty("threshold", 20, 0, 255))

    def compute(self, t, src):
        med = np.empty_like(src)
        for c in range(4):
            med[..., c] = _rank(src[..., c], t["radius"], "median")
        diff = np.abs(src[..., :3].astype(np.int32) - med[..., :3].astype(np.int32)).max(axis=-1)
        out = src.copy()
        out[diff > t["threshold"]] = med[diff > t["threshold"]]
        return out


class ReduceNoise(_Filter):
    """Smooth flat areas, keep edges: a gentle edge-aware blur."""
    name, label, category = "reduce_noise", "Reduce Noise", "Noise"

    def declare(self):
        return _props(IntProperty("strength", 50, 1, 100))

    def compute(self, t, src):
        sb = SurfaceBlur()
        return sb.compute({"radius": 2 + t["strength"] // 20, "threshold": 5 + t["strength"] // 3}, src)


class Spread(_Filter):
    """Scatter pixels randomly: GIMP's Spread and Hurl's gentler cousin,
    Photoshop's Diffuse."""
    name, label, category = "spread", "Spread (Diffuse)", "Noise"

    def declare(self):
        return _props(IntProperty("amount", 5, 1, 100), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        a = t["amount"]
        dx = np.rint((_hash01(xx, yy, t["seed"]) * 2 - 1) * a).astype(np.int64)
        dy = np.rint((_hash01(xx, yy, t["seed"] + 7) * 2 - 1) * a).astype(np.int64)
        return src[np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]


class FrostedGlass(_Filter):
    """Paint.NET's Frosted Glass: scattered, then softened."""
    name, label, category = "frosted_glass", "Frosted Glass", "Distort"

    def declare(self):
        return _props(IntProperty("amount", 6, 1, 50), _seed())

    def compute(self, t, src):
        return _blur_rgba(Spread().compute({"amount": t["amount"], "seed": t["seed"]}, src), 1.0)


# ==== Distort ===========================================================================

def _centre_px(t, w, h):
    return w * t.get("center_x", 50) / 100.0, h * t.get("center_y", 50) / 100.0


class Twirl(_Filter):
    """Swirl round the centre, most at the middle: Twist / Whirl / Twirl."""
    name, label, category = "twirl", "Twirl", "Distort"

    def declare(self):
        return _props(FloatProperty("angle", 90.0, -720.0, 720.0, decimals=1), IntProperty("radius", 100, 1, 200,
                                                                                label="Radius (%)"),
                      *_centre())

    def compute(self, t, src):
        h, w = src.shape[:2]
        cx, cy = _centre_px(t, w, h)
        xs, ys = _grid(h, w)
        R = min(w, h) / 2.0 * t["radius"] / 100.0
        dx, dy = xs - cx, ys - cy
        d = np.hypot(dx, dy)
        k = np.clip(1 - d / max(R, 1e-6), 0, 1) ** 2
        th = math.radians(t["angle"]) * k
        c, s = np.cos(th), np.sin(th)
        return _warp(src, cx + c * dx - s * dy, cy + s * dx + c * dy)


class PinchBulge(_Filter):
    """Pull the middle in (pinch, negative) or push it out (bulge, positive)."""
    name, label, category = "bulge", "Bulge / Pinch", "Distort"

    def declare(self):
        return _props(IntProperty("amount", 50, -100, 100), IntProperty("radius", 100, 1, 200,
                                                                         label="Radius (%)"),
                      *_centre())

    def compute(self, t, src):
        h, w = src.shape[:2]
        cx, cy = _centre_px(t, w, h)
        xs, ys = _grid(h, w)
        R = min(w, h) / 2.0 * t["radius"] / 100.0
        dx, dy = xs - cx, ys - cy
        r = np.hypot(dx, dy) / max(R, 1e-6)
        inside = r < 1
        amount = t["amount"] / 100.0
        scale = np.where(inside, np.power(np.clip(r, 1e-6, 1), amount * 0.8), 1.0)
        return _warp(src, cx + dx * scale, cy + dy * scale)


class Spherize(PinchBulge):
    """Wrap the image round a sphere (Photoshop's Spherize)."""
    name, label = "spherize", "Spherize"

    def compute(self, t, src):
        h, w = src.shape[:2]
        cx, cy = _centre_px(t, w, h)
        xs, ys = _grid(h, w)
        R = min(w, h) / 2.0 * t["radius"] / 100.0
        dx, dy = (xs - cx) / R, (ys - cy) / R
        r = np.hypot(dx, dy)
        amount = t["amount"] / 100.0
        inside = r < 1
        rr = np.clip(r, 0, 0.999999)
        sphere = np.where(inside, np.arcsin(rr) / (math.pi / 2) / np.maximum(rr, 1e-6), 1.0)
        scale = 1 + (sphere - 1) * amount
        return _warp(src, cx + dx * R * scale, cy + dy * R * scale)


class Ripple(_Filter):
    """Waves across the image: GIMP's Ripple, Photoshop's Wave (sine)."""
    name, label, category = "ripple", "Ripple", "Distort"

    def declare(self):
        return _props(IntProperty("amplitude", 6, 0, 200), IntProperty("wavelength", 40, 2, 1000),
                      FloatProperty("phase", 0.0, 0.0, 360.0, decimals=1),
                      ChoiceProperty("direction", "horizontal", ["horizontal", "vertical", "both"]),
                      _edge_choice("reflect"))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        ph = math.radians(t["phase"])
        k = 2 * math.pi / t["wavelength"]
        sx, sy = xs.copy(), ys.copy()
        if t["direction"] in ("horizontal", "both"):
            sx = sx + t["amplitude"] * np.sin(ys * k + ph)
        if t["direction"] in ("vertical", "both"):
            sy = sy + t["amplitude"] * np.sin(xs * k + ph)
        return _warp(src, sx, sy, t["edges"])


class Waves(_Filter):
    """Concentric rings spreading from a point: GIMP's Waves, Photoshop's
    Ocean/Zigzag family."""
    name, label, category = "waves", "Waves", "Distort"

    def declare(self):
        return _props(IntProperty("amplitude", 6, 0, 100), IntProperty("wavelength", 30, 2, 500),
                      FloatProperty("phase", 0.0, 0.0, 360.0, decimals=1), *_centre())

    def compute(self, t, src):
        h, w = src.shape[:2]
        cx, cy = _centre_px(t, w, h)
        xs, ys = _grid(h, w)
        dx, dy = xs - cx, ys - cy
        d = np.hypot(dx, dy)
        off = t["amplitude"] * np.sin(d * 2 * math.pi / t["wavelength"] + math.radians(t["phase"]))
        safe = np.maximum(d, 1e-6)
        return _warp(src, xs + dx / safe * off, ys + dy / safe * off)


class PolarCoordinates(_Filter):
    """Rectangular to polar and back: GIMP's Polar Coordinates, which also
    makes "little planet" panoramas."""
    name, label, category = "polar", "Polar Coordinates", "Distort"

    def declare(self):
        return _props(ChoiceProperty("mode", "to_polar", ["to_polar", "from_polar"],
                                     labels={"to_polar": "Rectangular to polar",
                                             "from_polar": "Polar to rectangular"}))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        cx, cy = w / 2.0, h / 2.0
        R = math.hypot(cx, cy)
        if t["mode"] == "to_polar":
            dx, dy = xs - cx, ys - cy
            ang = (np.arctan2(dx, -dy) + math.pi) / (2 * math.pi)
            rad = np.hypot(dx, dy) / R
            return _warp(src, ang * w, rad * h, "clamp")
        ang = xs / w * 2 * math.pi - math.pi
        rad = ys / h * R
        return _warp(src, cx + rad * np.sin(ang), cy - rad * np.cos(ang), "transparent")


class PolarInversion(_Filter):
    """Paint.NET's Polar Inversion: z -> 1/z, the image turned inside out."""
    name, label, category = "polar_inversion", "Polar Inversion", "Distort"

    def declare(self):
        return _props(FloatProperty("amount", 1.0, 0.1, 4.0, decimals=2), _edge_choice("wrap"))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        cx, cy = w / 2.0, h / 2.0
        s = min(w, h) / 2.0
        zx, zy = (xs - cx) / s, (ys - cy) / s
        d2 = np.maximum(zx * zx + zy * zy, 1e-9)
        a = t["amount"]
        return _warp(src, cx + zx / d2 * a * s, cy + zy / d2 * a * s, t["edges"])


class LensDistortion(_Filter):
    """Barrel (positive) or pincushion (negative) lens distortion, for
    adding or correcting it: GIMP's Lens Distortion, Photoshop's correction."""
    name, label, category = "lens_distortion", "Lens Distortion", "Distort"

    def declare(self):
        return _props(IntProperty("main", 20, -100, 100), IntProperty("edge", 0, -100, 100),
                      IntProperty("zoom", 0, -100, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        cx, cy = w / 2.0, h / 2.0
        norm = math.hypot(cx, cy)
        dx, dy = (xs - cx) / norm, (ys - cy) / norm
        r2 = dx * dx + dy * dy
        k = 1 + t["main"] / 100.0 * r2 * 0.5 + t["edge"] / 100.0 * r2 * r2 * 0.5
        z = 2 ** (-t["zoom"] / 100.0)
        return _warp(src, cx + dx * k * z * norm, cy + dy * k * z * norm, "transparent")


class Kaleidoscope(_Filter):
    """Mirror one wedge round the centre, GIMP's Kaleidoscope."""
    name, label, category = "kaleidoscope", "Kaleidoscope", "Distort"

    def declare(self):
        return _props(IntProperty("segments", 6, 2, 32), FloatProperty("angle", 0.0, 0.0, 360.0, decimals=1),
                      *_centre())

    def compute(self, t, src):
        h, w = src.shape[:2]
        cx, cy = _centre_px(t, w, h)
        xs, ys = _grid(h, w)
        dx, dy = xs - cx, ys - cy
        r = np.hypot(dx, dy)
        seg = 2 * math.pi / t["segments"]
        a = np.mod(np.arctan2(dy, dx) - math.radians(t["angle"]), seg)
        a = np.where(a > seg / 2, seg - a, a) + math.radians(t["angle"])
        return _warp(src, cx + r * np.cos(a), cy + r * np.sin(a), "reflect")


class TileReflection(_Filter):
    """Paint.NET's Tile Reflection / GIMP's Glass Tile: a grid of glass
    blocks, each refracting its patch."""
    name, label, category = "tile_reflection", "Tile Reflection", "Distort"

    def declare(self):
        return _props(IntProperty("tile", 32, 4, 200, label="Tile size"),
                      IntProperty("curvature", 8, -50, 50), FloatProperty("angle", 30.0, -45.0, 45.0, decimals=1))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        a = math.radians(t["angle"])
        c, s = math.cos(a), math.sin(a)
        u, v = xs * c + ys * s, -xs * s + ys * c
        size = t["tile"]
        fu = np.mod(u, size) / size - 0.5
        fv = np.mod(v, size) / size - 0.5
        k = t["curvature"] / 10.0
        du, dv = np.sin(fu * math.pi) * k * size / 8, np.sin(fv * math.pi) * k * size / 8
        return _warp(src, xs + du * c - dv * s, ys + du * s + dv * c, "reflect")


class Dents(_Filter):
    """Paint.NET's Dents: a smooth random displacement, like hammered metal or
    water, from fractal noise."""
    name, label, category = "dents", "Dents", "Distort"

    def declare(self):
        return _props(IntProperty("scale", 30, 2, 400), IntProperty("refraction", 15, 0, 200),
                      IntProperty("roughness", 50, 0, 100), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        nx = _value_noise(h, w, t["scale"], t["seed"], 4, t["roughness"] / 100.0)
        ny = _value_noise(h, w, t["scale"], t["seed"] + 99, 4, t["roughness"] / 100.0)
        return _warp(src, xs + (nx - 0.5) * t["refraction"], ys + (ny - 0.5) * t["refraction"],
                     "reflect")


class Wind(_Filter):
    """Photoshop's Wind: bright detail streaks off in one direction."""
    name, label, category = "wind", "Wind", "Stylize"

    def declare(self):
        return _props(IntProperty("strength", 20, 1, 100),
                      ChoiceProperty("direction", "right", ["left", "right"]))

    def compute(self, t, src):
        work = src.astype(np.float32)
        step = 1 if t["direction"] == "right" else -1
        out = work.copy()
        decay = 1.0 - 1.0 / t["strength"]
        trail = work.copy()
        for _ in range(int(t["strength"])):
            trail = _shifted(trail, 0, step) * decay
            out = np.maximum(out, trail)
        res = src.copy()
        res[..., :3] = np.clip(np.rint(out[..., :3]), 0, 255)
        return res


class TileSeamless(_Filter):
    """GIMP's Tile Seamless: blend the image with itself shifted by half, so
    its edges meet when tiled."""
    name, label, category = "tile_seamless", "Tile Seamless", "Distort"

    def declare(self):
        return _props()

    def compute(self, t, src):
        h, w = src.shape[:2]
        shifted = np.roll(np.roll(src, h // 2, axis=0), w // 2, axis=1)
        xs, ys = _grid(h, w)
        # 1 in the middle, 0 at the edges: the edges come from the shifted
        # copy, whose own edges are the image's middle, so they wrap cleanly.
        wx = 1 - np.abs(xs / w * 2 - 1)
        wy = 1 - np.abs(ys / h * 2 - 1)
        k = np.clip(np.minimum(wx, wy) * 2, 0, 1)[..., None].astype(np.float32)
        a, b = _pm(src), _pm(shifted)
        return _unpm(b + (a - b) * k)


# ==== Stylize ===========================================================================

def _sobel(gray):
    gx = _convolve(gray, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = _convolve(gray, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return gx, gy


class Emboss(_Filter):
    """Grey relief lit from one angle: Emboss in all three editors."""
    name, label, category = "emboss", "Emboss", "Stylize"

    def declare(self):
        return _props(FloatProperty("angle", 135.0, 0.0, 360.0, decimals=1), IntProperty("depth", 3, 1, 20))

    def compute(self, t, src):
        g = _lum(src[..., :3].astype(np.float64))
        gx, gy = _sobel(g)
        a = math.radians(t["angle"])
        v = 128 + (gx * math.cos(a) - gy * math.sin(a)) * t["depth"] / 4.0
        out = src.copy()
        out[..., :3] = np.clip(np.rint(v), 0, 255)[..., None]
        return out


class Relief(Emboss):
    """Paint.NET's Relief: the emboss shading laid over the colour."""
    name, label = "relief", "Relief"

    def compute(self, t, src):
        e = super().compute(t, src)[..., 0].astype(np.float32) - 128
        out = src.copy()
        out[..., :3] = np.clip(np.rint(src[..., :3].astype(np.float32) + e[..., None]), 0, 255)
        return out


class EdgeDetect(_Filter):
    """Edges, by Sobel, Laplace or difference of Gaussians (GIMP's choices);
    "Find Edges" style draws them dark on white like Photoshop's."""
    name, label, category = "edge_detect", "Edge Detect", "Stylize"

    def declare(self):
        return _props(ChoiceProperty("method", "sobel", ["sobel", "laplace", "dog"],
                                     labels={"sobel": "Sobel", "laplace": "Laplace",
                                             "dog": "Difference of Gaussians"}),
                      IntProperty("amount", 100, 1, 1000),
                      ChoiceProperty("style", "find_edges", ["find_edges", "glowing", "colour"],
                                     labels={"find_edges": "Dark on white (Find Edges)",
                                             "glowing": "Bright on black (Glowing Edges)",
                                             "colour": "Coloured edges"}))

    def compute(self, t, src):
        rgb = src[..., :3].astype(np.float64)
        k = t["amount"] / 100.0
        if t["method"] == "laplace":
            mag = np.abs(np.stack([_convolve(rgb[..., c], [[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                                   for c in range(3)], -1))
        elif t["method"] == "dog":
            a = _gauss(rgb.astype(np.float32), 1.0)
            b = _gauss(rgb.astype(np.float32), 3.0)
            mag = np.abs(a - b) * 3
        else:
            mag = np.stack([np.hypot(*_sobel(rgb[..., c])) for c in range(3)], -1) / 4
        mag = np.clip(mag * k, 0, 255)
        if t["style"] == "colour":
            v = mag
        else:
            grey = mag.max(axis=-1, keepdims=True).repeat(3, -1)
            v = 255 - grey if t["style"] == "find_edges" else grey
        out = src.copy()
        out[..., :3] = np.rint(v)
        return out


class Outline(_Filter):
    """Paint.NET's Outline effect: thick ink lines where colour changes."""
    name, label, category = "outline", "Outline", "Stylize"

    def declare(self):
        return _props(IntProperty("thickness", 3, 1, 50), IntProperty("intensity", 50, 0, 100))

    def compute(self, t, src):
        g = _lum(src[..., :3].astype(np.float64))
        r = t["thickness"]
        spread = _rank(g, r, "max") - _rank(g, r, "min")
        dark = np.clip(spread * t["intensity"] / 50.0, 0, 255)
        out = src.copy()
        out[..., :3] = np.clip(np.rint(src[..., :3] - dark[..., None]), 0, 255)
        return out


class Solarize(_Filter):
    """Invert everything above a level, like an overexposed print."""
    name, label, category = "solarize", "Solarize", "Stylize"
    pointwise = True

    def declare(self):
        return _props(IntProperty("threshold", 128, 0, 255))

    def compute(self, t, src):
        out = src.copy()
        rgb = src[..., :3]
        out[..., :3] = np.where(rgb >= t["threshold"], 255 - rgb, rgb)
        return out


class MaximumMinimum(_Filter):
    """Grow light (Maximum / Dilate) or dark (Minimum / Erode) areas."""
    name, label, category = "maximum_minimum", "Maximum / Minimum", "Stylize"

    def declare(self):
        return _props(ChoiceProperty("mode", "maximum", ["maximum", "minimum"],
                                     labels={"maximum": "Maximum (dilate)", "minimum": "Minimum (erode)"}),
                      IntProperty("radius", 2, 1, 50))

    def compute(self, t, src):
        out = src.copy()
        kind = "max" if t["mode"] == "maximum" else "min"
        for c in range(3):
            out[..., c] = _rank(src[..., c], t["radius"], kind)
        return out


class Neon(_Filter):
    """GIMP's Neon: glowing coloured edges on black."""
    name, label, category = "neon", "Neon", "Stylize"

    def declare(self):
        return _props(IntProperty("radius", 3, 1, 20), IntProperty("amount", 150, 10, 500))

    def compute(self, t, src):
        blur = _blur_rgba(src, t["radius"]).astype(np.float32)
        edge = np.abs(src[..., :3].astype(np.float32) - blur[..., :3]) * t["amount"] / 50.0
        glow = _gauss(edge, t["radius"])
        out = src.copy()
        out[..., :3] = np.clip(np.rint(edge + glow), 0, 255)
        return out


# ==== Artistic ===========================================================================

class OilPainting(_Filter):
    """Flat, brushy colour with edges kept: the Kuwahara filter (each pixel
    takes the mean of whichever quarter of its neighbourhood varies least).
    Paint.NET's Oil Painting, GIMP's Oilify, Photoshop's Oil Paint."""
    name, label, category = "oil_painting", "Oil Painting", "Artistic"

    def declare(self):
        return _props(IntProperty("brush", 4, 1, 20, label="Brush size"))

    def compute(self, t, src):
        r = t["brush"]
        work = _pm(src).astype(np.float64)
        lum = _lum(work)
        h, w = lum.shape

        def window_mean(a, oy, ox):
            m = _box(_box(a.astype(np.float32), r // 2 if r > 1 else 1, 0), r // 2 if r > 1 else 1, 1)
            return _shifted(m, oy, ox)
        best_var = np.full((h, w), np.inf)
        result = np.zeros_like(work)
        half = max(1, r // 2)
        for oy, ox in ((-half, -half), (-half, half), (half, -half), (half, half)):
            mean_l = window_mean(lum, oy, ox)
            var = window_mean(lum * lum, oy, ox) - mean_l * mean_l
            better = var < best_var
            best_var = np.where(better, var, best_var)
            mean_c = np.stack([window_mean(work[..., c], oy, ox) for c in range(4)], -1)
            result = np.where(better[..., None], mean_c, result)
        return _unpm(result.astype(np.float32))


class PencilSketch(_Filter):
    """Paint.NET's Pencil Sketch: greyscale, colour-dodged by its own blurred
    negative, which leaves the lines."""
    name, label, category = "pencil_sketch", "Pencil Sketch", "Artistic"

    def declare(self):
        return _props(IntProperty("tip", 3, 1, 20, label="Pencil tip"),
                      IntProperty("range", 0, -20, 20))

    def compute(self, t, src):
        g = _lum(src[..., :3].astype(np.float32))
        inv = 255 - g
        blur = _gauss(inv[..., None], t["tip"])[..., 0]
        dodge = np.where(blur >= 255, 255, np.minimum(255, g * 255 / np.maximum(255 - blur, 1)))
        dodge = np.clip(dodge + t["range"] * 4, 0, 255)
        out = src.copy()
        out[..., :3] = np.rint(dodge)[..., None]
        return out


class InkSketch(_Filter):
    """Paint.NET's Ink Sketch: dark ink edges over lightened colour."""
    name, label, category = "ink_sketch", "Ink Sketch", "Artistic"

    def declare(self):
        return _props(IntProperty("ink", 50, 0, 99, label="Ink outline"),
                      IntProperty("coloring", 50, 0, 100))

    def compute(self, t, src):
        rgb = src[..., :3].astype(np.float64)
        g = _lum(rgb)
        mag = np.hypot(*_sobel(g)) / 4
        ink = np.clip(mag * (t["ink"] / 25.0), 0, 255)
        grey = g[..., None]
        col = grey + (rgb - grey) * (t["coloring"] / 100.0)
        col = col + (255 - col) * 0.35
        out = src.copy()
        out[..., :3] = np.clip(np.rint(col - ink[..., None]), 0, 255)
        return out


class Cartoon(_Filter):
    """GIMP's Cartoon: blacken wherever a pixel is darker than its
    surroundings, leaving bold outlines on the original colour."""
    name, label, category = "cartoon", "Cartoon", "Artistic"

    def declare(self):
        return _props(IntProperty("radius", 7, 1, 50), IntProperty("black", 20, 0, 100,
                                                                   label="Percent black"))

    def compute(self, t, src):
        g = _lum(src[..., :3].astype(np.float32))
        local = _gauss(g[..., None], t["radius"])[..., 0]
        ratio = g / np.maximum(local, 1)
        thresh = 1 - t["black"] / 400.0
        dark = np.clip((thresh - ratio) * 10, 0, 1)
        out = src.copy()
        out[..., :3] = np.rint(src[..., :3] * (1 - dark[..., None]))
        return out


class Halftone(_Filter):
    """Colour halftone / Newsprint: each channel as rotated dots whose size
    carries the value."""
    name, label, category = "halftone", "Color Halftone", "Artistic"

    def declare(self):
        return _props(IntProperty("cell", 8, 3, 64, label="Dot spacing"),
                      BoolProperty("monochrome", False))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        cell = float(t["cell"])
        rgb = src[..., :3].astype(np.float64) / 255.0
        angles = (15, 75, 0) if not t["monochrome"] else (45, 45, 45)
        out = src.copy()
        chans = [_lum(rgb)] * 3 if t["monochrome"] else [rgb[..., c] for c in range(3)]
        for c, (val, ang) in enumerate(zip(chans, angles)):
            a = math.radians(ang)
            u = xs * math.cos(a) + ys * math.sin(a)
            v = -xs * math.sin(a) + ys * math.cos(a)
            cu, cv = (np.floor(u / cell) + 0.5) * cell, (np.floor(v / cell) + 0.5) * cell
            d = np.hypot(u - cu, v - cv) / (cell * 0.7071)
            ink = 1 - val
            r = np.sqrt(np.clip(ink, 0, 1))
            cov = np.clip((r - d) * cell * 0.7071 + 0.5, 0, 1)
            out[..., c] = np.rint(255 * (1 - cov))
        return out


class Crystallize(_Filter):
    """Polygonal cells of one colour each (a jittered Voronoi diagram):
    Photoshop's Crystallize, Paint.NET's Crystallize."""
    name, label, category = "crystallize", "Crystallize", "Artistic"

    def declare(self):
        return _props(IntProperty("cell", 16, 3, 200, label="Cell size"), _seed())

    def compute(self, t, src):
        h, w = src.shape[:2]
        c = t["cell"]
        gh, gw = h // c + 2, w // c + 2
        gy, gx = np.mgrid[0:gh, 0:gw]
        sx = (gx + _hash01(gx, gy, t["seed"])) * c
        sy = (gy + _hash01(gx, gy, t["seed"] + 5)) * c
        ys, xs = np.mgrid[0:h, 0:w]
        cx, cy = xs // c, ys // c
        best = np.full((h, w), np.inf)
        owner = np.zeros((h, w), np.int64)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                nx, ny = np.clip(cx + ox, 0, gw - 1), np.clip(cy + oy, 0, gh - 1)
                d = (sx[ny, nx] - xs) ** 2 + (sy[ny, nx] - ys) ** 2
                better = d < best
                best = np.where(better, d, best)
                owner = np.where(better, ny * gw + nx, owner)
        work = _pm(src).reshape(-1, 4).astype(np.float64)
        flat = owner.ravel()
        counts = np.bincount(flat, minlength=gh * gw)
        means = np.stack([np.bincount(flat, weights=work[:, k], minlength=gh * gw)
                          for k in range(4)], -1) / np.maximum(counts, 1)[:, None]
        return _unpm(means[flat].reshape(h, w, 4).astype(np.float32))


class Pointillize(Crystallize):
    """Dots of colour on the background colour: Photoshop's Pointillize."""
    name, label = "pointillize", "Pointillize"

    def declare(self):
        return _props(IntProperty("cell", 10, 3, 100, label="Dot size"), _seed(),
                      ColorProperty("background", (255, 255, 255, 255)))

    def compute(self, t, src):
        cells = super().compute({"cell": t["cell"], "seed": t["seed"]}, src)
        h, w = src.shape[:2]
        c = t["cell"]
        ys, xs = np.mgrid[0:h, 0:w]
        jx = _hash01(xs // c, ys // c, t["seed"] + 11)
        cxp = (xs // c + 0.5) * c + (jx - 0.5) * c * 0.3
        cyp = (ys // c + 0.5) * c + (_hash01(xs // c, ys // c, t["seed"] + 12) - 0.5) * c * 0.3
        inside = np.clip(c * 0.45 - np.hypot(xs + 0.5 - cxp, ys + 0.5 - cyp) + 0.5, 0, 1)
        bg = np.array(t["background"], np.float32)
        out = cells.astype(np.float32) * inside[..., None] + bg * (1 - inside[..., None])
        res = src.copy()
        res[..., :3] = np.rint(out[..., :3])
        return res


class CanvasTexture(_Filter):
    """GIMP's Apply Canvas: a woven texture embossed into the picture."""
    name, label, category = "canvas_texture", "Canvas Texture", "Artistic"

    def declare(self):
        return _props(IntProperty("depth", 4, 1, 20), IntProperty("scale", 4, 1, 20))

    def compute(self, t, src):
        h, w = src.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        s = t["scale"]
        weave = np.sin(xs / s * math.pi) * np.sin(ys / s * math.pi)
        bump = _convolve(weave, [[-1, 0, 0], [0, 0, 0], [0, 0, 1]]) * t["depth"] * 8
        out = src.copy()
        out[..., :3] = np.clip(np.rint(src[..., :3] + bump[..., None]), 0, 255)
        return out


# ==== Render ==============================================================================

class _Render(_Filter):
    """Render filters paint over the layer; Blend mixes them with what is there."""

    def _blend(self, t, src, rgb, alpha=None):
        out = src.copy()
        k = t.get("opacity", 100) / 100.0
        base = src[..., :3].astype(np.float64)
        out[..., :3] = np.clip(np.rint(base + (rgb - base) * k), 0, 255)
        out[..., 3] = 255 if alpha is None else alpha
        return out


class Clouds(_Render):
    """Fractal noise between the two colours: Clouds / Solid Noise / Plasma."""
    name, label, category = "clouds", "Clouds", "Render"

    def declare(self):
        return _props(IntProperty("scale", 120, 2, 1000), IntProperty("roughness", 60, 0, 100),
                      _seed(), ColorProperty("color1", (0, 0, 0, 255)),
                      ColorProperty("color2", (255, 255, 255, 255)),
                      IntProperty("opacity", 100, 0, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        n = _value_noise(h, w, t["scale"], t["seed"], 7, t["roughness"] / 100.0)
        n = np.clip((n - n.min()) / max(1e-9, n.max() - n.min()), 0, 1)[..., None]
        a = np.array(t["color1"][:3], np.float64)
        b = np.array(t["color2"][:3], np.float64)
        return self._blend(t, src, a + (b - a) * n)


class Mandelbrot(_Render):
    """The Mandelbrot set, smoothly coloured (Paint.NET's Mandelbrot Fractal)."""
    name, label, category = "mandelbrot", "Mandelbrot Fractal", "Render"

    def declare(self):
        return _props(FloatProperty("zoom", 1.0, 0.1, 1000.0, decimals=2),
                      FloatProperty("center_re", -0.6, -2.0, 1.0, decimals=4),
                      FloatProperty("center_im", 0.0, -1.5, 1.5, decimals=4),
                      IntProperty("iterations", 100, 10, 1000), IntProperty("opacity", 100, 0, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        scale = 3.0 / t["zoom"] / min(w, h)
        cr = t["center_re"] + (xs - w / 2.0) * scale
        ci = t["center_im"] + (ys - h / 2.0) * scale
        zr, zi = np.zeros_like(cr), np.zeros_like(ci)
        count = np.zeros(cr.shape)
        alive = np.ones(cr.shape, bool)
        for i in range(t["iterations"]):
            zr2, zi2 = zr * zr, zi * zi
            esc = alive & (zr2 + zi2 > 16)
            count[esc] = i + 1 - np.log2(np.log2(np.maximum(zr2 + zi2, 1.0001))[esc])
            alive &= ~esc
            zi = np.where(alive, 2 * zr * zi + ci, zi)
            zr = np.where(alive, zr2 - zi2 + cr, zr)
        v = np.where(alive, 0, count / t["iterations"])
        rgb = np.stack([0.5 + 0.5 * np.cos(6.28 * (v * 3 + d)) for d in (0.0, 0.33, 0.67)], -1)
        rgb = np.where(alive[..., None], 0, rgb) * 255
        return self._blend(t, src, rgb)


class Julia(Mandelbrot):
    """A Julia set, Paint.NET's Julia Fractal."""
    name, label = "julia", "Julia Fractal"

    def declare(self):
        return _props(FloatProperty("zoom", 1.0, 0.1, 1000.0, decimals=2),
                      FloatProperty("c_re", -0.8, -2.0, 2.0, decimals=4),
                      FloatProperty("c_im", 0.156, -2.0, 2.0, decimals=4),
                      IntProperty("iterations", 120, 10, 1000), IntProperty("opacity", 100, 0, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        scale = 3.0 / t["zoom"] / min(w, h)
        zr = (xs - w / 2.0) * scale
        zi = (ys - h / 2.0) * scale
        count = np.zeros(zr.shape)
        alive = np.ones(zr.shape, bool)
        for i in range(t["iterations"]):
            zr2, zi2 = zr * zr, zi * zi
            esc = alive & (zr2 + zi2 > 16)
            count[esc] = i + 1
            alive &= ~esc
            zi = np.where(alive, 2 * zr * zi + t["c_im"], zi)
            zr = np.where(alive, zr2 - zi2 + t["c_re"], zr)
        v = count / t["iterations"]
        rgb = np.stack([0.5 + 0.5 * np.sin(6.28 * (v * 2 + d)) for d in (0.6, 0.3, 0.0)], -1) * 255
        rgb = np.where(alive[..., None], 0, rgb)
        return self._blend(t, src, rgb)


class Voronoi(_Render):
    """Paint.NET's Voronoi Diagram: random coloured cells."""
    name, label, category = "voronoi", "Voronoi Cells", "Render"

    def declare(self):
        return _props(IntProperty("cell", 40, 4, 400, label="Cell size"), _seed(),
                      IntProperty("opacity", 100, 0, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        noise = np.zeros((h, w, 4), np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        for c in range(3):
            noise[..., c] = (_hash01(xx // t["cell"], yy // t["cell"], t["seed"] + c) * 255).astype(np.uint8)
        noise[..., 3] = 255
        cells = Crystallize().compute({"cell": t["cell"], "seed": t["seed"]}, noise)
        return self._blend(t, src, cells[..., :3].astype(np.float64))


class Checkerboard(_Render):
    name, label, category = "checkerboard", "Checkerboard", "Render"

    def declare(self):
        return _props(IntProperty("size", 16, 1, 1000), ColorProperty("color1", (0, 0, 0, 255)),
                      ColorProperty("color2", (255, 255, 255, 255)), IntProperty("opacity", 100, 0, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        k = (((xx // t["size"]) + (yy // t["size"])) % 2)[..., None]
        a, b = np.array(t["color1"][:3], np.float64), np.array(t["color2"][:3], np.float64)
        return self._blend(t, src, a + (b - a) * k)


class GridRender(_Render):
    """GIMP's Render > Grid: lines at a spacing, over the image."""
    name, label, category = "render_grid", "Grid", "Render"

    def declare(self):
        return _props(IntProperty("spacing", 16, 2, 1000), IntProperty("width", 1, 1, 50),
                      IntProperty("offset", 0, 0, 1000), ColorProperty("color", (0, 0, 0, 255)),
                      IntProperty("opacity", 100, 0, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        line = (((xx - t["offset"]) % t["spacing"]) < t["width"]) | \
               (((yy - t["offset"]) % t["spacing"]) < t["width"])
        base = src[..., :3].astype(np.float64)
        rgb = np.where(line[..., None], np.array(t["color"][:3], np.float64), base)
        out = src.copy()
        k = t["opacity"] / 100.0
        out[..., :3] = np.clip(np.rint(base + (rgb - base) * k), 0, 255)
        out[..., 3] = np.where(line, 255, src[..., 3])
        return out


class LensFlare(_Filter):
    """A simple lens flare: a hot core, a halo and ghost rings along the axis
    to the centre (GIMP's Lens Flare)."""
    name, label, category = "lens_flare", "Lens Flare", "Render"

    def declare(self):
        return _props(IntProperty("center_x", 30, 0, 100, label="Position X (%)"),
                      IntProperty("center_y", 30, 0, 100, label="Position Y (%)"),
                      IntProperty("brightness", 100, 10, 300), IntProperty("size", 15, 2, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        xs, ys = _grid(h, w)
        fx, fy = w * t["center_x"] / 100.0, h * t["center_y"] / 100.0
        R = min(w, h) * t["size"] / 100.0
        d = np.hypot(xs - fx, ys - fy) / R
        light = np.exp(-d * d * 4) * 1.0 + np.exp(-d * 1.5) * 0.35
        light = light[..., None] * np.array([1.0, 0.95, 0.85])
        for k, col in ((0.5, (0.3, 0.6, 1.0)), (1.2, (1.0, 0.4, 0.4)), (1.7, (0.4, 1.0, 0.5))):
            gx, gy = fx + (w / 2 - fx) * k, fy + (h / 2 - fy) * k
            gd = np.hypot(xs - gx, ys - gy) / (R * 0.4)
            ring = np.exp(-(gd - 1) ** 2 * 8) * 0.25
            light = light + ring[..., None] * np.array(col)
        add = light * 255 * t["brightness"] / 100.0
        out = src.copy()
        out[..., :3] = np.clip(np.rint(src[..., :3] + add), 0, 255)
        return out


# ==== Edge & Generic ==========================================================================

class ConvolutionMatrix(_Filter):
    """GIMP's Generic > Convolution Matrix, 3x3: write your own filter."""
    name, label, category = "convolution", "Convolution Matrix", "Edge & Generic"

    def declare(self):
        cells = [IntProperty("k%d%d" % (r, c), 1 if (r, c) == (1, 1) else 0, -100, 100)
                 for r in range(3) for c in range(3)]
        return _props(*cells, IntProperty("divisor", 1, -100, 100), IntProperty("offset", 0, -255, 255))

    def compute(self, t, src):
        k = np.array([[t["k%d%d" % (r, c)] for c in range(3)] for r in range(3)], np.float64)
        div = t["divisor"] or 1
        out = src.copy()
        for c in range(3):
            out[..., c] = np.clip(np.rint(_convolve(src[..., c].astype(np.float64), k) / div
                                          + t["offset"]), 0, 255)
        return out


class DropShadowFilter(_Filter):
    """GIMP's Light & Shadow > Drop Shadow, baked into the layer (for a live
    one, use Layer > Layer Style)."""
    name, label, category = "drop_shadow_filter", "Drop Shadow", "Edge & Generic"

    def declare(self):
        return _props(IntProperty("offset_x", 6, -200, 200), IntProperty("offset_y", 6, -200, 200),
                      IntProperty("blur", 6, 0, 100), ColorProperty("color", (0, 0, 0, 255)),
                      IntProperty("opacity", 70, 0, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        a = src[..., 3].astype(np.float32)
        sh = np.zeros_like(a)
        dx, dy = t["offset_x"], t["offset_y"]
        ys = slice(max(0, dy), min(h, h + dy))
        yd = slice(max(0, -dy), min(h, h - dy))
        xs = slice(max(0, dx), min(w, w + dx))
        xd = slice(max(0, -dx), min(w, w - dx))
        sh[ys, xs] = a[yd, xd]
        sh = _gauss(sh[..., None], t["blur"])[..., 0] * t["opacity"] / 100.0
        shadow = np.zeros(src.shape, np.uint8)
        shadow[..., :3] = t["color"][:3]
        shadow[..., 3] = np.clip(np.rint(sh), 0, 255)
        from .accel import spec
        return spec.blend_rect(shadow, src, spec.NORMAL, 255)


FILTERS = (MotionBlur, RadialBlur, ZoomBlur, BoxBlur, LensBlur, SurfaceBlur, TiltShift,
           FragmentBlur, Pixelate, UnsharpMask, HighPass, Vignette, Glow, SoftenPortrait,
           ShadowsHighlights, Median, DustScratches, ReduceNoise, Spread, FrostedGlass, Twirl,
           PinchBulge, Spherize, Ripple, Waves, PolarCoordinates, PolarInversion, LensDistortion,
           Kaleidoscope, TileReflection, Dents, Wind, TileSeamless, Emboss, Relief, EdgeDetect,
           Outline, Solarize, MaximumMinimum, Neon, OilPainting, PencilSketch, InkSketch, Cartoon,
           Halftone, Crystallize, Pointillize, CanvasTexture, Clouds, Mandelbrot, Julia, Voronoi,
           Checkerboard, GridRender, LensFlare, ConvolutionMatrix, DropShadowFilter)
