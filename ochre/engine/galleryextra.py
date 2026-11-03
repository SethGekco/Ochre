# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop filters Ochre lacked outside the six Filter Gallery families:
Diffuse Glow, Glass and Ocean Ripple (Distort), Glowing Edges (Stylize),
Smart Blur, Shape Blur, Average, Spin Blur and Path Blur (Blurs), Sharpen
Edges and Sharpen More (Photo), NTSC Colors (Video) and Embed Watermark
(Watermark), with read_watermark() to find what Embed Watermark hid.

Ported from Kooka Pictura's filter crates (crates/pictura-filters, GPL-3.0):
distort/diffuse_glow.rs, glass.rs, ripples.rs; photorust/stylize (Glowing
Edges) and photorust/sharpen.rs (Sharpen Edges); blur/smart.rs, shape.rs,
spin.rs, path.rs and blur.rs (Average); sharpen.rs (Sharpen More); video.rs
with pictura-color's NTSC clamp; digimarc.rs. The watermark is Kooka's own
spread-spectrum code and does not read or write Digimarc's.

Where Kooka moves pixels (Glass, Ocean Ripple, the blurs) it moves alpha
with them, and so does this port -- premultiplied here, so the hidden
color of transparent pixels never bleeds in. One simplification: Ocean
Ripple's lattice comes from Ochre's seeded hash, not Kooka's ChaCha8
stream (same look, different ripples per seed).

Spin and Path Blur are Kooka's whole effects: any number of spins (largest
mask wins) with the strobe's flash duration, any number of curved paths
(Catmull-Rom through their points, a blended motion field, trails that
bend along its streamlines), and the Blur Gallery's Noise pass (field.rs,
Gaussian / Uniform / Grain). The sliders make one spin / one straight path;
the `spins` / `paths` JSON, which the Blur Gallery tool writes, holds more.
Kooka's bokeh pass is off for both effects there too.
"""

import json
import math

import numpy as np

from . import fields as F
from .filters import _Filter, _centre
from .props import BoolProperty, ChoiceProperty, ColorProperty, FloatProperty, IntProperty, \
    PropertyCollection, TextProperty
from .sketch import WHITE


def _props(*items):
    return PropertyCollection(list(items))


def _rnd8(a):
    return np.clip(np.floor(np.asarray(a, np.float64) + 0.5), 0, 255).astype(np.uint8)


def _premul(src):
    a = src[..., 3:4].astype(np.float64) / 255.0
    return np.concatenate([src[..., :3].astype(np.float64) * a, src[..., 3:4].astype(np.float64)], -1)


def _unpremul(pm, src):
    """Back from premultiplied floats; where nothing is left, the old color."""
    a = pm[..., 3:4]
    rgb = np.where(a > 1e-6, pm[..., :3] * 255.0 / np.maximum(a, 1e-6), src[..., :3])
    out = np.empty(src.shape, np.uint8)
    out[..., :3] = _rnd8(rgb)
    out[..., 3] = _rnd8(a[..., 0])
    return out


def _hash_unit(x, y, salt):
    """Kooka's 32-bit coordinate hash: 0..1 (24 bits)."""
    h = (F._u32(x) * np.uint32(0x9E3779B1)) ^ (F._u32(y) * np.uint32(0x85EBCA77)) ^ F._u32(salt)
    h ^= h >> np.uint32(16)
    h = h * np.uint32(0x7FEB352D)
    h ^= h >> np.uint32(15)
    h = h * np.uint32(0x846CA68B)
    h ^= h >> np.uint32(16)
    return (h >> np.uint32(8)).astype(np.float64) / float(1 << 24)


def _bilinear(planes, x, y):
    """Clamp-to-edge bilinear samples of (H, W, C) at float pixel coords."""
    h, w = planes.shape[:2]
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    xi, yi = np.clip(x0, 0, w - 1).astype(np.intp), np.clip(y0, 0, h - 1).astype(np.intp)
    xj, yj = np.clip(x0 + 1, 0, w - 1).astype(np.intp), np.clip(y0 + 1, 0, h - 1).astype(np.intp)
    top = planes[yi, xi] * (1 - fx) + planes[yi, xj] * fx
    bottom = planes[yj, xi] * (1 - fx) + planes[yj, xj] * fx
    return top * (1 - fy) + bottom * fy


def _displace(src, dx, dy):
    """Each pixel sampled (bilinear, premultiplied) from itself + (dx, dy)."""
    h, w = src.shape[:2]
    xs, ys = F.grid(h, w)
    return _unpremul(_bilinear(_premul(src), xs + dx, ys + dy), src)


# ==== Distort ==========================================================================

class DiffuseGlow(_Filter):
    """Soft glow of the background color in the lights, through a grainy
    veil, as though seen through a diffusion filter."""
    name, label, category = "diffuse_glow", "Diffuse Glow", "Distort"

    def declare(self):
        return _props(IntProperty("graininess", 6, 0, 10), IntProperty("glow_amount", 10, 0, 20),
                      IntProperty("clear_amount", 15, 0, 20),
                      ColorProperty("background", WHITE, label="Glow (Background)"),
                      IntProperty("seed", 1, 0, 999))

    def compute(self, t, src):
        h, w = src.shape[:2]
        grain, glow, clear = float(t["graininess"]), float(t["glow_amount"]), float(t["clear_amount"])
        lit = F.blur(F.luma(src) / 255.0, 2.37 + 0.54 * glow)
        threshold = 0.56 + 0.016 * clear - 0.028 * glow
        gain = 0.71 + 0.186 * glow
        cover = 0.93 * (1 - math.exp(-glow / 0.71))
        veil = 0.98 * max(1 - clear / 10.0, 0.0) ** 2.5
        speckle = 0.0126 * grain + 0.0016 * grain * grain
        ramp = F.smooth(0.5 + gain * (lit - threshold))
        mix = 1 - (1 - veil) * (1 - cover * ramp)
        if speckle > 0:
            xs, ys = F.grid(h, w)
            salt = int(t["seed"]) & 0xFFFFFFFF
            mix = mix + speckle * (_hash_unit(xs, ys, salt) + _hash_unit(xs, ys, ~salt & 0xFFFFFFFF) - 1)
        mix = np.clip(mix, 0, 1)[..., None]
        v = src[..., :3].astype(np.float64)
        glow_rgb = np.asarray(t["background"][:3], np.float64)
        return F.put_rgb(src, v + (glow_rgb - v) * mix)


GLASS_TEXTURES = ("blocks", "canvas", "frosted", "tiny_lens")
GLASS_LABELS = {"blocks": "Blocks", "canvas": "Canvas", "frosted": "Frosted", "tiny_lens": "Tiny Lens"}


def _glass_noise(u, v, cell, salt):
    gu, gv = u / cell, v / cell
    iu, iv = np.floor(gu), np.floor(gv)
    fu, fv = F.smooth(gu - iu), F.smooth(gv - iv)
    iu, iv = iu.astype(np.int64), iv.astype(np.int64)

    def at(du, dv):
        return _glass_unit(iu + du, iv + dv, salt)
    top = at(0, 0) * (1 - fu) + at(1, 0) * fu
    bottom = at(0, 1) * (1 - fu) + at(1, 1) * fu
    return top * (1 - fv) + bottom * fv


def _glass_unit(x, y, salt):
    return _hash_unit(x, y, (salt * 0xC2B2AE3D) & 0xFFFFFFFF)


def _glass_height(texture, u, v):
    if texture == "frosted":
        return 0.6 * _glass_noise(u, v, 6.0, 1) + 0.4 * _glass_noise(u, v, 13.0, 2)
    if texture == "tiny_lens":
        a = (u + v) * math.sqrt(0.5)
        b = (u - v) * math.sqrt(0.5)
        cx = np.mod(a, 16.0) / 16.0 - 0.5
        cy = np.mod(b, 16.0) / 16.0 - 0.5
        return np.sqrt(np.maximum(1 - 4 * (cx * cx + cy * cy), 0))
    if texture == "blocks":
        row = np.floor(v / 40.0)
        col = np.floor(u / 40.0 + 0.5 * np.mod(row, 2.0))
        return _glass_unit(col.astype(np.int64), row.astype(np.int64), 3)
    a = np.sin(u / 24.0 * math.pi) ** 2                     # canvas
    b = np.sin(v / 24.0 * math.pi) ** 2
    over = np.mod(np.floor(u / 24.0) + np.floor(v / 24.0), 2.0)
    weave = np.where(over > 0.5, 0.6 * a + 0.3 * b, 0.3 * a + 0.6 * b)
    return 0.5 * weave + 0.6 * _glass_noise(u, v, 0.7 * 24.0, 7)


def _glass_surface(texture, h, w, zoom, sigma):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    field = _glass_height(texture, xs / zoom, ys / zoom)
    if texture == "blocks":
        field = F.blur(field, 0.25 * 40.0 * zoom)
    return F.blur(field, sigma)


def _slope(field):
    """Central differences, one-sided at the border."""
    h, w = field.shape
    xl, xr = np.maximum(np.arange(w) - 1, 0), np.minimum(np.arange(w) + 1, w - 1)
    yu, yd = np.maximum(np.arange(h) - 1, 0), np.minimum(np.arange(h) + 1, h - 1)
    gx = (field[:, xr] - field[:, xl]) / np.maximum(xr - xl, 1)
    gy = (field[yd] - field[yu]) / np.maximum(yd - yu, 1)[:, None]
    return gx, gy


class Glass(_Filter):
    """The picture seen through textured glass."""
    name, label, category = "glass", "Glass", "Distort"
    moves_alpha = True

    def declare(self):
        return _props(IntProperty("distortion", 5, 0, 20), IntProperty("smoothness", 3, 1, 15),
                      ChoiceProperty("texture", "frosted", list(GLASS_TEXTURES), labels=GLASS_LABELS),
                      IntProperty("scaling", 100, 50, 200), BoolProperty("invert", False))

    def compute(self, t, src):
        if t["distortion"] == 0:
            return src.copy()
        h, w = src.shape[:2]
        texture = t["texture"]
        zoom = t["scaling"] / 100.0
        sigma = 0.4 * t["smoothness"] * zoom
        gx, gy = _slope(_glass_surface(texture, h, w, zoom, sigma))
        # Scaled so each texture bends the picture as much at any size.
        rx, ry = _slope(_glass_surface(texture, 256, 256, zoom, sigma))
        rms = math.sqrt(float(np.mean(rx * rx + ry * ry)))
        if rms <= 1e-7:
            return src.copy()
        relief = {"frosted": 0.6, "canvas": 0.45}.get(texture, 1.0)
        reach = (-1.0 if t["invert"] else 1.0) * 1.4 * t["distortion"] * relief / rms
        return _displace(src, reach * gx, reach * gy)


class OceanRipple(_Filter):
    """Randomly spaced ripples, as though the picture lay under water."""
    name, label, category = "ocean_ripple", "Ocean Ripple", "Distort"
    moves_alpha = True

    def declare(self):
        return _props(IntProperty("ripple_size", 9, 1, 15), IntProperty("ripple_magnitude", 9, 0, 20),
                      IntProperty("seed", 1, 0, 999))

    def compute(self, t, src):
        if t["ripple_magnitude"] == 0:
            return src.copy()
        h, w = src.shape[:2]
        cell = 3.0 + 0.5 * t["ripple_size"]
        reach = 0.15 * float(t["ripple_magnitude"]) ** 1.5 * math.sqrt(cell / 4.0)
        gw, gh = int(w / cell) + 2, int(h / cell) + 2
        gi, gj = np.meshgrid(np.arange(gw), np.arange(gh))
        lattice = 2.0 * F.noise(gi, gj, t["seed"]).astype(np.float64) - 1.0
        xs, ys = F.grid(h, w)
        u, v = xs / cell, ys / cell
        i, j = u.astype(np.intp), v.astype(np.intp)
        fu, fv = u - i, v - j
        su, sv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
        du, dv = 6 * fu * (1 - fu), 6 * fv * (1 - fv)

        def at(a, b):
            return lattice[np.minimum(b, gh - 1), np.minimum(a, gw - 1)]
        a, b, c, d = at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)
        gx = ((b - a) * (1 - sv) + (d - c) * sv) * du
        gy = ((c - a) * (1 - su) + (d - b) * su) * dv
        return _displace(src, reach * gx, reach * gy)


# ==== Stylize ==========================================================================

def _sobel_rgb(rgb):
    """Per-channel Sobel magnitude, edge-clamped, truncated to bytes."""
    h, w = rgb.shape[:2]
    p = np.pad(rgb.astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")

    def s(dx, dy):
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    gx = -s(-1, -1) - 2 * s(-1, 0) - s(-1, 1) + s(1, -1) + 2 * s(1, 0) + s(1, 1)
    gy = -s(-1, -1) - 2 * s(0, -1) - s(1, -1) + s(-1, 1) + 2 * s(0, 1) + s(1, 1)
    return np.minimum(np.sqrt(gx * gx + gy * gy), 255).astype(np.uint8)


def _gauss_rgb(src, sigma):
    """Kooka's premultiplied Gaussian on a picture, back to straight bytes."""
    if sigma <= 0:
        return src[..., :3]
    return _rnd8(F.blur_rgb(src, sigma))


class GlowingEdges(_Filter):
    """The edges lit up like neon over black."""
    name, label, category = "glowing_edges", "Glowing Edges", "Stylize"

    def declare(self):
        return _props(IntProperty("edge_width", 2, 1, 14), IntProperty("edge_brightness", 6, 0, 20),
                      IntProperty("smoothness", 1, 1, 15))

    def compute(self, t, src):
        soft = _gauss_rgb(src, t["smoothness"] * 0.3).astype(np.float64)
        # Premultiplied, so a soft layer edge is not read as a step.
        soft = np.floor(soft * src[..., 3:4] / 255.0 + 0.5)
        edges = _sobel_rgb(soft)
        if t["edge_width"] // 2 > 0:
            edges = F.widest_nearby(edges, t["edge_width"] // 2)
        gain = t["edge_brightness"] / 5.0
        return F.put_rgb(src, np.floor(np.minimum(edges * gain, 255.0)))


# ==== Blurs ============================================================================

SMART_QUALITY = {"low": 3, "medium": 6, "high": 12}
SMART_MODES = ("normal", "edge_only", "overlay_edge")


def _smart_taps(radius, per_side):
    reach = int(math.ceil(radius))
    step = (reach + per_side - 1) // per_side
    sigma = max(radius / 3.0, 0.1)
    span = reach // step
    taps = [(0, 0, 1.0)]
    for j in range(-span, span + 1):
        for i in range(-span, span + 1):
            dx, dy = i * step, j * step
            d2 = dx * dx + dy * dy
            if (dx, dy) != (0, 0) and d2 <= radius * radius:
                taps.append((dx, dy, math.exp(-d2 / (2 * sigma * sigma))))
    return taps


class SmartBlur(_Filter):
    """Blurs only between colors within Threshold of each other, so edges
    stay sharp; Edge Only and Overlay Edge draw the edges in white."""
    name, label, category = "smart_blur", "Smart Blur", "Blurs"
    moves_alpha = True

    def declare(self):
        return _props(FloatProperty("radius", 3.0, 0.1, 100.0, decimals=1),
                      FloatProperty("threshold", 25.0, 0.1, 100.0, decimals=1),
                      ChoiceProperty("quality", "low", list(SMART_QUALITY),
                                     labels={"low": "Low", "medium": "Medium", "high": "High"}),
                      ChoiceProperty("mode", "normal", list(SMART_MODES),
                                     labels={"normal": "Normal", "edge_only": "Edge Only",
                                             "overlay_edge": "Overlay Edge"}))

    def compute(self, t, src):
        h, w = src.shape[:2]
        limit = math.floor(t["threshold"])
        mode = t["mode"]
        planes = 4 if mode == "normal" else 3
        reach = int(math.ceil(t["radius"]))
        p = np.pad(src[..., :planes].astype(np.int32), ((reach, reach), (reach, reach), (0, 0)),
                   mode="edge")
        me = src[..., :planes].astype(np.int32)

        def at(dx, dy):
            return p[reach + dy:reach + dy + h, reach + dx:reach + dx + w]

        def far(q):
            return np.any(np.abs(q[..., :3] - me[..., :3]) > limit, axis=-1)
        out = src.copy()
        edge = np.zeros((h, w), bool)
        if mode != "normal":
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                edge |= far(at(dx, dy))
        if mode == "edge_only":
            out[..., :3] = np.where(edge, 255, 0)[..., None]
            return out
        acc = np.zeros((h, w, planes))
        total = np.zeros((h, w))
        for dx, dy, weight in _smart_taps(t["radius"], SMART_QUALITY[t["quality"]]):
            q = at(dx, dy)
            near = ~far(q)
            acc += np.where(near[..., None], weight * q, 0)
            total += np.where(near, weight, 0)
        blurred = _rnd8(acc / total[..., None])           # the centre always counts
        out[..., :planes] = blurred
        if mode == "overlay_edge":
            out[..., :3] = np.where(edge[..., None], 255, out[..., :3])
        return out


SHAPES = ("star", "heart", "arrow", "cross", "lightning", "check")
SHAPE_LABELS = {s: s.title() for s in SHAPES}


def _shape_points(shape):
    if shape == "heart":
        t = np.arange(48) / 48.0 * 2 * math.pi
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return np.stack([x, -y], -1)
    fixed = {
        "arrow": [(0.0, 0.32), (0.55, 0.32), (0.55, 0.06), (1.0, 0.5), (0.55, 0.94), (0.55, 0.68),
                  (0.0, 0.68)],
        "cross": [(0.34, 0.0), (0.66, 0.0), (0.66, 0.34), (1.0, 0.34), (1.0, 0.66), (0.66, 0.66),
                  (0.66, 1.0), (0.34, 1.0), (0.34, 0.66), (0.0, 0.66), (0.0, 0.34), (0.34, 0.34)],
        "lightning": [(0.58, 0.0), (0.9, 0.0), (0.62, 0.4), (0.86, 0.4), (0.3, 1.0), (0.44, 0.56),
                      (0.16, 0.56), (0.34, 0.0)],
        "check": [(0.9, 0.12), (1.0, 0.28), (0.42, 0.95), (0.0, 0.58), (0.12, 0.42), (0.44, 0.7)],
    }
    if shape in fixed:
        return np.asarray(fixed[shape], np.float64)
    i = np.arange(10)                                       # star
    r = np.where(i % 2 == 0, 0.5, 0.2)
    a = -math.pi / 2 + i / 10.0 * 2 * math.pi
    return np.stack([0.5 + r * np.cos(a), 0.5 + r * np.sin(a)], -1)


def shape_kernel(shape, size, rows=4):
    """The shape fitted (proportions kept) into a size x size square,
    as 0..255 coverage: `rows` scanlines a pixel, exact across."""
    pts = _shape_points(shape)
    lo, hi = pts.min(0), pts.max(0)
    aspect = (hi[0] - lo[0]) / (hi[1] - lo[1])
    bw, bh = (size, size / aspect) if aspect >= 1 else (size * aspect, size)
    pts = (pts - lo) / (hi - lo) * (bw, bh) + ((size - bw) / 2, (size - bh) / 2)
    a, b = pts, np.roll(pts, -1, 0)
    cover = np.zeros((size, size))
    for k in range(size * rows):
        y = (k + 0.5) / rows
        crosses = ((a[:, 1] <= y) & (b[:, 1] > y)) | ((b[:, 1] <= y) & (a[:, 1] > y))
        if not crosses.any():
            continue
        aa, bb = a[crosses], b[crosses]
        xs = np.sort(aa[:, 0] + (y - aa[:, 1]) / (bb[:, 1] - aa[:, 1]) * (bb[:, 0] - aa[:, 0]))
        row = cover[k // rows]
        for left, right in zip(xs[0::2], xs[1::2]):        # even-odd spans
            cols = np.arange(size)
            row += np.clip(np.minimum(cols + 1, right) - np.maximum(cols, left), 0, 1) / rows
    return np.floor(np.clip(cover, 0, 1) * 255.0 + 0.5).astype(np.int64)


class ShapeBlur(_Filter):
    """A blur whose kernel is a shape: lights spread into stars, hearts..."""
    name, label, category = "shape_blur", "Shape Blur", "Blurs"
    moves_alpha = True

    def declare(self):
        return _props(IntProperty("radius", 10, 5, 1000),
                      ChoiceProperty("shape", "star", list(SHAPES), labels=SHAPE_LABELS))

    def compute(self, t, src):
        h, w = src.shape[:2]
        r = int(t["radius"])
        kernel = shape_kernel(t["shape"], 2 * r + 1)
        total = kernel.sum()
        if total == 0:
            return src.copy()
        pm = _premul(src)
        # Rows padded by the edge, prefix sums across: each run of equal
        # weight in a kernel row is two lookups.
        rows = np.clip(np.arange(-r, h + r), 0, h - 1)
        padded = np.pad(pm[rows], ((0, 0), (r, r), (0, 0)), mode="edge")
        pre = np.concatenate([np.zeros((h + 2 * r, 1, 4)), np.cumsum(padded, axis=1)], axis=1)
        acc = np.zeros((h, w, 4))
        xs = np.arange(w)
        for j in range(2 * r + 1):
            line = kernel[j]
            starts = np.flatnonzero(np.diff(np.concatenate([[-1], line])) != 0)
            ends = np.concatenate([starts[1:], [2 * r + 1]]) - 1
            dy = r - j                                       # the kernel is flipped
            band = pre[r + dy:r + dy + h]
            for s, e in zip(starts, ends):
                weight = line[s]
                if weight == 0:
                    continue
                lo, hi = r - e, r - s                         # offsets x+lo..x+hi
                acc += weight * (band[:, xs + r + hi + 1] - band[:, xs + r + lo])
        return _unpremul(acc / total, src)


class Average(_Filter):
    """The whole layer filled with its average color."""
    name, label, category = "average", "Average", "Blurs"
    moves_alpha = True

    def declare(self):
        return _props()

    def compute(self, t, src):
        pm = _premul(src).reshape(-1, 4).mean(0)
        return _unpremul(np.broadcast_to(pm, src.shape).astype(np.float64), src)


def _gather(src, chunk_taps, n_max, chunk=1 << 14):
    """Per-pixel tap gather. chunk_taps(idx) takes flat pixel indices and
    returns (moving, tap): which of them blur, and tap(k) -> (sel, x, y,
    weight) for their k-th tap (sel picks the moving pixels that have one)."""
    h, w = src.shape[:2]
    pm = _premul(src)
    out = pm.reshape(-1, 4).copy()
    for i0 in range(0, h * w, chunk):
        idx = np.arange(i0, min(h * w, i0 + chunk))
        moving, tap = chunk_taps(idx)
        if not moving.any():
            continue
        acc = np.zeros((len(idx), 4))
        for k in range(n_max):
            sel, tx, ty, weight = tap(k)
            if not sel.any():
                break
            acc[sel] += weight[:, None] * _bilinear(pm, tx - 0.5, ty - 0.5)
        out[idx[moving]] = acc[moving]
    return _unpremul(out.reshape(h, w, 4), src)


# ---- Spin Blur and Path Blur ---------------------------------------------------------
#
# Kooka's blur/spin.rs and blur/path.rs with the Blur Gallery's Noise pass
# (blur/field.rs). Both take any number of spins / curved paths as JSON, the
# way the Blur Gallery tool places them; without it, the sliders' one.

MAX_BLUR_PINS = 16                # spins or paths, as Kooka's MAX_FIELD_PINS
MAX_PATH_POINTS = 32
MAX_SPIN_TAPS = 128
MAX_PATH_TAPS = 64
_SPAN_SEGMENTS = 16               # curve samples per span between two path points
_FIELD_CELLS = 512                # the motion field's grid, cells on the long edge
NOISE_KINDS = ("gaussian", "uniform", "grain")


def _noise_props():
    """The Blur Gallery's Noise tab. Amount 0 adds nothing."""
    return (IntProperty("noise_amount", 0, 0, 100, label="Noise Amount"),
            ChoiceProperty("noise_kind", "gaussian", list(NOISE_KINDS),
                           labels={k: k.title() for k in NOISE_KINDS}, label="Noise"),
            IntProperty("noise_size", 50, 0, 100, label="Grain Size"),
            IntProperty("noise_roughness", 50, 0, 100, label="Grain Roughness"),
            IntProperty("noise_color", 0, 0, 100, label="Noise Color"),
            IntProperty("noise_highlights", 0, 0, 100, label="Noise Highlights"),
            IntProperty("noise_seed", 1, 0, 9999, label="Noise Seed"))


def _noise_of(t):
    return {k: t[k] for k in ("noise_amount", "noise_kind", "noise_size", "noise_roughness",
                              "noise_color", "noise_highlights", "noise_seed")}


_U64 = np.uint64


def _unit64(x, y, salt):
    """Kooka's 64-bit coordinate hash (field.rs `unit`): 0..1 (53 bits)."""
    x = np.atleast_1d(np.asarray(x, np.int64)).astype(_U64)
    y = np.atleast_1d(np.asarray(y, np.int64)).astype(_U64)
    h = (x * _U64(0x9E3779B97F4A7C15)) ^ (y * _U64(0xC2B2AE3D27D4EB4F)) ^ _U64(int(salt))
    h ^= h >> _U64(30)
    h = h * _U64(0xBF58476D1CE4E5B9)
    h ^= h >> _U64(27)
    h = h * _U64(0x94D049BB133111EB)
    h ^= h >> _U64(31)
    return (h >> _U64(11)).astype(np.float64) * (1.0 / float(1 << 53))


def _noise_value(noise, x, y, channel):
    """One noise sample in [-1, 1) per pixel; `channel` separates the color draws."""
    seed = int(noise["noise_seed"])
    rot = ((seed >> 32) & 0xFFFFFFFF)
    rot = ((rot << 13) | (rot >> 19)) & 0xFFFFFFFF
    salt = (seed & 0xFFFFFFFF) ^ rot ^ ((channel * 0x9E3779B1) & 0xFFFFFFFF)
    kind = noise["noise_kind"]
    if kind == "uniform":
        return _unit64(x, y, salt) * 2.0 - 1.0
    if kind == "grain":
        cell = 1.0 + noise["noise_size"] / 100.0 * 15.0
        fx, fy = x / cell, y / cell
        x0, y0 = np.floor(fx), np.floor(fy)
        tx, ty = fx - x0, fy - y0
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        top = _unit64(x0, y0, salt) * (1 - tx) + _unit64(x0 + 1, y0, salt) * tx
        bottom = _unit64(x0, y0 + 1, salt) * (1 - tx) + _unit64(x0 + 1, y0 + 1, salt) * tx
        smooth = top * (1 - ty) + bottom * ty
        r = noise["noise_roughness"] / 100.0
        return (smooth * (1 - r) + _unit64(x, y, salt) * r) * 2.0 - 1.0
    # Three uniform draws summed: a near-Gaussian bell, 1.732 restoring the
    # uniform draw's variance.
    bell = (_unit64(x, y, salt) + _unit64(x, y, salt ^ 0x9E3779B1)
            + _unit64(x, y, salt ^ 0x85EBCA77)) / 3.0
    return np.clip((bell * 2.0 - 1.0) * 1.732, -1.0, 1.0)


def _blur_noise(out, weights, noise):
    """The Noise pass over the blurred pixels (field.rs `apply_effects`
    without bokeh): grain gated by each pixel's blur weight (0..255), so
    sharp pixels never change. Color only; alpha is left as it is."""
    if not noise or noise["noise_amount"] <= 0:
        return out
    h, w = out.shape[:2]
    flat = np.flatnonzero(weights.reshape(-1) > 0)
    if not len(flat):
        return out
    rgb = out.reshape(-1, 4)[flat, :3].astype(np.float64)
    weight = weights.reshape(-1)[flat] / 255.0
    luma = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    gate = np.clip(1.0 - noise["noise_highlights"] / 100.0 * luma / 255.0, 0.0, 1.0)
    amplitude = noise["noise_amount"] / 100.0 * 127.5 * weight * gate
    x, y = flat % w, flat // w
    mono = _noise_value(noise, x, y, 0)
    mix = noise["noise_color"] / 100.0
    for c in range(3):
        draw = mono * (1 - mix) + _noise_value(noise, x, y, c) * mix if mix > 0 else mono
        rgb[:, c] += np.where(amplitude > 0, draw * amplitude, 0.0)
    res = out.reshape(-1, 4).copy()
    res[flat, :3] = _rnd8(rgb)
    return res.reshape(out.shape)


def _finite(values):
    return all(math.isfinite(v) for v in values)


def parse_spins(text):
    """The `spins` JSON -- [[x, y, radius_x, radius_y, angle, feather,
    blur_angle, pivot_x, pivot_y], ...], Kooka's nine slots: the centre as
    fractions of the image, the radii as fractions of its width, the
    pivot's offset as fractions of the image -- or None when there is none
    (or it does not parse). Bad rows are skipped, values clamped."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        rows = json.loads(text)
        spins = []
        for row in list(rows)[:MAX_BLUR_PINS]:
            v = [float(x) for x in row]
            if len(v) != 9 or not _finite(v):
                continue
            v[2], v[3] = (min(max(r, 1e-4), 10.0) for r in v[2:4])
            v[5] = min(max(v[5], 0.0), 1.0)
            v[6] = min(max(v[6], 0.0), 360.0)
            spins.append(v)
        return spins
    except (ValueError, TypeError):
        return None


def spin_blur(src, spins, strength=0.0, flashes=4, duration=1.0, noise=None):
    """Kooka's Spin Blur: each spin an ellipse with a feather ring and a
    rotation centre; a pixel takes the spin whose mask is largest and
    averages taps along the circle through it round that spin's centre,
    the strobe flashes mixed in (`strength` 0..100, `flashes` over the arc,
    each lighting `duration` degrees of it). An empty list changes nothing."""
    h, w = src.shape[:2]
    if not spins or all(s[6] <= 0 for s in spins):
        return src.copy()
    flat = np.arange(h * w)
    x, y = flat % w + 0.5, flat // w + 0.5
    best = np.full(h * w, -1, np.int64)
    best_mask = np.zeros(h * w)
    best_rho = np.zeros(h * w)
    for i, (sx, sy, rx, ry, angle, feather, _blur, _px, _py) in enumerate(spins):
        a = math.radians(angle)
        cos, sin = math.cos(a), math.sin(a)
        cx, cy = sx * w, sy * h
        u = ((x - cx) * cos + (y - cy) * sin) / (rx * w)
        v = ((y - cy) * cos - (x - cx) * sin) / (ry * w)
        rho = np.hypot(u, v)
        tt = np.clip((rho - feather) / max(1 - feather, 1e-12), 0, 1)
        mask = np.where(rho >= 1, 0.0, np.where(rho <= feather, 1.0, 1 - tt * tt * (3 - 2 * tt)))
        better = mask > best_mask
        best = np.where(better, i, best)
        best_mask = np.where(better, mask, best_mask)
        best_rho = np.where(better, rho, best_rho)
    table = np.array([[s[0] * w + s[7] * w, s[1] * h + s[8] * h, s[5], math.radians(s[6])]
                      for s in spins])
    pick = np.maximum(best, 0)
    px, py, feather, full = (table[pick, k] for k in range(4))
    ring = np.where(best_rho <= feather, 1.0, feather / np.maximum(best_rho, 1e-12))
    sweep = np.where(best >= 0, full * best_mask * ring, 0.0)
    dx, dy = x - px, y - py
    length = np.hypot(dx, dy) * sweep
    moving = (best >= 0) & (length >= 1)
    if not moving.any():
        return src.copy()
    count = np.clip(np.ceil(length) + 1, 2, MAX_SPIN_TAPS).astype(np.int64)
    step = sweep / (count - 1)
    strength = strength / 100.0
    flashes = int(flashes)
    half = math.radians(duration) / 2.0
    ks = np.arange(MAX_SPIN_TAPS)[:, None]

    def chunk_taps(idx):
        m = idx[moving[idx]]
        n, sw, st = count[m], sweep[m], step[m]
        cols = np.arange(len(m))
        phase = -sw / 2.0 + st * ks
        weights = np.where(ks < n, 1.0 / n, 0.0)
        if strength > 0 and len(m):
            # Each flash lights the taps within half its duration, and at
            # least the nearest one, so a short flash on a coarse arc lands.
            lit = np.zeros(phase.shape, bool)
            if flashes == 1:
                lit |= np.abs(phase - 0.0) <= half
            else:
                gap = sw / (flashes - 1)
                near = np.floor(st * ks / gap + 0.5)
                for d in (-1, 0, 1):
                    j = np.clip(near + d, 0, flashes - 1)
                    lit |= np.abs(phase - (-sw / 2.0 + sw * j / (flashes - 1))) <= half
            for j in range(flashes):
                at = 0.0 if flashes == 1 else -sw / 2.0 + sw * j / (flashes - 1)
                nearest = np.floor((at + sw / 2.0) / st + 0.5).astype(np.int64)
                lit[np.minimum(nearest, n - 1), cols] = True
            lit &= ks < n
            flash = np.where(lit, 1.0 / np.maximum(lit.sum(0), 1), 0.0)
            weights = weights * (1 - strength) + flash * strength
        mdx, mdy, mpx, mpy = dx[m], dy[m], px[m], py[m]

        def tap(k):
            sub = k < n
            sel = moving[idx] & (k < count[idx])
            c, s = np.cos(phase[k, sub]), np.sin(phase[k, sub])
            return (sel, mpx[sub] + mdx[sub] * c - mdy[sub] * s,
                    mpy[sub] + mdx[sub] * s + mdy[sub] * c, weights[k, sub])
        return moving[idx], tap
    out = _gather(src, chunk_taps, MAX_SPIN_TAPS)
    weight = np.where(moving, np.floor(best_mask * 255.0 + 0.5), 0).reshape(h, w)
    return _blur_noise(out, weight, noise)


class SpinBlur(_Filter):
    """Rotational blur inside ellipses, as a spinning wheel photographed:
    full strength inside Feather, fading to nothing at the rim. `spins`
    (JSON, placed with the Blur Gallery tool) holds any number; without it
    the sliders make one."""
    name, label, category = "spin_blur", "Spin Blur", "Blurs"
    moves_alpha = True

    def declare(self):
        return _props(*_centre(),
                      FloatProperty("radius_x", 0.13, 0.01, 10.0, label="Radius X (of width)"),
                      FloatProperty("radius_y", 0.13, 0.01, 10.0, label="Radius Y (of width)"),
                      FloatProperty("angle", 0.0, -360.0, 360.0, decimals=0),
                      FloatProperty("feather", 0.5, 0.0, 1.0),
                      FloatProperty("blur_angle", 15.0, 0.0, 360.0, decimals=0),
                      FloatProperty("rotation_x", 0.0, -10.0, 10.0, label="Rotation Center X"),
                      FloatProperty("rotation_y", 0.0, -10.0, 10.0, label="Rotation Center Y"),
                      IntProperty("strobe_strength", 0, 0, 100),
                      IntProperty("strobe_flashes", 4, 1, 100),
                      FloatProperty("strobe_duration", 1.0, 0.0, 360.0, decimals=0,
                                    label="Strobe Flash Duration"),
                      *_noise_props(),
                      TextProperty("spins", "", label="Spins (from the Blur Gallery tool)"))

    def compute(self, t, src):
        spins = parse_spins(t.get("spins"))
        if spins is None:
            spins = [[t["center_x"] / 100.0, t["center_y"] / 100.0, t["radius_x"], t["radius_y"],
                      t["angle"], t["feather"], t["blur_angle"], t["rotation_x"], t["rotation_y"]]]
        return spin_blur(src, spins, t["strobe_strength"], t["strobe_flashes"],
                         t["strobe_duration"], _noise_of(t))


def parse_paths(text):
    """The `paths` JSON -- [{"points": [[x, y], ...], "start_speed": s,
    "end_speed": e}, ...], points as fractions of the image (2 to 32 of
    them), speeds in pixels (0..500) -- or None when there is none."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        out = []
        for item in list(json.loads(text))[:MAX_BLUR_PINS]:
            pts = [(float(px), float(py)) for px, py in item["points"]]
            speeds = [float(item.get("start_speed", 0)), float(item.get("end_speed", 0))]
            if not 2 <= len(pts) <= MAX_PATH_POINTS or not _finite(
                    [c for p in pts for c in p] + speeds):
                continue
            out.append({"points": pts, "start_speed": min(max(speeds[0], 0.0), 500.0),
                        "end_speed": min(max(speeds[1], 0.0), 500.0)})
        return out
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def catmull_rom(points):
    """The uniform Catmull-Rom curve through `points` (N, 2), the ends' outer
    control points mirrored, 16 samples a span: (M, 2)."""
    p = np.asarray(points, np.float64)
    ext = np.concatenate([2 * p[:1] - p[1:2], p, 2 * p[-1:] - p[-2:-1]])
    t = (np.arange(_SPAN_SEGMENTS) / float(_SPAN_SEGMENTS))[None, :, None]
    t2, t3 = t * t, t * t * t
    a, b, c, d = (ext[k:k + len(p) - 1][:, None, :] for k in range(4))
    s = 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2
               + (3 * b - a - 3 * c + d) * t3)
    return np.concatenate([s.reshape(-1, 2), p[-1:]])


class _MotionField:
    """Kooka's motion vector field: every path's nearest-point tangent times
    its interpolated speed, blended by inverse-square distance, on a grid of
    at most 512 cells on the long edge, read back bilinearly."""

    def __init__(self, paths, scale, w, h):
        cell = max(-(-max(w, h) // _FIELD_CELLS), 1)
        self.cols, self.rows = -(-w // cell) + 1, -(-h // cell) + 1
        self.cell = float(cell)
        gy, gx = np.mgrid[0:self.rows, 0:self.cols].astype(np.float64) * self.cell
        gx, gy = gx.ravel(), gy.ravel()
        sx, sy, wsum = np.zeros(len(gx)), np.zeros(len(gx)), np.zeros(len(gx))
        for path in paths:
            d2, vx, vy = self._nearest(path, gx, gy, w, h)
            wgt = 1.0 / (d2 / (float(w) * w) + 1e-6)
            sx += wgt * vx
            sy += wgt * vy
            wsum += wgt
        self.vx = (sx / wsum * scale).reshape(self.rows, self.cols)
        self.vy = (sy / wsum * scale).reshape(self.rows, self.cols)

    @staticmethod
    def _nearest(path, x, y, w, h):
        pts = np.asarray(path["points"], np.float64) * (w, h)
        samples = catmull_rom(pts)
        seg = np.diff(samples, axis=0)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        if arc[-1] > 0:
            arc = arc / arc[-1]
        len2 = (seg ** 2).sum(1)
        live = len2 > 0
        d2 = np.full(len(x), np.inf)
        vx, vy = np.zeros(len(x)), np.zeros(len(x))
        if live.any():
            a, s, l2, arc0 = samples[:-1][live], seg[live], len2[live], arc[:-1][live]
            arc1 = arc[1:][live]
            step = max(1, (1 << 21) // len(a))
            for i0 in range(0, len(x), step):
                px, py = x[i0:i0 + step, None], y[i0:i0 + step, None]
                t = np.clip(((px - a[:, 0]) * s[:, 0] + (py - a[:, 1]) * s[:, 1]) / l2, 0, 1)
                qx, qy = a[:, 0] + s[:, 0] * t - px, a[:, 1] + s[:, 1] * t - py
                dist = qx * qx + qy * qy
                j = np.argmin(dist, axis=1)
                rows = np.arange(len(j))
                tj = t[rows, j]
                frac = arc0[j] + (arc1[j] - arc0[j]) * tj
                speed = path["start_speed"] + (path["end_speed"] - path["start_speed"]) * frac
                n = np.sqrt(l2[j])
                d2[i0:i0 + step] = dist[rows, j]
                vx[i0:i0 + step] = s[j, 0] / n * speed
                vy[i0:i0 + step] = s[j, 1] / n * speed
        else:
            # Every point coincides: the path has no direction.
            d2 = (samples[0, 0] - x) ** 2 + (samples[0, 1] - y) ** 2
        return d2, vx, vy

    def longest(self):
        return float(np.hypot(self.vx, self.vy).max())

    def at(self, x, y):
        gx = np.clip(x / self.cell, 0, self.cols - 1)
        gy = np.clip(y / self.cell, 0, self.rows - 1)
        x0, y0 = np.floor(gx).astype(np.intp), np.floor(gy).astype(np.intp)
        x1, y1 = np.minimum(x0 + 1, self.cols - 1), np.minimum(y0 + 1, self.rows - 1)
        tx, ty = gx - x0, gy - y0
        out = []
        for f in (self.vx, self.vy):
            top = f[y0, x0] + (f[y0, x1] - f[y0, x0]) * tx
            bottom = f[y1, x0] + (f[y1, x1] - f[y1, x0]) * tx
            out.append(top + (bottom - top) * ty)
        return out


def path_blur(src, paths, speed=50, taper=0, centered=True, rear_sync=False,
              strobe_strength=0, strobe_flashes=4, noise=None):
    """Kooka's Path Blur: motion blur along the streamlines of the field the
    paths make (each the Catmull-Rom curve through its points, speeds eased
    from its start to its end, all scaled by `speed` %), half behind and
    half ahead of a pixel with `centered`, weighted by taper, strobe flashes
    and Rear Sync Flash. An empty list changes nothing."""
    h, w = src.shape[:2]
    if not paths:
        return src.copy()
    field = _MotionField(paths, speed / 100.0, w, h)
    longest = field.longest()
    if longest < 1:
        return src.copy()
    flat = np.arange(h * w)
    x, y = flat % w + 0.5, flat // w + 0.5
    vx, vy = field.at(x, y)
    length = np.hypot(vx, vy)
    moving = length >= 1
    if not moving.any():
        return src.copy()
    count = np.clip(np.ceil(length) + 1, 2, MAX_PATH_TAPS).astype(np.int64)
    step = length / (count - 1)
    behind = (count - 1) // 2 if centered else count - 1
    taper = taper / 100.0
    strength = strobe_strength / 100.0
    flashes = int(strobe_flashes)
    ks = np.arange(MAX_PATH_TAPS)[:, None]

    def chunk_taps(idx):
        m = idx[moving[idx]]
        n, be, st = count[m], behind[m], step[m]
        ah = n - 1 - be
        cols = np.arange(len(m))
        # Walk the streamline each way, re-reading the direction at every
        # step, so a curved path's trail bends with it.
        tx = np.zeros((MAX_PATH_TAPS, len(m)))
        ty = np.zeros((MAX_PATH_TAPS, len(m)))
        tx[be, cols], ty[be, cols] = x[m], y[m]
        ux, uy = vx[m] / length[m], vy[m] / length[m]
        for sign, steps in ((-1.0, be), (1.0, ah)):
            px, py, dx, dy = x[m].copy(), y[m].copy(), ux.copy(), uy.copy()
            for k in range(1, int(steps.max(initial=0)) + 1):
                act = np.flatnonzero(k <= steps)
                px[act] += sign * dx[act] * st[act]
                py[act] += sign * dy[act] * st[act]
                at = be[act] - k if sign < 0 else be[act] + k
                tx[at, act], ty[at, act] = px[act], py[act]
                fx, fy = field.at(px[act], py[act])
                norm = np.hypot(fx, fy)
                ok = norm > 1e-9
                dx[act[ok]], dy[act[ok]] = fx[ok] / norm[ok], fy[ok] / norm[ok]
        reach = np.maximum(np.maximum(be, ah), 1).astype(np.float64)
        base = np.where(ks < n, 1 - taper * np.abs(ks - be) / reach, 0.0)
        weights = base / base.sum(0)
        if strength > 0:
            strobe = np.zeros_like(base)
            for j in range(flashes):
                at = be if flashes == 1 else np.floor(j * (n - 1) / (flashes - 1.0) + 0.5).astype(
                    np.int64)
                np.add.at(strobe, (at, cols), base[at, cols])
            total = strobe.sum(0)
            weights = np.where(total > 0, weights * (1 - strength)
                               + strobe / np.where(total > 0, total, 1) * strength, weights)
        if rear_sync:
            weights[be, cols] += 1.0
            weights = weights / 2.0

        def tap(k):
            sub = k < n
            sel = moving[idx] & (k < count[idx])
            return sel, tx[k, sub], ty[k, sub], weights[k, sub]
        return moving[idx], tap
    out = _gather(src, chunk_taps, MAX_PATH_TAPS)
    weight = np.where(moving, np.clip(np.floor(length / longest * 255.0 + 0.5), 0, 255), 0)
    return _blur_noise(out, weight.reshape(h, w), noise)


class PathBlur(_Filter):
    """Motion blur along paths, each speeding from its Start Speed to its End
    Speed. `paths` (JSON, drawn with the Blur Gallery tool) holds any number
    of curved paths; without it the sliders make one straight one."""
    name, label, category = "path_blur", "Path Blur", "Blurs"
    moves_alpha = True

    def declare(self):
        return _props(IntProperty("start_x", 50, 0, 100, label="Start X (%)"),
                      IntProperty("start_y", 50, 0, 100, label="Start Y (%)"),
                      IntProperty("end_x", 58, 0, 100, label="End X (%)"),
                      IntProperty("end_y", 50, 0, 100, label="End Y (%)"),
                      IntProperty("start_speed", 33, 0, 500), IntProperty("end_speed", 33, 0, 500),
                      IntProperty("speed", 50, 0, 500), IntProperty("taper", 0, 0, 100),
                      BoolProperty("centered", True, label="Centered Blur"),
                      BoolProperty("rear_sync", False, label="Rear Sync Flash"),
                      IntProperty("strobe_strength", 0, 0, 100),
                      IntProperty("strobe_flashes", 4, 1, 100),
                      *_noise_props(),
                      TextProperty("paths", "", label="Paths (from the Blur Gallery tool)"))

    def compute(self, t, src):
        paths = parse_paths(t.get("paths"))
        if paths is None:
            paths = [{"points": [(t["start_x"] / 100.0, t["start_y"] / 100.0),
                                 (t["end_x"] / 100.0, t["end_y"] / 100.0)],
                      "start_speed": float(t["start_speed"]), "end_speed": float(t["end_speed"])}]
        return path_blur(src, paths, t["speed"], t["taper"], t["centered"], t["rear_sync"],
                         t["strobe_strength"], t["strobe_flashes"], _noise_of(t))


# ==== Sharpen ==========================================================================

class SharpenEdges(_Filter):
    """Sharpens only where there are edges, leaving smooth areas alone."""
    name, label, category = "sharpen_edges", "Sharpen Edges", "Photo"

    def declare(self):
        return _props()

    def compute(self, t, src):
        rgb = src[..., :3].astype(np.float64)
        blurred = _gauss_rgb(src, 1.0).astype(np.float64)
        lum = F.luma(blurred)
        p = np.pad(lum, 1, mode="edge")
        h, w = lum.shape

        def s(dx, dy):
            return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        gx = -s(-1, -1) - 2 * s(-1, 0) - s(-1, 1) + s(1, -1) + 2 * s(1, 0) + s(1, 1)
        gy = -s(-1, -1) - 2 * s(0, -1) - s(1, -1) + s(-1, 1) + 2 * s(0, 1) + s(1, 1)
        weight = F.smooth((np.sqrt(gx * gx + gy * gy) / 4.0 - 6.0) / 30.0)
        weight = np.where(src[..., 3] > 0, weight, 0)[..., None]
        return F.put_rgb(src, np.floor(np.clip(rgb + (rgb - blurred) * weight, 0, 255)))


class SharpenMore(_Filter):
    """A stronger Sharpen: Unsharp Mask at 100%, radius 3 (sigma 1)."""
    name, label, category = "sharpen_more", "Sharpen More", "Photo"

    def declare(self):
        return _props()

    def compute(self, t, src):
        rgb = src[..., :3].astype(np.float64)
        blurred = np.floor(F.blur(rgb, 1.0) + 0.5)
        return F.put_rgb(src, rgb + (rgb - blurred) * 1.0)


# ==== Video ============================================================================

class NTSCColors(_Filter):
    """Pulls colors too saturated for broadcast television back inside
    the NTSC gamut, keeping their brightness and hue."""
    name, label, category = "ntsc_colors", "NTSC Colors", "Video"

    def declare(self):
        return _props()

    def compute(self, t, src):
        rgb = src[..., :3].astype(np.float64) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        yy = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.595716 * r - 0.274453 * g - 0.321263 * b
        q = 0.211456 * r - 0.522591 * g + 0.311135 * b
        chroma = np.hypot(i, q)
        scale = np.where(chroma > 0.5, 0.5 / np.maximum(chroma, 0.5), 1.0)
        i, q = i * scale, q * scale
        out = np.stack([yy + 0.956296 * i + 0.621024 * q,
                        yy - 0.272122 * i - 0.647380 * q,
                        yy - 1.106989 * i + 1.704615 * q], -1)
        return F.put_rgb(src, np.clip(out, 0, 1) * 255.0)


# ==== Watermark ========================================================================

_MAGIC, _CELL, _BITS = 0xA5, 8, 66
DURABILITY = {"low": (1, 4), "medium": (2, 6), "high": (4, 8)}
_DURABILITY_CODE = {"low": 0, "medium": 1, "high": 2}
MAX_CREATOR_ID, MAX_ID = 8388607, 16777215


def _push(bits, value, count):
    bits.extend((value >> s) & 1 == 1 for s in range(count - 1, -1, -1))


def _crc8(bits):
    crc = 0
    for bit in bits:
        high = (crc >> 7) & 1
        crc = (crc << 1) & 0xFF
        if int(bit) ^ high:
            crc ^= 0x07
    return crc


def _codeword(kind, creator, ident, durability):
    bits = []
    _push(bits, _MAGIC, 8)
    _push(bits, _DURABILITY_CODE[durability], 2)
    _push(bits, 1 if kind == "transaction_id" else 0, 1)
    _push(bits, creator & MAX_CREATOR_ID, 23)
    _push(bits, ident & MAX_ID, 24)
    _push(bits, _crc8(bits), 8)
    return bits


def _chip(x, y):
    """+1 or -1 per pixel: Kooka's 64-bit coordinate hash."""
    m = np.uint64
    h = m(0x9E3779B97F4A7C15) ^ (x.astype(np.uint64) * m(0x00000100000001B3)) \
        ^ (y.astype(np.uint64) * m(0xC2B2AE3D27D4EB4F))
    h ^= h >> m(30)
    h = h * m(0xBF58476D1CE4E5B9)
    h ^= h >> m(27)
    h = h * m(0x94D049BB133111EB)
    h ^= h >> m(31)
    return np.where(h & m(1) == 0, 1, -1).astype(np.int64)


WATERMARK_KINDS = ("image_id", "transaction_id")


class EmbedWatermark(_Filter):
    """Hides a creator and image (or transaction) ID in the picture as
    faint noise, readable later with Read Watermark. Durability trades
    visibility for surviving edits."""
    name, label, category = "embed_watermark", "Embed Watermark", "Watermark"

    def declare(self):
        return _props(ChoiceProperty("watermark_type", "image_id", list(WATERMARK_KINDS),
                                     labels={"image_id": "Image ID",
                                             "transaction_id": "Transaction ID"}),
                      IntProperty("creator_id", 0, 0, MAX_CREATOR_ID),
                      IntProperty("image_id", 0, 0, MAX_ID, label="Image ID"),
                      ChoiceProperty("durability", "medium", list(DURABILITY),
                                     labels={"low": "Low", "medium": "Medium", "high": "High"}))

    def compute(self, t, src):
        h, w = src.shape[:2]
        repetition, amplitude = DURABILITY[t["durability"]]
        code = np.asarray(_codeword(t["watermark_type"], int(t["creator_id"]), int(t["image_id"]),
                                    t["durability"]))
        xs, ys = F.grid(h, w)
        cols = -(-w // _CELL)
        cell = (ys // _CELL) * cols + xs // _CELL
        bit = (cell % (_BITS * repetition)) // repetition
        delta = amplitude * _chip(xs, ys) * np.where(code[bit], 1, -1)
        out = src.copy()
        out[..., :3] = np.clip(src[..., :3].astype(np.int64) + delta[..., None], 0, 255)
        return out


def read_watermark(rgba):
    """What Embed Watermark hid in an RGBA picture: a dict with kind,
    creator_id, image_id, durability and margin (the weakest bit's vote),
    or None when there is no readable watermark."""
    h, w = rgba.shape[:2]
    if h == 0 or w == 0:
        return None
    cols, rows = -(-w // _CELL), -(-h // _CELL)
    xs, ys = F.grid(h, w)
    chips = _chip(xs, ys)
    # Per cell and channel: the residual about the cell's (truncated)
    # mean, correlated with the chips.
    ph, pw = rows * _CELL, cols * _CELL
    corr = np.zeros((rows, cols), np.int64)
    for c in range(3):
        v = rgba[..., c].astype(np.int64)
        vp = np.zeros((ph, pw), np.int64)
        vp[:h, :w] = v
        cp = np.zeros((ph, pw), np.int64)
        cp[:h, :w] = chips
        inside = np.zeros((ph, pw), np.int64)
        inside[:h, :w] = 1
        blocks = (rows, _CELL, cols, _CELL)
        sums = vp.reshape(blocks).sum((1, 3))
        counts = inside.reshape(blocks).sum((1, 3))
        means = sums // counts
        resid = (vp - np.repeat(np.repeat(means, _CELL, 0), _CELL, 1)) * inside
        corr += (resid * cp).reshape(blocks).sum((1, 3))
    corr = corr.ravel()
    for repetition, durability in ((1, "low"), (2, "medium"), (4, "high")):
        length = _BITS * repetition
        if length > corr.size:
            continue
        votes = np.bincount(np.arange(corr.size) % length, weights=corr, minlength=length)
        margins = votes.reshape(_BITS, repetition).sum(1)
        bits = margins > 0
        value = 0
        for b in bits:
            value = (value << 1) | int(b)
        if value >> 58 != _MAGIC or _crc8(bits[:58]) != value & 0xFF:
            continue
        code = (value >> 56) & 3
        if code != _DURABILITY_CODE[durability]:
            continue
        return {"kind": "transaction_id" if (value >> 55) & 1 else "image_id",
                "creator_id": int((value >> 32) & MAX_CREATOR_ID),
                "image_id": int((value >> 8) & MAX_ID),
                "durability": durability, "margin": int(np.abs(margins).min())}
    return None


EXTRA = (DiffuseGlow, Glass, OceanRipple, GlowingEdges, SmartBlur, ShapeBlur, Average,
         SpinBlur, PathBlur, SharpenEdges, SharpenMore, NTSCColors, EmbedWatermark)
