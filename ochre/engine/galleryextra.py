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
color of transparent pixels never bleeds in. Two simplifications: Ocean
Ripple's lattice comes from Ochre's seeded hash, not Kooka's ChaCha8
stream (same look, different ripples per seed), and Spin and Path Blur
take one spin / one straight path from sliders, without the Blur Gallery's
bokeh and noise panels (Ochre's other Blur Gallery filters have none).
"""

import math

import numpy as np

from . import fields as F
from .filters import _Filter, _centre
from .props import BoolProperty, ChoiceProperty, ColorProperty, FloatProperty, IntProperty, \
    PropertyCollection
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


class SpinBlur(_Filter):
    """Rotational blur inside an ellipse, as a spinning wheel photographed:
    full strength inside Feather, fading to nothing at the rim."""
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
                      FloatProperty("rotation_y", 0.0, -10.0, 10.0, label="Rotation Center Y"))

    def compute(self, t, src):
        h, w = src.shape[:2]
        if t["blur_angle"] <= 0:
            return src.copy()
        cx, cy = t["center_x"] / 100.0 * w, t["center_y"] / 100.0 * h
        px, py = cx + t["rotation_x"] * w, cy + t["rotation_y"] * h
        a = math.radians(t["angle"])
        cos, sin = math.cos(a), math.sin(a)
        rx, ry = t["radius_x"] * w, t["radius_y"] * w
        feather = t["feather"]
        flat = np.arange(h * w)
        x, y = flat % w + 0.5, flat // w + 0.5
        u = ((x - cx) * cos + (y - cy) * sin) / rx
        v = ((y - cy) * cos - (x - cx) * sin) / ry
        rho = np.hypot(u, v)
        tt = np.clip((rho - feather) / max(1 - feather, 1e-12), 0, 1)
        mask = np.where(rho >= 1, 0.0, np.where(rho <= feather, 1.0, 1 - tt * tt * (3 - 2 * tt)))
        ring = np.where(rho <= feather, 1.0, feather / np.maximum(rho, 1e-12))
        sweep = math.radians(t["blur_angle"]) * mask * ring
        dx, dy = x - px, y - py
        length = np.hypot(dx, dy) * sweep
        count = np.clip(np.ceil(length) + 1, 2, 128).astype(np.int64)
        moving = (mask > 0) & (length >= 1)
        step = sweep / (count - 1)

        def chunk_taps(idx):
            def tap(k):
                sel = moving[idx] & (k < count[idx])
                i = idx[sel]
                phase = -sweep[i] / 2 + step[i] * k
                c, s = np.cos(phase), np.sin(phase)
                return sel, px + dx[i] * c - dy[i] * s, py + dx[i] * s + dy[i] * c, 1.0 / count[i]
            return moving[idx], tap
        return _gather(src, chunk_taps, 128)


class PathBlur(_Filter):
    """Motion blur along a path (here one straight path, Start to End),
    its speed eased from Start Speed to End Speed along the way."""
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
                      IntProperty("strobe_flashes", 4, 1, 100))

    def compute(self, t, src):
        h, w = src.shape[:2]
        ax, ay = t["start_x"] / 100.0 * w, t["start_y"] / 100.0 * h
        bx, by = t["end_x"] / 100.0 * w, t["end_y"] / 100.0 * h
        ex, ey = bx - ax, by - ay
        span = math.hypot(ex, ey)
        if span <= 0:
            return src.copy()
        ux, uy = ex / span, ey / span
        flat = np.arange(h * w)
        x, y = flat % w + 0.5, flat // w + 0.5
        s = np.clip(((x - ax) * ex + (y - ay) * ey) / (span * span), 0, 1)
        length = (t["start_speed"] + (t["end_speed"] - t["start_speed"]) * s) * t["speed"] / 100.0
        moving = length >= 1
        if not moving.any():
            return src.copy()
        count = np.clip(np.ceil(length) + 1, 2, 64).astype(np.int64)
        step = length / (count - 1)
        behind = (count - 1) // 2 if t["centered"] else count - 1
        taper = t["taper"] / 100.0
        strength = t["strobe_strength"] / 100.0
        flashes = int(t["strobe_flashes"])
        ks = np.arange(64)[:, None]

        def chunk_taps(idx):
            n, be = count[idx], behind[idx]
            reach = np.maximum(np.maximum(be, n - 1 - be), 1).astype(np.float64)
            base = np.where(ks < n, 1 - taper * np.abs(ks - be) / reach, 0.0)
            weights = base / base.sum(0)
            cols = np.arange(len(idx))
            if strength > 0:
                strobe = np.zeros_like(base)
                for j in range(flashes):
                    at = be if flashes == 1 else F.rnd(j * (n - 1) / (flashes - 1)).astype(np.int64)
                    np.add.at(strobe, (at, cols), base[at, cols])
                total = strobe.sum(0)
                weights = np.where(total > 0, weights * (1 - strength)
                                   + strobe / np.where(total > 0, total, 1) * strength, weights)
            if t["rear_sync"]:
                weights[be, cols] += 1.0
                weights = weights / 2.0

            def tap(k):
                sel = moving[idx] & (k < n)
                i = idx[sel]
                offset = (k - behind[i]) * step[i]
                return sel, x[i] + ux * offset, y[i] + uy * offset, weights[k, sel]
            return moving[idx], tap
        return _gather(src, chunk_taps, 64)


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
