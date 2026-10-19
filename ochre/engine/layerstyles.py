# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Layer styles: live effects drawn from a layer's shape, Photoshop's set.

A style never touches the layer's pixels. The compositor asks for two
images -- what goes BEHIND the layer (drop shadow, outer glow) and what goes
IN FRONT of it (overlays, inner shadow and glow, bevel, stroke) -- and
blends them around the layer itself, so editing the layer updates its
shadow and stroke as you paint. Paint.NET has nothing like this and GIMP
only gained non-destructive filters in 3.0; this is the Photoshop feature.

Everything is computed from the layer's alpha alone, in float32, over the
layer's content box grown by however far the effects reach, and the
compositor caches the result until the layer, or its styles, change.

A style is a name plus a parameter dict; layer.styles maps one to the other
and only enabled entries render. Names and defaults are in STYLES.
"""

import math

import numpy as np

from .morph import distance_sq

# name: (label, defaults). Colours are (r, g, b) tuples; opacity is 0..100.
STYLES = {
    "drop_shadow": ("Drop Shadow", {"color": (0, 0, 0), "opacity": 75, "angle": 120,
                                    "distance": 5, "spread": 0, "size": 5}),
    "outer_glow": ("Outer Glow", {"color": (255, 255, 190), "opacity": 75,
                                  "spread": 0, "size": 10}),
    "inner_shadow": ("Inner Shadow", {"color": (0, 0, 0), "opacity": 75, "angle": 120,
                                      "distance": 5, "size": 5}),
    "inner_glow": ("Inner Glow", {"color": (255, 255, 190), "opacity": 75, "size": 10}),
    "bevel": ("Bevel & Emboss", {"size": 5, "depth": 100, "angle": 120, "altitude": 30,
                                 "highlight": (255, 255, 255), "highlight_opacity": 75,
                                 "shadow": (0, 0, 0), "shadow_opacity": 75}),
    "satin": ("Satin", {"color": (0, 0, 0), "opacity": 50, "angle": 19, "distance": 11,
                        "size": 14, "invert": 1}),
    "color_overlay": ("Color Overlay", {"color": (255, 0, 0), "opacity": 100}),
    "gradient_overlay": ("Gradient Overlay", {"start": (0, 0, 0), "end": (255, 255, 255),
                                              "angle": 90, "opacity": 100}),
    "pattern_overlay": ("Pattern Overlay", {"pattern": "checker", "tile": "", "scale": 100,
                                            "opacity": 100}),
    "stroke": ("Stroke", {"color": (255, 0, 0), "size": 3, "position": "outside",
                          "opacity": 100}),
}
BEHIND = ("outer_glow", "drop_shadow")
# Bottom-first, Photoshop's stacking: Pattern, Gradient and Color Overlay,
# Satin, Inner Glow, Inner Shadow, Bevel, Stroke.
FRONT = ("pattern_overlay", "gradient_overlay", "color_overlay", "satin", "inner_glow",
         "inner_shadow", "bevel", "stroke")


def defaults(name):
    return dict(STYLES[name][1])


def params(styles, name):
    """The effective parameters of one style: defaults, then the layer's."""
    p = defaults(name)
    p.update((styles or {}).get(name) or {})
    return p


def enabled(styles):
    return [n for n in STYLES if (styles or {}).get(n) and (styles[n].get("enabled", True))]


def reach(styles):
    """How far, in pixels, the styles can draw outside the layer's content."""
    r = 0.0
    for name in enabled(styles):
        p = params(styles, name)
        if name == "drop_shadow":
            r = max(r, p["distance"] + p["spread"] + 3 * p["size"])
        elif name == "outer_glow":
            r = max(r, p["spread"] + 3 * p["size"])
        elif name == "stroke" and p["position"] != "inside":
            r = max(r, p["size"] + 1)
    return int(math.ceil(r)) + 2


def _blur(a, size):
    """Gaussian-ish blur: three box passes, the classic approximation."""
    size = float(size)
    if size <= 0.5:
        return a
    r = max(1, int(round(size / 2.0)))
    out = a
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis, dtype=np.float64)
            n = out.shape[axis]
            hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
            lo = np.take(c, np.arange(0, n), axis=axis)
            out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def _shift(a, dx, dy):
    """a moved by a whole-pixel offset; what slides in is empty."""
    dx, dy = int(round(dx)), int(round(dy))
    out = np.zeros_like(a)
    h, w = a.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    if h - abs(dy) > 0 and w - abs(dx) > 0:
        out[yd, xd] = a[ys, xs]
    return out


def _grow(a, radius):
    """Soft dilation of an alpha field by a Euclidean radius."""
    if radius <= 0:
        return a
    d = np.sqrt(distance_sq(a >= 0.5, radius + 1))
    return np.maximum(a, np.clip(radius + 0.5 - d, 0, 1).astype(np.float32))


def _offset(angle, distance):
    # Photoshop's angle is where the LIGHT comes from; the shadow falls the
    # other way. y grows downward on screen.
    t = math.radians(angle)
    return -math.cos(t) * distance, math.sin(t) * distance


def _layer(colour, alpha):
    out = np.zeros(alpha.shape + (4,), dtype=np.float32)
    out[..., 0], out[..., 1], out[..., 2] = colour[:3]
    out[..., 3] = np.clip(alpha, 0, 1) * 255.0
    return out


def _over(dst, src):
    """Straight-alpha source-over, float, in place on dst."""
    sa = src[..., 3:4] / 255.0
    da = dst[..., 3:4] / 255.0
    oa = sa + da * (1 - sa)
    safe = np.where(oa > 1e-6, oa, 1.0)
    dst[..., :3] = (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / safe
    dst[..., 3:4] = oa * 255.0
    return dst


def render(rgba, styles):
    """(behind, front) float32 RGBA for a layer's pixels under its styles.

    rgba is the layer's pixels over some window (uint8). Either result may
    be None when no enabled style draws there.
    """
    names = enabled(styles)
    if not names:
        return None, None
    A = rgba[..., 3].astype(np.float32) / 255.0
    h, w = A.shape
    behind = None
    for name in BEHIND:
        if name not in names:
            continue
        p = params(styles, name)
        if name == "drop_shadow":
            dx, dy = _offset(p["angle"], p["distance"])
            alpha = _blur(_grow(_shift(A, dx, dy), p["spread"]), p["size"])
        else:
            alpha = _blur(_grow(A, p["spread"]), p["size"])
        # "Layer knocks out drop shadow": none of it shows through the layer.
        alpha = alpha * (1.0 - A) * (p["opacity"] / 100.0)
        piece = _layer(p["color"], alpha)
        behind = piece if behind is None else _over(behind, piece)

    front = None
    for name in FRONT:
        if name not in names:
            continue
        p = params(styles, name)
        if name == "color_overlay":
            piece = _layer(p["color"], A * p["opacity"] / 100.0)
        elif name == "gradient_overlay":
            t = math.radians(p["angle"])
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            ux, uy = math.cos(t), -math.sin(t)
            proj = (xx - w / 2.0) * ux + (yy - h / 2.0) * uy
            span = max(1.0, abs(w * ux) + abs(h * uy))
            k = np.clip(proj / span + 0.5, 0, 1)[..., None]
            piece = np.zeros((h, w, 4), dtype=np.float32)
            s, e = np.array(p["start"][:3], np.float32), np.array(p["end"][:3], np.float32)
            piece[..., :3] = s + (e - s) * k
            piece[..., 3] = A * 255.0 * p["opacity"] / 100.0
        elif name == "satin":
            # Two offset, blurred copies of the shape, differenced: the soft
            # interior sheen of satin.
            dx, dy = _offset(p["angle"], p["distance"] / 2.0)
            a1 = _blur(_shift(A, dx, dy), p["size"])
            a2 = _blur(_shift(A, -dx, -dy), p["size"])
            sheen = np.abs(a1 - a2)
            if int(p.get("invert", 1)):
                sheen = 1.0 - sheen
            piece = _layer(p["color"], np.clip(sheen, 0, 1) * A * (p["opacity"] / 100.0))
        elif name == "pattern_overlay":
            from . import patterns
            t = patterns.decode_tile(p.get("tile")) if p.get("tile") else None
            if t is None:
                t = patterns.builtin_tile(p.get("pattern", "checker"))
            from .geometry import Rect
            tiled = patterns.fill(t, Rect(0, 0, w, h), max(0.05, float(p["scale"]) / 100.0))
            piece = tiled.astype(np.float32)
            piece[..., 3] = piece[..., 3] * A * (p["opacity"] / 100.0)
        elif name == "inner_glow":
            alpha = _blur(1.0 - A, p["size"]) * A * (p["opacity"] / 100.0)
            piece = _layer(p["color"], alpha)
        elif name == "inner_shadow":
            dx, dy = _offset(p["angle"], p["distance"])
            outside = _shift(1.0 - A, dx, dy)
            # Pixels shifted in from beyond the window count as outside.
            alpha = _blur(outside, p["size"]) * A * (p["opacity"] / 100.0)
            piece = _layer(p["color"], alpha)
        elif name == "bevel":
            height = _blur(A, max(1.0, p["size"])) * (p["depth"] / 100.0) * p["size"]
            gy, gx = np.gradient(height)
            t, alt = math.radians(p["angle"]), math.radians(p["altitude"])
            lx, ly, lz = math.cos(t) * math.cos(alt), -math.sin(t) * math.cos(alt), math.sin(alt)
            norm = np.sqrt(gx * gx + gy * gy + 1.0)
            shade = (-gx * lx - gy * ly + lz) / norm - lz
            hi = np.clip(shade, 0, None) * 2.5 * A * (p["highlight_opacity"] / 100.0)
            lo = np.clip(-shade, 0, None) * 2.5 * A * (p["shadow_opacity"] / 100.0)
            piece = _layer(p["shadow"], lo)
            _over(piece, _layer(p["highlight"], hi))
        else:   # stroke
            size = float(p["size"])
            inside = A >= 0.5
            if p["position"] == "inside":
                d = np.sqrt(distance_sq(~inside, size + 1))
                alpha = np.clip(size + 0.5 - d, 0, 1) * A
            elif p["position"] == "center":
                half = size / 2.0
                d_out = np.sqrt(distance_sq(inside, half + 1))
                d_in = np.sqrt(distance_sq(~inside, half + 1))
                alpha = np.maximum(np.clip(half + 0.5 - d_out, 0, 1) * (1 - A),
                                   np.clip(half + 0.5 - d_in, 0, 1) * A)
            else:
                d = np.sqrt(distance_sq(inside, size + 1))
                alpha = np.clip(size + 0.5 - d, 0, 1) * (1.0 - A)
                # The rim pixels of an antialiased edge belong to the stroke too.
                alpha = np.maximum(alpha, (1.0 - A) * (A > 0))
            piece = _layer(p["color"], alpha * p["opacity"] / 100.0)
        front = piece if front is None else _over(front, piece)
    return behind, front


def to_uint8(img):
    if img is None:
        return None
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ---- serialisation (layer sections of the .ochre manifest) -------------------

def encode(styles):
    """{name: params} -> {"Style.name.key": "value"} INI keys."""
    out = {}
    for name, p in (styles or {}).items():
        if name not in STYLES or not p:
            continue
        for key, value in p.items():
            if isinstance(value, (tuple, list)):
                value = ",".join(str(int(v)) for v in value)
            elif isinstance(value, bool):
                value = "1" if value else "0"
            out["Style.%s.%s" % (name, key)] = str(value)
        out.setdefault("Style.%s.enabled" % name, "1")
    return out


def decode(items):
    """Inverse of encode, typed by each style's defaults."""
    styles = {}
    for key, raw in items:
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "Style" or parts[1] not in STYLES:
            continue
        name, field = parts[1], parts[2]
        base = STYLES[name][1].get(field)
        try:
            if field == "enabled":
                value = str(raw).strip() not in ("0", "off", "false", "no")
            elif isinstance(base, tuple):
                value = tuple(int(v) for v in str(raw).split(","))[:3]
            elif isinstance(base, (int, float)) and not isinstance(base, bool):
                value = float(raw) if "." in str(raw) else int(raw)
            else:
                value = str(raw)
        except (TypeError, ValueError):
            continue
        styles.setdefault(name, {})[field] = value
    return styles
