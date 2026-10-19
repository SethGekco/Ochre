# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Text layers: text that stays text.

A text layer holds no pixels. Its params -- the string, font, size,
alignment, line spacing, direction, colour, position, an optional paragraph
box width and a warp -- are rendered by the compositor whenever it needs
them, the same way a fill layer is. So the text can be re-edited at any
time (click it with the Text tool), restyled, warped, given a layer style
(stroke and drop shadow are the classic uses), and it stays sharp through a
resize. Rasterize Layer turns it into ordinary pixels when you want to paint
on it.

Paragraph text: with box_width set, lines wrap at spaces to fit. A wrap
REPLACES a space with a newline, so character indices are the same in the
wrapped and unwrapped string and the caret needs no translation.
"""

import math
from collections import OrderedDict

import numpy as np

from . import text as textmod
from .geometry import Rect

DEFAULTS = {"text": "", "font": "DejaVu Sans", "size": 24, "align": "left",
            "line_spacing": 1.0, "orientation": "horizontal", "antialias": True,
            "color": (0, 0, 0, 255), "x": 0.0, "y": 0.0, "box_width": 0,
            "warp": "none", "bend": 50}
WARPS = ("none", "arc", "arch", "bulge", "flag", "wave", "rise", "fish")

_cache = OrderedDict()
_CACHE_SIZE = 32


def full(params):
    p = dict(DEFAULTS)
    p.update(params or {})
    return p


def wrap(body, family, size, width, book=None):
    """Break lines at spaces so none is wider than `width` px.

    Only spaces become newlines (same length), so indices are preserved. A
    single word wider than the box overflows it, as in Photoshop.
    """
    if width <= 0 or not body:
        return body
    font = (book or textmod.DEFAULT_BOOK).load(family, size)

    def measure(s):
        try:
            return font.getlength(s)
        except AttributeError:
            return font.getbbox(s)[2]
    out = list(body)
    line_start = 0
    last_space = -1
    i = 0
    while i < len(out):
        ch = out[i]
        if ch == "\n":
            line_start, last_space = i + 1, -1
        elif ch == " ":
            if measure("".join(out[line_start:i])) > width and last_space >= 0:
                out[last_space] = "\n"
                line_start = last_space + 1
                last_space = -1
                continue          # re-test this space against the new line
            last_space = i
        i += 1
    if last_space >= 0 and measure("".join(out[line_start:])) > width:
        out[last_space] = "\n"
    return "".join(out)


def warp_coverage(cov, style, bend):
    """Bend a rendered text block: Photoshop's Warp Text, the common styles."""
    if style in (None, "", "none") or cov.size == 0 or bend == 0:
        return cov
    from .warp import warp_field
    h, w = cov.shape
    k = float(bend) / 100.0
    pad = int(math.ceil(h * abs(k))) + 2
    H = h + 2 * pad
    ys, xs = np.mgrid[0:H, 0:w].astype(np.float64)
    u = (xs + 0.5) / w * 2 - 1                    # -1..1 across
    v = ys + 0.5 - pad                           # rows in the source
    if style == "arc":
        sy = v + k * h * (u * u) * 0.8
        sx = xs + 0.5
    elif style == "arch":
        sy = v + k * h * (u * u - 1) * 0.6
        sx = xs + 0.5
    elif style == "bulge":
        mid = (v - h / 2.0) / (1 + k * (1 - u * u) * 0.8)
        sy, sx = mid + h / 2.0, xs + 0.5
    elif style == "fish":
        mid = (v - h / 2.0) / (1 + k * (1 - np.abs(u)) * 0.8)
        sy, sx = mid + h / 2.0, xs + 0.5
    elif style == "flag":
        sy = v - k * h * 0.35 * np.sin(u * math.pi)
        sx = xs + 0.5
    elif style == "wave":
        sy = v - k * h * 0.35 * np.sin(u * 2 * math.pi)
        sx = xs + 0.5
    elif style == "rise":
        sy = v + k * h * 0.5 * u
        sx = xs + 0.5
    else:
        return cov
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = cov
    out = warp_field(rgba, sx, sy)[..., 3]
    ys_, xs_ = np.nonzero(out)
    if ys_.size == 0:
        return np.zeros((0, 0), np.uint8)
    return np.ascontiguousarray(out[ys_.min():ys_.max() + 1])


def _key(p):
    return repr(sorted((k, p[k]) for k in DEFAULTS))


def render(params):
    """(rect, coverage) of a text layer in document coordinates, or None."""
    p = full(params)
    key = _key(p)
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit
    body = str(p["text"])
    result = None
    if body.strip():
        vertical = p["orientation"] == "vertical"
        if vertical:
            cov = textmod.vertical_layout(body, family=p["font"], size=int(p["size"]),
                                          line_spacing=float(p["line_spacing"]),
                                          antialias=bool(p["antialias"])).coverage
            align = textmod.LEFT
        else:
            if int(p["box_width"]) > 0:
                body = wrap(body, p["font"], int(p["size"]), int(p["box_width"]))
            cov, _w, _h = textmod.render_coverage(body, family=p["font"], size=int(p["size"]),
                                                  align=p["align"],
                                                  line_spacing=float(p["line_spacing"]),
                                                  antialias=bool(p["antialias"]))
            align = p["align"]
        cov = warp_coverage(cov, p["warp"], int(p["bend"]))
        if cov.size:
            x = float(p["x"])
            if int(p["box_width"]) > 0 and not vertical:
                # Inside a paragraph box, alignment is relative to the box.
                bw = int(p["box_width"])
                x = x + (bw / 2.0 if align == textmod.CENTER else bw if align == textmod.RIGHT else 0)
            rect = textmod.place(cov, x, float(p["y"]), align)
            result = (rect, cov)
    _cache[key] = result
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return result


def bounds(params):
    """The text's rectangle in the document (a box's full width, if boxed)."""
    hit = render(params)
    p = full(params)
    if hit is None:
        return None
    rect = hit[0]
    if int(p["box_width"]) > 0 and p["orientation"] != "vertical":
        x0 = int(math.floor(float(p["x"])))
        rect = rect.union(Rect(x0, rect.y, int(p["box_width"]), rect.h))
    return rect


def contains(params, x, y, pad=4):
    r = bounds(params)
    return r is not None and r.inflated(int(pad)).contains(int(x), int(y))


def pixels(params, r):
    """The text layer's RGBA over document rect r."""
    out = np.zeros((r.h, r.w, 4), np.uint8)
    hit = render(params)
    if hit is None:
        return out
    rect, cov = hit
    inter = rect.intersect(r)
    if inter is None:
        return out
    colour = full(params)["color"]
    if isinstance(colour, str):
        colour = tuple(int(v) for v in colour.split(","))
    colour = tuple(colour) + (255,) * (4 - len(colour))
    sub = cov[inter.y - rect.y:inter.y - rect.y + inter.h, inter.x - rect.x:inter.x - rect.x + inter.w]
    win = out[inter.y - r.y:inter.y - r.y + inter.h, inter.x - r.x:inter.x - r.x + inter.w]
    win[..., :3] = colour[:3]
    win[..., 3] = ((sub.astype(np.uint32) * colour[3] + 127) // 255).astype(np.uint8)
    return out
