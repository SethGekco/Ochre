# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Shape layers: Photoshop's shape layers, GIMP 3.2's vector layers.

A shape layer holds a vector path and how to paint it -- fill colour (or
none), stroke colour (or none), stroke width and alignment (inside, centre,
outside) -- and the compositor draws it at whatever size the document is.
Resizing the image re-draws it sharp; the fill and stroke stay editable;
Rasterize Layer turns it into pixels.

params: {"path": [encoded subpath, ...], "fill": rgba | None,
         "stroke": rgba | None, "stroke_width": float, "stroke_align": str}
"""

import numpy as np

from . import paths as P


def path_of(params):
    return P.Path("Shape", [P.decode_subpath(s) for s in params.get("path") or []])


def encode_path(path):
    return [P.encode_subpath(s) for s in path.subpaths]


def make(path, fill=None, stroke=(0, 0, 0, 255), stroke_width=2.0, stroke_align="center"):
    return {"path": encode_path(path), "fill": None if fill is None else list(fill),
            "stroke": None if stroke is None else list(stroke),
            "stroke_width": float(stroke_width), "stroke_align": stroke_align}


def mapped(params, fn):
    """The shape after a point mapping (resize, crop, flip, rotate)."""
    path = path_of(params)
    path.map_points(fn)
    return dict(params, path=encode_path(path))


def _paint(out, cov, colour):
    """Source-over a flat colour at coverage cov (uint8) onto out (float RGBA)."""
    a = cov.astype(np.float64) / 255.0 * (colour[3] / 255.0 if len(colour) > 3 else 1.0)
    rgb = np.asarray(colour[:3], np.float64)
    da = out[..., 3]
    na = a + da * (1 - a)
    safe = np.where(na > 1e-9, na, 1.0)
    out[..., :3] = (rgb * a[..., None] + out[..., :3] * (da * (1 - a))[..., None]) / safe[..., None]
    out[..., 3] = na


def render(params, width, height):
    """The shape layer as full-canvas RGBA uint8."""
    path = path_of(params)
    out = np.zeros((height, width, 4), np.float64)
    if path.is_empty:
        return out.astype(np.uint8)
    fill_cov = P.fill_mask(path, width, height) if params.get("fill") is not None \
        or params.get("stroke_align", "center") != "center" else None
    if params.get("fill") is not None:
        _paint(out, fill_cov, params["fill"])
    stroke = params.get("stroke")
    w = float(params.get("stroke_width", 0) or 0)
    if stroke is not None and w > 0:
        align = params.get("stroke_align", "center")
        if align == "center":
            cov = P.stroke_mask(path, width, height, w)
        else:
            band = P.stroke_mask(path, width, height, 2 * w).astype(np.int32)
            f = fill_cov.astype(np.int32)
            cov = band * f // 255 if align == "inside" else band * (255 - f) // 255
            cov = np.clip(cov, 0, 255).astype(np.uint8)
        _paint(out, cov, stroke)
    res = np.empty((height, width, 4), np.uint8)
    res[..., :3] = np.clip(np.rint(out[..., :3]), 0, 255)
    res[..., 3] = np.clip(np.rint(out[..., 3] * 255), 0, 255)
    return res


def path_from_rings(rings, closed=True):
    """A Path of corner anchors from point rings (shapelib geometry)."""
    return P.Path("Shape", [P.Subpath([P.Anchor(x, y) for x, y in ring], closed)
                            for ring in rings if len(ring) >= 2])
