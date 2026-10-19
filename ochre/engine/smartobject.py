# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Smart objects: Photoshop's non-destructive layers.

A smart object keeps its ORIGINAL pixels (its SmartSource, at their own
size) plus a transform (a 3x3 homography from source to document) and a
list of smart filters. The compositor renders it from those every time, so:

  * scaling it down and back up loses nothing -- each render resamples the
    original once, from scratch;
  * a filter applied to it stays editable and removable (smart filters),
    applied in order to the transformed result.

params: {"matrix": [9 floats], "src_w": int, "src_h": int,
         "filters": [{"name", "params", "enabled"}, ...]}
"""

import numpy as np

from . import warp

IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def matrix(params):
    return np.asarray(params.get("matrix", IDENTITY), np.float64).reshape(3, 3)


def source_pixels(doc, layer, frame=None):
    src = getattr(layer, "smart_source", None)
    if src is None:
        return None
    frame = frame or doc.frame
    cell = doc.cells.get((src.id, frame.id))
    if cell is None:
        cell = next((c for (lid, _f), c in doc.cells.items() if lid == src.id), None)
    return None if cell is None else cell.plane("rgba")


def render(doc, layer, frame=None, effects=None):
    """The smart object as the document sees it: full-canvas RGBA."""
    src = source_pixels(doc, layer, frame)
    out = np.zeros((doc.height, doc.width, 4), np.uint8)
    if src is None:
        return out
    p = layer.params
    sw, sh = int(p.get("src_w", src.shape[1])), int(p.get("src_h", src.shape[0]))
    src = src[:sh, :sw]
    out = warp.warp_homography(src, matrix(p), (0, 0, doc.width, doc.height), warp.BILINEAR)
    filters = [f for f in p.get("filters", []) if f.get("enabled", True)]
    if filters:
        if effects is None:
            from .effects import EffectRegistry
            effects = EffectRegistry()
        for f in filters:
            effect = effects.create(f.get("name"), **dict(f.get("params") or {}))
            if effect is not None:
                out = effect.apply(out)
    return out


def corners(params):
    """Where the source's corners land in the document."""
    w, h = int(params.get("src_w", 1)), int(params.get("src_h", 1))
    return warp.apply(matrix(params), [(0, 0), (w, 0), (w, h), (0, h)])


def fit_matrix(src_w, src_h, doc_w, doc_h, enlarge=False):
    """Scale-to-fit and centre, as File > Place does."""
    k = min(doc_w / float(src_w), doc_h / float(src_h))
    if not enlarge:
        k = min(1.0, k)
    tx = (doc_w - src_w * k) / 2.0
    ty = (doc_h - src_h * k) / 2.0
    return [k, 0.0, tx, 0.0, k, ty, 0.0, 0.0, 1.0]
