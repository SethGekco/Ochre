# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Geometry: resize, canvas size, crop, flip, rotate, merge, flatten.

Everything here rewrites cells in place and knows nothing about history.
The controller wraps each operation in a GeometryDelta or StructureDelta
captured beforehand; keeping that out of here is what lets a batch script
resize a hundred sprites without carrying an undo stack around.

Two rules are load-bearing and both come from the design document:

  * EVERY AVERAGE OF RGBA WEIGHTS COLOUR BY ALPHA. Resampling straight
    alpha without premultiplying produces dark fringes around every
    transparent edge, and it is the single commonest defect in hand-rolled
    editors. So colour is premultiplied, filtered, and divided back out.
  * INDICES ARE NEVER INTERPOLATED. An index plane is resampled with
    nearest-neighbour whatever the user chose for colour, because halfway
    between two palette entries is a third, unrelated colour.
"""

import numpy as np
from PIL import Image

from . import accel
from .geometry import Rect
from .surface import INDEX, RGBA

NEAREST, BILINEAR, BICUBIC, LANCZOS, BOX = "nearest", "bilinear", "bicubic", "lanczos", "box"
METHODS = (NEAREST, BILINEAR, BICUBIC, LANCZOS, BOX)
_PIL = {BILINEAR: Image.BILINEAR, BICUBIC: Image.BICUBIC,
        LANCZOS: Image.LANCZOS, BOX: Image.BOX}

# Anchor fractions for canvas resize: where the OLD image sits in the new one.
ANCHORS = {
    "top-left": (0.0, 0.0), "top": (0.5, 0.0), "top-right": (1.0, 0.0),
    "left": (0.0, 0.5), "center": (0.5, 0.5), "right": (1.0, 0.5),
    "bottom-left": (0.0, 1.0), "bottom": (0.5, 1.0), "bottom-right": (1.0, 1.0),
}


# ---- plane primitives -------------------------------------------------------

def _nearest_grid(src_len, dst_len):
    """Source index for each destination pixel, sampling pixel CENTRES."""
    return np.minimum(((np.arange(dst_len) + 0.5) * src_len / dst_len).astype(np.int64),
                      src_len - 1)


def resample_nearest(plane, width, height):
    ys = _nearest_grid(plane.shape[0], height)
    xs = _nearest_grid(plane.shape[1], width)
    return np.ascontiguousarray(plane[ys][:, xs])


def resample_rgba(rgba, width, height, method=BILINEAR):
    """Resize straight-alpha RGBA, alpha-weighted. Returns (h, w, 4) uint8."""
    rgba = np.asarray(rgba, dtype=np.uint8)
    if method == NEAREST or method not in _PIL:
        return resample_nearest(rgba, width, height)
    if (rgba.shape[1], rgba.shape[0]) == (width, height):
        return rgba.copy()

    alpha = rgba[..., 3].astype(np.float32) / 255.0
    out = np.empty((height, width, 4), dtype=np.float32)
    filt = _PIL[method]
    # Premultiply, filter each channel as a float image, then divide back.
    # Filtering in float keeps the 8-bit rounding to a single final step.
    for c in range(3):
        pre = rgba[..., c].astype(np.float32) * alpha
        out[..., c] = np.asarray(Image.fromarray(pre, "F").resize((width, height), filt))
    out[..., 3] = np.asarray(Image.fromarray(alpha, "F").resize((width, height), filt))

    a = out[..., 3]
    safe = np.where(a > 1e-6, a, 1.0)
    rgb = np.clip(out[..., :3] / safe[..., None], 0.0, 255.0)
    rgb[a <= 1e-6] = 0
    result = np.empty((height, width, 4), dtype=np.uint8)
    result[..., :3] = np.rint(rgb).astype(np.uint8)
    result[..., 3] = np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return result


def flip_plane(plane, horizontal):
    return np.ascontiguousarray(np.flip(plane, axis=1 if horizontal else 0))


def rotate_plane(plane, quarter_turns):
    """Clockwise quarter turns. np.rot90 is counter-clockwise, hence -k."""
    return np.ascontiguousarray(np.rot90(plane, k=-int(quarter_turns)))


# ---- document rebuilds ------------------------------------------------------

def _transparent_for(surface):
    pal = surface.palette
    return 0 if pal is None or pal.transparent is None else int(pal.transparent)


def _rebuild(doc, width, height, convert, bbox_fn):
    """Replace every cell with convert(surface, name, plane) at a new size.

    All frames are thawed first: a frozen frame holds compressed planes and
    a geometry change must rewrite every one of them. Cells are recreated
    through doc.cell() so palette wiring and plane declarations come from the
    layer, exactly as they do on load.
    """
    for frame in doc.frames:
        doc.ensure_warm(frame)
    old = dict(doc.cells)
    doc.width, doc.height = int(width), int(height)
    doc.cells.clear()
    for (layer_id, frame_id), surface in old.items():
        layer = doc.layer(layer_id)
        frame = doc.frame_by_id(frame_id)
        if layer is None or frame is None:
            continue
        fresh = doc.cell(layer, frame)
        for name in surface.authoritative:
            plane = surface.planes.get(name)
            if plane is None:
                continue
            fresh.plane(name)[...] = convert(surface, name, plane)
        # Measured, not predicted: a filter's footprint is wider than the
        # scaled rectangle, and a bbox one pixel too tight loses the edge on
        # the next undo (history snapshots are cropped to it).
        fresh.recompute_content_bbox()
        fresh.refresh_derived()
    return doc.bounds


def resize_image(doc, width, height, method=BILINEAR):
    """Resample every layer to a new size."""
    width, height = max(1, int(width)), max(1, int(height))
    sx, sy = width / float(doc.width), height / float(doc.height)

    def convert(surface, name, plane):
        if name == RGBA:
            return resample_rgba(plane, width, height, method)
        return resample_nearest(plane, width, height)

    def bbox(r):
        return Rect.from_bounds(int(np.floor(r.x * sx)), int(np.floor(r.y * sy)),
                                int(np.ceil(r.x1 * sx)), int(np.ceil(r.y1 * sy)))

    return _rebuild(doc, width, height, convert, bbox)


def reframe(doc, width, height, dx, dy):
    """New canvas of width x height with the old content shifted by (dx, dy).

    Crop and canvas-size are both this: crop shifts negatively and shrinks,
    canvas-size grows and anchors. Index planes fill with the palette's
    transparent entry rather than zero, since those need not be the same.
    """
    width, height = max(1, int(width)), max(1, int(height))
    dx, dy = int(dx), int(dy)

    def convert(surface, name, plane):
        fill = _transparent_for(surface) if name == INDEX else 0
        shape = (height, width) + plane.shape[2:]
        out = np.full(shape, fill, dtype=np.uint8)
        src = Rect(0, 0, plane.shape[1], plane.shape[0])
        dst = src.translated(dx, dy).intersect(Rect(0, 0, width, height))
        if dst is not None:
            out[dst.slice()] = plane[dst.y - dy:dst.y1 - dy, dst.x - dx:dst.x1 - dx]
        return out

    return _rebuild(doc, width, height, convert, lambda r: r.translated(dx, dy))


def resize_canvas(doc, width, height, anchor="center"):
    ax, ay = ANCHORS.get(anchor, ANCHORS["center"]) if isinstance(anchor, str) else anchor
    dx = int(round((int(width) - doc.width) * ax))
    dy = int(round((int(height) - doc.height) * ay))
    return reframe(doc, width, height, dx, dy)


def crop(doc, rect):
    r = rect.clipped_to(doc.width, doc.height)
    if r is None or r.is_empty:
        return None
    return reframe(doc, r.w, r.h, -r.x, -r.y)


def flip_document(doc, horizontal=True):
    w, h = doc.width, doc.height

    def bbox(r):
        return (Rect(w - r.x1, r.y, r.w, r.h) if horizontal
                else Rect(r.x, h - r.y1, r.w, r.h))

    return _rebuild(doc, w, h, lambda s, n, p: flip_plane(p, horizontal), bbox)


def rotate_document(doc, quarter_turns):
    k = int(quarter_turns) % 4
    if k == 0:
        return doc.bounds
    w, h = doc.width, doc.height
    nw, nh = (h, w) if k % 2 else (w, h)

    def bbox(r):
        if k == 2:
            return Rect(w - r.x1, h - r.y1, r.w, r.h)
        if k == 1:        # clockwise: (x, y) -> (h - y1, x)
            return Rect(h - r.y1, r.x, r.h, r.w)
        return Rect(r.y, w - r.x1, r.h, r.w)   # k == 3, counter-clockwise

    return _rebuild(doc, nw, nh, lambda s, n, p: rotate_plane(p, k), bbox)


# ---- single-layer geometry --------------------------------------------------

def _rewrite_layer(doc, layer, fn, bbox_fn):
    touched = None
    for frame in doc.frames:
        doc.ensure_warm(frame)
        surface = doc.cells.get((layer.id, frame.id))
        if surface is None or surface.content_bbox is None:
            continue
        for name in surface.authoritative:
            plane = surface.planes.get(name)
            if plane is not None:
                surface.planes[name] = fn(plane)
        surface.recompute_content_bbox()
        surface.refresh_derived()
        surface.dirty.add(doc.bounds)
        touched = doc.bounds
    return touched


def flip_layer(doc, layer, horizontal=True):
    w, h = doc.width, doc.height
    return _rewrite_layer(
        doc, layer, lambda p: flip_plane(p, horizontal),
        lambda r: Rect(w - r.x1, r.y, r.w, r.h) if horizontal
        else Rect(r.x, h - r.y1, r.w, r.h))


def rotate_layer_180(doc, layer):
    w, h = doc.width, doc.height
    return _rewrite_layer(doc, layer, lambda p: rotate_plane(p, 2),
                          lambda r: Rect(w - r.x1, h - r.y1, r.w, r.h))


# ---- merging ----------------------------------------------------------------

def layer_below(doc, layer):
    """The leaf directly beneath `layer` in its group, or None.

    Merging into a GROUP is refused rather than guessed at: the group's own
    blend and opacity would have to be baked in, and Paint.NET and GIMP
    disagree about what the right answer is.
    """
    parent = layer.parent
    if parent is None or layer.is_group:
        return None
    i = parent.index_of(layer)
    if i is None or i == 0:
        return None
    below = parent.children[i - 1]
    return None if below.is_group else below


def merge_rects(doc, upper, lower):
    """{frame_id: rect} the merge will touch on `lower`. For undo capture."""
    out = {}
    for frame in doc.frames:
        a = doc.cells.get((upper.id, frame.id))
        if a is None or a.content_bbox is None:
            continue
        b = doc.cells.get((lower.id, frame.id))
        rect = a.content_bbox
        if b is not None and b.content_bbox is not None:
            rect = rect.union(b.content_bbox)
        out[frame.id] = rect
    return out


def composite_onto(lower, upper_rgba, rect, opcode, opacity):
    """Blend an rgba block onto `lower` over rect, honouring index locking."""
    from .quantize import index_for

    if not lower.index_locked:
        dst = lower.plane(RGBA)[rect.slice()]
        accel.blend_rect(dst, upper_rgba, opcode, opacity, None, out=dst)
        lower._after_write(rect, RGBA)
        return rect

    # Blend in colour, then land the result as indices: opaque pixels take
    # the nearest palette entry, transparent ones the transparent index.
    temp = lower.plane(RGBA)[rect.slice()].copy()
    accel.blend_rect(temp, upper_rgba, opcode, opacity, None, out=temp)
    idx = index_for(temp[..., :3], lower.palette)
    clear = _transparent_for(lower)
    idx = np.where(temp[..., 3] >= 128, idx, clear).astype(np.uint8)
    lower.plane(INDEX)[rect.slice()] = idx
    lower._after_write(rect, INDEX)
    return rect


def merge_down(doc, upper, lower, blends):
    """Composite `upper` onto `lower` in every frame, then detach `upper`.

    The caller captures history first (see merge_rects). Opacity and blend
    mode are the upper layer's own, which is what "merge down" has meant in
    every editor since Photoshop 3.
    """
    opcode = blends.opcode(upper.blend)
    for frame in doc.frames:
        doc.ensure_warm(frame)
        a = doc.cells.get((upper.id, frame.id))
        if a is None or a.content_bbox is None:
            continue
        rect = merge_rects(doc, upper, lower).get(frame.id)
        if rect is None:
            continue
        b = doc.cell(lower, frame)
        composite_onto(b, a.plane(RGBA)[rect.slice()], rect, opcode, upper.opacity)
    doc.remove_layer(upper)
    return doc.bounds
