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

import math

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


def _map_paths(doc, fn):
    """Apply a point mapping to every vector path and mark, so they follow
    the image through a resize, crop, flip or rotation."""
    for path in getattr(doc, "paths", ()):
        path.map_points(fn)
    _map_marks(doc, fn)


def _map_text(doc, fn, scale=None):
    """Text layers follow the image: their anchor moves with it and, on a
    resize, the type and its box scale too -- text that stays text."""
    for layer in doc.layers():
        if getattr(layer, "kind", "raster") != "text" or not layer.params:
            continue
        p = dict(layer.params)
        x, y = fn((float(p.get("x", 0)), float(p.get("y", 0))))
        p["x"], p["y"] = x, y
        if scale is not None:
            sx, sy = scale
            p["size"] = max(1, int(round(float(p.get("size", 24)) * sy)))
            if p.get("box_width"):
                p["box_width"] = max(1, int(round(float(p["box_width"]) * sx)))
        layer.params = p


def _map_marks(doc, fn):
    w, h = doc.width, doc.height
    doc.samplers = [tuple(fn((x, y))) for x, y in getattr(doc, "samplers", [])]
    doc.counts = [tuple(fn((x, y))) for x, y in getattr(doc, "counts", [])]
    doc.notes = [tuple(fn((x, y))) + (t,) for x, y, t in getattr(doc, "notes", [])]
    slices = []
    for name, (x, y, sw, sh) in getattr(doc, "slices", []):
        pts = [fn(p) for p in ((x, y), (x + sw, y), (x + sw, y + sh), (x, y + sh))]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0 = int(round(min(xs))), int(round(min(ys)))
        slices.append((name, (x0, y0, int(round(max(xs))) - x0, int(round(max(ys))) - y0)))
    doc.slices = slices
    guides = []
    for orient, pos in getattr(doc, "guides", []):
        a, b = ((pos, 0.0), (pos, float(h))) if orient == "v" else ((0.0, pos), (float(w), pos))
        (ax, ay), (bx, by) = fn(a), fn(b)
        if abs(ax - bx) < 1e-6:
            guides.append(("v", ax))
        elif abs(ay - by) < 1e-6:
            guides.append(("h", ay))
        else:
            # Turned off the axes (rotate by an angle): keep its centre.
            guides.append((orient, (ax + bx) / 2.0 if orient == "v" else (ay + by) / 2.0))
    doc.guides = guides


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

    _map_paths(doc, lambda p: (p[0] * sx, p[1] * sy))
    _map_text(doc, lambda p: (p[0] * sx, p[1] * sy), (sx, sy))
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

    _map_paths(doc, lambda p: (p[0] + dx, p[1] + dy))
    _map_text(doc, lambda p: (p[0] + dx, p[1] + dy))
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

    _map_paths(doc, (lambda p: (w - p[0], p[1])) if horizontal else (lambda p: (p[0], h - p[1])))
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

    _map_paths(doc, {1: lambda p: (h - p[1], p[0]),
                     2: lambda p: (w - p[0], h - p[1]),
                     3: lambda p: (p[1], w - p[0])}[k])
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


# ---- arbitrary geometry through the warp engine --------------------------------

def _warp_plane(surface, name, plane, h, width, height, method):
    from . import warp
    box = (0, 0, int(width), int(height))
    if name == INDEX:
        idx = warp.warp_homography(plane, h, box, index=True)
        fill = _transparent_for(surface)
        return np.where(idx >= 0, idx, fill).astype(np.uint8)
    if name == RGBA:
        return warp.warp_homography(plane, h, box,
                                    warp.NEAREST if method == NEAREST else warp.BILINEAR)
    idx = warp.warp_homography(plane, h, box, index=True)
    return np.where(idx >= 0, idx, 0).astype(np.uint8)


def perspective_crop(doc, quad, method=BILINEAR):
    """Straighten the quadrilateral `quad` (TL, TR, BR, BL) into a rectangle.

    The output is as wide as the quad's longer horizontal side and as tall
    as its longer vertical one, so nothing in it is shrunk. Every layer and
    frame goes through the same homography.
    """
    from . import warp
    q = np.asarray(quad, dtype=np.float64)
    w = int(round(max(np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[3])))))
    h = int(round(max(np.hypot(*(q[3] - q[0])), np.hypot(*(q[2] - q[1])))))
    if w < 1 or h < 1:
        return None
    hm = warp.homography(q, [(0, 0), (w, 0), (w, h), (0, h)])
    _map_paths(doc, lambda p: tuple(warp.apply(hm, [p])[0]))
    return _rebuild(doc, w, h,
                    lambda s, n, p: _warp_plane(s, n, p, hm, w, h, method), lambda r: r)


def rotate_image_by(doc, degrees, method=BILINEAR, expand=True):
    """Rotate the whole image by any angle, growing the canvas to fit."""
    from . import warp
    rad = math.radians(float(degrees))
    w, h = doc.width, doc.height
    if expand:
        c, s = abs(math.cos(rad)), abs(math.sin(rad))
        nw, nh = int(math.ceil(w * c + h * s)), int(math.ceil(w * s + h * c))
    else:
        nw, nh = w, h
    hm = warp.translate(nw / 2.0, nh / 2.0) @ warp.rotate_about(rad, 0, 0) @ \
        warp.translate(-w / 2.0, -h / 2.0)
    _map_paths(doc, lambda p: tuple(warp.apply(hm, [p])[0]))
    return _rebuild(doc, nw, nh,
                    lambda s_, n, p: _warp_plane(s_, n, p, hm, nw, nh, method), lambda r: r)


def transform_layer(doc, layer, hm, method=BILINEAR):
    """Send one layer, every frame, through a homography. Canvas size kept."""
    w, h = doc.width, doc.height

    def fn(plane, name="rgba"):
        return plane

    touched = None
    for frame in doc.frames:
        doc.ensure_warm(frame)
        surface = doc.cells.get((layer.id, frame.id))
        if surface is None or surface.content_bbox is None:
            continue
        for name in surface.authoritative:
            plane = surface.planes.get(name)
            if plane is not None:
                surface.planes[name] = _warp_plane(surface, name, plane, hm, w, h, method)
        surface.recompute_content_bbox()
        surface.refresh_derived()
        surface.dirty.add(doc.bounds)
        touched = doc.bounds
    return touched


def offset_layer(doc, layer, dx, dy, wrap=False):
    """Shift a layer's pixels. With wrap, what leaves one edge enters the
    opposite one -- GIMP's Layer > Offset, the classic way to check that a
    texture tiles."""
    dx, dy = int(dx), int(dy)

    def shift(surface, name, plane):
        if wrap:
            return np.roll(np.roll(plane, dy, axis=0), dx, axis=1)
        fill = _transparent_for(surface) if name == INDEX else 0
        out = np.full_like(plane, fill)
        h, w = plane.shape[:2]
        src = Rect(0, 0, w, h)
        dst = src.translated(dx, dy).intersect(src)
        if dst is not None:
            out[dst.slice()] = plane[dst.y - dy:dst.y1 - dy, dst.x - dx:dst.x1 - dx]
        return out

    touched = None
    for frame in doc.frames:
        doc.ensure_warm(frame)
        surface = doc.cells.get((layer.id, frame.id))
        if surface is None or surface.content_bbox is None:
            continue
        for name in surface.authoritative:
            plane = surface.planes.get(name)
            if plane is not None:
                surface.planes[name] = shift(surface, name, plane)
        surface.recompute_content_bbox()
        surface.refresh_derived()
        surface.dirty.add(doc.bounds)
        touched = doc.bounds
    return touched


ALIGNMENTS = ("left", "hcenter", "right", "top", "vcenter", "bottom")


def align_offset(content, ref, how):
    """(dx, dy) that aligns a content rect to a reference rect."""
    if how == "left":
        return ref.x - content.x, 0
    if how == "right":
        return ref.x1 - content.x1, 0
    if how == "hcenter":
        return int(round((ref.x + ref.x1 - content.x - content.x1) / 2.0)), 0
    if how == "top":
        return 0, ref.y - content.y
    if how == "bottom":
        return 0, ref.y1 - content.y1
    if how == "vcenter":
        return 0, int(round((ref.y + ref.y1 - content.y - content.y1) / 2.0))
    raise ValueError("unknown alignment %r" % (how,))
