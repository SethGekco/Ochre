# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Reduced-resolution previews: the document composited at a pyramid level.

At Fit to Window an 8000x6000 canvas is drawn about 1000 pixels wide, so a
full-resolution live preview computes some sixty pixels for every one the
screen can show. Measured on the development machine (C accelerator, three
layers over a background, zoom 12%): one opacity-slider tick took 1.0 s and
one Gaussian Blur preview 11-17 s. Here they are computed on a level of the
document reduced by a power of two -- the largest whose pixels are still no
smaller than the screen's -- and the full-resolution result is computed only
on commit: OK, Apply, mouse release.

The rule that makes this safe: NOTHING here writes a document pixel. A
ProxyCompositor reads the document, reduces what it reads, and composites
into an array the canvas draws instead of the display buffer. The committed
result is computed exactly as it always was, so it is bit-identical with or
without previews -- including on index-locked layers, whose reduced preview is
an approximation of their colours and is never written back.

What is reduced, and how:

  * pixel layers and raster masks: box-averaged (accel.reduce_box,
    premultiplied), cached per cell and kept current from the same dirty
    rects the compositor uses, so a second scrub costs only the composite;
  * shape layers, smart objects, non-destructive filters, vector masks and
    layer styles: rendered at full resolution by the real compositor (which
    caches them) and reduced from that;
  * fill layers: drawn on the reduced grid (a pattern's scale divided);
  * text: rendered at full resolution over its own bounds, then reduced;
  * adjustment layers and filter previews: run on the reduced pixels with
    their PIXEL-SIZED settings divided by the factor (scaled_token).

Approximations, stated: a radius divided by eight and rounded is not the same
blur seen through a box filter; Kooka Pictura records the same caveat for its
proxies. A filter that declares neither `pointwise` nor `proxy_scale` is never
previewed reduced -- its preview stays at full resolution, slow but true.
"""

import numpy as np

from . import accel, parallel
from .compositor import Compositor, _active_filters, fill_pixels, mask_value
from .geometry import Rect
from .layer import FILL, SHAPE, SMART, TEXT

#: Largest reduction used for a preview. At 1/32 a 30000-pixel-wide canvas is
#: still about 940 pixels across, which is more than a fitted view shows.
MAX_FACTOR = 32

# A cell with more pending rects than this is re-reduced whole: one big
# reduce beats many small ones, and the list stops growing.
_MAX_STALE = 32


def factor_for_zoom(zoom):
    """The largest power of two f, at most MAX_FACTOR, with f * zoom <= 1.

    1 at 100% and down to just above 50%: a reduced preview there would show
    fewer pixels than the screen does. 2 from 50%, 4 from 25%, and so on, so
    each level has at least one pixel per screen pixel.
    """
    f = 1
    while f < MAX_FACTOR and zoom * (f * 2) <= 1:
        f *= 2
    return f


def reduce_box(img, f):
    """accel.reduce_box, in row bands on the thread pool for a big image.
    Bands start on the block grid, so the result is the same bytes."""
    f = int(f)
    h, w = img.shape[:2]
    oh, ow = -(-h // f), -(-w // f)
    if f == 1 or parallel.bands(Rect(0, 0, w, h), align=f) is None:
        return accel.reduce_box(img, f)
    out = np.empty((oh, ow) + img.shape[2:], np.uint8)

    def work(_out, band):
        out[band.y // f:-(-band.y1 // f)] = accel.reduce_box(img[band.slice()], f)
    parallel.banded(out, Rect(0, 0, w, h), work, align=f)
    return out


def reduce_rect(rect, f):
    """The level-grid rect covering rect at 1/f (rounded outward)."""
    return Rect.from_bounds(rect.x // f, rect.y // f, -(-rect.x1 // f), -(-rect.y1 // f))


def expand_rect(rect, f, width, height):
    """The document rect a level-grid rect stands for, clipped to the canvas."""
    return Rect(rect.x * f, rect.y * f, rect.w * f, rect.h * f).clipped_to(width, height)


def reduce_region(img, rect, f, width, height):
    """(level rect, reduced image) for a full-resolution image covering document
    rect `rect`. The image is placed on the block grid -- padded with
    transparency to whole blocks, except at the canvas edge, where the blocks
    are short exactly as reduce_box makes them for a whole layer."""
    rk = reduce_rect(rect, f)
    full = expand_rect(rk, f, width, height)
    if full == rect:
        return rk, reduce_box(img, f)
    pad = np.zeros((full.h, full.w) + img.shape[2:], np.uint8)
    pad[rect.y - full.y:rect.y1 - full.y, rect.x - full.x:rect.x1 - full.x] = img
    return rk, reduce_box(pad, f)


def scaled_token(effect, f):
    """The effect's settings for a preview at 1/f, or None when it cannot be
    previewed reduced.

    A pointwise effect is the same at any scale. Any other effect must say
    which of its settings are lengths in pixels (`proxy_scale`, a tuple, empty
    when nothing scales -- a percentage, an angle); those are divided by f,
    rounded for integers and kept inside the property's range. An effect that
    says nothing is not guessed at.
    """
    token = dict(effect.props.token())
    if f == 1 or getattr(effect, "pointwise", False):
        return token
    names = getattr(effect, "proxy_scale", None)
    if names is None:
        return None
    for name in names:
        if name not in token:
            continue
        prop = effect.props[name]
        value = token[name] / float(f)
        if isinstance(token[name], int) and not isinstance(token[name], bool):
            value = int(round(value))
        low = getattr(prop, "minimum", None)
        token[name] = value if low is None else max(low, value)
    return token


class _ScaledEffect:
    """An adjustment layer's effect, run on reduced pixels with its settings
    scaled -- what Compositor._blend_leaf calls `apply` on."""

    __slots__ = ("effect", "token")

    def __init__(self, effect, token):
        self.effect = effect
        self.token = token

    def apply(self, source):
        return self.effect.apply(source, token=self.token)


class ProxyCompositor(Compositor):
    """The document composited at 1/factor of its size.

    Every rect handed to or returned by this compositor is on the LEVEL grid,
    ceil(width / factor) x ceil(height / factor). It shares the document and
    the layer tree with the real compositor `full`, borrows its caches for
    anything rendered from a description, and keeps its own cache of reduced
    pixel cells.

    `overrides` maps a layer (or mask) id to a level-sized RGBA array shown in
    place of that cell's pixels: how a filter dialog shows its preview without
    writing it into the layer.
    """

    def __init__(self, full, factor):
        super().__init__(full.doc, full.blends, full._effects_registry)
        self.full = full
        self.factor = int(factor)
        self.width = -(-self.doc.width // self.factor)
        self.height = -(-self.doc.height // self.factor)
        self.overrides = {}
        # (layer id, frame id) -> [surface serial, plane id, reduced, stale]
        # where stale is a list of document rects to re-reduce, or None for
        # the whole cell.
        self._cells = {}
        # (kind, id) -> (what it was reduced from, the reduction)
        self._derived = {}

    @property
    def bounds(self):
        return Rect(0, 0, self.width, self.height)

    def fits(self, document, factor):
        """Still the right compositor for this document at this factor?"""
        return (document is self.doc and factor == self.factor
                and -(-document.width // factor) == self.width
                and -(-document.height // factor) == self.height)

    # ---- keeping reduced cells current -----------------------------------

    def touch(self, key, rects):
        """Cell `key` changed over `rects` (document coordinates)."""
        entry = self._cells.get(key)
        if entry is None or entry[3] is None:
            return
        entry[3].extend(rects)
        if len(entry[3]) > _MAX_STALE:
            entry[3] = None

    def prune(self):
        """Forget reduced cells whose layer or surface no longer exists."""
        for key, entry in list(self._cells.items()):
            cell = self.doc.cells.get(key)
            if cell is None or cell.serial != entry[0]:
                del self._cells[key]
        self._derived = {}

    def reduced_cell(self, key, surface):
        """The reduced RGBA of cell `key` (an override, if one is set)."""
        hit = self.overrides.get(key[0])
        if hit is not None:
            return hit
        with self._lock:
            return self._reduced_cell(key, surface)

    def _reduced_cell(self, key, surface):
        f = self.factor
        plane = surface.pixels
        entry = self._cells.get(key)
        if entry is None or entry[0] != surface.serial or entry[1] != id(plane) \
                or entry[3] is None:
            reduced = reduce_box(plane, f)
            self._cells[key] = [surface.serial, id(plane), reduced, []]
            return reduced
        reduced = entry[2]
        for rect in entry[3]:
            r = rect.clipped_to(surface.width, surface.height)
            if r is None:
                continue
            rk = reduce_rect(r, f)
            full = expand_rect(rk, f, surface.width, surface.height)
            reduced[rk.slice()] = accel.reduce_box(plane[full.slice()], f)
        entry[3] = []
        return reduced

    def _reduced_from(self, kind, ident, source, make):
        """make() reduced once per `source` object -- a cached full-resolution
        render whose identity changes exactly when its content does."""
        with self._lock:
            hit = self._derived.get((kind, ident))
            if hit is not None and hit[0] is source:
                return hit[1]
            out = make()
            self._derived[(kind, ident)] = (source, out)
            return out

    def _whole(self, kind, layer_id, img):
        return self._reduced_from(kind, layer_id, img,
                                  lambda: reduce_box(np.ascontiguousarray(img), self.factor))

    # ---- the compositor's sources, on the level grid ---------------------

    def _source(self, leaf, r):
        layer = leaf.layer
        frame = self._frame_for_group or self.doc.frame
        f = self.factor
        if layer.is_group:
            region = getattr(layer, "region", None)
            if not region:
                return super()._source(leaf, r)
            from . import regions
            out = np.zeros((r.h, r.w, 4), np.uint8)
            if region.get("bg"):
                out[...] = region["bg"]
            self._run(out, self._visible_leaves(layer, frame), r, frame, top=False)
            x, y, w, h = region["rect"]
            scaled = dict(region, rect=[x / float(f), y / float(f), w / float(f), h / float(f)])
            out[..., 3] = np.rint(out[..., 3] * regions.coverage(scaled, r)).astype(np.uint8)
            return out
        if layer.id in self.overrides:
            return self.overrides[layer.id][r.slice()]
        kind = getattr(layer, "kind", "raster")
        if kind == FILL:
            params = layer.params
            if params.get("type") == "pattern":
                params = dict(params, scale=float(params.get("scale", 100)) / f)
            return fill_pixels(params, r, self.width, self.height)
        with self.full._lock:
            if kind == TEXT:
                return self._text(layer)[r.slice()]
            if kind == SMART:
                return self._whole("smart", layer.id, self.full._smart_image(layer))[r.slice()]
            if kind == SHAPE:
                return self._whole("shape", layer.id, self.full._shape_image(layer))[r.slice()]
            if layer.params and _active_filters(layer.params):
                return self._whole("nde", layer.id, self.full._filtered(leaf))[r.slice()]
        return self.reduced_cell((layer.id, frame.id), leaf.surface)[r.slice()]

    def _text(self, layer):
        """A text layer rendered over its own bounds, reduced onto the level."""
        with self._lock:
            return self._text_locked(layer)

    def _text_locked(self, layer):
        from . import textlayer
        key = (repr(sorted(layer.params.items())), self.doc.width, self.doc.height)
        hit = self._derived.get(("text", layer.id))
        if hit is not None and hit[0] == key:
            return hit[1]
        out = np.zeros((self.height, self.width, 4), np.uint8)
        box = textlayer.bounds(layer.params)
        box = None if box is None else box.clipped_to(self.doc.width, self.doc.height)
        if box is not None:
            rk, img = reduce_region(textlayer.pixels(layer.params, box), box, self.factor,
                                    self.doc.width, self.doc.height)
            out[rk.slice()] = img
        self._derived[("text", layer.id)] = (key, out)
        return out

    def _layer_mask(self, layer, r, frame):
        if layer.is_group:
            return None
        out = None
        m = getattr(layer, "mask", None)
        if m is not None and getattr(layer, "mask_enabled", True):
            cell = self.doc.cells.get((m.id, frame.id))
            if m.id in self.overrides or (cell is not None and not cell.is_empty()):
                out = mask_value(self.reduced_cell((m.id, frame.id), cell)[r.slice()])
        vm = getattr(layer, "vector_mask", None)
        if vm is not None and getattr(layer, "vector_mask_enabled", True):
            with self.full._lock:
                full = self.full._vector_coverage(vm)
            cov = self._whole("vmask", id(vm), full)[r.slice()]
            out = cov if out is None else accel.spec._mul(
                out.astype(np.uint32), cov.astype(np.uint32)).astype(np.uint8)
        return out

    def _style_images(self, leaf, frame):
        hit = self.full._style_images(leaf, frame)
        if hit is None:
            return None
        rect, behind, front = hit
        with self._lock:
            cached = self._derived.get(("style", leaf.layer.id))
            if cached is not None and cached[0][0] is behind and cached[0][1] is front:
                return cached[1]
            f, w, h = self.factor, self.doc.width, self.doc.height
            out = (reduce_rect(rect, f),
                   None if behind is None else reduce_region(behind, rect, f, w, h)[1],
                   None if front is None else reduce_region(front, rect, f, w, h)[1])
            self._derived[("style", leaf.layer.id)] = ((behind, front), out)
            return out

    def _effect(self, layer):
        effect = super()._effect(layer)
        if effect is None:
            return None
        token = scaled_token(effect, self.factor)
        return _ScaledEffect(effect, token if token is not None else effect.props.token())

    def _extent(self):
        return self.width, self.height
