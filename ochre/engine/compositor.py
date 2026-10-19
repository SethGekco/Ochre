# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Compositing, and the cache that makes it independent of layer count.

All figures below are measured against this implementation on the development
machine, numpy backend, ten layers at 4000x4000. They are not estimates.

A full-canvas composite takes 18.5 s for a mostly-Normal stack and 30 s for a
blend-heavy one. That is not a number any optimisation rescues -- it is
memory-bandwidth bound across gigabytes -- and it is precisely why nothing in
this design ever composites the full canvas on an input event. Full-canvas
work belongs to export and flatten, which are progress-bar operations. For
display the caller passes the visible viewport or a dirty rect within it, and
never more.

Interactive costs, cached, which is what actually matters:

    dirty rect    painting on TOP    painting at BOTTOM
                  (flattenable)      (blends above, replayed)
      128x128        0.98 ms              3.09 ms
      256x256        2.09 ms             13.51 ms
      512x512        6.97 ms             77.25 ms   <-- over budget

A brush dab produces roughly an 80x80 dirty rect, coalescing to 128-256 per
displayed frame, so ordinary editing sits in the first two rows and is
comfortable either way. The bottom-right cell is the honest weak spot: a large
coalesced region, painted at the bottom of a deep stack, with non-Normal modes
above that must be replayed individually. That combination exceeds a 60 Hz
budget on pure numpy, and it is the specific case the optional C extension
exists to fix. Until then it degrades to a slower repaint rather than to
anything incorrect.

"When legal" is doing real work in that sentence, and getting it wrong is
subtle enough to be worth stating here rather than only at the call site.
Caching the layers BELOW the active one is always exact. Caching those ABOVE
is exact only if every one of them uses Normal, because source-over is
associative while blend modes in general are not: a Multiply layer above the
active layer must multiply against everything beneath it, the active layer
included, and pre-flattening it against transparency computes a different
image. When the upper half cannot be flattened it is replayed per composite
instead -- still correct, and still cheap, because the common case is painting
at or near the top of the stack where there is nothing above to replay.

The caches are invalidated by anything that changes the stack: reorder,
visibility, opacity, blend mode, or a change of active layer. All of those are
non-interactive events where a full recomposite is acceptable.

Group semantics are isolated: a group flattens its children into a scratch
buffer, then blends that result into its parent with the group's own opacity,
blend mode and visibility.
"""

import math

import numpy as np

from . import accel
from .blend import DEFAULT as DEFAULT_BLENDS
from .geometry import Rect
from .layer import ADJUSTMENT, FILL, PASS_THROUGH

GROUP = "group"      # kind of a _Leaf standing for a whole isolated group

_EMPTY = object()      # a clipping base with no pixels: its clipped layers show nothing


class _Leaf:
    """One visible leaf, as the compositor needs it."""

    __slots__ = ("surface", "mode", "opacity", "layer", "base")

    def __init__(self, surface, mode, opacity, layer, base=None):
        self.surface = surface
        self.mode = mode
        self.opacity = opacity
        self.layer = layer
        self.base = base           # the _Leaf this one is clipped to, or None

    # The old (surface, mode, opacity, layer) tuple shape, for callers that
    # unpack leaves.
    def __iter__(self):
        return iter((self.surface, self.mode, self.opacity, self.layer))


class Compositor:
    """Flattens a document's layer tree over a rectangle."""

    def __init__(self, document, blends=None, effects=None):
        self.doc = document
        self.blends = blends or DEFAULT_BLENDS
        self._effects_registry = effects
        self._effects = {}          # (layer id, params) -> Effect, for adjustment layers
        self._styles = {}           # layer id -> (key, rect, behind, front)
        self._below = None          # flattened everything under the active layer
        self._above = None          # flattened upper half, when that is legal
        self._above_layers = None   # else the upper layers, replayed per composite
        self._active_leaf = None
        self._cache_frame = None
        self._cache_active = None
        self._frame_for_group = None

    def invalidate(self):
        """Drop the below/above caches. Cheap; they rebuild on next use."""
        self._below = None
        self._above = None
        self._above_layers = None
        self._active_leaf = None
        self._cache_frame = None
        self._cache_active = None

    @property
    def cache_valid(self):
        return self._below is not None

    @property
    def above_flattened(self):
        """True when the upper half could legally be pre-flattened."""
        return self._above is not None

    def set_active(self, layer_id):
        if layer_id != self._cache_active:
            self.invalidate()
            self._cache_active = layer_id

    # ---- the leaves ------------------------------------------------------------

    def _visible_leaves(self, group, frame, scale=255):
        """Visible leaves bottom-first, with opacity multiplied down through
        groups and each clipped layer pointing at its base.

        A clipping mask (Photoshop's Create Clipping Mask) shows a layer only
        where the nearest unclipped layer beneath it, in the same group, has
        pixels. Approximation, stated: Photoshop composites a clipping group
        and then applies the base's blend mode to the whole; here each clipped
        layer blends with its own mode, clipped to the base's coverage. The
        two agree whenever the base uses Normal, which is nearly always.
        """
        out = []
        base = None
        for node in group.children:
            if not node.visible or node.opacity == 0:
                if not getattr(node, "clip", False):
                    base = _EMPTY
                continue
            opacity = (node.opacity * scale + 127) // 255
            if node.is_group:
                if self._isolated(node):
                    # One leaf for the whole group: its children composite
                    # among themselves, then blend in with the group's mode.
                    leaf = _Leaf(None, self.blends.opcode(node.blend), opacity, node)
                    out.append(leaf)
                    base = leaf
                else:
                    out.extend(self._visible_leaves(node, frame, opacity))
                    base = None
                continue
            kind = getattr(node, "kind", "raster")
            surface = None
            if kind not in (ADJUSTMENT, FILL):
                surface = self.doc.cells.get((node.id, frame.id))
                if surface is None or surface.is_empty():
                    if not node.clip:
                        base = _EMPTY
                    continue
            clipped_to = None
            if node.clip and base is not None:
                if base is _EMPTY:
                    continue
                clipped_to = base
            leaf = _Leaf(surface, self.blends.opcode(node.blend), opacity, node, clipped_to)
            if not node.clip:
                base = leaf
            out.append(leaf)
        return out

    def _isolated(self, group):
        """Does isolating this group change the picture?

        Only a group NOT set to Pass Through is isolated -- and even then,
        when the group blends Normal and every leaf inside is a plain Normal
        layer, isolation and pass-through give the same pixels (source-over
        is associative), so the fast flattened path is used.
        """
        if group.blend == PASS_THROUGH:
            return False
        if group.blend != "Normal":
            return True
        for n in group.walk(include_self=False):
            if n.is_group:
                if n.blend not in ("Normal", PASS_THROUGH):
                    return True
                continue
            if (n.blend != "Normal" or getattr(n, "kind", "raster") == ADJUSTMENT
                    or getattr(n, "clip", False)):
                return True
        return False

    def _source(self, leaf, r):
        """The leaf's own pixels over r: stored, generated (a fill layer), or
        composited (an isolated group)."""
        layer = leaf.layer
        if layer.is_group:
            frame = self._frame_for_group or self.doc.frame
            out = np.zeros((r.h, r.w, 4), np.uint8)
            for child in self._visible_leaves(layer, frame):
                self._blend_leaf(out, child, r, frame)
            return out
        if getattr(layer, "kind", "raster") == FILL:
            return fill_pixels(layer.params, r, self.doc.width, self.doc.height)
        return leaf.surface.view(r, "rgba")

    def _layer_mask(self, layer, r, frame):
        if layer.is_group:
            return None
        m = getattr(layer, "mask", None)
        if m is None or not getattr(layer, "mask_enabled", True):
            return None
        cell = self.doc.cells.get((m.id, frame.id))
        if cell is None or cell.is_empty():
            return None
        px = cell.view(r, "rgba")
        if px is None:
            return None
        return mask_value(px)

    def _coverage(self, leaf, r, frame):
        """Mask for a leaf over r: its layer mask, times its clipping base."""
        mask = self._layer_mask(leaf.layer, r, frame)
        base = leaf.base
        if base is not None:
            src = self._source(base, r)
            a = src[..., 3].astype(np.uint32) if src is not None else \
                np.zeros((r.h, r.w), np.uint32)
            if base.opacity != 255:
                a = accel.spec._mul(a, np.uint32(base.opacity))
            bm = self._layer_mask(base.layer, r, frame)
            if bm is not None:
                a = accel.spec._mul(a, bm.astype(np.uint32))
            a = a.astype(np.uint8)
            mask = a if mask is None else accel.spec._mul(
                mask.astype(np.uint32), a.astype(np.uint32)).astype(np.uint8)
        return mask

    def _effect(self, layer):
        from .effects import EffectRegistry
        if self._effects_registry is None:
            self._effects_registry = EffectRegistry()
        name = layer.params.get("effect")
        settings = {k: v for k, v in layer.params.items() if k != "effect"}
        key = (layer.id, name, repr(sorted(settings.items())))
        effect = self._effects.get(key)
        if effect is None:
            effect = self._effects_registry.create(name, **settings)
            self._effects = {k: v for k, v in self._effects.items() if k[0] != layer.id}
            self._effects[key] = effect
        return effect

    def _style_images(self, leaf, frame):
        """(rect, behind, front) for a styled layer, cached until it changes."""
        from . import layerstyles
        layer = leaf.layer
        styles = getattr(layer, "styles", None)
        if not styles or not layerstyles.enabled(styles):
            return None
        surface = leaf.surface
        revision = None if surface is None else (id(surface), surface.revision,
                                                 surface.content_bbox)
        key = (frame.id, revision, repr(sorted((k, sorted(v.items())) for k, v in styles.items())),
               repr(sorted(layer.params.items())))
        hit = self._styles.get(layer.id)
        if hit is not None and hit[0] == key:
            return hit[1:]
        if surface is not None:
            box = surface.content_bbox
        else:
            box = self.doc.bounds
        rect = box.inflated(layerstyles.reach(styles)).clipped_to(self.doc.width, self.doc.height)
        if rect is None:
            return None
        src = self._source(leaf, rect)
        behind, front = layerstyles.render(np.ascontiguousarray(src), styles)
        out = (rect, layerstyles.to_uint8(behind), layerstyles.to_uint8(front))
        self._styles[layer.id] = (key,) + out
        return out

    def _blend_leaf(self, window, leaf, r, frame):
        """Composite one leaf over window (which covers document rect r)."""
        self._frame_for_group = frame
        layer = leaf.layer
        mask = self._coverage(leaf, r, frame)
        styled = None if layer.is_group else (
            self._style_images(leaf, frame) if leaf.surface is not None or
            getattr(layer, "kind", "raster") == FILL else None)
        sub = None
        if styled is not None:
            srect = styled[0].intersect(r)
            if srect is not None:
                oy, ox = srect.y - r.y, srect.x - r.x
                sy, sx = srect.y - styled[0].y, srect.x - styled[0].x
                sub = (window[oy:oy + srect.h, ox:ox + srect.w],
                       (slice(sy, sy + srect.h), slice(sx, sx + srect.w)),
                       None if mask is None else mask[oy:oy + srect.h, ox:ox + srect.w])
            if sub is not None and styled[1] is not None:
                accel.blend_rect(sub[0], styled[1][sub[1]], accel.NORMAL, leaf.opacity,
                                 sub[2], out=sub[0])

        if getattr(layer, "kind", "raster") == ADJUSTMENT:
            effect = self._effect(layer)
            if effect is not None:
                adjusted = effect.apply(window)
                accel.blend_rect(window, adjusted, leaf.mode, leaf.opacity, mask, out=window)
        else:
            src = self._source(leaf, r)
            if src is not None:
                accel.blend_rect(window, src, leaf.mode, leaf.opacity, mask, out=window)

        if sub is not None and styled[2] is not None:
            accel.blend_rect(sub[0], styled[2][sub[1]], accel.NORMAL, leaf.opacity,
                             sub[2], out=sub[0])

    # ---- full composites --------------------------------------------------------

    def composite_into(self, dst, rect, frame=None):
        """Flatten the document over rect into dst (an (H,W,4) uint8 array).

        dst is indexed in document coordinates, so dst must be canvas-sized.
        """
        frame = self.doc.frame if frame is None else frame
        r = rect.clipped_to(self.doc.width, self.doc.height)
        if r is None:
            return None
        window = dst[r.slice()]
        window[...] = 0
        for leaf in self._visible_leaves(self.doc.root, frame):
            self._blend_leaf(window, leaf, r, frame)
        return r

    def composite(self, rect=None, frame=None):
        """Flatten over rect and return a fresh array of just that region."""
        frame = self.doc.frame if frame is None else frame
        r = (self.doc.bounds if rect is None else rect).clipped_to(
            self.doc.width, self.doc.height)
        if r is None:
            return None
        out = np.zeros((r.h, r.w, 4), dtype=np.uint8)
        for leaf in self._visible_leaves(self.doc.root, frame):
            self._blend_leaf(out, leaf, r, frame)
        return out

    # ---- the stroke cache -------------------------------------------------------

    def build_cache(self, active_layer, frame=None):
        """Flatten what can legitimately be flattened, once, at stroke start.

        `below` is ALWAYS safe: layers beneath the active one composite among
        themselves, whatever their modes.

        `above` is safe only when every upper layer is Normal, is not an
        adjustment (which acts on everything beneath it, the active layer
        included), and is not clipped to the active layer (whose shape is
        what is being edited). Otherwise the upper layers are replayed per
        composite -- still correct, proportionally slower.
        """
        frame = self.doc.frame if frame is None else frame
        layers = self._visible_leaves(self.doc.root, frame)
        active_id = getattr(active_layer, "id", active_layer)
        if isinstance(active_id, str) and active_id.endswith("~mask"):
            active_id = active_id[:-len("~mask")]      # editing a mask: its owner

        split = None
        for i, leaf in enumerate(layers):
            if leaf.layer is not None and leaf.layer.id == active_id:
                split = i
                break

        if split is None and self.doc.layer(active_id) is not None \
                and not self.doc.layer(active_id).is_group:
            # The active layer is inside an isolated group, which composites
            # as one: nothing can be cached around it. Every composite takes
            # the full walk -- correct, just slower for that case.
            self.invalidate()
            return self
        lower = layers if split is None else layers[:split]
        upper = [] if split is None else layers[split + 1:]
        active = None if split is None else layers[split]

        bounds = self.doc.bounds
        shape = (self.doc.height, self.doc.width, 4)
        below = np.zeros(shape, dtype=np.uint8)
        for leaf in lower:
            self._blend_leaf(below, leaf, bounds, frame)

        flattenable = all(leaf.mode == accel.NORMAL
                          and getattr(leaf.layer, "kind", "raster") != ADJUSTMENT
                          and (leaf.base is None or leaf.base is not active)
                          for leaf in upper)
        if flattenable:
            above = np.zeros(shape, dtype=np.uint8)
            for leaf in upper:
                self._blend_leaf(above, leaf, bounds, frame)
            self._above = above
            self._above_layers = None
        else:
            self._above = None
            self._above_layers = upper

        self._below = below
        self._active_leaf = active
        self._cache_frame = frame.id
        self._cache_active = active_id
        return self

    def composite_cached(self, dst, rect, active_layer, frame=None):
        """Three blends: below, the active layer, above.

        Falls back to a full walk when the cache is cold or stale, so a caller
        can always use this path without checking first.
        """
        frame = self.doc.frame if frame is None else frame
        active_id = getattr(active_layer, "id", active_layer)
        if isinstance(active_id, str) and active_id.endswith("~mask"):
            active_id = active_id[:-len("~mask")]
        if (not self.cache_valid or self._cache_frame != frame.id
                or self._cache_active != active_id):
            return self.composite_into(dst, rect, frame)

        r = rect.clipped_to(self.doc.width, self.doc.height)
        if r is None:
            return None

        window = dst[r.slice()]
        window[...] = self._below[r.slice()]

        leaf = self._active_leaf
        if leaf is not None:
            node = self.doc.layer(active_id)
            if node is not None and node.effectively_visible() and node.opacity > 0:
                # The surface may have been created since the cache was built.
                if leaf.surface is None and getattr(node, "kind", "raster") not in (ADJUSTMENT, FILL):
                    leaf.surface = self.doc.cells.get((active_id, frame.id))
                if leaf.surface is not None or getattr(node, "kind", "raster") in (ADJUSTMENT, FILL):
                    self._blend_leaf(window, leaf, r, frame)

        if self._above is not None:
            accel.blend_rect(window, self._above[r.slice()], accel.NORMAL, 255,
                             None, out=window)
        else:
            for upper in (self._above_layers or ()):
                self._blend_leaf(window, upper, r, frame)
        return r


# ---- helpers shared with merge / flatten ----------------------------------------

def mask_value(px):
    """A mask cell's coverage: its luminance, over white where transparent.

    Luminance rather than one channel, so a mask painted in colour still
    behaves sensibly; transparent counts as white, so an erased patch of a
    mask reveals rather than leaving a hole.
    """
    p = px.astype(np.uint32)
    lum = (77 * p[..., 0] + 151 * p[..., 1] + 28 * p[..., 2] + 128) >> 8
    a = p[..., 3]
    return (accel.spec._mul(lum, a) + (255 - a)).astype(np.uint8)


def _colour(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        parts = [int(v) for v in value.split(",") if v.strip()]
    else:
        parts = [int(v) for v in value]
    if len(parts) == 3:
        parts.append(255)
    return tuple(parts[:4]) if len(parts) >= 4 else default


def fill_pixels(params, r, width, height):
    """A fill layer's content over r: a solid colour or a linear gradient."""
    kind = params.get("type", "solid")
    out = np.empty((r.h, r.w, 4), dtype=np.uint8)
    if kind == "gradient":
        a = np.array(_colour(params.get("start"), (0, 0, 0, 255)), np.float64)
        b = np.array(_colour(params.get("end"), (255, 255, 255, 255)), np.float64)
        t = math.radians(float(params.get("angle", 90)))
        ux, uy = math.cos(t), -math.sin(t)
        yy, xx = np.mgrid[r.y:r.y1, r.x:r.x1].astype(np.float64)
        proj = (xx + 0.5 - width / 2.0) * ux + (yy + 0.5 - height / 2.0) * uy
        span = max(1.0, abs(width * ux) + abs(height * uy))
        k = np.clip(proj / span + 0.5, 0, 1)[..., None]
        # Alpha-weighted, like every colour average here.
        alpha = a[3] + (b[3] - a[3]) * k[..., 0]
        rgb = (a[:3] * a[3] * (1 - k) + b[:3] * b[3] * k) / np.maximum(alpha, 1e-6)[..., None]
        out[..., :3] = np.clip(np.rint(rgb), 0, 255)
        out[..., 3] = np.clip(np.rint(alpha), 0, 255)
    else:
        out[...] = _colour(params.get("color"), (0, 0, 0, 255))
    return out
