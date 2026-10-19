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
from .layer import ADJUSTMENT, FILL, PASS_THROUGH, SMART, TEXT

_GENERATED = (FILL, TEXT, SMART)

GROUP = "group"      # kind of a _Leaf standing for a whole isolated group

_EMPTY = object()      # a clipping base with no pixels: its clipped layers show nothing


class _Leaf:
    """One visible leaf, as the compositor needs it."""

    __slots__ = ("surface", "mode", "opacity", "layer", "base", "group_start")

    def __init__(self, surface, mode, opacity, layer, base=None):
        self.surface = surface
        self.mode = mode
        self.opacity = opacity
        self.layer = layer
        self.base = base           # the _Leaf this one is clipped to, or None
        # Index (in the leaf list) where this leaf's pass-through group starts;
        # None at the top level. A shallow knockout reveals that point.
        self.group_start = None

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

    def _visible_leaves(self, group, frame, scale=255, _offset=0, _nested=False):
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
                    out.extend(self._visible_leaves(node, frame, opacity,
                                                    _offset + len(out), True))
                    base = None
                continue
            kind = getattr(node, "kind", "raster")
            surface = None
            if kind not in (ADJUSTMENT,) + _GENERATED:
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
            if _nested:
                leaf.group_start = _offset
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
        if getattr(group, "region", None):
            return True              # an artboard or frame: cut to its region
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
                    or getattr(n, "clip", False) or getattr(n, "blend_if", None)
                    or _knocks(n)):
                return True
        return False

    def _run(self, window, leaves, r, frame, top=True):
        """Composite leaves bottom-up over window -- the one loop every
        composite goes through, so knockouts behave the same everywhere.

        A knockout needs the picture as it was further down the stack: the
        start of the layer's group (shallow) or just above the bottom layer
        (deep, Photoshop's 'to the Background'). Those states are copied
        only when some leaf asks for them. Inside an isolated group (top
        False) both kinds reach the group's own transparent start -- an
        approximation: Photoshop's deep knockout there reaches the image
        background, through the group.
        """
        need = {}
        for i, leaf in enumerate(leaves):
            ko = _knocks(leaf.layer)
            if not ko:
                continue
            if ko == "shallow" and leaf.group_start is not None:
                start = leaf.group_start
            else:
                start = 1 if top else 0
            if start < i:
                need[i] = start
        wanted = set(need.values())
        snaps = {}
        for i, leaf in enumerate(leaves):
            if i in wanted:
                snaps[i] = window.copy()
            if i in need:
                self._knock(window, snaps[need[i]], leaf, r, frame)
            self._blend_leaf(window, leaf, r, frame)
        return window

    def _knock(self, window, backdrop, leaf, r, frame):
        """Where the leaf has pixels, put back the backdrop: window becomes
        lerp(window, backdrop, footprint) in premultiplied space. The leaf
        itself is blended afterwards at its Fill opacity."""
        if getattr(leaf.layer, "kind", "raster") == ADJUSTMENT:
            return
        src = self._source(leaf, r)
        if src is None:
            return
        k = src[..., 3].astype(np.float64) * (leaf.opacity / (255.0 * 255.0))
        mask = self._coverage(leaf, r, frame)
        if mask is not None:
            k = k * (mask / 255.0)
        k = k[..., None]
        w = window.astype(np.float64)
        b = backdrop.astype(np.float64)
        wa, ba = w[..., 3:4], b[..., 3:4]
        a = wa * (1 - k) + ba * k
        rgb = w[..., :3] * wa * (1 - k) + b[..., :3] * ba * k
        rgb = np.divide(rgb, a, out=np.zeros_like(rgb), where=a > 0)
        window[..., :3] = np.clip(np.rint(rgb), 0, 255)
        window[..., 3:4] = np.clip(np.rint(a), 0, 255)

    def _source(self, leaf, r):
        """The leaf's own pixels over r: stored, generated (a fill layer), or
        composited (an isolated group)."""
        layer = leaf.layer
        if layer.is_group:
            frame = self._frame_for_group or self.doc.frame
            out = np.zeros((r.h, r.w, 4), np.uint8)
            region = getattr(layer, "region", None)
            if region and region.get("bg"):
                out[...] = region["bg"]            # an artboard's background
            self._run(out, self._visible_leaves(layer, frame), r, frame, top=False)
            if region:
                from . import regions
                cov = regions.coverage(region, r)
                out[..., 3] = np.rint(out[..., 3] * cov).astype(np.uint8)
            return out
        kind = getattr(layer, "kind", "raster")
        if kind == FILL:
            return fill_pixels(layer.params, r, self.doc.width, self.doc.height)
        if kind == TEXT:
            from . import textlayer
            return textlayer.pixels(layer.params, r)
        if kind == SMART:
            return self._smart_image(layer)[r.slice()]
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

    def _smart_image(self, layer):
        """A smart object rendered from its source, cached until its
        transform, filters or source pixels change."""
        from . import smartobject
        frame = self._frame_for_group or self.doc.frame
        src = getattr(layer, "smart_source", None)
        cell = None if src is None else self.doc.cells.get((src.id, frame.id))
        rev = None if cell is None else (id(cell), cell.revision)
        key = (repr(layer.params.get("matrix")), repr(layer.params.get("filters")),
               layer.params.get("src_w"), layer.params.get("src_h"), rev,
               self.doc.width, self.doc.height, frame.id)
        cache = getattr(self, "_smart", None)
        if cache is None:
            cache = self._smart = {}
        hit = cache.get(layer.id)
        if hit is not None and hit[0] == key:
            return hit[1]
        img = smartobject.render(self.doc, layer, frame, self._effects_registry)
        cache[layer.id] = (key, img)
        return img

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
        revision = None if surface is None or getattr(layer, "kind", "raster") != "raster" \
            else (id(surface), surface.revision, surface.content_bbox)
        key = (frame.id, revision, repr(sorted((k, sorted(v.items())) for k, v in styles.items())),
               repr(sorted(layer.params.items())))
        hit = self._styles.get(layer.id)
        if hit is not None and hit[0] == key:
            return hit[1:]
        if getattr(layer, "kind", "raster") == TEXT:
            from . import textlayer
            box = textlayer.bounds(layer.params) or Rect(0, 0, 1, 1)
        elif surface is not None:
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
            getattr(layer, "kind", "raster") in _GENERATED else None)
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

        blend_if = getattr(layer, "blend_if", None)
        fill = getattr(layer, "fill_opacity", 255)
        opacity = leaf.opacity if fill >= 255 else (leaf.opacity * fill + 127) // 255
        if getattr(layer, "kind", "raster") == ADJUSTMENT:
            effect = self._effect(layer)
            if effect is not None:
                adjusted = effect.apply(window)
                if blend_if:
                    mask = _combine(mask, blend_if_factor(adjusted, window, blend_if))
                accel.blend_rect(window, adjusted, leaf.mode, opacity, mask, out=window)
        else:
            src = self._source(leaf, r)
            if src is not None:
                if blend_if:
                    mask = _combine(mask, blend_if_factor(src, window, blend_if))
                accel.blend_rect(window, src, leaf.mode, opacity, mask, out=window)

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
        self._run(window, self._visible_leaves(self.doc.root, frame), r, frame)
        return r

    def composite(self, rect=None, frame=None):
        """Flatten over rect and return a fresh array of just that region."""
        frame = self.doc.frame if frame is None else frame
        r = (self.doc.bounds if rect is None else rect).clipped_to(
            self.doc.width, self.doc.height)
        if r is None:
            return None
        out = np.zeros((r.h, r.w, 4), dtype=np.uint8)
        return self._run(out, self._visible_leaves(self.doc.root, frame), r, frame)

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

        if any(_knocks(leaf.layer) for leaf in layers):
            # A knockout reads the picture further down the stack, which the
            # three-part cache does not keep: take the full walk.
            self.invalidate()
            return self
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
                          and not getattr(leaf.layer, "blend_if", None)
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
                if leaf.surface is None and getattr(node, "kind", "raster") not in (ADJUSTMENT,) + _GENERATED:
                    leaf.surface = self.doc.cells.get((active_id, frame.id))
                if leaf.surface is not None or getattr(node, "kind", "raster") in (ADJUSTMENT,) + _GENERATED:
                    self._blend_leaf(window, leaf, r, frame)

        if self._above is not None:
            accel.blend_rect(window, self._above[r.slice()], accel.NORMAL, 255,
                             None, out=window)
        else:
            for upper in (self._above_layers or ()):
                self._blend_leaf(window, upper, r, frame)
        return r


# ---- helpers shared with merge / flatten ----------------------------------------

def _knocks(node):
    """The node's knockout kind, or None."""
    if node is None or node.is_group:
        return None
    ko = getattr(node, "knockout", "none")
    return ko if ko in ("shallow", "deep") else None


def _combine(mask, factor):
    if mask is None:
        return factor
    return accel.spec._mul(mask.astype(np.uint32), factor.astype(np.uint32)).astype(np.uint8)


def _ramp(lum, lo0, lo1, hi0, hi1):
    """1 inside [lo1, hi0], ramping to 0 at lo0 and hi1 (split sliders)."""
    lum = lum.astype(np.float64)
    if lo1 > lo0:
        up = np.clip((lum - lo0) / float(lo1 - lo0), 0, 1)
    else:
        up = (lum >= lo0).astype(np.float64)
    if hi1 > hi0:
        down = np.clip((hi1 - lum) / float(hi1 - hi0), 0, 1)
    else:
        down = (lum <= hi1).astype(np.float64)
    return up * down


def blend_if_factor(src, under, ranges):
    """Photoshop's Blend If: show this layer only where ITS lightness, and
    the lightness of what is under it, fall inside the given ranges -- each
    (black lo, black hi, white lo, white hi), split pairs fading smoothly.
    The classic use: drop out a layer's white background, or let clouds
    show only over the sky."""
    def lum(x):
        p = x.astype(np.uint32)
        return (77 * p[..., 0] + 151 * p[..., 1] + 28 * p[..., 2] + 128) >> 8
    this = _ramp(lum(src), *ranges[:4])
    below = _ramp(lum(under), *ranges[4:8])
    return np.clip(np.rint(this * below * 255), 0, 255).astype(np.uint8)

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
