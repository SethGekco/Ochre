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

import numpy as np

from . import accel
from .blend import DEFAULT as DEFAULT_BLENDS
from .geometry import Rect


class Compositor:
    """Flattens a document's layer tree over a rectangle."""

    def __init__(self, document, blends=None):
        self.doc = document
        self.blends = blends or DEFAULT_BLENDS
        self._below = None          # flattened everything under the active layer
        self._above = None          # flattened upper half, when that is legal
        self._above_layers = None   # else the upper layers, replayed per composite
        self._cache_frame = None
        self._cache_active = None

    # ---- cache lifecycle -------------------------------------------------

    def invalidate(self):
        """Drop the below/above caches. Cheap; they rebuild on next use."""
        self._below = None
        self._above = None
        self._above_layers = None
        self._cache_frame = None
        self._cache_active = None

    @property
    def cache_valid(self):
        return self._below is not None

    @property
    def above_flattened(self):
        """True when the upper half could legally be pre-flattened.

        False means every composite replays the upper layers individually --
        still correct, just proportional to how many sit above the active
        layer with a non-Normal blend mode.
        """
        return self._above is not None

    def set_active(self, layer_id):
        if layer_id != self._cache_active:
            self.invalidate()
            self._cache_active = layer_id

    # ---- the composite ---------------------------------------------------

    def _visible_leaves(self, group, frame):
        """(surface, mode, opacity, layer) bottom-first, skipping the invisible.

        Opacity multiplies down through groups, which is what makes an
        isolated group's opacity behave the way a user expects without the
        compositor needing a separate group pass for the common case of a
        group whose blend mode is Normal.
        """
        out = []
        for node in group.children:
            if not node.visible or node.opacity == 0:
                continue
            if node.is_group:
                for surface, mode, opacity, layer in self._visible_leaves(node, frame):
                    scaled = (opacity * node.opacity + 127) // 255
                    out.append((surface, mode, scaled, layer))
            else:
                surface = self.doc.cells.get((node.id, frame.id))
                if surface is None or surface.is_empty():
                    continue
                out.append((surface, self.blends.opcode(node.blend),
                            node.opacity, node))
        return out

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

        for surface, mode, opacity, _layer in self._visible_leaves(self.doc.root, frame):
            src = surface.view(r, "rgba")
            if src is None:
                continue
            accel.blend_rect(window, src, mode, opacity, None, out=window)
        return r

    def composite(self, rect=None, frame=None):
        """Flatten over rect and return a fresh array of just that region."""
        frame = self.doc.frame if frame is None else frame
        r = (self.doc.bounds if rect is None else rect).clipped_to(
            self.doc.width, self.doc.height)
        if r is None:
            return None
        out = np.zeros((r.h, r.w, 4), dtype=np.uint8)
        for surface, mode, opacity, _layer in self._visible_leaves(self.doc.root, frame):
            src = surface.view(r, "rgba")
            if src is None:
                continue
            accel.blend_rect(out, src, mode, opacity, None, out=out)
        return out

    # ---- the below/above cache ------------------------------------------

    def build_cache(self, active_layer, frame=None):
        """Flatten what can legitimately be flattened, once, at stroke start.

        A correctness constraint bounds how much of this is cacheable, and it
        is easy to get wrong:

        `below` is ALWAYS safe. Layers beneath the active one composite among
        themselves and their result does not depend on the active layer, so
        pre-flattening them is exact for any mix of blend modes.

        `above` is NOT generally safe. A Multiply layer sitting above the
        active layer must multiply against everything beneath it -- which
        includes the active layer. Pre-flattening it against transparency and
        then source-over-ing the result onto the finished stack computes a
        different picture. Flattening the upper layers is valid exactly when
        every one of them uses Normal, because source-over is associative:
        (a over b) over c == a over (b over c). Opacity is fine either way,
        being only a scale on alpha.

        So the upper half is flattened when it legally can be, and kept as a
        list to replay per composite when it cannot. Painting on or near the
        top layer -- overwhelmingly the common case -- gets the fast path
        regardless, because there is nothing above to replay.
        """
        frame = self.doc.frame if frame is None else frame
        layers = self._visible_leaves(self.doc.root, frame)
        active_id = getattr(active_layer, "id", active_layer)

        split = None
        for i, (_s, _m, _o, layer) in enumerate(layers):
            if layer is not None and layer.id == active_id:
                split = i
                break

        lower = layers if split is None else layers[:split]
        upper = [] if split is None else layers[split + 1:]

        shape = (self.doc.height, self.doc.width, 4)
        below = np.zeros(shape, dtype=np.uint8)
        for surface, mode, opacity, _l in lower:
            src = surface.view(self.doc.bounds, "rgba")
            if src is not None:
                accel.blend_rect(below, src, mode, opacity, None, out=below)

        flattenable = all(mode == accel.NORMAL for _s, mode, _o, _l in upper)
        if flattenable:
            above = np.zeros(shape, dtype=np.uint8)
            for surface, mode, opacity, _l in upper:
                src = surface.view(self.doc.bounds, "rgba")
                if src is not None:
                    accel.blend_rect(above, src, mode, opacity, None, out=above)
            self._above = above
            self._above_layers = None
        else:
            self._above = None
            self._above_layers = upper

        self._below = below
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
        if (not self.cache_valid or self._cache_frame != frame.id
                or self._cache_active != active_id):
            return self.composite_into(dst, rect, frame)

        r = rect.clipped_to(self.doc.width, self.doc.height)
        if r is None:
            return None

        node = self.doc.layer(active_id)
        window = dst[r.slice()]
        window[...] = self._below[r.slice()]

        if node is not None and node.effectively_visible() and node.opacity > 0:
            surface = self.doc.cells.get((active_id, frame.id))
            if surface is not None and not surface.is_empty():
                src = surface.view(r, "rgba")
                if src is not None:
                    accel.blend_rect(window, src, self.blends.opcode(node.blend),
                                     node.opacity, None, out=window)

        if self._above is not None:
            accel.blend_rect(window, self._above[r.slice()], accel.NORMAL, 255,
                             None, out=window)
        else:
            for surface, mode, opacity, _l in (self._above_layers or ()):
                src = surface.view(r, "rgba")
                if src is not None:
                    accel.blend_rect(window, src, mode, opacity, None, out=window)
        return r
