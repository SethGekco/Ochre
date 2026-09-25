# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Frames, and the residency policy that keeps them affordable.

The frame axis is generic, not a concession to any one format. A multi-frame
sprite, an animated GIF or WebP, a multi-page document and an onion-skinned
animation are all the same structure: N frames sharing one canvas size, one
palette and one selection. It is 1-D with a layout hint; a 2-D axis would
permanently complicate cell keys, undo, selection, playback and the file
format to serve a single format family, so a tile grid is expressed as
metadata plus one panel behaviour instead.

Memory is the one place where the naive approach fails outright. 100 frames
of 4000x4000 with a single RGBA layer each is 6.4 GB resident -- impossible,
and still 1.6 GB even index-locked. (The realistic sprite case, 100 frames of
200x200, is about 20 MB and needs none of this.) So frames have three
residency states under an LRU:

    warm      live numpy arrays
    cold      zlib blobs, one per authoritative plane
    unloaded  still in the file, never read

Documents open with everything unloaded but the first frame. The 100-frame
stress case becomes 8 warm plus 92 cold, which is resident and workable.

A caution on the numbers: cold-frame compression was measured at roughly
22:1 on synthetic gradients, which is optimistic. Real photographic content
lands nearer 2-3:1. MaxWarmBytes is the real safety bound; treat any ratio as
a best case.
"""

import itertools
import zlib

import numpy as np

from .geometry import DirtyRegion

WARM = "warm"
COLD = "cold"
UNLOADED = "unloaded"

_ids = itertools.count(1)


class Frame:
    """One position on the document's frame axis."""

    def __init__(self, name="", duration_ms=100, frame_id=None, max_dirty_rects=8):
        self.id = frame_id or "F%04d" % next(_ids)
        self.name = name
        # Generic: a GIF/WebP/APNG inter-frame delay, or sprite timing.
        self.duration_ms = int(duration_ms)
        self.residency = WARM
        self.dirty = DirtyRegion(max_dirty_rects)
        self.meta = {}              # str -> str, round-tripped by the format
        self._frozen = None         # {(layer_id, plane): (bytes, shape)}

    @property
    def is_warm(self):
        return self.residency == WARM

    def __repr__(self):
        return "Frame(id=%r, name=%r, residency=%r)" % (self.id, self.name,
                                                        self.residency)

    # ---- residency -------------------------------------------------------

    def freeze(self, cells, level=1):
        """Compress this frame's authoritative planes and release the arrays.

        Only authoritative planes are stored. Derived caches rebuild from
        their source on thaw, so compressing them would be paying twice for
        the same information.
        """
        if self.residency != WARM:
            return 0
        blobs = {}
        freed = 0
        for (layer_id, frame_id), surface in cells.items():
            if frame_id != self.id:
                continue
            for name in surface.authoritative:
                arr = surface.planes.get(name)
                if arr is None:
                    continue
                blobs[(layer_id, name)] = (zlib.compress(arr.tobytes(), level),
                                           arr.shape)
                freed += arr.nbytes
            surface.planes = {k: None for k in surface.planes}
        self._frozen = blobs
        self.residency = COLD
        return freed

    def thaw(self, cells):
        """Restore compressed planes.

        Derived caches are deliberately NOT rebuilt here. Surface.plane()
        builds them coherently on first access, so rebuilding eagerly only
        matters if something is about to read them -- and the commonest
        reason to thaw a pile of frames is saving, which reads authoritative
        planes and never touches the cache. On a real 136-frame sprite that
        eager rebuild was 3.1 GB of the 4.8 GB a save cost, to populate
        caches nothing would read before they were freed again.
        """
        if self.residency != COLD or self._frozen is None:
            return 0
        restored = 0
        for (layer_id, frame_id), surface in cells.items():
            if frame_id != self.id:
                continue
            for name in surface.authoritative:
                blob = self._frozen.get((layer_id, name))
                if blob is None:
                    continue
                data, shape = blob
                arr = np.frombuffer(zlib.decompress(data), dtype=np.uint8)
                surface.planes[name] = arr.reshape(shape).copy()
                restored += arr.nbytes
        self._frozen = None
        self.residency = WARM
        return restored

    def cold_bytes(self):
        if not self._frozen:
            return 0
        return sum(len(data) for data, _ in self._frozen.values())
