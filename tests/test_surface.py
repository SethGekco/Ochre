#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Surface: lazy allocation, plane coherence, and the index-locked contract.

Run: python3 tests/test_surface.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.geometry import Rect
from ochre.engine.palette import Palette, grayscale
from ochre.engine.surface import Surface


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- lazy allocation -------------------------------------------------
    s = Surface(64, 48)
    check("nothing allocated up front", not s.is_allocated("rgba"))
    check("fresh surface costs no pixel memory", s.nbytes() == 0)
    check("fresh surface is empty", s.is_empty())
    check("bounds", s.bounds == Rect(0, 0, 64, 48))

    px = s.pixels
    check("rgba allocated on access", s.is_allocated("rgba"))
    check("rgba shape", px.shape == (48, 64, 4))
    check("rgba dtype", px.dtype == np.uint8)
    check("rgba starts transparent", int(px.sum()) == 0)
    check("nbytes reflects allocation", s.nbytes() == 48 * 64 * 4)

    # Allocation alone is not a write.
    check("allocation does not mark content", s.is_empty())

    # ---- content_bbox and dirty ------------------------------------------
    s = Surface(64, 48)
    s.fill(Rect(10, 10, 4, 4), (255, 0, 0, 255))
    check("fill sets content_bbox", s.content_bbox == Rect(10, 10, 4, 4))
    check("fill marks dirty", len(s.dirty) == 1)
    s.fill(Rect(30, 30, 2, 2), (0, 255, 0, 255))
    check("content_bbox grows to union", s.content_bbox == Rect(10, 10, 22, 22))
    check("content_bbox never shrinks on new write",
          s.content_bbox.contains_rect(Rect(10, 10, 4, 4)))

    # Writes outside the canvas clip rather than raising.
    s = Surface(16, 16)
    r = s.fill(Rect(-8, -8, 32, 32), (1, 2, 3, 255))
    check("out-of-bounds fill clips", r == Rect(0, 0, 16, 16))
    check("fully-outside fill is a no-op",
          s.fill(Rect(100, 100, 4, 4), (9, 9, 9, 255)) is None)

    # ---- views are zero-copy --------------------------------------------
    s = Surface(32, 32)
    s.pixels[:] = 0
    v = s.view(Rect(4, 4, 8, 8))
    v[0, 0] = (7, 7, 7, 255)
    check("view aliases the plane", tuple(s.pixels[4, 4]) == (7, 7, 7, 255))
    check("view is a numpy view", v.base is not None)
    check("view of off-canvas rect is None", s.view(Rect(99, 99, 4, 4)) is None)

    # read() copies
    s.fill(Rect(0, 0, 4, 4), (5, 5, 5, 255))
    got = s.read(Rect(0, 0, 4, 4), ("rgba",))
    got["rgba"][0, 0] = (0, 0, 0, 0)
    check("read returns copies", tuple(s.pixels[0, 0]) == (5, 5, 5, 255))

    # ---- write shape discipline -----------------------------------------
    s = Surface(16, 16)
    try:
        s.write(Rect(0, 0, 4, 4), np.zeros((2, 2, 4), np.uint8))
        check("write rejects wrong shape", False, "-- no error raised")
    except ValueError:
        check("write rejects wrong shape", True)

    # ---- apply_masked ----------------------------------------------------
    s = Surface(8, 8)
    s.fill(s.bounds, (255, 0, 0, 255))
    src = np.zeros((4, 4, 4), np.uint8)
    src[..., 2] = 255
    src[..., 3] = 255
    s.apply_masked(Rect(0, 0, 4, 4), src, mask=np.zeros((4, 4), np.uint8))
    check("zero mask writes nothing", tuple(s.pixels[0, 0]) == (255, 0, 0, 255))
    s.apply_masked(Rect(0, 0, 4, 4), src, mask=np.full((4, 4), 255, np.uint8))
    check("full mask writes the source", tuple(s.pixels[0, 0]) == (0, 0, 255, 255))
    check("outside the rect is untouched", tuple(s.pixels[6, 6]) == (255, 0, 0, 255))

    try:
        s.apply_masked(Rect(0, 0, 4, 4), src, mask=np.zeros((2, 2), np.uint8))
        check("apply_masked rejects bad mask shape", False, "-- no error")
    except ValueError:
        check("apply_masked rejects bad mask shape", True)

    # ---- scalar (non-colour) planes -------------------------------------
    s = Surface(16, 16, planes=("rgba", "height"), authoritative=("rgba", "height"))
    h = s.plane("height")
    check("scalar plane is 2-D", h.shape == (16, 16))
    check("scalar plane is uint8", h.dtype == np.uint8)

    before = s.pixels.copy()
    s.fill(Rect(2, 2, 3, 3), 200, name="height")
    check("height write lands", int(s.plane("height")[2, 2]) == 200)
    check("height write leaves colour alone", np.array_equal(s.pixels, before))
    check("height write shares one content_bbox", s.content_bbox == Rect(2, 2, 3, 3))
    check("height write shares one dirty region", len(s.dirty) == 1)

    # An unknown plane is an error, not a silently orphaned array.
    try:
        s.plane("nonsense")
        check("unknown plane rejected", False, "-- no error")
    except KeyError:
        check("unknown plane rejected", True)

    # Authoritative planes must exist.
    try:
        Surface(8, 8, planes=("rgba",), authoritative=("index",))
        check("authoritative must be present", False, "-- no error")
    except ValueError:
        check("authoritative must be present", True)

    # ---- index-locked ----------------------------------------------------
    pal = grayscale()
    pal.set_entry(1, (10, 20, 30, 255))
    pal.set_entry(17, (200, 100, 50, 255))
    s = Surface(16, 16, planes=("index", "rgba"), authoritative=("index",),
                palette=pal)
    check("index is authoritative", s.index_locked)
    check("rgba is derived", s.is_derived("rgba"))
    check("index is not derived", not s.is_derived("index"))

    s.fill(Rect(0, 0, 4, 4), 17, name="index")
    check("index write lands", int(s.plane("index")[0, 0]) == 17)
    check("rgba cache refreshed automatically",
          tuple(s.pixels[0, 0]) == (200, 100, 50, 255),
          "-- got %r" % (tuple(s.pixels[0, 0]),))
    check("refresh is scoped to the write", tuple(s.pixels[8, 8]) == (0, 0, 0, 255))

    # read() defaults to authoritative planes only -- a history delta must
    # never record a cache.
    rec = s.read(Rect(0, 0, 4, 4))
    check("read defaults to authoritative only", set(rec) == {"index"},
          "-- got %r" % (sorted(rec),))

    # Palette edit must recolour exactly the pixels using that index.
    s.fill(Rect(8, 8, 2, 2), 1, name="index")
    pal.set_entry(17, (1, 2, 3, 255))
    s.recolor_index(17)
    check("recolor_index updates matching pixels",
          tuple(s.pixels[0, 0]) == (1, 2, 3, 255))
    check("recolor_index leaves other indices alone",
          tuple(s.pixels[8, 8]) == (10, 20, 30, 255))
    check("recolor_index leaves the index plane untouched",
          int(s.plane("index")[0, 0]) == 17)
    check("recolor_index of an unused index is a no-op",
          s.recolor_index(200) is None)

    # The capability no RGB-based approach has: two indices with identical
    # RGB must stay distinguishable, and editing one must not touch the other.
    pal2 = grayscale()
    pal2.set_entry(5, (60, 60, 60, 255))
    pal2.set_entry(6, (60, 60, 60, 255))       # deliberately identical
    s2 = Surface(8, 8, planes=("index", "rgba"), authoritative=("index",),
                 palette=pal2)
    s2.fill(Rect(0, 0, 2, 2), 5, name="index")
    s2.fill(Rect(4, 4, 2, 2), 6, name="index")
    check("duplicate-RGB entries render identically",
          tuple(s2.pixels[0, 0]) == tuple(s2.pixels[4, 4]))
    pal2.set_entry(6, (0, 255, 0, 255))
    s2.recolor_index(6)
    check("editing one of a duplicate pair moves only its pixels",
          tuple(s2.pixels[0, 0]) == (60, 60, 60, 255)
          and tuple(s2.pixels[4, 4]) == (0, 255, 0, 255))

    # Dropping the derived cache must be recoverable.
    freed = s2.drop_derived()
    check("drop_derived frees the cache", freed > 0 and not s2.is_allocated("rgba"))
    s2.refresh_derived()
    check("refresh_derived rebuilds from the index",
          tuple(s2.pixels[4, 4]) == (0, 255, 0, 255))

    # ---- readonly view ---------------------------------------------------
    s = Surface(8, 8)
    ro = s.readonly()
    try:
        ro[0, 0] = (1, 1, 1, 1)
        check("readonly view rejects writes", False, "-- write succeeded")
    except ValueError:
        check("readonly view rejects writes", True)

    # ---- copy / clear ----------------------------------------------------
    s = Surface(8, 8)
    s.fill(Rect(0, 0, 2, 2), (9, 9, 9, 255))
    c = s.copy()
    c.fill(Rect(0, 0, 2, 2), (1, 1, 1, 255))
    check("copy is independent", tuple(s.pixels[0, 0]) == (9, 9, 9, 255))
    check("copy carries content_bbox", c.content_bbox == Rect(0, 0, 2, 2))

    s.clear()
    check("clear releases planes", not s.is_allocated("rgba"))
    check("clear resets content_bbox", s.content_bbox is None)
    check("clear resets dirty", len(s.dirty) == 0)

    # ---- degenerate sizes ------------------------------------------------
    try:
        Surface(0, 10)
        check("zero-size surface rejected", False, "-- no error")
    except ValueError:
        check("zero-size surface rejected", True)

    print("\nall surface checks passed")


if __name__ == "__main__":
    main()
