#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Compositing, groups, and the below/above cache.

The load-bearing assertion here is that the cached path produces output
BYTE-IDENTICAL to an uncached full walk. A cache that is merely close is
worse than no cache at all, because the discrepancy only shows up as pixels
that change when you touch an unrelated layer.

Run: python3 tests/test_composite.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import accel
from ochre.engine.compositor import Compositor
from ochre.engine.document import Document
from ochre.engine.geometry import Rect


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fill(doc, layer, rgba, rect=None):
    doc.cell(layer).fill(rect or doc.bounds, rgba)


def main():
    # ---- basics ----------------------------------------------------------
    d = Document(16, 16)
    c = Compositor(d)
    out = c.composite()
    check("empty document composites to transparent", int(out.sum()) == 0)
    check("composite shape matches canvas", out.shape == (16, 16, 4))

    base = d.add_layer("base")
    fill(d, base, (255, 0, 0, 255))
    out = c.composite()
    check("single layer shows through", tuple(out[0, 0]) == (255, 0, 0, 255))

    top = d.add_layer("top")
    fill(d, top, (0, 0, 255, 255))
    out = c.composite()
    check("upper layer wins", tuple(out[0, 0]) == (0, 0, 255, 255))

    # ---- visibility and opacity -----------------------------------------
    top.visible = False
    check("hidden layer is skipped",
          tuple(c.composite()[0, 0]) == (255, 0, 0, 255))
    top.visible = True

    top.opacity = 0
    check("opacity 0 is skipped",
          tuple(c.composite()[0, 0]) == (255, 0, 0, 255))
    top.opacity = 128
    mid = c.composite()[0, 0]
    check("opacity 128 blends", 120 <= int(mid[2]) <= 135 and 120 <= int(mid[0]) <= 135,
          "-- got %r" % (tuple(mid),))
    top.opacity = 255

    # An empty cell must not contribute -- this is what makes sparseness pay.
    ghost = d.add_layer("ghost")
    d.cell(ghost)
    check("empty cell contributes nothing",
          tuple(c.composite()[0, 0]) == (0, 0, 255, 255))

    # ---- blend modes through the compositor ------------------------------
    d = Document(8, 8)
    c = Compositor(d)
    lo = d.add_layer("lo")
    hi = d.add_layer("hi")
    fill(d, lo, (200, 200, 200, 255))
    fill(d, hi, (128, 128, 128, 255))
    hi.blend = "Multiply"
    got = int(c.composite()[0, 0, 0])
    want = int(accel.blend_channel(accel.MULTIPLY, np.uint32(200), np.uint32(128)))
    check("blend mode reaches the compositor", got == want,
          "-- got %d want %d" % (got, want))

    hi.blend = "NoSuchMode"
    check("unknown blend renders as Normal rather than failing",
          tuple(c.composite()[0, 0]) == (128, 128, 128, 255))
    hi.blend = "Normal"

    # ---- groups are isolated --------------------------------------------
    d = Document(8, 8)
    c = Compositor(d)
    bg = d.add_layer("bg")
    fill(d, bg, (255, 255, 255, 255))
    g = d.add_group("g")
    inner = d.add_layer("inner", parent=g)
    fill(d, inner, (0, 0, 0, 255))
    check("group child composites", tuple(c.composite()[0, 0]) == (0, 0, 0, 255))

    g.visible = False
    check("hiding a group hides its children",
          tuple(c.composite()[0, 0]) == (255, 255, 255, 255))
    g.visible = True

    g.opacity = 128
    blended = int(c.composite()[0, 0, 0])
    check("group opacity scales its children", 120 <= blended <= 136,
          "-- got %d" % blended)
    g.opacity = 255

    # ---- region compositing ---------------------------------------------
    d = Document(32, 32)
    c = Compositor(d)
    lay = d.add_layer("a")
    d.cell(lay).fill(Rect(4, 4, 8, 8), (10, 20, 30, 255))
    part = c.composite(Rect(4, 4, 8, 8))
    check("region composite shape", part.shape == (8, 8, 4))
    check("region composite content", tuple(part[0, 0]) == (10, 20, 30, 255))
    check("off-canvas region is None", c.composite(Rect(99, 99, 4, 4)) is None)

    dst = np.zeros((32, 32, 4), dtype=np.uint8)
    r = c.composite_into(dst, Rect(4, 4, 8, 8))
    check("composite_into reports the clipped rect", r == Rect(4, 4, 8, 8))
    check("composite_into writes in document coordinates",
          tuple(dst[4, 4]) == (10, 20, 30, 255))
    check("composite_into leaves the rest alone", int(dst[0, 0].sum()) == 0)

    # A rect straddling the edge must clip, not raise.
    r = c.composite_into(dst, Rect(-4, -4, 12, 12))
    check("composite_into clips at the canvas edge", r == Rect(0, 0, 8, 8))

    # ---- THE cache test --------------------------------------------------
    # Ten layers, mixed blend modes and opacities, so the cache has real work
    # to be wrong about.
    d = Document(48, 48)
    c = Compositor(d)
    modes = ["Normal", "Multiply", "Screen", "Overlay", "Difference",
             "Lighten", "Darken", "Additive", "Negation", "Normal"]
    layers = []
    rng = np.random.default_rng(1234)
    for i, mode in enumerate(modes):
        lay = d.add_layer("L%d" % i)
        lay.blend = mode
        lay.opacity = int(rng.integers(64, 256))
        px = rng.integers(0, 256, size=(48, 48, 4), dtype=np.uint8)
        d.cell(lay).write(d.bounds, px)
        layers.append(lay)

    reference = c.composite()

    active = layers[5]
    c.build_cache(active)
    check("cache reports valid", c.cache_valid)

    cached = np.zeros((48, 48, 4), dtype=np.uint8)
    c.composite_cached(cached, d.bounds, active)
    check("CACHED OUTPUT IS BYTE-IDENTICAL TO A FULL WALK",
          np.array_equal(cached, reference),
          "-- %d pixels differ" % int(np.count_nonzero((cached != reference).any(axis=2))))

    # ...and for every possible active layer, not just one.
    worst = None
    for lay in layers:
        c.invalidate()
        c.build_cache(lay)
        got = np.zeros((48, 48, 4), dtype=np.uint8)
        c.composite_cached(got, d.bounds, lay)
        if not np.array_equal(got, reference):
            worst = lay.name
            break
    check("cache is exact for every active layer", worst is None,
          "-- failed with active=%s" % worst)

    # ...and over sub-rectangles, which is how a stroke actually uses it.
    c.invalidate()
    c.build_cache(active)
    for rect in (Rect(0, 0, 1, 1), Rect(10, 10, 7, 5), Rect(40, 40, 8, 8),
                 Rect(0, 20, 48, 3)):
        got = np.zeros((48, 48, 4), dtype=np.uint8)
        c.composite_cached(got, rect, active)
        want = reference[rect.slice()]
        if not np.array_equal(got[rect.slice()], want):
            check("cache exact over %r" % (rect,), False)
    check("cache is exact over sub-rectangles", True)

    # The associativity constraint, asserted directly rather than implied.
    # Above the bottom layer sit Multiply/Screen/Overlay/..., so the upper
    # half must NOT be pre-flattened; above the top layer there is nothing,
    # so it trivially can be.
    c.invalidate()
    c.build_cache(layers[0])
    check("upper half with blend modes is not pre-flattened",
          not c.above_flattened)
    c.invalidate()
    c.build_cache(layers[-1])
    check("nothing above the top layer, so flattening is legal",
          c.above_flattened)

    # And an all-Normal stack above the active layer IS flattenable.
    d2 = Document(16, 16)
    c2 = Compositor(d2)
    stack = []
    for i in range(4):
        lay = d2.add_layer("N%d" % i)
        d2.cell(lay).fill(d2.bounds, (i * 40, 10, 20, 200))
        stack.append(lay)
    ref2 = c2.composite()
    c2.build_cache(stack[0])
    check("all-Normal upper half is pre-flattened", c2.above_flattened)
    got2 = np.zeros((16, 16, 4), dtype=np.uint8)
    c2.composite_cached(got2, d2.bounds, stack[0])
    check("flattened upper half is still byte-exact",
          np.array_equal(got2, ref2))

    # Editing the active layer must show through the cache.
    c.invalidate()
    c.build_cache(active)
    d.cell(active).fill(Rect(0, 0, 4, 4), (255, 0, 255, 255))
    got = np.zeros((48, 48, 4), dtype=np.uint8)
    c.composite_cached(got, Rect(0, 0, 4, 4), active)
    fresh = c.composite(Rect(0, 0, 4, 4))
    check("edits to the active layer appear through the cache",
          np.array_equal(got[0:4, 0:4], fresh))

    # Changing the active layer must invalidate rather than serve stale data.
    c.set_active(layers[0].id)
    check("changing active layer invalidates", not c.cache_valid)

    # A cold cache must still produce correct output, not garbage.
    c.invalidate()
    got = np.zeros((48, 48, 4), dtype=np.uint8)
    c.composite_cached(got, d.bounds, active)
    check("cold cache falls back to a full walk correctly",
          np.array_equal(got, c.composite()))

    # ---- frames are composited independently ----------------------------
    d = Document(8, 8)
    c = Compositor(d)
    lay = d.add_layer("a")
    f1 = d.frame
    f2 = d.add_frame()
    d.cell(lay, f1).fill(d.bounds, (255, 0, 0, 255))
    d.cell(lay, f2).fill(d.bounds, (0, 255, 0, 255))
    check("frame 1 composites its own cells",
          tuple(c.composite(frame=f1)[0, 0]) == (255, 0, 0, 255))
    check("frame 2 composites its own cells",
          tuple(c.composite(frame=f2)[0, 0]) == (0, 255, 0, 255))

    print("\nall composite checks passed")


if __name__ == "__main__":
    main()
