#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Banded multi-threaded compositing must be BIT-IDENTICAL to one thread.

Compared with array_equal on a stack that exercises every per-pixel feature
the compositor has -- blend modes, opacity, masks, vector masks, clipping,
isolated and pass-through groups, Blend If, knockout, styles, fill, shape
and adjustment layers -- through every entry point that bands: a full
composite, a composite into the display buffer, the stroke cache, and the
controller's own flush, which is what the canvas draws. Built-in effects
that render their tiles on the pool are compared the same way.

Runs under both backends (tests/run_all.sh runs everything twice), so the
numpy fallback is banded and checked too.

Run: python3 tests/test_bands.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import parallel, proxy
from ochre.engine.compositor import Compositor
from ochre.engine.document import Document
from ochre.engine.geometry import Rect
from ochre.engine.layer import ADJUSTMENT, FILL, LayerMask

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def _paint(cell, seed, alpha):
    h, w = cell.height, cell.width
    yy, xx = np.mgrid[0:h, 0:w]
    px = cell.plane("rgba")
    px[..., 0] = (xx * (seed + 3) // 7 + yy) % 256
    px[..., 1] = (yy * (seed + 5) // 11) % 256
    px[..., 2] = ((xx + yy) * (seed + 1) // 5) % 256
    px[..., 3] = alpha if alpha is not None else (xx * 3 + yy) % 256
    cell._touched(Rect(0, 0, w, h))


def build(w, h):
    """A document using every per-pixel feature the compositor has."""
    from ochre.engine import paths
    doc = Document(w, h)
    base = doc.add_layer("base")
    _paint(doc.cell(base), 1, 255)
    mul = doc.add_layer("multiply, masked")
    _paint(doc.cell(mul), 4, None)
    mul.blend, mul.opacity = "Multiply", 200
    mul.mask = LayerMask(mul)
    m = doc.cell(mul.mask).plane("rgba")
    m[...] = 255
    m[:, : w // 2, :3] = 40
    doc.cell(mul.mask)._touched(doc.bounds)
    clipped = doc.add_layer("clipped hue")
    _paint(doc.cell(clipped), 9, 180)
    clipped.clip, clipped.blend = True, "Hue"
    group = doc.add_group("isolated overlay group")
    inner = doc.add_layer("inner screen", parent=group)
    _paint(doc.cell(inner), 7, 150)
    inner.blend = "Screen"
    group.blend = "Overlay"
    through = doc.add_group("pass-through")
    knock = doc.add_layer("knockout", parent=through)
    doc.cell(knock).fill(Rect(w // 5, h // 6, w // 3, h // 2), (10, 200, 30, 255))
    knock.knockout = "shallow"
    knock.fill_opacity = 120
    soft = doc.add_layer("soft light, blend if, styled")
    doc.cell(soft).fill(Rect(w // 3, h // 4, w // 3, h // 3), (40, 200, 240, 220))
    soft.blend = "SoftLight"
    soft.blend_if = (0, 30, 220, 255, 10, 60, 200, 250)
    soft.styles = {"stroke": {"size": 3, "color": (0, 0, 255)},
                   "drop_shadow": {"distance": 6, "size": 5}}
    vec = doc.add_layer("vector masked")
    _paint(doc.cell(vec), 2, 160)
    vec.blend = "Difference"
    vec.vector_mask = paths.Path("triangle", [paths.Subpath(
        [paths.Anchor(w * 0.1, h * 0.2), paths.Anchor(w * 0.9, h * 0.35),
         paths.Anchor(w * 0.5, h * 0.95)], closed=True)])
    fill = doc.add_layer("gradient fill")
    fill.kind = FILL
    fill.params = {"type": "gradient", "start": "200,40,40,120", "end": "20,40,220,60",
                   "angle": 30}
    fill.blend = "LinearLight"
    adj = doc.add_layer("adjust")
    adj.kind = ADJUSTMENT
    adj.params = {"effect": "hue_saturation", "hue": 25, "saturation": 130, "lightness": 5}
    adj2 = doc.add_layer("curves")
    adj2.kind = ADJUSTMENT
    adj2.params = {"effect": "levels", "input_low": 20, "input_high": 230, "gamma": 1.3}
    return doc


def serial_then_banded(fn, threads=4):
    parallel.set_threads(1)
    a = fn()
    parallel.set_threads(threads)
    before = parallel.stats["banded"]
    b = fn()
    ran = parallel.stats["banded"] > before
    return a, b, ran


def main():
    saved = parallel.threads()
    doc = build(700, 520)
    W, H = doc.width, doc.height

    # ---- the splitting itself -------------------------------------------
    parallel.set_threads(4)
    split = parallel.bands(Rect(3, 5, 700, 520))
    check("a big rect splits into a short warm-up band and one per thread",
          split is not None and split[0].h == parallel.WARM_ROWS and len(split) == 5,
          "-- %r" % (split,))
    check("the bands tile the rect exactly",
          sum(b.area for b in split) == 700 * 520 and split[0].y == 5
          and all(a.y1 == b.y for a, b in zip(split, split[1:])) and split[-1].y1 == 525)
    check("a brush-sized rect is not split", parallel.bands(Rect(0, 0, 256, 256)) is None)
    aligned = parallel.bands(Rect(0, 0, 999, 1000), align=8)
    check("aligned bands start on the block grid",
          aligned is not None and all(b.y % 8 == 0 for b in aligned))
    parallel.set_threads(1)
    check("one thread means no bands", parallel.bands(Rect(0, 0, 4000, 4000)) is None)

    # ---- full composite ---------------------------------------------------
    a, b, ran = serial_then_banded(lambda: Compositor(doc).composite())
    check("banded composite ran on the pool", ran)
    check("banded composite == single-threaded, byte for byte",
          np.array_equal(a, b), "-- %d pixels differ" % int((a != b).any(-1).sum()))

    # ---- into a canvas-sized buffer, over an unaligned rect ---------------
    rect = Rect(13, 7, 651, 499)

    def into():
        dst = np.full((H, W, 4), 77, np.uint8)
        Compositor(doc).composite_into(dst, rect)
        return dst
    a, b, ran = serial_then_banded(into)
    check("banded composite_into == single-threaded, outside the rect untouched",
          ran and np.array_equal(a, b))

    # ---- the stroke cache: below, active, above ---------------------------
    # (A knockout makes the cache step aside for the full walk; take it out.)
    knocker = [l for l in doc.layers() if getattr(l, "knockout", "none") != "none"][0]
    knocker.knockout = "none"
    active = doc.layer(doc.root.children[1].id)

    def cached():
        comp = Compositor(doc)
        comp.build_cache(active)
        dst = np.zeros((H, W, 4), np.uint8)
        comp.composite_cached(dst, doc.bounds, active)
        return dst, comp._below.copy()
    (a, ab), (b, bb), ran = serial_then_banded(cached)
    check("banded stroke cache == single-threaded (below and composite)",
          ran and np.array_equal(ab, bb) and np.array_equal(a, b))
    check("...and the cached composite is the full composite",
          np.array_equal(b, Compositor(doc).composite()))
    knocker.knockout = "shallow"

    # ---- the reduced-preview compositor bands too -------------------------
    a, b, ran = serial_then_banded(lambda: proxy.ProxyCompositor(Compositor(doc), 2).composite())
    check("banded reduced preview == single-threaded", ran and np.array_equal(a, b))
    big = np.random.default_rng(5).integers(0, 256, (1203, 997, 4), dtype=np.uint8)
    a, b, ran = serial_then_banded(lambda: proxy.reduce_box(big, 8))
    check("banded box reduction == single-threaded", ran and np.array_equal(a, b))

    # ---- effect tiles on the pool -----------------------------------------
    # Every built-in effect that renders its tiles on several threads, on an
    # image just big enough to fan out: the same bytes as one thread.
    from ochre.engine.effects import EffectRegistry
    reg = EffectRegistry()
    img = np.empty((520, 520, 4), np.uint8)
    yy, xx = np.mgrid[0:520, 0:520]
    img[..., 0], img[..., 1], img[..., 2] = (xx * 3) % 256, (yy * 5) % 256, (xx + yy) % 256
    img[..., 3] = (xx * 7 + yy) % 256
    tiled = [n for n in reg.names() if getattr(reg.create(n), "parallel_tiles", False)]
    differ = []
    pooled = False
    for name in tiled:
        settings = {"radius": 3} if name == "gaussian_blur" else {}
        a, b, ran = serial_then_banded(lambda: reg.create(name, **settings).apply(img))
        pooled = pooled or ran
        if not np.array_equal(a, b):
            differ.append(name)
    check("%d effects render tiles on the pool, identical to one thread" % len(tiled),
          len(tiled) >= 10 and pooled and not differ, "-- %r" % differ)

    # ---- a neighbourhood adjustment layer is never banded -----------------
    blur = doc.add_layer("blur adjustment")
    blur.kind = ADJUSTMENT
    blur.params = {"effect": "gaussian_blur", "radius": 3}
    a, b, _ = serial_then_banded(lambda: Compositor(doc).composite())
    # (The blur's own tiles may still use the pool -- inside one rect, which
    # is exact. What must not happen is the rect being cut into bands.)
    check("a stack with a non-pointwise adjustment is not cut into bands",
          not Compositor(doc)._bandable() and np.array_equal(a, b))
    doc.remove_layer(blur)

    # ---- the real display path: the controller's flush --------------------
    from ochre.engine.settings import Settings
    from ochre.ui.controller import Controller
    ctl = Controller(Settings(DATA), DATA, None, addon_dirs=[],
                     state_dir=tempfile.mkdtemp(prefix="ochre-previews-"))
    ctl.new_document(640, 480, (255, 255, 255, 255))
    for i, mode in enumerate(("Normal", "Overlay", "ColorBurn")):
        layer = ctl.add_layer("L%d" % i)
        _paint(ctl.doc.cell(layer), i, 255 if i == 0 else 150)
        layer.blend = mode

    def flushed():
        ctl.compositor.invalidate()
        ctl.invalidate()
        ctl.flush()
        return ctl.display.copy()
    ctl.settings._cache[("Performance", "Threads")] = 4
    parallel.configure(ctl.settings)
    check("the Performance/Threads setting is what the pool uses", parallel.threads() == 4)
    a, b, ran = serial_then_banded(flushed)
    check("the display buffer composited in bands == on one thread", ran and np.array_equal(a, b))

    parallel.set_threads(saved)
    print("\nall banded-compositing checks passed")


if __name__ == "__main__":
    main()
