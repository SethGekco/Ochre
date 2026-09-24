#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Integer pixel arithmetic -- the normative specification.

The mul255 identity is checked exhaustively rather than sampled. It is the
foundation the whole accelerator parity contract rests on, so "probably right"
is not good enough.

Run: python3 tests/test_arith.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.arith import (mul255, div255, lerp255, blend_over,
                                apply_coverage, expand_palette)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- the identity, over ALL 65536 pairs, not a sample ----
    a = np.arange(256, dtype=np.uint32).reshape(-1, 1)
    b = np.arange(256, dtype=np.uint32).reshape(1, -1)
    got = mul255(np.broadcast_to(a, (256, 256)), np.broadcast_to(b, (256, 256)))
    want = np.rint(a.astype(np.float64) * b.astype(np.float64) / 255.0).astype(np.uint8)
    bad = int(np.count_nonzero(got != want))
    check("mul255 == round(a*b/255) for all 65536 pairs", bad == 0,
          "-- %d mismatches" % bad)

    check("mul255 identity element", int(mul255(200, 255)) == 200)
    check("mul255 by zero", int(mul255(200, 0)) == 0)
    check("mul255 full scale", int(mul255(255, 255)) == 255)
    check("mul255 half rounds to nearest", int(mul255(255, 128)) == 128)
    check("mul255 returns uint8", mul255(10, 10).dtype == np.uint8)

    # div255 must agree with mul255 on the same products.
    prods = (a * b).ravel()
    check("div255 matches mul255 on products",
          np.array_equal(div255(prods), got.ravel()))

    # ---- lerp255 ----
    check("lerp t=0 gives a", int(lerp255(10, 200, 0)) == 10)
    check("lerp t=255 gives b", int(lerp255(10, 200, 255)) == 200)
    check("lerp is monotone", int(lerp255(0, 255, 128)) == 128)
    # Exhaustive over t for a representative pair, against float reference.
    ts = np.arange(256, dtype=np.uint32)
    ref = np.rint(40 + (220 - 40) * ts / 255.0).astype(np.uint8)
    check("lerp255 matches float reference across all t",
          np.array_equal(lerp255(np.full(256, 40), np.full(256, 220), ts), ref))
    check("lerp255 never overflows at extremes",
          int(lerp255(255, 255, 255)) == 255 and int(lerp255(0, 0, 0)) == 0)

    # ---- blend_over ----
    opaque_red = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
    opaque_blu = np.array([[[0, 0, 255, 255]]], dtype=np.uint8)
    clear = np.zeros((1, 1, 4), dtype=np.uint8)

    out = blend_over(opaque_red, opaque_blu)
    check("opaque over opaque replaces", tuple(out[0, 0]) == (0, 0, 255, 255))

    out = blend_over(opaque_red, clear)
    check("transparent src leaves dst", tuple(out[0, 0]) == (255, 0, 0, 255))

    out = blend_over(clear, opaque_blu)
    check("src over nothing is src", tuple(out[0, 0]) == (0, 0, 255, 255))

    out = blend_over(clear, clear)
    check("nothing over nothing is nothing", tuple(out[0, 0]) == (0, 0, 0, 0))

    half_blue = np.array([[[0, 0, 255, 128]]], dtype=np.uint8)
    out = blend_over(opaque_red, half_blue)
    r, g, b_, al = (int(v) for v in out[0, 0])
    check("half-alpha over opaque stays opaque", al == 255)
    check("half-alpha blends roughly evenly", 120 <= b_ <= 135 and 120 <= r <= 135,
          "-- got rgba=(%d,%d,%d,%d)" % (r, g, b_, al))

    # The fringe test: blending a coloured-but-transparent pixel must not
    # drag its colour in. This is the defect that produces dark halos.
    ghost = np.array([[[0, 0, 0, 0]]], dtype=np.uint8)      # black, invisible
    white = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
    out = blend_over(ghost, white)
    check("invisible black does not darken", tuple(out[0, 0]) == (255, 255, 255, 255))
    out = blend_over(white, ghost)
    check("invisible black over white leaves white",
          tuple(out[0, 0]) == (255, 255, 255, 255))

    # mask scales source alpha
    out = blend_over(opaque_red, opaque_blu, mask=np.array([[0]], dtype=np.uint8))
    check("zero mask blocks the source", tuple(out[0, 0]) == (255, 0, 0, 255))
    out = blend_over(opaque_red, opaque_blu, mask=np.array([[255]], dtype=np.uint8))
    check("full mask passes the source", tuple(out[0, 0]) == (0, 0, 255, 255))

    # Inputs must not be mutated.
    src_copy = opaque_blu.copy()
    dst_copy = opaque_red.copy()
    blend_over(opaque_red, opaque_blu)
    check("blend_over does not mutate inputs",
          np.array_equal(opaque_blu, src_copy) and np.array_equal(opaque_red, dst_copy))

    # ---- apply_coverage (scalar planes) ----
    d = np.full((4, 4), 100, dtype=np.uint8)
    s = np.full((4, 4), 200, dtype=np.uint8)
    check("coverage 0 keeps dst",
          np.array_equal(apply_coverage(d, s, np.zeros((4, 4), np.uint8)), d))
    check("coverage 255 takes src",
          np.array_equal(apply_coverage(d, s, np.full((4, 4), 255, np.uint8)), s))
    mid = apply_coverage(d, s, np.full((4, 4), 128, np.uint8))
    check("coverage 128 lands between", 148 <= int(mid[0, 0]) <= 152,
          "-- got %d" % int(mid[0, 0]))

    # ---- expand_palette ----
    pal = np.zeros((256, 4), dtype=np.uint8)
    pal[1] = (10, 20, 30, 255)
    pal[17] = (99, 88, 77, 255)
    idx = np.array([[0, 1], [17, 1]], dtype=np.uint8)
    rgba = expand_palette(pal, idx)
    check("expand_palette shape", rgba.shape == (2, 2, 4))
    check("expand_palette maps indices",
          tuple(rgba[1, 0]) == (99, 88, 77, 255) and tuple(rgba[0, 1]) == (10, 20, 30, 255))

    # out= must write in place, not allocate -- this is the hot path.
    dest = np.zeros((2, 2, 4), dtype=np.uint8)
    ret = expand_palette(pal, idx, out=dest)
    check("expand_palette writes into out=", ret is dest)
    check("expand_palette out= contents correct",
          tuple(dest[1, 0]) == (99, 88, 77, 255))

    # ---- dtype discipline ----
    for name, fn in (("mul255", lambda: mul255(np.uint8(3), np.uint8(4))),
                     ("lerp255", lambda: lerp255(np.uint8(3), np.uint8(4), np.uint8(5))),
                     ("div255", lambda: div255(np.uint32(1000)))):
        check("%s returns uint8" % name, fn().dtype == np.uint8)

    print("\nall arithmetic checks passed")


if __name__ == "__main__":
    main()
