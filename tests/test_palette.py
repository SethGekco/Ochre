#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Palette: annotations, the protection guarantee, and nearest-colour snap.

The protection test is the important one. Protected indices must be
*structurally* unreachable from a snap, not merely unlikely -- that is what
lets a format addon guarantee a remap ramp or a shadow index survives a
resample or a quantise untouched.

Run: python3 tests/test_palette.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.palette import LENGTH, Palette, grayscale


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- construction ----------------------------------------------------
    p = Palette()
    check("default palette length", len(p) == LENGTH)
    check("default is opaque", int(p.entries[:, 3].min()) == 255)

    rgb = np.zeros((LENGTH, 3), dtype=np.uint8)
    rgb[5] = (1, 2, 3)
    p = Palette(rgb)
    check("RGB input is widened to RGBA", p.rgba(5) == (1, 2, 3, 255))

    try:
        Palette(np.zeros((100, 4), np.uint8))
        check("wrong-size palette rejected", False, "-- no error")
    except ValueError:
        check("wrong-size palette rejected", True)

    g = grayscale()
    check("grayscale ramp", g.rgba(0) == (0, 0, 0, 255) and g.rgba(255) == (255, 255, 255, 255))
    check("grayscale is the only shipped palette", g.name == "Grayscale")

    # ---- annotations are opaque to the core ------------------------------
    p = Palette()
    p.annotate(16, 31, "Remap ramp", role="whatever-the-addon-says")
    p.annotate(4, 4, "Shadow")
    check("annotation found by index", p.annotation_for(20)[2] == "Remap ramp")
    check("annotation misses outside range", p.annotation_for(40) is None)
    check("single-index annotation", p.annotation_for(4)[2] == "Shadow")
    p.annotate(90, 80, "Reversed")
    check("reversed range is normalised", p.annotation_for(85)[2] == "Reversed")
    # The core stores the role string and never interprets it.
    check("role is carried verbatim",
          p.annotation_for(20)[3] == "whatever-the-addon-says")

    # ---- protection ------------------------------------------------------
    p = grayscale()
    p.protect((16, 31), 4)
    check("range protection expands", 16 in p.protected and 31 in p.protected)
    check("scalar protection", 4 in p.protected)
    check("unprotected neighbour", 15 not in p.protected and 32 not in p.protected)
    check("protected count", len(p.protected) == 17)

    cand = p.candidates()
    check("candidates exclude protected",
          not any(int(c) in p.protected for c in cand))

    # THE guarantee: sweep every possible RGB through the snap and assert no
    # protected index is ever emitted. Not sampled -- every LUT cell.
    lut = p.build_snap_lut(bits=5)
    check("snap LUT size", lut.size == 32768, "-- got %d" % lut.size)
    emitted = set(int(v) for v in np.unique(lut))
    leaked = emitted & set(p.protected)
    check("no protected index is reachable from any colour", not leaked,
          "-- leaked %r" % (sorted(leaked),))

    # And via the public snap() path over a dense colour sweep.
    sweep = np.stack(np.meshgrid(np.arange(0, 256, 7, dtype=np.uint8),
                                 np.arange(0, 256, 7, dtype=np.uint8),
                                 np.arange(0, 256, 7, dtype=np.uint8),
                                 indexing="ij"), axis=-1).reshape(-1, 3)
    snapped = p.snap(sweep)
    leaked = set(int(v) for v in np.unique(snapped)) & set(p.protected)
    check("snap() never emits a protected index over a dense sweep", not leaked,
          "-- leaked %r" % (sorted(leaked),))

    # Snapping an exact palette colour should land on that colour's value.
    p2 = grayscale()
    got = int(p2.snap(np.array([[130, 130, 130]], dtype=np.uint8))[0])
    check("snap of an exact entry is near-exact", abs(got - 130) <= 4,
          "-- snapped 130 to index %d" % got)

    # Snap is idempotent: snapping a palette colour then expanding then
    # snapping again must not drift.
    idx1 = p2.snap(p2.entries[100:101, :3])
    idx2 = p2.snap(p2.entries[idx1, :3])
    check("snap is idempotent", np.array_equal(idx1, idx2))

    # Transparent index is excluded from snap candidates.
    p3 = grayscale()
    p3.set_transparent(0)
    check("transparent excluded from candidates",
          0 not in set(int(c) for c in p3.candidates()))
    lut3 = p3.build_snap_lut(bits=5)
    check("snap never emits the transparent index",
          0 not in set(int(v) for v in np.unique(lut3)))

    # A palette with nothing left to snap to is an error, not a crash later.
    p4 = grayscale()
    p4.protect(*range(LENGTH))
    try:
        p4.build_snap_lut()
        check("fully-protected palette rejected", False, "-- no error")
    except ValueError:
        check("fully-protected palette rejected", True)

    # ---- render_entries --------------------------------------------------
    p5 = grayscale()
    p5.set_entry(0, (255, 0, 255, 255))
    check("opaque by default", p5.render_entries[0][3] == 255)
    p5.set_transparent(0)
    check("transparent index renders with alpha 0", p5.render_entries[0][3] == 0)
    # The authored colour must survive -- a format may legitimately store one.
    check("authored colour of transparent index is preserved",
          p5.rgba(0) == (255, 0, 255, 255))
    p5.set_transparent(None)
    check("clearing transparent restores opacity", p5.render_entries[0][3] == 255)

    # ---- cache invalidation ---------------------------------------------
    p6 = grayscale()
    first = p6.build_snap_lut(bits=5)
    check("LUT is cached", p6.build_snap_lut(bits=5) is first)
    p6.set_entry(10, (1, 2, 3, 255))
    check("entry edit invalidates the LUT", p6.build_snap_lut(bits=5) is not first)
    rev = p6.revision
    p6.protect(200)
    check("protection bumps revision", p6.revision > rev)

    # ---- exact lookup ----------------------------------------------------
    p7 = grayscale()
    check("index_of finds an exact entry", p7.index_of((77, 77, 77, 255)) == 77)
    check("index_of misses a non-entry", p7.index_of((1, 2, 3, 255)) is None)
    check("index_of widens RGB", p7.index_of((77, 77, 77)) == 77)

    # ---- copy independence ----------------------------------------------
    a = grayscale()
    a.protect(5)
    a.annotate(0, 9, "low")
    b = a.copy()
    b.set_entry(5, (9, 9, 9, 255))
    check("copy does not alias entries", a.rgba(5) == (5, 5, 5, 255))
    check("copy carries protection", 5 in b.protected)
    check("copy carries annotations", b.annotation_for(3)[2] == "low")

    # ---- build cost is tolerable ----------------------------------------
    p8 = grayscale()
    t0 = time.perf_counter()
    p8.build_snap_lut(bits=5)
    dt = (time.perf_counter() - t0) * 1000.0
    check("5-bit LUT builds in under 2 s", dt < 2000.0, "-- took %.0f ms" % dt)
    print("    (5-bit snap LUT built in %.0f ms)" % dt)

    print("\nall palette checks passed")


if __name__ == "__main__":
    main()
