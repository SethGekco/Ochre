#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Quantisation into a fixed palette.

The assertion that matters: PROTECTED INDICES ARE UNREACHABLE. A reserved
ramp or a designated shadow index must survive a conversion untouched, and
that has to be structural -- excluded from the lookup table -- rather than
merely unlikely.

Run: python3 tests/test_quantize.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.document import Document
from ochre.engine.geometry import Rect
from ochre.engine.palette import LENGTH, Palette, grayscale
from ochre.engine.quantize import (FLOYD, NONE, ORDERED, convert_layer_to_indexed,
                                   palette_usage, quantization_error, quantize)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def rgba(h, w, colour):
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[...] = colour
    return out


def main():
    pal = grayscale()

    # ---- basic mapping ---------------------------------------------------
    img = rgba(8, 8, (128, 128, 128, 255))
    idx = quantize(img, pal)
    check("output shape is a single plane", idx.shape == (8, 8))
    check("output is uint8", idx.dtype == np.uint8)
    check("mid grey maps near index 128", abs(int(idx[0, 0]) - 128) <= 4,
          "-- got %d" % int(idx[0, 0]))

    black = quantize(rgba(4, 4, (0, 0, 0, 255)), pal)
    white = quantize(rgba(4, 4, (255, 255, 255, 255)), pal)
    check("black maps to a low index", int(black[0, 0]) <= 4)
    check("white maps to a high index", int(white[0, 0]) >= 250)

    try:
        quantize(np.zeros((4, 4), np.uint8), pal)
        check("non-RGBA input rejected", False, "-- no error")
    except ValueError:
        check("non-RGBA input rejected", True)
    try:
        quantize(img, pal, dither="nonsense")
        check("unknown dither mode rejected", False, "-- no error")
    except ValueError:
        check("unknown dither mode rejected", True)

    # ---- THE protection guarantee ---------------------------------------
    protected = grayscale()
    protected.protect((16, 31), 4)
    # A colour sitting exactly on a protected entry must still not produce it.
    exact = rgba(4, 4, tuple(int(v) for v in protected.entries[20]))
    got = quantize(exact, protected)
    check("a colour matching a protected entry does not produce it",
          int(got[0, 0]) not in protected.protected,
          "-- produced protected index %d" % int(got[0, 0]))

    # Sweep the whole colour cube through every dither mode.
    sweep = np.zeros((32, 32, 4), dtype=np.uint8)
    rng = np.random.default_rng(3)
    sweep[..., :3] = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    sweep[..., 3] = 255
    for mode in (NONE, ORDERED, FLOYD):
        out = quantize(sweep, protected, dither=mode)
        leaked = set(int(v) for v in np.unique(out)) & set(protected.protected)
        check("%s dithering never emits a protected index" % mode, not leaked,
              "-- leaked %r" % sorted(leaked))

    # ---- transparency ----------------------------------------------------
    trans = grayscale()
    trans.set_transparent(0)
    mixed = np.zeros((4, 4, 4), dtype=np.uint8)
    mixed[..., :3] = 200
    mixed[0, 0, 3] = 0             # transparent, but with meaningless RGB
    mixed[1, 1, 3] = 255
    out = quantize(mixed, trans)
    check("transparent pixels take the transparent index",
          int(out[0, 0]) == 0,
          "-- a transparent pixel's RGB must not be matched as a colour")
    check("opaque pixels are matched normally", int(out[1, 1]) != 0)
    check("the transparent index is not a snap target",
          0 not in set(int(c) for c in trans.candidates()))

    # Without a transparent index declared, alpha is simply ignored.
    out = quantize(mixed, grayscale())
    check("no transparent index means no special case", int(out[0, 0]) != 0
          or int(out[1, 1]) == int(out[0, 0]))

    # ---- ordered dithering ----------------------------------------------
    # A flat colour halfway between two palette entries should break up into
    # a mix rather than banding to one.
    coarse = Palette(np.array([[i * 64 if i < 4 else 255] * 3 for i in range(LENGTH)],
                              dtype=np.uint8))
    flat = rgba(16, 16, (96, 96, 96, 255))       # between 64 and 128
    plain = quantize(flat, coarse, dither=NONE)
    dithered = quantize(flat, coarse, dither=ORDERED)
    check("undithered flat colour picks one index",
          len(np.unique(plain)) == 1)
    check("ordered dithering mixes two indices",
          len(np.unique(dithered)) > 1,
          "-- dithering produced a single flat index")

    # Deterministic, and independent of where the region sits.
    a = quantize(flat, coarse, dither=ORDERED)
    b = quantize(flat, coarse, dither=ORDERED)
    check("ordered dithering is deterministic", np.array_equal(a, b))

    big = rgba(24, 24, (96, 96, 96, 255))
    whole = quantize(big, coarse, dither=ORDERED)
    part = quantize(big[8:16, 8:16], coarse, dither=ORDERED, origin=(8, 8))
    check("ordered dithering tiles seamlessly",
          np.array_equal(part, whole[8:16, 8:16]),
          "-- a region quantised alone must match the same region of the whole")

    # ---- Floyd-Steinberg -------------------------------------------------
    fs = quantize(flat, coarse, dither=FLOYD)
    check("Floyd-Steinberg mixes indices", len(np.unique(fs)) > 1)
    check("Floyd-Steinberg is deterministic",
          np.array_equal(fs, quantize(flat, coarse, dither=FLOYD)))

    # It must refuse a job it cannot do promptly, rather than hanging.
    huge = np.zeros((1200, 1200, 4), dtype=np.uint8)
    huge[..., 3] = 255
    try:
        quantize(huge, coarse, dither=FLOYD)
        check("Floyd-Steinberg refuses oversized input", False, "-- no error")
    except ValueError as exc:
        check("Floyd-Steinberg refuses oversized input", True)
        check("...and says why, with an estimate", "force=True" in str(exc))

    # ---- error reporting -------------------------------------------------
    err = quantization_error(flat, quantize(flat, coarse), coarse)
    check("error report counts opaque pixels", err["pixels"] == 16 * 16)
    check("error report has a mean and a max",
          err["mean"] >= 0 and err["max"] >= 0)

    exact_pal = grayscale()
    grey = rgba(8, 8, (77, 77, 77, 255))
    err = quantization_error(grey, quantize(grey, exact_pal), exact_pal)
    check("an exact palette match reports zero error", err["max"] == 0,
          "-- got max %d" % err["max"])

    # What dithering actually buys, stated precisely: it does NOT reduce
    # per-pixel error -- every pixel is still snapped to a palette entry and
    # is just as far from the original. It makes the LOCAL AVERAGE correct.
    # Asserting the per-pixel figure would be asserting the wrong thing.
    plain_px = quantization_error(flat, quantize(flat, coarse, NONE), coarse)
    dith_px = quantization_error(flat, quantize(flat, coarse, ORDERED), coarse)
    check("dithering does not reduce per-pixel error",
          abs(dith_px["mean"] - plain_px["mean"]) < 1.0,
          "-- plain %.1f vs dithered %.1f" % (plain_px["mean"], dith_px["mean"]))

    source_mean = float(flat[..., :3].mean())
    plain_mean = float(coarse.render_entries[quantize(flat, coarse, NONE)][..., :3].mean())
    dith_mean = float(coarse.render_entries[quantize(flat, coarse, ORDERED)][..., :3].mean())
    check("dithering makes the LOCAL AVERAGE far closer to the original",
          abs(dith_mean - source_mean) < abs(plain_mean - source_mean),
          "-- source %.1f, plain %.1f, dithered %.1f"
          % (source_mean, plain_mean, dith_mean))
    print("    (source %.0f -> undithered %.0f, dithered %.1f)"
          % (source_mean, plain_mean, dith_mean))

    usage = palette_usage(quantize(flat, coarse, ORDERED))
    check("usage reports only the indices present", all(c > 0 for c in usage.values()))
    check("usage covers every pixel", sum(usage.values()) == 16 * 16)

    # ---- document conversion --------------------------------------------
    doc = Document(32, 32)
    layer = doc.add_layer("Photo")
    cell = doc.cell(layer)
    cell.fill(Rect(0, 0, 16, 32), (200, 40, 40, 255))
    cell.fill(Rect(16, 0, 16, 32), (40, 40, 200, 255))
    original = cell.pixels.copy()

    pal = grayscale()
    report = convert_layer_to_indexed(doc, layer, pal)
    check("conversion makes the layer index-locked", layer.index_locked)
    check("conversion binds the palette", doc.palette is pal)
    check("conversion reports its error", len(report) == 1)
    converted = doc.cell(layer)
    check("converted cell has an index plane", converted.has("index"))
    check("index plane is authoritative", "index" in converted.authoritative)
    check("derived colour was rebuilt",
          int(converted.pixels[0, 0].sum()) > 0)

    # Converting across frames must convert EVERY frame.
    doc2 = Document(16, 16)
    lay2 = doc2.add_layer("Art")
    f2 = doc2.add_frame()
    doc2.cell(lay2).fill(doc2.bounds, (100, 100, 100, 255))
    doc2.cell(lay2, f2).fill(doc2.bounds, (200, 200, 200, 255))
    report = convert_layer_to_indexed(doc2, lay2, grayscale())
    check("conversion covers every frame", len(report) == 2,
          "-- converted %d cells" % len(report))
    check("each frame kept its own pixels",
          int(doc2.cell(lay2).plane("index")[0, 0])
          != int(doc2.cell(lay2, f2).plane("index")[0, 0]))

    doc3 = Document(8, 8)
    lay3 = doc3.add_layer("A")
    try:
        convert_layer_to_indexed(doc3, lay3)
        check("conversion without a palette is refused", False, "-- no error")
    except ValueError:
        check("conversion without a palette is refused", True)

    # ---- round-trip fidelity --------------------------------------------
    # Converting an image that ALREADY uses only palette colours must be
    # lossless -- that is the case a sprite round-trip depends on.
    pal = grayscale()
    exact_img = np.zeros((16, 16, 4), dtype=np.uint8)
    for i in range(16):
        exact_img[i, :, :3] = i * 17
    exact_img[..., 3] = 255
    idx = quantize(exact_img, pal)
    rebuilt = pal.render_entries[idx]
    check("an already-in-palette image converts losslessly",
          np.array_equal(rebuilt[..., :3], exact_img[..., :3]),
          "-- a sprite round-trip depends on this being exact")

    # ---- cost ------------------------------------------------------------
    photo = np.zeros((512, 512, 4), dtype=np.uint8)
    photo[..., :3] = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    photo[..., 3] = 255
    pal = grayscale()
    pal.build_snap_lut(5)
    t0 = time.perf_counter()
    quantize(photo, pal, dither=ORDERED)
    dt = (time.perf_counter() - t0) * 1000
    check("ordered dithering of 512x512 stays under 250 ms", dt < 250,
          "-- took %.0f ms" % dt)
    print("    (512x512 ordered dither: %.0f ms)" % dt)

    print("\nall quantisation checks passed")


if __name__ == "__main__":
    main()
