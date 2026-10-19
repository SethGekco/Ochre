#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Refine Edge, sky selection and replacement, Merge to HDR, PDF export.

Run: python3 tests/test_refine.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import refine
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def landscape(h=120, w=160):
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 3] = 255
    t = np.linspace(0, 1, 70)[:, None]
    img[:70, :, 0] = (60 + t * 80).astype(np.uint8)
    img[:70, :, 1] = (110 + t * 70).astype(np.uint8)
    img[:70, :, 2] = (200 + t * 30).astype(np.uint8)
    rng = np.random.default_rng(0)
    img[70:, :, :3] = np.clip(np.array([110, 80, 50]) + rng.integers(-25, 25, (h - 70, w, 3)), 0, 255)
    img[90:110, 60:90, :3] = (80, 130, 215)        # a blue thing on the ground
    return img


def fresh(img):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(img.shape[1], img.shape[0], background=None)
    cell = ctl.doc.cell(ctl.active_layer)
    cell.plane("rgba")[...] = img
    cell._touched(ctl.doc.bounds)
    return ctl


def main():
    # ---- guided filter / refine edge ---------------------------------------------
    yy, xx = np.mgrid[:100, :100]
    disc = (xx - 50) ** 2 + (yy - 50) ** 2 < 25 ** 2
    img = np.full((100, 100, 4), 255, np.uint8)
    img[disc, :3] = 20
    rough = ((xx - 50) ** 2 + (yy - 50) ** 2 < 31 ** 2).astype(np.uint8) * 255
    r = refine.refine_mask(img, rough, radius=8)
    err_rough = int(((rough >= 128) != disc).sum())
    err_refined = int(((r >= 128) != disc).sum())
    check("Refine Edge pulls a rough selection onto the real edge (%d -> %d wrong pixels)"
          % (err_rough, err_refined), err_refined < err_rough * 0.75)
    check("...keeping the inside selected", r[50, 50] == 255 and r[2, 2] == 0)
    grown = refine.refine_mask(img, rough, radius=0, shift=60)
    check("shift edge grows the selection", (grown >= 128).sum() > (rough >= 128).sum() - 1)
    soft = refine.refine_mask(img, rough, radius=0, feather=6)
    check("feather softens it", ((soft > 10) & (soft < 245)).sum() > ((rough > 10) & (rough < 245)).sum())

    ctl = fresh(img)
    ctl.selection.mask = rough.copy()
    ctl.refine_selection(radius=8)
    check("Select > Refine Edge refines the real selection, undoably",
          int(((ctl.selection.mask >= 128) != disc).sum()) < err_rough * 0.75
          and ctl.history_labels()[-1] == "Refine Edge")

    # ---- sky ------------------------------------------------------------------------
    land = landscape()
    m = refine.sky_mask(land)
    check("the sky is found down to the horizon, even as it shades",
          (m[:68] >= 128).mean() > 0.97)
    check("...but not the ground, nor a sky-coloured thing on the ground",
          (m[72:] >= 128).mean() == 0 and m[100, 75] < 128)
    ctl = fresh(land)
    ctl.select_sky()
    check("Select > Sky", ctl.selection.mask is not None and ctl.selection.mask[10, 10] >= 128
          and ctl.selection.mask[100, 10] == 0)
    ctl.deselect()
    n0 = len(ctl.history_labels())
    layer = ctl.sky_replace("sunset")
    check("Sky Replacement adds a masked Sky layer", layer is not None and layer.mask is not None
          and ctl.history_labels()[n0:] == ["Sky Replacement"])
    ctl.compositor.invalidate()
    out = ctl.compositor.composite(ctl.doc.bounds)
    check("...the sky changes and the ground does not",
          not np.array_equal(out[10, 10, :3], land[10, 10, :3])
          and np.array_equal(out[100, 10, :3], land[100, 10, :3]))
    ctl.undo()
    check("...undone in one step", ctl.doc.layer(layer.id) is None)

    # ---- merge to HDR ------------------------------------------------------------------
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (48, 64, 3)).astype(float)
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    for k in (0.25, 1.0, 2.5):
        ctl.new_document(64, 48, background=None)
        cell = ctl.doc.cell(ctl.active_layer)
        cell.plane("rgba")[..., :3] = np.clip(base * k, 0, 255)
        cell.plane("rgba")[..., 3] = 255
        cell._touched(ctl.doc.bounds)
        ctl.set_tool("pencil")
        ctl.begin_stroke(0, 0)
        ctl.end_stroke(0, 0)               # touched, so the next opens beside it
    check("three exposures open", len(ctl.documents) == 3)
    ctl.merge_hdr([0, 1, 2])
    fused = ctl.doc.cell(ctl.active_layer).plane("rgba")[..., :3].astype(float)
    check("Merge to HDR makes a new image from the brackets", len(ctl.documents) == 4)
    clipped = lambda a: float(((a <= 2) | (a >= 253)).mean())
    darkest = np.clip(base * 0.25, 0, 255)
    brightest = np.clip(base * 2.5, 0, 255)
    check("...with fewer crushed or blown pixels than the extremes",
          clipped(fused) < clipped(brightest) and fused.mean() > darkest.mean())

    # ---- PDF ---------------------------------------------------------------------------
    path = os.path.join(tempfile.mkdtemp(), "out.pdf")
    ctl.export_pdf(path)
    with open(path, "rb") as fh:
        check("Export PDF writes a PDF", fh.read(5) == b"%PDF-")
    print("\nall refine/sky/HDR checks passed")


if __name__ == "__main__":
    main()
