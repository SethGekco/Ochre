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


def _refine_edge_dialog(shot, rough, m, edge, tint):
    """Select > Refine Edge as the user drives it: the dialog's Output
    section, ticked and set, then OK."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("ok: Refine Edge dialog not checked (PySide6 unavailable)")
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ochre.ui.bus import EventBus
    from ochre.ui.config import UiSettings
    from ochre.ui.dialogs import RefineEdgeDialog
    from ochre.ui.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[],
                     state_dir=tempfile.mkdtemp(prefix="ochre-refine-"))
    win = MainWindow(ctl, UiSettings(DATA))
    ctl.new_document(100, 100, background=None)
    cell = ctl.doc.cell(ctl.active_layer)
    cell.plane("rgba")[...] = shot
    cell._touched(ctl.doc.bounds)
    ctl.selection.mask = rough.copy()

    seen = {}

    def run(dlg):
        dlg.sliders["radius"].setValue(6)
        out = dlg.output
        seen["before"] = out.currentData()
        dlg.decontaminate.setChecked(True)
        seen["switched"] = out.currentData()
        seen["locked"] = not out.model().item(0).isEnabled() and dlg.amount.isEnabled()
        dlg.amount.setValue(100)
        return 1
    real = RefineEdgeDialog.exec
    RefineEdgeDialog.exec = run
    try:
        win.do_refine_edge()
    finally:
        RefineEdgeDialog.exec = real
    check("the Refine Edge dialog: ticking Decontaminate Colors moves Output To from "
          "Selection to New Layer with Layer Mask and locks the in-place outputs",
          seen == {"before": "selection", "switched": "new_layer_mask", "locked": True})
    layer = ctl.active_layer
    px = ctl.doc.cell(layer).plane("rgba")
    check("...and OK makes the cleaned, masked copy",
          layer.mask is not None and len(ctl.doc.layers()) == 2
          and np.array_equal(ctl.doc.cell(layer.mask).plane("rgba")[..., 0], m)
          and tint(px)[edge].mean() < 0.5 * tint(shot)[edge].mean())
    app.processEvents()


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

    # ---- Refine Edge: Decontaminate Colors and Output To ----------------------------
    # A subject with a soft 6 px edge, photographed on a green screen: the
    # edge pixels are part subject, part green.
    d = np.hypot(xx - 50, yy - 50)
    cover = np.clip((28 - d) / 6.0, 0, 1)[..., None]
    skin, green_bg = np.array([230, 200, 170.0]), np.array([0, 170, 0.0])
    shot = np.full((100, 100, 4), 255, np.uint8)
    shot[..., :3] = np.rint(cover * skin + (1 - cover) * green_bg)
    rough = (d < 25).astype(np.uint8) * 255
    tint = lambda a: a[..., 1].astype(float) - (a[..., 0].astype(float) + a[..., 2]) / 2
    m = refine.refine_mask(shot, rough, radius=6)
    edge = (m > 20) & (m < 235)
    clean = refine.decontaminate(shot, m, 100)
    check("Decontaminate Colors takes the green out of the soft edge (tint %.0f -> %.0f)"
          % (tint(shot)[edge].mean(), tint(clean)[edge].mean()),
          edge.sum() > 200 and tint(clean)[edge].mean() < 0.5 * tint(shot)[edge].mean())
    half = refine.decontaminate(shot, m, 50)
    check("...Amount scales it", tint(clean)[edge].mean() < tint(half)[edge].mean()
          < tint(shot)[edge].mean())
    check("...and leaves solid subject, background and alpha alone",
          np.array_equal(clean[m == 255], shot[m == 255])
          and np.array_equal(clean[m == 0], shot[m == 0])
          and np.array_equal(clean[..., 3], shot[..., 3]))

    ctl = fresh(shot)
    source = ctl.active_layer
    ctl.selection.mask = rough.copy()
    n0 = len(ctl.history_labels())
    ctl.refine_selection(radius=6, decontaminate=True, amount=100, output="selection")
    layer = ctl.active_layer
    check("Decontaminate with Output To Selection becomes New Layer with Layer Mask",
          layer is not source and layer.mask is not None and not source.visible
          and ctl.selection.mask is None)
    px = ctl.doc.cell(layer).plane("rgba")
    mpx = ctl.doc.cell(layer.mask).plane("rgba")[..., 0]
    check("...the copy is cleaned and masked by the refined edge",
          np.array_equal(mpx, m) and tint(px)[edge].mean() < 0.5 * tint(shot)[edge].mean())
    check("...as ONE undo step", ctl.history_labels()[n0:] == ["Refine Edge"])
    ctl.undo()
    check("...which undoes it all",
          ctl.doc.layer(layer.id) is None and source.visible
          and np.array_equal(ctl.selection.mask, rough))

    ctl.selection.mask = rough.copy()
    ctl.refine_selection(radius=6, output="new_layer")
    cut = ctl.doc.cell(ctl.active_layer).plane("rgba")
    check("Output To New Layer bakes the edge into the copy's alpha, colours untouched",
          ctl.active_layer.mask is None and np.array_equal(cut[..., 3], m)
          and np.array_equal(cut[..., :3], shot[..., :3]))
    ctl = fresh(shot)
    ctl.selection.mask = rough.copy()
    ctl.refine_selection(radius=6, output="layer_mask")
    check("Output To Layer Mask masks the active layer itself",
          len(ctl.doc.layers()) == 1 and ctl.active_layer.mask is not None
          and np.array_equal(ctl.doc.cell(ctl.active_layer.mask).plane("rgba")[..., 0], m))

    _refine_edge_dialog(shot, rough, m, edge, tint)

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
