#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Camera Raw, Lens Correction, Adaptive Wide Angle and the Filter Gallery.

Run: python3 tests/test_photo.py
"""

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine.effects import EffectRegistry
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REG = EffectRegistry()


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def run(name, img, **params):
    return REG.create(name, **params).apply(img)


def solid(h, w, rgb):
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = 255
    return out


def photo(h=96, w=128, seed=1):
    """Something photograph-like: smooth gradients, a few edges, some noise."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0] = np.clip(60 + 120 * xs / w + rng.normal(0, 6, (h, w)), 0, 255)
    out[..., 1] = np.clip(40 + 150 * ys / h + rng.normal(0, 6, (h, w)), 0, 255)
    out[..., 2] = np.clip(90 + 60 * np.sin(xs / 9.0) + rng.normal(0, 6, (h, w)), 0, 255)
    out[30:60, 40:80, :3] = (220, 210, 40)
    out[..., 3] = 255
    return out


def main():
    img = photo()

    # ---- Camera Raw ------------------------------------------------------------------
    check("Camera Raw at its defaults changes nothing", np.array_equal(run("camera_raw", img), img))
    grey = solid(8, 8, (128, 128, 128))
    up = run("camera_raw", grey, exposure=1.0)[0, 0, 0]
    check("+1 stop of exposure doubles LINEAR light (128 -> ~%d, not 255)" % up, 170 <= up <= 182)
    warm = run("camera_raw", grey, temperature=50)[0, 0]
    check("temperature warms: red up, blue down", warm[0] > 128 > warm[2])
    tint = run("camera_raw", grey, tint=50)[0, 0]
    check("tint toward magenta: green down", tint[1] < tint[0] and tint[1] < tint[2])
    flat = run("camera_raw", img, saturation=-100)
    spread = int((flat[..., :3].max(-1).astype(int) - flat[..., :3].min(-1)).max())
    check("saturation -100 is greyscale", spread <= 1, str(spread))

    def sat(px):
        return int(px.max()) - int(px.min())
    duo = np.concatenate([solid(4, 4, (140, 120, 110)), solid(4, 4, (230, 40, 30))], axis=1)
    v = run("camera_raw", duo, vibrance=80)
    gain_dull = sat(v[0, 0, :3]) / sat(duo[0, 0, :3])
    gain_vivid = sat(v[0, 6, :3]) / max(1, sat(duo[0, 6, :3]))
    check("vibrance boosts dull colours more than vivid ones (%.2f vs %.2f)" % (gain_dull, gain_vivid),
          gain_dull > gain_vivid)

    half = np.concatenate([solid(32, 32, (30, 30, 30)), solid(32, 32, (220, 220, 220))], axis=1)
    sh = run("camera_raw", half, shadows=80)
    check("shadows lift the dark side and leave the bright side nearly alone",
          sh[16, 4, 0] > 45 and abs(int(sh[16, 60, 0]) - 220) <= 4, str((sh[16, 4, 0], sh[16, 60, 0])))
    hl = run("camera_raw", half, highlights=-80)
    check("highlights -80 pull the bright side down", hl[16, 60, 0] < 205 and abs(int(hl[16, 4, 0]) - 30) <= 4)
    cl = run("camera_raw", img, clarity=100)

    def local_contrast(a):
        g = a[..., 1].astype(float)
        return float(np.abs(np.diff(g, axis=1)).mean())
    check("clarity raises local contrast", local_contrast(cl) > local_contrast(img) * 1.05)

    # Dehaze: a scene seen through haze, then cleared.
    # A scene of coherent patches (each with a dark channel near zero, as
    # natural colour patches have), with a band of sky that is pure haze --
    # where the airlight is seen, as in a real hazy photograph.
    rng = np.random.default_rng(3)
    blocks = rng.uniform(0, 200, (12, 12, 3))
    dark_chan = rng.integers(0, 3, (12, 12))
    for i in range(12):
        for j in range(12):
            blocks[i, j, dark_chan[i, j]] = rng.uniform(0, 15)
    scene = np.kron(blocks, np.ones((8, 8, 1)))
    t = np.full((96, 96, 1), 0.45)
    t[:16] = 0.0                                            # the sky
    hazy = np.zeros((96, 96, 4), np.uint8)
    hazy[..., :3] = np.clip(scene * t + 235 * (1 - t), 0, 255)
    hazy[..., 3] = 255
    clear = run("camera_raw", hazy, dehaze=100)
    ground = slice(24, 96)
    err_before = np.abs(hazy[ground, :, :3].astype(float) - scene[ground]).mean()
    err_after = np.abs(clear[ground, :, :3].astype(float) - scene[ground]).mean()
    check("dehaze brings a hazy scene back toward the real one (error %.0f -> %.0f)"
          % (err_before, err_after), err_after < err_before * 0.6)
    vg = run("camera_raw", solid(64, 64, (128, 128, 128)), vignette=-80)
    check("a negative vignette darkens the corners, not the centre",
          vg[0, 0, 0] < 100 and abs(int(vg[32, 32, 0]) - 128) <= 2)

    # ---- Lens Correction ---------------------------------------------------------------
    check("Lens Correction at its defaults changes nothing",
          np.abs(run("lens_correction", img).astype(int) - img.astype(int)).max() <= 1)
    grid = solid(120, 160, (255, 255, 255))
    grid[:, 20:22, :3] = 0                                  # a vertical line near the left edge

    def line_x(a, y):
        row = a[y, :80, 0].astype(float)
        dark = 255 - row
        return float((dark * np.arange(80)).sum() / max(dark.sum(), 1e-6))
    bends = {}
    for d in (-60, 0, 60):
        out = run("lens_correction", grid, distortion=d, edges="edge")
        bends[d] = line_x(out, 60) - line_x(out, 5)
    check("removing distortion bends edge lines one way or the other, 0 leaves them straight (%s)"
          % {k: round(v, 1) for k, v in bends.items()},
          abs(bends[0]) < 0.6 and bends[-60] * bends[60] < 0 and min(abs(bends[-60]), abs(bends[60])) > 1)
    box = solid(120, 160, (0, 0, 0))
    box[50:70, 130:150, :3] = 255                           # a white square near the right edge
    box[50:70, 70:90, :3] = 255                             # and one in the middle
    ca = run("lens_correction", box, red_cyan=100)
    r_edge = np.nonzero(ca[60, 100:, 0] > 128)[0]
    g_edge = np.nonzero(ca[60, 100:, 1] > 128)[0]
    check("fixing red/cyan fringing moves the red channel at the edge of the frame...",
          r_edge.size and g_edge.size and (r_edge.min() != g_edge.min() or r_edge.max() != g_edge.max()))
    check("...but not at the centre",
          np.array_equal(ca[60, 60:100, 0] > 128, ca[60, 60:100, 1] > 128))
    lv = run("lens_correction", solid(64, 64, (100, 100, 100)), vignette=80)
    check("lens vignette +80 brightens the corners", lv[0, 0, 0] > 115 and abs(int(lv[32, 32, 0]) - 100) <= 2)
    pair = solid(120, 160, (255, 255, 255))
    pair[:, 40:42, :3] = 0
    pair[:, 118:120, :3] = 0
    pv = run("lens_correction", pair, vertical=60)

    def gap(a, y):
        xs = np.nonzero(a[y, :, 0] < 128)[0]
        return xs.max() - xs.min() if xs.size else 0
    check("vertical perspective makes parallel verticals converge (gap %d top vs %d bottom)"
          % (gap(pv, 10), gap(pv, 110)), abs(gap(pv, 10) - gap(pv, 110)) >= 4)

    # ---- Adaptive Wide Angle -------------------------------------------------------------
    dot = solid(121, 161, (0, 0, 0))
    dot[60, 80, :3] = 255                                   # centre
    dot[60, 120, :3] = 255                                  # 40 px right of it
    fe = run("adaptive_wide_angle", dot, correction="fisheye", focal=10.0)
    xs = np.nonzero(fe[60, :, 0] > 20)[0]
    f = 10.0 / 36.0 * 161
    expect = 80 + f * math.tan(40 / f)
    check("fisheye to rectilinear keeps the centre and pushes off-centre points out (%s, expect ~%.0f)"
          % (list(xs), expect), 80 in xs and any(abs(x - expect) <= 2.5 for x in xs))
    cyl = run("adaptive_wide_angle", dot, correction="cylindrical", focal=10.0)
    xs = np.nonzero(cyl[60, :, 0] > 20)[0]
    check("wide angle to cylindrical pulls edge content IN", 80 in xs and xs.max() < 120)

    # ---- Filter Gallery: the stack effect ----------------------------------------------------
    stack = [{"name": "invert", "params": {}, "enabled": True},
             {"name": "pixelate", "params": {"cell": 8}, "enabled": True},
             {"name": "camera_raw", "params": {"exposure": 2.0}, "enabled": False}]
    got = run("filter_stack", img, stack=json.dumps(stack))
    want = run("pixelate", run("invert", img), cell=8)
    check("a filter stack applies its filters in order, skipping disabled ones", np.array_equal(got, want))
    check("an empty stack changes nothing", np.array_equal(run("filter_stack", img), img))

    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(128, 96, background=None)
    lay = ctl.active_layer
    ctl.doc.cell(lay).plane("rgba")[...] = img
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    n = len(ctl.history_labels())
    ctl.apply_effect("filter_stack", {"stack": json.dumps(stack)})
    check("the gallery's stack applies as ONE undo step",
          len(ctl.history_labels()) == n + 1 and np.array_equal(ctl.doc.cell(lay).plane("rgba"), want))
    ctl.undo()
    smart = ctl.convert_to_smart(lay)
    ctl.apply_effect("filter_stack", {"stack": json.dumps(stack)})
    check("...and on a smart object as ONE smart filter",
          [f["name"] for f in smart.params["filters"]] == ["filter_stack"])

    # ---- the dialog --------------------------------------------------------------------
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from ochre.ui.gallery import FilterGalleryDialog
    app = QApplication.instance() or QApplication([])
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(600, 400, background=None)
    big = np.tile(img, (5, 5, 1))[:400, :600]
    ctl.doc.cell(ctl.active_layer).plane("rgba")[...] = big
    reg = EffectRegistry(ctl.db)
    dlg = FilterGalleryDialog(ctl, reg)
    check("the gallery opens with one filter in its stack and a 100% preview",
          len(dlg.stack) == 1 and dlg.last_preview.shape[:2] == (320, 420))
    top = dlg.tree.topLevelItem(0)
    dlg.tree.expandItem(top)
    check("expanding a category draws a thumbnail per filter",
          top.childCount() > 3 and all(not top.child(i).icon(0).isNull() for i in range(top.childCount())))
    pix = [top.child(i) for i in range(top.childCount())]
    dlg._picked(pix[1])
    check("clicking a filter replaces the selected stack entry", dlg.stack[0]["name"] == pix[1].data(0, Qt.UserRole))
    dlg._new()
    dlg._picked(pix[2])
    check("New adds a layer to the stack, then a click sets it",
          [e["name"] for e in dlg.stack] == [pix[1].data(0, Qt.UserRole), pix[2].data(0, Qt.UserRole)])
    dlg.refresh_preview()
    expect = run(pix[2].data(0, Qt.UserRole), run(pix[1].data(0, Qt.UserRole), dlg.sample))
    check("the preview is the stack, at 100%", np.array_equal(dlg.last_preview, expect))
    dlg.list.item(0).setCheckState(Qt.Unchecked)
    check("unticking an entry turns it off", dlg.stack[0]["enabled"] is False)
    del app
    print("\nall photo-filter checks passed")


if __name__ == "__main__":
    main()
