#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Selective Color's nine ranges, the on-canvas Blur Gallery, Select Subject
and Remove Background, and content-aware canvas growth.

Run: python3 tests/test_moreactions.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ochre.engine import morefilters as MF
from ochre.engine import segment
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


def fresh(w, h, img=None):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=None)
    if img is not None:
        ctl.doc.cell(ctl.active_layer).plane("rgba")[...] = img
        ctl.doc.cell(ctl.active_layer)._touched(ctl.doc.bounds)
    return ctl


def texture(w, h, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.kron(rng.integers(30, 220, (h // 4 + 1, w // 4 + 1, 3)), np.ones((4, 4, 1)))[:h, :w]
    out[..., 3] = 255
    return out


def main():
    # ---- Selective Color, several ranges at once ------------------------------------------
    img = np.zeros((4, 12, 4), np.uint8)
    img[..., 3] = 255
    img[:, 0:4] = (220, 40, 40, 255)
    img[:, 4:8] = (40, 40, 220, 255)
    img[:, 8:12] = (128, 128, 128, 255)
    both = REG.create("selective_color", ranges=json.dumps({"reds": [100, 0, 0, 0],
                                                            "blues": [0, 0, 100, 0]})).apply(img)
    reds = REG.create("selective_color", colors="reds", cyan=100).apply(img)
    check("Selective Color applies several ranges in one pass",
          both[0, 0, 0] < 200 and both[0, 5, 2] < 210 and np.array_equal(both[0, 0], reds[0, 0]))
    check("...and leaves what is in none of them", np.array_equal(both[0, 9], img[0, 9]))
    check("...the old single-range form still works (recorded actions)", reds[0, 0, 0] < 200
          and np.array_equal(reds[0, 5], img[0, 5]))
    from PySide6.QtWidgets import QApplication
    from ochre.ui.effectdialog import EffectDialog
    app = QApplication.instance() or QApplication([])
    ctl = fresh(12, 4, img)
    fx = REG.create("selective_color")
    dlg = EffectDialog(fx, ctl)
    check("its dialog shows Photoshop's face (range menu + four inks), not raw settings",
          hasattr(dlg, "panel") and all(r.prop.name not in ("ranges", "cyan") for r in dlg.rows))
    dlg.panel.range.setCurrentIndex(dlg.panel.range.findData("reds"))
    dlg.panel.spins[0].setValue(80)
    dlg.panel.range.setCurrentIndex(dlg.panel.range.findData("blues"))
    dlg.panel.spins[2].setValue(60)
    check("...edits in each range are all kept", json.loads(fx.props.get("ranges")) ==
          {"reds": [80, 0, 0, 0], "blues": [0, 0, 60, 0]})
    dlg.reject()

    # ---- Field Blur with many pins ----------------------------------------------------------
    tex = texture(96, 64)
    out = MF.field_blur(tex, [(10, 10, 0), (86, 10, 0), (48, 54, 12)])
    change = lambda a, b, ys, xs: np.abs(a[ys, xs].astype(int) - b[ys, xs]).mean()
    check("Field Blur with three pins: sharp at the 0 px pins, blurred at the 12 px one",
          change(out, tex, slice(5, 15), slice(5, 15)) < 6
          and change(out, tex, slice(48, 60), slice(42, 54)) > 15)

    # ---- the Blur Gallery tool ---------------------------------------------------------------------
    ctl = fresh(96, 64, tex)
    lay = ctl.active_layer
    ctl.set_tool("blur_gallery", mode="field", blur=0)
    ctl.begin_stroke(10, 10)
    ctl.end_stroke(10, 10)
    ctl.begin_stroke(48, 54)
    ctl.end_stroke(48, 54)
    ctl.set_tool_option("blur", 12)              # sets the selected (new) pin
    t = ctl.tool
    check("Blur Gallery: clicks drop pins, each with its own Amount",
          [p[2] for p in t.pins] == [0.0, 12.0])
    ctl.set_tool_option("blur", 20)
    check("...changing Amount sets the selected pin", t.pins[1][2] == 20.0)
    ctl.begin_stroke(48, 54)
    ctl.motion_stroke(60, 50)
    ctl.end_stroke(60, 50)
    check("...a pin can be dragged", t.pins[1][:2] == [60.0, 50.0])
    live = ctl.doc.cell(lay).plane("rgba").copy()
    check("...and the layer previews it", not np.array_equal(live, tex))
    pos = ctl.history.position
    ctl.finish_tool()
    check("Enter applies it as one undo step", ctl.history.position == pos + 1
          and np.array_equal(ctl.doc.cell(lay).plane("rgba"), live))
    ctl.undo()
    check("...undoable", np.array_equal(ctl.doc.cell(lay).plane("rgba"), tex))
    ctl.set_tool("blur_gallery", mode="iris", blur=10, feather=30)
    ctl.begin_stroke(28, 12)
    ctl.motion_stroke(50, 30)
    ctl.end_stroke(68, 52)
    iris = ctl.doc.cell(lay).plane("rgba")
    check("Iris on the canvas: sharp in the middle of the ellipse, blurred in the corners",
          change(iris, tex, slice(28, 36), slice(44, 52)) < 3
          and change(iris, tex, slice(0, 8), slice(0, 8)) > 10)
    ctl.cancel_stroke()
    check("Escape puts the layer back", np.array_equal(ctl.doc.cell(lay).plane("rgba"), tex))
    ctl.set_tool("blur_gallery", mode="tilt_shift", blur=10, band=12, feather=0)
    ctl.begin_stroke(0, 32)
    ctl.end_stroke(96, 32)
    tilt = ctl.doc.cell(lay).plane("rgba")
    check("Tilt-Shift along a dragged line: sharp on the line, blurred far from it",
          change(tilt, tex, slice(29, 35), slice(10, 86)) < 1
          and change(tilt, tex, slice(0, 8), slice(10, 86)) > 10)
    ctl.finish_tool()

    # ---- Select Subject, Remove Background ----------------------------------------------------------
    im = Image.new("RGB", (120, 90), (70, 120, 200))
    d = ImageDraw.Draw(im)
    for y in range(90):
        d.line([(0, y), (119, y)], fill=(60 + y // 3, 110 + y // 4, 200 - y // 2))
    d.ellipse([40, 20, 85, 75], fill=(230, 180, 60))
    d.rectangle([55, 35, 70, 50], fill=(200, 60, 40))
    im = im.filter(ImageFilter.GaussianBlur(0.7))
    photo = np.zeros((90, 120, 4), np.uint8)
    photo[..., :3] = np.asarray(im)
    photo[..., 3] = 255
    m = segment.subject_mask(photo)
    inside = m[30:65, 48:78].mean()
    outside = np.concatenate([m[:12].ravel(), m[:, :25].ravel(), m[:, 100:].ravel()]).mean()
    check("Select Subject finds the subject (%.0f inside, %.0f outside)" % (inside, outside),
          inside > 230 and outside < 10)
    ctl = fresh(120, 90, photo)
    ctl.select_subject()
    check("...as an undoable selection", ctl.selection.mask is not None
          and ctl.selection.mask[45, 62] > 200 and ctl.history_labels()[-1] == "Select Subject")
    ctl.selection.select_all()
    lay = ctl.active_layer
    ctl.remove_background()
    shown = ctl.compositor.composite(ctl.doc.bounds)
    check("Remove Background hides the background with a layer mask, pixels kept",
          lay.mask is not None and shown[45, 62, 3] > 230 and shown[5, 5, 3] < 20
          and np.array_equal(ctl.doc.cell(lay).plane("rgba"), photo))

    # ---- content-aware canvas growth ----------------------------------------------------------------
    ctl = fresh(64, 48, texture(64, 48, 3))
    lay = ctl.active_layer
    pos = ctl.history.position
    ctl.resize_canvas(96, 48, "center", fill="content_aware")
    px = ctl.doc.cell(lay).plane("rgba")
    check("Canvas Size with Content-Aware fills the new area from the picture",
          (ctl.doc.width, ctl.doc.height) == (96, 48) and px[..., 3].min() == 255
          and px[:, :16, :3].std() > 20)
    check("...as one undo step", ctl.history.position == pos + 1)
    ctl.undo()
    check("...undoable", (ctl.doc.width, ctl.doc.height) == (64, 48))
    ctl.set_tool("crop", fill="content_aware")
    ctl.begin_stroke(-16, 0)
    ctl.motion_stroke(30, 30)
    ctl.end_stroke(64, 48)
    ctl.finish_tool()
    px = ctl.doc.cell(ctl.active_layer).plane("rgba")
    check("cropping past the edge grows the canvas, Content-Aware fills it",
          (ctl.doc.width, ctl.doc.height) == (80, 48) and px[..., 3].min() == 255)
    ctl.undo()
    ctl.set_tool("crop", fill="transparent")
    ctl.begin_stroke(0, -8)
    ctl.end_stroke(64, 48)
    ctl.finish_tool()
    check("...or leaves it transparent", (ctl.doc.width, ctl.doc.height) == (64, 56)
          and ctl.doc.cell(ctl.active_layer).plane("rgba")[:8, :, 3].max() == 0)
    del app
    print("\nall more-action checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
