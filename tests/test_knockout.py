#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Fill opacity and knockouts (Photoshop's Blending Options), through the
controller.

Run: python3 tests/test_knockout.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine.fileio import ochre_format, psd
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WHITE, RED, GREEN, BLUE = (255, 255, 255, 255), (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh():
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(80, 60, background=WHITE)
    return ctl


def shown(ctl):
    ctl.compositor.invalidate()
    return ctl.compositor.composite(ctl.doc.bounds)


def solid(ctl, name, colour, rect=None, parent=None):
    lay = ctl.add_layer(name, parent=parent)
    ctl.doc.cell(lay).fill(rect or ctl.doc.bounds, colour)
    return lay


def px(img, x, y):
    return tuple(int(v) for v in img[y, x])


def main():
    # ---- fill opacity ----------------------------------------------------------------
    ctl = fresh()
    top = solid(ctl, "blue", BLUE, Rect(20, 20, 40, 20))
    ctl.set_layer_styles(top, {"stroke": {"size": 2, "color": (0, 200, 0), "position": "outside"}})
    ctl.set_layer_property(top, "fill_opacity", 0)
    img = shown(ctl)
    check("Fill 0 hides the layer's own pixels...", px(img, 40, 30) == WHITE)
    check("...but not its layer style (the stroke stays)", px(img, 19, 30)[1] == 200)
    ctl.undo()
    check("fill opacity is one undo step", top.fill_opacity == 255 and px(shown(ctl), 40, 30) == BLUE)
    ctl.set_layer_property(top, "fill_opacity", 128)
    p = px(shown(ctl), 40, 30)
    check("Fill 50% mixes the layer with what is beneath", p[2] == 255 and 120 < p[0] < 136
          and 120 < p[1] < 136, str(p))

    # ---- knockout at the top level reaches the bottom layer -----------------------------
    ctl = fresh()
    red = solid(ctl, "red", RED)
    hole = solid(ctl, "hole", BLUE, Rect(20, 20, 40, 20))
    ctl.set_layer_property(hole, "knockout", "deep")
    check("an opaque knockout layer at Fill 100% looks like a normal layer",
          px(shown(ctl), 40, 30) == BLUE)
    ctl.set_layer_property(hole, "fill_opacity", 0)
    img = shown(ctl)
    check("at Fill 0 it punches a hole through red to the bottom layer",
          px(img, 40, 30) == WHITE and px(img, 5, 5) == RED)
    ctl.set_layer_property(hole, "fill_opacity", 128)
    p = px(shown(ctl), 40, 30)
    check("at Fill 50% the hole shows blue over white -- no red in it",
          p[2] == 255 and 120 < p[0] < 136 and p[0] == p[1], str(p))
    for _ in range(3):              # fill 50%, fill 0, knockout
        ctl.undo()
    check("knockout is undoable", hole.knockout == "none" and px(shown(ctl), 40, 30) == BLUE)

    # ---- shallow vs deep inside a pass-through group -------------------------------------
    ctl = fresh()
    red = solid(ctl, "red", RED)
    group = ctl.add_group("G")
    check("new groups pass through", group.blend == "PassThrough")
    green = solid(ctl, "green", GREEN, Rect(0, 0, 40, 60), parent=group)
    hole = solid(ctl, "hole", BLUE, Rect(20, 20, 40, 20), parent=group)
    hole.fill_opacity = 0
    hole.knockout = "shallow"
    img = shown(ctl)
    check("shallow knockout reaches the bottom of its group: red shows through green",
          px(img, 30, 30) == RED and px(img, 5, 5) == GREEN)
    hole.knockout = "deep"
    img = shown(ctl)
    check("deep knockout reaches the bottom layer: white", px(img, 30, 30) == WHITE
          and px(img, 50, 30) == WHITE and px(img, 5, 5) == GREEN)

    # ---- an isolated group: the hole is in the group's own picture ------------------------
    group.blend = "Normal"
    hole.knockout = "shallow"
    img = shown(ctl)
    check("in an isolated group, a shallow hole shows what is under the group",
          px(img, 30, 30) == RED and px(img, 5, 5) == GREEN)

    # ---- the stroke cache agrees with the full walk -----------------------------------------
    group.blend = "PassThrough"
    hole.knockout = "deep"
    ctl.set_active_layer(green)
    ctl.compositor.build_cache(green)
    dst = np.zeros((60, 80, 4), np.uint8)
    ctl.compositor.composite_cached(dst, ctl.doc.bounds, green)
    check("painting with a knockout in the stack: cached composite == full composite",
          np.array_equal(dst, shown(ctl)))

    # ---- files --------------------------------------------------------------------------
    before = shown(ctl)
    path = os.path.join(tempfile.mkdtemp(), "k.ochre")
    ochre_format.save(ctl.doc, path)
    doc2, _ = ochre_format.load(path)
    h2 = doc2.layer(hole.id)
    from ochre.engine.compositor import Compositor
    check(".ochre keeps fill opacity and knockout", h2.fill_opacity == 0 and h2.knockout == "deep"
          and np.array_equal(Compositor(doc2).composite(doc2.bounds), before))
    hole.knockout = "shallow"
    hole.fill_opacity = 64
    before = shown(ctl)
    path = os.path.join(tempfile.mkdtemp(), "k.psd")
    psd.save(ctl.doc, path)
    doc3 = psd.load(path)
    doc3 = doc3[0] if isinstance(doc3, tuple) else doc3
    h3 = [n for n in doc3.layers() if n.name == "hole"][0]
    check("PSD keeps them (iOpa, knko)", h3.fill_opacity == 64 and h3.knockout == "shallow")
    diff = np.abs(Compositor(doc3).composite(doc3.bounds).astype(int) - before.astype(int)).max()
    check("...and the PSD composites the same", diff <= 1, str(diff))

    # ---- rasterize bakes fill ---------------------------------------------------------------
    ctl = fresh()
    top = solid(ctl, "blue", BLUE, Rect(20, 20, 40, 20))
    ctl.set_layer_property(top, "fill_opacity", 128)
    before = shown(ctl)
    ctl.rasterize_layer(top)
    check("Rasterize bakes Fill into the pixels and resets it",
          top.fill_opacity == 255 and np.abs(shown(ctl).astype(int) - before.astype(int)).max() <= 1)

    # ---- the dialog ------------------------------------------------------------------------
    from PySide6.QtWidgets import QApplication
    from ochre.ui.layerdialogs import LayerStyleDialog
    app = QApplication.instance() or QApplication([])
    ctl = fresh()
    top = solid(ctl, "blue", BLUE, Rect(20, 20, 40, 20))
    n = len(ctl.history_labels())
    dlg = LayerStyleDialog(ctl, top)
    dlg.fill_spin.setValue(0)
    dlg.knockout_box.setCurrentIndex(dlg.knockout_box.findData("deep"))
    check("the Layer Style dialog previews Fill and Knockout live",
          top.fill_opacity == 0 and top.knockout == "deep")
    dlg.reject()
    check("...Cancel puts them back", top.fill_opacity == 255 and top.knockout == "none")
    dlg = LayerStyleDialog(ctl, top)
    dlg.fill_spin.setValue(50)
    dlg.accept()
    check("...OK is one undo step", ctl.history_labels()[n:] == ["Layer Style"]
          and top.fill_opacity == 128)
    del app
    print("\nall knockout checks passed")


if __name__ == "__main__":
    main()
