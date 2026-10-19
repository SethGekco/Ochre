#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Non-destructive filters on pixel layers (GIMP 3), the filter-list editor,
and linked smart objects.

Run: python3 tests/test_ndefilters.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image

from ochre.engine.compositor import Compositor
from ochre.engine.fileio import ochre_format
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(state=None):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state or tempfile.mkdtemp())
    ctl.new_document(64, 48, background=None)
    lay = ctl.active_layer
    rng = np.random.default_rng(0)
    px = ctl.doc.cell(lay).plane("rgba")
    px[..., :3] = rng.integers(0, 256, (48, 64, 3))
    px[..., 3] = 255
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    return ctl, lay


def shown(ctl):
    ctl.compositor.invalidate()
    return ctl.compositor.composite(ctl.doc.bounds)


def main():
    state = tempfile.mkdtemp()
    ctl, lay = fresh(state)
    orig = ctl.doc.cell(lay).plane("rgba").copy()
    ctl.apply_effect("invert")
    check("by default an effect is baked into the pixels",
          not np.array_equal(ctl.doc.cell(lay).plane("rgba"), orig) and not lay.params.get("filters"))
    ctl.undo()

    ctl.set_non_destructive(True)
    ctl.apply_effect("invert")
    img = shown(ctl)
    check("non-destructively, the pixels stay as they were...",
          np.array_equal(ctl.doc.cell(lay).plane("rgba"), orig))
    check("...the filter is listed on the layer...", [f["name"] for f in lay.params["filters"]] == ["invert"])
    check("...and the image shows it", np.array_equal(img[..., :3], 255 - orig[..., :3]))
    ctl.apply_effect("brightness_contrast", {"brightness": 40})
    a = shown(ctl).copy()
    filters = ctl.layer_filters()
    ctl.set_smart_filters(list(reversed(filters)))
    check("the order matters, and can be changed", not np.array_equal(shown(ctl), a))
    ctl.set_smart_filters(filters)
    off = [dict(filters[0], enabled=False), filters[1]]
    ctl.set_smart_filters(off)
    check("a filter can be switched off without removing it",
          np.array_equal(shown(ctl), ctl.effects.create("brightness_contrast", brightness=40).apply(orig)))
    ctl.set_smart_filters(filters)

    # painting stays live under the filters
    ctl.set_primary((0, 0, 0, 255))
    ctl.set_tool("pencil", size=4)
    ctl.begin_stroke(10, 10)
    ctl.end_stroke(10, 10)
    p = shown(ctl)[10, 10]
    want = ctl.effects.create("brightness_contrast", brightness=40).apply(
        np.full((1, 1, 4), (255, 255, 255, 255), np.uint8))[0, 0]
    check("painting on the layer shows through its filters (black, inverted, brightened)",
          tuple(p) == tuple(want), "%s vs %s" % (tuple(p), tuple(want)))
    ctl.undo()

    f = os.path.join(tempfile.mkdtemp(), "n.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    check(".ochre keeps the filter list", np.array_equal(Compositor(doc2).composite(doc2.bounds), shown(ctl))
          and len(doc2.layer(lay.id).params["filters"]) == 2)

    before = shown(ctl).copy()
    ctl.merge_layer_filters()
    check("Merge Filters bakes them in and empties the list, same picture, one step",
          not lay.params.get("filters") and np.array_equal(shown(ctl), before)
          and ctl.history_labels()[-1] == "Merge Filters")
    ctl.undo()
    check("...undoable", len(lay.params["filters"]) == 2
          and np.array_equal(ctl.doc.cell(lay).plane("rgba"), orig))

    ctl.selection.select_rect((0, 0, 20, 20))
    ctl.apply_effect("invert")
    check("with a selection, the filter is applied for good (a list entry cannot hold a selection)",
          len(lay.params["filters"]) == 2 and not np.array_equal(ctl.doc.cell(lay).plane("rgba"), orig))
    ctl.undo()
    ctl.selection.select_all()

    ctl2, _ = fresh(state)
    check("the non-destructive setting is remembered", ctl2.non_destructive is True)

    # ---- the editor ---------------------------------------------------------------------
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from ochre.ui.filterlist import LayerFiltersDialog
    app = QApplication.instance() or QApplication([])
    start = shown(ctl).copy()
    dlg = LayerFiltersDialog(ctl, lay, ctl.effects)
    dlg.list.item(0).setCheckState(Qt.Unchecked)
    check("the filter editor previews live", not np.array_equal(shown(ctl), start))
    dlg.reject()
    check("...Cancel puts it all back", np.array_equal(shown(ctl), start)
          and all(f.get("enabled", True) for f in lay.params["filters"]))
    pos = ctl.history.position          # (labels also list redo-able steps)
    dlg = LayerFiltersDialog(ctl, lay, ctl.effects)
    dlg.list.setCurrentRow(1)
    dlg.delete()
    dlg.accept()
    check("...OK is one undo step", ctl.history.position == pos + 1
          and [f["name"] for f in lay.params["filters"]] == ["invert"])
    del app

    # ---- linked smart objects ----------------------------------------------------------------
    d = tempfile.mkdtemp()
    path = os.path.join(d, "logo.png")
    Image.fromarray(np.full((20, 30, 4), (255, 0, 0, 255), np.uint8)).save(path)
    ctl, _ = fresh()
    placed = ctl.place_linked(path)
    check("Place Linked makes a smart object that remembers its file",
          placed.kind == "smart" and placed.params["link"] == os.path.abspath(path)
          and ctl.linked_status() == [(placed, "ok")])
    Image.fromarray(np.full((20, 30, 4), (0, 0, 255, 255), np.uint8)).save(path)
    t = time.time() + 5
    os.utime(path, (t, t))
    check("...notices when the file changes", ctl.linked_status() == [(placed, "modified")])
    red = shown(ctl).copy()
    check("Update Modified Content re-reads it", ctl.update_linked() == 1
          and shown(ctl)[24, 32, 2] == 255 and ctl.linked_status() == [(placed, "ok")])
    ctl.undo()
    check("...undoable", np.array_equal(shown(ctl), red))
    os.remove(path)
    check("a missing file is reported, and the layer still renders",
          ctl.linked_status() == [(placed, "missing")] and np.array_equal(shown(ctl), red))
    other = os.path.join(d, "other.png")
    Image.fromarray(np.full((20, 30, 4), (0, 255, 0, 255), np.uint8)).save(other)
    ctl.relink(other)
    check("Relink to File points it at another file",
          placed.params["link"] == os.path.abspath(other) and shown(ctl)[24, 32, 1] == 255)
    print("\nall non-destructive filter checks passed")


if __name__ == "__main__":
    main()
