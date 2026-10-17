#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Geometry, structure and the clipboard -- through the controller.

Driven through the CONTROLLER rather than the engine functions, on
purpose. Twice in this project a test exercised the layer beneath the one a
user touches and passed while the user-facing path was broken; the lesson
recorded in docs/DESIGN.md is to drive the real entry point. Here that is
`Controller.resize_image`, `.merge_down`, `.paste` and so on, each of which
must also leave exactly one undo entry that restores the document bit for
bit.

Run: python3 tests/test_transform.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import transform
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller
from ochre.ui.recent import RecentFiles

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=40, h=30):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=None)
    return ctl


def snapshot(ctl):
    """Everything that defines the document, for exact comparison.

    Authoritative planes only, and an unallocated plane is materialised as
    zeros: a cell nothing has written to and a cell holding all-zero pixels
    are the same document, and the two are reached by different paths (a
    rebuild allocates, an undo of one leaves the plane lazy).
    """
    doc = ctl.doc
    cells = {}
    for k, s in doc.cells.items():
        planes = {}
        for n in s.authoritative:
            arr = s.planes.get(n)
            if arr is None:
                shape = (s.height, s.width) + ((4,) if n == "rgba" else ())
                arr = np.zeros(shape, dtype=np.uint8)
            planes[n] = arr.copy()
        cells[k] = planes
    return (doc.width, doc.height,
            [(l.id, l.name, l.opacity, l.blend, l.visible) for l in doc.root.walk()],
            cells,
            None if ctl.selection.mask is None else ctl.selection.mask.copy())


def same(a, b):
    if a[:3] != b[:3] or set(a[3]) != set(b[3]):
        return False
    for k in a[3]:
        pa, pb = a[3][k], b[3][k]
        if set(pa) != set(pb):
            return False
        if any(not np.array_equal(pa[n], pb[n]) for n in pa):
            return False
    if (a[4] is None) != (b[4] is None):
        return False
    return a[4] is None or np.array_equal(a[4], b[4])


def undo_restores(ctl, before, label):
    """One undo must give back `before` exactly; one redo must re-apply."""
    after = snapshot(ctl)
    ctl.undo()
    check("%s: ONE UNDO RESTORES THE DOCUMENT EXACTLY" % label, same(snapshot(ctl), before))
    ctl.redo()
    check("%s: redo re-applies it" % label, same(snapshot(ctl), after))


def main():
    # ---- alpha-weighted resampling ---------------------------------------
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[2:6, 2:6] = (0, 0, 255, 128)
    small = transform.resample_rgba(rgba, 4, 4, transform.BILINEAR)
    core = small[1:3, 1:3]
    check("RESAMPLING IS ALPHA-WEIGHTED: half-alpha blue stays pure blue",
          bool((core[..., 2] == 255).all()) and bool((core[..., :2] == 0).all()),
          "-- got %s; a dark fringe here is the classic straight-alpha bug" % core[0, 0].tolist())
    up = transform.resample_rgba(rgba, 16, 16, transform.LANCZOS)
    check("upscaling keeps fully transparent pixels transparent",
          int(up[0, 0, 3]) == 0 and int(up[8, 8, 3]) > 0)
    near = transform.resample_nearest(np.arange(16, dtype=np.uint8).reshape(4, 4), 2, 2)
    check("nearest samples pixel CENTRES", near.tolist() == [[5, 7], [13, 15]],
          "-- got %s" % near.tolist())

    # ---- resize, canvas, crop, flip, rotate: each one undo entry ---------
    ctl = fresh()
    a = ctl.active_layer
    ctl.doc.cell(a).fill(Rect(0, 0, 10, 30), (200, 10, 10, 255))
    b = ctl.add_layer("B")
    ctl.doc.cell(b).fill(Rect(30, 0, 10, 10), (10, 10, 200, 255))
    ctl.selection.select_rect((5, 5, 25, 15))   # bounds, not size
    n0 = len(ctl.history)
    before = snapshot(ctl)

    check("resize image changes the canvas", ctl.resize_image(20, 15) and
          (ctl.doc.width, ctl.doc.height) == (20, 15))
    check("...every cell is at the new size",
          all(s.width == 20 and s.height == 15 for s in ctl.doc.cells.values()))
    check("...the display buffer followed", ctl.display.shape[:2] == (15, 20))
    check("...the selection followed", (ctl.selection.width, ctl.selection.height) == (20, 15))
    check("...as exactly one history entry", len(ctl.history) == n0 + 1)
    undo_restores(ctl, before, "resize")
    ctl.undo()

    before = snapshot(ctl)
    ctl.resize_canvas(60, 30, "right")
    cell = ctl.doc.cell(a)
    check("canvas size anchored RIGHT moves content by the growth",
          ctl.doc.width == 60 and cell.plane("rgba")[0, 20, 3] == 255
          and cell.plane("rgba")[0, 0, 3] == 0)
    undo_restores(ctl, before, "canvas size")
    ctl.undo()

    before = snapshot(ctl)
    check("crop to selection", ctl.crop_to_selection() and (ctl.doc.width, ctl.doc.height) == (20, 10))
    check("...clears the selection, which no longer means anything", ctl.selection.selects_all())
    undo_restores(ctl, before, "crop")
    ctl.undo()
    check("...and undo brings the selection back", ctl.selection.bbox == Rect(5, 5, 20, 10))

    before = snapshot(ctl)
    ctl.flip_image(True)
    check("flip horizontal mirrors content", ctl.doc.cell(a).plane("rgba")[0, 39, 3] == 255
          and ctl.doc.cell(a).plane("rgba")[0, 0, 3] == 0)
    undo_restores(ctl, before, "flip")
    ctl.undo()

    before = snapshot(ctl)
    ctl.rotate_image(1)
    check("rotate 90 swaps the dimensions", (ctl.doc.width, ctl.doc.height) == (30, 40))
    # Layer A was the left 10 columns; clockwise that becomes the TOP 10 rows.
    check("...and content goes where a clockwise turn puts it",
          ctl.doc.cell(a).plane("rgba")[0, 15, 3] == 255 and ctl.doc.cell(a).plane("rgba")[20, 15, 3] == 0)
    undo_restores(ctl, before, "rotate")
    ctl.undo()
    # A rotation changes the canvas shape, and a selection mask drawn on the
    # old shape means nothing on the new one, so it is dropped (undo brings
    # it back, as checked above). Deselect first so the comparison is about
    # pixels.
    ctl.selection.select_all()
    ctl.emit("selection.changed")
    before = snapshot(ctl)
    check("four quarter turns are the identity",
          all([ctl.rotate_image(1) for _ in range(4)]) and same(snapshot(ctl), before))

    # ---- layer-level geometry --------------------------------------------
    ctl = fresh()
    a = ctl.active_layer
    ctl.doc.cell(a).fill(Rect(0, 0, 10, 30), (200, 10, 10, 255))
    before = snapshot(ctl)
    ctl.flip_layer(True)
    check("flip layer only touches that layer", ctl.doc.cell(a).plane("rgba")[0, 39, 3] == 255)
    undo_restores(ctl, before, "flip layer")

    # ---- structure: add, delete, duplicate, merge, flatten ---------------
    ctl = fresh()
    a = ctl.active_layer
    ctl.doc.cell(a).fill(ctl.doc.bounds, (255, 0, 0, 255))
    before = snapshot(ctl)
    b = ctl.add_layer("B")
    check("ADD LAYER IS UNDOABLE", len(ctl.history) == 1)
    ctl.undo()
    check("...and undo removes it", [l.name for l in ctl.doc.layers()] == ["Background"])
    ctl.redo()
    check("...redo brings it back", [l.name for l in ctl.doc.layers()] == ["Background", "B"])
    b = ctl.doc.layers()[-1]
    ctl.set_active_layer(b)
    ctl.doc.cell(b).fill(Rect(10, 5, 8, 6), (0, 0, 255, 128))

    before = snapshot(ctl)
    ctl.remove_layer(b)
    check("DELETE LAYER IS UNDOABLE", [l.name for l in ctl.doc.layers()] == ["Background"])
    ctl.undo()
    check("...undo restores the layer AND its pixels", same(snapshot(ctl), before))
    b = ctl.doc.layers()[-1]

    before = snapshot(ctl)
    ctl.set_active_layer(b)
    copy = ctl.duplicate_layer()
    check("duplicate is undoable", copy is not None and len(ctl.doc.layers()) == 3)
    ctl.undo()
    check("...undo removes the copy", same(snapshot(ctl), before))

    before = snapshot(ctl)
    ctl.set_active_layer(b)
    check("merge down composites onto the layer below", ctl.merge_down()
          and [l.name for l in ctl.doc.layers()] == ["Background"])
    px = ctl.doc.cell(a).plane("rgba")[7, 12]
    check("...with the upper layer's alpha honoured (half blue over red)",
          px[0] in (127, 128) and px[2] in (127, 128) and px[3] == 255, "-- got %s" % px.tolist())
    undo_restores(ctl, before, "merge down")
    ctl.undo()

    ctl.add_layer("C")
    ctl.doc.cell(ctl.active_layer).fill(Rect(0, 0, 5, 5), (0, 255, 0, 255))
    before = snapshot(ctl)
    check("flatten leaves one layer", ctl.flatten() and len(ctl.doc.layers()) == 1)
    check("...holding the composite", ctl.doc.cell(ctl.doc.layers()[0]).plane("rgba")[2, 2].tolist() == [0, 255, 0, 255])
    undo_restores(ctl, before, "flatten")

    # Merge into a group is refused, not guessed at.
    ctl = fresh()
    g = ctl.add_group("G")
    top = ctl.add_layer("Top", parent=ctl.doc.root, index=len(ctl.doc.root.children))
    ctl.doc.cell(top).fill(Rect(0, 0, 4, 4), (1, 2, 3, 255))
    ctl.set_active_layer(top)
    check("merge down refuses when the layer below is a group",
          transform.layer_below(ctl.doc, top) is None and not ctl.merge_down())

    # ---- frames -----------------------------------------------------------
    ctl = fresh()
    before = snapshot(ctl)
    ctl.add_frame()
    check("add frame is undoable", len(ctl.doc.frames) == 2 and len(ctl.history) == 1)
    ctl.undo()
    check("...undo removes it", len(ctl.doc.frames) == 1)
    ctl.redo()
    ctl.set_current_frame(1)
    ctl.doc.cell(ctl.active_layer).fill(Rect(0, 0, 3, 3), (7, 7, 7, 255))
    check("delete frame", ctl.remove_frame() and len(ctl.doc.frames) == 1)
    ctl.undo()
    check("...undo restores the frame with its pixels",
          len(ctl.doc.frames) == 2
          and ctl.doc.cells[(ctl.active_layer.id, ctl.doc.frames[1].id)].plane("rgba")[0, 0, 0] == 7)

    # ---- clipboard --------------------------------------------------------
    ctl = fresh()
    a = ctl.active_layer
    ctl.doc.cell(a).fill(Rect(2, 2, 4, 4), (9, 8, 7, 255))
    ctl.selection.select_rect((2, 2, 6, 6))
    clip = ctl.copy()
    check("copy lifts the selection", clip is not None and (clip.width, clip.height) == (4, 4)
          and clip.rgba[0, 0].tolist() == [9, 8, 7, 255])
    ctl.selection.select_all()
    before = snapshot(ctl)
    landed = ctl.paste(at=(20, 10))
    check("paste lands where asked", landed == Rect(20, 10, 4, 4)
          and ctl.doc.cell(a).plane("rgba")[10, 20].tolist() == [9, 8, 7, 255])
    check("...and selects what it pasted", ctl.selection.bbox == Rect(20, 10, 4, 4))
    check("...and hands over to Move Selected Pixels", ctl.tool_name == "move_pixels")
    ctl.selection.select_all()
    undo_restores(ctl, before, "paste")
    ctl.undo()

    before = snapshot(ctl)
    ctl.paste(at=(0, 0), new_layer=True)
    check("paste into new layer adds a layer", len(ctl.doc.layers()) == 2
          and ctl.doc.cell(ctl.doc.layers()[-1]).plane("rgba")[0, 0, 0] == 9)
    ctl.selection.select_all()
    check("...as ONE undo entry for layer and pixels together",
          (ctl.undo(), same(snapshot(ctl), before))[1])

    ctl.selection.select_rect((2, 2, 6, 6))
    before = snapshot(ctl)
    ctl.cut()
    check("cut erases the source", ctl.doc.cell(a).plane("rgba")[3, 3, 3] == 0)
    ctl.undo()
    check("...undoably", same(snapshot(ctl), before))

    before = snapshot(ctl)
    ctl.fill_selection((1, 2, 3, 255))
    check("fill selection paints the selected rect",
          ctl.doc.cell(a).plane("rgba")[3, 3].tolist() == [1, 2, 3, 255]
          and ctl.doc.cell(a).plane("rgba")[0, 0, 3] == 0)
    ctl.undo()
    check("...undoably", same(snapshot(ctl), before))
    ctl.erase_selection()
    check("erase selection clears it", ctl.doc.cell(a).plane("rgba")[3, 3, 3] == 0)

    new = ctl.paste_new_image()
    check("paste into new image sizes the document to the clip",
          (new.width, new.height) == (4, 4) and ctl.doc is new)

    # Indices travel as indices between documents sharing a palette.
    from ochre.engine.palette import grayscale
    from ochre.engine.document import Document
    pal = grayscale()
    ctl = fresh()
    ctl._adopt(Document(10, 10, palette=pal), path=None)
    lay = ctl.add_layer("i", planes=("index", "rgba"), authoritative=("index",))
    ctl.doc.cell(lay).plane("index")[2:5, 2:5] = 77
    ctl.doc.cell(lay).refresh_derived()
    ctl.selection.select_rect((2, 2, 5, 5))
    clip = ctl.copy()
    check("copying from an indexed layer keeps the INDICES", clip.index is not None
          and int(clip.index[0, 0]) == 77)
    ctl.selection.select_all()
    ctl.paste(at=(6, 6))
    check("...and pasting writes them back verbatim",
          int(ctl.doc.cell(lay).plane("index")[6, 6]) == 77)

    # ---- recent files ------------------------------------------------------
    state = tempfile.mkdtemp()
    recent = RecentFiles(state, limit=3)
    for name in ("a", "b", "c", "d"):
        path = os.path.join(state, name + ".ochre")
        open(path, "w").close()
        recent.add(path)
    check("recent files keeps the newest first, bounded",
          [os.path.basename(p) for p in recent.items] == ["d.ochre", "c.ochre", "b.ochre"])
    recent.add(os.path.join(state, "b.ochre"))
    check("re-opening moves a file to the front, no duplicate",
          [os.path.basename(p) for p in recent.items] == ["b.ochre", "d.ochre", "c.ochre"])
    os.remove(os.path.join(state, "d.ochre"))
    check("existing() hides what is gone without forgetting it",
          [os.path.basename(p) for p in recent.existing()] == ["b.ochre", "c.ochre"]
          and len(recent.items) == 3)
    check("the list survives a reload", RecentFiles(state, limit=3).items == recent.items)

    print("\nall transform checks passed")


if __name__ == "__main__":
    main()
