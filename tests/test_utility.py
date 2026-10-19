#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Guides, snapping, marks (samplers, counts, notes, slices), measuring and
straightening, the histogram, and several open documents.

Run: python3 tests/test_utility.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import annotations as A
from ochre.engine.fileio import ochre_format
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ALT, SHIFT = 4, 1


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=200, h=150, background=None):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=background)
    return ctl


def click(ctl, x, y, mods=0):
    ctl.begin_stroke(x, y, 1.0, mods)
    ctl.end_stroke(x, y, 1.0, mods)


def drag(ctl, a, b, mods=0):
    ctl.begin_stroke(a[0], a[1], 1.0, mods)
    ctl.motion_stroke((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, 1.0, mods)
    ctl.end_stroke(b[0], b[1], 1.0, mods)


def main():
    # ---- engine helpers ------------------------------------------------------
    x, y, sx, sy = A.snap(31.5, 18.0, [("v", 30.0), ("h", 20.0)], tolerance=3)
    check("snap pulls each axis to its nearest guide", (x, y, sx, sy) == (30.0, 20.0, True, True))
    x, y, sx, sy = A.snap(37.0, 9.0, [], grid=(16, 16), tolerance=3)
    check("...or to the grid, and only within tolerance", (x, y) == (37.0, 9.0) and not sx)
    x, y, _a, _b = A.snap(33.0, 1.0, [], grid=(16, 16), bounds=(100, 80), tolerance=3)
    check("grid and canvas edges both count", (x, y) == (32.0, 0.0))
    check("slices from guides cut the canvas into cells",
          len(A.slices_from_guides([("v", 50), ("h", 40)], 100, 80)) == 4)
    img = np.zeros((10, 10, 4), np.uint8)
    img[:, :5] = (255, 0, 0, 255)
    check("an averaged sample weights colour by alpha (no dark edge)",
          A.sample(img, 5, 5, 3)[:3] == (255, 0, 0))
    check("straighten finds the angle to level a tilted line",
          abs(A.straighten_angle(0, 0, 100, 10) + math.degrees(math.atan(0.1))) < 1e-9)
    check("...and to plumb a nearly vertical one",
          abs(A.straighten_angle(0, 0, 5, 100) - math.degrees(math.atan2(5, 100))) < 1e-9)

    # ---- guides, snapping ------------------------------------------------------
    ctl = fresh()
    ctl.add_guide("v", 50)
    ctl.add_guide("h", 40)
    check("guides are added", ctl.doc.guides == [("v", 50.0), ("h", 40.0)])
    ctl.undo()
    check("...undoably", ctl.doc.guides == [("v", 50.0)])
    ctl.redo()
    ctl.set_tool("select_rect")
    ctl.pick_radius = 2.0
    drag(ctl, (10, 10), (52.5, 41.5))
    b = ctl.selection.bbox
    check("a marquee drawn near guides SNAPS to them", (b.x1, b.y1) == (50, 40),
          "(bbox %s)" % (b,))
    ctl.snap_guides = False
    drag(ctl, (10, 10), (52.5, 41.5))
    check("...and does not when snapping is off", ctl.selection.bbox.x1 != 50)
    ctl.snap_guides = True
    ctl.deselect()
    ctl.set_tool("brush")
    before = ctl.doc.cell(ctl.active_layer).plane("rgba").copy()
    drag(ctl, (51.5, 80), (51.5, 90))
    painted = np.nonzero(ctl.doc.cell(ctl.active_layer).plane("rgba")[85, :, 3])[0]
    check("freehand painting is never snapped", painted.size and abs(painted.mean() - 51.5) < 2)
    ctl.move_guide(0, 70)
    check("a guide can be moved", ctl.doc.guides[0] == ("v", 70.0))
    ctl.move_guide(0, -5)
    check("...and dragging it off the canvas removes it", ctl.doc.guides == [("h", 40.0)])
    check("guide_at finds a guide near a point", ctl.guide_at(100, 41, 3) == 0
          and ctl.guide_at(100, 60, 3) is None)
    ctl.clear_guides()
    check("Clear Guides", ctl.doc.guides == [])

    # ---- Count, Color Sampler ----------------------------------------------------
    ctl = fresh()
    ctl.set_tool("count")
    for p in ((10, 10), (30, 30), (50, 50)):
        click(ctl, *p)
    check("the count tool drops numbered markers", len(ctl.doc.counts) == 3)
    drag(ctl, (30, 30), (35, 40))
    check("...which drag", ctl.doc.counts[1] == (35.0, 40.0))
    click(ctl, 10, 10, ALT)
    check("...and Alt-click removes", len(ctl.doc.counts) == 2 and ctl.doc.counts[0] == (35.0, 40.0))
    n = len(ctl.history_labels())
    ctl.undo()
    check("each is one undo step", len(ctl.doc.counts) == 3 and n >= 5)
    ctl.set_tool("color_sampler")
    for i in range(12):
        click(ctl, 10 + 12 * i, 100)
    check("color samplers stop at ten, as in Photoshop", len(ctl.doc.samplers) == 10)

    # ---- eyedropper sampling -----------------------------------------------------
    ctl = fresh(40, 40, background=(255, 255, 255, 255))
    top = ctl.add_layer("top")
    ctl.doc.cell(top).fill(Rect(0, 0, 20, 40), (0, 0, 255, 255))
    ctl.set_active_layer(ctl.doc.layers()[0])
    ctl.set_tool("eyedropper", sample_from="image")
    click(ctl, 5, 5)
    check("the color picker can sample the merged image", ctl.primary[:3] == (0, 0, 255))
    ctl.set_tool("eyedropper", sample_from="layer")
    click(ctl, 5, 5)
    check("...or just the active layer", ctl.primary[:3] == (255, 255, 255))
    ctl.set_active_layer(top)
    ctl.set_tool("eyedropper", sample_from="image", sample_size="3")
    click(ctl, 20, 5)
    check("a 3x3 sample averages across an edge", ctl.primary[:3] == (170, 170, 255),
          str(ctl.primary))

    # ---- notes -----------------------------------------------------------------------
    ctl = fresh()
    seen = []
    ctl.bus = type("Bus", (), {"emit": lambda self, topic, **kw: seen.append((topic, kw))})()
    ctl.set_tool("note")
    click(ctl, 40, 40)
    req = [kw["request"] for t, kw in seen if t == "tool.request"]
    check("clicking with the note tool asks the window for text, adding nothing yet",
          req == [("new_note", (40.0, 40.0))] and ctl.doc.notes == [])
    ctl.add_note(40, 40, "fix this")
    check("the note is added with its text", ctl.doc.notes == [(40.0, 40.0, "fix this")])
    seen.clear()
    click(ctl, 40.5, 40.5)
    req = [kw["request"] for t, kw in seen if t == "tool.request"]
    check("clicking a note asks to edit it", req == [("edit_note", 0)])
    ctl.set_note_text(0, "fixed")
    drag(ctl, (40, 40), (60, 50))
    check("notes edit and drag", ctl.doc.notes == [(60.0, 50.0, "fixed")])

    # ---- slices ------------------------------------------------------------------------
    ctl = fresh(100, 80, background=(10, 200, 30, 255))
    ctl.set_tool("slice")
    drag(ctl, (10, 10), (40, 30))
    check("the slice tool makes a slice from a drag",
          ctl.doc.slices == [("slice_01", (10, 10, 30, 20))])
    drag(ctl, (20, 20), (30, 25))
    check("...dragging inside moves it", ctl.doc.slices[0][1] == (20, 15, 30, 20))
    ctl.set_tool("slice_select")
    drag(ctl, (60, 60), (90, 75))
    check("slice select never makes a new slice", len(ctl.doc.slices) == 1)
    ctl.add_guide("v", 50)
    ctl.slices_from_guides()
    check("Slices From Guides replaces them with the guide cells", len(ctl.doc.slices) == 2)
    out = ctl.export_slices(tempfile.mkdtemp())
    from PIL import Image
    sizes = sorted(Image.open(p).size for p in out)
    check("Export Slices writes one image per slice, the right size",
          len(out) == 2 and sizes == [(50, 80), (50, 80)])
    check("...holding the merged pixels",
          Image.open(out[0]).convert("RGBA").getpixel((5, 5)) == (10, 200, 30, 255))

    # ---- measure and straighten -------------------------------------------------------
    ctl = fresh(200, 150, background=(255, 255, 255, 255))
    cell = ctl.doc.cell(ctl.active_layer)
    ang = math.radians(8.0)
    for t in np.linspace(0, 140, 600):
        px, py = 30 + t * math.cos(ang), 40 + t * math.sin(ang)
        cell.plane("rgba")[int(py), int(px)] = (0, 0, 0, 255)
    cell._touched(ctl.doc.bounds)
    ctl.set_tool("measure")
    drag(ctl, (30, 40), (30 + 140 * math.cos(ang), 40 + 140 * math.sin(ang)))
    m = ctl.tool.measurement
    check("the measure tool reports distance and angle",
          abs(m["distance"] - 140) < 0.01 and abs(m["angle"] + 8.0) < 0.01)
    check("...and offers a status line", "140.0 px" in ctl.tool.status())
    ctl.straighten()
    rgba = ctl.doc.cell(ctl.active_layer).plane("rgba")
    ys, xs = np.nonzero((rgba[..., 3] > 0) & (rgba[..., 0] < 100))
    slope = np.polyfit(xs, ys, 1)[0]
    check("Straighten rotates the image so the measured line is level (slope %.4f)" % slope,
          abs(slope) < 0.01)
    ctl.undo()
    check("...as one undo step", ctl.doc.width == 200)

    # ---- histogram ---------------------------------------------------------------------
    ctl = fresh(100, 100, background=(0, 0, 0, 255))
    ctl.doc.cell(ctl.active_layer).fill(Rect(0, 0, 50, 100), (255, 255, 255, 255))
    h = ctl.histogram()
    check("the histogram counts the image's levels",
          h["value"][0] == 5000 and h["value"][255] == 5000 and h["pixels"] == 10000)
    check("...with statistics", abs(h["mean"] - 127.5) < 1e-6 and h["median"] in (0, 255))
    ctl.selection.select_rect((0, 0, 50, 100))
    h = ctl.histogram()
    check("...over the selection when there is one", h["value"][255] == 5000
          and h["value"][0] == 0)

    # ---- marks follow the image and are saved -------------------------------------------
    ctl = fresh(100, 80)
    ctl.add_guide("h", 20)
    ctl.doc.counts = [(10.0, 20.0)]
    ctl.doc.slices = [("a", (0, 0, 50, 40))]
    ctl.resize_image(200, 160)
    check("resizing scales guides, marks and slices",
          ctl.doc.guides == [("h", 40.0)] and ctl.doc.counts == [(20.0, 40.0)]
          and ctl.doc.slices == [("a", (0, 0, 100, 80))])
    ctl.undo()
    check("...and undo restores them", ctl.doc.guides == [("h", 20.0)]
          and ctl.doc.counts == [(10.0, 20.0)])
    ctl.doc.notes = [(1.0, 2.0, "a, b\nc")]
    ctl.doc.samplers = [(3.0, 4.0)]
    f = os.path.join(tempfile.mkdtemp(), "marks.ochre")
    ochre_format.save(ctl.doc, f)
    d2, _ = ochre_format.load(f)
    check("guides and every mark survive save and load",
          (d2.guides, d2.counts, d2.notes, d2.samplers, d2.slices)
          == (ctl.doc.guides, ctl.doc.counts, ctl.doc.notes, ctl.doc.samplers, ctl.doc.slices))

    # ---- several documents --------------------------------------------------------------
    ctl = fresh(64, 48)
    first = ctl.doc
    ctl.new_document(32, 32)
    check("a new document REPLACES an untouched Untitled",
          len(ctl.documents) == 1 and ctl.doc is not first)
    ctl.set_tool("pencil")
    drag(ctl, (2, 2), (20, 20))
    a = ctl.doc
    ctl.new_document(50, 40)
    check("...but opens beside one with work in it", len(ctl.documents) == 2
          and ctl.current_doc == 1 and ctl.doc is not a)
    ctl.selection.select_rect((0, 0, 10, 10))
    check("titles mark unsaved work", ctl.document_titles() == ["*Untitled", "Untitled"])
    check("modified_documents lists only the edited one", ctl.modified_documents() == [0])
    ctl.switch_document(0)
    check("switching brings back that document, its history and its selection",
          ctl.doc is a and ctl.history_labels()[-1] == "Pencil" and ctl.selection.mask is None)
    ctl.undo()
    ctl.switch_document(1)
    check("...each document keeps its own", ctl.doc.width == 50
          and ctl.selection.mask is not None and not ctl.history_labels()[:1] == ["Pencil"])
    ctl.close_document()
    check("closing a document leaves the others", len(ctl.documents) == 1 and ctl.doc is a)
    ctl.close_document()
    check("closing the last leaves a fresh canvas to draw on",
          len(ctl.documents) == 1 and ctl.doc is not a and not ctl.is_modified())

    print("\nall utility checks passed")


if __name__ == "__main__":
    main()
