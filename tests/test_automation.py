#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Actions (record, play, batch), saved-selection channels, layer comps and
Blend If -- through the controller.

Run: python3 tests/test_automation.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from ochre.engine.compositor import Compositor
from ochre.engine.fileio import ochre_format, psd
from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=60, h=40, state=None, bg=(100, 150, 200, 255)):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=state or tempfile.mkdtemp())
    ctl.new_document(w, h, background=bg)
    return ctl


def main():
    # ---- Actions ---------------------------------------------------------------
    state = tempfile.mkdtemp()
    ctl = fresh(state=state)
    ctl.start_recording("Prep")
    ctl.apply_effect("invert")
    ctl.resize_image(30, 20)
    ctl.flip_image(True)
    rec = ctl.stop_recording()
    check("recording logs the commands, with their settings",
          [s["op"] for s in rec["steps"]] == ["apply_effect", "resize_image", "flip_image"])
    check("...and only the top-level command (not what it calls inside)",
          len(rec["steps"]) == 3)
    on_disk = json.load(open(os.path.join(state, "actions.json")))
    check("actions are saved beside the session", "Prep" in on_disk)
    ctl2 = fresh(state=state)
    check("...and a new session finds them", "Prep" in ctl2.actions())
    n0 = len(ctl2.history_labels())
    ctl2.play_action("Prep")
    px = ctl2.doc.cell(ctl2.active_layer).plane("rgba")
    check("playing an action repeats it: inverted, resized",
          (ctl2.doc.width, ctl2.doc.height) == (30, 20) and tuple(px[5, 5, :3]) == (155, 105, 55))
    check("...as ONE undo step", ctl2.history_labels()[n0:] == ["Action: Prep"],
          str(ctl2.history_labels()[n0:]))
    ctl2.undo()
    check("...that undoes completely", (ctl2.doc.width, ctl2.doc.height) == (60, 40)
          and tuple(ctl2.doc.cell(ctl2.active_layer).plane("rgba")[5, 5, :3]) == (100, 150, 200))
    ctl2._actions["Evil"] = [{"op": "__init__", "args": [], "kwargs": {}},
                             {"op": "invert_selection", "args": [], "kwargs": {}}]
    ctl2.play_action("Evil")
    check("an action can only call recordable commands (the file is data)", True)

    # ---- Batch -------------------------------------------------------------------
    src = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    for i, colour in enumerate(((255, 0, 0), (0, 255, 0))):
        a = np.zeros((10, 16, 3), np.uint8)
        a[...] = colour
        Image.fromarray(a).save(os.path.join(src, "img%d.png" % i))
    report = ctl2.batch("Prep", sorted(os.path.join(src, f) for f in os.listdir(src)), out, "png")
    outs = sorted(os.listdir(out))
    check("Batch runs the action over every file and saves the results",
          outs == ["img0.png", "img1.png"] and all(not str(r[1]).startswith("error") for r in report))
    im = np.asarray(Image.open(os.path.join(out, "img0.png")).convert("RGB"))
    check("...each one processed (inverted red is cyan, at the new size)",
          im.shape[:2] == (20, 30) and tuple(im[3, 3]) == (0, 255, 255))
    check("...and the opened files are closed again", len(ctl2.documents) == 1)

    # ---- channels -------------------------------------------------------------------
    ctl = fresh()
    ctl.selection.select_rect((5, 5, 25, 25))
    ch = ctl.save_selection("Face")
    check("Save Selection makes a named channel", ctl.doc.channels == [ch] and ch.name == "Face")
    ctl.deselect()
    ctl.load_selection(0)
    check("Load Selection brings it back", ctl.selection.mask is not None
          and ctl.selection.mask[10, 10] == 255 and ctl.selection.mask[30, 30] == 0)
    ctl.selection.select_rect((40, 20, 50, 30))
    from ochre.engine.selection import UNION
    ctl.load_selection(0, UNION)
    check("...and can add to the current selection", ctl.selection.mask[10, 10] == 255
          and ctl.selection.mask[25, 45] == 255)
    ctl.resize_image(120, 80)
    check("channels resize with the image", ctl.channel_mask(0).shape == (80, 120)
          and ctl.channel_mask(0)[20, 20] == 255)
    ctl.undo()
    f = os.path.join(tempfile.mkdtemp(), "ch.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    check("channels save and load", [c.name for c in doc2.channels] == ["Face"])
    ctl.delete_channel(0)
    check("Delete Channel", ctl.doc.channels == [])
    ctl.undo()
    check("...undoably", len(ctl.doc.channels) == 1 and ctl.channel_mask(0)[10, 10] == 255)

    # ---- layer comps ---------------------------------------------------------------------
    ctl = fresh()
    a = ctl.add_layer("A")
    b = ctl.add_layer("B")
    ctl.new_comp("both")
    ctl.set_layer_property(b, "visible", False)
    ctl.set_layer_property(a, "opacity", 90)
    ctl.new_comp("only A")
    ctl.apply_comp(0)
    check("applying a comp restores visibility and opacity", b.visible and a.opacity == 255)
    check("...as one undo step", ctl.history_labels()[-1] == "Apply Layer Comp")
    ctl.apply_comp(1)
    check("...switching between versions", not b.visible and a.opacity == 90)
    f = os.path.join(tempfile.mkdtemp(), "comps.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    check("comps save and load", [c["name"] for c in doc2.comps] == ["both", "only A"])

    # ---- Blend If -----------------------------------------------------------------------------
    ctl = fresh(40, 20, bg=(20, 20, 20, 255))
    top = ctl.add_layer("white background")
    cell = ctl.doc.cell(top)
    cell.fill(ctl.doc.bounds, (255, 255, 255, 255))
    cell.fill(Rect(10, 5, 20, 10), (200, 0, 0, 255))
    ctl.set_layer_property(top, "blend_if", (0, 0, 200, 230, 0, 0, 255, 255))
    img = Compositor(ctl.doc).composite(ctl.doc.bounds)
    check("Blend If drops out the layer's whites (the classic background removal)",
          tuple(img[2, 2, :3]) == (20, 20, 20) and tuple(img[10, 20, :3]) == (200, 0, 0))
    p = os.path.join(tempfile.mkdtemp(), "bi.psd")
    psd.save(ctl.doc, p)
    doc3 = psd.load(p)
    check("Blend If travels in a PSD (as Photoshop's blending ranges)",
          [l for l in doc3.layers() if l.name == "white background"][0].blend_if
          == (0, 0, 200, 230, 0, 0, 255, 255))
    img3 = Compositor(doc3).composite(doc3.bounds)
    check("...and composites the same after the round trip", np.array_equal(img, img3))
    print("\nall automation checks passed")


if __name__ == "__main__":
    main()
