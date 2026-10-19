#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Text layers: text that stays editable, through the controller.

Run: python3 tests/test_textlayers.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import textlayer
from ochre.engine.fileio import ochre_format
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WHITE = (255, 255, 255, 255)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def fresh(w=240, h=160):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(w, h, background=WHITE)
    return ctl


def shown(ctl):
    ctl.compositor.invalidate()
    return ctl.compositor.composite(ctl.doc.bounds)


def type_text(ctl, s):
    for ch in s:
        ctl.tool.insert(ctl._ctx, ch)


def click(ctl, x, y):
    ctl.begin_stroke(x, y)
    ctl.end_stroke(x, y)


def ink(img, box=None):
    """How many pixels are not white (in an optional (x0, y0, x1, y1) box)."""
    if box is not None:
        img = img[box[1]:box[3], box[0]:box[2]]
    return int((img[..., :3].min(axis=-1) < 200).sum())


def main():
    # ---- create, type, commit, undo ---------------------------------------------
    ctl = fresh()
    bg = ctl.active_layer
    bg_before = ctl.doc.cell(bg).plane("rgba").copy()
    ctl.set_tool("text", size=32)
    ctl.set_primary((200, 0, 0, 255))
    n0 = len(ctl.history_labels())
    click(ctl, 20, 20)
    lay = ctl.active_layer
    check("a click with the Text tool starts a new TEXT layer", lay is not bg
          and lay.kind == "text" and ctl.tool.active)
    check("...which is not in history yet (an empty click must leave no trace)",
          len(ctl.history_labels()) == n0)
    type_text(ctl, "Hello")
    check("typing renders the text live", ink(shown(ctl)) > 100)
    ctl.finish_tool()
    check("committing records the layer AND its text as ONE undo step",
          ctl.history_labels()[n0:] == ["Text"], str(ctl.history_labels()[n0:]))
    check("...names the layer after its text", lay.name == "Hello")
    check("...and never touches the layer beneath",
          np.array_equal(ctl.doc.cell(bg).plane("rgba"), bg_before))
    img = shown(ctl)
    solid = img[..., 1] == 0                 # fully inked pixels of the text
    check("the text is the colour that was chosen",
          solid.any() and set(img[solid][:, 0].tolist()) == {200})
    ctl.undo()
    check("undo removes the text layer", ctl.doc.layer(lay.id) is None and ink(shown(ctl)) == 0)
    ctl.redo()
    check("redo brings it back, text and all", ctl.doc.layer(lay.id) is not None
          and lay.params["text"] == "Hello")

    # ---- an empty click, and Escape -------------------------------------------------
    n1 = len(ctl.doc.layers())
    ctl.set_active_layer(bg)
    click(ctl, 150, 120)
    ctl.finish_tool()
    check("an empty click leaves no layer and no history",
          len(ctl.doc.layers()) == n1 and ctl.active_layer is bg)
    click(ctl, 150, 120)
    type_text(ctl, "gone")
    ctl.cancel_stroke()
    check("Escape while typing a new text layer removes it",
          len(ctl.doc.layers()) == n1 and ctl.active_layer is bg)

    # ---- re-edit ----------------------------------------------------------------------
    box = textlayer.bounds(lay.params)
    n2 = len(ctl.history_labels())
    click(ctl, box.x + 2, box.y + box.h // 2)
    check("clicking the text EDITS that layer, no new one", ctl.active_layer is lay
          and len(ctl.doc.layers()) == n1 and ctl.tool.active)
    ctl.tool.caret_end()
    type_text(ctl, " world")
    ctl.finish_tool()
    check("re-edited text is one 'Edit Text' step", lay.params["text"] == "Hello world"
          and ctl.history_labels()[n2:] == ["Edit Text"])
    ctl.undo()
    check("...that undoes back to the old text", lay.params["text"] == "Hello")
    ctl.redo()

    # ---- options and colour while editing ------------------------------------------------
    click(ctl, box.x + 2, box.y + box.h // 2)
    w_before = textlayer.bounds(lay.params).w
    ctl.set_tool_option("size", 48)
    check("changing the size while editing re-renders the text bigger",
          textlayer.bounds(lay.params).w > w_before)
    ctl.set_primary((0, 0, 230, 255))
    img = shown(ctl)
    check("picking a colour while editing recolours it", tuple(lay.params["color"]) == (0, 0, 230, 255)
          and (img[..., 0] == 0).any() and set(img[img[..., 0] == 0][:, 2].tolist()) == {230})
    ctl.set_tool_option("warp", "arc")
    ctl.set_tool_option("bend", 60)
    flat_h = textlayer.bounds(dict(lay.params, warp="none")).h
    check("Warp Text bends it (taller than unwarped)", textlayer.bounds(lay.params).h > flat_h)
    ctl.set_tool_option("warp", "none")
    ctl.finish_tool()

    # ---- paragraph box --------------------------------------------------------------------
    ctl.set_active_layer(bg)
    ctl.set_tool("text", size=20)
    ctl.begin_stroke(10, 90)
    ctl.motion_stroke(60, 90)
    ctl.end_stroke(110, 90)
    para = ctl.active_layer
    type_text(ctl, "the quick brown fox jumps over the lazy dog")
    ctl.finish_tool()
    pb = textlayer.bounds(para.params)
    check("dragging starts a paragraph box (width %d)" % para.params["box_width"],
          para.params["box_width"] == 100)
    check("...and the text wraps inside it: several lines, no wider than the box",
          pb.h > 3 * 20 and pb.w <= 100 + 2, str(pb))

    # ---- styles, resize, rasterize, save -----------------------------------------------------
    ctl.set_layer_styles(lay, {"stroke": {"size": 3, "color": (0, 200, 0), "position": "outside"}})
    img = shown(ctl)
    check("a layer style strokes the text", (img[..., 1] == 200).sum() > 50)
    params_before = dict(lay.params)
    ctl.resize_image(480, 320)
    check("resizing scales text layers: position and type size, still text",
          lay.kind == "text" and lay.params["size"] == params_before["size"] * 2
          and abs(lay.params["x"] - params_before["x"] * 2) < 1e-6)
    check("...and boxes scale with it", para.params["box_width"] == 200)
    ctl.undo()
    check("undoing the resize puts the text back", lay.params == params_before)
    f = os.path.join(tempfile.mkdtemp(), "t.ochre")
    ochre_format.save(ctl.doc, f)
    doc2, _ = ochre_format.load(f)
    l2 = doc2.layer(lay.id)
    check("text layers save and load as TEXT (editable)", l2.kind == "text"
          and l2.params["text"] == lay.params["text"] and tuple(l2.params["color"]) == tuple(lay.params["color"]))
    ctl.rasterize_layer(lay)
    check("Rasterize Layer turns the text into ordinary pixels",
          lay.kind == "raster" and ink(ctl.doc.cell(lay).plane("rgba")[..., :4]) >= 0
          and ctl.doc.cell(lay).plane("rgba")[..., 3].max() == 255)

    # ---- vertical, and the painted mode ----------------------------------------------------------
    ctl = fresh()
    ctl.set_tool("text_vertical", size=24)
    click(ctl, 100, 10)
    type_text(ctl, "ABC")
    ctl.finish_tool()
    vb = textlayer.bounds(ctl.active_layer.params)
    check("vertical text layers stack top to bottom", ctl.active_layer.kind == "text" and vb.h > 2 * vb.w)
    ctl = fresh()
    bg = ctl.active_layer
    ctl.set_tool("text", layer=False, size=30)
    click(ctl, 20, 20)
    type_text(ctl, "paint")
    ctl.finish_tool()
    check("with 'Editable text layer' off, text paints onto the layer (Paint.NET)",
          len(ctl.doc.layers()) == 1 and ink(ctl.doc.cell(bg).plane("rgba")) > 50)
    print("\nall text-layer checks passed")


if __name__ == "__main__":
    main()
