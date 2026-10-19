#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Index-locked layers, driven THROUGH THE TOOLS.

The design document promised this file and it was never written, and the
cost of that was exact: every paint tool wrote the derived rgba cache and
left the authoritative index plane untouched, so you could open a sprite,
paint on it, save, and get a byte-identical file back. The edit was visible
on screen the whole time, because the screen shows the cache.

Nothing caught it because the tests that existed wrote `plane("index")`
directly. They proved the path from the plane to the file and never asked
whether anything could reach the plane. So the rule this file exists to
enforce is not "indices round-trip" -- that was already true and already
tested -- it is:

    EVERY TOOL A USER CAN PICK MUST REACH THE AUTHORITATIVE PLANE.

The loop below is deliberately over the tool registry rather than a
hand-listed set, so a tool added later is covered without anyone
remembering to add it here.

Run: python3 tests/test_indexlock.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.document import Document
from ochre.engine.history import HistoryStack
from ochre.engine.ini import IniDB
from ochre.engine.palette import Palette
from ochre.engine.quantize import index_for
from ochre.engine.selection import Selection
from ochre.engine.tools import ToolContext, ToolEvent, ToolRegistry

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# A palette shaped like a C&C one: a transparent index 0, a protected remap
# ramp at 16-31 whose entries are near-duplicates of each other, and ordinary
# colour elsewhere.
def cnc_like():
    entries = np.zeros((256, 4), dtype=np.uint8)
    entries[:, 3] = 255
    for i in range(256):
        entries[i, :3] = (i, (i * 7) % 256, (i * 13) % 256)
    # The ramp: sixteen near-identical greens, exactly the case nearest-RGB
    # matching cannot tell apart.
    for n in range(16):
        entries[16 + n, :3] = (0, 200 + n, 0)
    pal = Palette(entries, name="cnc-like")
    pal.protect((16, 31))
    pal.set_transparent(0)
    return pal


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def setup(palette=None, fill=5):
    pal = palette or cnc_like()
    doc = Document(40, 32, palette=pal)
    layer = doc.add_layer("Sprite", planes=("index", "rgba"),
                          authoritative=("index",))
    cell = doc.cell(layer)
    cell.plane("index")[...] = fill
    cell.refresh_derived()
    cell._touched(doc.bounds)          # writes through plane() must record bounds
    ctx = ToolContext(doc, layer, doc.frame, Selection(40, 32),
                      primary=(255, 255, 255, 255), secondary=(0, 0, 0, 255),
                      history=HistoryStack())
    return doc, layer, cell, ctx


def _ink(ctx, x, y, w, h, index):
    """Scaffolding: put something on the canvas to clone FROM."""
    ctx.cell.plane("index")[y:y + h, x:x + w] = index
    ctx.cell.refresh_derived()


def _checker(ctx, x, y, w, h, a, b):
    """Scaffolding: texture, for tools that move or transfer detail."""
    yy, xx = np.mgrid[0:h, 0:w]
    ctx.cell.plane("index")[y:y + h, x:x + w] = np.where((xx // 2 + yy // 2) % 2, a, b)
    ctx.cell.refresh_derived()


def _stroke(t, c, pts, mods=0):
    t.begin(c, ToolEvent(pts[0][0], pts[0][1], 1.0, mods))
    for x, y in pts[1:]:
        t.motion(c, ToolEvent(x, y, 1.0, mods))
    return t.end(c, ToolEvent(pts[-1][0], pts[-1][1], 1.0, mods))


def _set(ctx, **kw):
    for k, v in kw.items():
        setattr(ctx, k, v)


# How to drive each tool that paints. Anything in the registry not listed
# here is checked to be deliberately exempt rather than quietly skipped.
DRIVERS = {
    "pencil":   lambda t, c: (t.begin(c, ToolEvent(8, 8)),
                              t.motion(c, ToolEvent(24, 24)),
                              t.end(c, ToolEvent(24, 24))),
    "brush":    lambda t, c: (t.begin(c, ToolEvent(8, 8)),
                              t.motion(c, ToolEvent(24, 24)),
                              t.end(c, ToolEvent(24, 24))),
    "eraser":   lambda t, c: (t.begin(c, ToolEvent(8, 8)),
                              t.motion(c, ToolEvent(24, 24)),
                              t.end(c, ToolEvent(24, 24))),
    "bucket":   lambda t, c: (t.begin(c, ToolEvent(20, 16)),
                              t.end(c, ToolEvent(20, 16))),
    "line":     lambda t, c: (t.begin(c, ToolEvent(4, 4)),
                              t.motion(c, ToolEvent(34, 28)),
                              t.end(c, ToolEvent(34, 28))),
    "rectangle": lambda t, c: (t.begin(c, ToolEvent(4, 4)),
                               t.motion(c, ToolEvent(34, 28)),
                               t.end(c, ToolEvent(34, 28))),
    "ellipse":  lambda t, c: (t.begin(c, ToolEvent(4, 4)),
                              t.motion(c, ToolEvent(34, 28)),
                              t.end(c, ToolEvent(34, 28))),
    "freeform": lambda t, c: (t.begin(c, ToolEvent(4, 4)),
                              t.motion(c, ToolEvent(30, 10)),
                              t.motion(c, ToolEvent(20, 28)),
                              t.end(c, ToolEvent(20, 28))),
    "gradient": lambda t, c: (t.begin(c, ToolEvent(0, 0)),
                              t.motion(c, ToolEvent(39, 31)),
                              t.end(c, ToolEvent(39, 31))),
    # Cloning needs something to clone: a uniform canvas copies index 5 onto
    # index 5 and legitimately changes nothing, which would make this pass
    # for the wrong reason.
    "clone":    lambda t, c: (_ink(c, 2, 2, 10, 10, 200),
                              t.begin(c, ToolEvent(6, 6, 1.0, 2)),
                              t.begin(c, ToolEvent(24, 20)),
                              t.motion(c, ToolEvent(30, 24)),
                              t.end(c, ToolEvent(30, 24))),
    "shape":    lambda t, c: (t.begin(c, ToolEvent(4, 4)),
                              t.motion(c, ToolEvent(34, 28)),
                              t.end(c, ToolEvent(34, 28))),
    "text":     lambda t, c: (t.begin(c, ToolEvent(2, 10)), t.commit_text(c)),
    # ---- retouch family: each needs content it can actually change -------
    # An airbrush is used by HOLDING it: ticks while the pointer rests are
    # what build coverage past an index's 50% threshold. A quick swipe at
    # 12% flow legitimately changes no index -- just as it paints only a
    # faint wash on an RGB layer.
    "airbrush": lambda t, c: (t.begin(c, ToolEvent(20, 16)),
                              [t.tick(c) for _ in range(12)],
                              t.end(c, ToolEvent(20, 16))),
    "ink":      lambda t, c: _stroke(t, c, [(4, 4), (34, 28)]),
    "mixer":    lambda t, c: _stroke(t, c, [(4, 4), (34, 28)]),
    "recolor":  lambda t, c: (_set(c, secondary=tuple(int(v) for v in c.doc.palette.entries[5])),
                              _stroke(t, c, [(6, 6), (30, 24)])),
    "blur":     lambda t, c: (_checker(c, 0, 0, 40, 16, 200, 5),
                              _stroke(t, c, [(4, 12), (36, 12)])),
    "sharpen":  lambda t, c: (_ink(c, 0, 0, 20, 32, 200),
                              _stroke(t, c, [(20, 4), (20, 28)])),
    "smudge":   lambda t, c: (_ink(c, 0, 0, 16, 32, 200),
                              _stroke(t, c, [(10, 16), (16, 16), (24, 16), (32, 16)])),
    "dodge":    lambda t, c: _stroke(t, c, [(6, 6), (30, 24)]),
    "burn":     lambda t, c: (_ink(c, 0, 0, 40, 32, 120), _stroke(t, c, [(6, 6), (30, 24)])),
    "sponge":   lambda t, c: (_ink(c, 0, 0, 40, 32, 200), _stroke(t, c, [(6, 6), (30, 24)])),
    "background_eraser": lambda t, c: _stroke(t, c, [(6, 6), (30, 24)]),
    "magic_eraser": lambda t, c: (t.begin(c, ToolEvent(20, 16)), t.end(c, ToolEvent(20, 16))),
    "pattern_stamp": lambda t, c: _stroke(t, c, [(6, 6), (30, 24)]),
    "history_brush": lambda t, c: (_set(c, history_source=lambda lid, fid: np.full(
                                       (c.doc.height, c.doc.width, 4), (200, 120, 40, 255), np.uint8)),
                                   _stroke(t, c, [(6, 6), (30, 24)])),
    "heal":     lambda t, c: (_checker(c, 0, 0, 16, 16, 200, 5),
                              _stroke(t, c, [(8, 8)], mods=2),
                              _stroke(t, c, [(28, 22), (30, 24)])),
    "red_eye":  lambda t, c: (_ink(c, 0, 0, 40, 32, 200), _stroke(t, c, [(6, 6), (30, 24)])),
    # ---- transforms: lift, change, commit ---------------------------------
    "transform":  lambda t, c: (_stroke(t, c, [(20, 16), (28, 22)]), t.finish(c)),
    "rotate_tool": lambda t, c: (_ink(c, 4, 4, 12, 24, 200),
                                 _stroke(t, c, [(36, 16), (20, 30)]), t.finish(c)),
    "scale_tool": lambda t, c: (_stroke(t, c, [(39.9, 31.9), (20, 16)]), t.finish(c)),
    "shear_tool": lambda t, c: (_stroke(t, c, [(20, 16), (30, 16)]), t.finish(c)),
    "perspective_tool": lambda t, c: (_stroke(t, c, [(39.9, 31.9), (25, 20)]), t.finish(c)),
    "transform3d": lambda t, c: (_stroke(t, c, [(20, 16), (32, 16)]), t.finish(c)),
    "handle_transform": lambda t, c: (_stroke(t, c, [(10, 10)]),
                                      _stroke(t, c, [(10, 10), (20, 18)]), t.finish(c)),
    "flip_tool":  lambda t, c: (_ink(c, 0, 0, 12, 32, 200),
                                t.begin(c, ToolEvent(20, 16)), t.end(c, ToolEvent(20, 16))),
    "cage":       lambda t, c: (_ink(c, 8, 8, 16, 16, 200),
                                [_stroke(t, c, [p]) for p in ((6, 6), (26, 6), (26, 26), (6, 26))],
                                t.finish(c),
                                _stroke(t, c, [(26, 26), (36, 30)]), t.finish(c)),
    "warp":       lambda t, c: (_checker(c, 0, 0, 40, 32, 200, 5),
                                _stroke(t, c, [(10, 16), (16, 16), (22, 16), (28, 16)])),
    "move_layer": lambda t, c: _stroke(t, c, [(20, 16), (30, 22)]),
}

# Tools that legitimately write no pixels. Listed explicitly so that a tool
# which stops painting is a failure rather than a silent exemption.
NON_PAINTING = {"eyedropper", "select_rect", "select_ellipse", "lasso",
                "magic_wand", "move_selection", "zoom", "pan",
                "select_row", "select_col", "polygon_lasso", "scissors",
                # Crops resize the document through the controller; the tool
                # itself only reports the rectangle.
                "crop", "perspective_crop", "rotate_view",
                "foreground_select", "quick_select", "select_color", "object_select",
                # Moves existing pixels rather than laying down new colour;
                # covered by test_movetools.py against an indexed layer.
                "move_pixels"}


def main():
    reg = ToolRegistry(IniDB(DATA))

    # ---- THE ONE THAT MATTERED -------------------------------------------
    unchecked = []
    for name in reg.names():
        if name in NON_PAINTING:
            continue
        driver = DRIVERS.get(name)
        if driver is None:
            unchecked.append(name)
            continue

        opts = {"size": 5} if name in ("pencil", "brush", "eraser", "clone") else {}
        if name == "text":
            opts = {"text": "Hi", "size": 14}
        _doc, _layer, cell, ctx = setup()
        before = cell.plane("index").copy()
        tool = reg.create(name, **opts)
        driver(tool, ctx)
        changed = int((before != cell.plane("index")).sum())
        check("%s reaches the AUTHORITATIVE index plane" % name, changed > 0,
              "-- it wrote %d index pixels; a tool that only writes the rgba "
              "cache paints something the user watches disappear on save"
              % changed)

    check("every painting tool in the registry was exercised", not unchecked,
          "-- no driver for %s; add one rather than letting it go unchecked"
          % ", ".join(unchecked))

    # Every SHAPE the one tool offers, not just its default. They share a
    # render path, but each reaches it through its own geometry, and a
    # polygon that produced no vertices would silently draw nothing.
    shape_tool = reg.create("shape")
    for choice in shape_tool.choices():
        _doc, _layer, cell, ctx = setup()
        before = cell.plane("index").copy()
        tool = reg.create("shape", shape=choice)
        DRIVERS["shape"](tool, ctx)
        check("shape=%s paints" % choice,
              int((before != cell.plane("index")).sum()) > 0)

    # ---- the rgba cache must agree with what was painted -----------------
    doc, layer, cell, ctx = setup()
    tool = reg.create("pencil", size=5)
    DRIVERS["pencil"](tool, ctx)
    idx = cell.plane("index")
    expected = doc.palette.render_entries[idx]
    check("the rgba cache still matches the indices afterwards",
          np.array_equal(cell.plane("rgba"), expected),
          "-- the cache is what the screen shows; if it drifts, the canvas "
          "lies about the file")

    # ---- coverage is a threshold, not a blend ----------------------------
    # Averaging index 5 and index 200 is not index 102, it is a different
    # colour -- and on a C&C palette often the shadow or a remap entry.
    doc, layer, cell, ctx = setup()
    ctx.primary = tuple(int(v) for v in doc.palette.entries[200, :3]) + (255,)
    tool = reg.create("brush", size=9, hardness=0.1)   # deliberately very soft
    DRIVERS["brush"](tool, ctx)
    written = set(np.unique(cell.plane("index")).tolist())
    check("A SOFT BRUSH PRODUCES NO INTERMEDIATE INDICES",
          written <= {5, 200},
          "-- got %s; partial coverage must round to painted or not painted, "
          "never to an index halfway between two unrelated colours"
          % sorted(written))

    # ---- the protected ramp: deliberate yes, accidental no ---------------
    pal = cnc_like()
    ramp = tuple(int(v) for v in pal.entries[17, :3])
    check("an EXACT protected colour is honoured",
          int(index_for(np.array([ramp], dtype=np.uint8), pal)[0]) == 17,
          "-- picking index 17 out of the palette means it; refusing to "
          "paint the remap ramp would make the editor useless for C&C")

    near = tuple(min(255, v + 3) for v in ramp)
    landed = int(index_for(np.array([near], dtype=np.uint8), pal)[0])
    check("a NEARBY colour never lands in the protected ramp",
          landed not in pal.protected,
          "-- %s snapped to %d; this is the whole reason protection exists, "
          "because ramp entries are near-duplicates in RGB" % (near, landed))

    doc, layer, cell, ctx = setup()
    ctx.primary = ramp + (255,)
    tool = reg.create("pencil", size=3)
    DRIVERS["pencil"](tool, ctx)
    check("...and a tool painting it writes that exact index",
          17 in set(np.unique(cell.plane("index")).tolist()))

    # ---- erasing means the transparent INDEX, not zero alpha -------------
    doc, layer, cell, ctx = setup()
    tool = reg.create("eraser", size=7)
    DRIVERS["eraser"](tool, ctx)
    check("the eraser writes the palette's transparent index",
          0 in set(np.unique(cell.plane("index")).tolist()),
          "-- erasing the rgba cache erases nothing that survives a save")

    # A palette whose transparent entry is not 0 must be respected: RA2's
    # alpha-image sprites use 127.
    pal = cnc_like()
    pal.set_transparent(127)
    doc, layer, cell, ctx = setup(palette=pal)
    tool = reg.create("eraser", size=7)
    DRIVERS["eraser"](tool, ctx)
    check("...whatever index the palette says is transparent",
          127 in set(np.unique(cell.plane("index")).tolist()),
          "-- transparency is a per-document setting, never the constant 0")

    # ---- undo and history operate on indices -----------------------------
    doc, layer, cell, ctx = setup()
    before = cell.plane("index").copy()
    tool = reg.create("pencil", size=5)
    DRIVERS["pencil"](tool, ctx)
    after = cell.plane("index").copy()
    check("painting pushed one history entry", len(ctx.history) == 1)
    ctx.history.undo(doc)
    check("UNDO RESTORES THE INDICES", np.array_equal(cell.plane("index"), before))
    check("...and the rgba cache follows it back",
          np.array_equal(cell.plane("rgba"),
                         doc.palette.render_entries[cell.plane("index")]))
    ctx.history.redo(doc)
    check("redo re-applies them", np.array_equal(cell.plane("index"), after))

    print("\nall index-lock checks passed")


if __name__ == "__main__":
    main()
