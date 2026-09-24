#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Text rasterisation and the text tool.

The point worth proving: text is rasterised in the ENGINE, with no Qt. The
design document assumed the UI would have to do it because glyphs need font
machinery; Pillow already ships FreeType and is already a dependency, so the
engine stays Qt-free and text stays scriptable.

The other property is the live-edit loop. Retyping must leave no residue of
previous strings, because it restores from the same buffer that becomes the
undo entry.

Run: python3 tests/test_text.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import text as textmod
from ochre.engine.document import Document
from ochre.engine.geometry import Rect
from ochre.engine.history import HistoryStack
from ochre.engine.ini import IniDB
from ochre.engine.selection import Selection
from ochre.engine.tools import ToolContext, ToolEvent, ToolRegistry

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def setup(w=200, h=120, bg=(0, 0, 0, 255)):
    d = Document(w, h)
    lay = d.add_layer("paint")
    d.cell(lay).fill(d.bounds, bg)
    sel = Selection(w, h)
    hist = HistoryStack()
    ctx = ToolContext(d, lay, d.frame, sel, primary=(255, 0, 0, 255),
                      secondary=(0, 0, 255, 255), history=hist)
    return d, lay, sel, hist, ctx


def main():
    # ---- the whole point -------------------------------------------------
    check("TEXT RASTERISES WITH NO QT LOADED",
          not any(m.startswith(("PySide", "PyQt", "shiboken")) for m in sys.modules),
          "-- the engine must stay Qt-free; that is why Pillow does this")

    # ---- font discovery --------------------------------------------------
    book = textmod.FontBook()
    faces = book.faces()
    if not faces:
        print("ok: skipped (no fonts installed)")
        return
    check("fonts discovered", len(faces) > 0, "-- %d faces" % len(faces))
    check("a known family resolves",
          book.resolve("DejaVu Sans") is not None)
    check("a fontconfig ALIAS resolves",
          book.resolve("sans-serif") is not None,
          "-- fontconfig understands aliases; reimplementing that badly is waste")
    check("an absolute path passes through",
          book.resolve(book.resolve("DejaVu Sans")) == book.resolve("DejaVu Sans"))
    check("names() lists the faces", len(book.names()) == len(faces))

    # A missing font must fall back, never raise -- a document should open on
    # a machine that lacks the font it was authored with.
    font = book.load("This Font Certainly Does Not Exist", 24)
    check("a missing font falls back rather than raising", font is not None)
    check("loaded fonts are cached", book.load("DejaVu Sans", 24)
          is book.load("DejaVu Sans", 24))

    # ---- rasterising -----------------------------------------------------
    cov, w, h = textmod.render_coverage("Hello", size=32)
    check("text rasterises", w > 0 and h > 0, "-- %dx%d" % (w, h))
    check("coverage is uint8", cov.dtype == np.uint8)
    check("coverage is 2-D", cov.ndim == 2)
    check("glyphs actually marked pixels", int(cov.max()) == 255)
    check("output is cropped tight",
          int(cov[:, 0].max()) > 0 and int(cov[:, -1].max()) > 0,
          "-- leading or trailing blank columns were not cropped")

    soft = int(((cov > 0) & (cov < 255)).sum())
    check("text is antialiased", soft > 0)
    hard, _w, _h = textmod.render_coverage("Hello", size=32, antialias=False)
    check("antialias=off gives hard edges", set(np.unique(hard)) <= {0, 255})

    big, bw, bh = textmod.render_coverage("Hello", size=64)
    check("a larger size rasterises larger", bw > w and bh > h)

    empty, ew, eh = textmod.render_coverage("", size=24)
    check("empty text yields an empty plane", ew == 0 and eh == 0)
    space, sw, sh = textmod.render_coverage("   ", size=24)
    check("whitespace-only text yields nothing to draw", sw == 0 or sh == 0)

    multi, mw, mh = textmod.render_coverage("one\ntwo\nthree", size=24)
    single, _sw, sh1 = textmod.render_coverage("one", size=24)
    check("multi-line text is taller than one line", mh > sh1 * 2,
          "-- %d vs %d" % (mh, sh1))

    tight, _w, th = textmod.render_coverage("one\ntwo", size=24, line_spacing=1.0)
    loose, _w2, lh = textmod.render_coverage("one\ntwo", size=24, line_spacing=2.0)
    check("line spacing widens the block", lh > th, "-- %d vs %d" % (lh, th))

    # ---- placement -------------------------------------------------------
    cov, w, h = textmod.render_coverage("Anchor", size=24)
    left = textmod.place(cov, 100, 50, textmod.LEFT)
    centre = textmod.place(cov, 100, 50, textmod.CENTER)
    right = textmod.place(cov, 100, 50, textmod.RIGHT)
    check("left alignment starts at the anchor", left.x == 100)
    check("centre alignment straddles the anchor",
          centre.x < 100 and centre.x1 > 100)
    check("right alignment ends at the anchor", abs(right.x1 - 100) <= 1)
    check("all alignments keep the same size",
          left.w == centre.w == right.w and left.h == centre.h == right.h)
    check("baseline placement lifts the block",
          textmod.place(cov, 100, 50, baseline=True).y < 50)

    # ---- the tool --------------------------------------------------------
    reg = ToolRegistry(IniDB(DATA))
    check("text tool registered", "text" in reg)
    check("INI supplies the font", reg.create("text").option("font") == "DejaVu Sans")
    check("INI supplies the size", reg.create("text").option("size") == 24)

    d, lay, sel, hist, ctx = setup()
    tool = reg.create("text", text="Hi", size=48)
    tool.begin(ctx, ToolEvent(20, 30))
    px = d.cell(lay).pixels
    check("the tool painted text", int((px[..., 0] > 0).sum()) > 0)
    check("text used the primary colour",
          tuple(px[np.nonzero(px[..., 0] > 200)][0])[:3] == (255, 0, 0))

    # The session stays OPEN after mouse-up, because typing continues.
    tool.end(ctx, ToolEvent(20, 30))
    check("the session survives mouse-up so typing can continue", tool.active,
          "-- on_end finishes positioning, not the edit")
    check("nothing committed yet", len(hist) == 0)

    tool.commit_text(ctx)
    check("commit_text pushes exactly one entry", len(hist) == 1)
    check("the tool is finished after commit", not tool.active)

    after = d.cell(lay).pixels.copy()
    hist.undo(d)
    check("text undoes cleanly", int((d.cell(lay).pixels[..., 0] > 0).sum()) == 0)
    hist.redo(d)
    check("text redoes cleanly", np.array_equal(d.cell(lay).pixels, after))

    # ---- THE live-edit loop ---------------------------------------------
    # Retyping must leave no residue: a long string replaced by a short one
    # must look exactly like the short one drawn alone.
    d, lay, sel, hist, ctx = setup()
    tool = reg.create("text", text="", size=40)
    tool.begin(ctx, ToolEvent(10, 20))
    for partial in ("W", "Wo", "Wor", "Worrrrrrld", "Wo", "Hi"):
        tool.set_text(ctx, partial)
    tool.commit_text(ctx)
    typed = d.cell(lay).pixels.copy()

    d2, lay2, _s, _h, ctx2 = setup()
    direct = reg.create("text", text="Hi", size=40)
    direct.begin(ctx2, ToolEvent(10, 20))
    direct.commit_text(ctx2)

    check("RETYPING LEAVES NO RESIDUE OF EARLIER STRINGS",
          np.array_equal(typed, d2.cell(lay2).pixels),
          "-- a longer previous string left pixels behind")

    # Dragging repositions rather than resizing.
    d, lay, sel, hist, ctx = setup()
    tool = reg.create("text", text="Drag", size=32)
    tool.begin(ctx, ToolEvent(10, 20))
    tool.motion(ctx, ToolEvent(90, 60))
    tool.end(ctx, ToolEvent(90, 60))
    tool.commit_text(ctx)
    ys, xs = np.nonzero(d.cell(lay).pixels[..., 0] > 0)
    check("dragging repositions the text", int(xs.min()) >= 80,
          "-- text landed at x=%d, expected near 90" % int(xs.min()))

    # Cancelling leaves nothing.
    d, lay, sel, hist, ctx = setup()
    pristine = d.cell(lay).pixels.copy()
    tool = reg.create("text", text="Gone", size=32)
    tool.begin(ctx, ToolEvent(10, 20))
    tool.cancel(ctx)
    check("cancelling text restores the canvas",
          np.array_equal(d.cell(lay).pixels, pristine))
    check("cancelling text leaves no history", len(hist) == 0)

    # Empty text draws nothing at all.
    d, lay, sel, hist, ctx = setup()
    tool = reg.create("text", text="", size=32)
    tool.begin(ctx, ToolEvent(10, 20))
    tool.commit_text(ctx)
    check("empty text paints nothing", int(d.cell(lay).pixels[..., 0].sum()) == 0)

    # ---- and it clips like everything else ------------------------------
    d, lay, sel, hist, ctx = setup()
    sel.select_rect((0, 0, 100, 120))          # left half only
    tool = reg.create("text", text="WIDE TEXT HERE", size=40)
    tool.begin(ctx, ToolEvent(10, 40))
    tool.commit_text(ctx)
    outside = d.cell(lay).pixels[:, 102:]
    check("TEXT CANNOT PAINT OUTSIDE THE SELECTION",
          int((outside[..., 0] != 0).sum()) == 0,
          "-- no tool implements clipping; it falls out of apply_masked")
    check("text still paints inside the selection",
          int((d.cell(lay).pixels[:, :98, 0] != 0).sum()) > 0)

    # Off-canvas placement must clip, not raise.
    d, lay, sel, hist, ctx = setup()
    tool = reg.create("text", text="Offscreen", size=32)
    tool.begin(ctx, ToolEvent(500, 500))
    tool.commit_text(ctx)
    check("text placed off-canvas is harmless", True)

    # ---- the caret -------------------------------------------------------
    # The caret lives in the ENGINE as an integer index, so the whole edit
    # loop is testable with no Qt and no display.
    lay = textmod.layout("Hi\nthere", size=32)
    check("layout reports a line step", lay.line_step > 0)
    check("layout keeps the crop delta",
          isinstance(lay.crop_x, int) and isinstance(lay.crop_y, int),
          "-- a caret computed in uncropped space needs this to line up")

    x0, y0, ch = lay.caret_at(0)
    x1, _y1, _h = lay.caret_at(1)
    check("the caret advances along a line", x1 > x0)
    check("the caret has a height", ch > 0)
    _x, y3, _h = lay.caret_at(3)
    check("the caret drops to the next line", y3 > y0,
          "-- index 3 is past the newline")

    check("an EMPTY string still has a caret position",
          textmod.layout("", size=32).caret_at(0) is not None,
          "-- otherwise clicking to start typing shows nothing")

    check("clicking maps back to an index",
          lay.index_at(*lay.caret_at(2)[:2]) == 2,
          "-- round-tripping a caret position must return the same index")

    # ---- editing operations ---------------------------------------------
    d, lay_, sel, hist, ctx = setup()
    tool = reg.create("text", size=28)
    tool.begin(ctx, ToolEvent(20, 30))

    for ch in "Hello":
        tool.insert(ctx, ch)
    check("typing builds the string", tool.option("text") == "Hello")
    check("the caret follows typing", tool.caret == 5)

    tool.move_caret(-2)
    check("left arrow moves the caret back", tool.caret == 3)
    tool.insert(ctx, "XY")
    check("typing inserts AT the caret", tool.option("text") == "HelXYlo",
          "-- got %r" % tool.option("text"))
    check("the caret follows the insertion", tool.caret == 5)

    tool.backspace(ctx)
    check("backspace deletes before the caret",
          tool.option("text") == "HelXlo" and tool.caret == 4)
    tool.delete(ctx)
    check("delete removes after the caret", tool.option("text") == "HelXo")

    tool.caret_home()
    check("home goes to the line start", tool.caret == 0)
    tool.backspace(ctx)
    check("backspace at the start is harmless", tool.option("text") == "HelXo")
    tool.caret_end()
    check("end goes to the line end", tool.caret == 5)
    tool.delete(ctx)
    check("delete at the end is harmless", tool.option("text") == "HelXo")

    tool.move_caret(-100)
    check("the caret cannot go negative", tool.caret == 0)
    tool.move_caret(1000)
    check("the caret cannot run past the end", tool.caret == 5)

    # Multi-line navigation
    tool.set_text(ctx, "one\ntwo\nthree")
    tool.caret = 5                       # inside "two"
    tool.caret_line(-1)
    check("up a line keeps the column", tool.caret == 1,
          "-- got %d" % tool.caret)
    tool.caret_line(1)
    check("down a line keeps the column", tool.caret == 5)
    tool.caret = 1
    tool.caret_line(-1)
    check("up from the first line stays put", tool.caret == 1)
    tool.caret = 10
    tool.caret_line(1)
    check("down from the last line stays put", tool.caret == 10)

    tool.caret_home()
    check("home finds the LINE start, not the string start", tool.caret == 8,
          "-- got %d" % tool.caret)

    # ---- the caret's drawn position -------------------------------------
    rect = tool.caret_rect(ctx)
    check("the caret has a document rect", rect is not None)
    check("the caret is one pixel wide", rect.w == 1)
    check("the caret is as tall as a line", rect.h > 1)

    tool.set_text(ctx, "")
    empty_rect = tool.caret_rect(ctx)
    check("an empty text box still shows a caret", empty_rect is not None,
          "-- this is exactly when a caret matters most")

    tool.set_text(ctx, "Positioned")
    tool.caret = 0
    left = tool.caret_rect(ctx)
    tool.caret = 10
    right = tool.caret_rect(ctx)
    check("the caret rect moves with the caret", right.x > left.x)

    tool.caret_from_point(ctx, right.x, right.y + 2)
    check("clicking positions the caret near the click",
          abs(tool.caret - 10) <= 1, "-- landed at %d" % tool.caret)

    tool.commit_text(ctx)
    check("editing then committing pushes one entry", len(hist) == 1)

    # Typing is still just set_text underneath, so the no-residue property
    # from earlier holds for keystroke-driven editing too.
    d, lay_, sel, hist, ctx = setup()
    tool = reg.create("text", size=36)
    tool.begin(ctx, ToolEvent(10, 20))
    for ch in "Wide text":
        tool.insert(ctx, ch)
    for _ in range(7):
        tool.backspace(ctx)
    tool.commit_text(ctx)
    typed = d.cell(lay_).pixels.copy()

    d2, lay2, _s, _h, ctx2 = setup()
    direct = reg.create("text", text="Wi", size=36)
    direct.begin(ctx2, ToolEvent(10, 20))
    direct.commit_text(ctx2)
    check("TYPING AND DELETING LEAVES NO RESIDUE",
          np.array_equal(typed, d2.cell(lay2).pixels),
          "-- deleted characters left pixels behind")

    print("\nall text checks passed")


if __name__ == "__main__":
    main()
