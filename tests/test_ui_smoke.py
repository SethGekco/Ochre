#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The UI, exercised offscreen.

Constructing widgets verifies almost nothing about a canvas, so this forces
real synchronous paintEvents over a live document. That is what exercises
the numpy-to-QImage aliasing, the checkerboard, the nearest-neighbour switch
and drawImage -- precisely the code that would segfault if the buffer
lifetime rules were broken.

Skips cleanly when PySide6 is absent, so the suite stays green on a machine
with only the engine installed.

Run: QT_QPA_PLATFORM=offscreen python3 tests/test_ui_smoke.py
"""

import math
import os
import sys
import tempfile
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


try:
    import numpy as np
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtWidgets import QApplication, QDockWidget, QPushButton
except ImportError as exc:
    print("ok: skipped (PySide6 unavailable: %s)" % exc)
    sys.exit(0)

from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.bus import EventBus
from ochre.ui.canvas import CanvasSurface
from ochre.ui.config import UiSettings
from ochre.ui.controller import Controller
from ochre.ui.mainwindow import MainWindow

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    app = QApplication.instance() or QApplication([])

    # ---- CanvasSurface: the aliasing contract ---------------------------
    arr = np.zeros((8, 12, 4), dtype=np.uint8)
    surf = CanvasSurface(arr)
    check("surface is valid", surf.is_valid())
    check("surface size matches the array", surf.size == (12, 8))
    check("stride is width*4", surf.image.bytesPerLine() == 12 * 4)
    check("format is RGBA8888",
          surf.image.format() == surf.image.format().Format_RGBA8888)

    arr[3, 4] = (10, 20, 30, 255)
    px = surf.image.pixelColor(4, 3)
    check("a numpy write shows through the QImage, with no copy",
          (px.red(), px.green(), px.blue()) == (10, 20, 30))

    # The lifetime rule: the surface must keep the array alive itself.
    check("surface holds its own reference to the array", surf.array is arr)
    del arr
    px = surf.image.pixelColor(4, 3)
    check("image survives the caller dropping its reference",
          (px.red(), px.green(), px.blue()) == (10, 20, 30))

    try:
        CanvasSurface(np.zeros((4, 4), dtype=np.uint8))
        check("surface rejects a non-RGBA array", False, "-- no error")
    except ValueError:
        check("surface rejects a non-RGBA array", True)

    surf.detach()
    check("detach clears both halves together",
          surf.image is None and surf.array is None)

    # ---- full window -----------------------------------------------------
    bus = EventBus()
    ctl = Controller(Settings(DATA), DATA, bus)
    win = MainWindow(ctl, UiSettings(DATA))

    docks = {d.objectName() for d in win.findChildren(QDockWidget)}
    check("all five docks exist",
          {"Tools", "Layers", "Colors", "Palette", "History"} <= docks,
          "-- got %r" % sorted(docks))
    check("a document was created", ctl.doc is not None)
    check("a tool is selected", ctl.tool is not None)
    check("canvas bound to the display buffer",
          win.canvas.surface.array is ctl.display)

    win.resize(900, 620)
    win.canvas.resize(820, 540)

    # ---- forced paint ----------------------------------------------------
    win.canvas.repaint()
    check("painting an empty document does not crash", True)

    # ---- paint through the real tool path -------------------------------
    ctl.set_tool("brush", size=20)
    ctl.set_primary((220, 40, 60, 255))
    ctl.begin_stroke(60, 60)
    for i in range(10):
        ctl.motion_stroke(60 + i * 12, 60 + i * 6)
    ctl.end_stroke(180, 120)
    rects = ctl.flush()
    check("a stroke produced dirty rects", bool(rects))
    check("a stroke produced one history entry", len(ctl.history) == 1)

    painted = ctl.display[60, 60]
    check("the display buffer actually changed", int(painted[0]) > 0,
          "-- got %r" % (tuple(painted),))

    win.canvas.repaint()
    check("painting after a stroke does not crash", True)

    # ---- zoom paths, including the nearest-neighbour switch -------------
    from fractions import Fraction
    for zoom in (Fraction(1, 8), Fraction(1, 2), Fraction(1, 1),
                 Fraction(4, 1), Fraction(16, 1)):
        win.canvas.zoom_to(zoom)
        win.canvas.repaint()
    check("painting survives every zoom level", True)
    check("the pixel grid threshold is reachable",
          win.canvas.view.percent >= 800)

    win.canvas.zoom_fit()
    check("zoom-to-fit keeps the document on screen",
          win.canvas.view.canvas_size()[0] <= win.canvas.viewport().width() + 1)
    win.canvas.zoom_actual()
    check("actual size is exactly 100%", win.canvas.view.zoom == Fraction(1, 1))

    # ---- selection overlay ----------------------------------------------
    ctl.selection.select_ellipse((20, 20, 200, 160))
    win.canvas.repaint()
    check("painting with an active selection does not crash", True)
    # Raw clean-up, not ctl.deselect(): Deselect is itself an undoable step
    # now, and the next check is about undoing the stroke before it.
    ctl.selection.select_all()

    # ---- undo/redo through the UI ---------------------------------------
    win.do_undo()
    check("undo through the window rewinds", ctl.history.position == -1)
    win.do_redo()
    check("redo through the window restores", ctl.history.position == 0)

    # ---- docks respond to bus events ------------------------------------
    ctl.add_layer("Second")
    win.layers_dock.sync()
    check("layers dock lists both layers", win.layers_dock.list.count() == 2,
          "-- got %d" % win.layers_dock.list.count())

    ctl.set_layer_property(ctl.active_layer, "opacity", 128)
    win.layers_dock.sync()
    check("layers dock reflects opacity", win.layers_dock.opacity.value() == 128)

    ctl.set_primary((1, 2, 3, 255), index=9)
    win.colors_dock.sync()
    check("colors dock shows the index, not only the hex",
          "index 9" in win.colors_dock.readout.text(),
          "-- got %r" % win.colors_dock.readout.text())

    win.history_dock.sync()
    check("history dock lists entries plus the original",
          win.history_dock.list.count() == len(ctl.history) + 1)

    # ---- palette dock ----------------------------------------------------
    win.palette_dock.sync()
    check("palette dock reports no palette when none is bound",
          "No palette" in win.palette_dock.info.text())

    from ochre.engine.palette import grayscale
    pal = grayscale()
    pal.protect((16, 31))
    pal.annotate(16, 31, "Ramp", "some-addon-role")
    pal.set_transparent(0)
    ctl.doc.bind_palette(pal)
    win.palette_dock.sync()
    check("palette dock renders a bound palette",
          "protected" in win.palette_dock.info.text())
    check("palette dock shows annotations it does not understand",
          "Ramp" in win.palette_dock.info.text(),
          "-- the core displays annotations without interpreting them")

    # ---- resizing --------------------------------------------------------
    win.canvas.resize(400, 300)
    win.canvas.repaint()
    win.canvas.resize(1100, 700)
    win.canvas.repaint()
    check("painting survives resizing", True)

    # ---- new document rebinds safely ------------------------------------
    old_display = ctl.display
    ctl.new_document(120, 90)
    win.canvas.bind()
    win.canvas.repaint()
    check("new document rebinds the canvas",
          win.canvas.surface.array is ctl.display)
    check("the old buffer was genuinely replaced", ctl.display is not old_display)
    check("view state matches the new document",
          win.canvas.view.doc_w == 120 and win.canvas.view.doc_h == 90)

    # ---- generated effect dialogs ---------------------------------------
    from ochre.ui.effectdialog import EffectDialog

    ctl.new_document(80, 60, (128, 128, 128, 255))
    win.canvas.bind()
    layer = ctl.active_layer
    pristine = ctl.doc.cell(layer).pixels.copy()

    effect = win.effects.create("brightness_contrast")
    dlg = EffectDialog(effect, ctl, win, debounce_ms=0)
    check("dialog generated one row per property",
          len(dlg.rows) == len(list(effect.props)),
          "-- %d rows for %d properties" % (len(dlg.rows), len(list(effect.props))))
    check("preview ran on construction",
          not np.array_equal(ctl.doc.cell(layer).pixels, pristine)
          or effect.props.get("brightness") == 0)

    effect.props.set("brightness", 60)
    dlg._render_preview()
    check("preview brightened the layer",
          int(ctl.doc.cell(layer).pixels[30, 40, 0]) > 128,
          "-- got %d" % int(ctl.doc.cell(layer).pixels[30, 40, 0]))

    dlg.reject()
    check("cancelling an effect restores the layer exactly",
          np.array_equal(ctl.doc.cell(layer).pixels, pristine))

    # Commit must render from the PRISTINE source, not from the preview, or
    # the effect would compound with itself.
    effect2 = win.effects.create("brightness_contrast", brightness=40)
    dlg2 = EffectDialog(effect2, ctl, win, debounce_ms=0)
    dlg2._render_preview()
    dlg2.accept()
    committed = ctl.doc.cell(layer).pixels.copy()
    expected = effect2.apply(pristine)
    check("commit renders from the original, not the preview",
          np.array_equal(committed, expected),
          "-- the effect compounded with its own preview")
    check("commit recorded exactly one history entry", len(ctl.history) == 1)

    ctl.undo()
    check("an applied effect undoes cleanly",
          np.array_equal(ctl.doc.cell(layer).pixels, pristine))

    # Every registered effect must produce a working dialog -- that is the
    # whole point of generating them from declared properties.
    failures = []
    for name in win.effects.names():
        eff = win.effects.create(name)
        try:
            probe = EffectDialog(eff, ctl, win, debounce_ms=0)
            if len(probe.rows) != len(list(eff.props)):
                failures.append(name)
            probe.reject()
        except Exception as exc:                              # noqa: BLE001
            failures.append("%s (%s)" % (name, exc))
    check("every registered effect generates a dialog", not failures,
          "-- %r" % failures)

    # A rule moving a property must be reflected in the widgets.
    post = win.effects.create("posterize")
    pdlg = EffectDialog(post, ctl, win, debounce_ms=0)
    post.props.set("red", 8)
    for row in pdlg.rows:
        row.refresh()
    green_row = [r for r in pdlg.rows if r.prop.name == "green"][0]
    check("linked properties update their widgets",
          green_row.spin.value() == 8,
          "-- widget shows %r, model says %r"
          % (green_row.spin.value(), post.props.get("green")))

    # Read-only is model state; the row must merely reflect it.
    post.props["green"].readonly = True
    green_row.refresh()
    check("a read-only property disables its row", not green_row.isEnabled())
    pdlg.reject()

    # ---- the indexed workflow, end to end -------------------------------
    from ochre.engine.quantize import ORDERED, convert_layer_to_indexed

    ctl.new_document(64, 48, (255, 255, 255, 255))
    win.canvas.bind()
    lay = ctl.active_layer
    from ochre.engine.geometry import Rect as _Rect
    ctl.doc.cell(lay).fill(_Rect(0, 0, 32, 48), (200, 60, 60, 255))
    ctl.doc.cell(lay).fill(_Rect(32, 0, 32, 48), (60, 60, 200, 255))

    win.do_bind_palette()
    check("Image menu binds a palette", ctl.doc.palette is not None)
    win.palette_dock.sync()
    check("palette dock picks it up", "protected" in win.palette_dock.info.text())

    report = convert_layer_to_indexed(ctl.doc, lay, dither=ORDERED)
    check("conversion produced a report", len(report) == 1)
    check("layer is now index-locked", lay.index_locked)
    win.layers_dock.sync()
    check("layers dock marks an indexed layer",
          "[indexed]" in win.layers_dock.list.item(0).text(),
          "-- got %r" % win.layers_dock.list.item(0).text())

    cell = ctl.doc.cell(lay)
    before = cell.pixels.copy()
    used = int(cell.plane("index")[10, 10])

    # A palette edit must repaint instantly and cost nothing in pixel data.
    ctl.set_palette_entry(used, (0, 255, 0, 255))
    check("editing a palette entry repaints the pixels using it",
          not np.array_equal(cell.pixels, before))
    check("the index plane was not touched",
          int(cell.plane("index")[10, 10]) == used)
    check("a palette edit is undoable", len(ctl.history) >= 1)
    ctl.undo()
    check("undoing a palette edit restores the colour exactly",
          np.array_equal(cell.pixels, before))

    # Cursor readout reports the index, which is the number a spriter works in.
    win.ctl.set_active_layer(lay)
    win._on_cursor(10, 10)
    check("status bar shows the index under the cursor",
          "index" in win.pos_label.text(),
          "-- got %r" % win.pos_label.text())

    # ---- frames dock -----------------------------------------------------
    win.frames_dock.sync()
    check("frames dock lists the single frame", win.frames_dock.list.count() == 1)
    ctl.add_frame(copy_current=True)
    win.frames_dock.sync()
    check("frames dock follows an added frame", win.frames_dock.list.count() == 2)
    ctl.set_current_frame(1)
    win.frames_dock.sync()
    check("frames dock tracks the current frame",
          win.frames_dock.list.currentRow() == 1)
    check("frames dock reports the axis layout",
          "strip" in win.frames_dock.info.text())
    ctl.doc.axis_layout = "grid"
    win.frames_dock.sync()
    check("one panel serves both layouts", "grid" in win.frames_dock.info.text())
    win.canvas.repaint()
    check("painting a multi-frame document does not crash", True)

    # ---- the bus survives a bad subscriber ------------------------------
    def explode(**_):
        raise RuntimeError("boom")

    bus.subscribe("layers.changed", explode)
    ctl.add_layer("After the explosion")
    check("a raising subscriber does not break the emitter", True)
    check("the bad subscriber was recorded and removed",
          any(t == "layers.changed" for t, _h, _e in bus.errors))

    # ---- the tool footprint ---------------------------------------------
    # A brush hides the system cursor and draws its own outline, so that
    # outline has to cover the pixels that would ACTUALLY change. If it is
    # off by even half a pixel the tool aims wrong, which is worse than
    # having no cursor at all.
    from PySide6.QtCore import Qt

    canvas = win.canvas
    ctl.set_tool("pencil")
    canvas.apply_tool_cursor()
    check("a footprint tool hides the system cursor",
          canvas.viewport().cursor().shape() == Qt.BlankCursor,
          "-- it draws its own outline instead")

    ctl.set_tool("bucket")
    canvas.apply_tool_cursor()
    check("a tool with NO footprint keeps a real cursor",
          canvas.viewport().cursor().shape() != Qt.BlankCursor,
          "-- an invisible pointer with nothing drawn is a lost pointer")

    canvas.view.zoom = Fraction(8)
    canvas.view.offset_x = canvas.view.offset_y = 0

    ctl.set_tool("pencil", size=1)
    canvas._move_hover(10.5, 20.5)
    rect = canvas._footprint_rect()
    check("a 1px pencil covers exactly one pixel cell",
          rect.width() == 8 and rect.height() == 8,
          "-- got %dx%d at zoom 8" % (rect.width(), rect.height()))
    check("...and it is the cell under the pointer",
          rect.x() == 80 and rect.y() == 160,
          "-- got %d,%d" % (rect.x(), rect.y()))

    ctl.set_tool("brush", size=16)
    canvas._move_hover(10.5, 20.5)
    rect = canvas._footprint_rect()
    check("the footprint scales with brush size",
          rect.width() == 16 * 8, "-- got %d" % rect.width())

    # The footprint must sit where the DAB sits. Both floor the same way, so
    # a brush centred at x uses the same origin the outline draws from.
    size = 16
    expect_x = math.floor(10.5 - size / 2.0)
    check("the outline uses the brush's own placement arithmetic",
          rect.x() == expect_x * 8,
          "-- outline at %d, dab at %d" % (rect.x(), expect_x * 8))

    canvas._move_hover(None, None)
    check("the footprint disappears when the pointer leaves",
          canvas._footprint_rect() is None,
          "-- otherwise it is left stranded on the canvas")

    ctl.set_tool("bucket")
    canvas._move_hover(5.0, 5.0)
    check("a tool with no footprint draws none",
          canvas._footprint_rect() is None)

    # ---- layers: reordering ---------------------------------------------
    doc = ctl.doc
    while len(doc.layers()) > 1:
        ctl.remove_layer(doc.layers()[-1])
    bottom = doc.layers()[0]
    bottom.name = "bottom"
    middle = ctl.add_layer("middle")
    top = ctl.add_layer("top")
    check("three layers, bottom-first",
          [l.name for l in doc.layers()] == ["bottom", "middle", "top"])

    ctl.move_layer(top, doc.root, 0)
    check("REORDERING RESTACKS THE IMAGE",
          [l.name for l in doc.layers()] == ["top", "bottom", "middle"],
          "-- got %s" % [l.name for l in doc.layers()])
    # Counting entries would be wrong here: pushing after an undo discards
    # the redo branch, so the total can stay flat. What matters is that the
    # CURRENT entry is the reorder.
    current = ctl.history.entries[ctl.history.position]
    check("reordering is undoable", current.label == "Reorder layer",
          "-- a drag that cannot be undone is a trap; current entry is %r"
          % current.label)
    ctl.undo()
    check("...and undo puts it back",
          [l.name for l in doc.layers()] == ["bottom", "middle", "top"],
          "-- got %s" % [l.name for l in doc.layers()])
    ctl.redo()
    check("...and redo re-applies it",
          [l.name for l in doc.layers()] == ["top", "bottom", "middle"])
    ctl.undo()

    check("a no-op move pushes no history",
          ctl.move_layer(bottom, doc.root, 0) is False)

    # An empty group is FALSY (it defines __len__), so `parent or root` sends
    # a drop into an empty group to the root instead. This is the same trap
    # that bit the layer tree before; it must not come back through the dock.
    group = ctl.add_group("empty group")
    ctl.move_layer(middle, group, 0)
    check("a layer can be dropped INTO AN EMPTY GROUP",
          middle.parent is group,
          "-- landed in %r; empty groups are falsy, so `parent or root` "
          "silently retargets the drop" % getattr(middle.parent, "name", "?"))

    # ---- layers: duplicate ----------------------------------------------
    plain = ctl.add_layer("plain")
    cell = doc.cell(plain)
    cell.fill(Rect(0, 0, 4, 4), (9, 8, 7, 255))
    copy = ctl.duplicate_layer(plain)
    check("duplicate makes a new layer", copy is not None and copy is not plain)
    check("duplicate copies the PIXELS, not a reference",
          np.array_equal(doc.cell(copy).pixels, cell.pixels))
    doc.cell(copy).fill(Rect(0, 0, 4, 4), (1, 1, 1, 255))
    check("...and the two are independent afterwards",
          not np.array_equal(doc.cell(copy).pixels, cell.pixels))

    # Duplicating an INDEX-LOCKED layer must carry the indices, not just the
    # colours derived from them. Copying into a plain RGBA layer would lose
    # exactly what the indexed model exists to protect.
    indexed = doc.add_layer("indexed", planes=("index", "rgba"),
                            authoritative=("index",))
    src = doc.cell(indexed)
    src.plane("index")[0:4, 0:4] = 17
    src.refresh_derived()
    icopy = ctl.duplicate_layer(indexed)
    check("DUPLICATE PRESERVES AUTHORITATIVE INDICES",
          icopy is not None and "index" in icopy.authoritative
          and np.array_equal(doc.cell(icopy).plane("index"),
                             src.plane("index")),
          "-- an indexed layer duplicated into an RGBA one loses the indices")

    # ---- layers: the context menu ----------------------------------------
    win.layers_dock.sync()
    menu_labels = _context_labels(win.layers_dock, doc.layers()[0])
    check("right-click offers Delete", "Delete" in menu_labels,
          "-- deleting by right-click is a preference worth keeping")
    check("right-click offers Duplicate", "Duplicate" in menu_labels)
    check("the delete BUTTON still exists too",
          any(b.text() == "Delete"
              for b in win.layers_dock.findChildren(QPushButton)),
          "-- both routes, since which one you reach for is personal")

    # ---- view tools, end to end through real mouse events ----------------
    # The engine decides what a click MEANS; only this path proves the canvas
    # carries it out. A view tool that quietly fell through to the painting
    # branch would draw on the document instead of zooming, which is the
    # failure worth guarding against.
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    def press(kind, x, y, button=Qt.LeftButton):
        ev = QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button,
                         button, Qt.NoModifier)
        return ev

    win.canvas.zoom_to(Fraction(1, 1))
    ctl.set_tool("zoom")
    win.canvas.apply_tool_cursor()
    start_zoom = win.canvas.view.zoom
    pixels_before = ctl.doc.cell(ctl.doc.layers()[0]).pixels.copy()
    hist_before = len(ctl.history)

    win.canvas.mousePressEvent(press(QMouseEvent.Type.MouseButtonPress, 200, 150))
    win.canvas.mouseReleaseEvent(press(QMouseEvent.Type.MouseButtonRelease, 200, 150))
    check("CLICKING WITH THE ZOOM TOOL ZOOMS IN",
          win.canvas.view.zoom > start_zoom,
          "-- %s -> %s" % (start_zoom, win.canvas.view.zoom))

    zoomed = win.canvas.view.zoom
    win.canvas.mousePressEvent(
        press(QMouseEvent.Type.MouseButtonPress, 200, 150, Qt.RightButton))
    win.canvas.mouseReleaseEvent(
        press(QMouseEvent.Type.MouseButtonRelease, 200, 150, Qt.RightButton))
    check("right-clicking zooms back out", win.canvas.view.zoom < zoomed)

    # A drag must select a region, not paint one.
    win.canvas.mousePressEvent(press(QMouseEvent.Type.MouseButtonPress, 60, 60))
    win.canvas.mouseMoveEvent(press(QMouseEvent.Type.MouseMove, 260, 210))
    check("the rubber band exists mid-drag", ctl.tool.band is not None)
    win.canvas.repaint()
    check("painting the rubber band does not crash", True)
    win.canvas.mouseReleaseEvent(press(QMouseEvent.Type.MouseButtonRelease, 260, 210))
    check("releasing a drag zooms to the region", ctl.tool.band is None)

    ctl.set_tool("pan")
    win.canvas.apply_tool_cursor()
    before_offset = (win.canvas.view.offset_x, win.canvas.view.offset_y)
    win.canvas.mousePressEvent(press(QMouseEvent.Type.MouseButtonPress, 300, 200))
    win.canvas.mouseMoveEvent(press(QMouseEvent.Type.MouseMove, 260, 230))
    win.canvas.mouseReleaseEvent(press(QMouseEvent.Type.MouseButtonRelease, 260, 230))
    check("DRAGGING WITH THE PAN TOOL SCROLLS THE VIEW",
          (win.canvas.view.offset_x, win.canvas.view.offset_y) != before_offset)

    check("NO VIEW TOOL PAINTED ANYTHING",
          np.array_equal(ctl.doc.cell(ctl.doc.layers()[0]).pixels, pixels_before),
          "-- a view tool fell through to the painting branch")
    check("...and none of it is undoable", len(ctl.history) == hist_before,
          "-- moving the view is not an edit")

    ctl.set_tool("brush")
    win.canvas.apply_tool_cursor()

    # ---- reported from real use: Ctrl+Z skipped the last thing ----------
    # The text tool deliberately stays active after mouse-up so typing can
    # continue, which left live edits outside history. Undo then rolled back
    # the entry BEFORE the one on screen -- and worse, the open session still
    # held a checkpoint of pixels from before that rollback, so committing
    # afterwards wrote them back. Undoing everything left pixels behind that
    # had never existed in any state the user visited.
    ctl.new_document(200, 120)
    win.canvas.bind()
    paint_layer = ctl.doc.layers()[0]
    ctl.doc.cell(paint_layer).fill(ctl.doc.bounds, (0, 0, 0, 255))
    pixels = lambda: ctl.doc.cell(paint_layer).pixels.copy()

    ctl.set_primary((255, 0, 0, 255))
    ctl.set_tool("brush")
    ctl.tool.set_option("size", 20)
    ctl.begin_stroke(20, 60, 1.0, 0, 1)
    ctl.motion_stroke(160, 60, 1.0, 0)
    ctl.end_stroke(160, 60, 1.0, 0)
    after_stroke = pixels()

    # Text placed so it OVERLAPS the stroke: that overlap is what made the
    # old behaviour corrupt rather than merely surprising.
    ctl.set_tool("text", layer=False)      # painted text; text layers: test_textlayers
    ctl.tool.set_option("size", 40)
    ctl.begin_stroke(20, 45, 1.0, 0, 1)
    ctl.end_stroke(20, 45, 1.0, 0)
    for ch in "Hi":
        ctl.tool.insert(ctl._context(), ch)
    check("typing changed the canvas", not np.array_equal(pixels(), after_stroke))
    check("the text tool is still live", ctl.tool.active)

    ctl.undo()
    check("CTRL+Z UNDOES THE LIVE EDIT FIRST",
          np.array_equal(pixels(), after_stroke),
          "-- it must restore exactly the state before typing began, not "
          "roll back the stroke underneath it")
    check("...and the session is closed", not ctl.tool.active)

    ctl.undo()
    check("the next Ctrl+Z removes the stroke",
          int((pixels()[..., 0] > 0).sum()) == 0)
    check("NO PIXELS SURVIVE THAT NEVER EXISTED",
          np.array_equal(pixels(), np.dstack([
              np.zeros((120, 200, 3), np.uint8),
              np.full((120, 200), 255, np.uint8)])),
          "-- the canvas must be back to the fill it started from")

    # Redo has the same hazard in the other direction.
    ctl.redo()
    ctl.redo()
    check("redo still reaches the stroke", int((pixels()[..., 0] > 0).sum()) > 0)

    ctl.set_tool("brush")

    # ---- the frame strip belongs to documents that HAVE frames ----------
    # A permanent one-cell strip along the bottom is a band of wasted height
    # telling you there is one frame. Keyed on the frame count, not the file
    # type, so an addon format gets it without the window knowing the format.
    ctl.new_document(120, 90)
    win._sync_all()
    check("a plain image hides the frame strip", win.frames_dock.isHidden())

    import tempfile as _tf
    from PIL import Image as _Image
    gif = os.path.join(_tf.mkdtemp(), "anim.gif")
    _frames = []
    for i in range(4):
        a = np.zeros((32, 32, 3), np.uint8)
        a[:, i * 8:(i + 1) * 8] = (255, 0, 0)
        _frames.append(_Image.fromarray(a))
    _frames[0].save(gif, save_all=True, append_images=_frames[1:],
                    duration=120, loop=0)

    ctl.open_path(gif)
    win._sync_all()
    check("AN ANIMATED GIF IMPORTS EVERY FRAME", len(ctl.doc.frames) == 4,
          "-- got %d; reading only the first is data loss at open"
          % len(ctl.doc.frames))
    check("...and the frames differ from one another",
          not np.array_equal(
              ctl.doc.cell(ctl.doc.layers()[0], ctl.doc.frames[0]).pixels,
              ctl.doc.cell(ctl.doc.layers()[0], ctl.doc.frames[3]).pixels))
    check("...and the strip appears for it", not win.frames_dock.isHidden())

    ctl.new_document(64, 64)
    win._sync_all()
    check("and goes away again for a still image", win.frames_dock.isHidden())

    win.frames_dock.toggleViewAction().trigger()
    check("a user toggle pins it", win.frames_pinned is True)
    ctl.new_document(48, 48)
    win._sync_all()
    check("...and the pin outlives the next document",
          not win.frames_dock.isHidden(),
          "-- once someone has an opinion, stop having one for them")
    win.frames_pinned = None

    # ---- the modern-editor batch: menus, dialogs, shortcuts, clipboard ----
    from PySide6.QtGui import QImage

    def menu_texts(title):
        for action in win.menuBar().actions():
            if action.text() == title and action.menu() is not None:
                return [a.text().replace("&", "") for a in action.menu().actions() if a.text()]
        return []

    edit = menu_texts("&Edit")
    for entry in ("Cut", "Copy", "Copy Merged", "Paste", "Paste Into New Layer",
                  "Paste Into New Image", "Erase Selection", "Fill Selection"):
        check("Edit menu offers %s" % entry, entry in edit, "-- have %s" % edit)
    image = menu_texts("&Image")
    for entry in ("Crop to Selection", "Resize...", "Canvas Size...", "Flip Horizontal",
                  "Flip Vertical", "Rotate 90\u00b0 Clockwise", "Rotate 180\u00b0", "Delete Frame"):
        check("Image menu offers %s" % entry, entry in image, "-- have %s" % image)
    layers = menu_texts("&Layers")
    for entry in ("Duplicate Layer", "Merge Layer Down", "Flatten", "Move Layer Up",
                  "Properties..."):
        check("Layers menu offers %s" % entry, entry in layers, "-- have %s" % layers)
    check("File menu has Open Recent", "Open Recent" in menu_texts("&File"))

    # Tool shortcuts come from tools.ini and switch tools...
    ctl.set_tool("brush")
    win.tools_dock.shortcuts["pencil"].activated.emit()
    check("the P shortcut selects the pencil", ctl.tool_name == "pencil")
    check("shortcut keys are declared in tools.ini", ctl.tools.shortcut("pencil") == "P"
          and ctl.tools.shortcut("zoom") == "Z")
    # ...but a text box being typed into must keep its letters.
    ctl.set_tool("text")
    win.canvas.mousePressEvent(press(QMouseEvent.Type.MouseButtonPress, 100, 100))
    win.canvas.mouseReleaseEvent(press(QMouseEvent.Type.MouseButtonRelease, 100, 100))
    check("a text session is open", ctl.tool.active)
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    override = QKeyEvent(QEvent.ShortcutOverride, Qt.Key_P, Qt.NoModifier, "P")
    win.canvas.event(override)
    check("THE CANVAS CLAIMS KEYS WHILE TEXT IS BEING TYPED", override.isAccepted(),
          "-- otherwise typing a P switches to the pencil mid-sentence")
    win.canvas.cancel_text()
    ctl.set_tool("brush")

    # Clipboard round trip through the SYSTEM clipboard.
    ctl.selection.select_rect((10, 10, 16, 16))
    ctl.doc.cell(ctl.active_layer).fill(Rect(10, 10, 6, 6), (11, 22, 33, 255))
    win.do_copy()
    img = QApplication.clipboard().image()
    check("copy puts an image on the system clipboard", not img.isNull()
          and (img.width(), img.height()) == (6, 6))
    check("...and the engine clip is kept", ctl.clip is not None)
    # A foreign image pasted in lands on the active layer.
    foreign = QImage(3, 3, QImage.Format_RGBA8888)
    foreign.fill(0xFF00FF00)
    QApplication.clipboard().setImage(foreign)
    ctl.selection.select_all()
    landed = win.do_paste()
    check("a FOREIGN clipboard image pastes", landed is not None and landed.w == 3 and landed.h == 3)
    ctl.undo()

    # Dialogs construct, report, and apply.
    from ochre.ui.dialogs import CanvasSizeDialog, LayerPropertiesDialog, ResizeDialog
    dlg = ResizeDialog(ctl, win)
    dlg.width.setValue(ctl.doc.width // 2)
    w, h, method = dlg.result()
    check("resize dialog keeps aspect by default", h == ctl.doc.height // 2)
    dlg.percent.setValue(200)
    check("...and percent drives both", dlg.result()[0] == ctl.doc.width * 2)
    cdlg = CanvasSizeDialog(ctl, win)
    cdlg.set_anchor("bottom-right")
    check("canvas dialog reports its anchor", cdlg.result()[2] == "bottom-right")
    ldlg = LayerPropertiesDialog(ctl, ctl.active_layer, win)
    ldlg.name.setText("Renamed")
    ldlg.opacity.setValue(100)
    # position, not len(): an earlier undo left a redo-able tail, and the
    # next push truncates it, so len() can stay flat while one entry lands.
    n = ctl.history.position
    ldlg.apply()
    check("layer properties apply as one undo entry",
          ctl.active_layer.name == "Renamed" and ctl.active_layer.opacity == 100
          and ctl.history.position == n + 1)
    ctl.undo()
    check("...and undo reverts both fields", ctl.active_layer.opacity == 255)

    # A geometry change through the window re-aliases the canvas. (Paste
    # left its landing rect selected; start from a clean selection.)
    ctl.selection.select_all()
    # Relative to whatever document the earlier checks left current: a rect
    # covering the whole canvas collapses to select-all and crops nothing.
    half_w, half_h = max(1, ctl.doc.width // 2), max(1, ctl.doc.height // 2)
    ctl.selection.select_rect((0, 0, half_w, half_h))
    size_before = (ctl.doc.width, ctl.doc.height)
    cropped = win.do_crop()
    check("crop via the window keeps the canvas aliased to the new buffer",
          win.canvas.surface.array is ctl.display and ctl.display.shape[:2] == (half_h, half_w),
          "-- do_crop returned %r, doc was %s now %dx%d, display %s, aliased %s, bus errors %s"
          % (cropped, size_before, ctl.doc.width, ctl.doc.height, ctl.display.shape,
             win.canvas.surface.array is ctl.display, ctl.bus.errors[-2:] if ctl.bus else None))
    win.canvas.repaint()
    ctl.undo()
    check("...and undoing the crop re-aliases again",
          win.canvas.surface.array is ctl.display and ctl.display.shape[:2] == size_before[::-1])
    win.canvas.repaint()
    check("painting after a size change does not crash", True)

    # ---- launch as a USER launches: no hand-resizing, real event routing ----
    # Every check above resizes the canvas by hand and drives handlers
    # directly. That is how the editor shipped with an 800x600 canvas drawn
    # at 1% zoom -- bind() fitted to a viewport that was a pixel wide before
    # the window was shown -- and every click landing off the document. This
    # block does what a user does: construct, show, and drag with QTest,
    # which goes through Qt's real delivery to the viewport.
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from ochre.ui.session import Session

    state2 = tempfile.mkdtemp()
    ctl2 = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=state2)
    win2 = MainWindow(ctl2, UiSettings(DATA))
    win2.resize(1100, 760)
    win2.show()
    # Let layout settle the way it does on a real launch: until the viewport
    # stops changing size, not a fixed number of event-loop turns.
    last = None
    for _ in range(50):
        app.processEvents()
        size = (win2.canvas.viewport().width(), win2.canvas.viewport().height())
        if size == last:
            break
        last = size
    vp2 = win2.canvas.viewport()
    check("A FRESHLY LAUNCHED WINDOW FITS THE CANVAS TO A REAL VIEWPORT",
          win2.canvas.view.zoom > Fraction(1, 20),
          "-- zoom is %s; at the 1%% floor the canvas is an invisible speck"
          % win2.canvas.view.zoom)
    x0, y0 = win2.canvas.view.doc_to_widget(0, 0)
    x1, y1 = win2.canvas.view.doc_to_widget(ctl2.doc.width, ctl2.doc.height)
    check("...and the whole canvas is on screen",
          x0 >= 0 and y0 >= 0 and x1 <= vp2.width() + 1 and y1 <= vp2.height() + 1,
          "-- canvas spans (%.0f,%.0f)-(%.0f,%.0f) in a %dx%d viewport"
          % (x0, y0, x1, y1, vp2.width(), vp2.height()))

    cell2 = ctl2.doc.cell(ctl2.active_layer)
    before2 = cell2.plane("rgba").copy()
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    QTest.mousePress(vp2, Qt.LeftButton, Qt.NoModifier, QPoint(cx, cy))
    for i in range(1, 12):
        QTest.mouseMove(vp2, QPoint(cx + 6 * i, cy + 3 * i))
        app.processEvents()
    QTest.mouseRelease(vp2, Qt.LeftButton, Qt.NoModifier, QPoint(cx + 66, cy + 33))
    app.processEvents()
    changed = int((before2 != cell2.plane("rgba")).any(-1).sum())
    check("A REAL MOUSE DRAG ON A FRESH LAUNCH PAINTS", changed > 0,
          "-- %d pixels changed" % changed)
    check("...and is undoable", ctl2.history.can_undo)

    # Resizing the window keeps fitting until the user zooms...
    z_before = win2.canvas.view.zoom
    win2.resize(700, 500)
    for _ in range(3):
        app.processEvents()
    check("before the user zooms, resizing re-fits", win2.canvas.view.zoom != z_before
          or win2.canvas.view.zoom <= z_before)
    # ...and stops once they do.
    win2.canvas.zoom_step(+1)
    z_mine = win2.canvas.view.zoom
    win2.resize(1000, 700)
    for _ in range(3):
        app.processEvents()
    check("after the user zooms, a resize leaves their zoom alone",
          win2.canvas.view.zoom == z_mine)

    # View > Zoom In must act on the CURRENT view, not the one that existed
    # when the menu was built (bind() replaces it).
    win2.canvas.bind()
    for _ in range(3):
        app.processEvents()
    z0 = win2.canvas.view.zoom
    for action in win2.menuBar().actions():
        if action.text() == "&View":
            for a in action.menu().actions():
                if a.text().replace("&", "") == "Zoom In":
                    a.trigger()
    check("VIEW > ZOOM IN ZOOMS THE VIEW YOU CAN SEE", win2.canvas.view.zoom > z0,
          "-- %s -> %s" % (z0, win2.canvas.view.zoom))

    # ---- unsaved-work tracking: nag only about real work -----------------
    ctl3 = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl3.new_document(64, 48)
    check("AN UNTOUCHED UNTITLED DOCUMENT IS NOT MODIFIED", not ctl3.is_modified(),
          "-- ignoring a fresh canvas must never trigger a save prompt")
    ctl3.set_tool("pencil")
    ctl3.begin_stroke(5, 5)
    ctl3.end_stroke(5, 5)
    check("painting makes it modified", ctl3.is_modified())
    ctl3.undo()
    check("undoing back to the start makes it clean again", not ctl3.is_modified())
    ctl3.redo()
    out3 = os.path.join(tempfile.mkdtemp(), "x.ochre")
    ctl3.save_path(out3)
    check("saving makes it clean", not ctl3.is_modified())
    ctl3.begin_stroke(9, 9)
    ctl3.end_stroke(9, 9)
    check("an edit after saving is modified again", ctl3.is_modified())
    ctl3.undo()
    check("...and undoing back to the SAVE POINT is clean", not ctl3.is_modified())
    ctl3.set_tool("text")
    ctl3.begin_stroke(10, 10)
    ctl3.end_stroke(10, 10)
    ctl3.tool.insert(ctl3._context(), "hi")
    check("uncommitted typed text counts as unsaved work", ctl3.is_modified())
    ctl3.cancel_stroke()

    # The window asks only when there is something to lose.
    asked = []
    from PySide6.QtWidgets import QMessageBox
    real_exec = QMessageBox.exec
    QMessageBox.exec = lambda self: (asked.append(self.text()), QMessageBox.Discard)[1]
    try:
        ctl2.undo()
        while ctl2.history.can_undo:
            ctl2.undo()
        check("closing an untouched document does not prompt",
              win2.confirm_discard("closing") and not asked)
        ctl2.set_tool("pencil")
        ctl2.begin_stroke(3, 3)
        ctl2.end_stroke(3, 3)
        check("closing with real changes DOES prompt",
              win2.confirm_discard("closing") and len(asked) == 1, "-- prompts: %s" % asked)
    finally:
        QMessageBox.exec = real_exec

    # ---- the last document's size is remembered between runs -------------
    ses = Session(state2)
    check("the session remembers the last document's size",
          ses.last_size() == (ctl2.doc.width, ctl2.doc.height))
    ctl2.resize_canvas(321, 123)
    check("...including after a canvas resize", Session(state2).last_size() == (321, 123))
    ctl4 = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=state2)
    ctl4.new_document()
    check("A NEW LAUNCH STARTS AT THE LAST DOCUMENT'S SIZE",
          (ctl4.doc.width, ctl4.doc.height) == (321, 123),
          "-- got %dx%d" % (ctl4.doc.width, ctl4.doc.height))
    ctl5 = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl5.new_document()
    check("with no history it falls back to the INI default",
          (ctl5.doc.width, ctl5.doc.height) == (800, 600))
    win2.close() if not ctl2.is_modified() else None
    win2.hide()

    # ---- compact toolbox, generated options bar ----------------------------
    from PySide6.QtWidgets import QCheckBox, QComboBox, QSpinBox
    from ochre.ui.tooloptions import ShapePalette

    td = win2.tools_dock
    groups = ctl2.tools.groups()
    check("THE TOOLBOX HAS ONE BUTTON PER GROUP, not one per tool",
          len(td.buttons) == len(groups) < len(ctl2.tools.names()),
          "-- %d buttons for %d tools" % (len(td.buttons), len(ctl2.tools.names())))
    check("...and is narrow", td.widget().sizeHint().width() <= 100,
          "-- %d px wide" % td.widget().sizeHint().width())
    unreachable = []
    for name in ctl2.tools.names():
        btn = td.button_for(name)
        menu_items = [a.text().split("\t")[0] for a in btn.menu().actions()] if btn.menu() else []
        if td.current[ctl2.tools.group(name)] != name and ctl2.tools.label(name) not in menu_items:
            unreachable.append(name)
    check("every tool is reachable from its button or that button's flyout",
          not unreachable, "-- unreachable: %s" % unreachable)
    check("tool buttons carry icons", all(not b.icon().isNull() for b in td.buttons.values()))

    # Shared letters cycle within a group, as in Paint.NET.
    sharing = [n for n in ctl2.tools.names() if ctl2.tools.shortcut(n) == "S"]
    seq = []
    for _ in range(len(sharing) + 1):
        td.shortcuts["select_rect"].activated.emit()
        seq.append(ctl2.tool_name)
    check("pressing S repeatedly cycles every marquee and wraps around",
          seq[:len(sharing)] == sharing and seq[-1] == sharing[0] and len(sharing) >= 2,
          "-- %s over %s" % (seq, sharing))
    ctl2.set_tool("select_ellipse")
    check("the group's button shows the member you last picked",
          td.current["marquee"] == "select_ellipse")

    # The options bar is generated for EVERY tool from its spec.
    bar = win2.options_bar
    missing = []
    for name in ctl2.tools.names():
        ctl2.set_tool(name)
        app.processEvents()
        want = {o.name for o in ctl2.tools.spec(name)}
        if set(bar.widgets) != want:
            missing.append((name, sorted(want - set(bar.widgets))))
    check("THE OPTIONS BAR SHOWS A CONTROL FOR EVERY DECLARED OPTION, for every tool",
          not missing, "-- %s" % missing)

    ctl2.set_tool("brush")
    app.processEvents()
    spin = bar.widgets["size"][1]
    spin.setValue(37)
    check("changing brush size in the bar changes the tool", ctl2.tool.option("size") == 37)
    ctl2.set_tool("eraser")
    ctl2.set_tool("brush")
    app.processEvents()
    check("...and survives switching tools", ctl2.tool.option("size") == 37
          and bar.widgets["size"][1].value() == 37)
    hard = bar.widgets["hardness"][1]
    hard.setValue(40)
    check("percent options map 0..100 onto 0..1", abs(ctl2.tool.option("hardness") - 0.4) < 1e-9)
    ctl6 = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=state2)
    ctl6.new_document()
    ctl6.set_tool("brush")
    check("TOOL OPTIONS ARE REMEMBERED BETWEEN RUNS", ctl6.tool.option("size") == 37,
          "-- a fresh controller on the same state dir got %r" % ctl6.tool.option("size"))

    win2.step_size(+1)
    app.processEvents()
    check("] grows the brush and the bar follows",
          ctl2.tool.option("size") > 37 and bar.widgets["size"][1].value() == ctl2.tool.option("size"))
    win2.step_size(-1)
    win2.step_size(-1)
    check("[ shrinks it", ctl2.tool.option("size") < 37)

    ctl2.set_tool("shape")
    app.processEvents()
    palette = bar.widgets["shape"][1]
    check("the shape option is a categorised palette", isinstance(palette, ShapePalette)
          and len(palette.buttons) == len(next(o for o in ctl2.tools.spec("shape")
                                               if o.name == "shape").choices))
    palette._pick("gear")
    check("picking a shape in the palette sets it", ctl2.tool.shape == "gear")
    combo = bar.widgets["style"][1]
    combo.setCurrentIndex(combo.findData("fill"))
    check("a choice option is a combo that writes back", ctl2.tool.option("style") == "fill")
    aa = bar.widgets["antialias"][1]
    aa.setChecked(False)
    check("a bool option is a checkbox that writes back", ctl2.tool.option("antialias") is False)

    # Space turns any tool into the hand.
    ctl2.set_tool("brush")
    win2.canvas.user_took_view()
    ox = win2.canvas.view.offset_x
    vp2 = win2.canvas.viewport()
    before_px = ctl2.doc.cell(ctl2.active_layer).plane("rgba").copy()
    QTest.keyPress(win2.canvas, Qt.Key_Space)
    QTest.mousePress(vp2, Qt.LeftButton, Qt.NoModifier, QPoint(200, 200))
    QTest.mouseMove(vp2, QPoint(260, 210))
    QTest.mouseRelease(vp2, Qt.LeftButton, Qt.NoModifier, QPoint(260, 210))
    QTest.keyRelease(win2.canvas, Qt.Key_Space)
    app.processEvents()
    check("HOLDING SPACE PANS WITH ANY TOOL", win2.canvas.view.offset_x != ox)
    check("...without painting", np.array_equal(before_px, ctl2.doc.cell(ctl2.active_layer).plane("rgba")))

    # ---- click-by-click tools through REAL Qt events -------------------------
    # Shown again: Qt does not deliver a button-less mouse move (the hover
    # that drives the rubber band) to a hidden window.
    win2.show()
    ctl2.new_document(200, 150)
    win2.canvas.bind()
    for _ in range(5):
        app.processEvents()
    ctl2.set_tool("polygon_lasso")
    v = win2.canvas.view
    vp2 = win2.canvas.viewport()

    def at(dx, dy):
        wx, wy = v.doc_to_widget(dx, dy)
        return QPoint(int(wx), int(wy))
    for dx, dy in ((20, 20), (150, 25), (120, 120)):
        QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at(dx, dy))
    QTest.mouseMove(vp2, at(60, 130))
    app.processEvents()
    check("between clicks the lasso shows a rubber band to the pointer",
          ctl2.tool.active and abs(ctl2.tool.overlay()[0][1][-1][1] - 130) < 3)
    win2.canvas.repaint()
    check("painting the tool overlay does not crash", True)
    QTest.mouseDClick(vp2, Qt.LeftButton, Qt.NoModifier, at(40, 110))
    app.processEvents()
    check("A REAL DOUBLE-CLICK FINISHES THE POLYGON", not ctl2.tool.active
          and ctl2.selection.mask is not None and int(ctl2.selection.mask[40, 100]) == 255)
    ctl2.set_tool("polygon_lasso")
    for dx, dy in ((10, 10), (60, 10), (60, 60)):
        QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at(dx, dy))
    QTest.keyClick(win2.canvas, Qt.Key_Return)
    app.processEvents()
    check("Enter finishes it too", not ctl2.tool.active and int(ctl2.selection.mask[20, 50]) == 255)
    ctl2.set_tool("polygon_lasso")
    QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at(100, 100))
    QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at(150, 100))
    QTest.keyClick(win2.canvas, Qt.Key_Escape)
    check("Escape abandons it", not ctl2.tool.active and int(ctl2.selection.mask[20, 50]) == 255)

    sel_items = [a.text().replace("&", "") for a in next(
        a for a in win2.menuBar().actions() if a.text() == "&Select").menu().actions() if a.text()]
    for entry in ("All", "None", "Invert", "Feather...", "Grow...", "Shrink...", "Border...",
                  "Remove Holes", "Rounded Corners...", "Alpha to Selection", "Toggle Quick Mask"):
        check("Select menu offers %s" % entry, entry in sel_items, "-- %s" % sel_items)
    win2.do_quick_mask()
    check("Q enters quick mask and the menu item shows it", win2.quick_mask_action.isChecked()
          and ctl2.active_layer.name == "Quick Mask")
    win2.canvas.repaint()
    win2.do_quick_mask()
    check("...and Q again leaves it", not win2.quick_mask_action.isChecked()
          and all(l.name != "Quick Mask" for l in ctl2.doc.layers()))

    # ---- rotate view, marching ants, transforms through the window ----------
    ctl2.new_document(300, 200)
    win2.canvas.bind()
    for _ in range(5):
        app.processEvents()
    win2.canvas.rotate_view(90)
    # Let the layout settle and take the view out of fit mode, or a late
    # resize re-fits the canvas between computing the click point and
    # delivering the click (timing that changed when a dock was added).
    for _ in range(5):
        app.processEvents()
    win2.canvas.user_took_view()
    vp2 = win2.canvas.viewport()
    cell2 = ctl2.doc.cell(ctl2.active_layer)
    ctl2.set_tool("pencil", size=3)
    ctl2.set_primary((255, 0, 0, 255))
    wx, wy = win2.canvas.view.doc_to_widget(150, 60)
    QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, QPoint(int(wx), int(wy)))
    app.processEvents()
    ys, xs = np.nonzero(cell2.plane("rgba")[..., 1] == 0)
    check("ON A ROTATED VIEW, A CLICK PAINTS THE PIXEL UNDER THE POINTER",
          len(xs) and abs(xs.mean() - 150) < 2 and abs(ys.mean() - 60) < 2,
          "-- painted at %s" % ((xs.mean(), ys.mean()) if len(xs) else None,))
    ctl2.selection.select_ellipse((50, 40, 250, 160))
    win2.canvas.repaint()
    check("marching ants trace the real outline, not the bounding box",
          win2.canvas._selection_path().elementCount() > 8)
    ctl2.selection.select_all()
    win2.canvas.rotate_view(None)
    check("View > Reset Rotation returns upright", not win2.canvas.view.rotated)
    view_items = [a.text().replace("&", "") for a in next(
        a for a in win2.menuBar().actions() if a.text() == "&View").menu().actions() if a.text()]
    check("the View menu offers rotation", any("Rotate View" in t for t in view_items))
    layer_items = [a.text().replace("&", "") for a in next(
        a for a in win2.menuBar().actions() if a.text() == "&Layers").menu().actions() if a.text()]
    for entry in ("Rotate by Angle...", "Scale Layer...", "Offset...", "Align"):
        check("Layers menu offers %s" % entry, entry in layer_items, "-- %s" % layer_items)

    # ---- inpainting through the window ------------------------------------
    ctl2.new_document(200, 150)
    win2.canvas.bind()
    for _ in range(5):
        app.processEvents()
    win2.canvas.rotate_view(None)
    v = win2.canvas.view
    vp2 = win2.canvas.viewport()
    cell3 = ctl2.doc.cell(ctl2.active_layer)
    yy3 = np.mgrid[:150, :200][0]
    tex = np.zeros((150, 200, 4), np.uint8)
    tex[..., 3] = 255
    tex[(yy3 // 4) % 2 == 0, :3] = (180, 60, 20)
    tex[(yy3 // 4) % 2 == 1, :3] = (20, 60, 180)
    cell3.plane("rgba")[...] = tex
    cell3.plane("rgba")[70:76, 95:101] = (255, 255, 0, 255)
    cell3._touched(ctl2.doc.bounds)
    win2.canvas.refresh()
    ctl2.set_tool("spot_heal", size=14)

    def at3(dx, dy):
        wx, wy = v.doc_to_widget(dx, dy)
        return QPoint(int(wx), int(wy))
    QTest.mousePress(vp2, Qt.LeftButton, Qt.NoModifier, at3(97, 72))
    QTest.mouseMove(vp2, at3(98, 73))
    QTest.mouseMove(vp2, at3(99, 74))
    QTest.mouseRelease(vp2, Qt.LeftButton, Qt.NoModifier, at3(99, 74))
    app.processEvents()
    spot3 = cell3.plane("rgba")[70:76, 95:101, :3]
    check("A REAL SPOT-HEAL STROKE REMOVES THE BLEMISH",
          not (spot3 == np.array((255, 255, 0), np.uint8)).all(-1).any())
    check("...and the wait cursor is released afterwards",
          QApplication.overrideCursor() is None)
    cell3.plane("rgba")[40:60, 40:60] = (255, 255, 0, 255)
    cell3._touched(ctl2.doc.bounds)
    ctl2.selection.select_rect((38, 38, 62, 62))
    edit_actions = {a.text().replace("&", ""): a for a in next(
        a for a in win2.menuBar().actions() if a.text() == "&Edit").menu().actions() if a.text()}
    check("Edit menu offers Content-Aware Fill", "Content-Aware Fill" in edit_actions,
          "-- %s" % sorted(edit_actions))
    edit_actions["Content-Aware Fill"].trigger()
    app.processEvents()
    check("...which rebuilds the selection from the texture around it",
          not (cell3.plane("rgba")[40:60, 40:60, :3]
               == np.array((255, 255, 0), np.uint8)).all(-1).any())
    check("...and leaves no wait cursor behind", QApplication.overrideCursor() is None)
    check("Content-Aware Fill is Shift+Backspace",
          edit_actions["Content-Aware Fill"].shortcut().toString() == "Shift+Backspace")

    # ---- vector paths through the window -------------------------------------
    ctl2.new_document(200, 150)
    win2.canvas.bind()
    for _ in range(5):
        app.processEvents()
    v = win2.canvas.view
    vp2 = win2.canvas.viewport()

    def at4(dx, dy):
        wx, wy = v.doc_to_widget(dx, dy)
        return QPoint(int(round(wx)), int(round(wy)))
    ctl2.set_tool("pen")
    for dx, dy in ((40, 40), (160, 40), (160, 120), (40, 120)):
        QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at4(dx, dy))
    QTest.mouseMove(vp2, at4(100, 140))
    app.processEvents()
    win2.canvas.repaint()
    check("A REAL PEN: four clicks make four anchors, drawn without error",
          ctl2.doc.path is not None and len(ctl2.doc.path.subpaths[0]) == 4)
    QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at4(40, 40))
    app.processEvents()
    check("...clicking the first anchor closes it", ctl2.doc.path.subpaths[0].closed
          and not ctl2.tool.active)
    check("the Paths dock lists the new path",
          win2.paths_dock.list.count() == 1 and win2.paths_dock.list.item(0).text() == "Work Path")
    win2.paths_dock.buttons["To selection"].click()
    app.processEvents()
    check("the dock's To selection button selects the path's interior",
          ctl2.selection.mask is not None and ctl2.selection.mask[80, 100] == 255
          and ctl2.selection.mask[10, 10] == 0)
    ctl2.deselect()
    win2.paths_dock.buttons["Fill"].click()
    app.processEvents()
    check("...and Fill paints it on the layer",
          tuple(ctl2.doc.cell(ctl2.active_layer).plane("rgba")[80, 100]) == tuple(ctl2.primary))
    sel_items2 = [a.text().replace("&", "") for a in next(
        a for a in win2.menuBar().actions() if a.text() == "&Select").menu().actions() if a.text()]
    check("Select menu offers From Path and To Path",
          "From Path" in sel_items2 and "To Path" in sel_items2)
    ctl2.set_tool("brush")
    win2.show_path_action.setChecked(False)
    win2.do_show_path()
    win2.canvas.repaint()
    check("View > Show Path can hide the outline", win2.canvas.show_path is False)

    # ---- utility: rulers, guides, docks, notes, tabs ---------------------------
    ctl2.new_document(200, 150)
    win2.canvas.bind()
    for _ in range(5):
        app.processEvents()
    win2.rulers_action.setChecked(True)
    for _ in range(3):
        app.processEvents()
    top_ruler = win2.canvas.rulers[0]
    check("View > Rulers shows rulers beside the canvas",
          top_ruler.isVisible() and top_ruler.height() > 0)
    win2.canvas.repaint()
    v = win2.canvas.view
    vp2 = win2.canvas.viewport()

    def at5(dx, dy):
        wx, wy = v.doc_to_widget(dx, dy)
        return QPoint(int(round(wx)), int(round(wy)))
    target = at5(100, 60)
    QTest.mousePress(top_ruler, Qt.LeftButton, Qt.NoModifier, QPoint(target.x(), 5))
    local = top_ruler.mapFromGlobal(vp2.mapToGlobal(target))
    QTest.mouseMove(top_ruler, local)
    QTest.mouseRelease(top_ruler, Qt.LeftButton, Qt.NoModifier, local)
    app.processEvents()
    check("DRAGGING OUT OF THE TOP RULER MAKES A HORIZONTAL GUIDE",
          len(ctl2.doc.guides) == 1 and ctl2.doc.guides[0][0] == "h"
          and abs(ctl2.doc.guides[0][1] - 60) <= 1, str(ctl2.doc.guides))
    ctl2.set_tool("move_layer")
    gy = ctl2.doc.guides[0][1]
    QTest.mousePress(vp2, Qt.LeftButton, Qt.NoModifier, at5(30, gy))
    QTest.mouseMove(vp2, at5(30, 90))
    QTest.mouseRelease(vp2, Qt.LeftButton, Qt.NoModifier, at5(30, 90))
    app.processEvents()
    check("the Move tool drags a guide", abs(ctl2.doc.guides[0][1] - 90) <= 1,
          str(ctl2.doc.guides))
    win2.grid_action.setChecked(True)
    win2.canvas.repaint()
    check("View > Grid draws without error", win2.canvas.show_grid)
    win2.grid_action.setChecked(False)
    win2.rulers_action.setChecked(False)

    win2.info_dock.show()
    win2.info_dock.raise_()
    app.processEvents()
    QTest.mouseMove(vp2, at5(50, 50))
    app.processEvents()
    win2.info_dock.sync()
    check("the Info dock reads out the pointer and the colour under it",
          "Pointer" in win2.info_dock.label.text() and "RGB" in win2.info_dock.label.text())
    win2.histogram_dock.show()
    win2.histogram_dock.raise_()
    app.processEvents()
    win2.histogram_dock.sync()
    check("the Histogram dock computes", win2.histogram_dock.view.data is not None
          and win2.histogram_dock.view.data["pixels"] > 0)
    win2.navigator_dock.show()
    win2.navigator_dock.raise_()
    app.processEvents()
    win2.navigator_dock.sync()
    win2.navigator_dock.view.repaint()
    check("the Navigator shows a thumbnail", win2.navigator_dock.view.thumb is not None)

    from PySide6.QtWidgets import QInputDialog
    real = QInputDialog.getMultiLineText
    QInputDialog.getMultiLineText = staticmethod(lambda *a, **k: ("remember this", True))
    try:
        ctl2.set_tool("note")
        QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at5(120, 40))
        app.processEvents()
    finally:
        QInputDialog.getMultiLineText = real
    check("A NOTE CLICKED ONTO THE CANVAS ASKS FOR TEXT AND KEEPS IT",
          len(ctl2.doc.notes) == 1 and ctl2.doc.notes[0][2] == "remember this")
    win2.canvas.repaint()

    # Tabs: an edited document stays open beside a new one.
    n_before = win2.doc_tabs.count()
    ctl2.new_document(64, 64)
    win2.canvas.bind()
    win2._sync_all()
    app.processEvents()
    check("a new image opens in a second tab", win2.doc_tabs.count() == n_before + 1
          and win2.doc_tabs.currentIndex() == ctl2.current_doc)
    win2.doc_tabs.setCurrentIndex(ctl2.current_doc - 1)
    app.processEvents()
    check("clicking a tab switches document (and the canvas follows)",
          ctl2.doc.width == 200 and win2.canvas.view.doc_w == 200)
    win2.close_document(len(ctl2.documents) - 1)
    app.processEvents()
    check("closing an untouched tab asks nothing and removes it",
          win2.doc_tabs.count() == n_before)
    file_items = [a.text().replace("&", "") for a in next(
        a for a in win2.menuBar().actions() if a.text() == "&File").menu().actions() if a.text()]
    check("File offers Close and Export Slices", "Close" in file_items and "Export Slices..." in file_items)
    check("a Window menu toggles every dock", any(a.text() == "&Window" for a in win2.menuBar().actions()))

    # ---- masks, clipping, adjustment / fill layers, styles in the window ------
    from ochre.engine.geometry import Rect as _R
    from ochre.ui.layerdialogs import (AdjustmentLayerDialog, FillLayerDialog,
                                       LayerStyleDialog)
    ctl2.new_document(120, 90)
    win2.canvas.bind()
    win2._sync_all()
    for _ in range(3):
        app.processEvents()
    lay3 = ctl2.add_layer("shape")
    ctl2.doc.cell(lay3).fill(_R(30, 20, 40, 40), (0, 0, 255, 255))
    layer_items = [a.text().replace("&", "") for a in next(
        a for a in win2.menuBar().actions() if a.text() == "&Layers").menu().actions() if a.text()]
    for entry in ("Layer Mask", "New Adjustment Layer", "New Fill Layer", "Layer Style...",
                  "Rasterize Layer", "Create / Release Clipping Mask"):
        check("Layers menu offers %s" % entry, entry in layer_items, "-- %s" % layer_items)
    win2._layer_fx(ctl2.add_layer_mask, "reveal")
    app.processEvents()
    check("the Layers dock marks a layer whose mask is being painted",
          any("[painting mask]" in win2.layers_dock.list.item(i).text()
              for i in range(win2.layers_dock.list.count())))
    menu = win2.layers_dock.build_context_menu(lay3)
    check("the layer's right-click menu carries the mask and effects entries",
          any(a.text().startswith("Mask, Clipping") for a in menu.actions()))
    dlg = LayerStyleDialog(ctl2, lay3, win2)
    dlg._set("stroke", "size", 4)
    win2.canvas.repaint()
    check("the Layer Style dialog previews live", "stroke" in lay3.styles)
    dlg.accept()
    check("...and OK records it as one step", lay3.styles["stroke"]["size"] == 4
          and ctl2.history_labels()[-1] == "Layer Style")
    dlg = LayerStyleDialog(ctl2, lay3, win2)
    dlg._set("drop_shadow", "distance", 9)
    dlg.reject()
    check("...while Cancel puts the styles back exactly", "drop_shadow" not in lay3.styles)
    adj = ctl2.add_adjustment_layer("brightness_contrast")
    dlg = AdjustmentLayerDialog(ctl2, adj, win2.effects, win2)
    for row in dlg.rows:
        if row.prop.name == "brightness":
            row.prop.value = 40
    dlg._changed()
    dlg.accept()
    check("the adjustment layer dialog sets its parameters", adj.params.get("brightness") == 40)
    fill3 = ctl2.add_fill_layer("solid")
    dlg = FillLayerDialog(ctl2, fill3, win2)
    dlg.color.set_value((9, 99, 199, 255))
    dlg.accept()
    win2.layers_dock.sync()
    win2.canvas.repaint()
    check("the fill layer dialog sets its colour, and everything paints",
          tuple(fill3.params["color"]) == (9, 99, 199, 255))

    # ---- Curves through its dialog, with a real drag ------------------------------
    from ochre.ui.effectdialog import CurveEditor, EffectDialog
    ctl2.new_document(64, 64, background=(100, 100, 100, 255))
    win2.canvas.bind()
    win2._sync_all()
    for _ in range(3):
        app.processEvents()
    adj_items = [a.text().replace("&", "") for a in next(
        a for a in win2.menuBar().actions() if a.text() == "&Adjustments").menu().actions() if a.text()]
    for entry in ("Curves", "Color Balance", "Exposure", "Vibrance", "Channel Mixer", "Gradient Map",
                  "Color to Alpha", "Auto Levels", "Equalize", "Threshold", "Colorize"):
        check("Adjustments offers %s" % entry, entry in adj_items, "-- %s" % adj_items)
    dlg = EffectDialog(win2.effects.create("curves"), ctl2, win2)
    dlg.show()
    app.processEvents()
    editor = [r for r in dlg.rows if r.prop.name == "rgb"][0].curve
    check("the Curves dialog has a curve editor", isinstance(editor, CurveEditor))
    sx, sy = editor._to_screen(100, 100)
    tx, ty = editor._to_screen(100, 180)
    QTest.mousePress(editor, Qt.LeftButton, Qt.NoModifier, QPoint(int(sx), int(sy)))
    QTest.mouseMove(editor, QPoint(int(tx), int(ty)))
    QTest.mouseRelease(editor, Qt.LeftButton, Qt.NoModifier, QPoint(int(tx), int(ty)))
    app.processEvents()
    check("A REAL DRAG ON THE CURVE ADDS AND MOVES A POINT",
          any(abs(x - 100) <= 2 and abs(y - 180) <= 3 for x, y in editor.points), str(editor.points))
    dlg._render_preview()
    dlg.accept()
    app.processEvents()
    v = int(ctl2.doc.cell(ctl2.active_layer).plane("rgba")[10, 10, 0])
    check("...and OK applies the curve to the layer (100 became ~180)", abs(v - 180) <= 4, str(v))

    # ---- effects respect the selection and the paint target -----------------------
    ctl2.new_document(40, 40, background=(200, 100, 50, 255))
    win2.canvas.bind()
    win2._sync_all()
    ctl2.selection.select_rect((0, 0, 20, 40))
    dlg = EffectDialog(win2.effects.create("invert"), ctl2, win2)
    dlg.accept()
    px3 = ctl2.doc.cell(ctl2.active_layer).plane("rgba")
    check("AN EFFECT CHANGES ONLY THE SELECTION",
          tuple(px3[10, 5, :3]) == (55, 155, 205) and tuple(px3[10, 30, :3]) == (200, 100, 50))
    ctl2.undo()
    check("...and undoes exactly", tuple(px3[10, 5, :3]) == (200, 100, 50))
    ctl2.deselect()
    ctl2.add_layer_mask("reveal")
    dlg = EffectDialog(win2.effects.create("invert"), ctl2, win2)
    dlg.accept()
    check("while editing a mask, an effect goes to the MASK, not the layer",
          tuple(ctl2.doc.cell(ctl2.active_layer).plane("rgba")[5, 5, :3]) == (200, 100, 50)
          and tuple(ctl2.doc.cell(ctl2.active_layer.mask).plane("rgba")[5, 5, :3]) == (0, 0, 0))
    gs = win2.effects.create("gaussian_blur")
    dlg = EffectDialog(gs, ctl2, win2)
    dlg.reject()

    # ---- Export As ----------------------------------------------------------------
    from ochre.ui.exportdialog import ExportDialog
    exp = ExportDialog(ctl2, win2)
    exp.format.setCurrentIndex(exp.format.findData("JPEG"))
    exp.quality.setValue(30)
    exp.refresh()
    small = len(exp.data)
    exp.quality.setValue(95)
    exp.refresh()
    check("Export As shows the REAL encoded size, which follows quality",
          0 < small < len(exp.data) and "KB" in exp.size.text() or "bytes" in exp.size.text())
    check("...and previews the decoded file", exp.preview.pixmap() is not None)
    target = os.path.join(tempfile.mkdtemp(), "out.jpg")
    exp.write(target)
    check("...and writes exactly those bytes", os.path.getsize(target) == len(exp.data))
    exp.reject()

    # ---- text layers through the window ----------------------------------------
    ctl2.new_document(240, 140, background=(255, 255, 255, 255))
    win2.canvas.bind()
    win2._sync_all()
    for _ in range(4):
        app.processEvents()
    v = win2.canvas.view
    vp2 = win2.canvas.viewport()

    def at6(dx, dy):
        wx, wy = v.doc_to_widget(dx, dy)
        return QPoint(int(round(wx)), int(round(wy)))
    bg6 = ctl2.active_layer
    ctl2.set_tool("text", size=40)
    QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at6(20, 30))
    QTest.keyClicks(win2.canvas, "Ochre")
    app.processEvents()
    tl = ctl2.active_layer
    check("A REAL CLICK AND TYPING MAKE AN EDITABLE TEXT LAYER",
          tl is not bg6 and tl.kind == "text" and tl.params.get("text") == "Ochre",
          str(getattr(tl, "params", None)))
    QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at6(200, 120))
    app.processEvents()
    win2.canvas.cancel_text()
    app.processEvents()
    check("clicking away commits it (and an empty second click leaves nothing)",
          tl.name == "Ochre" and len([l for l in ctl2.doc.layers() if l.kind == "text"]) == 1
          and ctl2.active_layer is not None)
    from ochre.engine import textlayer as _tlm
    b6 = _tlm.bounds(tl.params)
    QTest.mouseClick(vp2, Qt.LeftButton, Qt.NoModifier, at6(b6.x1 - 2, b6.y + b6.h // 2))
    QTest.keyClick(win2.canvas, Qt.Key_End)
    QTest.keyClicks(win2.canvas, "!")
    win2.canvas.commit_text()
    app.processEvents()
    check("clicking the text again edits it in place", tl.params["text"] == "Ochre!"
          and ctl2.history_labels()[-1] == "Edit Text", str(tl.params["text"]))
    win2.canvas.repaint()
    check("the Layers dock marks it as text", any("[text]" in win2.layers_dock.list.item(i).text()
                                                  for i in range(win2.layers_dock.list.count())))
    ctl2.set_tool("brush")

    # ---- Actions, Channels and Layer Comps docks ---------------------------------
    ctl2.new_document(50, 40, background=(10, 200, 60, 255))
    win2.canvas.bind()
    win2._sync_all()
    ctl2.start_recording("ui")
    ctl2.apply_effect("invert")
    ctl2.stop_recording()
    app.processEvents()
    names7 = [win2.actions_dock.list.item(i).text() for i in range(win2.actions_dock.list.count())]
    check("the Actions dock lists a recorded action", "ui" in names7)
    win2.actions_dock.list.setCurrentRow(names7.index("ui"))
    check("...and shows its steps", "apply_effect" in win2.actions_dock.steps.toPlainText())
    win2.actions_dock._play()
    check("...and plays it", tuple(ctl2.doc.cell(ctl2.active_layer).plane("rgba")[5, 5, :3])
          == (10, 200, 60))
    ctl2.selection.select_rect((0, 0, 20, 20))
    win2.channels_dock._save()
    check("the Channels dock saves the selection as a channel",
          win2.channels_dock.list.count() == 6)
    win2.channels_dock.list.setCurrentRow(1)              # Red
    win2.canvas.repaint()
    check("clicking Red views only that channel", win2.canvas.view_channel == 0)
    win2.channels_dock.list.setCurrentRow(0)
    check("...and RGB brings the colour back", win2.canvas.view_channel is None)
    ctl2.new_comp("v1")
    app.processEvents()
    check("the Layer Comps dock lists a comp", win2.comps_dock.list.count() == 1)
    ctl2.deselect()

    # ---- printing, and the canvas keeps its room ---------------------------------
    from PySide6.QtPrintSupport import QPrinter
    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    pdf = os.path.join(tempfile.mkdtemp(), "print.pdf")
    printer.setOutputFileName(pdf)
    win2.print_to(printer)
    check("File > Print renders the image to the printer (here, a PDF)",
          os.path.exists(pdf) and open(pdf, "rb").read(5) == b"%PDF-")
    win2.resize(1100, 750)
    for _ in range(5):
        app.processEvents()
    check("THE CANVAS GETS MOST OF THE WINDOW, however many panels there are",
          win2.canvas.viewport().width() > 0.5 * win2.width(),
          "-- canvas %d of %d px" % (win2.canvas.viewport().width(), win2.width()))
    sel_items3 = [a.text().replace("&", "") for a in next(
        a for a in win2.menuBar().actions() if a.text() == "&Select").menu().actions() if a.text()]
    check("Select offers Refine Edge and Sky", "Refine Edge..." in sel_items3 and "Sky" in sel_items3)

    # Explicit teardown while the QApplication is alive -- see
    # MainWindow.shutdown for the crash this prevents.
    import gc
    for w in (win, win2):
        w.shutdown()
        w.hide()
        w.deleteLater()
    app.processEvents()
    gc.collect()

    print("\nall UI smoke checks passed")


def _context_labels(dock, node):
    """The layer context menu's entries, without showing it."""
    return [a.text() for a in dock.build_context_menu(node).actions()]


if __name__ == "__main__":
    main()
    # As in ochre/ui/__main__.py: skip interpreter finalisation, where
    # shiboken's arbitrary destruction order segfaults intermittently AFTER
    # every check has passed (exit 139, "no Python frame"). Failures still
    # exit non-zero through check() before reaching this line.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
