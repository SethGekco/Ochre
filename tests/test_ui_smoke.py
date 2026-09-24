#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
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

import os
import sys

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
    from PySide6.QtWidgets import QApplication, QDockWidget
except ImportError as exc:
    print("ok: skipped (PySide6 unavailable: %s)" % exc)
    sys.exit(0)

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
    ctl.deselect()

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

    print("\nall UI smoke checks passed")


if __name__ == "__main__":
    main()
