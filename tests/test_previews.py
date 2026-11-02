#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Reduced-resolution previews when zoomed out (engine/proxy.py).

The contract under test has two halves, and the second matters more:

  * zoomed out on a big canvas, live previews -- a filter dialog's slider,
    the Layers opacity slider dragged with the mouse -- are drawn from a
    reduced copy of the document, so they cost what the screen shows;
  * what is COMMITTED (OK, mouse release) is computed at full resolution
    exactly as before: the layer pixels and the display buffer are compared
    byte for byte with a full-resolution reference, never approximately.

The UI half drives the real widgets offscreen: the slider is pressed, dragged
and released with mouse events, the effect dialog's own slider is moved.

Run: QT_QPA_PLATFORM=offscreen python3 tests/test_previews.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine import accel, proxy
from ochre.engine.compositor import Compositor
from ochre.engine.document import Document
from ochre.engine.effects import EffectRegistry
from ochre.engine.geometry import Rect
from ochre.engine.layer import ADJUSTMENT, FILL, LayerMask

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def _paint(cell, seed, alpha):
    """Smooth, structured content (a blur of noise would be dull to test)."""
    h, w = cell.height, cell.width
    yy, xx = np.mgrid[0:h, 0:w]
    px = cell.plane("rgba")
    px[..., 0] = (xx * (seed + 3) // 7) % 256
    px[..., 1] = (yy * (seed + 5) // 11) % 256
    px[..., 2] = ((xx + yy) * (seed + 1) // 5) % 256
    px[..., 3] = alpha
    cell._touched(Rect(0, 0, w, h))


def engine_checks():
    # ---- the level for a zoom --------------------------------------------
    from fractions import Fraction
    check("100% and just above 50% preview at full resolution",
          proxy.factor_for_zoom(Fraction(1)) == 1 and proxy.factor_for_zoom(Fraction(51, 100)) == 1)
    check("each halving of the zoom halves the level",
          [proxy.factor_for_zoom(Fraction(1, d)) for d in (2, 3, 4, 8, 9)] == [2, 2, 4, 8, 8])
    check("the level is capped", proxy.factor_for_zoom(Fraction(1, 1000)) == proxy.MAX_FACTOR)
    check("a rect maps outward onto the level grid",
          proxy.reduce_rect(Rect(5, 9, 10, 3), 4) == Rect(1, 2, 3, 1))

    # ---- which effects scale, and how ------------------------------------
    reg = EffectRegistry()
    blur = reg.create("gaussian_blur", radius=20)
    check("a blur radius is divided by the level",
          proxy.scaled_token(blur, 8)["radius"] == 2)
    check("...and kept inside its range", proxy.scaled_token(
        reg.create("box_blur", radius=1), 8)["radius"] == 1)
    check("a pointwise adjustment previews reduced unchanged",
          proxy.scaled_token(reg.create("brightness_contrast", brightness=30), 8)
          == dict(reg.create("brightness_contrast", brightness=30).props.token()))
    unmarked = [n for n in reg.names() if not getattr(reg.create(n), "pointwise", False)
                and getattr(reg.create(n), "proxy_scale", None) is None]
    check("an effect nobody checked keeps its full-resolution preview",
          unmarked and proxy.scaled_token(reg.create(unmarked[0]), 4) is None,
          "-- %r" % unmarked[:3])

    # ---- the proxy compositor at factor 1 IS the compositor ---------------
    # Every source kind goes through the proxy's own path, so at factor 1
    # anything it gets wrong shows up as a pixel difference here.
    doc = Document(96, 72)
    base = doc.add_layer("base")
    _paint(doc.cell(base), 1, 255)
    mul = doc.add_layer("multiply")
    _paint(doc.cell(mul), 4, 170)
    mul.blend = "Multiply"
    group = doc.add_group("group")
    inner = doc.add_layer("inner", parent=group)
    _paint(doc.cell(inner), 7, 120)
    inner.blend = "Screen"
    group.blend = "Overlay"
    fill = doc.add_layer("fill")
    fill.kind = FILL
    fill.params = {"type": "gradient", "start": "200,40,40,160", "end": "20,40,220,90"}
    adj = doc.add_layer("adjust")
    adj.kind = ADJUSTMENT
    adj.params = {"effect": "brightness_contrast", "brightness": 25, "contrast": 10}
    soft = doc.add_layer("soft, masked, stroked")
    doc.cell(soft).fill(Rect(30, 20, 40, 30), (40, 200, 240, 220))
    soft.blend = "SoftLight"
    soft.mask = LayerMask(soft)
    m = doc.cell(soft.mask).plane("rgba")
    m[...] = 255
    m[:, :45, :3] = 0
    doc.cell(soft.mask)._touched(doc.bounds)
    soft.styles = {"stroke": {"size": 2, "color": (0, 0, 255)}}
    clipped = doc.add_layer("clipped")
    doc.cell(clipped).fill(doc.bounds, (255, 0, 255, 255))
    clipped.clip = True
    full = Compositor(doc)
    want = full.composite()
    one = proxy.ProxyCompositor(full, 1).composite()
    check("at factor 1 the proxy composites the document byte for byte",
          np.array_equal(one, want), "-- %d pixels differ" % int((one != want).any(-1).sum()))

    # ---- at factor 4 it is the reduced picture ----------------------------
    four = proxy.ProxyCompositor(full, 4)
    got = four.composite()
    ref = accel.reduce_box(want, 4)
    err = np.abs(got.astype(np.int16) - ref.astype(np.int16)).mean()
    check("at factor 4 the proxy is the reduced document, closely (mean error %.2f)" % err,
          got.shape == ref.shape and err < 3.0)

    # ---- reduced cells follow the document's dirty rects ------------------
    cell = doc.cell(base)
    reduced = four.reduced_cell((base.id, doc.frame.id), cell)
    cell.fill(Rect(10, 10, 20, 9), (0, 255, 0, 255))
    doc.take_dirty(on_cell=four.touch)
    reduced = four.reduced_cell((base.id, doc.frame.id), cell)
    check("a write reaches the reduced cell through its dirty rect alone",
          np.array_equal(reduced, accel.reduce_box(cell.pixels, 4)))


def ui_checks():
    try:
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSlider
    except ImportError as exc:
        print("ok: UI part skipped (PySide6 unavailable: %s)" % exc)
        return
    from ochre.engine.settings import Settings
    from ochre.ui.bus import EventBus
    from ochre.ui.config import UiSettings
    from ochre.ui.controller import Controller
    from ochre.ui.effectdialog import EffectDialog
    from ochre.ui.mainwindow import MainWindow

    app = QApplication.instance() or QApplication([])
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[],
                     state_dir=tempfile.mkdtemp(prefix="ochre-previews-"))
    win = MainWindow(ctl, UiSettings(DATA))
    win.resize(900, 620)
    win.show()
    app.processEvents()

    # Big enough to be worth reducing, small enough for a test: 3000 x 2000.
    W, H = 3000, 2000
    ctl.new_document(W, H, (200, 190, 180, 255))
    for i, mode in enumerate(("Normal", "Multiply", "Screen")):
        layer = ctl.add_layer("L%d" % i)
        _paint(ctl.doc.cell(layer), i, 255 if i == 0 else 170)
        layer.blend = mode
    # An index-locked layer in the stack: its indices must come through a
    # scrub untouched, and its committed composite exact.
    from ochre.engine.palette import grayscale
    if ctl.doc.palette is None:
        ctl.doc.bind_palette(grayscale())
    sprite = ctl.add_layer("sprite", planes=("index", "rgba"), authoritative=("index",))
    scell = ctl.doc.cell(sprite)
    yy, xx = np.mgrid[0:H, 0:W]
    scell.plane("index")[...] = ((xx // 40 + yy // 40) % 7 * 30).astype(np.uint8)
    del yy, xx
    scell.refresh_derived()
    scell._touched(ctl.doc.bounds)
    indices = scell.plane("index").copy()
    sprite.opacity = 140
    win.canvas.bind()
    win.canvas.zoom_fit()
    app.processEvents()
    win.canvas.settle()
    factor = win.canvas.reduction_factor()
    check("Fit to Window on a 3000x2000 canvas previews reduced (1/%d at %.0f%%)"
          % (factor, win.canvas.view.percent), factor >= 2)

    def reference():
        return Compositor(ctl.doc).composite()

    check("settled, the display is the full-resolution composite",
          np.array_equal(ctl.display, reference()))

    # ---- opacity, scrubbed with the mouse --------------------------------
    target = ctl.doc.layer(ctl.doc.root.children[2].id)
    ctl.set_active_layer(target)
    win.layers_dock.sync()
    slider = win.layers_dock.opacity
    opt = QStyleOptionSlider()
    slider.initStyleOption(opt)
    pos = slider.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle,
                                        slider).center()
    before = ctl.display.copy()
    QTest.mousePress(slider, Qt.LeftButton, Qt.NoModifier, pos)
    ticks = []
    x = pos.x()
    for _ in range(3):
        x -= max(4, slider.width() // 10)
        t0 = time.perf_counter()
        QTest.mouseMove(slider, QPoint(x, pos.y()))
        app.processEvents()
        ticks.append((time.perf_counter() - t0) * 1000)
    print("    opacity ticks while dragging: %s" % ", ".join("%.0f ms" % t for t in ticks))
    check("dragging the opacity slider changes the layer", target.opacity < 255,
          "-- opacity %d" % target.opacity)
    check("...shows a reduced preview", win.canvas.showing_reduced)
    check("...and composites nothing at full resolution meanwhile",
          np.array_equal(ctl.display, before))
    preview = win.canvas._reduced.array
    want = reference()
    err = np.abs(preview.astype(np.int16)
                 - accel.reduce_box(want, factor).astype(np.int16)).mean()
    check("the preview is the reduced picture of the new opacity (mean error %.2f)" % err,
          err < 3.0)
    QTest.mouseRelease(slider, Qt.LeftButton, Qt.NoModifier, QPoint(x, pos.y()))
    app.processEvents()
    check("releasing the slider returns to full resolution", not win.canvas.showing_reduced)
    check("...whose display is byte-identical to a full-resolution composite",
          np.array_equal(ctl.display, want))
    check("the index-locked layer's indices are untouched",
          np.array_equal(scell.plane("index"), indices))

    # ---- a filter dialog -------------------------------------------------
    layer = ctl.doc.layer(ctl.doc.root.children[1].id)
    ctl.set_active_layer(layer)
    pristine = ctl.doc.cell(layer).pixels.copy()
    entries = len(ctl.history)
    dlg = EffectDialog(win.effects.create("gaussian_blur", radius=4), ctl, win, debounce_ms=0)
    row = [r for r in dlg.rows if r.prop.name == "radius"][0]
    t0 = time.perf_counter()
    row.slider.setValue(12)
    app.processEvents()
    print("    blur preview after a slider move: %.0f ms" % ((time.perf_counter() - t0) * 1000))
    check("moving the dialog's slider previews reduced", win.canvas.showing_reduced)
    check("...without writing the layer", np.array_equal(ctl.doc.cell(layer).pixels, pristine))
    blurred = win.effects.create("gaussian_blur", radius=12).apply(pristine)
    dlg.reject()
    check("Cancel leaves the layer, the history and the display as they were",
          np.array_equal(ctl.doc.cell(layer).pixels, pristine) and len(ctl.history) == entries
          and np.array_equal(ctl.display, want) and not win.canvas.showing_reduced)

    dlg = EffectDialog(win.effects.create("gaussian_blur", radius=4), ctl, win, debounce_ms=0)
    row = [r for r in dlg.rows if r.prop.name == "radius"][0]
    row.slider.setValue(12)
    app.processEvents()
    dlg.accept()
    app.processEvents()
    committed = ctl.doc.cell(layer).pixels.copy()
    check("OK commits the full-resolution blur, byte for byte",
          np.array_equal(committed, blurred))
    check("...and the display is the full-resolution composite",
          not win.canvas.showing_reduced and np.array_equal(ctl.display, reference()))
    check("...as one undo step", len(ctl.history) == entries + 1)

    # The same edit with reduced previews switched off: today's path.
    ctl.undo()
    win.canvas.settle()
    check("undo restores the layer", np.array_equal(ctl.doc.cell(layer).pixels, pristine))
    off = win.canvas.cfg
    saved = off.get("Canvas", "ReducedPreviews")
    _set(off, "Canvas", "ReducedPreviews", False)
    check("switched off, previews are full resolution", win.canvas.reduction_factor() == 1)
    dlg = EffectDialog(win.effects.create("gaussian_blur", radius=4), ctl, win, debounce_ms=0)
    row = [r for r in dlg.rows if r.prop.name == "radius"][0]
    row.slider.setValue(12)
    app.processEvents()
    check("...and write the layer as they always did",
          not np.array_equal(ctl.doc.cell(layer).pixels, pristine))
    dlg.accept()
    app.processEvents()
    check("the committed layer is identical with and without reduced previews",
          np.array_equal(ctl.doc.cell(layer).pixels, committed))
    _set(off, "Canvas", "ReducedPreviews", saved)


def _set(cfg, section, key, value):
    """Change one setting in memory, for this test only."""
    cfg._cache[(section, key)] = value


def main():
    engine_checks()
    ui_checks()
    print("\nall reduced-preview checks passed")


if __name__ == "__main__":
    main()
