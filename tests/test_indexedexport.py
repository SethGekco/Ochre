#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""GIF and PNG-8 from File > Export As, read back from the written file.

Each export goes the way a person's does: the File menu's Export As opens
the dialog, its widgets are set, the encoder runs on its own timer, the
Export button writes the file (the system file picker answers with a path
in a temp directory), and the file is then opened with Pillow and its
indices, palette and transparency compared with what was asked for.

The centre of it is EXACTNESS: an indexed document exported on its own
palette must come back with the very same indices -- duplicate palette
colours, transparent index and all -- not a re-quantization that merely
looks the same.

Skips cleanly when PySide6 is absent.

Run: QT_QPA_PLATFORM=offscreen python3 tests/test_indexedexport.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


try:
    from PySide6.QtCore import QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    print("ok: skipped (PySide6 unavailable: %s)" % exc)
    sys.exit(0)

import numpy as np
from PIL import Image

from ochre.engine.palette import LENGTH, Palette
from ochre.engine.settings import Settings
from ochre.ui import exportdialog
from ochre.ui.bus import EventBus
from ochre.ui.config import UiSettings
from ochre.ui.controller import Controller
from ochre.ui.mainwindow import MainWindow
from ochre.ui.prefsdialog import menu_actions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")


def main():
    app = QApplication.instance() or QApplication([])
    state = tempfile.mkdtemp(prefix="ochre-export-")
    out_dir = tempfile.mkdtemp(prefix="ochre-export-out-")
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=state)
    win = MainWindow(ctl, UiSettings(DATA))
    win.show()
    app.processEvents()
    export_action = dict(menu_actions(win.menuBar()))["File/Export As..."]

    # The system file picker, answered: whatever name the dialog proposes,
    # in the temp directory. Remembered so the test sees what was offered.
    offered = {}

    def pick(parent, title, start, filters):
        offered.update(start=start, filters=filters)
        return os.path.join(out_dir, offered["name"]), filters
    exportdialog.QFileDialog.getSaveFileName = staticmethod(pick)

    def export(name, setup):
        """File > Export As, set up by `setup(dialog)`, Export pressed.
        Returns (the dialog, the written file's path)."""
        seen = {}

        def drive():
            dlg = QApplication.activeModalWidget()
            seen["dlg"] = dlg
            setup(dlg)
            QTest.qWait(400)                 # the dialog re-encodes on a timer
            seen["note"] = dlg.indexed_note.text()
            seen["size"] = dlg.size.text()
            seen["data"] = dlg.data
            offered["name"] = name
            dlg.export_button.click()
        QTimer.singleShot(0, drive)
        export_action.trigger()
        return seen, os.path.join(out_dir, name)

    def choose(combo, data):
        combo.setCurrentIndex(combo.findData(data))

    # ---- an indexed document, on its own palette ---------------------------
    entries = np.zeros((LENGTH, 4), np.uint8)
    entries[:, 3] = 255
    rng = np.random.default_rng(7)
    entries[:, :3] = rng.integers(0, 256, (LENGTH, 3))
    entries[40] = entries[41] = (200, 30, 40, 255)        # the same colour twice
    entries[0] = (255, 0, 255, 255)                       # the transparent entry
    pal = Palette(entries, name="Test palette")
    pal.set_transparent(0)
    ctl.new_document(40, 24, background=None)
    ctl.doc.bind_palette(pal)
    lay = ctl.doc.add_layer("art", planes=("index", "rgba"), authoritative=("index",))
    for other in [l for l in ctl.doc.layers() if l is not lay]:
        ctl.doc.remove_layer(other)
    idx = rng.integers(1, 120, (24, 40)).astype(np.uint8)
    idx[:, :10] = 40
    idx[:, 10:20] = 41                                    # looks like the column left of it
    idx[:4, :] = 0                                        # a transparent band
    cell = ctl.doc.cell(lay)
    cell.plane("index")[...] = idx
    cell.content_bbox = ctl.doc.bounds
    cell.refresh_derived()
    ctl.compositor.invalidate()
    check("the document is indexed", ctl.is_indexed_document())

    def png8(dlg):
        choose(dlg.format, "PNG8")
    seen, path = export("sprite.png", png8)
    dlg = seen["dlg"]
    check("Export As offers PNG-8, named so", dlg.format.currentText() == "PNG-8 (.png)")
    check("...with the indexed options showing", dlg.indexed_box.isVisibleTo(dlg))
    check("...and the document's palette chosen for an indexed document",
          dlg.palette.currentData() == "document")
    check("the file picker proposed a .png", offered["start"].endswith(".png"))
    check("the file was written", os.path.isfile(path))
    with open(path, "rb") as fh:
        check("...exactly the bytes the dialog sized", fh.read() == seen["data"])
    img = Image.open(path)
    check("it is a paletted PNG", img.format == "PNG" and img.mode == "P", img.mode)
    check("EXACT: every index is the document's own",
          np.array_equal(np.asarray(img), idx))
    check("...the twin colours stay two indices",
          (np.asarray(img)[4:, :10] == 40).all() and (np.asarray(img)[4:, 10:20] == 41).all())
    back = np.array(img.getpalette("RGB"), np.uint8).reshape(-1, 3)
    check("...and the palette is the document's, entry for entry",
          len(back) == LENGTH and np.array_equal(back, entries[:, :3]))
    check("...with its transparent index", img.info.get("transparency") == 0,
          str(img.info.get("transparency")))
    check("the dialog says the export is exact", seen["note"].startswith("Exact"), seen["note"])

    def gif_doc(dlg):
        choose(dlg.format, "GIF")
    seen, path = export("sprite.gif", gif_doc)
    img = Image.open(path)
    check("GIF on the document's palette: exact indices too",
          img.format == "GIF" and np.array_equal(np.asarray(img), idx))
    check("...and transparency", img.info.get("transparency") == 0)

    def remapped(dlg):
        choose(dlg.format, "PNG8")
        dlg.scale.setValue(50)
    seen, path = export("half.png", remapped)
    img = Image.open(path)
    check("at 50% the picture is no longer its indices: mapped, and said so",
          seen["note"].startswith("Mapped") and img.size == (20, 12), seen["note"])
    check("...still on the document's palette",
          np.array_equal(np.array(img.getpalette("RGB"), np.uint8).reshape(-1, 3),
                         entries[:, :3]))

    lay.opacity = 128
    ctl.compositor.invalidate()

    def faded(dlg):
        choose(dlg.format, "PNG8")
    seen, _path = export("faded.png", faded)
    check("a half-opaque indexed layer is not exact either", seen["note"].startswith("Mapped"),
          seen["note"])
    lay.opacity = 255
    ctl.compositor.invalidate()

    # ---- a full-colour document, palette made for it -----------------------
    ctl.new_document(96, 48, background=None)
    rgb = ctl.doc.cell(ctl.active_layer)
    ramp = np.zeros((48, 96, 4), np.uint8)
    ramp[..., 0] = np.linspace(0, 255, 96)[None, :]
    ramp[..., 1] = np.linspace(0, 255, 48)[:, None]
    ramp[..., 2] = 90
    ramp[..., 3] = 255
    ramp[:8, :8, 3] = 0                                   # clear corner
    ramp[40:, 88:] = (0, 0, 0, 128)                       # half-clear black corner
    rgb.plane("rgba")[...] = ramp
    rgb._touched(ctl.doc.bounds)
    ctl.compositor.invalidate()

    def gif16(dlg):
        choose(dlg.format, "GIF")
        dlg.colours.setValue(16)
        choose(dlg.dither, "none")
    seen, path = export("photo.gif", gif16)
    dlg = seen["dlg"]
    check("a document without a palette cannot pick the document palette",
          not dlg.palette.model().item(1).isEnabled()
          and dlg.palette.currentData() == "generated")
    img = Image.open(path)
    a = np.asarray(img)
    clear = img.info.get("transparency")
    check("GIF with 16 colours: at most 16 indices used, one transparent",
          img.mode == "P" and len(np.unique(a)) <= 16 and clear is not None,
          "%d used, transparency %r" % (len(np.unique(a)), clear))
    check("...the clear corner is the transparent index", (a[:8, :8] == clear).all())
    check("...nothing else is", not (a[8:40, 8:88] == clear).any())
    shown = np.asarray(img.convert("RGB")).astype(int)
    check("...the half-clear black corner is laid over the white matte: grey",
          (80 < shown[44, 92]).all() and (shown[44, 92] < 180).all(), str(shown[44, 92]))
    err = np.abs(shown[8:40, 8:88] - ramp[8:40, 8:88, :3].astype(int)).mean()
    check("...and the colours are a fair approximation", err < 20, "%.1f" % err)
    check("the dialog says how many colours", seen["note"].startswith("16 colours"),
          seen["note"])

    def dithered(mode, amount=100):
        def setup(dlg):
            choose(dlg.format, "PNG8")
            dlg.colours.setValue(4)
            choose(dlg.dither, mode)
            dlg.amount.setValue(amount)
        return setup
    plain = np.asarray(Image.open(export("d0.png", dithered("none"))[1])).copy()
    ordered = np.asarray(Image.open(export("d1.png", dithered("ordered"))[1])).copy()
    floyd = np.asarray(Image.open(export("d2.png", dithered("floyd"))[1])).copy()
    zero = np.asarray(Image.open(export("d3.png", dithered("floyd", 0))[1])).copy()
    check("dithering changes the pixels (pattern and diffusion both)",
          not np.array_equal(plain, ordered) and not np.array_equal(plain, floyd))

    def mean_error(path, want):
        got = np.asarray(Image.open(path).convert("RGB")).astype(int)[8:40, 8:88]
        # Averaged over 8x8 blocks: what the eye sees of a dither.
        g = got.reshape(4, 8, 10, 8, 3).mean((1, 3))
        w = want[8:40, 8:88, :3].astype(int).reshape(4, 8, 10, 8, 3).mean((1, 3))
        return np.abs(g - w).mean()
    e_plain = mean_error(os.path.join(out_dir, "d0.png"), ramp)
    e_ordered = mean_error(os.path.join(out_dir, "d1.png"), ramp)
    e_floyd = mean_error(os.path.join(out_dir, "d2.png"), ramp)
    check("...and brings the local average closer to the original",
          e_ordered < e_plain and e_floyd < e_plain,
          "none %.1f ordered %.1f floyd %.1f" % (e_plain, e_ordered, e_floyd))
    check("amount 0 is no dither at all", np.array_equal(zero, plain))

    def opaque(dlg):
        choose(dlg.format, "PNG8")
        dlg.colours.setValue(32)
        dlg.transparency.setChecked(False)
        dlg.matte.set_value((0, 0, 0))
    seen, path = export("opaque.png", opaque)
    img = Image.open(path)
    shown = np.asarray(img.convert("RGBA")).astype(int)
    check("transparency off: no transparent index written",
          "transparency" not in img.info and (shown[..., 3] == 255).all())
    check("...the clear corner takes the matte colour (black)",
          shown[:8, :8, :3].max() < 40, str(shown[0, 0]))

    win.shutdown()
    shutil.rmtree(state, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    print("\nall indexed export checks passed")


if __name__ == "__main__":
    main()
