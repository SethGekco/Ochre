#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Colour management: document profiles, soft proofing, gamut warning,
profile conversion, Grayscale, and profiles through import and export.

Run: python3 tests/test_colour.py
"""

import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image

from ochre.engine import colormanage as C
from ochre.engine.fileio import ochre_format, psd
from ochre.engine.fileio.flat import export_flat, import_flat
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def swatches():
    img = np.zeros((8, 32, 4), np.uint8)
    img[..., 3] = 255
    img[:, 0:8] = (0, 255, 0, 255)        # saturated green: outside any press gamut
    img[:, 8:16] = (128, 128, 128, 255)   # grey: inside every gamut
    img[:, 16:24] = (200, 150, 120, 255)  # a skin tone
    img[:, 24:32] = (0, 0, 255, 255)
    return img


def main():
    profiles = C.system_profiles()
    # A WIDE-gamut RGB profile (round trips from sRGB through it are lossless).
    wide_names = ("adobe", "a98", "romm", "prophoto", "best", "wide", "bruce")
    rgb = [p for p in profiles if p[2] == "RGB" and any(k in p[0].lower() or k in p[1].lower()
                                                        for k in wide_names)]
    cmyk = [p for p in profiles if p[2] == "CMYK"]
    img = swatches()

    # ---- the engine ------------------------------------------------------------------
    grey = C.to_gray(img)
    lab = C.to_lab(img)
    lab_g = C.to_lab(grey)
    check("Grayscale keeps each pixel's lightness (max L* error %.1f)"
          % np.abs(lab[..., 0] - lab_g[..., 0]).max(),
          np.abs(lab[..., 0] - lab_g[..., 0]).max() < 1.5 and np.abs(lab_g[..., 1:]).max() < 1.5)
    if rgb:
        wide = C.open_profile(rgb[0][1])
        there = C.convert(img, None, wide)
        back = C.convert(there, wide, None, "relative")
        check("sRGB -> %s -> sRGB comes back (max error %d)" % (rgb[0][0], np.abs(back.astype(int) - img).max()),
              np.abs(back.astype(int) - img.astype(int)).max() <= 3 and not np.array_equal(there, img))
    if cmyk:
        press = C.open_profile(cmyk[0][1])
        oog = C.out_of_gamut(img, press)
        check("gamut check: saturated green is out of %s, grey and skin are in" % cmyk[0][0],
              oog[:, 0:8].all() and not oog[:, 8:24].any())
        vt = C.ViewTransform(proof=press, gamut=True)
        shown = vt.apply(img)
        check("the gamut warning paints out-of-gamut colour grey",
              tuple(shown[0, 0, :3]) == C.GAMUT_COLOUR and tuple(shown[0, 9, :3]) != C.GAMUT_COLOUR)
        soft = C.ViewTransform(proof=press).apply(img)
        check("soft proofing dulls what the press cannot print, and leaves grey grey",
              soft[0, 0, 1] < 250 and np.abs(soft[0, 9, :3].astype(int) - 128).max() <= 6)
    else:
        print("ok: (no CMYK profile installed here; proof and CMYK checks not run)")

    # ---- through files ------------------------------------------------------------------
    d = tempfile.mkdtemp()
    if rgb:
        wide = C.open_profile(rgb[0][1])
        p = os.path.join(d, "tagged.png")
        Image.fromarray(img).save(p, icc_profile=C.profile_bytes(wide))
        doc = import_flat(p)
        check("an imported image keeps its numbers and its profile",
              doc.meta.get("icc") == C.describe(wide)
              and np.array_equal(doc.cell(doc.layers()[0]).plane("rgba"), img))
        f = os.path.join(d, "t.ochre")
        ochre_format.save(doc, f)
        doc2, _ = ochre_format.load(f)
        check(".ochre keeps the profile", C.describe(C.doc_profile(doc2)) == C.describe(wide))
        out = os.path.join(d, "out.png")
        export_flat(doc2, out)
        check("export embeds it", Image.open(out).info.get("icc_profile") == C.profile_bytes(wide))
        pp = os.path.join(d, "t.psd")
        psd.save(doc2, pp)
        check("PSD carries it both ways (resource 1039)",
              psd.read_icc(pp) == C.profile_bytes(wide) and psd.load(pp).meta.get("icc") == C.describe(wide))
    if cmyk:
        press = C.open_profile(cmyk[0][1])
        photo = np.zeros((16, 16, 4), np.uint8)
        photo[..., 3] = 255
        photo[..., 0] = np.linspace(40, 220, 16)[None, :]
        photo[..., 1] = 120
        photo[..., 2] = np.linspace(200, 60, 16)[:, None]
        jp = os.path.join(d, "print.jpg")
        C.to_cmyk_image(photo, press).save(jp, quality=95, icc_profile=C.profile_bytes(press))
        doc = import_flat(jp)
        got = doc.cell(doc.layers()[0]).plane("rgba").astype(int)
        naive = np.asarray(Image.open(jp).convert("RGBA")).astype(int)
        e_icc = np.abs(got[..., :3] - photo[..., :3]).mean()
        e_naive = np.abs(naive[..., :3] - photo[..., :3]).mean()
        check("a CMYK JPEG opens through its profile (error %.1f, Pillow's formula %.1f)"
              % (e_icc, e_naive), e_icc < 12 and e_icc < e_naive)
        doc = import_flat(os.path.join(d, "print.jpg"))
        cj = os.path.join(d, "cmyk_out.jpg")
        export_flat(doc, cj, options={"cmyk_profile": cmyk[0][1]})
        im = Image.open(cj)
        check("Export can write CMYK through a press profile", im.mode == "CMYK"
              and im.info.get("icc_profile") == C.profile_bytes(press))

    # ---- the controller --------------------------------------------------------------------
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(32, 8, background=None)
    lay = ctl.active_layer
    ctl.doc.cell(lay).plane("rgba")[...] = img
    ctl.doc.cell(lay)._touched(ctl.doc.bounds)
    check("a plain sRGB document needs no view transform", ctl.view_transform() is None)
    if cmyk:
        ctl.set_proof(cmyk[0][1], on=True)
        check("Proof Colors gives the canvas a proofing transform", ctl.view_transform() is not None)
        ctl.set_proof(on=False)
        check("...and off again, none", ctl.view_transform() is None)
    if rgb:
        ctl.assign_profile(rgb[0][1])
        check("Assign Profile tags without touching pixels",
              ctl.doc.meta.get("icc") == rgb[0][0] and np.array_equal(ctl.doc.cell(lay).plane("rgba"), img)
              and ctl.view_transform() is not None)
        ctl.undo()
        check("...undoable", "icc" not in ctl.doc.meta)
        ctl.convert_to_profile(rgb[0][1], "relative")
        conv = ctl.doc.cell(lay).plane("rgba").copy()
        back = C.convert(conv, C.doc_profile(ctl.doc), None, "relative")
        check("Convert to Profile changes the numbers so the picture looks the same",
              not np.array_equal(conv, img) and np.abs(back.astype(int) - img.astype(int)).max() <= 3)
        ctl.undo()
        check("...undoable, pixels and tag", np.array_equal(ctl.doc.cell(lay).plane("rgba"), img)
              and "icc" not in ctl.doc.meta)
    ctl.convert_to_grayscale()
    g = ctl.doc.cell(lay).plane("rgba")
    check("Image > Mode > Grayscale", (g[..., 0] == g[..., 1]).all() and (g[..., 1] == g[..., 2]).all())
    ctl.undo()

    # ---- the canvas ------------------------------------------------------------------------
    if cmyk:
        from PySide6.QtWidgets import QApplication
        from ochre.ui.bus import EventBus
        from ochre.ui.config import UiSettings
        from ochre.ui.mainwindow import MainWindow
        app = QApplication.instance() or QApplication([])
        ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=tempfile.mkdtemp())
        win = MainWindow(ctl, UiSettings(DATA))
        ctl.new_document(32, 8, background=None)
        win.canvas.bind()
        ctl.doc.cell(ctl.active_layer).plane("rgba")[...] = img
        ctl.doc.cell(ctl.active_layer)._touched(ctl.doc.bounds)
        ctl.invalidate()
        win.canvas.refresh()
        check("the canvas shows the numbers directly when nothing needs managing",
              win.canvas._cm_buf is None)
        ctl.set_proof(cmyk[0][1], on=True, gamut=True)
        app.processEvents()
        check("Proof Colors + Gamut Warning: the canvas shows the proof, grey where out of gamut",
              win.canvas._cm_buf is not None and tuple(win.canvas._cm_buf[0, 0, :3]) == C.GAMUT_COLOUR)
        win.canvas.viewport().repaint()
        ctl.set_proof(on=False, gamut=False)
        app.processEvents()
        check("...and back", win.canvas._cm_buf is None)
        del app
    print("\nall colour checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
