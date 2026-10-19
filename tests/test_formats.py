#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop, OpenRaster and the flat formats; group isolation.

PSD is checked three ways: Ochre -> PSD -> Ochre keeps every layer property
and the composite exactly; a hand-built 16-bit file decodes; and, when the
real-world sample is present (OCHRE_PSD_SAMPLE, or Rex's RA2 file), Ochre's
composite of it matches the one Photoshop stored inside it. Set OCHRE_GIMP=1
to also have GIMP open what Ochre wrote and compare its flattening.

Run: python3 tests/test_formats.py
"""

import os
import struct
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from ochre.engine.compositor import Compositor
from ochre.engine.document import Document
from ochre.engine.fileio import ora, psd
from ochre.engine.fileio.flat import encode_flat, export_flat
from ochre.engine.geometry import Rect
from ochre.engine.layer import LayerGroup, LayerMask
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAMPLE = os.environ.get("OCHRE_PSD_SAMPLE") or (
    "/mnt/wwn-0x50014ee6b383afb8-part1/(3) Games Data/RA2 & Yuri's Revenge/"
    "Modding and Mapping/Modding/red alert 2 building parts.psd")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def sample_doc():
    doc = Document(96, 72)
    bg = doc.add_layer("Background")
    yy, xx = np.mgrid[:72, :96]
    px = doc.cell(bg).plane("rgba")
    px[..., 0], px[..., 1], px[..., 2], px[..., 3] = (xx * 2) % 256, (yy * 3) % 256, 90, 255
    doc.cell(bg)._touched(doc.bounds)
    mul = doc.add_layer("Multiply été")
    doc.cell(mul).fill(Rect(10, 10, 40, 30), (250, 160, 30, 255))
    mul.blend, mul.opacity = "Multiply", 170
    soft = doc.add_layer("soft")
    doc.cell(soft).fill(Rect(30, 20, 30, 30), (40, 200, 240, 220))
    soft.blend = "SoftLight"
    soft.mask = LayerMask(soft)
    m = doc.cell(soft.mask).plane("rgba")
    m[...] = 255
    m[:, :45, :3] = 0
    doc.cell(soft.mask)._touched(doc.bounds)
    clipped = doc.add_layer("clipped")
    doc.cell(clipped).fill(doc.bounds, (255, 0, 255, 255))
    clipped.clip = True
    hidden = doc.add_layer("hidden")
    doc.cell(hidden).fill(Rect(0, 0, 20, 20), (0, 255, 0, 255))
    hidden.visible = False
    iso = LayerGroup(name="isolated", blend="Normal")
    doc.root.add(iso)
    inner = doc.add_layer("screen inside", parent=iso)
    doc.cell(inner).fill(Rect(55, 35, 35, 30), (60, 220, 60, 255))
    inner.blend = "Screen"
    passing = LayerGroup(name="pass")
    doc.root.add(passing)
    inner2 = doc.add_layer("difference inside", parent=passing)
    doc.cell(inner2).fill(Rect(5, 45, 30, 20), (200, 200, 200, 255))
    inner2.blend = "Difference"
    return doc


def main():
    # ---- group isolation, the semantics both formats depend on ----------------
    doc = Document(20, 20)
    base = doc.add_layer("base")
    doc.cell(base).fill(doc.bounds, (200, 40, 40, 255))
    for blend, want in (("PassThrough", "blended"), ("Normal", "isolated")):
        g = LayerGroup(name="g", blend=blend)
        doc.root.add(g)
        lay = doc.add_layer("screen", parent=g)
        doc.cell(lay).fill(doc.bounds, (40, 120, 230, 255))
        lay.blend = "Screen"
        px = Compositor(doc).composite(doc.bounds)[5, 5, :3]
        if want == "isolated":
            check("an isolated group composites its layers among themselves (Screen on nothing = itself)",
                  tuple(px) == (40, 120, 230), str(px))
        else:
            check("a Pass Through group lets its layers blend with what is beneath",
                  px[0] > 200 and px[2] > 230, str(px))
        doc.remove_layer(g)
    check("new groups are Pass Through, as in Photoshop", LayerGroup().blend == "PassThrough")

    # ---- PSD: Ochre -> PSD -> Ochre ---------------------------------------------------
    doc = sample_doc()
    before = Compositor(doc).composite(doc.bounds)
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "t.psd")
    warnings = psd.save(doc, p)
    check("a PSD saves with nothing to warn about", warnings == [], str(warnings))
    with open(p, "rb") as fh:
        check("...and is a Photoshop file", fh.read(4) == b"8BPS")
    doc2 = psd.load(p)
    after = Compositor(doc2).composite(doc2.bounds)
    check("PSD round trip: the composite is IDENTICAL", np.array_equal(before, after),
          "max %d" % np.abs(before.astype(int) - after.astype(int)).max())
    names = [l.name for l in doc2.layers()]
    check("...layer names, Unicode included", "Multiply été" in names and "hidden" in names, str(names))
    by = {l.name: l for l in doc2.layers()}
    check("...blend modes and opacity", by["Multiply été"].blend == "Multiply"
          and by["Multiply été"].opacity == 170 and by["soft"].blend == "SoftLight")
    check("...visibility, clipping and the mask", not by["hidden"].visible and by["clipped"].clip
          and by["soft"].mask is not None)
    groups = {g.name: g for g in doc2.root.walk() if g.is_group and g is not doc2.root}
    check("...groups, isolated and Pass Through", groups.get("isolated") is not None
          and groups["isolated"].blend == "Normal" and groups["pass"].blend == "PassThrough")
    _w, _h, _layers, stored = psd.read(p)
    a = stored[..., 3:4].astype(float) / 255
    check("the stored composite is what other readers see",
          np.abs(stored[..., :3].astype(float) * a - before[..., :3].astype(float) * a).max() <= 1.01)

    # A hand-built 16-bit, layerless, raw PSD: the composite is the picture.
    h16, w16 = 4, 3
    head = b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, h16, w16, 16, 3)
    planes = b""
    for c, v in enumerate((0x1234, 0xABCD, 0xFFFF)):
        planes += struct.pack(">%dH" % (h16 * w16), *([v] * (h16 * w16)))
    raw = head + struct.pack(">III", 0, 0, 0) + struct.pack(">H", 0) + planes
    p16 = os.path.join(tmp, "deep.psd")
    with open(p16, "wb") as fh:
        fh.write(raw)
    d16 = psd.load(p16)
    check("16-bit PSDs read, reduced to 8 bits",
          tuple(d16.cell(d16.layers()[0]).plane("rgba")[1, 1]) == (0x12, 0xAB, 0xFF, 255))

    # An adjustment layer cannot travel: the saver says so.
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[], state_dir=tempfile.mkdtemp())
    ctl.new_document(40, 30)
    ctl.add_adjustment_layer("invert")
    ctl.save_path(os.path.join(tmp, "adj.psd"))
    check("saving an adjustment layer to PSD WARNS instead of dropping it silently",
          any("adjustment" in w for w in ctl.last_save_warnings), str(ctl.last_save_warnings))
    ctl.open_path(os.path.join(tmp, "t.psd"))
    check("opening a PSD keeps it as a PSD: Save writes it back", ctl.path.endswith("t.psd"))
    ctl.save_path()
    check("...and the file still opens", psd.load(os.path.join(tmp, "t.psd")).width == 96)

    # The real-world sample: Ochre's composite against Photoshop's own.
    if os.path.exists(SAMPLE):
        w, h, layers, comp = psd.read(SAMPLE)
        sdoc = psd.load(SAMPLE)
        mine = Compositor(sdoc).composite(sdoc.bounds)
        pm = lambda x: x[..., :3].astype(float) * x[..., 3:4] / 255.0
        diff = np.abs(pm(mine) - pm(comp)).max()
        check("a real %d-layer PSD composites as Photoshop did (max %.2f)" % (len(layers), diff),
              diff <= 1.5 and np.array_equal(mine[..., 3], comp[..., 3]))
    else:
        print("ok: (real-PSD comparison not run: no sample present)")

    # ---- OpenRaster ---------------------------------------------------------------------
    doc = sample_doc()
    for l in doc.layers():
        l.mask = None
        l.clip = False
    before = Compositor(doc).composite(doc.bounds)
    po = os.path.join(tmp, "t.ora")
    ora.save(doc, po)
    with zipfile.ZipFile(po) as zf:
        first = zf.infolist()[0]
        check("ORA: 'mimetype' comes first, stored, and says image/openraster",
              first.filename == "mimetype" and first.compress_type == zipfile.ZIP_STORED
              and zf.read("mimetype") == b"image/openraster")
        names = zf.namelist()
        check("...with stack.xml, a merged image and a thumbnail",
              "stack.xml" in names and "mergedimage.png" in names
              and "Thumbnails/thumbnail.png" in names)
        stack = zf.read("stack.xml").decode()
        check("...listing the TOP layer first", stack.index('name="pass"') < stack.index('name="Background"'))
    doc2 = ora.load(po)
    after = Compositor(doc2).composite(doc2.bounds)
    check("ORA round trip: the composite is identical", np.array_equal(before, after),
          "max %d" % np.abs(before.astype(int) - after.astype(int)).max())
    groups = {g.name: g.blend for g in doc2.root.walk() if g.is_group and g is not doc2.root}
    check("...isolation survives (isolate vs auto)", groups == {"isolated": "Normal", "pass": "PassThrough"},
          str(groups))
    check("ORA is recognised by content", ora.is_ora(po) and not ora.is_ora(p))

    # ---- flat formats, and Export As's encoder -----------------------------------------------
    doc = sample_doc()
    for ext in ("avif", "qoi", "tga", "ico", "webp", "jp2", "png"):
        target = os.path.join(tmp, "x." + ext)
        export_flat(doc, target)
        im = Image.open(target)
        check("exports and reopens as .%s" % ext, im.size[0] > 0)
    small = encode_flat(doc, "JPEG", {"quality": 20})
    big = encode_flat(doc, "JPEG", {"quality": 95})
    check("Export As: lower quality really is smaller (%d < %d bytes)" % (len(small), len(big)),
          len(small) < len(big))
    half = Image.open(__import__("io").BytesIO(encode_flat(doc, "PNG", {"scale": 0.5})))
    check("...and scale resizes the export, not the document", half.size == (48, 36) and doc.width == 96)
    lossless = Image.open(__import__("io").BytesIO(encode_flat(doc, "WEBP", {"lossless": True}))).convert("RGBA")
    check("...lossless WebP is exact", np.array_equal(np.asarray(lossless),
                                                      Compositor(doc).composite(doc.bounds)))

    # ---- optional: GIMP reads what Ochre wrote ------------------------------------------------
    if os.environ.get("OCHRE_GIMP") and subprocess.call(["which", "gimp"], stdout=subprocess.DEVNULL) == 0:
        # GIMP 2.10 has no clipping masks (it approximates them on import),
        # so the cross-check leaves clipping out; everything else stays.
        gdoc = sample_doc()
        for l in gdoc.layers():
            l.clip = False
        p = os.path.join(tmp, "gimp.psd")
        psd.save(gdoc, p)
        out = os.path.join(tmp, "g.png")
        script = ('(let* ((img (car (gimp-file-load RUN-NONINTERACTIVE "%s" "%s"))) '
                  '(lay (car (gimp-image-merge-visible-layers img CLIP-TO-IMAGE)))) '
                  '(file-png-save RUN-NONINTERACTIVE img lay "%s" "%s" 0 9 0 0 0 0 0))' % (p, p, out, out))
        subprocess.call(["gimp", "-i", "-n", "-b", script, "-b", "(gimp-quit 0)"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=240)
        gimp = np.asarray(Image.open(out).convert("RGBA")).astype(int)
        mine = Compositor(psd.load(p)).composite(psd.load(p).bounds).astype(int)
        check("GIMP flattens Ochre's PSD the same way (max diff %d)" % np.abs(gimp - mine).max(),
              np.abs(gimp - mine).max() <= 2)
    else:
        print("ok: (GIMP cross-check not run: set OCHRE_GIMP=1)")

    print("\nall format checks passed")


if __name__ == "__main__":
    main()
