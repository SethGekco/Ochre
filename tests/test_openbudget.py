#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The open budget: a file declaring absurd sizes, or carrying a zip bomb,
is refused quickly, in little memory, with a message meant for a person --
and the cap is PER FRAME, so a file with many frames is never refused for
having them.

Every file here is opened through Controller.open_path, the function File >
Open calls, so the refusal is the one the user would see: the addon host,
the format sniffing and the error wording are all on the path.

Run: python3 tests/test_openbudget.py
"""

import io
import os
import resource
import struct
import sys
import tempfile
import time
import zipfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image

from ochre.engine import budget
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ADDONS = os.path.join(ROOT, "addons")
TMP = tempfile.mkdtemp(prefix="ochre-fileio-budget-")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def controller():
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[ADDONS],
                     state_dir=tempfile.mkdtemp(prefix="ochre-fileio-budget-"))
    for ident in list(ctl.addons.addons):
        ctl.addons.trust(ident)
    ctl.load_addons(all_addons=True)
    return ctl


def put(name, data):
    path = os.path.join(TMP, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---- crafted files ---------------------------------------------------------------

def png_header(width, height):
    """A PNG that is all header: it declares width x height and carries one
    tiny (wrong-sized) IDAT. Enough for any reader that trusts the header."""
    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\0" * 16)) + chunk(b"IEND", b""))


def zeros_deflated(nbytes):
    """zlib of `nbytes` zeros, built a megabyte at a time (the test itself
    must not allocate the bomb it is building)."""
    comp = zlib.compressobj(9)
    out = []
    block = bytes(1 << 20)
    left = nbytes
    while left > 0:
        out.append(comp.compress(block[:min(left, len(block))]))
        left -= len(block)
    out.append(comp.flush())
    return b"".join(out)


def psd(width, height, layers=(), nchan=3):
    """A minimal 8-bit RGB PSD. `layers` is [(rect, [(channel id, comp,
    payload)])]; a raw composite of zeros follows."""
    head = b"8BPS" + struct.pack(">H6xHIIHH", 1, nchan, height, width, 8, 3)
    body = struct.pack(">I", 0) + struct.pack(">I", 0)
    info = b""
    if layers:
        records, data = b"", b""
        for (t, l, b, r), chans in layers:
            rec = struct.pack(">iiiiH", t, l, b, r, len(chans))
            for cid, comp, payload in chans:
                rec += struct.pack(">hI", cid, len(payload) + 2)
                data += struct.pack(">H", comp) + payload
            extra = struct.pack(">II", 0, 0) + b"\x00\x00\x00\x00"   # no mask, no ranges, name ""
            rec += b"8BIMnorm" + bytes((255, 0, 0, 0)) + struct.pack(">I", len(extra)) + extra
            records += rec
        info = struct.pack(">h", len(layers)) + records + data
        info = struct.pack(">I", len(info)) + info
    lm = info + struct.pack(">I", 0) if info else b""
    body += struct.pack(">I", len(lm)) + lm
    composite = struct.pack(">H", 0) + bytes(min(width * height, 64) * nchan)
    return head + body + composite


def zip_bytes(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data, zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def refused(label, ctl, path, words=("megapixels", "pixels")):
    """Open must fail with a FileTooLarge naming the file, fast and small."""
    before = peak_mb()
    t0 = time.time()
    exc = None
    try:
        ctl.open_path(path)
    except Exception as e:                                  # noqa: BLE001
        exc = e
    dt = time.time() - t0
    grew = peak_mb() - before
    check("%s: refused as too large, not crashed (%s)" % (label, type(exc).__name__),
          isinstance(exc, budget.FileTooLarge), repr(exc))
    msg = str(exc)
    check("%s: ...with a message for a person: %r" % (label, msg[:110]),
          msg.startswith("cannot open %s:" % os.path.basename(path))
          and any(w in msg for w in words))
    check("%s: ...in %.2f s and %.0f MB more peak memory" % (label, dt, grew),
          dt < 5.0 and grew < 128, "(%.2f s, +%.0f MB)" % (dt, grew))
    return exc


def main():
    ctl = controller()
    have_cnc = "cnc.shp" in [n for _l, n, _e, _i in ctl.addon_formats()]
    huge = 100000                              # 10 gigapixels: absurd, but 32-bit clean

    # ---- absurd declared sizes, one per reader ---------------------------------------
    refused("PNG declaring 100000 x 100000", ctl, put("huge.png", png_header(huge, huge)))
    # Between one and two times the cap Pillow only WARNS (and the suite runs
    # warnings as errors); the budget must still refuse it, in its own words.
    refused("PNG declaring 25000 x 25000 (just over the cap)", ctl,
            put("over.png", png_header(25000, 25000)))
    refused("PSD declaring a 100000 x 100000 canvas", ctl, put("huge.psd", psd(huge, huge)))
    refused("PSD layer whose rectangle is 2**30 wide", ctl, put("wide.psd", psd(
        64, 64, [((0, 0, 1, 1 << 30), [(0, 0, b"")])])))
    refused(".ochre manifest declaring a 100000 x 100000 canvas", ctl, put("huge.ochre", zip_bytes(
        [("manifest.ini", "[Document]\nWidth = %d\nHeight = %d\n" % (huge, huge))])))
    refused(".ora stack declaring a 100000 x 100000 canvas", ctl, put("huge.ora", zip_bytes(
        [("mimetype", "image/openraster"),
         ("stack.xml", '<image w="%d" h="%d"><stack/></image>' % (huge, huge))])))
    refused(".ora layer PNG declaring 60000 x 60000 in a 16 x 16 canvas", ctl, put("layer.ora", zip_bytes(
        [("mimetype", "image/openraster"),
         ("stack.xml", '<image w="16" h="16"><stack><layer src="data/a.png" name="a"/></stack></image>'),
         ("data/a.png", png_header(60000, 60000))])))

    # ---- decompression bombs -----------------------------------------------------
    bomb = zeros_deflated(256 << 20)           # 256 MB of zeros, about 250 KB deflated
    refused("PSD channel zipped to 256 MB inside a 64 x 64 layer", ctl, put("bomb.psd", psd(
        64, 64, [((0, 0, 64, 64), [(0, 2, bomb)])])), words=("inflates",))
    # A zip member whose header claims 100 bytes but whose stream inflates to
    # 64 MB: inflation stops at the declared size and the CRC fails it.
    buf = bytearray(zip_bytes([("manifest.ini", bytes(64 << 20))]))
    buf[22:26] = struct.pack("<I", 100)                             # local header
    cd = buf.rfind(b"PK\x01\x02")
    buf[cd + 24:cd + 28] = struct.pack("<I", 100)                   # central directory
    lying = put("lying.ochre", bytes(buf))
    before, t0 = peak_mb(), time.time()
    try:
        ctl.open_path(lying)
        exc = None
    except Exception as e:                                  # noqa: BLE001
        exc = e
    check(".ochre whose manifest lies about its size (64 MB declared as 100 bytes) is "
          "rejected in %.2f s, +%.0f MB" % (time.time() - t0, peak_mb() - before),
          exc is not None and time.time() - t0 < 5 and peak_mb() - before < 16, repr(exc))
    with budget.limits(text_bytes=1 << 16):
        refused(".ochre manifest larger than the text budget", ctl, put("bigmanifest.ochre", zip_bytes(
            [("manifest.ini", "[Document]\nWidth = 8\nHeight = 8\n" + " " * (1 << 17))])),
            words=("bytes",))

    # ---- addon readers ---------------------------------------------------------
    if have_cnc:
        shp_head = struct.Struct("<HHHH")
        shp_frame = struct.Struct("<HHHHI4sII")
        refused("sprite declaring a 65535 x 65535 canvas", ctl, put("huge.shp",
                shp_head.pack(0, 65535, 65535, 1) + shp_frame.pack(0, 0, 1, 1, 0, b"\0" * 4, 0, 32) + b"\1"))
        refused("sprite frame of 65535 x 65535 inside a 16 x 16 canvas", ctl, put("frame.shp",
                shp_head.pack(0, 16, 16, 1) + shp_frame.pack(0, 0, 65535, 65535, 0, b"\0" * 4, 0, 32) + b"\1"))
        addon = ctl.addons.addons["cnc"]
        check("a refused file is not counted against the addon that read it",
              addon.state == "loaded" and addon.faults == 0, "(%s, %d faults)" % (addon.state, addon.faults))
        tile = struct.Struct("<iiiiiiiiiIBBB3s3s")
        tmp = (struct.pack("<iiii", 1, 1, 48, 24) + struct.pack("<i", 20)
               + tile.pack(0, 0, 52 + 576, 0, 0, 1 << 30, 0, 1, 1, 1, 0, 0, 0, b"\0" * 3, b"\0" * 3)
               + b"\0" * 3 + bytes(576) + b"\1")
        refused("terrain tile whose extra sits 2**30 pixels away", ctl, put("far.tmp", tmp))
        refused("terrain declaring 60000 x 60000 tiles", ctl, put("huge.tmp",
                struct.pack("<iiii", 1, 1, 60000, 60000) + struct.pack("<i", 0)))
    else:
        print("note: C&C addon not present; its readers not exercised")

    # ---- the default cap admits real large images ----------------------------------
    for w, h, what in ((20000, 20000, "a 20000 x 20000 photo"), (2052, 2816, "a 2052 x 2816 sprite frame"),
                       (16384, 16384, "a 16384 x 16384 texture"), (100000, 4000, "a 100000 x 4000 panorama")):
        try:
            budget.open_image(io.BytesIO(png_header(w, h)))
            ok = True
        except budget.FileTooLarge:
            ok = False
        check("the default per-frame cap admits %s (header checked, not decoded)" % what, ok)

    # ---- PER FRAME: many frames that together exceed one frame's cap ---------------
    # A small cap makes the arithmetic visible: 100 x 100 = 10,000 px per
    # frame. Each file below has frames of 80 x 80 (6,400 px) -- several of
    # them, so their TOTAL is far past one frame's cap -- and must open.
    gif = os.path.join(TMP, "anim.gif")
    frames = [Image.new("RGB", (80, 80), (40 * i, 0, 0)) for i in range(6)]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=50)
    one = os.path.join(TMP, "big1.gif")
    Image.new("RGB", (120, 120), (9, 9, 9)).save(one)

    maker = controller()
    maker.new_document(80, 80, background=(255, 255, 255, 255))
    for i in range(4):
        maker.add_layer("Layer %d" % i)
        cell = maker.doc.cell(maker.active_layer)
        cell.plane("rgba")[5 * i:5 * i + 20, :] = (10 * i, 50, 90, 255)
        cell._touched(maker.doc.bounds)
    layered_psd = os.path.join(TMP, "layers.psd")
    layered_ora = os.path.join(TMP, "layers.ora")
    layered_doc = os.path.join(TMP, "layers.ochre")
    maker.save_path(layered_psd)
    maker.save_path(layered_ora)
    maker.save_path(layered_doc)

    with budget.limits(frame_pixels=10000):
        ctl.open_path(gif)
        check("a 6-frame GIF of 80 x 80 (38,400 px in all) opens under a 10,000 px frame cap",
              len(ctl.doc.frames) == 6)
        for path, label in ((layered_psd, "PSD"), (layered_ora, "OpenRaster"),
                            (layered_doc, ".ochre")):
            ctl.open_path(path)
            n = len([l for l in ctl.doc.layers() if not l.is_group])
            check("a 5-layer %s of 80 x 80 opens under the same cap (%d layers)" % (label, n), n >= 5)
        if have_cnc:
            sys.path.insert(0, ADDONS)
            from cnc import cncshp, cnctmp
            planes = [np.full((80, 80), i + 1, np.uint8) for i in range(8)]
            sprite = put("many.shp", cncshp.build_shp(80, 80, planes))
            ctl.open_path(sprite)
            check("an 8-frame 80 x 80 sprite (51,200 px in all) opens under it",
                  len(ctl.doc.frames) == 8)
            tiles = [{"x": 0, "y": 0, "height": 0, "land": 0, "ramp": 0,
                      "image": np.full((24, 48), i + 1, np.uint8), "z": None, "extra": None}
                     for i in range(16)]
            terrain = put("many.tmp", cnctmp.build_tmp({"bx": 4, "by": 4, "cx": 48, "cy": 24}, tiles))
            with budget.limits(frame_pixels=2000):
                ctl.open_path(terrain)
            check("a 16-tile terrain template (18,432 px in all) opens under a 2,000 px cap",
                  len(ctl.doc.frames) == 16)
        refused("...but ONE 120 x 120 frame (14,400 px) is refused under it", ctl, one)

    print("\nall open-budget checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
