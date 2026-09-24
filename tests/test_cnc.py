#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The C&C addon: SHP and TMP, end to end.

The assertion everything else serves: OPENING A FILE AND RE-SAVING IT
UNCHANGED PRODUCES THE SAME BYTES. That is what makes an editor safe to use
on someone's mod, and it is only achievable because indices are authoritative
rather than derived from colour.

Run: python3 tests/test_cnc.py
"""

import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDONS = os.path.join(REPO, "addons")
DATA = os.path.join(REPO, "data")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def controller(state):
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[ADDONS], state_dir=state)
    for ident in list(ctl.addons.addons):
        ctl.addons.trust(ident)
    ctl.load_addons()
    return ctl


def main():
    if not os.path.isdir(os.path.join(ADDONS, "cnc")):
        print("ok: skipped (C&C addon not present)")
        return

    sys.path.insert(0, ADDONS)
    from cnc import cncpal, cncshp, cnctmp

    state = tempfile.mkdtemp()
    work = tempfile.mkdtemp()

    # ---- palettes: the 6-bit expansion --------------------------------
    check("6-bit 63 expands to 255", int(cncpal.expand6(63)) == 255,
          "-- the v<<2 shortcut gives 252 and clips every channel")
    check("6-bit 0 expands to 0", int(cncpal.expand6(0)) == 0)
    check("6-bit 32 expands sensibly", 128 <= int(cncpal.expand6(32)) <= 132)
    every = np.arange(64, dtype=np.uint8)
    back = cncpal.compress8(cncpal.expand6(every))
    check("expand then compress round-trips all 64 levels",
          np.array_equal(back, every))

    raw = bytes(range(64)) * 12
    raw = (raw + b"\0" * 768)[:768]
    pal = cncpal.read_pal(raw, "probe")
    check(".pal parses to 256 entries", len(pal) == 256)
    check(".pal writes back byte-identically",
          cncpal.write_pal(pal) == raw,
          "-- a palette edit must not silently alter untouched entries")
    try:
        cncpal.read_pal(b"too short")
        check("a short .pal is rejected", False, "-- no error")
    except ValueError:
        check("a short .pal is rejected", True)

    # ---- RLE-Zero -------------------------------------------------------
    cases = {
        "all transparent, 16 wide": np.zeros(16, np.uint8),
        "all transparent, 600 wide": np.zeros(600, np.uint8),
        "no transparency": np.arange(1, 49, dtype=np.uint8),
        "trailing run": np.array([3, 4, 5] + [0] * 100, np.uint8),
        "alternating": np.array([0, 7] * 30, np.uint8),
        "single pixel": np.array([42], np.uint8),
    }
    for label, row in cases.items():
        blob = cncshp.encode_rle_line(row)
        got, nxt = cncshp.decode_rle_line(blob, 0, len(row))
        check("RLE-Zero round-trips: %s" % label,
              np.array_equal(got, row) and nxt == len(blob))

    wide = cncshp.encode_rle_line(np.zeros(600, np.uint8))
    check("transparent runs clamp at 255",
          wide.hex(" ") == "08 00 00 ff 00 ff 00 5a",
          "-- got %s; the games misparse an unclamped run" % wide.hex(" "))
    check("the line length counts itself",
          struct.unpack_from("<H", wide, 0)[0] == len(wide))

    # A truncated line must clamp rather than raise -- real files are messy.
    got, _ = cncshp.decode_rle_line(b"\x20\x00\x05", 0, 8)
    check("a truncated line decodes leniently", int(got[0]) == 5)

    # ---- SHP: synthesise, write, read back ------------------------------
    W, H, N = 32, 24, 6
    planes = []
    for i in range(N):
        plane = np.zeros((H, W), np.uint8)
        if i < N // 2:
            plane[4:20, 4 + i:20 + i] = 100 + i
            plane[6:10, 6:10] = 17            # inside the remap ramp
        else:
            plane[6:22, 6 + (i - N // 2):22 + (i - N // 2)] = 1   # shadow
        planes.append(plane)
    planes[2][...] = 0                        # a deliberately empty frame

    blob = cncshp.build_shp(W, H, planes)
    check("SHP was produced", len(blob) > 0)
    check("SHP sniffs as a SHP", cncshp.is_shp(blob))
    check("random bytes do not sniff as SHP", not cncshp.is_shp(b"\x01\x02notshp"))

    zero, width, height, count = struct.unpack_from("<HHHH", blob, 0)
    check("SHP header: leading word is zero", zero == 0)
    check("SHP header: canvas size", (width, height) == (W, H))
    check("SHP header: frame count", count == N)

    _w, _h, frames = cncshp.read_frames(blob)
    check("the empty frame has zero size and zero offset",
          frames[2]["w"] == 0 and frames[2]["h"] == 0 and frames[2]["offset"] == 0,
          "-- this is the canonical encoding of a blank frame")
    check("frames were auto-cropped to their content",
          frames[0]["w"] < W and frames[0]["h"] < H,
          "-- %dx%d" % (frames[0]["w"], frames[0]["h"]))
    check("frame data blocks are 8-byte aligned",
          all(f["offset"] % 8 == 0 for f in frames if f["offset"]))

    # Rebuild full-canvas planes and compare against what went in.
    bad = []
    for i, entry in enumerate(frames):
        rebuilt = np.zeros((H, W), np.uint8)
        if entry["plane"] is not None:
            rebuilt[entry["y"]:entry["y"] + entry["h"],
                    entry["x"]:entry["x"] + entry["w"]] = entry["plane"]
        if not np.array_equal(rebuilt, planes[i]):
            bad.append(i)
    check("EVERY SHP FRAME ROUND-TRIPS BIT-EXACTLY", not bad, "-- frames %r" % bad)

    # Uncompressed frames must decode too.
    plain = cncshp.build_shp(W, H, planes, compress=False)
    _w, _h, plain_frames = cncshp.read_frames(plain)
    check("uncompressed frames decode",
          np.array_equal(plain_frames[0]["plane"], frames[0]["plane"]))

    # ---- SHP through the application ------------------------------------
    path = os.path.join(work, "unit.shp")
    with open(path, "wb") as f:
        f.write(blob)

    ctl = controller(state)
    check("the C&C addon is loaded",
          ctl.addons.addons["cnc"].state == "loaded",
          "-- %s" % ctl.addons.addons["cnc"].error)
    check("SHP is a known extension", "shp" in ctl.addon_extensions())

    ctl.open_path(path)
    doc = ctl.doc
    check("SHP opened through the addon", doc is not None)
    check("canvas matches the file", (doc.width, doc.height) == (W, H))
    check("each SHP frame became a DOCUMENT FRAME", len(doc.frames) == N,
          "-- a multi-frame sprite is not a layer stack")
    check("the sprite layer is index-locked", doc.layers()[0].index_locked)
    check("a palette was bound", doc.palette is not None)
    check("the shadow split was noted",
          doc.meta.get("cnc.shadow_split") == str(N // 2))

    # The annotations the CORE never defined.
    labels = [a[2] for a in doc.palette.annotations]
    check("the addon supplied palette annotations", labels,
          "-- the core renders these without understanding them")
    check("...including the remap ramp",
          any("remap" in a[3] for a in doc.palette.annotations))
    check("index 0 is protected",
          0 in doc.palette.protected and doc.palette.transparent == 0)
    check("the remap ramp is protected",
          all(i in doc.palette.protected for i in range(16, 32)),
          "-- no quantiser may invent a house-colour index")

    layer = doc.layers()[0]
    original = {f.id: doc.cell(layer, f).plane("index").copy() for f in doc.frames}
    check("frame 0 carries its indices",
          int(original[doc.frames[0].id][6, 6]) == 17)

    # ---- THE round-trip -------------------------------------------------
    out = os.path.join(work, "unit-resaved.shp")
    ctl.save_path(out)
    with open(out, "rb") as f:
        resaved = f.read()
    check("RE-SAVING AN UNTOUCHED SHP REPRODUCES THE BYTES",
          resaved == blob,
          "-- %d vs %d bytes" % (len(resaved), len(blob)))

    # ...and again through a second open, to catch state that only survives
    # because it is still in memory.
    ctl2 = controller(state)
    ctl2.open_path(out)
    again = {i: ctl2.doc.cell(ctl2.doc.layers()[0], f).plane("index")
             for i, f in enumerate(ctl2.doc.frames)}
    drift = [i for i, f in enumerate(doc.frames)
             if not np.array_equal(again[i], original[f.id])]
    check("a second open/save cycle is still exact", not drift, "-- %r" % drift)

    # ---- editing then saving --------------------------------------------
    cell = doc.cell(layer, doc.frames[0])
    cell.fill(Rect(0, 0, 4, 4), 200, name="index")
    edited = os.path.join(work, "unit-edited.shp")
    ctl.save_path(edited)
    ctl3 = controller(state)
    ctl3.open_path(edited)
    check("an edit survives the round-trip",
          int(ctl3.doc.cell(ctl3.doc.layers()[0], ctl3.doc.frames[0])
              .plane("index")[1, 1]) == 200)

    # The remap ramp must survive an edit elsewhere in the sprite.
    check("editing elsewhere leaves the remap ramp untouched",
          int(ctl3.doc.cell(ctl3.doc.layers()[0], ctl3.doc.frames[0])
              .plane("index")[7, 7]) == 17,
          "-- a colour-based editor would have smeared this")

    # ---- the shadow-split command ---------------------------------------
    cmd = ctl.addons.registry.get("commands", "cnc.shadow_split")
    report = cmd.run(ctl.addons_host("cnc"))
    check("the shadow-split command reports", "%d frames" % N in report)
    check("...and identifies the shadow frames", "indices 0 and 1" in report)

    # ---- TMP -------------------------------------------------------------
    for cx, cy in ((48, 24), (60, 30)):
        rows = cnctmp.diamond_rows(cx, cy)
        total = sum(w for _y, _x, w in rows)
        check("diamond for %dx%d sums to cx*cy/2" % (cx, cy),
              total == cx * cy // 2, "-- %d vs %d" % (total, cx * cy // 2))
        check("diamond rows stay inside the tile (%dx%d)" % (cx, cy),
              all(0 <= x and x + w <= cx for _y, x, w in rows))

    cx, cy = 48, 24
    src = np.zeros((cy, cx), np.uint8)
    for y, x, w in cnctmp.diamond_rows(cx, cy):
        if w:
            src[y, x:x + w] = np.arange(1, w + 1, dtype=np.uint8)
    packed = cnctmp.encode_diamond(src, cx, cy)
    check("diamond packs to exactly cx*cy/2 bytes", len(packed) == cx * cy // 2)
    check("diamond round-trips exactly",
          np.array_equal(cnctmp.decode_diamond(packed, cx, cy), src))

    # A 2x2 template with one EMPTY slot -- the normal case for a diamond.
    tiles = []
    for slot in range(4):
        if slot == 1:
            tiles.append(None)
            continue
        image = np.zeros((cy, cx), np.uint8)
        zdata = np.full((cy, cx), cnctmp.Z_NONE, np.uint8)
        for y, x, w in cnctmp.diamond_rows(cx, cy):
            if w:
                image[y, x:x + w] = (slot * 20 + y) % 200 + 1
                zdata[y, x:x + w] = (y * 31) // max(1, cy - 1)
        tiles.append({"x": (slot % 2) * cx, "y": (slot // 2) * cy,
                      "height": slot, "land": 13, "ramp": 9,
                      "flags": 0, "extra_x": 0, "extra_y": 0,
                      "radar_low": b"\x10\x20\x30", "radar_high": b"\x40\x50\x60",
                      "image": image, "z": zdata, "extra": None, "extra_z": None})

    header = {"bx": 2, "by": 2, "cx": cx, "cy": cy}
    tmp_blob = cnctmp.build_tmp(header, tiles)
    check("TMP was produced", len(tmp_blob) > 0)
    check("TMP sniffs as a TMP", cnctmp.is_tmp(tmp_blob))
    check("a SHP does not sniff as a TMP", not cnctmp.is_tmp(blob))

    back_header, back_tiles = cnctmp.read_tmp(tmp_blob)
    check("TMP header round-trips", back_header == header)
    check("the empty slot stays empty", back_tiles[1] is None,
          "-- most slots in a diamond template are unused")
    bad = [i for i, t in enumerate(back_tiles)
           if t is not None and not np.array_equal(t["image"], tiles[i]["image"])]
    check("EVERY TMP TILE ROUND-TRIPS BIT-EXACTLY", not bad, "-- tiles %r" % bad)
    bad = [i for i, t in enumerate(back_tiles)
           if t is not None and not np.array_equal(t["z"], tiles[i]["z"])]
    check("Z-DATA ROUND-TRIPS BIT-EXACTLY", not bad, "-- tiles %r" % bad)
    check("per-tile land and ramp survive verbatim",
          back_tiles[0]["land"] == 13 and back_tiles[0]["ramp"] == 9,
          "-- several values share one label; normalising changes behaviour")
    check("both radar colours survive",
          back_tiles[0]["radar_low"] == b"\x10\x20\x30"
          and back_tiles[0]["radar_high"] == b"\x40\x50\x60",
          "-- the engine interpolates between them for the minimap")

    # ---- TMP through the application ------------------------------------
    tmp_path = os.path.join(work, "terrain.tem")
    with open(tmp_path, "wb") as f:
        f.write(tmp_blob)

    ctl4 = controller(state)
    ctl4.open_path(tmp_path)
    tdoc = ctl4.doc
    check("TMP opened through the addon", tdoc is not None)
    check("the canvas is one tile", (tdoc.width, tdoc.height) == (cx, cy))
    check("each tile became a document frame", len(tdoc.frames) == 4)
    check("the axis is laid out as a GRID",
          tdoc.axis_layout == "grid" and tdoc.axis_columns == 2,
          "-- a tile grid is a layout hint, not a second axis")
    tlayer = tdoc.layers()[0]
    check("the tile layer is index-locked", tlayer.index_locked)
    check("Z-DATA RIDES ON A NAMED NON-COLOUR PLANE",
          "height" in tlayer.planes and "height" in tlayer.authoritative,
          "-- this is why the core grew arbitrary uint8 planes")
    check("the empty slot has no cell",
          tdoc.frames[1].meta.get("cnc.empty") == "1")
    check("terrain metadata reached the frame",
          tdoc.frames[0].meta.get("cnc.land") == "13")

    zcell = tdoc.cell(tlayer, tdoc.frames[0])
    check("the height plane carries the z-data",
          int(zcell.plane("height")[cy // 2, cx // 2]) <= 31)

    tmp_out = os.path.join(work, "terrain-resaved.tem")
    ctl4.save_path(tmp_out)
    with open(tmp_out, "rb") as f:
        tmp_resaved = f.read()
    check("RE-SAVING AN UNTOUCHED TMP REPRODUCES THE BYTES",
          tmp_resaved == tmp_blob,
          "-- %d vs %d bytes" % (len(tmp_resaved), len(tmp_blob)))

    cmd = ctl4.addons.registry.get("commands", "cnc.terrain_info")
    info = cmd.run(ctl4.addons_host("cnc"))
    check("the terrain command names the land type", "Clear" in info,
          "-- got %r" % info)
    check("...and the ramp type", "Inner north-west" in info)

    # ---- providers are independent --------------------------------------
    check("both providers registered",
          ctl4.addons.registry.get("formats", "cnc.shp") is not None
          and ctl4.addons.registry.get("formats", "cnc.tmp") is not None)
    check("one addon owns both", ctl4.addons.registry.owner("formats", "cnc.shp")
          == ctl4.addons.registry.owner("formats", "cnc.tmp") == "cnc")

    # ---- and the core still knows nothing -------------------------------
    import subprocess
    hits = subprocess.run(
        ["grep", "-rniE", r"\bshp\b|tiberian|westwood|remap ramp|house colou?r",
         os.path.join(REPO, "ochre"), os.path.join(REPO, "data")],
        capture_output=True, text=True).stdout.strip().splitlines()
    hits = [h for h in hits if ".pyc" not in h and "extensions: tuple" not in h]
    check("THE CORE STILL CONTAINS NO C&C KNOWLEDGE", not hits,
          "-- %r" % hits[:3])

    print("\nall C&C checks passed")


if __name__ == "__main__":
    main()
