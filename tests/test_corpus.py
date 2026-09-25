#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The C&C codecs against REAL game files.

Every other test in this suite proves the codecs are self-consistent: what we
write, we can read back. That is necessary and it is not the same as being
correct, because a reader and a writer that share a misunderstanding agree
with each other perfectly. This is the only test that puts the codecs in
front of files Ochre did not author -- the shipped RA2 and Yuri's Revenge
art, written by Westwood's tools twenty-odd years ago.

The assertion is the strict one: READ A FILE, WRITE IT BACK UNCHANGED, GET
THE SAME BYTES. Not "loads without raising", not "looks about right" -- byte
equality, which is the only version of this claim that cannot be fudged.

Point it at a game directory:

    OCHRE_CNC_CORPUS="/path/to/RA2" python3 tests/test_corpus.py

With no corpus it skips, because the files are game assets and cannot be
redistributed with the source. That is also why nothing here is committed as
a fixture: the test travels, the data does not.

OCHRE_CNC_CORPUS_LIMIT caps how many files of each kind are checked (default
400) so the suite stays quick; whatever it drops is reported rather than
quietly ignored.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.geometry import Rect
from ochre.engine.settings import Settings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDONS = os.path.join(REPO, "addons")
DATA = os.path.join(REPO, "data")

# The theater extensions. Note these say which THEATER a file belongs to and
# not what format it is -- see the sniffing check below, which is the whole
# reason this list is shared between the SHP and TMP scans.
THEATER = (".tem", ".sno", ".urb", ".lun", ".des", ".ubn")
CANDIDATES = THEATER + (".shp", ".tmp")


def audit_shp(blob, shp):
    """Check a file against the written spec, independently of our reader.

    Deliberately re-derived from the format description rather than sharing
    code with the codec: a reader and a writer that agree with each other
    prove nothing, which is the whole reason this file exists. It has teeth
    -- run it on the shipped art and it reports the vanilla over-long-run
    quirk, 12 rows in burnt01.tem alone.
    """
    problems = []
    zero, width, height, count = shp.HEADER.unpack_from(blob, 0)
    if zero != 0:
        problems.append("leading word is %d, not 0" % zero)
    base = 8 + 24 * count
    for i in range(count):
        x, y, w, h, flags, _rad, _res, off = shp.FRAME.unpack_from(blob, 8 + 24 * i)
        if not (w and h):
            continue
        if off < base or off >= len(blob):
            problems.append("frame %d: offset %d outside the file" % (i, off))
            continue
        if x + w > width or y + h > height:
            problems.append("frame %d: %dx%d at %d,%d escapes the %dx%d canvas"
                            % (i, w, h, x, y, width, height))
        if not flags & shp.FLAG_RLE:
            if off + w * h > len(blob):
                problems.append("frame %d: raw block runs past EOF" % i)
            continue
        pos = off
        for row in range(h):
            if pos + 2 > len(blob):
                problems.append("frame %d row %d: truncated" % (i, row))
                break
            length = blob[pos] | (blob[pos + 1] << 8)
            end = pos + length
            if length < 2 or end > len(blob):
                problems.append("frame %d row %d: bad length %d" % (i, row, length))
                break
            k, total = pos + 2, 0
            while k < end:
                b = blob[k]
                k += 1
                if b:
                    total += 1
                else:
                    if k >= end:
                        problems.append("frame %d row %d: run marker with no count"
                                        % (i, row))
                        break
                    total += blob[k]
                    k += 1
            if total != w:
                problems.append("frame %d row %d: declares %d pixels, frame is %d wide"
                                % (i, row, total, w))
            pos = end
    return problems


def mutations(good, shp):
    """(name, bytes) for each single-field corruption of a valid file.

    The negative control for audit_shp. Each one breaks a different rule, so
    an audit that silently stopped checking any of them shows up here.
    """
    out = []
    first = next((i for i in range(len(good) // 24)
                  if shp.FRAME.unpack_from(good, 8 + 24 * i)[2]), None)
    if first is None:
        return out
    off = 8 + 24 * first
    x, y, w, h, flags, radar, res, data_off = shp.FRAME.unpack_from(good, off)

    for name, fields in (
            ("an offset pointing past EOF",
             (x, y, w, h, flags, radar, res, len(good) + 999)),
            ("a rect escaping the canvas",
             (x + 8192, y, w, h, flags, radar, res, data_off)),
            ("a frame claiming to be wider than it is",
             (x, y, w + 7, h, flags, radar, res, data_off))):
        broken = bytearray(good)
        shp.FRAME.pack_into(broken, off, *fields)
        out.append((name, bytes(broken)))

    if flags & shp.FLAG_RLE and data_off + 2 <= len(good):
        broken = bytearray(good)
        broken[data_off] = 1              # a scanline length below the minimum
        broken[data_off + 1] = 0
        out.append(("a corrupted scanline length", bytes(broken)))
    return out


def canvas_planes(width, height, frames):
    """Every frame's pixels on a full canvas, so rects need not match."""
    out = []
    for entry in frames:
        full = np.zeros((height, width), dtype=np.uint8)
        plane = entry["plane"]
        if plane is not None:
            fh, fw = plane.shape
            if entry["y"] + fh <= height and entry["x"] + fw <= width:
                full[entry["y"]:entry["y"] + fh,
                     entry["x"]:entry["x"] + fw] = plane
        out.append(full)
    return out


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def walk(root):
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(CANDIDATES) or fn.lower().endswith(".pal"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def main():
    root = os.environ.get("OCHRE_CNC_CORPUS")
    if not root or not os.path.isdir(root):
        print("ok: skipped (set OCHRE_CNC_CORPUS to a game directory)")
        return
    if not os.path.isdir(os.path.join(ADDONS, "cnc")):
        print("ok: skipped (C&C addon not present)")
        return

    sys.path.insert(0, ADDONS)
    from cnc import cncpal, cncshp, cnctmp

    limit = int(os.environ.get("OCHRE_CNC_CORPUS_LIMIT", "400"))
    paths = walk(root)
    check("the corpus has files in it", len(paths) > 0,
          "-- nothing with a C&C extension under %s" % root)

    # ---- sniffing, which the extensions actively lie about ---------------
    shps, tmps, pals, jasc, unknown = [], [], [], [], []
    for p in paths:
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if p.lower().endswith(".pal") and len(data) == 768:
            pals.append(p)
        elif p.lower().endswith(".pal") and data[:8] == b"JASC-PAL":
            jasc.append(p)
        elif cncshp.is_shp(data):
            shps.append(p)
        elif cnctmp.is_tmp(data):
            tmps.append(p)
        else:
            unknown.append(p)

    # A directory of units has no terrain and a theater directory has no
    # sprites, so neither kind is required -- but finding NOTHING means the
    # corpus is wrong or the sniffers are broken, and either way the strict
    # checks below would pass vacuously.
    check("the sniffers recognised something",
          shps or tmps,
          "-- %d files, none of them a SHP or a TMP" % len(paths))
    check("no file is claimed by both sniffers",
          not (set(shps) & set(tmps)))

    # The extension names the theater, not the format: terrain templates and
    # theater-specific sprites share .tem/.sno/.urb. A sniffer that trusted
    # the extension would hand a SHP to the TMP reader on the first one.
    mislabelled = [p for p in shps if p.lower().endswith(THEATER)]
    if mislabelled:
        check("SHPs HIDE BEHIND THEATER EXTENSIONS, and sniffing catches them",
              True, "-- %d of them" % len(mislabelled))

    print("corpus: %d SHP, %d TMP, %d PAL, %d JASC-PAL, %d other (of %d files)"
          % (len(shps), len(tmps), len(pals), len(jasc), len(unknown),
             len(paths)))

    # ---- palettes ---------------------------------------------------------
    bad = []
    for p in pals[:limit]:
        with open(p, "rb") as f:
            data = f.read()
        if cncpal.write_pal(cncpal.read_pal(data, "p")) != data:
            bad.append(p)
    check("EVERY REAL PALETTE ROUND-TRIPS BYTE-EXACTLY", not bad,
          "-- %d of %d failed, e.g. %s"
          % (len(bad), len(pals[:limit]),
             os.path.basename(bad[0]) if bad else ""))

    # JASC-PAL is the TEXT palette format, sharing the .pal extension. Read
    # as binary it does not fail -- it turns ASCII digits into 6-bit colour
    # and yields confident nonsense. So the check is that the parsed entries
    # match the numbers actually written in the file.
    bad = []
    for p in jasc[:limit]:
        with open(p, "rb") as f:
            rows = f.read().decode("ascii", "replace").splitlines()
        want = [int(v) for v in rows[3].split()[:3]]
        got = list(cncpal.load_pal_file(p).entries[0, :3])
        if want != got:
            bad.append("%s: first entry %s, parsed %s"
                       % (os.path.basename(p), want, got))
    if jasc:
        check("JASC-PAL TEXT PALETTES PARSE AS TEXT, not as binary garbage",
              not bad, "-- %s" % "; ".join(bad[:4]))

    # ---- SHP --------------------------------------------------------------
    dropped = max(0, len(shps) - limit)
    bad, frames_seen = [], 0
    for p in shps[:limit]:
        with open(p, "rb") as f:
            data = f.read()
        try:
            width, height, frames = cncshp.read_frames(data)
            frames_seen += len(frames)
            planes, bounds, radar = [], [], []
            for entry in frames:
                full = np.zeros((height, width), dtype=np.uint8)
                plane = entry["plane"]
                if plane is not None:
                    fh, fw = plane.shape
                    if entry["y"] + fh <= height and entry["x"] + fw <= width:
                        full[entry["y"]:entry["y"] + fh,
                             entry["x"]:entry["x"] + fw] = plane
                planes.append(full)
                bounds.append(Rect(entry["x"], entry["y"], entry["w"], entry["h"])
                              if plane is not None else None)
                radar.append(entry["radar"])
            rebuilt = cncshp.build_shp(
                width, height, planes, radar_colours=radar, bounds=bounds,
                source=frames, tail=cncshp.trailing_bytes(data, frames))
        except Exception as exc:                 # noqa: BLE001 - report, don't mask
            bad.append((p, "%s: %s" % (type(exc).__name__, exc)))
            continue
        if rebuilt != data:
            n = min(len(rebuilt), len(data))
            at = next((i for i in range(n) if rebuilt[i] != data[i]), n)
            bad.append((p, "len %d -> %d, first difference at 0x%x"
                        % (len(data), len(rebuilt), at)))

    if shps:
        check("EVERY REAL SHP ROUND-TRIPS BYTE-EXACTLY", not bad,
              "-- %d of %d failed:\n    %s"
              % (len(bad), len(shps[:limit]),
                 "\n    ".join("%s  %s" % (os.path.basename(p), why)
                               for p, why in bad[:8])))
        print("   %d SHP files, %d frames" % (len(shps[:limit]), frames_seen)
              + (" (%d more not checked; raise OCHRE_CNC_CORPUS_LIMIT)" % dropped
                 if dropped else ""))

    # ---- TMP --------------------------------------------------------------
    dropped = max(0, len(tmps) - limit)
    bad, tiles_seen = [], 0
    for p in tmps[:limit]:
        with open(p, "rb") as f:
            data = f.read()
        try:
            header, tiles = cnctmp.read_tmp(data)
            tiles_seen += sum(1 for t in tiles if t is not None)
            rebuilt = cnctmp.build_tmp(header, tiles)
        except Exception as exc:                 # noqa: BLE001
            bad.append((p, "%s: %s" % (type(exc).__name__, exc)))
            continue
        if rebuilt != data:
            n = min(len(rebuilt), len(data))
            at = next((i for i in range(n) if rebuilt[i] != data[i]), n)
            bad.append((p, "len %d -> %d, first difference at 0x%x"
                        % (len(data), len(rebuilt), at)))

    if tmps:
        check("EVERY REAL TMP ROUND-TRIPS BYTE-EXACTLY", not bad,
              "-- %d of %d failed:\n    %s"
              % (len(bad), len(tmps[:limit]),
                 "\n    ".join("%s  %s" % (os.path.basename(p), why)
                               for p, why in bad[:8])))
        print("   %d TMP files, %d tiles" % (len(tmps[:limit]), tiles_seen)
              + (" (%d more not checked; raise OCHRE_CNC_CORPUS_LIMIT)" % dropped
                 if dropped else ""))

    # ---- OUR encoder, on its own ------------------------------------------
    # Everything above proves we can reproduce a file. That is a test of the
    # preservation path, and it deliberately avoids re-encoding -- so it says
    # nothing about the encoder a user actually invokes the moment they paint
    # a single pixel. This runs that path over the same real artwork: encode
    # from scratch, with none of the original's choices, then require the
    # bytes to be spec-clean and to decode back to exactly the same pixels.
    limit_enc = int(os.environ.get("OCHRE_CNC_ENCODE_LIMIT", "300"))
    dropped = max(0, len(shps) - limit_enc)
    bad, frames_done = [], 0
    for p in shps[:limit_enc]:
        with open(p, "rb") as f:
            data = f.read()
        try:
            width, height, frames = cncshp.read_frames(data)
            before = canvas_planes(width, height, frames)
            ours = cncshp.build_shp(width, height, before)   # no source=
            w2, h2, back = cncshp.read_frames(ours)
            after = canvas_planes(w2, h2, back)
        except Exception as exc:                 # noqa: BLE001
            bad.append((p, "%s: %s" % (type(exc).__name__, exc)))
            continue
        frames_done += len(frames)
        if (w2, h2) != (width, height) or len(back) != len(frames):
            bad.append((p, "shape changed: %dx%d/%d -> %dx%d/%d"
                        % (width, height, len(frames), w2, h2, len(back))))
            continue
        if not all(np.array_equal(a, b) for a, b in zip(before, after)):
            bad.append((p, "pixels changed through our own encoder"))
            continue
        problems = audit_shp(ours, cncshp)
        if problems:
            bad.append((p, "%d spec violations: %s" % (len(problems), problems[:2])))

    if shps:
        check("WHAT OUR ENCODER WRITES IS SPEC-CLEAN AND LOSSLESS", not bad,
              "-- %d of %d failed:\n    %s"
              % (len(bad), len(shps[:limit_enc]),
                 "\n    ".join("%s  %s" % (os.path.basename(p), why)
                               for p, why in bad[:8])))
        print("   re-encoded %d files, %d frames, from scratch"
              % (len(shps[:limit_enc]), frames_done)
              + (" (%d more not checked; raise OCHRE_CNC_ENCODE_LIMIT)" % dropped
                 if dropped else ""))

    # A NEGATIVE CONTROL, because "spec-clean" is worthless unless the audit
    # can fail. Corrupt one byte of a known-good file in three different ways
    # and require each to be caught. This is deliberately synthetic rather
    # than "some real files trip it": whether the corpus happens to contain
    # quirky art depends on which directory it was pointed at, and a check
    # whose teeth come and go with the input is not a check.
    if shps:
        with open(shps[0], "rb") as f:
            width, height, frames = cncshp.read_frames(f.read())
        good = cncshp.build_shp(width, height,
                                canvas_planes(width, height, frames))
        check("the audit passes a file we just wrote",
              not audit_shp(good, cncshp))

        missed = [name for name, broken in mutations(good, cncshp)
                  if not audit_shp(broken, cncshp)]
        check("THE AUDIT CATCHES DELIBERATE CORRUPTION", not missed,
              "-- these went undetected, so a clean verdict above proves "
              "nothing: %s" % ", ".join(missed))

    # ---- through the actual editor ---------------------------------------
    # The byte checks above drive the codec functions. A user drives the
    # PROVIDER: open the file into a document, save the document back. That
    # path also crosses the index-locked layer, the frame axis and the
    # metadata, so it is the one that proves the document model does not
    # quietly lose something on the way through.
    from ochre.ui.controller import Controller

    state = tempfile.mkdtemp()
    work = tempfile.mkdtemp()
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[ADDONS], state_dir=state)
    for ident in list(ctl.addons.addons):
        ctl.addons.trust(ident)
    ctl.load_addons()

    bad = []
    sample = shps[:25] + tmps[:25]
    for p in sample:
        with open(p, "rb") as f:
            original = f.read()
        # Keep the original extension: the provider is chosen by extension
        # first, and a theater-suffixed SHP has to stay one.
        out = os.path.join(work, "rt_" + os.path.basename(p))
        try:
            ctl.open_path(p)
            ctl.save_path(out)
        except Exception as exc:                 # noqa: BLE001
            bad.append((p, "%s: %s" % (type(exc).__name__, exc)))
            continue
        with open(out, "rb") as f:
            if f.read() != original:
                bad.append((p, "differs after a load/save round trip"))

    check("OPENING AND RE-SAVING A REAL FILE CHANGES NOTHING", not bad,
          "-- %d of %d failed:\n    %s"
          % (len(bad), len(sample),
             "\n    ".join("%s  %s" % (os.path.basename(p), why)
                           for p, why in bad[:8])))

    # Painting a REAL extra, with real offsets. The synthetic fixture proves
    # the mechanism; this proves it against the geometry that actually ships,
    # where the block starts above the tile and overlaps it.
    with_extra = None
    for p in tmps[:limit]:
        with open(p, "rb") as f:
            _h, ts = cnctmp.read_tmp(f.read())
        hit = next((i for i, t in enumerate(ts)
                    if t is not None and t["extra"] is not None), None)
        if hit is not None:
            with_extra = (p, hit)
            break

    if with_extra is not None:
        p, slot = with_extra
        doc = ctl.open_path(p)
        elayer = next((l for l in doc.layers() if l.name == "Extra"), None)
        frame = next(f for f in doc.frames if f.meta.get("cnc.slot") == str(slot))
        ox = int(doc.meta["cnc.origin_x"])
        oy = int(doc.meta["cnc.origin_y"])
        ex = ox + int(frame.meta["cnc.extra_x"]) - int(frame.meta["cnc.x"])
        ey = oy + int(frame.meta["cnc.extra_y"]) - int(frame.meta["cnc.y"])
        cell = doc.cells[(elayer.id, frame.id)]
        whole = cell.plane("index").copy()
        before = whole[ey:ey + 4, ex:ex + 6]
        # Paint a value the block does not already contain, or the assertion
        # would pass without the stroke ever having happened.
        ink = next(v for v in range(1, 256) if not (before == v).any())
        cell.plane("index")[ey:ey + 4, ex:ex + 6] = ink
        cell.refresh_derived()

        out = os.path.join(work, "painted_" + os.path.basename(p))
        ctl.save_path(out)
        with open(out, "rb") as f:
            _h, after = cnctmp.read_tmp(f.read())
        got = after[slot]["extra"]
        check("PAINTING A REAL TILE'S OVERHANG REACHES THE FILE",
              got is not None and bool((got[0:4, 0:6] == ink).all()),
              "-- %s slot %d: the stroke did not survive"
              % (os.path.basename(p), slot))

        restored = got.copy()
        restored[0:4, 0:6] = before
        original = whole[ey:ey + got.shape[0], ex:ex + got.shape[1]]
        check("...and nothing else in the block moved",
              np.array_equal(restored, original),
              "-- painting one corner disturbed the rest of the overhang")


    print("\nall corpus checks passed")


if __name__ == "__main__":
    main()
