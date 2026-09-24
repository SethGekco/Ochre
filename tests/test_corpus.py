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

    print("\nall corpus checks passed")


if __name__ == "__main__":
    main()
