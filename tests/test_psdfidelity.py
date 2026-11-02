#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""PSD fidelity: deep files read at 8 bits, rounded and said so.

Every file here is built byte by byte in the test (no PSD library, nothing
from outside the repository), small, and opened and saved through the
controller's open_path / save_path -- the calls the window, the headless
CLI and the control server make.

Run: python3 tests/test_psdfidelity.py
"""

import os
import shutil
import struct
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.fileio import psd
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TMP = tempfile.mkdtemp(prefix="ochre-psd-fidelity-")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


# ---- a tiny PSD writer, independent of the one under test -----------------------------

def samples(values, depth):
    """Raw big-endian samples at `depth` bits."""
    fmt = {8: "B", 16: "H", 32: "f"}[depth]
    return struct.pack(">%d%s" % (len(values), fmt), *values)


def raw_channel(values, depth):
    return struct.pack(">H", 0) + samples(values, depth)


def zip_predicted(values, w, depth):
    """Compression 3 (ZIP with prediction) of one row-major plane."""
    h = len(values) // w
    if depth == 16:
        a = np.array(values, np.uint16).reshape(h, w)
        d = a.copy()
        d[:, 1:] = a[:, 1:] - a[:, :-1]
        payload = d.astype(">u2").tobytes()
    else:
        f = np.array(values, ">f4").reshape(h, w)
        # Each row: the floats' byte planes one after another, then deltas.
        planes = f.view(np.uint8).reshape(h, w, 4).transpose(0, 2, 1).reshape(h, 4 * w)
        d = planes.copy()
        d[:, 1:] = planes[:, 1:] - planes[:, :-1]
        payload = d.tobytes()
    return struct.pack(">H", 3) + zlib.compress(payload)


def rle_channel(values, w, depth):
    """Compression 1 (PackBits) of one plane, rows of w samples."""
    raw = samples(values, depth)
    rowbytes = w * depth // 8
    rows = [psd.packbits_encode(raw[i:i + rowbytes]) for i in range(0, len(raw), rowbytes)]
    return (struct.pack(">H", 1) + b"".join(struct.pack(">H", len(x)) for x in rows)
            + b"".join(rows))


def block(key, data, pad=2):
    """A tagged block: signature, key, length, data, padding."""
    return b"8BIM" + key + struct.pack(">I", len(data)) + data + bytes((-len(data)) % pad)


def layer(name, rect, chans, blend=b"norm", opacity=255, clip=0, flags=0, blocks=b"",
          ranges=b""):
    """(record, channel data) for one layer. chans: [(id, channel bytes)]."""
    t, l, b, r = rect
    rec = struct.pack(">iiiiH", t, l, b, r, len(chans))
    for cid, data in chans:
        rec += struct.pack(">hI", cid, len(data))
    raw = name.encode("latin-1")
    pascal = bytes((len(raw),)) + raw
    pascal += bytes((-len(pascal)) % 4)
    extra = struct.pack(">I", 0) + struct.pack(">I", len(ranges)) + ranges + pascal + blocks
    rec += b"8BIM" + blend + bytes((opacity, clip, flags, 0)) + struct.pack(">I", len(extra))
    return rec + extra, b"".join(data for _cid, data in chans)


def layer_info(layers):
    records = [layer(*a, **k) for a, k in layers]
    return (struct.pack(">h", -len(records)) + b"".join(r for r, _ in records)
            + b"".join(d for _, d in records))


def build(w, h, depth, layers, composite, resources=b"", global_blocks=b"", tagged=False,
          global_mask=b""):
    """A whole RGB PSD. `composite` is four planes of w*h samples (RGBA).
    tagged: the layers go in a document-level Lr16/Lr32 block, as
    Photoshop stores them at 16 and 32 bits."""
    head = b"8BPS" + struct.pack(">H6xHIIHH", 1, 4, h, w, depth, 3)
    body = struct.pack(">I", 0) + struct.pack(">I", len(resources)) + resources
    info = layer_info(layers) if layers else b""
    info += bytes(len(info) % 2)
    if tagged:
        key = {16: b"Lr16", 32: b"Lr32"}[depth]
        lm = struct.pack(">I", 0) + struct.pack(">I", len(global_mask)) + global_mask
        lm += block(key, info, 4) + global_blocks
    else:
        lm = struct.pack(">I", len(info)) + info
        lm += struct.pack(">I", len(global_mask)) + global_mask + global_blocks
    body += struct.pack(">I", len(lm)) + lm
    body += struct.pack(">H", 0) + b"".join(samples(p, depth) for p in composite)
    return head + body


def put(name, data):
    path = os.path.join(TMP, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def controller():
    return Controller(Settings(DATA), DATA, addon_dirs=[],
                      state_dir=tempfile.mkdtemp(prefix="ochre-psd-fidelity-state-"))


def pixels(ctl, name):
    node = next(l for l in ctl.doc.layers() if l.name == name)
    return ctl.doc.cell(node).plane("rgba")


# ---- deep files -----------------------------------------------------------------------

def test_deep():
    # Eight samples whose rounding and truncation disagree, in both
    # directions: v / 257 = 0.498, 0.502, 0.992, 127.498, 127.502, 254.502,
    # 254.498, 255.
    v16 = [128, 129, 255, 32767, 32768, 65407, 65406, 65535]
    want16 = [0, 1, 1, 127, 128, 255, 254, 255]
    check("the 16-bit samples are ones truncation gets wrong",
          [v >> 8 for v in v16] != want16)
    w, h = 8, 1
    opaque16 = [65535] * 8
    layers = [
        (("raw", (0, 0, 1, 8), [(-1, raw_channel(opaque16, 16)), (0, raw_channel(v16, 16)),
                                (1, raw_channel(v16[::-1], 16)), (2, raw_channel(v16, 16))]), {}),
        (("zip", (0, 0, 1, 8), [(-1, raw_channel(opaque16, 16)),
                                (0, zip_predicted(v16, w, 16)),
                                (1, zip_predicted(v16, w, 16)),
                                (2, zip_predicted(v16, w, 16))]), {}),
        (("rle", (0, 0, 1, 8), [(-1, rle_channel(opaque16, w, 16)),
                                (0, rle_channel(v16, w, 16)),
                                (1, rle_channel(v16, w, 16)),
                                (2, rle_channel(v16, w, 16))]), {}),
    ]
    path = put("deep16.psd", build(w, h, 16, layers, [v16, v16, v16, opaque16], tagged=True))
    ctl = controller()
    ctl.open_path(path)
    check("a 16-bit PSD's layers (stored in its Lr16 block) open as layers",
          [l.name for l in ctl.doc.layers()] == ["raw", "zip", "rle"],
          str([l.name for l in ctl.doc.layers()]))
    got = pixels(ctl, "raw")
    check("16-bit samples are ROUNDED to 8 bits (round(v / 257)), not truncated",
          got[0, :, 0].tolist() == want16 and got[0, :, 1].tolist() == want16[::-1],
          str(got[0, :, 0].tolist()))
    check("...in ZIP-with-prediction channels too",
          pixels(ctl, "zip")[0, :, 0].tolist() == want16, str(pixels(ctl, "zip")[0, :, 0]))
    check("...and in PackBits channels",
          pixels(ctl, "rle")[0, :, 2].tolist() == want16, str(pixels(ctl, "rle")[0, :, 2]))
    check("...alpha included", int(got[0, :, 3].min()) == 255)
    notes = " ".join(ctl.open_notes)
    check("opening says the depth was reduced: 16-bit, read at 8 bits",
          "16-bit" in notes and "8 bits" in notes, repr(ctl.open_notes))
    _w, _h, _layers, comp = psd.read(path)
    check("the 16-bit composite is rounded the same way", comp[0, :, 0].tolist() == want16,
          str(comp[0, :, 0].tolist()))
    out = os.path.join(TMP, "deep16-saved.psd")
    ctl.save_path(out)
    saved = psd.parse(out)
    check("saving writes an 8-bit PSD with the layers",
          saved.depth == 8 and [L.name for L in saved.layers] == ["raw", "zip", "rle"])
    ctl.open_path(out)
    check("...and reopening an 8-bit PSD says nothing about depth", ctl.open_notes == [],
          repr(ctl.open_notes))

    # 32-bit: floats clipped to 0..1 and rounded; NaN reads as 0.
    v32 = [-0.5, float("nan"), 0.001, 0.25, 0.5, 0.75, 1.0, 1.5]
    want32 = [0, 0, 0, 64, 128, 191, 255, 255]
    one = [1.0] * 8
    layers = [
        (("float", (0, 0, 1, 8), [(-1, raw_channel(one, 32)), (0, raw_channel(v32, 32)),
                                  (1, zip_predicted(v32, w, 32)),
                                  (2, rle_channel(v32, w, 32))]), {}),
    ]
    path = put("deep32.psd", build(w, h, 32, layers, [v32, v32, v32, one], tagged=True))
    ctl.open_path(path)
    got = pixels(ctl, "float")
    check("32-bit floats read clipped to 0..1 and rounded (raw, ZIP-predicted, PackBits)",
          got[0, :, 0].tolist() == want32 and got[0, :, 1].tolist() == want32
          and got[0, :, 2].tolist() == want32, str(got[0, :, :3].tolist()))
    notes = " ".join(ctl.open_notes)
    check("...and opening says so", "32-bit" in notes and "8 bits" in notes, repr(ctl.open_notes))


def main():
    try:
        test_deep()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\nall PSD fidelity checks passed")


if __name__ == "__main__":
    main()
