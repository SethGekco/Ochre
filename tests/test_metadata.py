#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photo metadata through the real paths: a phone photo stored sideways with
an EXIF Orientation opens upright (all eight orientations), its EXIF and XMP
come back out of Save and Export As, a re-export is never turned twice, and
the location can be stripped -- by Preferences for Save, per export in the
Export As dialog.

Run: python3 tests/test_metadata.py
"""

import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image

from ochre.engine import metadata
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TMP = tempfile.mkdtemp(prefix="ochre-fileio-meta-")

XMP = (b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
       b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
       b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
       b'<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
       b'xmlns:exif="http://ns.adobe.com/exif/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/" '
       b'tiff:Orientation="6" exif:GPSLatitude="40,26.767N" exif:GPSLongitude="79,58.933W">'
       b'<dc:creator><rdf:Seq><rdf:li>A. Photographer</rdf:li></rdf:Seq></dc:creator>'
       b'<exif:GPSAltitude>310/1</exif:GPSAltitude>'
       b'</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def scene():
    """The picture as it should LOOK: 60 wide, 40 tall, with a red block in
    the top-left corner and a blue one at the bottom right."""
    a = np.full((40, 60, 3), 230, np.uint8)
    a[0:12, 0:18] = (220, 20, 20)
    a[28:40, 44:60] = (20, 20, 220)
    return Image.fromarray(a, "RGB")


# For each Orientation, the transpose that turns the upright picture into
# what the camera STORED (the inverse of what a viewer applies).
STORE = {1: None, 2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
         4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
         6: Image.Transpose.ROTATE_90, 7: Image.Transpose.TRANSVERSE,
         8: Image.Transpose.ROTATE_270}


def phone_exif(orientation):
    ex = Image.Exif()
    ex[0x010F] = "PhoneCo"
    ex[0x0110] = "Pocket 9"
    ex[0x0112] = orientation
    ex[0x8769] = {0x9003: "2026:10:01 12:00:00", 0xA002: 40, 0xA003: 60}
    ex[0x8825] = {1: "N", 2: (40.0, 26.0, 46.0), 3: "W", 4: (79.0, 58.0, 56.0)}
    return ex.tobytes()


def photo(name, orientation, fmt):
    up = scene()
    stored = up if STORE[orientation] is None else up.transpose(STORE[orientation])
    path = os.path.join(TMP, name)
    kw = {"exif": phone_exif(orientation)}
    if fmt == "JPEG":
        kw.update(quality=95, xmp=XMP)
    stored.save(path, fmt, **kw)
    return path


def merged(ctl):
    return np.asarray(ctl.merged())[..., :3].astype(int)


def upright(arr, tol):
    """Red top-left, blue bottom-right, grey elsewhere: the scene, upright."""
    if arr.shape[:2] != (40, 60):
        return False
    red = np.abs(arr[5, 8] - (220, 20, 20)).max() <= tol
    blue = np.abs(arr[34, 52] - (20, 20, 220)).max() <= tol
    grey = np.abs(arr[34, 8] - 230).max() <= tol and np.abs(arr[5, 52] - 230).max() <= tol
    return red and blue and grey


def main():
    ctl = Controller(Settings(DATA), DATA, state_dir=tempfile.mkdtemp(prefix="ochre-fileio-meta-"))

    # ---- orientation on import ---------------------------------------------------
    for o in range(1, 9):
        ctl.open_path(photo("o%d.png" % o, o, "PNG"))
        check("EXIF Orientation %d opens upright (60 x 40, red top-left)" % o,
              upright(merged(ctl), 0), str(merged(ctl).shape))

    # ---- a phone JPEG: sideways pixels, Orientation 6, GPS, XMP --------------------
    src = photo("phone.jpg", 6, "JPEG")
    raw = Image.open(src)
    check("(the test photo really is stored sideways: %s)" % (raw.size,), raw.size == (40, 60))
    ctl.open_path(src)
    check("a sideways phone JPEG opens upright", upright(merged(ctl), 24))
    kept = Image.Exif()
    kept.load(metadata._decode(ctl.doc.meta.get("exif_data")))
    check("its EXIF is kept, with Orientation now 1",
          kept.get(0x010F) == "PhoneCo" and kept.get(0x0112) == 1, str(dict(kept)))
    check("its XMP is kept, with tiff:Orientation now 1",
          b'tiff:Orientation="1"' in metadata._decode(ctl.doc.meta.get("xmp_data")))

    out = os.path.join(TMP, "edited.jpg")
    ctl.save_path(out)
    back = Image.open(out)
    ex = back.getexif()
    check("Save writes the EXIF back: camera, date, GPS",
          ex.get(0x010F) == "PhoneCo" and ex.get(0x0110) == "Pocket 9"
          and ex.get_ifd(0x8769).get(0x9003) == "2026:10:01 12:00:00"
          and bool(ex.get_ifd(0x8825)), str(dict(ex)))
    check("...with Orientation 1 and the pixels upright: no double rotation",
          ex.get(0x0112) == 1 and back.size == (60, 40))
    check("...and the recorded pixel size is the new one (60 x 40, not 40 x 60)",
          ex.get_ifd(0x8769).get(0xA002) == 60 and ex.get_ifd(0x8769).get(0xA003) == 40)
    xmp = back.info.get("xmp") or b""
    check("Save writes the XMP back (creator kept, orientation 1)",
          b"A. Photographer" in xmp and b'tiff:Orientation="1"' in xmp)
    ctl.open_path(out)
    check("re-opening the export: still upright, not turned again", upright(merged(ctl), 24))

    # Through .ochre and out again, into each format that can carry it.
    native = os.path.join(TMP, "kept.ochre")
    ctl.save_path(native)
    ctl.open_path(native)
    for ext in ("png", "webp", "tif"):
        path = os.path.join(TMP, "again." + ext)
        ctl.save_path(path)
        im = Image.open(path)
        ex = im.getexif()
        check(".ochre keeps the metadata, and .%s carries EXIF and XMP out" % ext,
              ex.get(0x010F) == "PhoneCo" and ex.get(0x0112) in (1, None)
              and b"A. Photographer" in (im.info.get("xmp") or b"") and im.size == (60, 40),
              "%s %r" % (dict(ex), (im.info.get("xmp") or b"")[:40]))

    # ---- stripping the location ---------------------------------------------------
    private = Controller(Settings(DATA, env={"OCHRE_STRIPGPS": "on"}), DATA,
                         state_dir=tempfile.mkdtemp(prefix="ochre-fileio-meta-"))
    private.open_path(src)
    out = os.path.join(TMP, "private.jpg")
    private.save_path(out)
    im = Image.open(out)
    ex = im.getexif()
    xmp = im.info.get("xmp") or b""
    check("Preferences > Export > StripGPS: Save drops the GPS block, keeps the camera",
          not ex.get_ifd(0x8825) and 0x8825 not in ex and ex.get(0x010F) == "PhoneCo", str(dict(ex)))
    check("...and every GPS property in the XMP, keeping the rest",
          b"GPS" not in xmp and b"A. Photographer" in xmp, xmp.decode("latin-1"))

    # The Export As dialog, as the user drives it.
    from PySide6.QtWidgets import QApplication
    from ochre.ui.exportdialog import ExportDialog
    app = QApplication.instance() or QApplication([])
    ctl.open_path(src)
    dlg = ExportDialog(ctl)
    dlg.format.setCurrentIndex(dlg.format.findData("JPEG"))
    check("Export As offers the metadata choice for a photo that has some", dlg.meta.isEnabled())
    for index, label, want_exif, want_gps in ((0, "Keep all", True, True),
                                              (1, "Remove location (GPS)", True, False),
                                              (2, "Remove all", False, False)):
        dlg.meta.setCurrentIndex(index)
        dlg.refresh()
        im = Image.open(io.BytesIO(dlg.data))
        ex = im.getexif()
        xmp = im.info.get("xmp") or b""
        check("Export As, %s: EXIF %s, GPS %s" % (label, want_exif, want_gps),
              (ex.get(0x010F) == "PhoneCo") == want_exif
              and bool(ex.get_ifd(0x8825)) == want_gps
              and (b"GPSLatitude" in xmp) == want_gps
              and (b"A. Photographer" in xmp) == want_exif
              and im.size == (60, 40), "%s %r" % (dict(ex), xmp[:60]))
    dlg.close()
    del app

    print("\nall metadata checks passed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
