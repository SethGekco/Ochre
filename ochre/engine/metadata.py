# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photo metadata: EXIF orientation on import, EXIF and XMP carried through
to export, and location stripped on request.

ORIENTATION. A phone stores its pixels the way the sensor read them and an
EXIF Orientation tag saying how to turn them. Ochre applies the tag on
import -- the pixels are turned, so the picture opens upright -- and then
records Orientation = 1 in the metadata it keeps. That normalization is the
whole guard against double rotation: whatever is exported afterwards says
"already upright", so neither Ochre nor anything else turns it again. XMP
can carry its own tiff:Orientation; it is normalized the same way.

PASS-THROUGH. The EXIF block (minus anything describing the old file's
layout, and minus its embedded thumbnail, which would still show the
picture as it was before any edit or crop) and the XMP packet live in
doc.meta as base64 text, like the ICC profile, so they survive a trip
through .ochre. Export writes them back where Pillow can: EXIF to JPEG,
PNG, WebP, TIFF and AVIF; XMP to all five. Pixel dimensions recorded in
EXIF are updated to the exported size.

PRIVACY. `strip_gps` removes the GPS block from EXIF and every GPS property
from XMP; `strip_metadata` writes neither. Both are export options and
[Export] settings, so a Save can honor them as well as Export As.
"""

import base64
import re

ORIENTATION = 0x0112
EXIF_IFD = 0x8769
GPS_IFD = 0x8825
INTEROP_IFD = 0xA005
MAKER_NOTE = 0x927C
PIXEL_X = 0xA002
PIXEL_Y = 0xA003

# IFD0 tags that describe the SOURCE file's own layout (a TIFF's strips, a
# thumbnail's offsets) or that travel separately (ICC, XMP, IPTC). Copied
# into another file they would be wrong, so they never enter doc.meta.
_LAYOUT_TAGS = frozenset((
    256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 317, 320, 322, 323,
    324, 325, 338, 339, 513, 514, 530, 700, 33723, 34377, 34675, 37724))

# What each Pillow writer can carry.
EXIF_FORMATS = ("JPEG", "PNG", "WEBP", "TIFF", "AVIF")
XMP_FORMATS = ("JPEG", "PNG", "WEBP", "TIFF", "AVIF")

# A JPEG APP1 segment holds at most 65533 bytes, and Pillow refuses more.
JPEG_EXIF_MAX = 65533


def _transpose_for(orientation):
    """The Pillow transpose that turns stored pixels upright, or None."""
    from PIL import Image
    T = Image.Transpose
    return {2: T.FLIP_LEFT_RIGHT, 3: T.ROTATE_180, 4: T.FLIP_TOP_BOTTOM,
            5: T.TRANSPOSE, 6: T.ROTATE_270, 7: T.TRANSVERSE,
            8: T.ROTATE_90}.get(orientation)


def orientation(img):
    """The EXIF Orientation of an opened image: 1 (upright) to 8."""
    try:
        value = int(img.getexif().get(ORIENTATION, 1) or 1)
    except (TypeError, ValueError, OSError, SyntaxError):
        return 1
    return value if 1 <= value <= 8 else 1


def upright(img, value):
    """`img` turned upright for an Orientation `value` (a new image)."""
    method = _transpose_for(value)
    return img if method is None else img.transpose(method)


# ---- capture on import ----------------------------------------------------

def capture(img):
    """{doc.meta key: value} for an opened image's EXIF and XMP, normalized
    to Orientation 1 (the caller turns the pixels with upright())."""
    out = {}
    exif = _clean_exif(img)
    if exif is not None:
        out["exif_data"] = base64.b64encode(exif).decode("ascii")
    xmp = _xmp_of(img)
    if xmp:
        out["xmp_data"] = base64.b64encode(set_xmp_orientation(xmp, 1)).decode("ascii")
    return out


def _clean_exif(img):
    """The image's EXIF as bytes, layout tags and thumbnail dropped and
    Orientation set to 1; None when there is nothing worth keeping."""
    from PIL import Image
    try:
        src = img.getexif()
        if not len(src):
            return None
        clean = Image.Exif()
        for tag, value in src.items():
            if tag in _LAYOUT_TAGS or tag in (EXIF_IFD, GPS_IFD):
                continue
            clean[tag] = value
        sub = dict(src.get_ifd(EXIF_IFD)) if EXIF_IFD in src else {}
        # The interop pointer is an offset into the OLD file; written into a
        # new one it would point at garbage. It only names the color space,
        # which the ICC profile already says.
        sub.pop(INTEROP_IFD, None)
        if sub:
            clean[EXIF_IFD] = sub
        gps = dict(src.get_ifd(GPS_IFD)) if GPS_IFD in src else {}
        if gps:
            clean[GPS_IFD] = gps
        if not len(clean):
            return None
        if ORIENTATION in clean:
            clean[ORIENTATION] = 1
        return clean.tobytes()
    except Exception:                                       # noqa: BLE001
        # Metadata is a courtesy. A damaged EXIF block must never stop a
        # picture from opening.
        return None


def _xmp_of(img):
    """The raw XMP packet, wherever this Pillow and this format put it."""
    for key in ("xmp", "XML:com.adobe.xmp"):
        value = img.info.get(key)
        if value:
            return value.encode("utf-8") if isinstance(value, str) else bytes(value)
    tags = getattr(img, "tag_v2", None)
    if tags is not None and tags.get(700):
        value = tags.get(700)
        return value.encode("utf-8") if isinstance(value, str) else bytes(value)
    return None


# ---- XMP surgery ----------------------------------------------------------
# XMP is XML, but a regular expression is the right tool here: the edits are
# two narrow ones, and a parse-and-reserialise round trip would rewrite the
# whole packet (namespaces, padding, whitespace) to change one value.

_ORIENT_ATTR = re.compile(rb'(\btiff:Orientation\s*=\s*["\'])\d(["\'])')
_ORIENT_ELEM = re.compile(rb'(<tiff:Orientation>\s*)\d(\s*</tiff:Orientation>)')
_GPS_ATTR = re.compile(rb'\s+[A-Za-z][\w.-]*:GPS\w*\s*=\s*("[^"]*"|\'[^\']*\')')
_GPS_ELEM = re.compile(rb'<([A-Za-z][\w.-]*:GPS\w*)\b[^>]*?(/>|>.*?</\1\s*>)', re.S)


def set_xmp_orientation(xmp, value):
    xmp = _ORIENT_ATTR.sub(rb"\g<1>%d\g<2>" % value, xmp)
    return _ORIENT_ELEM.sub(rb"\g<1>%d\g<2>" % value, xmp)


def strip_xmp_gps(xmp):
    """XMP with every GPS property (exif:GPSLatitude and the rest) removed."""
    return _GPS_ELEM.sub(b"", _GPS_ATTR.sub(b"", xmp))


# ---- write on export ------------------------------------------------------

def export_kwargs(meta, fmt, size, strip_gps=False, strip_all=False):
    """Pillow save() keyword arguments carrying `meta`'s EXIF and XMP into a
    `fmt` file of `size` (w, h). Empty when the format carries neither, or
    when asked to strip everything."""
    if strip_all or not meta:
        return {}
    out = {}
    xmp = _decode(meta.get("xmp_data")) if fmt in XMP_FORMATS else None
    if xmp:
        xmp = set_xmp_orientation(xmp, 1)
        if strip_gps:
            xmp = strip_xmp_gps(xmp)
    exif = _decode(meta.get("exif_data")) if fmt in EXIF_FORMATS else None
    if exif:
        # TIFF keeps XMP as tag 700 of the very directory the EXIF goes
        # into, and Pillow takes either exif= or tiffinfo=, not both.
        exif = _exif_for_export(exif, size, strip_gps, fmt,
                                xmp if fmt == "TIFF" else None)
        if exif:
            out["exif"] = exif
            if fmt == "TIFF":
                xmp = None
    if xmp:
        if fmt == "PNG":
            from PIL import PngImagePlugin
            info = PngImagePlugin.PngInfo()
            info.add_itxt("XML:com.adobe.xmp", xmp.decode("utf-8", "replace"), zip=False)
            out["pnginfo"] = info
        elif fmt == "TIFF":
            out["tiffinfo"] = {700: xmp}
        else:
            out["xmp"] = xmp
    return out


def _decode(text):
    if not text:
        return None
    try:
        return base64.b64decode(text)
    except (ValueError, TypeError):
        return None


def _exif_for_export(raw, size, strip_gps, fmt, xmp=None):
    from PIL import Image
    try:
        exif = Image.Exif()
        exif.load(raw)
        # Always upright: the pixels were turned on import.
        if ORIENTATION in exif:
            exif[ORIENTATION] = 1
        sub = exif.get_ifd(EXIF_IFD) if EXIF_IFD in exif else None
        if sub:
            # What the camera recorded is history; the size is now this one.
            if PIXEL_X in sub:
                sub[PIXEL_X] = int(size[0])
            if PIXEL_Y in sub:
                sub[PIXEL_Y] = int(size[1])
        if strip_gps and GPS_IFD in exif:
            del exif[GPS_IFD]
        if xmp:
            exif[700] = xmp
        data = exif.tobytes()
        if fmt == "JPEG" and len(data) > JPEG_EXIF_MAX and sub and MAKER_NOTE in sub:
            # A maker note is the usual reason a block outgrows a JPEG
            # segment, and the least useful part of it outside the camera
            # maker's own software.
            del sub[MAKER_NOTE]
            data = exif.tobytes()
        if fmt == "JPEG" and len(data) > JPEG_EXIF_MAX:
            return None
        return data
    except Exception:                                       # noqa: BLE001
        return None
