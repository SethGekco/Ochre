# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Flat import and export, through Pillow.

Format quirks live in data/fileformats.ini as [Format:NAME] sections rather
than as `if ext == ".jpg"` at a call site -- so adding a format's rules, or
changing the matte colour JPEG flattens onto, is an INI edit.

Export composites the whole document once. That is a progress-bar operation,
not an interactive one: a full-canvas composite of a large multi-layer
document runs into the tens of seconds, which is exactly why nothing in the
interactive path ever does it.
"""

import os

import numpy as np
from PIL import Image

from ..geometry import Rect

DEFAULTS = {
    "PNG": {"alpha": True, "extensions": ["png"]},
    "BMP": {"alpha": False, "extensions": ["bmp"]},
    "JPEG": {"alpha": False, "extensions": ["jpg", "jpeg"], "quality": 92},
    "GIF": {"alpha": True, "extensions": ["gif"], "frames": True},
    "WEBP": {"alpha": True, "extensions": ["webp"], "frames": True},
    "TIFF": {"alpha": True, "extensions": ["tif", "tiff"]},
    "AVIF": {"alpha": True, "extensions": ["avif"], "quality": 80},
    "QOI": {"alpha": True, "extensions": ["qoi"]},
    "TGA": {"alpha": True, "extensions": ["tga"]},
    "ICO": {"alpha": True, "extensions": ["ico"]},
    "JPEG2000": {"alpha": True, "extensions": ["jp2", "j2k"], "quality": 90},
    "PPM": {"alpha": False, "extensions": ["ppm", "pnm"]},
    "PCX": {"alpha": False, "extensions": ["pcx"]},
    "SGI": {"alpha": True, "extensions": ["sgi", "rgb"]},
}

MATTE = (255, 255, 255)


class FormatTable:
    """What each flat format can carry, and how to coerce onto it."""

    def __init__(self, db=None):
        self.formats = {name: dict(spec) for name, spec in DEFAULTS.items()}
        if db is not None:
            self.load(db)

    def load(self, db):
        for name in db.sections("Format"):
            section = "Format:" + name
            spec = self.formats.setdefault(name.upper(), {})
            if db.has(section, "Alpha"):
                spec["alpha"] = db.getbool(section, "Alpha", True)
            exts = db.getlist(section, "Extensions")
            if exts:
                spec["extensions"] = [e.lower().lstrip(".") for e in exts]
            if db.has(section, "Quality"):
                spec["quality"] = db.getint(section, "Quality", 92)
            if db.has(section, "Frames"):
                spec["frames"] = db.getbool(section, "Frames", False)
            if db.has(section, "Matte"):
                spec["matte"] = db.getcolor(section, "Matte", MATTE + (255,))[:3]
        return self

    def for_extension(self, ext):
        ext = ext.lower().lstrip(".")
        for name, spec in self.formats.items():
            if ext in spec.get("extensions", []):
                return name, spec
        return None, {}

    def extensions(self):
        out = []
        for spec in self.formats.values():
            out.extend(spec.get("extensions", []))
        return sorted(set(out))

    def supports_alpha(self, name):
        return self.formats.get(name.upper(), {}).get("alpha", True)


DEFAULT_TABLE = FormatTable()


def supported_extensions(table=None):
    return (table or DEFAULT_TABLE).extensions()


# ---- import -------------------------------------------------------------

def import_flat(path, settings=None):
    """Any Pillow-readable image becomes a single-layer document.

    A MULTI-FRAME source -- an animated GIF, a multi-page TIFF, an APNG --
    becomes one document frame per source frame on the one layer, which is
    exactly the shape the frame axis exists for. Reading only the first one
    is silent data loss at the moment of opening, and the frames would be
    gone for good the first time the file was saved back.

    Pillow composites GIF frames for us as it seeks, so disposal methods and
    partial inter-frame rectangles are already resolved; what arrives here
    is always a full canvas.
    """
    from ..document import Document
    from ..frame import Frame

    img = Image.open(path)
    img.load()
    count = max(1, int(getattr(img, "n_frames", 1)))
    icc = img.info.get("icc_profile")
    if img.mode == "CMYK" and count == 1:
        # Through its profile (or the system's CMYK default), not Pillow's
        # naive formula, which turns every print file's colours to mud.
        img = _cmyk_to_srgb(img, icc)
        icc = None

    doc = None
    layer = None
    for index in range(count):
        if count > 1:
            img.seek(index)
        frame_img = img if img.mode == "RGBA" else img.convert("RGBA")
        arr = np.asarray(frame_img, dtype=np.uint8)
        h, w = arr.shape[:2]

        if doc is None:
            doc = Document(w, h, settings=settings)
            layer = doc.add_layer(
                os.path.splitext(os.path.basename(path))[0] or "Layer")
            if count > 1:
                doc.frames = []
        if count > 1:
            frame = Frame(name="Frame %d" % (index + 1))
            duration = frame_img.info.get("duration")
            if duration:
                frame.meta["duration_ms"] = str(int(duration))
            doc.frames.append(frame)
            cell = doc.cell(layer, frame)
        else:
            cell = doc.cell(layer)
        # Frames after the first can be smaller than the canvas on some
        # encoders; write what fits rather than refusing the file.
        region = doc.bounds if (w, h) == (doc.width, doc.height) \
            else Rect(0, 0, min(w, doc.width), min(h, doc.height))
        cell.write(region, arr[:region.h, :region.w])

    doc.current = 0
    doc.meta["source"] = os.path.basename(path)
    if icc:
        # The numbers stay as they are; the profile says what they mean
        # (Photoshop's "use the embedded profile").
        from .. import colormanage
        try:
            prof = colormanage.open_profile(icc)
            if colormanage.colour_space(prof) == "RGB" and not colormanage.is_srgb(prof):
                colormanage.set_doc_profile(doc, prof)
        except Exception:                                   # noqa: BLE001
            pass
    if count > 1:
        doc.meta["frames"] = str(count)
    return doc


def _cmyk_to_srgb(img, icc=None):
    from .. import colormanage
    from PIL import ImageCms
    try:
        src = colormanage.open_profile(icc) if icc else None
        if src is None:
            cmyk = [p for p in colormanage.system_profiles() if p[2] == "CMYK"]
            src = colormanage.open_profile(cmyk[0][1]) if cmyk else None
        if src is None:
            return img.convert("RGB")
        t = ImageCms.buildTransform(src, colormanage.open_profile(None), "CMYK", "RGB")
        return ImageCms.applyTransform(img, t)
    except Exception:                                       # noqa: BLE001
        return img.convert("RGB")


def import_indexed(path, settings=None):
    """A paletted image becomes an index-locked layer, losslessly.

    Going through RGBA would destroy the index identity -- two palette
    entries with the same colour would collapse into one -- so a paletted
    source is read as indices and the palette is bound to the document.
    """
    from ..document import Document
    from ..palette import LENGTH, Palette

    img = Image.open(path)
    img.load()
    if img.mode != "P":
        return import_flat(path, settings)

    raw = img.getpalette() or []
    entries = np.zeros((LENGTH, 4), dtype=np.uint8)
    entries[:, 3] = 255
    triples = min(LENGTH, len(raw) // 3)
    entries[:triples, :3] = np.array(raw[:triples * 3], dtype=np.uint8).reshape(-1, 3)

    palette = Palette(entries, os.path.basename(path))
    transparency = img.info.get("transparency")
    if isinstance(transparency, int):
        palette.set_transparent(transparency)

    idx = np.asarray(img, dtype=np.uint8)
    h, w = idx.shape[:2]
    doc = Document(w, h, palette=palette, settings=settings)
    layer = doc.add_layer(os.path.splitext(os.path.basename(path))[0] or "Layer",
                          planes=("index", "rgba"), authoritative=("index",))
    cell = doc.cell(layer)
    cell.plane("index")[...] = idx
    cell.content_bbox = doc.bounds
    cell.refresh_derived()
    doc.meta["source"] = os.path.basename(path)
    return doc


# ---- export -------------------------------------------------------------

def _prepare(doc, name, spec, frame=None, matte=None, scale=1.0, flat=None):
    """The merged image, scaled and matted as the format needs, as a PIL Image."""
    from ..compositor import Compositor
    if flat is None:
        flat = Compositor(doc).composite(frame=frame)
    if flat is None:
        raise ValueError("nothing to export")
    img_rgba = Image.fromarray(flat, "RGBA")
    if scale and abs(float(scale) - 1.0) > 1e-6:
        w = max(1, int(round(doc.width * float(scale))))
        h = max(1, int(round(doc.height * float(scale))))
        # Resampled premultiplied, so transparent edges do not darken.
        from ..transform import resample_rgba
        img_rgba = Image.fromarray(resample_rgba(flat, w, h, "lanczos" if scale < 1 else "bicubic"),
                                   "RGBA")
    if not spec.get("alpha", True):
        arr = flatten_onto(np.asarray(img_rgba), matte or spec.get("matte", MATTE))
        return Image.fromarray(arr[..., :3], "RGB")
    return img_rgba


def _save_kwargs(name, spec, options):
    kw = {}
    quality = options.get("quality", spec.get("quality"))
    if name in ("JPEG", "WEBP", "AVIF") and quality is not None:
        kw["quality"] = int(quality)
    if name == "WEBP" and options.get("lossless"):
        kw["lossless"] = True
    if name == "PNG":
        kw["optimize"] = bool(options.get("optimize", False))
        kw["compress_level"] = int(options.get("compress_level", 6))
    if name == "JPEG":
        kw["optimize"] = True
        kw["progressive"] = bool(options.get("progressive", False))
    if name == "ICO":
        kw["sizes"] = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
    return kw


def encode_flat(doc, fmt, options=None, table=None, flat=None):
    """The exported file's bytes, without writing it -- what Export As uses
    to show the real size before you commit to a quality."""
    import io
    table = table or DEFAULT_TABLE
    name = fmt.upper()
    spec = table.formats.get(name, {})
    options = options or {}
    img = _prepare(doc, name, spec, options.get("frame"), options.get("matte"),
                   options.get("scale", 1.0), flat)
    img, extra = _colour_out(doc, name, img, options)
    buf = io.BytesIO()
    img.save(buf, format=name, **dict(_save_kwargs(name, spec, options), **extra))
    return buf.getvalue()


def export_flat(doc, path, fmt=None, table=None, frame=None, matte=None, options=None):
    """Composite the document and write it as a flat image."""
    table = table or DEFAULT_TABLE
    ext = os.path.splitext(path)[1]
    name, spec = table.for_extension(ext)
    if fmt:
        name = fmt.upper()
        spec = table.formats.get(name, spec)
    if not name:
        raise ValueError("no known format for %r" % (ext or path))
    options = dict(options or {})
    img = _prepare(doc, name, spec, frame, matte or options.get("matte"),
                   options.get("scale", 1.0))
    img, extra = _colour_out(doc, name, img, options)
    img.save(path, format=name, **dict(_save_kwargs(name, spec, options), **extra))
    return path


ICC_FORMATS = ("PNG", "JPEG", "TIFF", "WEBP")
CMYK_FORMATS = ("JPEG", "TIFF")


def _colour_out(doc, name, img, options):
    """The image as written: embedded profile (the document's, when it has
    one), or converted to CMYK through a chosen profile. Returns (image,
    extra save kwargs)."""
    from .. import colormanage
    cmyk = options.get("cmyk_profile")
    if cmyk and name in CMYK_FORMATS:
        prof = colormanage.open_profile(cmyk)
        rgba = np.asarray(img.convert("RGBA"))
        out = colormanage.to_cmyk_image(rgba, prof, colormanage.doc_profile(doc),
                                        options.get("intent", "perceptual"))
        return out, {"icc_profile": colormanage.profile_bytes(prof)}
    prof = colormanage.doc_profile(doc)
    if name in ICC_FORMATS and (prof is not None or options.get("embed_srgb")):
        prof = prof or colormanage.open_profile(None)
        return img, {"icc_profile": colormanage.profile_bytes(prof)}
    return img, {}


def flatten_onto(rgba, matte=MATTE):
    """Composite straight-alpha RGBA onto an opaque background.

    Needed by every format that cannot carry alpha. Done with the same
    alpha-weighted arithmetic as everything else, so a half-transparent pixel
    lands where it should rather than being naively averaged.
    """
    from ..arith import lerp255

    rgba = np.asarray(rgba, dtype=np.uint8)
    out = np.empty(rgba.shape, dtype=np.uint8)
    alpha = rgba[..., 3]
    for c in range(3):
        back = np.full(rgba.shape[:2], matte[c], dtype=np.uint8)
        out[..., c] = lerp255(back, rgba[..., c], alpha)
    out[..., 3] = 255
    return out


def export_frames(doc, path, table=None, loop=0):
    """Write every frame as an animation, for formats that carry one."""
    from ..compositor import Compositor

    table = table or DEFAULT_TABLE
    name, spec = table.for_extension(os.path.splitext(path)[1])
    if not spec.get("frames"):
        raise ValueError("%s cannot hold multiple frames" % (name or path))

    comp = Compositor(doc)
    images, durations = [], []
    for frame in doc.frames:
        doc.ensure_warm(frame)
        flat = comp.composite(frame=frame)
        images.append(Image.fromarray(flat, "RGBA"))
        durations.append(max(1, frame.duration_ms))
    if not images:
        raise ValueError("nothing to export")

    images[0].save(path, format=name, save_all=True, append_images=images[1:],
                   duration=durations, loop=loop, disposal=2)
    return path
