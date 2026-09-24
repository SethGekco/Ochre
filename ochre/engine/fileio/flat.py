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

DEFAULTS = {
    "PNG": {"alpha": True, "extensions": ["png"]},
    "BMP": {"alpha": False, "extensions": ["bmp"]},
    "JPEG": {"alpha": False, "extensions": ["jpg", "jpeg"], "quality": 92},
    "GIF": {"alpha": True, "extensions": ["gif"], "frames": True},
    "WEBP": {"alpha": True, "extensions": ["webp"], "frames": True},
    "TIFF": {"alpha": True, "extensions": ["tif", "tiff"]},
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
    """Any Pillow-readable image becomes a single-layer document."""
    from ..document import Document

    img = Image.open(path)
    img.load()
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]

    doc = Document(w, h, settings=settings)
    layer = doc.add_layer(os.path.splitext(os.path.basename(path))[0] or "Layer")
    cell = doc.cell(layer)
    cell.write(doc.bounds, arr)
    doc.meta["source"] = os.path.basename(path)
    return doc


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

def export_flat(doc, path, fmt=None, table=None, frame=None, matte=None):
    """Composite the document and write it as a flat image."""
    from ..compositor import Compositor

    table = table or DEFAULT_TABLE
    ext = os.path.splitext(path)[1]
    name, spec = table.for_extension(ext)
    if fmt:
        name = fmt.upper()
        spec = table.formats.get(name, spec)
    if not name:
        raise ValueError("no known format for %r" % (ext or path))

    flat = Compositor(doc).composite(frame=frame)
    if flat is None:
        raise ValueError("nothing to export")

    if not spec.get("alpha", True):
        flat = flatten_onto(flat, matte or spec.get("matte", MATTE))
        img = Image.fromarray(flat[..., :3], "RGB")
    else:
        img = Image.fromarray(flat, "RGBA")

    kwargs = {}
    if "quality" in spec:
        kwargs["quality"] = spec["quality"]
    img.save(path, format=name, **kwargs)
    return path


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
