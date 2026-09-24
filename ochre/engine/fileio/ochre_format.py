# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The .ochre document format: a zip of PNGs with an INI manifest.

    document.ochre
    |-- manifest.ini
    |-- thumbnail.png
    |-- selection.png          (only when a selection is active)
    |-- palette.png            (256x1 RGBA, only when a palette is bound)
    `-- frames/
        |-- 0000/layers/L0001.rgba.png
        |               L0001.index.png
        |               L0002.height.png
        `-- 0001/layers/...

Why this and not a custom binary format: `unzip -l` inspects it, `cat
manifest.ini` explains it, any tool on earth can read the layer PNGs,
corruption is localised to one entry, and there is no serialisation code to
write or version.

The forward-compatibility story falls out of the house INI rule rather than
needing any machinery. Unknown keys are ignored, so a file written by a newer
Ochre opens correctly in an older one, minus the features it does not know
about. There is no version negotiation because there is nothing to negotiate.

Single-frame documents still write frames/0000/, so there is exactly one
layout and a reader needs no special case.

Only AUTHORITATIVE planes are loaded. An index-locked layer also writes its
derived RGBA purely so external tools can read the file; Ochre ignores that
entry and regenerates from the index, which is what keeps a round-trip exact
rather than merely close.
"""

import io
import os
import zipfile

import numpy as np
from PIL import Image

from ..geometry import Rect
from ..ini import IniDB
from ..layer import Layer, LayerGroup
from ..palette import LENGTH, Palette
from ..surface import plane_channels

FORMAT_VERSION = 1
MANIFEST = "manifest.ini"
THUMBNAIL = "thumbnail.png"
SELECTION = "selection.png"
PALETTE = "palette.png"


def _esc(value):
    """INI values are one line; keep stray newlines out of metadata."""
    return str(value).replace("\r", " ").replace("\n", " ")


def _plane_path(frame_index, layer_id, plane):
    return "frames/%04d/layers/%s.%s.png" % (frame_index, layer_id, plane)


def _encode_plane(arr, plane):
    buf = io.BytesIO()
    mode = "RGBA" if plane_channels(plane) == 4 else "L"
    Image.fromarray(arr, mode).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _decode_plane(data, plane):
    img = Image.open(io.BytesIO(data))
    img.load()
    want = "RGBA" if plane_channels(plane) == 4 else "L"
    if img.mode != want:
        img = img.convert(want)
    # Pillow hands back a read-only view; callers own what they load.
    return np.array(img, dtype=np.uint8, copy=True)


# ---- saving -------------------------------------------------------------

def save(doc, path, selection=None, thumbnail_max=256):
    """Write a document. Returns the path."""
    lines = []
    add = lines.append

    add("; Ochre document. The pixel data lives in the PNGs alongside this")
    add("; file; everything else is here, in plain text, on purpose.")
    add("")
    add("[Document]")
    add("FormatVersion = %d" % FORMAT_VERSION)
    add("Width = %d" % doc.width)
    add("Height = %d" % doc.height)
    add("Generator = Ochre")
    add("CurrentFrame = %d" % doc.current)
    add("AxisLayout = %s" % _esc(doc.axis_layout))
    add("AxisColumns = %d" % doc.axis_columns)
    if doc.palette is not None:
        add("Palette = %s" % _esc(doc.palette.name or "palette"))
    for key, value in sorted(doc.meta.items()):
        add("Meta.%s = %s" % (key, _esc(value)))
    add("")

    # Layers, flattened. Parent/Index rebuild the tree, so the INI stays flat
    # and a human can read it without tracking indentation.
    def emit_nodes(group, parent_id=""):
        for index, node in enumerate(group.children):
            add("[Layer:%s]" % node.id)
            add("Name = %s" % _esc(node.name))
            add("Type = %s" % ("group" if node.is_group else "raster"))
            add("Parent = %s" % parent_id)
            add("Index = %d" % index)
            add("Opacity = %d" % node.opacity)
            add("Visible = %s" % ("1" if node.visible else "0"))
            add("BlendMode = %s" % _esc(node.blend))
            if not node.is_group:
                add("Planes = %s" % ",".join(node.planes))
                add("Authoritative = %s" % ",".join(node.authoritative))
            for key, value in sorted(node.meta.items()):
                add("Meta.%s = %s" % (key, _esc(value)))
            add("")
            if node.is_group:
                emit_nodes(node, node.id)

    emit_nodes(doc.root)

    for index, frame in enumerate(doc.frames):
        add("[Frame:%s]" % frame.id)
        add("Name = %s" % _esc(frame.name))
        add("Index = %d" % index)
        add("DurationMs = %d" % frame.duration_ms)
        for key, value in sorted(frame.meta.items()):
            add("Meta.%s = %s" % (key, _esc(value)))
        add("")

    if doc.palette is not None:
        pal = doc.palette
        add("[Palette]")
        add("Name = %s" % _esc(pal.name or "palette"))
        add("File = %s" % PALETTE)
        if pal.transparent is not None:
            add("Transparent = %d" % pal.transparent)
        if pal.protected:
            add("Protected = %s" % ",".join(str(i) for i in sorted(pal.protected)))
        for n, (lo, hi, label, role) in enumerate(pal.annotations):
            add("Annotation.%d = %d-%d,%s,%s" % (n, lo, hi, _esc(label), _esc(role)))
        add("")

    # Cells: one section per (layer, frame) that actually has pixels.
    blobs = {}
    frame_index = {f.id: i for i, f in enumerate(doc.frames)}
    for (layer_id, frame_id), surface in sorted(doc.cells.items()):
        node = doc.layer(layer_id)
        if node is None or frame_id not in frame_index:
            continue
        fi = frame_index[frame_id]
        written = []
        for plane in surface.planes:
            arr = surface.planes.get(plane)
            derived = surface.is_derived(plane)
            if arr is None:
                if not derived:
                    continue
                arr = surface.plane(plane)       # materialise for interchange
            rel = _plane_path(fi, layer_id, plane)
            blobs[rel] = _encode_plane(arr, plane)
            written.append((plane, rel, derived))
        if not written:
            continue
        add("[Cell:%s@%s]" % (layer_id, frame_id))
        for plane, rel, derived in written:
            add("%sPlane.%s = %s" % ("; derived, not loaded -- " if derived else "",
                                     plane, rel))
        if surface.content_bbox is not None:
            b = surface.content_bbox
            add("ContentBBox = %d,%d,%d,%d" % (b.x, b.y, b.w, b.h))
        add("")

    manifest = "\n".join(lines) + "\n"

    with zipfile.ZipFile(path, "w") as zf:
        # The manifest compresses well and is tiny; the PNGs are already
        # compressed, so deflating them again costs CPU for nothing.
        zf.writestr(MANIFEST, manifest, zipfile.ZIP_DEFLATED)
        for rel, data in sorted(blobs.items()):
            zf.writestr(rel, data, zipfile.ZIP_STORED)
        if doc.palette is not None:
            zf.writestr(PALETTE,
                        _encode_plane(doc.palette.entries.reshape(1, LENGTH, 4), "rgba"),
                        zipfile.ZIP_STORED)
        if selection is not None and not selection.selects_all():
            zf.writestr(SELECTION, _encode_plane(selection.mask, "mask"),
                        zipfile.ZIP_STORED)
        thumb = _thumbnail(doc, thumbnail_max)
        if thumb is not None:
            zf.writestr(THUMBNAIL, thumb, zipfile.ZIP_STORED)
    return path


def _thumbnail(doc, max_dim):
    try:
        from ..compositor import Compositor
        flat = Compositor(doc).composite()
        if flat is None:
            return None
        img = Image.fromarray(flat, "RGBA")
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # A thumbnail is a convenience. Never let one stop a save.
        return None


# ---- loading ------------------------------------------------------------

def load(path, settings=None):
    """Read a document. Returns (document, selection_mask_or_None)."""
    from ..document import Document

    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        if MANIFEST not in names:
            raise ValueError("%s is not an Ochre document (no manifest)" % path)
        db = IniDB()
        db.load_string(zf.read(MANIFEST).decode("utf-8"), source=path)

        width = db.getint("Document", "Width", 0)
        height = db.getint("Document", "Height", 0)
        if width <= 0 or height <= 0:
            raise ValueError("%s has no usable canvas size" % path)

        palette = None
        if db.has("Palette") and PALETTE in names:
            entries = _decode_plane(zf.read(PALETTE), "rgba").reshape(LENGTH, 4)
            palette = Palette(entries, db.get("Palette", "Name", "palette"))
            transparent = db.get("Palette", "Transparent")
            if transparent is not None:
                palette.set_transparent(db.getint("Palette", "Transparent", 0))
            protected = db.getints("Palette", "Protected")
            if protected:
                palette.protect(*protected)
            for key in db.keys("Palette"):
                if not key.startswith("Annotation."):
                    continue
                parts = (db.get("Palette", key) or "").split(",")
                if len(parts) >= 2:
                    span = parts[0].split("-")
                    try:
                        lo = int(span[0])
                        hi = int(span[1]) if len(span) > 1 else lo
                    except ValueError:
                        continue
                    palette.annotate(lo, hi, parts[1],
                                     parts[2] if len(parts) > 2 else "")

        doc = Document(width, height, palette=palette, settings=settings)
        doc.axis_layout = db.get("Document", "AxisLayout", "strip")
        doc.axis_columns = db.getint("Document", "AxisColumns", 0)
        doc.meta = _meta_of(db, "Document")

        # Frames, in declared order. The document starts with one frame, so
        # reuse it rather than leaving an orphan.
        frame_sections = sorted(db.sections("Frame"),
                                key=lambda fid: db.getint("Frame:" + fid, "Index", 0))
        doc.frames = []
        for fid in frame_sections:
            section = "Frame:" + fid
            from ..frame import Frame
            frame = Frame(name=db.get(section, "Name", ""),
                          duration_ms=db.getint(section, "DurationMs", 100),
                          frame_id=fid)
            frame.meta = _meta_of(db, section)
            doc.frames.append(frame)
        if not doc.frames:
            from ..frame import Frame
            doc.frames = [Frame(name="Frame 1")]
        doc.current = max(0, min(db.getint("Document", "CurrentFrame", 0),
                                 len(doc.frames) - 1))

        # Layers: build every node first, then attach, so a child declared
        # before its parent still lands in the right place.
        nodes, parents = {}, {}
        for lid in db.sections("Layer"):
            section = "Layer:" + lid
            is_group = db.get(section, "Type", "raster") == "group"
            if is_group:
                node = LayerGroup(name=db.get(section, "Name", ""), layer_id=lid)
            else:
                planes = tuple(db.getlist(section, "Planes", ["rgba"]))
                auth = tuple(db.getlist(section, "Authoritative", [planes[0]]))
                auth = tuple(p for p in auth if p in planes) or (planes[0],)
                node = Layer(name=db.get(section, "Name", ""),
                             planes=planes, authoritative=auth, layer_id=lid)
            node.opacity = max(0, min(255, db.getint(section, "Opacity", 255)))
            node.visible = db.getbool(section, "Visible", True)
            node.blend = db.get(section, "BlendMode", "Normal")
            node.meta = _meta_of(db, section)
            nodes[lid] = node
            parents[lid] = (db.get(section, "Parent", "") or "",
                            db.getint(section, "Index", 0))

        for lid in sorted(nodes, key=lambda k: parents[k][1]):
            parent_id = parents[lid][0]
            parent = nodes.get(parent_id) if parent_id else doc.root
            if parent is None or not getattr(parent, "is_group", False):
                parent = doc.root
            parent.add(nodes[lid])

        # Cells. Derived planes are deliberately skipped and rebuilt.
        frame_by_index = {i: f for i, f in enumerate(doc.frames)}
        for key in db.sections("Cell"):
            if "@" not in key:
                continue
            layer_id, frame_id = key.split("@", 1)
            node = nodes.get(layer_id)
            frame = doc.frame_by_id(frame_id)
            if node is None or frame is None or node.is_group:
                continue
            surface = doc.cell(node, frame)
            if surface is None:
                continue
            section = "Cell:" + key
            for opt in db.keys(section):
                if not opt.startswith("Plane."):
                    continue
                plane = opt[len("Plane."):]
                if plane not in surface.planes or surface.is_derived(plane):
                    continue
                rel = db.get(section, opt)
                if rel not in names:
                    continue
                arr = _decode_plane(zf.read(rel), plane)
                target = surface.plane(plane)
                if arr.shape != target.shape:
                    continue
                target[...] = arr
            box = db.getints(section, "ContentBBox")
            if len(box) == 4:
                surface.content_bbox = Rect(*box)
            else:
                surface.content_bbox = surface.bounds
            surface.refresh_derived()

        selection_mask = None
        if SELECTION in names:
            selection_mask = _decode_plane(zf.read(SELECTION), "mask")

    return doc, selection_mask


def _meta_of(db, section):
    out = {}
    for key in db.keys(section):
        if key.startswith("Meta."):
            out[key[len("Meta."):]] = db.get(section, key)
    return out


def is_ochre(path):
    """True if path looks like an Ochre document. Cheap, no full parse."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return MANIFEST in zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return False
