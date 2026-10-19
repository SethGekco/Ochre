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
            if node.is_group and getattr(node, "region", None):
                from ..regions import encode as encode_region
                add("Region = %s" % _esc(encode_region(node.region)))
            if not node.is_group:
                add("Planes = %s" % ",".join(node.planes))
                add("Authoritative = %s" % ",".join(node.authoritative))
                if getattr(node, "kind", "raster") != "raster":
                    add("Kind = %s" % node.kind)
                for key, value in sorted(getattr(node, "params", {}).items()):
                    add("Param.%s = %s" % (key, _esc(_encode_value(value))))
                if getattr(node, "clip", False):
                    add("Clip = 1")
                if getattr(node, "blend_if", None):
                    add("BlendIf = %s" % ",".join(str(int(v)) for v in node.blend_if))
                if getattr(node, "fill_opacity", 255) != 255:
                    add("FillOpacity = %d" % node.fill_opacity)
                if getattr(node, "knockout", "none") != "none":
                    add("Knockout = %s" % node.knockout)
                if getattr(node, "mask", None) is not None:
                    add("Mask = 1")
                    if not node.mask_enabled:
                        add("MaskEnabled = 0")
                from ..layerstyles import encode as encode_styles
                for key, value in sorted(encode_styles(getattr(node, "styles", {})).items()):
                    add("%s = %s" % (key, _esc(value)))
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

    # Vector paths, as text: one value per subpath, anchors and handles in
    # document coordinates (engine/paths.py encode_subpath).
    from ..paths import encode_subpath
    for n, vpath in enumerate(getattr(doc, "paths", ())):
        add("[Path:%04d]" % n)
        add("Name = %s" % _esc(vpath.name))
        if doc.active_path == n:
            add("Active = 1")
        for k, sub in enumerate(vpath.subpaths):
            add("Subpath.%d = %s" % (k, encode_subpath(sub)))
        add("")

    # Saved selections (their pixels are cells, written with the rest) and
    # layer comps (plain state, as one JSON line each).
    for ch in getattr(doc, "channels", ()):
        add("[Channel:%s]" % ch.id)
        add("Name = %s" % _esc(ch.name))
        add("")
    import json as _json
    for n, comp in enumerate(getattr(doc, "comps", ())):
        add("[Comp:%04d]" % n)
        add("Name = %s" % _esc(comp.get("name", "")))
        add("State = %s" % _esc(_json.dumps(comp.get("state", {}), separators=(",", ":"))))
        add("")

    # Guides and the other non-pixel marks (engine/annotations.py).
    from ..annotations import encode_note
    if getattr(doc, "guides", None):
        add("[Guides]")
        for n, (orient, pos) in enumerate(doc.guides):
            add("%s.%d = %g" % ("H" if orient == "h" else "V", n, pos))
        add("")
    marks = [("Sampler", "%g,%g" % p) for p in getattr(doc, "samplers", [])]
    marks += [("Count", "%g,%g" % p) for p in getattr(doc, "counts", [])]
    marks += [("Note", encode_note(*n)) for n in getattr(doc, "notes", [])]
    marks += [("Slice", "%s,%d,%d,%d,%d" % ((_esc(nm).replace(",", "_"),) + tuple(r)))
              for nm, r in getattr(doc, "slices", [])]
    marks += [("Plane", ",".join("%g" % v for p in quad for v in p))
              for quad in getattr(doc, "planes", [])]
    if marks:
        add("[Marks]")
        for n, (kind, value) in enumerate(marks):
            add("%s.%d = %s" % (kind, n, value))
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
            if is_group and db.get(section, "Region"):
                from ..regions import decode as decode_region
                node.region = decode_region(db.get(section, "Region"))
            node.meta = _meta_of(db, section)
            if not is_group:
                from ..layer import LayerMask
                from ..layerstyles import decode as decode_styles
                node.kind = db.get(section, "Kind", "raster")
                node.params = {k[len("Param."):]: _decode_value(db.get(section, k))
                               for k in db.keys(section) if k.startswith("Param.")}
                node.clip = db.getbool(section, "Clip", False)
                bi = db.getints(section, "BlendIf")
                node.blend_if = tuple(bi) if len(bi) == 8 else None
                node.fill_opacity = max(0, min(255, int(db.get(section, "FillOpacity", 255) or 255)))
                ko = db.get(section, "Knockout", "none")
                node.knockout = ko if ko in ("shallow", "deep") else "none"
                if node.kind == "smart":
                    from ..layer import SmartSource
                    sw, sh = node.params.get("src_w"), node.params.get("src_h")
                    node.smart_source = SmartSource(node, (int(sw), int(sh)) if sw and sh else None)
                if db.getbool(section, "Mask", False):
                    node.mask = LayerMask(node)
                    node.mask_enabled = db.getbool(section, "MaskEnabled", True)
                node.styles = decode_styles((k, db.get(section, k)) for k in db.keys(section)
                                            if k.startswith("Style."))
            nodes[lid] = node
            parents[lid] = (db.get(section, "Parent", "") or "",
                            db.getint(section, "Index", 0))

        for lid in sorted(nodes, key=lambda k: parents[k][1]):
            parent_id = parents[lid][0]
            parent = nodes.get(parent_id) if parent_id else doc.root
            if parent is None or not getattr(parent, "is_group", False):
                parent = doc.root
            parent.add(nodes[lid])

        from ..layer import Channel
        for cid in db.sections("Channel"):
            doc.channels.append(Channel(db.get("Channel:" + cid, "Name", "Alpha"), layer_id=cid))
        import json as _json
        for comp_id in sorted(db.sections("Comp")):
            sec = "Comp:" + comp_id
            try:
                state = _json.loads(db.get(sec, "State", "{}"))
            except ValueError:
                state = {}
            doc.comps.append({"name": db.get(sec, "Name", "Layer Comp"), "state": state})

        # Cells. Derived planes are deliberately skipped and rebuilt.
        frame_by_index = {i: f for i, f in enumerate(doc.frames)}
        for key in db.sections("Cell"):
            if "@" not in key:
                continue
            layer_id, frame_id = key.split("@", 1)
            node = nodes.get(layer_id) or doc.layer(layer_id)     # masks resolve via owner
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

        from ..paths import Path, decode_subpath
        for pid in sorted(db.sections("Path")):
            section = "Path:" + pid
            vpath = Path(db.get(section, "Name", "Path"))
            keys = sorted((k for k in db.keys(section) if k.startswith("Subpath.")),
                          key=lambda k: int(k.split(".", 1)[1]) if k.split(".", 1)[1].isdigit() else 0)
            for key in keys:
                vpath.subpaths.append(decode_subpath(db.get(section, key)))
            if db.get(section, "Active", "0") == "1":
                doc.active_path = len(doc.paths)
            doc.paths.append(vpath)

        from ..annotations import decode_note

        def ordered(section):
            def key(k):
                tail = k.split(".", 1)[-1]
                return int(tail) if tail.isdigit() else 0
            return sorted(db.keys(section), key=key)
        if db.has("Guides"):
            for k in ordered("Guides"):
                try:
                    doc.guides.append(("h" if k.startswith("H") else "v",
                                       float(db.get("Guides", k))))
                except (TypeError, ValueError):
                    pass
        if db.has("Marks"):
            for k in ordered("Marks"):
                kind, value = k.split(".", 1)[0], db.get("Marks", k)
                try:
                    if kind in ("Sampler", "Count"):
                        x, y = (float(v) for v in value.split(",")[:2])
                        (doc.samplers if kind == "Sampler" else doc.counts).append((x, y))
                    elif kind == "Note":
                        n = decode_note(value)
                        if n is not None:
                            doc.notes.append(n)
                    elif kind == "Slice":
                        parts = value.split(",")
                        doc.slices.append((parts[0], tuple(int(v) for v in parts[1:5])))
                    elif kind == "Plane":
                        v = [float(t) for t in value.split(",")]
                        if len(v) == 8:
                            doc.planes.append(tuple((v[i], v[i + 1]) for i in range(0, 8, 2)))
                except (TypeError, ValueError, IndexError):
                    pass

        selection_mask = None
        if SELECTION in names:
            selection_mask = _decode_plane(zf.read(SELECTION), "mask")

    return doc, selection_mask


def _encode_value(value):
    import json
    if isinstance(value, dict) or (isinstance(value, (tuple, list)) and value
                                   and isinstance(value[0], dict)):
        return "json:" + json.dumps(value, separators=(",", ":"))
    if isinstance(value, (tuple, list)) and any(isinstance(v, float) for v in value):
        return "json:" + json.dumps(list(value))
    if isinstance(value, (tuple, list)) and not value:
        return "json:[]"
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], (tuple, list)):
        return ";".join(",".join(str(int(v)) for v in pair) for pair in value)
    if isinstance(value, (tuple, list)):
        return ",".join(str(int(v)) for v in value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _decode_value(text):
    """Typed back: comma lists of ints become tuples, numbers numbers."""
    t = str(text).strip()
    if t.startswith("json:"):
        import json
        try:
            return json.loads(t[5:])
        except ValueError:
            return t
    if ";" in t:
        try:
            return tuple(tuple(int(v) for v in pair.split(",")) for pair in t.split(";") if pair)
        except ValueError:
            return t
    if "," in t:
        try:
            return tuple(int(v) for v in t.split(","))
        except ValueError:
            return t
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t


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
