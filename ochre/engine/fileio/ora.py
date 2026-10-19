# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""OpenRaster (.ora): the open layered format GIMP, Krita and MyPaint share.

A zip holding `mimetype` (stored first, uncompressed: "image/openraster"),
`stack.xml` describing the layer tree, one PNG per layer, the flattened
`mergedimage.png` and a thumbnail. Layer order in stack.xml is TOP first.

Kept: layer names, position, opacity, visibility, blend modes (the svg:
composite operations), groups. Masks, styles and fill layers are flattened
into the layer's pixels -- OpenRaster has no masks -- and adjustment layers
cannot be expressed, which the saver reports as a warning.
"""

import io
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

MIMETYPE = b"image/openraster"

OPS = {
    "svg:src-over": "Normal", "svg:multiply": "Multiply", "svg:screen": "Screen",
    "svg:overlay": "Overlay", "svg:darken": "Darken", "svg:lighten": "Lighten",
    "svg:color-dodge": "ColorDodge", "svg:color-burn": "ColorBurn",
    "svg:hard-light": "HardLight", "svg:soft-light": "SoftLight",
    "svg:difference": "Difference", "svg:exclusion": "Exclusion", "svg:hue": "Hue",
    "svg:saturation": "Saturation", "svg:color": "Color", "svg:luminosity": "Luminosity",
    "svg:plus": "Additive",
}
OP_FOR = {v: k for k, v in OPS.items()}


def is_ora(path):
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.read("mimetype").strip() == MIMETYPE
    except (OSError, KeyError, zipfile.BadZipFile):
        return False


def load(path, settings=None):
    from ..document import Document
    from ..geometry import Rect
    from ..layer import LayerGroup
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("stack.xml"))
        width, height = int(root.get("w")), int(root.get("h"))
        doc = Document(width, height, settings=settings)

        def walk(stack_el, parent):
            # stack.xml lists the TOP layer first; Ochre's children are
            # bottom-first, so walk in reverse.
            for el in reversed(list(stack_el)):
                visible = el.get("visibility", "visible") != "hidden"
                opacity = int(round(float(el.get("opacity", "1")) * 255))
                blend = OPS.get(el.get("composite-op", "svg:src-over"), "Normal")
                if el.tag == "stack":
                    group = LayerGroup(name=el.get("name", "Group"))
                    if el.get("isolation", "auto") != "isolate" and blend == "Normal":
                        blend = "PassThrough"
                    group.visible, group.opacity, group.blend = visible, opacity, blend
                    parent.add(group)
                    walk(el, group)
                elif el.tag == "layer":
                    src = el.get("src")
                    node = doc.add_layer(el.get("name", "Layer"), parent=parent)
                    node.visible, node.opacity, node.blend = visible, opacity, blend
                    if not src:
                        continue
                    try:
                        img = Image.open(io.BytesIO(zf.read(src))).convert("RGBA")
                    except (KeyError, OSError):
                        continue
                    px = np.asarray(img)
                    x, y = int(float(el.get("x", "0"))), int(float(el.get("y", "0")))
                    dst = Rect(x, y, px.shape[1], px.shape[0]).clipped_to(width, height)
                    if dst is None:
                        continue
                    cell = doc.cell(node)
                    cell.plane("rgba")[dst.slice()] = px[dst.y - y:dst.y - y + dst.h,
                                                         dst.x - x:dst.x - x + dst.w]
                    cell._touched(dst)
        top = root.find("stack")
        if top is not None:
            walk(top, doc.root)
    if not doc.layers():
        doc.add_layer("Background")
    return doc


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr), "RGBA").save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def save(doc, path, compositor=None):
    """Write OpenRaster. Returns warnings about what could not be kept."""
    from ..compositor import Compositor, _Leaf
    comp = compositor or Compositor(doc)
    frame = doc.frame
    warnings = []
    files = {}
    counter = [0]

    def emit(group, parent_el):
        # Top first in stack.xml.
        for node in reversed(group.children):
            attrs = {"name": node.name or "", "opacity": "%.4f" % (node.opacity / 255.0),
                     "visibility": "visible" if node.visible else "hidden",
                     "composite-op": OP_FOR.get(node.blend, "svg:src-over")}
            if node.is_group and node.blend == "PassThrough":
                attrs["composite-op"] = "svg:src-over"
            elif node.blend not in OP_FOR:
                warnings.append("blend mode %s on %r has no OpenRaster equivalent; saved as Normal"
                                % (node.blend, node.name))
            if node.is_group:
                attrs["isolation"] = "auto" if node.blend == "PassThrough" else "isolate"
                el = ET.SubElement(parent_el, "stack", attrs)
                emit(node, el)
                continue
            kind = getattr(node, "kind", "raster")
            if kind == "smart":
                warnings.append("smart object %r is saved as pixels (it stays smart in .ochre)"
                                % node.name)
            if kind == "text":
                warnings.append("text layer %r is saved as pixels (it stays editable in .ochre)"
                                % node.name)
            if kind == "adjustment":
                warnings.append("adjustment layer %r is not kept in OpenRaster" % node.name)
                continue
            if getattr(node, "mask", None) is not None or getattr(node, "styles", None) \
                    or getattr(node, "clip", False):
                warnings.append("mask / styles / clipping on %r are flattened into its pixels"
                                % node.name)
            cell = doc.cells.get((node.id, frame.id))
            px = np.zeros((doc.height, doc.width, 4), np.uint8)
            if kind != "raster" or (cell is not None and not cell.is_empty()):
                leaf = _Leaf(cell, comp.blends.opcode("Normal"), 255, node)
                comp._blend_leaf(px, leaf, doc.bounds, frame)
            ys, xs = np.nonzero(px[..., 3])
            if ys.size:
                y0, x0, y1, x1 = int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
            else:
                y0 = x0 = 0
                y1 = x1 = 1
            counter[0] += 1
            src = "data/layer%03d.png" % counter[0]
            files[src] = _png(px[y0:y1, x0:x1])
            attrs.update({"src": src, "x": str(x0), "y": str(y0)})
            ET.SubElement(parent_el, "layer", attrs)

    root = ET.Element("image", {"version": "0.0.5", "w": str(doc.width), "h": str(doc.height),
                                "xres": "72", "yres": "72"})
    top = ET.SubElement(root, "stack")
    emit(doc.root, top)
    merged = comp.composite(doc.bounds, frame)
    thumb = Image.fromarray(merged, "RGBA")
    thumb.thumbnail((256, 256))
    tb = io.BytesIO()
    thumb.save(tb, "PNG")
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, MIMETYPE)
        zf.writestr("stack.xml", ET.tostring(root, encoding="utf-8", xml_declaration=True),
                    zipfile.ZIP_DEFLATED)
        for name, data in files.items():
            zf.writestr(name, data, zipfile.ZIP_STORED)
        zf.writestr("mergedimage.png", _png(merged), zipfile.ZIP_STORED)
        zf.writestr("Thumbnails/thumbnail.png", tb.getvalue(), zipfile.ZIP_STORED)
    return warnings
