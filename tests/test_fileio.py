#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""File I/O: the .ochre round-trip must be BIT-EXACT, and flat import/export.

A document format that is nearly right is a data-loss bug with a delay on it,
so the round-trip assertions compare pixels with array_equal rather than
eyeballing structure.

Run: python3 tests/test_fileio.py
"""

import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from ochre.engine.document import Document
from ochre.engine.fileio import (export_flat, export_frames, flatten_onto,
                                 FormatTable, import_flat, import_indexed,
                                 is_ochre, load, save, supported_extensions)
from ochre.engine.geometry import Rect
from ochre.engine.ini import IniDB
from ochre.engine.palette import grayscale
from ochre.engine.selection import Selection


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def build_document():
    """A document exercising every structural feature at once."""
    d = Document(48, 32)
    pal = grayscale()
    pal.set_entry(7, (200, 30, 40, 255))
    pal.set_entry(8, (200, 30, 40, 255))     # deliberate duplicate RGB
    pal.set_transparent(0)
    pal.protect((16, 31), 4)
    pal.annotate(16, 31, "Ramp", "remap")
    d.bind_palette(pal)

    bg = d.add_layer("Background")
    d.cell(bg).fill(d.bounds, (10, 20, 30, 255))

    group = d.add_group("Group")
    group.opacity = 200
    group.blend = "Multiply"
    inner = d.add_layer("Inner", parent=group)
    d.cell(inner).fill(Rect(4, 4, 10, 10), (255, 0, 0, 128))
    inner.meta["role"] = "body"

    il = d.add_layer("Indexed", planes=("index", "rgba"), authoritative=("index",))
    cell = d.cell(il)
    cell.fill(Rect(0, 0, 8, 8), 7, name="index")
    cell.fill(Rect(8, 0, 8, 8), 8, name="index")    # the duplicate-RGB pair
    cell.fill(Rect(16, 0, 8, 8), 20, name="index")  # a protected index

    terrain = d.add_layer("Terrain", planes=("rgba", "height"),
                          authoritative=("rgba", "height"))
    tc = d.cell(terrain)
    tc.fill(Rect(0, 16, 48, 16), (0, 90, 0, 255))
    tc.fill(Rect(2, 18, 20, 10), 31, name="height")

    f2 = d.add_frame(name="Second")
    f2.duration_ms = 250
    f2.meta["tag"] = "walk"
    d.cell(bg, f2).fill(d.bounds, (99, 99, 99, 255))

    d.meta["kind"] = "test"
    d.axis_layout = "grid"
    d.axis_columns = 4
    return d


def main():
    tmp = tempfile.mkdtemp(prefix="ochre-test-")

    # ---- .ochre round-trip ----------------------------------------------
    doc = build_document()
    sel = Selection(48, 32).select_ellipse((4, 4, 40, 28))
    path = os.path.join(tmp, "doc.ochre")
    save(doc, path, selection=sel)

    check("file was written", os.path.exists(path))
    check("is_ochre recognises it", is_ochre(path))
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    check("manifest present", "manifest.ini" in names)
    check("thumbnail present", "thumbnail.png" in names)
    check("palette present", "palette.png" in names)
    check("selection present", "selection.png" in names)
    check("single-frame layout is still frames/0000",
          any(n.startswith("frames/0000/") for n in names))
    check("second frame written", any(n.startswith("frames/0001/") for n in names))

    got, sel_mask = load(path)

    check("canvas size preserved", (got.width, got.height) == (48, 32))
    check("document meta preserved", got.meta.get("kind") == "test")
    check("axis layout preserved",
          got.axis_layout == "grid" and got.axis_columns == 4)
    check("frame count preserved", len(got.frames) == 2)
    check("frame name preserved", got.frames[1].name == "Second")
    check("frame duration preserved", got.frames[1].duration_ms == 250)
    check("frame meta preserved", got.frames[1].meta.get("tag") == "walk")

    names_before = [n.name for n in doc.root.leaves()]
    names_after = [n.name for n in got.root.leaves()]
    check("layers preserved in order", names_before == names_after,
          "-- %r vs %r" % (names_before, names_after))

    grp = [n for n in got.root.children if n.is_group]
    check("group survived", len(grp) == 1)
    check("group opacity preserved", grp[0].opacity == 200)
    check("group blend preserved", grp[0].blend == "Multiply")
    check("group child reattached", len(grp[0].children) == 1)
    check("layer meta preserved",
          grp[0].children[0].meta.get("role") == "body")

    # Pixels, bit-exactly, on every plane of every cell.
    mismatches = []
    for (layer_id, frame_id), surface in doc.cells.items():
        node_before = doc.layer(layer_id)
        node_after = None
        for n in got.root.leaves():
            if n.name == node_before.name:
                node_after = n
        if node_after is None:
            mismatches.append("missing layer %s" % node_before.name)
            continue
        fi = doc.frame_index(frame_id)
        other = got.cells.get((node_after.id, got.frames[fi].id))
        if other is None:
            mismatches.append("missing cell %s@%d" % (node_before.name, fi))
            continue
        for plane in surface.authoritative:
            a = surface.plane(plane)
            b = other.plane(plane)
            if not np.array_equal(a, b):
                mismatches.append("%s/%s differs" % (node_before.name, plane))
    check("ALL AUTHORITATIVE PLANES ROUND-TRIP BIT-EXACTLY", not mismatches,
          "-- %r" % mismatches)

    # ---- palette fidelity is where a lossy format would show -------------
    pal = got.palette
    check("palette loaded", pal is not None)
    check("palette entries exact", pal.rgba(7) == (200, 30, 40, 255))
    check("duplicate-RGB entries stay distinct",
          pal.rgba(7) == pal.rgba(8))
    check("transparent index preserved", pal.transparent == 0)
    check("protected indices preserved",
          20 in pal.protected and 4 in pal.protected and 32 not in pal.protected)
    ann = pal.annotation_for(20)
    check("annotation preserved", ann is not None and ann[2] == "Ramp")
    check("annotation role preserved", ann[3] == "remap")

    # The pixels using the duplicate pair must still be independently
    # editable -- the whole point of storing indices rather than colour.
    il_after = [n for n in got.root.leaves() if n.name == "Indexed"][0]
    cell = got.cell(il_after)
    check("indices survived, not just their colours",
          int(cell.plane("index")[0, 0]) == 7 and int(cell.plane("index")[0, 8]) == 8)
    got.set_palette_entry(8, (0, 255, 0, 255))
    check("editing one of the duplicate pair still moves only its pixels",
          tuple(cell.pixels[0, 0]) == (200, 30, 40, 255)
          and tuple(cell.pixels[0, 8]) == (0, 255, 0, 255))

    # A protected index must survive a round-trip untouched.
    check("protected index survived in pixel data",
          int(cell.plane("index")[0, 16]) == 20)

    # ---- non-colour planes ----------------------------------------------
    terr = [n for n in got.root.leaves() if n.name == "Terrain"][0]
    check("extra plane declared", "height" in terr.planes)
    check("extra plane authoritative", "height" in terr.authoritative)
    check("height data round-tripped",
          int(got.cell(terr).plane("height")[20, 4]) == 31)

    # ---- selection -------------------------------------------------------
    check("selection round-tripped", sel_mask is not None)
    check("selection is bit-exact", np.array_equal(sel_mask, sel.mask))

    # ---- unknown keys are ignored (the compat story) --------------------
    with zipfile.ZipFile(path) as zf:
        manifest = zf.read("manifest.ini").decode()
    manifest2 = manifest.replace("[Document]",
                                 "[Document]\nFutureFeature = 42\nAnotherThing = yes")
    path2 = os.path.join(tmp, "future.ochre")
    with zipfile.ZipFile(path, "r") as src, zipfile.ZipFile(path2, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "manifest.ini":
                data = manifest2.encode()
            dst.writestr(item, data)
    future, _ = load(path2)
    check("a file from a newer Ochre still opens",
          future.width == 48 and len(future.frames) == 2)

    # ---- malformed input fails cleanly ----------------------------------
    bad = os.path.join(tmp, "bad.ochre")
    with open(bad, "wb") as f:
        f.write(b"not a zip at all")
    check("a non-zip is not mistaken for a document", not is_ochre(bad))
    try:
        load(bad)
        check("loading garbage raises", False, "-- no error")
    except Exception:
        check("loading garbage raises", True)

    empty_zip = os.path.join(tmp, "empty.ochre")
    with zipfile.ZipFile(empty_zip, "w") as zf:
        zf.writestr("readme.txt", "hello")
    check("a zip without a manifest is rejected", not is_ochre(empty_zip))
    try:
        load(empty_zip)
        check("loading a manifest-less zip raises", False, "-- no error")
    except ValueError:
        check("loading a manifest-less zip raises", True)

    # ---- empty document --------------------------------------------------
    blank = Document(16, 16)
    bpath = os.path.join(tmp, "blank.ochre")
    save(blank, bpath)
    back, _ = load(bpath)
    check("an empty document round-trips",
          back.width == 16 and back.layers() == [] and len(back.frames) == 1)

    # ---- flat export/import ---------------------------------------------
    doc = build_document()
    png = os.path.join(tmp, "flat.png")
    export_flat(doc, png)
    check("PNG exported", os.path.exists(png))
    img = Image.open(png)
    check("PNG has alpha", img.mode == "RGBA")
    check("PNG size matches canvas", img.size == (48, 32))

    reimported = import_flat(png)
    check("flat import makes a one-layer document", len(reimported.layers()) == 1)
    check("flat import size matches", (reimported.width, reimported.height) == (48, 32))
    from ochre.engine.compositor import Compositor
    check("flat export/import round-trips the composite",
          np.array_equal(np.asarray(img, dtype=np.uint8),
                         Compositor(doc).composite()))

    jpg = os.path.join(tmp, "flat.jpg")
    export_flat(doc, jpg)
    check("JPEG exported despite having no alpha", os.path.exists(jpg))
    check("JPEG is RGB, flattened onto the matte",
          Image.open(jpg).mode == "RGB")

    try:
        export_flat(doc, os.path.join(tmp, "nope.xyz"))
        check("unknown extension rejected", False, "-- no error")
    except ValueError:
        check("unknown extension rejected", True)

    # ---- flatten_onto is alpha-correct -----------------------------------
    half = np.zeros((1, 1, 4), dtype=np.uint8)
    half[0, 0] = (0, 0, 0, 128)                 # half-transparent black
    onto_white = flatten_onto(half, (255, 255, 255))
    check("half-alpha black over white lands mid-grey",
          120 <= int(onto_white[0, 0, 0]) <= 135,
          "-- got %d" % int(onto_white[0, 0, 0]))
    check("flatten produces opaque output", int(onto_white[0, 0, 3]) == 255)
    clear = np.zeros((1, 1, 4), dtype=np.uint8)
    check("fully transparent takes the matte exactly",
          tuple(flatten_onto(clear, (10, 20, 30))[0, 0]) == (10, 20, 30, 255))

    # ---- indexed import is lossless --------------------------------------
    pal_img = Image.new("P", (8, 8))
    palette_bytes = []
    for i in range(256):
        palette_bytes += [i, (i * 3) % 256, (i * 7) % 256]
    pal_img.putpalette(palette_bytes)
    pal_img.putpixel((0, 0), 5)
    pal_img.putpixel((1, 0), 200)
    # PNG-8, not GIF: see the note below on why GIF cannot be used here.
    indexed_png = os.path.join(tmp, "indexed.png")
    pal_img.save(indexed_png)

    idoc = import_indexed(indexed_png)
    check("indexed import produced an index-locked layer",
          idoc.layers()[0].index_locked)
    check("indexed import bound a palette", idoc.palette is not None)
    icell = idoc.cell(idoc.layers()[0])
    check("indices imported verbatim, not via RGB",
          int(icell.plane("index")[0, 0]) == 5
          and int(icell.plane("index")[0, 1]) == 200,
          "-- got %d and %d" % (int(icell.plane("index")[0, 0]),
                                int(icell.plane("index")[0, 1])))
    check("imported palette colours match the source",
          idoc.palette.rgba(5) == (5, 15, 35, 255))

    # An RGBA source must still go through the plain path.
    check("import_indexed falls back for non-paletted input",
          not import_indexed(png).layers()[0].index_locked)

    # Documented limitation, asserted so it cannot silently change: PIL's GIF
    # encoder OPTIMISES the palette down to the colours actually used, which
    # REMAPS indices. GIF therefore cannot carry index identity, and any
    # format addon that needs exact indices must not route through it.
    gif_remap = os.path.join(tmp, "remapped.gif")
    pal_img.save(gif_remap)
    regif = np.asarray(Image.open(gif_remap))
    check("GIF remaps indices (a limitation, not a bug)",
          int(regif[0, 0]) != 5,
          "-- GIF preserved indices; the workaround below may be removable")

    # ---- animation export ------------------------------------------------
    anim = os.path.join(tmp, "anim.gif")
    export_frames(build_document(), anim)
    check("animation exported", os.path.exists(anim))
    gif_in = Image.open(anim)
    check("animation has both frames", getattr(gif_in, "n_frames", 1) == 2,
          "-- got %d" % getattr(gif_in, "n_frames", 1))

    try:
        export_frames(build_document(), os.path.join(tmp, "still.png"))
        check("single-frame format rejects an animation", False, "-- no error")
    except ValueError:
        check("single-frame format rejects an animation", True)

    # ---- format table is INI-driven --------------------------------------
    table = FormatTable()
    check("known extensions listed", "png" in supported_extensions())
    check("PNG carries alpha", table.supports_alpha("PNG"))
    check("JPEG does not", not table.supports_alpha("JPEG"))

    db = IniDB()
    db.load_string("[Format:PNG]\nAlpha = off\nMatte = 1,2,3\n")
    table = FormatTable(db)
    check("INI can override a format's alpha support",
          not table.supports_alpha("PNG"))
    check("INI can set the matte", table.formats["PNG"]["matte"] == (1, 2, 3))

    real = FormatTable(IniDB(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")))
    check("shipped fileformats.ini parses", "jpg" in real.extensions())

    print("\nall file I/O checks passed")


if __name__ == "__main__":
    main()
