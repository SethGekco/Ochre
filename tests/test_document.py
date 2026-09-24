#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Document: the layer tree, the frame axis, sparse cells, residency.

This covers the structural decisions the plan flags as un-retrofittable, so
it leans on the invariants rather than on API surface.

Run: python3 tests/test_document.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.document import Document
from ochre.engine.frame import COLD, WARM
from ochre.engine.geometry import Rect
from ochre.engine.layer import Layer, LayerGroup
from ochre.engine.palette import grayscale


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- the degenerate case costs nothing -------------------------------
    d = Document(64, 48)
    check("one frame by default", len(d.frames) == 1)
    check("no layers by default", d.layers() == [])
    check("no cells by default", len(d.cells) == 0)
    check("no pixel memory by default", d.nbytes() == 0)
    check("bounds", d.bounds == Rect(0, 0, 64, 48))

    try:
        Document(0, 10)
        check("zero-size document rejected", False, "-- no error")
    except ValueError:
        check("zero-size document rejected", True)

    # ---- layers ----------------------------------------------------------
    base = d.add_layer("Background")
    top = d.add_layer("Ink")
    check("layers are ordered bottom-first",
          [n.name for n in d.layers()] == ["Background", "Ink"])
    check("lookup by id", d.layer(base.id) is base)
    check("lookup of a stranger is None", d.layer("nope") is None)
    check("default opacity is integer 255",
          top.opacity == 255 and isinstance(top.opacity, int))

    g = d.add_group("Effects")
    inner = d.add_layer("Glow", parent=g)
    check("group holds its child", inner.parent is g)
    check("child depth", inner.depth() == 2)
    check("walk reaches nested leaves",
          {n.name for n in d.root.leaves()} == {"Background", "Ink", "Glow"})

    # flat() is what a panel renders: top-most first.
    flat = d.root.flat()
    check("flat lists top-most first", flat[0][1].name == "Effects")
    check("flat indents children", flat[1] == (1, inner))

    # Visibility is inherited through groups.
    check("visible by default", inner.effectively_visible())
    g.visible = False
    check("hidden group hides its children", not inner.effectively_visible())
    check("hidden group does not change the child's own flag", inner.visible)
    g.visible = True

    # A group cannot contain itself.
    try:
        g.add(g)
        check("self-parenting rejected", False, "-- no error")
    except ValueError:
        check("self-parenting rejected", True)
    try:
        inner_group = d.add_group("Nested", parent=g)
        inner_group.add(g)
        check("ancestor cycle rejected", False, "-- no error")
    except ValueError:
        check("ancestor cycle rejected", True)

    # Re-parenting detaches from the old parent rather than duplicating.
    d.root.add(inner)
    check("reparent moves, not copies", inner.parent is d.root and inner not in g.children)

    # ---- cells are sparse and lazy --------------------------------------
    d = Document(32, 32)
    a = d.add_layer("A")
    b = d.add_layer("B")
    check("no cell until asked", not d.has_cell(a))
    cell = d.cell(a)
    check("cell created on demand", d.has_cell(a))
    check("cell matches canvas", (cell.width, cell.height) == (32, 32))
    check("cell is still unallocated", cell.nbytes() == 0)
    check("cell is cached", d.cell(a) is cell)
    check("create=False does not create", d.cell(b, create=False) is None)
    check("absent layer yields None", d.cell("nope") is None)

    # ---- frames ----------------------------------------------------------
    d = Document(16, 16)
    layer = d.add_layer("Art")
    f1 = d.frame
    d.cell(layer).fill(Rect(0, 0, 4, 4), (255, 0, 0, 255))

    f2 = d.add_frame()
    check("frame appended", len(d.frames) == 2)
    check("new frame has no cells", not d.has_cell(layer, f2))
    check("a layer keeps its identity across frames", d.layers() == [layer])

    # This is the point of the cells model: same layer, different pixels.
    d.cell(layer, f2).fill(Rect(0, 0, 4, 4), (0, 0, 255, 255))
    check("cells are per (layer, frame)",
          tuple(d.cell(layer, f1).pixels[0, 0]) == (255, 0, 0, 255)
          and tuple(d.cell(layer, f2).pixels[0, 0]) == (0, 0, 255, 255))

    f3 = d.add_frame(copy_from=f1)
    check("copy_from duplicates cells",
          tuple(d.cell(layer, f3).pixels[0, 0]) == (255, 0, 0, 255))
    d.cell(layer, f3).fill(Rect(0, 0, 4, 4), (0, 255, 0, 255))
    check("copied cells are independent",
          tuple(d.cell(layer, f1).pixels[0, 0]) == (255, 0, 0, 255))

    # Insertion must not disturb existing cells -- they are keyed by frame
    # id, not index, precisely so this holds.
    mid = d.add_frame(index=1)
    check("insert shifts order", d.frames[1] is mid)
    check("insert does not disturb existing cells",
          tuple(d.cell(layer, f2).pixels[0, 0]) == (0, 0, 255, 255))

    check("frame_index tracks position", d.frame_index(f2.id) == 2)

    # Removing a frame drops its cells and nothing else.
    before = len(d.cells)
    d.remove_frame(f2)
    check("remove_frame drops its cells", len(d.cells) < before)
    check("remove_frame leaves others", d.has_cell(layer, f1))
    while len(d.frames) > 1:
        d.remove_frame(d.frames[-1])
    check("the last frame cannot be removed", d.remove_frame(d.frames[0]) is False)

    # Removing a layer drops its cells across every frame.
    d = Document(16, 16)
    lay = d.add_layer("Gone")
    keep = d.add_layer("Kept")
    fb = d.add_frame()
    d.cell(lay); d.cell(lay, fb); d.cell(keep)
    check("cells exist across frames", len(d.cells) == 3)
    d.remove_layer(lay)
    check("removing a layer drops all its cells", len(d.cells) == 1)
    check("other layers survive", d.has_cell(keep))

    # Removing a group drops its descendants' cells too.
    d = Document(16, 16)
    grp = d.add_group("G")
    child = d.add_layer("C", parent=grp)
    d.cell(child)
    check("group child has a cell", len(d.cells) == 1)
    d.remove_layer(grp)
    check("removing a group drops descendant cells", len(d.cells) == 0)

    # ---- palette is document-level --------------------------------------
    d = Document(16, 16)
    pal = grayscale()
    pal.set_entry(9, (200, 50, 50, 255))
    d.bind_palette(pal)
    art = d.add_layer("Art", planes=("index", "rgba"), authoritative=("index",))
    f_b = d.add_frame()
    d.cell(art).fill(Rect(0, 0, 4, 4), 9, name="index")
    d.cell(art, f_b).fill(Rect(0, 0, 4, 4), 9, name="index")
    check("index-locked cell renders via the palette",
          tuple(d.cell(art).pixels[0, 0]) == (200, 50, 50, 255))

    # One palette edit must reach every frame at once. That requirement is
    # exactly why the palette is document-level and not per-layer.
    touched = d.set_palette_entry(9, (10, 220, 10, 255))
    check("palette edit reaches every frame", len(touched) == 2,
          "-- touched %d cells" % len(touched))
    check("frame A recoloured", tuple(d.cell(art).pixels[0, 0]) == (10, 220, 10, 255))
    check("frame B recoloured", tuple(d.cell(art, f_b).pixels[0, 0]) == (10, 220, 10, 255))
    check("index planes untouched by a palette edit",
          int(d.cell(art).plane("index")[0, 0]) == 9)

    # ---- converting an existing layer to index-locked --------------------
    d = Document(16, 16)
    d.bind_palette(grayscale())
    rgba_layer = d.add_layer("Photo")
    d.cell(rgba_layer).fill(Rect(0, 0, 8, 8), (128, 128, 128, 255))
    d.lock_layer_to_index(rgba_layer)
    check("layer reports index-locked", rgba_layer.index_locked)
    cell = d.cell(rgba_layer)
    check("cell gained an index plane", cell.has("index") and cell.index_locked)
    got = int(cell.plane("index")[0, 0])
    check("existing pixels snapped into indices", abs(got - 128) <= 6,
          "-- 128 grey snapped to index %d" % got)

    # ---- non-colour planes ----------------------------------------------
    d = Document(16, 16)
    terrain = d.add_layer("Terrain", planes=("rgba", "height"),
                          authoritative=("rgba", "height"))
    cell = d.cell(terrain)
    before = cell.pixels.copy()
    cell.fill(Rect(2, 2, 4, 4), 31, name="height")
    check("height plane writes", int(cell.plane("height")[2, 2]) == 31)
    check("height write does not touch colour", np.array_equal(cell.pixels, before))
    check("one content_bbox for both planes", cell.content_bbox == Rect(2, 2, 4, 4))

    lay = Layer("L")
    check("add_plane attaches and makes authoritative",
          lay.add_plane("height") and "height" in lay.authoritative)
    check("add_plane is idempotent", lay.add_plane("height") is False)

    # ---- residency -------------------------------------------------------
    d = Document(64, 64)
    lay = d.add_layer("Art")
    frames = [d.frame] + [d.add_frame() for _ in range(5)]
    for f in frames:
        d.cell(lay, f).fill(Rect(0, 0, 32, 32), (7, 7, 7, 255))
    warm_before = d.warm_bytes()
    check("all frames warm initially", warm_before > 0)

    cooled = d.enforce_residency(max_warm=2)
    check("enforce_residency cools distant frames", cooled > 0)
    check("current frame stays warm", d.frames[d.current].residency == WARM)
    cold = [f for f in d.frames if f.residency == COLD]
    check("some frames went cold", len(cold) > 0)
    check("warm memory dropped", d.warm_bytes() < warm_before)

    # Cold -> warm must be bit-exact. A lossy residency policy would be a
    # silent data-corruption bug.
    target = cold[-1]
    idx = d.frame_index(target.id)
    d.set_current_frame(idx)
    check("switching thaws the target", d.frames[idx].residency == WARM)
    check("thawed pixels are bit-exact",
          tuple(d.cell(lay, d.frames[idx]).pixels[0, 0]) == (7, 7, 7, 255))
    check("thawed content_bbox survives",
          d.cell(lay, d.frames[idx]).content_bbox is not None)

    # Cold frames of an index-locked layer must rebuild their derived cache.
    d2 = Document(32, 32)
    d2.bind_palette(grayscale())
    il = d2.add_layer("IL", planes=("index", "rgba"), authoritative=("index",))
    f_other = d2.add_frame()
    d2.cell(il).fill(Rect(0, 0, 8, 8), 100, name="index")
    d2.cell(il, f_other).fill(Rect(0, 0, 8, 8), 200, name="index")
    d2.frames[0].freeze(d2.cells)
    check("index-locked frame freezes", d2.frames[0].residency == COLD)
    d2.frames[0].thaw(d2.cells)
    check("index plane restored", int(d2.cell(il).plane("index")[0, 0]) == 100)
    check("derived cache rebuilt on thaw",
          tuple(d2.cell(il).pixels[0, 0]) == (100, 100, 100, 255))

    # ---- dirty aggregation ----------------------------------------------
    d = Document(32, 32)
    lay = d.add_layer("A")
    d.cell(lay).fill(Rect(1, 1, 2, 2), (1, 1, 1, 255))
    d.invalidate(Rect(20, 20, 2, 2))
    rects = d.take_dirty()
    check("take_dirty collects cell and frame dirt", len(rects) >= 1)
    covered = rects[0]
    for r in rects[1:]:
        covered = covered.union(r)
    check("dirt covers both writes",
          covered.contains_rect(Rect(1, 1, 2, 2)) and covered.contains_rect(Rect(20, 20, 2, 2)))
    check("take_dirty drains", d.take_dirty() == [])

    # ---- meta dicts round-trip intent -----------------------------------
    d = Document(8, 8)
    lay = d.add_layer("A")
    d.meta["kind"] = "sprite"
    lay.meta["role"] = "body"
    d.frame.meta["tag"] = "walk"
    check("document meta", d.meta["kind"] == "sprite")
    check("layer meta", lay.meta["role"] == "body")
    check("frame meta", d.frame.meta["tag"] == "walk")

    # ---- axis layout is a hint, not a second axis ------------------------
    d = Document(8, 8)
    check("default layout is a strip", d.axis_layout == "strip")
    d.axis_layout, d.axis_columns = "grid", 4
    check("grid layout is metadata only",
          d.axis_layout == "grid" and d.axis_columns == 4 and len(d.frames) == 1)

    print("\nall document checks passed")


if __name__ == "__main__":
    main()
