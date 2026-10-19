# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Undoable commands, each storing exactly one direction.

The model is Paint.NET's inverse-generating memento, and it is better than the
obvious alternative in a way worth spelling out. A command does NOT store a
"before" and an "after". It stores one state, and `undo()` captures whatever
is currently on the canvas into a freshly-built opposite command *before*
restoring. Three consequences:

  - Undo memory halves. Only one direction is ever resident per entry.
  - Undo and redo become structurally incapable of disagreeing, because redo
    is always derived from what was actually there rather than from what
    something predicted would be there.
  - Undo and redo are the same code path. `undo()` on a redo-command is a
    redo. There is no second implementation to keep in sync.

Compression is zlib level 1, deliberately. Level 6 costs several times the CPU
for a modest ratio gain, and this runs on mouse-up where latency is felt.
Measured on realistic brush content: 41% at 256x256 in 2.8 ms.

Only AUTHORITATIVE planes are recorded. On an index-locked layer that means
the index plane alone -- the RGBA cache is reconstructible, so recording it
would be paying twice for the same information. That is why index-locked
layers cost about a quarter of the undo memory of plain RGBA ones.
"""

import zlib

import numpy as np

from .geometry import Rect


class Command:
    """Base: apply once, then undo/redo by generating the opposite."""

    label = "Command"

    def undo(self, doc):
        """Return (opposite_command, dirty_rect) and apply this command's state.

        Called for both undo and redo -- the direction is whichever one this
        instance happens to hold.
        """
        raise NotImplementedError

    def nbytes(self):
        return 0


class PixelDelta(Command):
    """A rectangle of one cell's authoritative planes."""

    __slots__ = ("label", "layer_id", "frame_id", "rect", "_planes", "_level")

    def __init__(self, layer_id, frame_id, rect, planes, label="Paint", level=1):
        self.label = label
        self.layer_id = layer_id
        self.frame_id = frame_id
        self.rect = rect
        self._level = level
        # {name: (compressed_bytes, shape, dtype_str)}
        self._planes = {}
        for name, arr in planes.items():
            arr = np.ascontiguousarray(arr)
            self._planes[name] = (zlib.compress(arr.tobytes(), level),
                                  arr.shape, arr.dtype.str)

    @staticmethod
    def capture(doc, layer_id, frame_id, rect, label="Paint", level=1):
        """Snapshot a cell's authoritative planes over rect."""
        surface = doc.cells.get((layer_id, frame_id))
        if surface is None:
            return None
        r = rect.clipped_to(surface.width, surface.height)
        if r is None:
            return None
        return PixelDelta(layer_id, frame_id, r, surface.read(r), label, level)

    def _decompress(self):
        out = {}
        for name, (blob, shape, dtype) in self._planes.items():
            arr = np.frombuffer(zlib.decompress(blob), dtype=np.dtype(dtype))
            out[name] = arr.reshape(shape)
        return out

    def undo(self, doc):
        surface = doc.cells.get((self.layer_id, self.frame_id))
        if surface is None:
            return None, None
        # Capture what is there NOW into the opposite command, first.
        opposite = PixelDelta(self.layer_id, self.frame_id, self.rect,
                              surface.read(self.rect), self.label, self._level)
        for name, arr in self._decompress().items():
            surface.plane(name)[self.rect.slice()] = arr
        # A derived cache is not stored, so rebuild it from its source.
        surface.refresh_derived(self.rect)
        surface.dirty.add(self.rect)
        return opposite, self.rect

    def nbytes(self):
        return sum(len(blob) for blob, _, _ in self._planes.values())

    def __repr__(self):
        return ("PixelDelta(%s@%s, %r, planes=%s, %d bytes)"
                % (self.layer_id, self.frame_id, self.rect,
                   sorted(self._planes), self.nbytes()))


class PropertyDelta(Command):
    """One attribute of one object. Tens of bytes, never a pixel copy."""

    __slots__ = ("label", "target_id", "attr", "value", "kind")

    def __init__(self, target_id, attr, value, label=None, kind="layer"):
        self.target_id = target_id
        self.attr = attr
        self.value = value
        self.kind = kind
        self.label = label or ("Change %s" % attr)

    def _resolve(self, doc):
        if self.kind == "layer":
            return doc.layer(self.target_id)
        if self.kind == "frame":
            return doc.frame_by_id(self.target_id)
        if self.kind == "document":
            return doc
        return None

    def undo(self, doc):
        target = self._resolve(doc)
        if target is None:
            return None, None
        opposite = PropertyDelta(self.target_id, self.attr,
                                 getattr(target, self.attr), self.label, self.kind)
        setattr(target, self.attr, self.value)
        return opposite, doc.bounds

    def nbytes(self):
        return 64

    def __repr__(self):
        return "PropertyDelta(%s.%s = %r)" % (self.target_id, self.attr, self.value)


class ReorderDelta(Command):
    """Where a node sits in the tree. Moves no pixels, so it costs two ints.

    A PropertyDelta cannot express this: position is not an attribute of the
    node, it is the parent's ordering of its children. Reordering changes
    what the composite looks like without touching a single pixel, which is
    exactly the case the property/pixel split was drawn around.
    """

    __slots__ = ("label", "target_id", "parent_id", "index")

    def __init__(self, target_id, parent_id, index, label="Reorder"):
        self.target_id = target_id
        self.parent_id = parent_id      # None means the root
        self.index = index
        self.label = label

    def undo(self, doc):
        node = doc.layer(self.target_id)
        if node is None:
            return None, None
        # Inverse-generating, like every other command here: capture where
        # the node is NOW before moving it, so redo is derived from reality
        # rather than from a prediction made earlier.
        here = node.parent
        opposite = ReorderDelta(
            self.target_id,
            None if here is None or here is doc.root else here.id,
            here.children.index(node) if here is not None else 0,
            self.label)

        parent = doc.root if self.parent_id is None else doc.layer(self.parent_id)
        if parent is None:
            return None, None
        parent.add(node, self.index)
        return opposite, doc.bounds

    def nbytes(self):
        return 64

    def __repr__(self):
        return "ReorderDelta(%s -> %s[%d])" % (self.target_id, self.parent_id,
                                               self.index)


class PaletteDelta(Command):
    """One palette entry. The cheapest meaningful edit in the system.

    A palette edit changes no authoritative pixel data at all -- the index
    planes are untouched -- so this carries four bytes and an index, and undo
    is exact no matter how many times an entry has been recoloured. An
    RGBA-only editor would accumulate rounding error on every recolour and
    could never reverse one precisely.
    """

    __slots__ = ("label", "index", "rgba")

    def __init__(self, index, rgba, label="Change palette colour"):
        self.label = label
        self.index = int(index)
        self.rgba = tuple(rgba)

    def undo(self, doc):
        if doc.palette is None:
            return None, None
        opposite = PaletteDelta(self.index, doc.palette.rgba(self.index), self.label)
        doc.set_palette_entry(self.index, self.rgba)
        return opposite, doc.bounds

    def nbytes(self):
        return 32


class CompoundCommand(Command):
    """Several commands undone as one.

    Undoing walks the children in REVERSE and collects each one's generated
    opposite into a new compound -- so "apply this effect to every frame"
    becomes a single entry that reverses with one keystroke.

    The ordering of the opposites is subtle enough to be worth writing down,
    because reversing them "so the list reads in original order" is a natural
    instinct and it is wrong. Given children [a, b, c] applied in that order,
    undo runs c, b, a and collects [c', b', a']. Redo must then apply
    a', b', c' -- and since undo walks in reverse, storing the opposites in
    exactly the order they were collected produces precisely that. Reversing
    them makes redo replay the group backwards, which round-trips only when
    the children happen to commute.
    """

    __slots__ = ("label", "children")

    def __init__(self, children, label="Multiple changes"):
        self.label = label
        self.children = list(children)

    def undo(self, doc):
        opposites = []
        covered = None
        for child in reversed(self.children):
            opposite, rect = child.undo(doc)
            if opposite is not None:
                opposites.append(opposite)
            if rect is not None:
                covered = rect if covered is None else covered.union(rect)
        return CompoundCommand(opposites, self.label), covered

    def nbytes(self):
        return sum(c.nbytes() for c in self.children)

    def __len__(self):
        return len(self.children)

    def __repr__(self):
        return "CompoundCommand(%d children, %r)" % (len(self.children), self.label)


class CellSnapshot(Command):
    """A whole cell, present or absent. Backs layer add/remove.

    Deleting a layer has to retain its pixels somewhere, and this is that
    somewhere. Snapshots are cropped to content_bbox, which on a sparse layer
    is usually far smaller than the canvas.
    """

    __slots__ = ("label", "layer_id", "frame_id", "_delta", "_existed")

    def __init__(self, layer_id, frame_id, delta, existed, label="Layer"):
        self.label = label
        self.layer_id = layer_id
        self.frame_id = frame_id
        self._delta = delta
        self._existed = existed

    @staticmethod
    def capture(doc, layer_id, frame_id, label="Layer", level=1):
        surface = doc.cells.get((layer_id, frame_id))
        if surface is None:
            return CellSnapshot(layer_id, frame_id, None, False, label)
        box = surface.content_bbox or Rect(0, 0, 1, 1)
        delta = PixelDelta(layer_id, frame_id, box, surface.read(box), label, level)
        return CellSnapshot(layer_id, frame_id, delta, True, label)

    def undo(self, doc):
        opposite = CellSnapshot.capture(doc, self.layer_id, self.frame_id, self.label)
        key = (self.layer_id, self.frame_id)
        if not self._existed:
            doc.cells.pop(key, None)
            return opposite, doc.bounds
        node = doc.layer(self.layer_id)
        if node is None:
            return opposite, None
        surface = doc.cell(node, doc.frame_by_id(self.frame_id))
        if surface is not None and self._delta is not None:
            for name, arr in self._delta._decompress().items():
                surface.plane(name)[self._delta.rect.slice()] = arr
            surface.refresh_derived(self._delta.rect)
            surface.content_bbox = self._delta.rect
            surface.dirty.add(self._delta.rect)
        return opposite, doc.bounds

    def nbytes(self):
        return 0 if self._delta is None else self._delta.nbytes()


# ---- structure --------------------------------------------------------------
#
# Adding or deleting a layer was not undoable at all for most of this
# project's life. The pixels were preserved (CellSnapshot) but the NODE was
# not, so Ctrl+Z after deleting a layer undid the stroke before it instead --
# one of the two causes of "undo did not undo the last thing I did". These
# commands hold the node object itself, detached, plus every cell it owned.

class StructureDelta(Command):
    """A layer node present or absent in the tree, with all its cells.

    Two-state, like every command here: `present` says what undo() should
    make true. A removal is captured BEFORE the node is detached, so it can
    put the node back exactly where it was; its opposite is an insertion
    whose undo detaches the node again, capturing the cells first.
    """

    __slots__ = ("label", "node", "parent_id", "index", "_cells", "present")

    def __init__(self, node, parent_id, index, cells, present, label="Layer"):
        self.label = label
        self.node = node
        self.parent_id = parent_id          # None means the document root
        self.index = index
        self._cells = cells                 # {(layer_id, frame_id): PixelDelta|None}
        self.present = present

    @staticmethod
    def _leaf_ids(node):
        leaves = [n for n in node.walk(include_self=False) if not n.is_group] \
            if node.is_group else [node]
        out = []
        for n in leaves:
            out.append(n.id)
            # A layer's mask travels with it: its cells are captured too, and
            # the mask object stays attached to the node while it is out.
            if getattr(n, "mask", None) is not None:
                out.append(n.mask.id)
            if getattr(n, "smart_source", None) is not None:
                out.append(n.smart_source.id)
        return out

    @staticmethod
    def removal(doc, node, label="Delete layer", level=1):
        """Capture a node that is ABOUT to be removed."""
        parent = node.parent
        parent_id = None if parent is None or parent is doc.root else parent.id
        index = parent.index_of(node) if parent is not None else 0
        cells = {}
        for layer_id in StructureDelta._leaf_ids(node):
            for frame in doc.frames:
                doc.ensure_warm(frame)
                key = (layer_id, frame.id)
                surface = doc.cells.get(key)
                if surface is None or surface.content_bbox is None:
                    continue
                box = surface.content_bbox
                cells[key] = PixelDelta(layer_id, frame.id, box,
                                        surface.read(box), label, level)
        return StructureDelta(node, parent_id, index, cells, True, label)

    @staticmethod
    def insertion(doc, node, label="Add layer"):
        """The opposite of a removal: undo() takes the node OUT."""
        parent = node.parent
        parent_id = None if parent is None or parent is doc.root else parent.id
        index = parent.index_of(node) if parent is not None else 0
        return StructureDelta(node, parent_id, index, {}, False, label)

    def undo(self, doc):
        if self.present:
            parent = doc.root if self.parent_id is None else doc.root.find(self.parent_id)
            if parent is None or not parent.is_group:
                parent = doc.root
            parent.add(self.node, self.index)
            for (layer_id, frame_id), delta in self._cells.items():
                layer = doc.layer(layer_id)
                frame = doc.frame_by_id(frame_id)
                if layer is None or frame is None:
                    continue
                surface = doc.cell(layer, frame)
                for name, arr in delta._decompress().items():
                    surface.plane(name)[delta.rect.slice()] = arr
                surface.content_bbox = delta.rect
                surface.refresh_derived(delta.rect)
                surface.dirty.add(delta.rect)
            return StructureDelta.insertion(doc, self.node, self.label), doc.bounds

        opposite = StructureDelta.removal(doc, self.node, self.label)
        doc.remove_layer(self.node)
        return opposite, doc.bounds

    def nbytes(self):
        return sum(d.nbytes() for d in self._cells.values())


class FrameDelta(Command):
    """A frame present or absent on the axis, with every cell in it."""

    __slots__ = ("label", "frame", "index", "_cells", "present")

    def __init__(self, frame, index, cells, present, label="Frame"):
        self.label = label
        self.frame = frame
        self.index = index
        self._cells = cells
        self.present = present

    @staticmethod
    def removal(doc, frame, label="Delete frame", level=1):
        doc.ensure_warm(frame)
        cells = {}
        for (layer_id, frame_id), surface in doc.cells.items():
            if frame_id != frame.id or surface.content_bbox is None:
                continue
            box = surface.content_bbox
            cells[(layer_id, frame_id)] = PixelDelta(layer_id, frame_id, box,
                                                     surface.read(box), label, level)
        return FrameDelta(frame, doc.frame_index(frame.id) or 0, cells, True, label)

    @staticmethod
    def insertion(doc, frame, label="Add frame"):
        return FrameDelta(frame, doc.frame_index(frame.id) or 0, {}, False, label)

    def undo(self, doc):
        if self.present:
            at = max(0, min(self.index, len(doc.frames)))
            doc.frames.insert(at, self.frame)
            self.frame.residency = "warm"
            for (layer_id, frame_id), delta in self._cells.items():
                layer = doc.layer(layer_id)
                if layer is None:
                    continue
                surface = doc.cell(layer, self.frame)
                for name, arr in delta._decompress().items():
                    surface.plane(name)[delta.rect.slice()] = arr
                surface.content_bbox = delta.rect
                surface.refresh_derived(delta.rect)
            doc.current = at
            return FrameDelta.insertion(doc, self.frame, self.label), doc.bounds

        opposite = FrameDelta.removal(doc, self.frame, self.label)
        doc.remove_frame(self.frame)
        return opposite, doc.bounds

    def nbytes(self):
        return sum(d.nbytes() for d in self._cells.values())


class GeometryDelta(Command):
    """The whole document's pixels and size. Backs resize, crop, flip, rotate.

    These operations touch every cell of every frame and may change the
    canvas dimensions, which nothing rectangular can express. So this is the
    one command that is allowed to be big: every authoritative plane of
    every cell, compressed, plus the selection mask -- a selection drawn on
    a 400-wide canvas means nothing on a 200-wide one. They are rare
    operations and the budget in history.ini bounds the damage.
    """

    __slots__ = ("label", "width", "height", "_cells", "_selection",
                 "_sel_blob", "_sel_shape", "_paths", "_marks", "_params", "_regions",
                 "_vmasks")

    def __init__(self, width, height, cells, selection=None, label="Transform",
                 paths=None):
        self.label = label
        # Paths and marks are transformed with the image, so they come back
        # with it.
        self._paths = paths
        self._marks = None
        self._params = None       # text layers' params (position and size move too)
        self._regions = None      # artboards' and frames' regions
        self._vmasks = None       # layers' vector masks (Path objects, never mutated)
        self.width = width
        self.height = height
        self._cells = cells             # {(layer_id, frame_id): (PixelDelta|None, bbox)}
        self._selection = selection
        self._sel_blob = None
        self._sel_shape = None
        if selection is not None and selection.mask is not None:
            self._sel_blob = zlib.compress(selection.mask.tobytes(), 1)
            self._sel_shape = selection.mask.shape

    @staticmethod
    def capture(doc, selection=None, label="Transform", level=1):
        cells = {}
        for frame in doc.frames:
            doc.ensure_warm(frame)
        for (layer_id, frame_id), surface in doc.cells.items():
            box = surface.content_bbox
            delta = None
            if box is not None:
                delta = PixelDelta(layer_id, frame_id, box, surface.read(box),
                                   label, level)
            cells[(layer_id, frame_id)] = (delta, box)
        paths = ([p.copy() for p in doc.paths], doc.active_path)
        out = GeometryDelta(doc.width, doc.height, cells, selection, label, paths)
        out._marks = {a: list(getattr(doc, a, [])) for a in MARKS}
        out._params = {l.id: dict(l.params) for l in doc.layers()
                       if getattr(l, "kind", "raster") in ("text", "smart")}
        out._regions = {g.id: dict(g.region) for g in doc.root.walk()
                        if g.is_group and getattr(g, "region", None)}
        out._vmasks = {l.id: l.vector_mask for l in doc.layers()
                       if getattr(l, "vector_mask", None) is not None}
        return out

    def undo(self, doc):
        opposite = GeometryDelta.capture(doc, self._selection, self.label)

        doc.width, doc.height = self.width, self.height
        doc.cells.clear()
        for (layer_id, frame_id), (delta, box) in self._cells.items():
            layer = doc.layer(layer_id)
            frame = doc.frame_by_id(frame_id)
            if layer is None or frame is None:
                continue
            surface = doc.cell(layer, frame)       # fresh, at the restored size
            if delta is not None:
                for name, arr in delta._decompress().items():
                    surface.plane(name)[delta.rect.slice()] = arr
                surface.content_bbox = box
                surface.refresh_derived(box)

        if self._selection is not None:
            self._selection.width, self._selection.height = self.width, self.height
            if self._sel_blob is None:
                self._selection.mask = None
            else:
                arr = np.frombuffer(zlib.decompress(self._sel_blob), dtype=np.uint8)
                self._selection.mask = arr.reshape(self._sel_shape).copy()
            self._selection._bbox = None
        if self._paths is not None:
            doc.paths = [p.copy() for p in self._paths[0]]
            doc.active_path = self._paths[1]
        if self._marks is not None:
            for a, v in self._marks.items():
                setattr(doc, a, list(v))
        for lid, params in (self._params or {}).items():
            node = doc.layer(lid)
            if node is not None:
                node.params = dict(params)
        for gid, region in (self._regions or {}).items():
            node = doc.layer(gid)
            if node is not None:
                node.region = dict(region)
        for lid, vm in (self._vmasks or {}).items():
            node = doc.layer(lid)
            if node is not None:
                node.vector_mask = vm
        return opposite, doc.bounds

    def nbytes(self):
        return sum(d.nbytes() for d, _ in self._cells.values() if d is not None) \
            + (len(self._sel_blob) if self._sel_blob else 0)



class SelectionDelta(Command):
    """The selection as it was. Selections are undoable, as in Paint.NET and
    GIMP: losing a painstaking lasso to a stray click with no way back is
    the kind of thing that makes people stop trusting a tool.

    The selection lives on the controller, not the document, so the command
    holds a reference to it -- the same arrangement GeometryDelta uses.
    """

    __slots__ = ("label", "selection", "_blob", "_shape")

    def __init__(self, selection, mask, label="Selection"):
        self.label = label
        self.selection = selection
        if mask is None:
            self._blob, self._shape = None, None
        else:
            self._blob = zlib.compress(np.ascontiguousarray(mask).tobytes(), 1)
            self._shape = mask.shape

    def undo(self, doc):
        sel = self.selection
        opposite = SelectionDelta(sel, sel.mask, self.label)
        if self._blob is None or self._shape != (sel.height, sel.width):
            sel.mask = None
        else:
            sel.mask = np.frombuffer(zlib.decompress(self._blob), np.uint8).reshape(self._shape).copy()
        sel._bbox = None
        return opposite, doc.bounds

    def nbytes(self):
        return len(self._blob) if self._blob else 0


class PathDelta(Command):
    """The document's vector paths as they were.

    Paths are small -- a few dozen floats per anchor -- so this simply keeps
    a copy of the whole list and the active index. Like every command here
    it stores one direction and builds the other when undone.
    """

    __slots__ = ("label", "_paths", "_active")

    def __init__(self, paths, active, label="Path"):
        self.label = label
        self._paths = [p.copy() for p in paths]
        self._active = active

    @staticmethod
    def capture(doc, label="Path"):
        return PathDelta(doc.paths, doc.active_path, label)

    def same_as(self, doc):
        """True if the document's paths still match this snapshot."""
        from .paths import encode_subpath
        if self._active != doc.active_path or len(self._paths) != len(doc.paths):
            return False
        for a, b in zip(self._paths, doc.paths):
            if a.name != b.name or len(a.subpaths) != len(b.subpaths):
                return False
            if any(encode_subpath(x) != encode_subpath(y)
                   for x, y in zip(a.subpaths, b.subpaths)):
                return False
        return True

    def undo(self, doc):
        opposite = PathDelta(doc.paths, doc.active_path, self.label)
        doc.paths = [p.copy() for p in self._paths]
        doc.active_path = self._active
        return opposite, doc.bounds

    def nbytes(self):
        return 64 + 56 * sum(len(s) for p in self._paths for s in p.subpaths)


MARKS = ("guides", "samplers", "counts", "notes", "slices", "planes")


class MarksDelta(Command):
    """Guides, samplers, counts, notes and slices as they were.

    All plain lists of tuples, so a shallow copy is a complete snapshot.
    Adding a guide or a count marker is undoable, as in Photoshop.
    """

    __slots__ = ("label", "_state")

    def __init__(self, doc, label="Marks", attrs=MARKS):
        self.label = label
        self._state = {a: list(getattr(doc, a, [])) for a in attrs}

    def same_as(self, doc):
        return all(list(getattr(doc, a, [])) == v for a, v in self._state.items())

    def undo(self, doc):
        opposite = MarksDelta(doc, self.label, tuple(self._state))
        for a, v in self._state.items():
            setattr(doc, a, list(v))
        return opposite, doc.bounds

    def nbytes(self):
        return 64 + 48 * sum(len(v) for v in self._state.values())


class MaskDelta(Command):
    """A layer's mask present or absent, with its cells.

    Holds the mask object and its Surfaces themselves -- detaching and
    re-attaching rather than copying -- so a mask's undo history (its own
    PixelDeltas, keyed by the mask's id) still lines up after the round trip.
    """

    __slots__ = ("label", "owner_id", "mask", "cells")

    def __init__(self, owner_id, mask, cells, label="Layer Mask"):
        self.label = label
        self.owner_id = owner_id
        self.mask = mask
        self.cells = cells

    @staticmethod
    def capture(doc, owner, label="Layer Mask"):
        mask = getattr(owner, "mask", None)
        cells = {}
        if mask is not None:
            cells = {k: v for k, v in doc.cells.items() if k[0] == mask.id}
        return MaskDelta(owner.id, mask, cells, label)

    def undo(self, doc):
        owner = doc.layer(self.owner_id)
        if owner is None:
            return None, None
        opposite = MaskDelta.capture(doc, owner, self.label)
        if owner.mask is not None:
            for key in [k for k in doc.cells if k[0] == owner.mask.id]:
                del doc.cells[key]
        owner.mask = self.mask
        doc.cells.update(self.cells)
        return opposite, doc.bounds

    def nbytes(self):
        return sum(c.nbytes() for c in self.cells.values())


class ChannelsDelta(Command):
    """The document's saved channels, with their cells, as they were."""

    __slots__ = ("label", "channels", "cells")

    def __init__(self, doc, label="Channel"):
        self.label = label
        self.channels = list(doc.channels)
        ids = {c.id for c in doc.channels}
        self.cells = {k: v for k, v in doc.cells.items() if k[0] in ids}

    def undo(self, doc):
        opposite = ChannelsDelta(doc, self.label)
        ids = {c.id for c in doc.channels}
        for key in [k for k in doc.cells if k[0] in ids]:
            del doc.cells[key]
        doc.channels = list(self.channels)
        doc.cells.update(self.cells)
        return opposite, doc.bounds

    def nbytes(self):
        return sum(c.nbytes() for c in self.cells.values())


class CompsDelta(Command):
    """Layer comps, as they were."""

    __slots__ = ("label", "comps")

    def __init__(self, doc, label="Layer Comp"):
        self.label = label
        self.comps = [dict(c) for c in doc.comps]

    def undo(self, doc):
        opposite = CompsDelta(doc, self.label)
        doc.comps = [dict(c) for c in self.comps]
        return opposite, doc.bounds

    def nbytes(self):
        return 64 * len(self.comps)


class SourceDelta(Command):
    """A smart object's source present or absent, with its cells, and the
    layer's kind and params -- so converting to a smart object (or
    rasterizing one) is one exact undo step."""

    __slots__ = ("label", "owner_id", "source", "cells", "kind", "params")

    def __init__(self, doc, owner, label="Smart Object"):
        self.label = label
        self.owner_id = owner.id
        self.source = getattr(owner, "smart_source", None)
        self.cells = {} if self.source is None else \
            {k: v for k, v in doc.cells.items() if k[0] == self.source.id}
        self.kind = owner.kind
        self.params = dict(owner.params)

    def undo(self, doc):
        owner = doc.layer(self.owner_id)
        if owner is None:
            return None, None
        opposite = SourceDelta(doc, owner, self.label)
        if owner.smart_source is not None:
            for key in [k for k in doc.cells if k[0] == owner.smart_source.id]:
                del doc.cells[key]
        owner.smart_source = self.source
        doc.cells.update(self.cells)
        owner.kind = self.kind
        owner.params = dict(self.params)
        return opposite, doc.bounds

    def nbytes(self):
        return sum(c.nbytes() for c in self.cells.values())
