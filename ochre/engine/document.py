# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Document: the layer tree, the frame axis, and the sparse cells between them.

This is the structural decision that cannot be retrofitted, so it is worth
stating plainly. A Document has a layer tree *and* a frame axis, and pixels
live in a sparse map keyed by (layer_id, frame_id). With a single frame the
model degenerates exactly to "one surface per layer", so the ordinary case
costs nothing -- but a multi-frame sprite stops having to pretend that its
frames are layers.

Sparseness is load-bearing rather than an optimisation. A cell that does not
exist simply is not there, which means a layer present in only some frames
costs nothing in the others, and per-frame-independent layer structures fall
out for free.

Canvas size, palette and selection are document-level; layers are
document-level and ordered; only pixels are per (layer, frame). That matches
how sprite and animation editing actually works: you select a region once and
apply it across frames, and a palette edit must reach every frame at once.
"""

from .frame import COLD, Frame, UNLOADED, WARM
from .geometry import Rect
from .layer import Layer, LayerGroup
from .palette import Palette
from .surface import Surface


class Document:
    """A canvas, a layer tree, a frame axis, and the cells between them."""

    def __init__(self, width, height, palette=None, settings=None):
        if width <= 0 or height <= 0:
            raise ValueError("document must be at least 1x1, got %dx%d"
                             % (width, height))
        self.width = int(width)
        self.height = int(height)
        self.root = LayerGroup(name="Document")
        self.frames = [Frame(name="Frame 1")]
        self.current = 0
        self.palette = palette
        self.cells = {}                 # (layer_id, frame_id) -> Surface
        self.meta = {}                  # str -> str
        self.axis_layout = "strip"      # "strip" | "grid" -- a hint, nothing more
        self.axis_columns = 0           # grid width when layout == "grid"
        self._settings = settings
        self._max_dirty = self._setting("Display", "MaxDirtyRects", 8)

    def _setting(self, section, key, default):
        if self._settings is None:
            return default
        value = self._settings.get(section, key)
        return default if value is None else value

    # ---- geometry --------------------------------------------------------

    @property
    def bounds(self):
        return Rect(0, 0, self.width, self.height)

    def __repr__(self):
        return ("Document(%dx%d, layers=%d, frames=%d, cells=%d)"
                % (self.width, self.height, len(self.root.leaves()),
                   len(self.frames), len(self.cells)))

    # ---- layers ----------------------------------------------------------

    def add_layer(self, name="", parent=None, index=None, planes=("rgba",),
                  authoritative=None):
        layer = Layer(name=name, planes=planes, authoritative=authoritative)
        (self.root if parent is None else parent).add(layer, index)
        return layer

    def add_group(self, name="", parent=None, index=None):
        group = LayerGroup(name=name)
        (self.root if parent is None else parent).add(group, index)
        return group

    def layer(self, layer_id):
        return self.root.find(layer_id)

    def layers(self):
        """Leaf layers, bottom-first."""
        return self.root.leaves()

    def remove_layer(self, node):
        """Detach a node and drop every cell it owns, across all frames."""
        parent = node.parent
        if parent is None:
            return False
        doomed = [n.id for n in ([node] if not node.is_group
                                 else list(node.walk(include_self=True)))]
        for key in [k for k in self.cells if k[0] in doomed]:
            del self.cells[key]
        return parent.remove(node)

    # ---- frames ----------------------------------------------------------

    @property
    def frame(self):
        return self.frames[self.current]

    def frame_by_id(self, frame_id):
        for f in self.frames:
            if f.id == frame_id:
                return f
        return None

    def frame_index(self, frame_id):
        for i, f in enumerate(self.frames):
            if f.id == frame_id:
                return i
        return None

    def add_frame(self, index=None, name="", copy_from=None):
        """Insert a frame. copy_from duplicates that frame's cells."""
        frame = Frame(name=name or "Frame %d" % (len(self.frames) + 1),
                      max_dirty_rects=self._max_dirty)
        at = len(self.frames) if index is None else max(0, min(index, len(self.frames)))
        self.frames.insert(at, frame)
        if copy_from is not None:
            src_id = copy_from.id if isinstance(copy_from, Frame) else copy_from
            for (layer_id, frame_id), surface in list(self.cells.items()):
                if frame_id == src_id:
                    self.cells[(layer_id, frame.id)] = surface.copy()
        return frame

    def remove_frame(self, frame):
        """Drop a frame and its cells. The last frame cannot be removed."""
        if len(self.frames) <= 1:
            return False
        target = frame if isinstance(frame, Frame) else self.frame_by_id(frame)
        if target is None:
            return False
        for key in [k for k in self.cells if k[1] == target.id]:
            del self.cells[key]
        self.frames.remove(target)
        self.current = min(self.current, len(self.frames) - 1)
        return True

    def set_current_frame(self, index):
        """Switch frames, thawing the target and cooling by LRU."""
        index = max(0, min(int(index), len(self.frames) - 1))
        if index == self.current:
            return self.frame
        self.current = index
        self.ensure_warm(self.frame)
        self.enforce_residency()
        return self.frame

    # ---- cells -----------------------------------------------------------

    def cell(self, layer, frame=None, create=True):
        """The Surface for (layer, frame), created on demand.

        Creation is where a layer's plane declaration becomes concrete, and
        where an index-locked layer is wired to the document palette -- which
        is why the palette has to be document-level rather than per-layer.
        """
        layer_id = layer.id if hasattr(layer, "id") else layer
        node = self.layer(layer_id)
        frame_obj = self.frame if frame is None else (
            frame if isinstance(frame, Frame) else self.frame_by_id(frame))
        if node is None or frame_obj is None:
            return None
        key = (layer_id, frame_obj.id)
        surface = self.cells.get(key)
        if surface is None and create:
            surface = Surface(self.width, self.height,
                              planes=node.planes,
                              authoritative=node.authoritative,
                              palette=self.palette if node.index_locked else None,
                              max_dirty_rects=self._max_dirty)
            self.cells[key] = surface
        return surface

    def has_cell(self, layer, frame=None):
        layer_id = layer.id if hasattr(layer, "id") else layer
        frame_obj = self.frame if frame is None else (
            frame if isinstance(frame, Frame) else self.frame_by_id(frame))
        return frame_obj is not None and (layer_id, frame_obj.id) in self.cells

    def cells_for_frame(self, frame=None):
        frame_obj = self.frame if frame is None else frame
        return {k: v for k, v in self.cells.items() if k[1] == frame_obj.id}

    def cells_for_layer(self, layer):
        layer_id = layer.id if hasattr(layer, "id") else layer
        return {k: v for k, v in self.cells.items() if k[0] == layer_id}

    # ---- palette ---------------------------------------------------------

    def bind_palette(self, palette):
        """Attach a palette document-wide. All frames share it."""
        self.palette = palette
        for (layer_id, _), surface in self.cells.items():
            node = self.layer(layer_id)
            if node is not None and node.index_locked:
                surface.palette = palette
                surface.refresh_derived()
        return palette

    def set_palette_entry(self, index, rgba):
        """Recolour an entry and update every cell that uses it.

        Costs no history beyond the entry itself: not one authoritative byte
        changes, because the index planes are untouched. Undo of a palette
        edit is therefore free and exact, and repeated edits cannot drift.
        """
        if self.palette is None:
            return []
        self.palette.set_entry(index, rgba)
        touched = []
        for key, surface in self.cells.items():
            r = surface.recolor_index(index)
            if r is not None:
                touched.append((key, r))
        return touched

    def lock_layer_to_index(self, layer):
        """Convert a layer to index-locked across every frame."""
        if self.palette is None:
            raise ValueError("bind a palette before locking a layer to it")
        layer.lock_to_index()
        for key, surface in list(self.cells_for_layer(layer).items()):
            replacement = Surface(self.width, self.height,
                                  planes=layer.planes,
                                  authoritative=layer.authoritative,
                                  palette=self.palette,
                                  max_dirty_rects=self._max_dirty)
            if surface.planes.get("rgba") is not None:
                replacement.plane("index")[...] = self.palette.snap(surface.pixels)
                replacement.refresh_derived()
                replacement.content_bbox = surface.content_bbox
            self.cells[key] = replacement
        return layer

    # ---- residency -------------------------------------------------------

    def ensure_warm(self, frame):
        if frame.residency == COLD:
            frame.thaw(self.cells)
        elif frame.residency == UNLOADED:
            frame.residency = WARM
        return frame

    def enforce_residency(self, max_warm=None, max_bytes=None):
        """Cool least-recently-used frames until within budget.

        Recency is distance from the current frame, which is the right proxy
        for a timeline: scrubbing keeps neighbours warm and lets distant
        frames cool.
        """
        max_warm = self._setting("Frames", "MaxWarmFrames", 8) if max_warm is None else max_warm
        max_bytes = self._setting("Frames", "MaxWarmBytes", 1 << 30) if max_bytes is None else max_bytes

        order = sorted(range(len(self.frames)), key=lambda i: abs(i - self.current))
        keep = set(order[:max(1, max_warm)])

        cooled = 0
        for i, frame in enumerate(self.frames):
            if i not in keep and frame.residency == WARM:
                cooled += frame.freeze(self.cells)

        if self.warm_bytes() > max_bytes:
            for i in reversed(order):
                if i == self.current:
                    continue
                frame = self.frames[i]
                if frame.residency == WARM:
                    cooled += frame.freeze(self.cells)
                if self.warm_bytes() <= max_bytes:
                    break
        return cooled

    def warm_bytes(self):
        return sum(s.nbytes() for s in self.cells.values())

    def nbytes(self):
        return self.warm_bytes() + sum(f.cold_bytes() for f in self.frames)

    # ---- dirty -----------------------------------------------------------

    def invalidate(self, rect, frame=None):
        (self.frame if frame is None else frame).dirty.add(rect)

    def take_dirty(self, frame=None):
        """Drain the frame's dirty rects, merged with its cells' own.

        Cells accumulate dirt as tools write; the frame is where it is
        collected. Draining both here keeps a single source of truth for
        "what needs recompositing" without the cells needing to know that a
        frame exists.
        """
        frame_obj = self.frame if frame is None else frame
        region = frame_obj.dirty
        for (_, frame_id), surface in self.cells.items():
            if frame_id == frame_obj.id:
                for r in surface.dirty.take():
                    region.add(r)
        return region.take()
