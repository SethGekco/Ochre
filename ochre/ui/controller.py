# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Qt-free boundary between the engine and the widgets.

Every mutation the UI performs goes through here, and this module imports no
Qt. That keeps the whole application logic testable headlessly, makes a
scripted batch mode fall out for free, and means the widget layer can be
replaced without touching a line of editing logic.

The contract with the UI is deliberately one-directional. Mutators return
dirty rectangles and nothing else; the only read-back path is `flush()`
followed by reading `display`. That single direction is what keeps state
from drifting between the two layers.

The engine never composites on its own schedule. Tools mark regions dirty;
compositing happens when the UI calls flush(), at display-frame rate. A
200 Hz tablet therefore causes 200 dabs but not 200 composites.
"""

import os

import numpy as np

from ..engine import accel
from ..engine.compositor import Compositor
from ..engine.document import Document
from ..engine.fileio import export_flat, import_flat, is_ochre, load, save
from ..engine.geometry import Rect
from ..engine.history import HistoryStack
from ..engine.ini import IniDB
from ..engine.selection import Selection
from ..engine.settings import Settings
from ..engine.tools import ToolContext, ToolEvent, ToolRegistry


class Controller:
    """One open document, plus everything the UI needs to drive it."""

    def __init__(self, settings=None, data_dir=None, bus=None,
                 addon_dirs=None, state_dir=None, trust_callback=None):
        if settings is None:
            settings = Settings(data_dir)
        self.settings = settings
        self.db = settings.db if hasattr(settings, "db") else IniDB(data_dir)
        self.tools = ToolRegistry(self.db)
        # Addons are discovered now but nothing of theirs runs until asked;
        # see ochre/addons/loader.py for why that split matters.
        from ..addons import AddonManager
        self.addons = AddonManager(addon_dirs or [], settings, self.db,
                                   state_dir, self, trust_callback)
        self.addons.discover()
        self.bus = bus
        self.doc = None
        self.compositor = None
        self.history = None
        self.selection = None
        self.display = None
        self.clip = None
        from .session import Session
        self.state_dir = state_dir
        self.session = Session(state_dir)
        self._saved_marker = None
        self._tool_overrides = {}
        # Document pixels that count as "on" a point; the canvas sets it to a
        # few screen pixels as the zoom changes.
        self.pick_radius = 4.0
        self.quick_mask = None
        self.path = None
        self.dirty_since_save = False
        self.active_layer = None
        self.tool = None
        self.tool_name = None
        self.primary = (0, 0, 0, 255)
        self.secondary = (255, 255, 255, 255)
        self.primary_index = None
        self.secondary_index = None
        self._pending = []

    # ---- events ----------------------------------------------------------

    def emit(self, topic, **payload):
        if self.bus is not None:
            self.bus.emit(topic, **payload)

    # ---- document lifecycle ---------------------------------------------

    def new_document(self, width=None, height=None, background=(255, 255, 255, 255)):
        # The size you were last working at, when there is one. A fresh launch
        # opening at whatever the INI says is a small daily annoyance for
        # anyone who works at one size -- sprites especially.
        last = self.session.last_size() if self.session else None
        if width is None and height is None and last is not None:
            width, height = last
        width = width or self.settings.get("Canvas", "DefaultWidth") or 800
        height = height or self.settings.get("Canvas", "DefaultHeight") or 600
        doc = Document(width, height, settings=self.settings)
        layer = doc.add_layer("Background")
        if background is not None:
            doc.cell(layer).fill(doc.bounds, background)
        self._adopt(doc, path=None)
        return doc

    def _addon_format_for(self, path, writing=False):
        """An addon provider claiming this file, or None.

        Extension first, then content sniffing -- extensions lie, and a
        format addon is the thing that knows how to tell.
        """
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        candidates = [p for p in self.addons.registry.formats_for(ext)
                      if (p.can_write if writing else p.can_read)]
        if len(candidates) <= 1:
            return candidates[0] if candidates else None

        # Writing cannot sniff -- there is nothing at the target path yet, or
        # worse, something unrelated. When two providers share an extension
        # the document itself is the authority on which one it came from.
        # Without this, saving to an ambiguous extension silently writes the
        # wrong FORMAT, which is the most destructive outcome available.
        if writing and self.doc is not None:
            came_from = self.doc.meta.get("format")
            for provider in candidates:
                if provider.name == came_from:
                    return provider

        try:
            with open(path, "rb") as f:
                head = f.read(4096)
        except OSError:
            return candidates[0]
        for provider in candidates:
            owner = self.addons.registry.owner("formats", provider.name)
            with self.addons.guard(owner, "sniff") as guard:
                if provider.sniff(head, path):
                    return provider
            if guard.failed:
                continue
        return candidates[0]

    def open_path(self, path):
        provider = self._addon_format_for(path)
        if provider is not None:
            owner = self.addons.registry.owner("formats", provider.name)
            with self.addons.guard(owner, "load") as guard:
                doc = provider.load(path, self.addons_host(owner))
                self._adopt(doc, path=path)
                return self.doc
            if guard.failed:
                raise ValueError("addon %r failed to open %s" % (owner, path))
        if is_ochre(path):
            doc, mask = load(path, settings=self.settings)
            self._adopt(doc, path=path)
            if mask is not None:
                self.selection.mask = mask
                self.selection._bbox = None
        else:
            try:
                doc = import_flat(path, settings=self.settings)
            except Exception as exc:
                raise self._unopenable(path, exc) from exc
            self._adopt(doc, path=None)   # imported, not "the same file"
        return self.doc

    def _unopenable(self, path, exc):
        """Explain a failed open, blaming the right thing.

        Pillow's "cannot identify image file" is technically true and
        actively misleading when the format IS supported and its addon
        simply is not loaded -- which happens routinely, because trust is
        keyed to a content hash and every addon update revokes it. Saying
        so turns a dead end into an instruction.
        """
        idle = sorted(a.id for a in self.addons.addons.values()
                      if "formats" in (a.manifest.provides or ())
                      and a.state != "loaded")
        if idle:
            return ValueError(
                "cannot open %s: no loaded format handles it, but %s %s "
                "format support and %s not loaded. Enable %s under "
                "Addons, or re-approve %s -- updating an addon revokes "
                "trust, by design."
                % (os.path.basename(path), " and ".join(repr(i) for i in idle),
                   "provide" if len(idle) > 1 else "provides",
                   "are" if len(idle) > 1 else "is",
                   "them" if len(idle) > 1 else "it",
                   "them" if len(idle) > 1 else "it"))
        return ValueError("cannot open %s: %s" % (os.path.basename(path), exc))

    def save_path(self, path=None):
        target = path or self.path
        if target is None:
            raise ValueError("no path to save to")
        provider = self._addon_format_for(target, writing=True)
        if provider is not None:
            owner = self.addons.registry.owner("formats", provider.name)
            with self.addons.guard(owner, "save") as guard:
                provider.save(self.doc, target, self.addons_host(owner), {})
                self.path = target
                self.dirty_since_save = False
                self._saved_marker = self.history.marker()
                self.emit("document.saved", path=target)
                return target
            if guard.failed:
                raise ValueError("addon %r failed to save %s" % (owner, target))
        if target.lower().endswith(".ochre"):
            save(self.doc, target, selection=self.selection)
            self.path = target
        else:
            export_flat(self.doc, target)
        self.dirty_since_save = False
        self._saved_marker = self.history.marker()
        self.emit("document.saved", path=target)
        return target

    def _adopt(self, doc, path):
        self.doc = doc
        self.history = HistoryStack(self.settings)
        self.selection = Selection(doc.width, doc.height)
        self.compositor = Compositor(doc)
        self.blends = self.compositor.blends
        self.display = np.zeros((doc.height, doc.width, 4), dtype=np.uint8)
        self.path = path
        self.dirty_since_save = False
        self._saved_marker = self.history.marker()
        if self.session is not None:
            self.session.remember_size(doc.width, doc.height)
        leaves = doc.layers()
        self.active_layer = leaves[-1] if leaves else None
        self._pending = [doc.bounds]
        self.set_history_source(auto=True)
        if self.tool_name:
            self.set_tool(self.tool_name)
        self.emit("document.opened", document=doc, path=path)

    def is_modified(self):
        """True only when there is real unsaved work.

        Measured against history rather than a dirty flag: an untouched
        untitled canvas is not modified, and undoing back to the point of the
        last save is not modified either. That is what lets the editor never
        nag about a document nobody touched -- while still not silently
        throwing away a document somebody did. An open tool session counts:
        typed text that has not been committed yet is work.
        """
        if self.doc is None:
            return False
        if self.tool is not None and self.tool.active:
            return True
        current = self.history.marker()
        saved = self._saved_marker
        if saved is None:
            return current[0] is not None or current[2] != 0
        return current[1] is not saved[1] or current[2] != saved[2]

    @property
    def title(self):
        name = os.path.basename(self.path) if self.path else "Untitled"
        return ("*" if self.is_modified() else "") + name

    def addons_host(self, addon_id):
        """A Host bound to one addon, for calls made after registration."""
        from ..addons.api import Host
        addon = self.addons.addons.get(addon_id)
        return Host(self.addons.registry, addon, self.settings, self.db, self)

    def load_addons(self):
        """Load discovered addons and fold what they registered into the app."""
        self.addons.load_all()
        for name, cls in self.addons.registry.all("tools").items():
            self.tools._classes[name] = cls
        self.emit("addons.changed")
        return self.addons.loaded()

    def addon_extensions(self):
        out = []
        for provider in self.addons.registry.all("formats").values():
            out.extend(getattr(provider, "extensions", ()))
        return sorted(set(out))

    # ---- rendering -------------------------------------------------------

    def invalidate(self, rect=None):
        self._pending.append(rect if rect is not None else self.doc.bounds)

    def flush(self):
        """Composite everything pending and return the rects the UI must blit.

        `display` is a stable, never-reallocated array so the UI can wrap it
        once in a QImage and blit from it forever. Resizing replaces its
        contents, never its identity.
        """
        if self.doc is None:
            return []
        for rect in self.doc.take_dirty():
            self._pending.append(rect)
        if not self._pending:
            return []

        pending, self._pending = self._pending, []
        merged = []
        for rect in pending:
            clipped = rect.clipped_to(self.doc.width, self.doc.height)
            if clipped is not None:
                merged.append(clipped)
        if not merged:
            return []

        for rect in merged:
            if self.active_layer is not None and self.compositor.cache_valid:
                self.compositor.composite_cached(self.display, rect, self.active_layer)
            else:
                self.compositor.composite_into(self.display, rect)
        return merged

    def composite_region(self, rect):
        return self.compositor.composite(rect)

    # ---- layers ----------------------------------------------------------

    def layer_rows(self):
        """(depth, node) top-most first, which is how a panel lists them."""
        return self.doc.root.flat()

    def add_layer(self, name=None, parent=None, index=None, planes=None,
                  authoritative=None):
        from ..engine.commands import StructureDelta
        kw = {}
        if planes is not None:
            kw["planes"] = tuple(planes)
            kw["authoritative"] = tuple(authoritative or (planes[0],))
        layer = self.doc.add_layer(name or "Layer %d" % (len(self.doc.layers()) + 1),
                                   parent=parent, index=index, **kw)
        self.doc.cell(layer)
        # Adding a layer is an edit. For most of this project's life it was
        # not in history, so Ctrl+Z after "Add Layer" undid the stroke
        # before it -- the second cause of "undo skipped what I just did".
        self.history.push(StructureDelta.insertion(self.doc, layer, "Add layer"),
                          "Add layer")
        self.set_active_layer(layer)
        self._after_structure_change()
        return layer

    def add_group(self, name=None):
        from ..engine.commands import StructureDelta
        group = self.doc.add_group(name or "Group")
        self.history.push(StructureDelta.insertion(self.doc, group, "Add group"),
                          "Add group")
        self._after_structure_change()
        return group

    def remove_layer(self, layer=None):
        from ..engine.commands import StructureDelta
        target = layer or self.active_layer
        if target is None or len(self.doc.layers()) <= 1:
            return False
        # Captured BEFORE removal: the node, its slot, and every cell.
        cmd = StructureDelta.removal(self.doc, target, "Delete layer")
        self.doc.remove_layer(target)
        self.history.push(cmd, "Delete layer")
        leaves = self.doc.layers()
        self.active_layer = leaves[-1] if leaves else None
        self._after_structure_change()
        return True

    def layer_at(self, x, y):
        """The topmost visible layer with paint at (x, y), or None."""
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.doc.width and 0 <= iy < self.doc.height):
            return None
        for layer in reversed(self.doc.layers()):
            if not layer.effectively_visible():
                continue
            cell = self.doc.cells.get((layer.id, self.doc.frame.id))
            if cell is None or cell.content_bbox is None:
                continue
            if cell.plane("rgba")[iy, ix, 3] > 0:
                return layer
        return None

    def align_layer(self, how, layer=None):
        """Align the layer's content to the selection, or to the canvas."""
        from ..engine import transform
        target = layer or self.active_layer
        if target is None or target.is_group:
            return False
        cell = self.doc.cells.get((target.id, self.doc.frame.id))
        if cell is None or cell.content_bbox is None:
            return False
        ref = self.doc.bounds if self.selection.selects_all() else self.selection.bbox
        dx, dy = transform.align_offset(cell.content_bbox, ref, how)
        if dx == 0 and dy == 0:
            return False
        return self._layer_pixels_op("Align layer",
                                     lambda l: transform.offset_layer(self.doc, l, dx, dy), target)

    def offset_layer(self, dx, dy, wrap=True, layer=None):
        from ..engine import transform
        return self._layer_pixels_op("Offset layer",
                                     lambda l: transform.offset_layer(self.doc, l, dx, dy, wrap),
                                     layer)

    def rotate_layer(self, degrees, layer=None):
        """Rotate a layer by any angle about the centre of its content."""
        import math as _m
        from ..engine import transform, warp
        target = layer or self.active_layer
        cell = None if target is None else self.doc.cells.get((target.id, self.doc.frame.id))
        if cell is None or cell.content_bbox is None:
            return False
        c = cell.content_bbox
        hm = warp.rotate_about(_m.radians(float(degrees)), c.x + c.w / 2.0, c.y + c.h / 2.0)
        return self._layer_pixels_op("Rotate layer",
                                     lambda l: transform.transform_layer(self.doc, l, hm), target)

    def scale_layer(self, factor_x, factor_y=None, layer=None):
        from ..engine import transform, warp
        target = layer or self.active_layer
        cell = None if target is None else self.doc.cells.get((target.id, self.doc.frame.id))
        if cell is None or cell.content_bbox is None:
            return False
        c = cell.content_bbox
        fy = factor_x if factor_y is None else factor_y
        hm = warp.scale_about(float(factor_x), float(fy), c.x + c.w / 2.0, c.y + c.h / 2.0)
        return self._layer_pixels_op("Scale layer",
                                     lambda l: transform.transform_layer(self.doc, l, hm), target)

    def rotate_image_by(self, degrees):
        from ..engine import transform
        return self._geometry("Rotate image",
                              lambda: transform.rotate_image_by(self.doc, degrees))

    def move_layer_by(self, delta, layer=None):
        """Nudge a layer up (+1, towards the top) or down within its group."""
        target = layer or self.active_layer
        if target is None or target.parent is None:
            return False
        parent = target.parent
        i = parent.index_of(target)
        j = max(0, min(i + int(delta), len(parent.children) - 1))
        if j == i:
            return False
        return self.move_layer(target, parent, j)

    def set_active_layer(self, layer):
        self.active_layer = layer
        self.compositor.set_active(getattr(layer, "id", layer))
        self.emit("layer.active", layer=layer)

    def set_layer_property(self, layer, attr, value):
        from ..engine.commands import PropertyDelta
        cmd = PropertyDelta(layer.id, attr, getattr(layer, attr))
        setattr(layer, attr, value)
        self.history.push(cmd, "Change %s" % attr)
        self._after_structure_change()
        return value

    def move_layer(self, layer, parent, index):
        # `parent or self.doc.root` would be wrong here: an empty LayerGroup
        # is falsy because it defines __len__, so dropping into an empty
        # group silently moved the layer to the root instead.
        if parent is None:
            parent = self.doc.root
        here = layer.parent
        if here is parent and here.children.index(layer) == index:
            return False

        from ..engine.commands import ReorderDelta
        undo = ReorderDelta(
            layer.id,
            None if here is None or here is self.doc.root else here.id,
            here.children.index(layer) if here is not None else 0,
            "Reorder layer")
        parent.add(layer, index)
        self.history.push(undo, "Reorder layer")
        self._after_structure_change()
        return True

    def duplicate_layer(self, layer=None):
        """Copy a layer and everything it holds across every frame."""
        source = layer or self.active_layer
        if source is None or source.is_group:
            return None
        # The copy must have the SAME planes as the source. An index-locked
        # layer duplicated into a plain RGBA one would lose the authoritative
        # indices -- the single thing the whole indexed model exists to keep.
        copy = self.doc.add_layer(
            "%s copy" % (source.name or source.id), parent=source.parent,
            planes=tuple(source.planes) if hasattr(source, "planes") else ("rgba",),
            authoritative=tuple(getattr(source, "authoritative", ("rgba",))))
        copy.opacity, copy.blend = source.opacity, source.blend
        copy.visible = source.visible
        for (layer_id, frame_id), cell in list(self.doc.cells.items()):
            if layer_id != source.id:
                continue
            frame = self.doc.frame_by_id(frame_id)
            if frame is None:
                continue
            target = self.doc.cell(copy, frame)
            for name in cell.authoritative:
                plane = cell.planes.get(name)
                if plane is not None:
                    target.plane(name)[...] = plane
            target.content_bbox = cell.content_bbox
            target.refresh_derived()
        from ..engine.commands import StructureDelta
        self.history.push(StructureDelta.insertion(self.doc, copy, "Duplicate layer"),
                          "Duplicate layer")
        self.set_active_layer(copy)
        self._after_structure_change()
        return copy

    # ---- merging ---------------------------------------------------------

    def merge_down(self, layer=None):
        """Composite the layer onto the one beneath it; one undo entry."""
        from ..engine import transform
        from ..engine.commands import PixelDelta, StructureDelta
        upper = layer or self.active_layer
        if upper is None:
            return False
        lower = transform.layer_below(self.doc, upper)
        if lower is None:
            return False
        rects = transform.merge_rects(self.doc, upper, lower)
        self.history.begin_group()
        for frame_id, rect in rects.items():
            self.doc.cell(lower, self.doc.frame_by_id(frame_id))
            delta = PixelDelta.capture(self.doc, lower.id, frame_id, rect, "Merge down")
            if delta is not None:
                self.history.push(delta, "Merge down", frame_id=frame_id)
        self.history.push(StructureDelta.removal(self.doc, upper, "Merge down"), "Merge down")
        self.history.end_group("Merge down")
        transform.merge_down(self.doc, upper, lower, self.blends)
        self.set_active_layer(lower)
        self._after_structure_change()
        return True

    def flatten(self):
        """Every visible layer into one. Indexed stays indexed when it can."""
        from ..engine import transform
        from ..engine.commands import StructureDelta
        leaves = self.doc.layers()
        if len(self.doc.root.children) <= 1 and len(leaves) <= 1:
            return False
        keep_indexed = (self.doc.palette is not None
                        and leaves and all(l.index_locked for l in leaves))

        self.history.begin_group()
        # Removals are pushed TOP-DOWN so that undo, which replays a group in
        # reverse, re-inserts bottom-up. Each delta stores an absolute slot,
        # and inserting the top layer first into an emptied group clamps its
        # index and ends with siblings swapped.
        for node in reversed(list(self.doc.root.children)):
            self.history.push(StructureDelta.removal(self.doc, node, "Flatten"), "Flatten")
        # Composite each frame BEFORE the stack is torn down.
        flats = {f.id: self.compositor.composite(self.doc.bounds, f) for f in self.doc.frames}
        for node in list(self.doc.root.children):
            self.doc.remove_layer(node)
        if keep_indexed:
            merged = self.doc.add_layer("Background", planes=("index", "rgba"),
                                        authoritative=("index",))
        else:
            merged = self.doc.add_layer("Background")
        for frame in self.doc.frames:
            cell = self.doc.cell(merged, frame)
            if keep_indexed:
                transform.composite_onto(cell, flats[frame.id], self.doc.bounds,
                                         self.blends.opcode("Normal"), 255)
            else:
                cell.plane("rgba")[...] = flats[frame.id]
                cell._after_write(self.doc.bounds, "rgba")
        self.history.push(StructureDelta.insertion(self.doc, merged, "Flatten"), "Flatten")
        self.history.end_group("Flatten")
        self.set_active_layer(merged)
        self._after_structure_change()
        return True

    # ---- layer geometry --------------------------------------------------

    def _layer_pixels_op(self, label, fn, layer=None):
        from ..engine.commands import PixelDelta
        target = layer or self.active_layer
        if target is None or target.is_group:
            return False
        self.history.begin_group()
        for frame in self.doc.frames:
            cell = self.doc.cells.get((target.id, frame.id))
            if cell is None or cell.content_bbox is None:
                continue
            self.doc.ensure_warm(frame)
            delta = PixelDelta.capture(self.doc, target.id, frame.id, self.doc.bounds, label)
            if delta is not None:
                self.history.push(delta, label, frame_id=frame.id)
        self.history.end_group(label)
        fn(target)
        self._after_structure_change()
        return True

    def flip_layer(self, horizontal=True, layer=None):
        from ..engine import transform
        return self._layer_pixels_op(
            "Flip layer", lambda l: transform.flip_layer(self.doc, l, horizontal), layer)

    def rotate_layer_180(self, layer=None):
        from ..engine import transform
        return self._layer_pixels_op(
            "Rotate layer", lambda l: transform.rotate_layer_180(self.doc, l), layer)

    # ---- document geometry -----------------------------------------------

    def _geometry(self, label, fn):
        """Run a whole-document geometry change as one undo entry."""
        from ..engine.commands import GeometryDelta
        if self.tool is not None and self.tool.active:
            self.cancel_stroke()
        cmd = GeometryDelta.capture(self.doc, self.selection, label)
        result = fn()
        if result is None:
            return False
        self.history.push(cmd, label)
        self._after_geometry_change()
        return True

    def resize_image(self, width, height, method="bilinear"):
        from ..engine import transform
        if (int(width), int(height)) == (self.doc.width, self.doc.height):
            return False
        return self._geometry("Resize image",
                              lambda: transform.resize_image(self.doc, width, height, method))

    def resize_canvas(self, width, height, anchor="center"):
        from ..engine import transform
        if (int(width), int(height)) == (self.doc.width, self.doc.height):
            return False
        return self._geometry("Canvas size",
                              lambda: transform.resize_canvas(self.doc, width, height, anchor))

    def crop_to_selection(self):
        from ..engine import transform
        if self.selection.selects_all():
            return False
        rect = self.selection.bbox
        if rect.is_empty:
            return False
        done = self._geometry("Crop", lambda: transform.crop(self.doc, rect))
        if done:
            self.selection.select_all()
            self.emit("selection.changed")
        return done

    def flip_image(self, horizontal=True):
        from ..engine import transform
        label = "Flip horizontal" if horizontal else "Flip vertical"
        return self._geometry(label, lambda: transform.flip_document(self.doc, horizontal))

    def rotate_image(self, quarter_turns):
        from ..engine import transform
        return self._geometry("Rotate image",
                              lambda: transform.rotate_document(self.doc, quarter_turns))

    def _after_geometry_change(self):
        """The canvas changed size: everything sized to it is rebuilt.

        The display buffer's stability contract is "replace contents, never
        identity" -- which holds for edits, and cannot for a canvas that is
        now a different shape. So this is the one sanctioned reallocation,
        and the signal tells the canvas to re-alias.
        """
        if self.display is None or self.display.shape[:2] != (self.doc.height, self.doc.width):
            self.display = np.zeros((self.doc.height, self.doc.width, 4), dtype=np.uint8)
        if (self.selection.width, self.selection.height) != (self.doc.width, self.doc.height):
            self.selection.width, self.selection.height = self.doc.width, self.doc.height
            self.selection.mask = None
            self.selection._bbox = None
        self.compositor.invalidate()
        self._pending = [self.doc.bounds]
        self.dirty_since_save = True
        if self.session is not None:
            self.session.remember_size(self.doc.width, self.doc.height)
        self.emit("document.resized", width=self.doc.width, height=self.doc.height)
        self.emit("layers.changed")
        self.emit("selection.changed")

    # ---- clipboard -------------------------------------------------------

    def copy(self, merged=False):
        from ..engine import clipboard
        if self.active_layer is None and not merged:
            return None
        self.clip = clipboard.copy_selection(
            self.doc, self.active_layer, self.doc.frame, self.selection,
            merged=merged, compositor=self.compositor)
        if self.clip is not None:
            self.emit("clipboard.changed", clip=self.clip)
        return self.clip

    def cut(self):
        clip = self.copy()
        if clip is not None:
            self.erase_selection()
        return clip

    def _selection_window(self):
        rect = self.selection.bbox.clipped_to(self.doc.width, self.doc.height)
        if rect is None or rect.is_empty:
            return None, None
        return rect, self.selection.mask_for(rect)

    def erase_selection(self):
        """Delete key: clear the selected pixels on the active layer."""
        from ..engine.commands import PixelDelta
        if self.active_layer is None or self.active_layer.is_group:
            return None
        rect, mask = self._selection_window()
        if rect is None:
            return None
        cell = self.doc.cell(self.active_layer)
        delta = PixelDelta.capture(self.doc, self.active_layer.id, self.doc.frame.id,
                                   rect, "Erase selection")
        cell.erase_masked(rect, mask)
        if delta is not None:
            self.history.push(delta, "Erase selection", frame_id=self.doc.frame.id)
        self.invalidate(rect)
        self.dirty_since_save = True
        self.emit("history.changed")
        return rect

    def fill_selection(self, colour=None):
        """Backspace: fill the selection with the primary colour."""
        from ..engine.commands import PixelDelta
        if self.active_layer is None or self.active_layer.is_group:
            return None
        rect, mask = self._selection_window()
        if rect is None:
            return None
        cell = self.doc.cell(self.active_layer)
        delta = PixelDelta.capture(self.doc, self.active_layer.id, self.doc.frame.id,
                                   rect, "Fill selection")
        rgba = np.array(colour or self.primary, dtype=np.uint8)
        block = np.broadcast_to(rgba, (rect.h, rect.w, 4))
        cell.apply_masked(rect, block, mask)
        if delta is not None:
            self.history.push(delta, "Fill selection", frame_id=self.doc.frame.id)
        self.invalidate(rect)
        self.dirty_since_save = True
        self.emit("history.changed")
        return rect

    def inpaint_selection(self, method="content_aware"):
        """Edit > Content-Aware Fill: rebuild the selected area from its surroundings.

        The selection is the hole; its antialiased edge becomes a soft blend
        between the fill and what was there. 'smooth' fills with the smoothest
        colour surface instead of texture, for flat areas.
        """
        from ..engine import inpaint
        from ..engine.commands import PixelDelta
        if self.active_layer is None or self.active_layer.is_group:
            return None
        if self.selection.mask is None:
            return None                       # nothing selected: no hole
        rect, mask = self._selection_window()
        if rect is None:
            return None
        cell = self.doc.cell(self.active_layer)
        hole = self.selection.mask >= 1
        # A one-shot command can afford more iterations than a brush release.
        kw = {"quality": 2.0} if method == "content_aware" else {}
        filled = inpaint.fill_hole(cell.plane("rgba"), hole, method=method, **kw)
        label = "Content-Aware Fill" if method == "content_aware" else "Smooth Fill"
        delta = PixelDelta.capture(self.doc, self.active_layer.id, self.doc.frame.id,
                                   rect, label)
        cell.lerp_masked(rect, filled[rect.slice()], mask)
        if delta is not None:
            self.history.push(delta, label, frame_id=self.doc.frame.id)
        self.invalidate(rect)
        self.dirty_since_save = True
        self.emit("history.changed")
        return rect

    def paste(self, clip=None, at=None, new_layer=False):
        """Put a clip on the active layer (or a new one) and select it.

        Afterwards the pasted region is the selection and the tool is Move
        Selected Pixels, so the next drag positions it -- which is what every
        editor's users expect a paste to do, and what makes "paste, then
        move it where I meant" one gesture rather than three.
        """
        from ..engine import clipboard
        from ..engine.commands import PixelDelta
        clip = clip or self.clip
        if clip is None or self.doc is None:
            return None
        if self.tool is not None and self.tool.active:
            self.cancel_stroke()

        self.history.begin_group()
        if new_layer or self.active_layer is None or self.active_layer.is_group:
            planes = ("index", "rgba") if (clip.index is not None and self.doc.palette is not None
                                           and clip.palette is not None
                                           and np.array_equal(self.doc.palette.entries,
                                                              clip.palette.entries)) else None
            layer = self.add_layer("Pasted", planes=planes,
                                   authoritative=("index",) if planes else None)
        else:
            layer = self.active_layer

        if at is None:
            origin = clip.origin
            fits = origin.clipped_to(self.doc.width, self.doc.height)
            at = (origin.x, origin.y) if fits is not None and fits.w == clip.width                 and fits.h == clip.height else (0, 0)

        cell = self.doc.cell(layer)
        target = Rect(int(at[0]), int(at[1]), clip.width, clip.height)
        landed_guess = target.clipped_to(self.doc.width, self.doc.height)
        delta = None
        if landed_guess is not None:
            delta = PixelDelta.capture(self.doc, layer.id, self.doc.frame.id,
                                       landed_guess, "Paste")
        landed = clipboard.paste(self.doc, layer, self.doc.frame, clip, at)
        if delta is not None and landed is not None:
            self.history.push(delta, "Paste", frame_id=self.doc.frame.id)
        self.history.end_group("Paste")

        if landed is not None:
            self.selection.select_rect((landed.x, landed.y, landed.x1, landed.y1),
                                       antialias=False)
            self.emit("selection.changed")
            self.invalidate(landed)
            if "move_pixels" in self.tools:
                self.set_tool("move_pixels")
        self.dirty_since_save = True
        self.emit("history.changed")
        return landed

    def paste_new_image(self, clip=None):
        clip = clip or self.clip
        if clip is None:
            return None
        indexed = clip.index is not None and clip.palette is not None
        doc = Document(clip.width, clip.height,
                       palette=clip.palette if indexed else None,
                       settings=self.settings)
        if indexed:
            layer = doc.add_layer("Pasted", planes=("index", "rgba"),
                                  authoritative=("index",))
            doc.cell(layer).plane("index")[...] = clip.index
            doc.cell(layer).content_bbox = doc.bounds
            doc.cell(layer).refresh_derived()
        else:
            layer = doc.add_layer("Pasted")
            doc.cell(layer).plane("rgba")[...] = clip.rgba
            doc.cell(layer).content_bbox = doc.bounds
        self._adopt(doc, path=None)
        return doc

    def _after_structure_change(self):
        """Anything that changes the stack invalidates the composite cache."""
        self.compositor.invalidate()
        self.invalidate()
        self.dirty_since_save = True
        self.emit("layers.changed")

    # ---- frames ----------------------------------------------------------

    def set_current_frame(self, index):
        self.doc.set_current_frame(index)
        self.compositor.invalidate()
        self.invalidate()
        self.emit("frame.changed", index=self.doc.current)
        return self.doc.current

    def add_frame(self, copy_current=False):
        from ..engine.commands import FrameDelta
        frame = self.doc.add_frame(index=self.doc.current + 1,
                                   copy_from=self.doc.frame if copy_current else None)
        self.history.push(FrameDelta.insertion(self.doc, frame, "Add frame"), "Add frame")
        self.dirty_since_save = True
        self.emit("frames.changed")
        return frame

    def remove_frame(self, frame=None):
        from ..engine.commands import FrameDelta
        target = frame or self.doc.frame
        if len(self.doc.frames) <= 1:
            return False
        cmd = FrameDelta.removal(self.doc, target, "Delete frame")
        if not self.doc.remove_frame(target):
            return False
        self.history.push(cmd, "Delete frame")
        self.compositor.invalidate()
        self.invalidate()
        self.dirty_since_save = True
        self.emit("frames.changed")
        self.emit("frame.changed", index=self.doc.current)
        return True

    # ---- palette ---------------------------------------------------------

    def set_palette_entry(self, index, rgba):
        from ..engine.commands import PaletteDelta
        if self.doc.palette is None:
            return None
        cmd = PaletteDelta(index, self.doc.palette.rgba(index))
        touched = self.doc.set_palette_entry(index, rgba)
        self.history.push(cmd, "Change palette colour")
        for _key, rect in touched:
            self.invalidate(rect)
        self.dirty_since_save = True
        self.emit("palette.changed", index=index)
        return touched

    def set_primary(self, rgba, index=None):
        self.primary = tuple(rgba)
        self.primary_index = index
        self.emit("colour.changed", which="primary", rgba=self.primary, index=index)

    def set_secondary(self, rgba, index=None):
        self.secondary = tuple(rgba)
        self.secondary_index = index
        self.emit("colour.changed", which="secondary", rgba=self.secondary, index=index)

    # ---- tools -----------------------------------------------------------

    def tool_option_values(self, name):
        """What a tool's options are right now: INI, then the user's choices.

        Kept per tool and remembered between runs, because a tool switch
        builds a fresh instance -- and losing your brush size every time you
        pick the eraser is the kind of thing that makes an editor feel broken.
        """
        values = self.tools.defaults(name)
        remembered = self.session.tool_options(name) if self.session else {}
        for opt in self.tools.spec(name):
            if opt.name in remembered:
                values[opt.name] = opt.coerce(remembered[opt.name])
        values.update(self._tool_overrides.get(name, {}))
        return values

    def set_tool_option(self, key, value, name=None):
        """Change one option of the active (or a named) tool. Remembered."""
        name = name or self.tool_name
        spec = {o.name: o for o in self.tools.spec(name)}
        if key in spec:
            value = spec[key].coerce(value)
        self._tool_overrides.setdefault(name, {})[key] = value
        if self.session is not None:
            self.session.remember_tool_option(name, key, value)
        if name == self.tool_name and self.tool is not None:
            self.tool.set_option(key, value)
        self.emit("tool.options", name=name, key=key, value=value)
        return value

    def set_tool(self, name, **options):
        values = self.tool_option_values(name) if name in self.tools else {}
        values.update(options)
        tool = self.tools.create(name, **values)
        if tool is None:
            return None
        if self.tool is not None and self.tool.active:
            # A pending transform, crop or text box is WORK: switching tools
            # applies it, the way Photoshop and Paint.NET do, instead of
            # silently discarding it.
            if getattr(self.tool, "commit_on_switch", False) and hasattr(self.tool, "finish"):
                self.finish_tool()
            else:
                self.tool.cancel(self._context())
        self.tool = tool
        self.tool_name = name
        self.emit("tool.changed", name=name, tool=tool)
        return tool

    def _context(self):
        ctx = ToolContext(self.doc, self.active_layer, self.doc.frame,
                          self.selection, self.primary, self.secondary,
                          self.settings, self.history)
        ctx.primary_index = self.primary_index
        ctx.secondary_index = self.secondary_index
        ctx.clip = self.clip
        ctx.history_source = self.history_source
        ctx.pick_radius = self.pick_radius
        return ctx

    def tick_stroke(self):
        """A pointer-at-rest tick, for tools that keep working (airbrush)."""
        if self.tool is None or not self.tool.active or not hasattr(self.tool, "tick"):
            return None
        return self.tool.tick(self._ctx)

    # ---- history brush source --------------------------------------------

    # Snapshotting at open costs memory proportional to the document, so it
    # is skipped above this many pixels (layers x canvas, current frame).
    HISTORY_SOURCE_AUTO_LIMIT = 64 * 1000 * 1000

    def set_history_source(self, auto=False):
        """Capture the current frame's layers for the History Brush.

        Compressed, like the history itself. At open this happens
        automatically for documents of sane size, so the history brush works
        out of the box -- painting back to "as opened" is what it is for.
        """
        import zlib
        doc = self.doc
        if doc is None:
            return False
        if auto and doc.width * doc.height * max(1, len(doc.layers())) > self.HISTORY_SOURCE_AUTO_LIMIT:
            self._history_src = {}
            return False
        snap = {}
        frame = doc.frame
        for layer in doc.layers():
            cell = doc.cells.get((layer.id, frame.id))
            if cell is None:
                continue
            rgba = cell.plane("rgba")
            snap[(layer.id, frame.id)] = (zlib.compress(rgba.tobytes(), 1), rgba.shape)
        self._history_src = snap
        self.emit("history.source")
        return True

    def history_source(self, layer_id, frame_id):
        import zlib
        entry = getattr(self, "_history_src", {}).get((layer_id, frame_id))
        if entry is None:
            return None
        blob, shape = entry
        return np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(shape)

    def begin_stroke(self, x, y, pressure=1.0, modifiers=0, button=1):
        if self.tool is None or self.active_layer is None:
            return None
        # A tool that is collecting clicks (polygonal lasso, scissors) keeps
        # its context and its "before" snapshot across presses.
        continuing = self.tool.active and getattr(self.tool, "multi_click", False)
        # Photoshop's Move tool with auto-select: the press first makes the
        # topmost layer with paint under the pointer the active one.
        if not continuing and getattr(self.tool, "picks_layer", False) \
                and self.tool.option("auto_select", True):
            picked = self.layer_at(x, y)
            if picked is not None and picked is not self.active_layer:
                self.set_active_layer(picked)
        if getattr(self.tool, "changes_selection", False) and not continuing:
            self._sel_before = (None if self.selection.mask is None
                                else self.selection.mask.copy())
        if continuing and getattr(self, "_ctx", None) is not None:
            self._ctx.pick_radius = self.pick_radius
            rect = self.tool.begin(self._ctx, ToolEvent(x, y, pressure, modifiers, button))
            return rect
        self._ctx = self._context()
        if self.tool.wants_stroke:
            self.compositor.build_cache(self.active_layer)
        rect = self.tool.begin(self._ctx, ToolEvent(x, y, pressure, modifiers, button))
        self._sync_from_context()
        self.dirty_since_save = True
        return rect

    def motion_stroke(self, x, y, pressure=1.0, modifiers=0, button=1):
        if self.tool is None or not self.tool.active:
            return None
        rect = self.tool.motion(self._ctx, ToolEvent(x, y, pressure, modifiers, button))
        self._sync_from_context()
        return rect

    def end_stroke(self, x, y, pressure=1.0, modifiers=0, button=1):
        if self.tool is None:
            return None
        # A tool that changes pixels AND the selection in one gesture (the
        # Patch tool's Destination mode, Content-Aware Move) must undo as one
        # step: otherwise the first Ctrl+Z only moves the marching ants back,
        # which reads as "undo did not undo what I just did".
        grouping = hasattr(self, "_sel_before")
        if grouping:
            self.history.begin_group()
        try:
            result = self.tool.end(self._ctx, ToolEvent(x, y, pressure, modifiers, button))
            self._sync_from_context()
            if not self.tool.active:
                self._record_selection_change()
        finally:
            if grouping:
                self.history.end_group(self.tools.label(self.tool_name)
                                       if self.tool_name else "Selection")
        if getattr(self.tool, "shows_paths", False):
            # Path tools edit the document's paths directly; the Paths dock
            # and the canvas outline follow from this.
            self.emit("paths.changed")
        self.emit("history.changed")
        return result

    def _record_selection_change(self, label=None):
        """Push the selection as it was before the tool, if the tool changed it."""
        if not hasattr(self, "_sel_before"):
            return False
        before = self._sel_before
        del self._sel_before
        after = self.selection.mask
        same = (before is None and after is None) or (
            before is not None and after is not None and np.array_equal(before, after))
        if same:
            return False
        from ..engine.commands import SelectionDelta
        name = label or (self.tools.label(self.tool_name) if self.tool_name else "Selection")
        self.history.push(SelectionDelta(self.selection, before, name), name)
        self.emit("selection.changed")
        return True

    def hover_tool(self, x, y):
        """Pointer moved with no button down while a tool is mid-task."""
        tool = self.tool
        if tool is None or not tool.active or not hasattr(tool, "hover"):
            return False
        return bool(tool.hover(self._ctx, ToolEvent(x, y)))

    def finish_tool(self):
        """Double-click / Enter: complete a multi-click tool."""
        tool = self.tool
        if tool is None or not tool.active or not hasattr(tool, "finish"):
            return None
        result = tool.finish(self._ctx)
        if getattr(tool, "shows_paths", False):
            self.emit("paths.changed")
        # The tool decides whether it is done: Enter on an open cage CLOSES
        # it and the tool stays active for dragging, so forcing it inactive
        # here broke that.
        self._sync_from_context()
        self._record_selection_change()
        # Some tools finish by asking for a whole-document change they cannot
        # make themselves: a crop resizes the canvas, which only the
        # controller (display buffer, selection, history) can do.
        if isinstance(result, dict):
            if "crop" in result:
                rect = result["crop"]
                from ..engine import transform
                self._geometry("Crop", lambda: transform.crop(self.doc, rect))
            elif "perspective_crop" in result:
                quad = result["perspective_crop"]
                from ..engine import transform
                self._geometry("Perspective Crop",
                               lambda: transform.perspective_crop(self.doc, quad))
        else:
            self.invalidate()
        self.emit("history.changed")
        return result

    def cancel_stroke(self):
        if self.tool is None or not self.tool.active:
            return None
        rect = self.tool.cancel(self._ctx)
        if hasattr(self, "_sel_before"):
            del self._sel_before
        if rect is not None:
            self.invalidate(rect if isinstance(rect, Rect) else self.doc.bounds)
        return rect

    def _sync_from_context(self):
        """Tools may pick colours; carry that back out."""
        ctx = getattr(self, "_ctx", None)
        if ctx is None:
            return
        if ctx.primary != self.primary or ctx.primary_index != self.primary_index:
            self.set_primary(ctx.primary, ctx.primary_index)
        if ctx.secondary != self.secondary or ctx.secondary_index != self.secondary_index:
            self.set_secondary(ctx.secondary, ctx.secondary_index)

    # ---- selection -------------------------------------------------------

    def _selection_op(self, label, fn):
        """Run a selection change as one undo entry. False if nothing changed."""
        self._sel_before = None if self.selection.mask is None else self.selection.mask.copy()
        fn(self.selection)
        self.selection._bbox = None
        if self.selection.mask is not None and self.selection.mask.min() == 255:
            self.selection.mask = None
        changed = self._record_selection_change(label)
        self.emit("history.changed")
        return changed

    # ---- vector paths -----------------------------------------------------

    def _path_change(self, label, fn):
        """Run a change to the document's paths as one undo entry."""
        from ..engine.commands import PathDelta
        before = PathDelta.capture(self.doc, label)
        result = fn()
        if not before.same_as(self.doc):
            self.history.push(before, label)
            self.dirty_since_save = True
        self.emit("paths.changed")
        self.emit("history.changed")
        self.invalidate()
        return result

    def path_names(self):
        return [p.name for p in self.doc.paths]

    def set_active_path(self, index):
        """Make a path the one the path tools edit (None: no active path)."""
        if index is not None and not (0 <= index < len(self.doc.paths)):
            index = None
        if index == self.doc.active_path:
            return False
        # Choosing a path is not an edit and is not undoable, as in the
        # Paths panel; it only changes what the tools act on.
        self.doc.active_path = index
        self.emit("paths.changed")
        self.invalidate()
        return True

    def new_path(self, name=None):
        from ..engine.paths import Path
        from ..engine.tools.pathtools import _new_name

        def run():
            self.doc.paths.append(Path(name or _new_name(self.doc)))
            self.doc.active_path = len(self.doc.paths) - 1
            return self.doc.active_path
        return self._path_change("New Path", run)

    def delete_path(self, index=None):
        index = self.doc.active_path if index is None else index
        if index is None or not (0 <= index < len(self.doc.paths)):
            return False

        def run():
            del self.doc.paths[index]
            if not self.doc.paths:
                self.doc.active_path = None
            else:
                self.doc.active_path = min(index, len(self.doc.paths) - 1)
            return True
        return self._path_change("Delete Path", run)

    def duplicate_path(self, index=None):
        index = self.doc.active_path if index is None else index
        if index is None or not (0 <= index < len(self.doc.paths)):
            return None

        def run():
            dup = self.doc.paths[index].copy()
            dup.name = dup.name + " copy"
            self.doc.paths.insert(index + 1, dup)
            self.doc.active_path = index + 1
            return index + 1
        return self._path_change("Duplicate Path", run)

    def rename_path(self, index, name):
        if not (0 <= index < len(self.doc.paths)) or not str(name).strip():
            return False

        def run():
            self.doc.paths[index].name = str(name).strip()
            return True
        return self._path_change("Rename Path", run)

    def path_to_selection(self, mode=None, index=None):
        """Select > From Path: the path's interior becomes the selection."""
        from ..engine import paths as P
        from ..engine.selection import REPLACE
        index = self.doc.active_path if index is None else index
        if index is None or not (0 <= index < len(self.doc.paths)):
            return False
        vpath = self.doc.paths[index]
        if vpath.is_empty:
            return False
        mask = P.fill_mask(vpath, self.doc.width, self.doc.height, antialias=True)
        return self._selection_op("Path to Selection",
                                  lambda sel: sel.combine(mask, mode or REPLACE))

    def selection_to_path(self, smooth=False):
        """Select > To Path: the selection's outline becomes a new path."""
        from ..engine import paths as P
        mask = self.selection.mask
        if mask is None or not mask.any():
            return None
        traced = P.from_mask(mask, smooth=smooth)
        if traced.is_empty:
            return None

        def run():
            from ..engine.tools.pathtools import _new_name
            traced.name = _new_name(self.doc) if self.doc.paths else "Selection"
            self.doc.paths.append(traced)
            self.doc.active_path = len(self.doc.paths) - 1
            return self.doc.active_path
        return self._path_change("Selection to Path", run)

    def _paint_coverage(self, label, cov, colour):
        from ..engine.commands import PixelDelta
        from ..engine.geometry import Rect
        if self.active_layer is None or self.active_layer.is_group:
            return None
        ys, xs = np.nonzero(cov)
        if ys.size == 0:
            return None
        rect = Rect.from_bounds(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        window = self.selection.combine_mask(cov[rect.slice()], rect)
        cell = self.doc.cell(self.active_layer)
        delta = PixelDelta.capture(self.doc, self.active_layer.id, self.doc.frame.id, rect, label)
        block = np.broadcast_to(np.array(colour, dtype=np.uint8), (rect.h, rect.w, 4))
        cell.apply_masked(rect, block, window)
        if delta is not None:
            self.history.push(delta, label, frame_id=self.doc.frame.id)
        self.invalidate(rect)
        self.dirty_since_save = True
        self.emit("history.changed")
        return rect

    def fill_path(self, colour=None, index=None, antialias=True):
        """Edit > Fill Path: paint the path's interior on the active layer."""
        from ..engine import paths as P
        index = self.doc.active_path if index is None else index
        if index is None or not (0 <= index < len(self.doc.paths)):
            return None
        cov = P.fill_mask(self.doc.paths[index], self.doc.width, self.doc.height, antialias)
        return self._paint_coverage("Fill Path", cov, colour or self.primary)

    def stroke_path(self, width=2.0, colour=None, index=None, antialias=True):
        """Edit > Stroke Path: paint along the path's outline."""
        from ..engine import paths as P
        index = self.doc.active_path if index is None else index
        if index is None or not (0 <= index < len(self.doc.paths)):
            return None
        cov = P.stroke_mask(self.doc.paths[index], self.doc.width, self.doc.height,
                            float(width), antialias)
        return self._paint_coverage("Stroke Path", cov, colour or self.primary)

    def select_all(self):
        return self._selection_op("Select All", lambda s: s.select_all())

    def deselect(self):
        return self._selection_op("Deselect", lambda s: s.select_all())

    def invert_selection(self):
        return self._selection_op("Invert Selection", lambda s: s.invert())

    def _mask_op(self, label, fn):
        """Transform the selection mask. A select-all selection is treated as
        a full mask, so 'shrink' on no selection still means something."""
        def run(sel):
            m = sel.mask if sel.mask is not None else \
                np.full((sel.height, sel.width), 255, np.uint8)
            sel.mask = np.asarray(fn(m), dtype=np.uint8)
        return self._selection_op(label, run)

    def grow_selection(self, radius):
        from ..engine import morph
        return self._mask_op("Grow Selection", lambda m: morph.grow(m, radius))

    def shrink_selection(self, radius):
        from ..engine import morph
        return self._mask_op("Shrink Selection", lambda m: morph.shrink(m, radius))

    def border_selection(self, radius):
        from ..engine import morph
        return self._mask_op("Border Selection", lambda m: morph.border(m, radius))

    def feather_selection(self, radius):
        return self._selection_op("Feather Selection", lambda s: s.feather(radius))

    def sharpen_selection(self):
        from ..engine import morph
        return self._mask_op("Sharpen Selection", morph.sharpen)

    def remove_selection_holes(self):
        from ..engine import morph
        return self._mask_op("Remove Holes", morph.remove_holes)

    def round_selection(self, radius):
        from ..engine import morph
        return self._mask_op("Rounded Selection", lambda m: morph.round_corners(m, radius))

    def alpha_to_selection(self):
        """The active layer's opacity becomes the selection. GIMP's Alpha to
        Selection; Photoshop's ctrl-click on a layer thumbnail."""
        if self.active_layer is None or self.active_layer.is_group:
            return False
        cell = self.doc.cell(self.active_layer)
        alpha = cell.plane("rgba")[..., 3].copy()
        return self._selection_op("Alpha to Selection",
                                  lambda s: setattr(s, "mask", alpha))

    def select_by_colour_at(self, x, y, tolerance=32):
        from ..engine.tools.select import SelectByColorTool
        tool = SelectByColorTool(tolerance=tolerance)
        ctx = self._context()
        return self._selection_op("Select by Color",
                                  lambda s: tool.on_begin(ctx, ToolEvent(x, y)))

    # ---- quick mask ------------------------------------------------------

    def toggle_quick_mask(self):
        """Edit the selection by PAINTING it, as a temporary layer.

        On: the selection becomes a layer on top -- white where selected,
        red where not, multiplied so the image shows through -- and the brush,
        eraser, gradient and every other tool edit it directly. Off: the
        layer's green channel becomes the selection again and the layer goes.
        GIMP's and Photoshop's Quick Mask, built on an ordinary layer so
        nothing new is needed to paint it.
        """
        if self.quick_mask is None:
            mask = self.selection.mask if self.selection.mask is not None else \
                np.full((self.doc.height, self.doc.width), 255, np.uint8)
            layer = self.doc.add_layer("Quick Mask")
            layer.blend = "Multiply"
            layer.opacity = 160
            cell = self.doc.cell(layer)
            px = cell.plane("rgba")
            px[..., 0] = 255
            px[..., 1] = mask
            px[..., 2] = mask
            px[..., 3] = 255
            cell._touched(self.doc.bounds)
            self.quick_mask = (layer, self.active_layer)
            self.set_active_layer(layer)
            # White paints "selected", black paints "not selected".
            self.set_primary((255, 255, 255, 255))
            self.set_secondary((0, 0, 0, 255))
        else:
            layer, previous = self.quick_mask
            self.quick_mask = None
            cell = self.doc.cells.get((layer.id, self.doc.frame.id))
            mask = None if cell is None else cell.plane("rgba")[..., 1].copy()
            self.doc.remove_layer(layer)
            if previous is not None and self.doc.layer(previous.id) is not None:
                self.set_active_layer(previous)
            else:
                leaves = self.doc.layers()
                self.set_active_layer(leaves[-1] if leaves else None)
            if mask is not None:
                self._selection_op("Quick Mask", lambda s: setattr(s, "mask", mask))
        self._after_structure_change()
        self.emit("quickmask.changed", on=self.quick_mask is not None)
        return self.quick_mask is not None

    # ---- history ---------------------------------------------------------

    def undo(self):
        # An open tool session IS the most recent thing the user did, and it
        # is not in history yet -- the text tool deliberately stays active
        # after mouse-up so typing can continue. Undoing past it was wrong
        # twice over: it rolled back the edit BEFORE the one on screen, and
        # the session still held a checkpoint of pixels from before that
        # rollback, so committing afterwards wrote them back. Undoing
        # everything then left pixels on the canvas that had never existed in
        # any state the user visited.
        if self.tool is not None and self.tool.active:
            rect = self.cancel_stroke()
            self.emit("history.changed")
            return rect

        rect = self.history.undo(self.doc)
        self._after_history_step(rect)
        return rect

    def redo(self):
        # Same hazard in the other direction: redoing with a live session
        # open would reapply pixels underneath a checkpoint taken before it.
        # Discard the session first, then redo normally -- there is nothing
        # uncommitted to redo.
        if self.tool is not None and self.tool.active:
            self.cancel_stroke()

        rect = self.history.redo(self.doc)
        self._after_history_step(rect)
        return rect

    def _after_history_step(self, rect):
        if rect is not None:
            resized = self.display.shape[:2] != (self.doc.height, self.doc.width)
            if resized:
                self._after_geometry_change()
            else:
                self.compositor.invalidate()
                self.invalidate(rect)
            self.dirty_since_save = True
            # Structure may have changed under the active layer.
            if self.active_layer is not None and self.doc.layer(self.active_layer.id) is None:
                leaves = self.doc.layers()
                self.active_layer = leaves[-1] if leaves else None
            self.emit("layers.changed")
            self.emit("frames.changed")
        self.emit("paths.changed")
        self.emit("history.changed")

    def history_labels(self):
        return self.history.labels()

    def history_goto(self, index):
        rect = self.history.goto(index, self.doc)
        self._after_history_step(rect if rect is not None else self.doc.bounds)
        return rect

    # ---- introspection ---------------------------------------------------

    def status(self):
        """A one-line summary. Handy for a status bar and for a self-test."""
        return {
            "backend": accel.BACKEND,
            "size": (self.doc.width, self.doc.height) if self.doc else None,
            "layers": len(self.doc.layers()) if self.doc else 0,
            "frames": len(self.doc.frames) if self.doc else 0,
            "history": len(self.history) if self.history else 0,
            "tool": self.tool_name,
            "indexed": self.doc.palette is not None if self.doc else False,
        }
