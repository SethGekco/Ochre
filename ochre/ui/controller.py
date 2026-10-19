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
        # Snapping and the grid. Engine-side so snapping is testable; the
        # canvas only draws the grid and guides.
        self.snap_guides = True
        self.snap_grid = False
        self.snap_bounds = False
        self.grid = (16, 16)
        self.last_measure = None
        # Open documents, one tab each. The current one lives in the
        # attributes above; the others are parked here as state bundles.
        self.documents = []
        self.current_doc = -1
        # Painting goes to the active layer's MASK rather than its pixels.
        self.editing_mask = False
        self.recording = None           # an Action being recorded

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
        from ..engine.fileio import ora, psd
        if is_ochre(path):
            doc, mask = load(path, settings=self.settings)
            self._adopt(doc, path=path)
            if mask is not None:
                self.selection.mask = mask
                self.selection._bbox = None
        elif psd.is_psd(path):
            # Layered formats open as THEMSELVES: Save writes the PSD (or ORA)
            # back, as Photoshop and GIMP do, rather than demanding Save As.
            try:
                doc = psd.load(path, settings=self.settings)
            except Exception as exc:
                raise self._unopenable(path, exc) from exc
            self._adopt(doc, path=path)
        elif ora.is_ora(path):
            doc = ora.load(path, settings=self.settings)
            self._adopt(doc, path=path)
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
        from ..engine.fileio import ora, psd
        lower = target.lower()
        self.last_save_warnings = []
        if lower.endswith(".ochre"):
            save(self.doc, target, selection=self.selection)
            self.path = target
        elif lower.endswith((".psd", ".psb")):
            self.last_save_warnings = psd.save(self.doc, target, self.compositor)
            self.path = target
        elif lower.endswith(".ora"):
            self.last_save_warnings = ora.save(self.doc, target, self.compositor)
            self.path = target
        else:
            export_flat(self.doc, target)
        self.dirty_since_save = False
        self._saved_marker = self.history.marker()
        self.emit("document.saved", path=target)
        return target

    # Everything that belongs to ONE open document rather than to the editor.
    _PER_DOC = ("doc", "history", "selection", "compositor", "blends", "display",
                "path", "dirty_since_save", "_saved_marker", "active_layer",
                "_pending", "quick_mask", "last_measure", "_history_src",
                "editing_mask")

    def _capture(self):
        return {k: getattr(self, k, None) for k in self._PER_DOC}

    def _restore(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def _pristine(self):
        """An untitled document nobody has touched: replaced, not kept."""
        return (self.doc is not None and self.path is None and not self.is_modified()
                and not self.history.entries)

    def document_titles(self):
        """One title per open document, '*' marking unsaved work."""
        out = []
        for i, state in enumerate(self.documents):
            if i == self.current_doc:
                out.append(self.title)
                continue
            path = state.get("path")
            name = os.path.basename(path) if path else "Untitled"
            out.append(("*" if self._state_modified(state) else "") + name)
        return out

    @staticmethod
    def _state_modified(state):
        history = state.get("history")
        if history is None:
            return False
        current = history.marker()
        saved = state.get("_saved_marker")
        if saved is None:
            return current[0] is not None or current[2] != 0
        return current[1] is not saved[1] or current[2] != saved[2]

    def modified_documents(self):
        """Indices of documents with unsaved work."""
        return [i for i, st in enumerate(self.documents)
                if (self.is_modified() if i == self.current_doc else self._state_modified(st))]

    def _leave_current(self):
        """Settle the tool before the document under it changes."""
        if self.tool is not None and self.tool.active:
            if getattr(self.tool, "commit_on_switch", False) and hasattr(self.tool, "finish"):
                self.finish_tool()
            else:
                self.tool.cancel(self._context())
        if 0 <= self.current_doc < len(self.documents):
            self.documents[self.current_doc] = self._capture()

    def switch_document(self, index):
        """Bring another open document to the front."""
        if not (0 <= index < len(self.documents)) or index == self.current_doc:
            return False
        self._leave_current()
        self.current_doc = index
        self._restore(self.documents[index])
        if self.tool_name:
            self.set_tool(self.tool_name)
        self.emit("document.opened", document=self.doc, path=self.path)
        self.emit("documents.changed")
        return True

    def close_document(self, index=None):
        """Close a document (the current one by default) WITHOUT asking.

        Asking is the window's job; by the time this runs the user has said
        yes. Closing the last one leaves a fresh untitled canvas, so there is
        always something to draw on.
        """
        index = self.current_doc if index is None else index
        if not (0 <= index < len(self.documents)):
            return False
        if index == self.current_doc:
            if self.tool is not None and self.tool.active:
                self.tool.cancel(self._context())
            del self.documents[index]
            self.current_doc = -1
            if not self.documents:
                self.doc = None
                self.new_document()
            else:
                target = min(index, len(self.documents) - 1)
                self.current_doc = target
                self._restore(self.documents[target])
                if self.tool_name:
                    self.set_tool(self.tool_name)
                self.emit("document.opened", document=self.doc, path=self.path)
        else:
            del self.documents[index]
            if index < self.current_doc:
                self.current_doc -= 1
        self.emit("documents.changed")
        return True

    def _adopt(self, doc, path):
        # A new tab, unless the current document is an untouched Untitled
        # -- then it is simply replaced, as Paint.NET does with its blank
        # start-up canvas.
        if self.doc is None or not self.documents:
            self.documents = [None]
            self.current_doc = 0
        elif self._pristine():
            pass
        else:
            self._leave_current()
            self.documents.append(None)
            self.current_doc = len(self.documents) - 1
        self.quick_mask = None
        self.last_measure = None
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
        self.documents[self.current_doc] = self._capture()
        self.emit("document.opened", document=doc, path=path)
        self.emit("documents.changed")

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
        # A layer style draws beyond the pixels that changed -- a stroke or
        # shadow follows new paint outward -- so repaints grow by the
        # furthest any style in the document reaches (nearly always zero).
        from ..engine import layerstyles
        grow = max([layerstyles.reach(l.styles) for l in self.doc.layers()
                    if getattr(l, "styles", None) and layerstyles.enabled(l.styles)] + [0])
        merged = []
        for rect in pending:
            if grow:
                rect = rect.inflated(grow)
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
        if layer is not self.active_layer:
            self.editing_mask = False
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
        copy.kind, copy.params = source.kind, dict(source.params)
        copy.clip, copy.mask_enabled = source.clip, source.mask_enabled
        copy.blend_if = source.blend_if
        copy.styles = {k: dict(v) for k, v in source.styles.items()}
        if source.mask is not None:
            from ..engine.layer import LayerMask
            copy.mask = LayerMask(copy)
            for (lid, fid), mcell in list(self.doc.cells.items()):
                if lid != source.mask.id:
                    continue
                target = self.doc.cell(copy.mask, self.doc.frame_by_id(fid))
                target.plane("rgba")[...] = mcell.plane("rgba")
                target.content_bbox = mcell.content_bbox
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

    def match_color(self, source_index, amount=1.0):
        """Image > Match Color: take another open image's colour statistics.

        Applied to the active layer, inside the selection; the source is the
        merged look of the other document.
        """
        from ..engine import colors
        from ..engine.commands import PixelDelta
        if not (0 <= source_index < len(self.documents)) or source_index == self.current_doc:
            return None
        layer = self.active_layer
        if layer is None or layer.is_group or getattr(layer, "kind", "raster") != "raster":
            return None
        state = self.documents[source_index]
        src_doc = state["doc"]
        source = state["compositor"].composite(src_doc.bounds)
        cell = self.doc.cell(layer)
        rect = self.doc.bounds
        delta = PixelDelta.capture(self.doc, layer.id, self.doc.frame.id, rect, "Match Color")
        matched = colors.match_color(cell.plane("rgba"), source, amount)
        cell.lerp_masked(rect, matched, self.selection.mask_for(rect))
        if delta is not None:
            self.history.push(delta, "Match Color", frame_id=self.doc.frame.id)
        self.invalidate(rect)
        self.dirty_since_save = True
        self.emit("history.changed")
        return rect

    # ---- masks, clipping, adjustment and fill layers, styles ----------------

    def add_layer_mask(self, mode="reveal", layer=None):
        """Layer > Layer Mask > Reveal All / Hide All / From Selection / From
        Transparency. The new mask becomes what you paint on, as in Photoshop."""
        from ..engine.commands import MaskDelta
        from ..engine.layer import LayerMask
        owner = layer or self.active_layer
        if owner is None or owner.is_group or getattr(owner, "is_mask", False) \
                or owner.mask is not None:
            return None
        before = MaskDelta.capture(self.doc, owner, "Add Layer Mask")
        owner.mask = LayerMask(owner)
        for frame in self.doc.frames:
            if frame is not self.doc.frame and (owner.id, frame.id) not in self.doc.cells:
                continue
            cell = self.doc.cell(owner.mask, frame)
            px = cell.plane("rgba")
            px[..., 3] = 255
            if mode == "hide":
                value = 0
            elif mode == "selection":
                value = 255 if self.selection.mask is None else self.selection.mask
            elif mode == "transparency":
                src = self.doc.cells.get((owner.id, frame.id))
                value = 0 if src is None else src.plane("rgba")[..., 3]
            else:
                value = 255
            px[..., 0] = value
            px[..., 1] = value
            px[..., 2] = value
            cell._touched(self.doc.bounds)
        self.history.push(before, "Add Layer Mask")
        self.editing_mask = True
        self._after_structure_change()
        return owner.mask

    def delete_layer_mask(self, apply=False, layer=None):
        """Delete the mask -- or Apply it: bake it into the layer's alpha first."""
        from ..engine.commands import MaskDelta, PixelDelta
        from ..engine.compositor import mask_value
        owner = layer or self.active_layer
        if owner is None or getattr(owner, "mask", None) is None:
            return False
        label = "Apply Layer Mask" if apply else "Delete Layer Mask"
        self.history.begin_group()
        if apply and getattr(owner, "kind", "raster") == "raster":
            for frame in self.doc.frames:
                mcell = self.doc.cells.get((owner.mask.id, frame.id))
                cell = self.doc.cells.get((owner.id, frame.id))
                if mcell is None or cell is None or cell.content_bbox is None:
                    continue
                delta = PixelDelta.capture(self.doc, owner.id, frame.id, self.doc.bounds, label)
                m = mask_value(mcell.plane("rgba"))
                if cell.index_locked:
                    idx = cell.plane("index")
                    clear = 0 if cell.palette is None or cell.palette.transparent is None \
                        else int(cell.palette.transparent)
                    idx[m < 128] = clear
                    cell.refresh_derived()
                else:
                    px = cell.plane("rgba")
                    a = px[..., 3].astype(np.uint32)
                    px[..., 3] = ((a * m + 127) // 255).astype(np.uint8)
                cell._touched(self.doc.bounds)
                if delta is not None:
                    self.history.push(delta, label, frame_id=frame.id)
        before = MaskDelta.capture(self.doc, owner, label)
        for key in [k for k in self.doc.cells if k[0] == owner.mask.id]:
            del self.doc.cells[key]
        owner.mask = None
        self.history.push(before, label)
        self.history.end_group(label)
        self.editing_mask = False
        self._after_structure_change()
        return True

    def set_editing_mask(self, on):
        layer = self.active_layer
        self.editing_mask = bool(on) and layer is not None and getattr(layer, "mask", None) is not None
        self.emit("layer.active", layer=layer)
        return self.editing_mask

    def set_mask_enabled(self, on, layer=None):
        owner = layer or self.active_layer
        if owner is None or getattr(owner, "mask", None) is None:
            return False
        self.set_layer_property(owner, "mask_enabled", bool(on))
        return True

    def toggle_clipping(self, layer=None):
        """Layer > Create / Release Clipping Mask (Ctrl+Alt+G)."""
        target = layer or self.active_layer
        if target is None or target.is_group:
            return None
        self.set_layer_property(target, "clip", not target.clip)
        return target.clip

    def add_adjustment_layer(self, effect, params=None, name=None):
        """A layer that applies an adjustment to everything beneath it, and
        stays editable: Photoshop's adjustment layers."""
        from ..engine.layer import ADJUSTMENT
        label = "New Adjustment Layer"
        self.history.begin_group()
        layer = self.add_layer(name or self._effect_label(effect))
        layer.kind = ADJUSTMENT
        layer.params = dict(params or {}, effect=effect)
        self.add_layer_mask("reveal", layer)
        self.history.end_group(label)
        self.editing_mask = False
        self._after_structure_change()
        return layer

    @property
    def effects(self):
        from ..engine.effects import EffectRegistry
        reg = getattr(self, "_effect_reg", None)
        if reg is None:
            reg = self._effect_reg = EffectRegistry(self.db)
        return reg

    def _effect_label(self, effect):
        return self.effects.label(effect)

    def apply_effect(self, name, params=None):
        """Run an adjustment or filter on what the tools paint (the layer, or
        its mask), inside the selection, as one undo step. What the effect
        dialogs call on OK -- and so what an Action records."""
        from ..engine.commands import PixelDelta
        effect = self.effects.create(name, **dict(params or {}))
        target = self.paint_target
        if effect is None or target is None or target.is_group:
            return None
        if getattr(target, "kind", "raster") not in ("raster",) and not getattr(target, "is_mask", False):
            return None
        cell = self.doc.cell(target)
        bounds = self.doc.bounds
        before = PixelDelta.capture(self.doc, target.id, self.doc.frame.id, bounds, effect.label)
        result = effect.apply(cell.plane("rgba"))
        cell.lerp_masked(bounds, result, self.selection.mask)
        cell._touched(bounds)
        if before is not None:
            self.history.push(before, effect.label, frame_id=self.doc.frame.id)
        self.invalidate(bounds)
        self.dirty_since_save = True
        self.emit("history.changed")
        return bounds

    # ---- channels (saved selections) ---------------------------------------------

    def save_selection(self, name=None):
        """Select > Save Selection: the selection becomes a named channel."""
        from ..engine.commands import ChannelsDelta
        from ..engine.layer import Channel
        before = ChannelsDelta(self.doc, "Save Selection")
        ch = Channel(name or "Alpha %d" % (len(self.doc.channels) + 1))
        self.doc.channels.append(ch)
        cell = self.doc.cell(ch)
        px = cell.plane("rgba")
        mask = self.selection.mask
        value = 255 if mask is None else mask
        px[..., 0] = value
        px[..., 1] = value
        px[..., 2] = value
        px[..., 3] = 255
        cell._touched(self.doc.bounds)
        self.history.push(before, "Save Selection")
        self.dirty_since_save = True
        self.emit("channels.changed")
        self.emit("history.changed")
        return ch

    def channel_mask(self, index):
        from ..engine.compositor import mask_value
        if not (0 <= index < len(self.doc.channels)):
            return None
        cell = self.doc.cells.get((self.doc.channels[index].id, self.doc.frame.id))
        if cell is None:
            cell = next((c for (lid, _f), c in self.doc.cells.items()
                         if lid == self.doc.channels[index].id), None)
        return None if cell is None else mask_value(cell.plane("rgba"))

    def load_selection(self, index, mode=None):
        """Select > Load Selection: a channel back into the selection, combined
        by mode (Replace / Union / Exclude / Intersect)."""
        from ..engine.selection import REPLACE
        mask = self.channel_mask(index)
        if mask is None:
            return False
        return self._selection_op("Load Selection", lambda sel: sel.combine(mask, mode or REPLACE))

    def delete_channel(self, index):
        from ..engine.commands import ChannelsDelta
        if not (0 <= index < len(self.doc.channels)):
            return False
        before = ChannelsDelta(self.doc, "Delete Channel")
        ch = self.doc.channels.pop(index)
        for key in [k for k in self.doc.cells if k[0] == ch.id]:
            del self.doc.cells[key]
        self.history.push(before, "Delete Channel")
        self.emit("channels.changed")
        self.emit("history.changed")
        return True

    def rename_channel(self, index, name):
        if not (0 <= index < len(self.doc.channels)) or not str(name).strip():
            return False
        self.set_layer_property(self.doc.channels[index], "name", str(name).strip())
        self.emit("channels.changed")
        return True

    # ---- layer comps ------------------------------------------------------------------

    _COMP_ATTRS = ("visible", "opacity", "blend", "mask_enabled")

    def _comp_state(self):
        state = {}
        for node in self.doc.root.walk(include_self=False):
            entry = {a: getattr(node, a) for a in self._COMP_ATTRS if hasattr(node, a)}
            if hasattr(node, "styles"):
                entry["styles"] = {k: dict(v) for k, v in node.styles.items()}
            state[node.id] = entry
        return state

    def new_comp(self, name=None):
        """Layer Comps: remember which layers are shown, their opacity, modes
        and styles, to switch between versions of a design."""
        from ..engine.commands import CompsDelta
        before = CompsDelta(self.doc, "New Layer Comp")
        self.doc.comps.append({"name": name or "Layer Comp %d" % (len(self.doc.comps) + 1),
                               "state": self._comp_state()})
        self.history.push(before, "New Layer Comp")
        self.emit("comps.changed")
        self.emit("history.changed")
        return len(self.doc.comps) - 1

    def update_comp(self, index):
        from ..engine.commands import CompsDelta
        if not (0 <= index < len(self.doc.comps)):
            return False
        before = CompsDelta(self.doc, "Update Layer Comp")
        comp = dict(self.doc.comps[index])
        comp["state"] = self._comp_state()
        self.doc.comps[index] = comp
        self.history.push(before, "Update Layer Comp")
        self.emit("comps.changed")
        return True

    def delete_comp(self, index):
        from ..engine.commands import CompsDelta
        if not (0 <= index < len(self.doc.comps)):
            return False
        before = CompsDelta(self.doc, "Delete Layer Comp")
        del self.doc.comps[index]
        self.history.push(before, "Delete Layer Comp")
        self.emit("comps.changed")
        return True

    def apply_comp(self, index):
        """Show a comp: each layer back to its remembered state, one undo step."""
        from ..engine.commands import PropertyDelta
        if not (0 <= index < len(self.doc.comps)):
            return False
        label = "Apply Layer Comp"
        self.history.begin_group()
        for node_id, entry in self.doc.comps[index]["state"].items():
            node = self.doc.layer(node_id)
            if node is None:
                continue
            for attr, value in entry.items():
                if attr == "styles":
                    value = {k: dict(v) for k, v in value.items()}
                if getattr(node, attr, None) != value:
                    self.history.push(PropertyDelta(node.id, attr, getattr(node, attr), label), label)
                    setattr(node, attr, value)
        self.history.end_group(label)
        self._after_structure_change()
        return True

    # ---- brush presets ------------------------------------------------------------

    BUILTIN_BRUSHES = {
        "brush": {
            "Hard round": {"size": 12, "hardness": 1.0, "size_jitter": 0, "opacity_jitter": 0,
                           "scatter": 0, "count": 1},
            "Soft round": {"size": 40, "hardness": 0.0, "size_jitter": 0, "opacity_jitter": 0,
                           "scatter": 0, "count": 1},
            "Spray": {"size": 4, "hardness": 0.6, "size_jitter": 60, "opacity_jitter": 50,
                      "scatter": 400, "count": 10},
            "Chalk": {"size": 18, "hardness": 0.9, "size_jitter": 30, "opacity_jitter": 60,
                      "scatter": 30, "count": 3},
            "Leaves": {"size": 22, "hardness": 0.8, "size_jitter": 70, "opacity_jitter": 30,
                       "scatter": 250, "count": 2},
        },
        "pencil": {"Pixel": {"size": 1}, "2 px": {"size": 2}, "Block 4": {"size": 4}},
        "eraser": {"Hard": {"size": 16, "hardness": 1.0}, "Soft": {"size": 40, "hardness": 0.0}},
    }

    def _brushes_path(self):
        return os.path.join(self.state_dir, "brushes.json") if self.state_dir else None

    def brush_presets(self, tool=None):
        """{preset: options} for a tool (the active one by default): the
        built-in ones and the user's own, which win on a name clash."""
        import json
        tool = tool or self.tool_name
        out = dict(self.BUILTIN_BRUSHES.get(tool, {}))
        path = self._brushes_path()
        if path and os.path.exists(path):
            try:
                with open(path) as fh:
                    out.update(json.load(fh).get(tool, {}))
            except (OSError, ValueError):
                pass
        return out

    def save_brush_preset(self, name, tool=None):
        """Remember the tool's current options under a name."""
        import json
        tool = tool or self.tool_name
        path = self._brushes_path()
        if not path or self.tool is None:
            return False
        data = {}
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                data = {}
        spec = [o.name for o in self.tools.spec(tool)]
        opts = {k: v for k, v in self.tool_option_values(tool).items() if k in spec}
        data.setdefault(tool, {})[str(name)] = _plain(opts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1)
        self.emit("brushes.changed")
        return True

    def apply_brush_preset(self, name, tool=None):
        tool = tool or self.tool_name
        preset = self.brush_presets(tool).get(name)
        if preset is None:
            return False
        for key, value in preset.items():
            self.set_tool_option(key, value, tool)
        return True

    def delete_brush_preset(self, name, tool=None):
        import json
        tool = tool or self.tool_name
        path = self._brushes_path()
        if not path or not os.path.exists(path):
            return False
        with open(path) as fh:
            data = json.load(fh)
        if data.get(tool, {}).pop(name, None) is None:
            return False
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1)
        self.emit("brushes.changed")
        return True

    # ---- actions: record, replay, batch --------------------------------------

    def start_recording(self, name):
        """Actions panel > Record. Menu commands are logged until stopped."""
        self.recording = {"name": str(name), "steps": []}
        self.emit("actions.changed")

    def stop_recording(self):
        rec, self.recording = getattr(self, "recording", None), None
        if rec is not None and rec["steps"]:
            self.actions()[rec["name"]] = rec["steps"]
            self.save_actions()
        self.emit("actions.changed")
        return rec

    def _record(self, op, args, kwargs):
        rec = getattr(self, "recording", None)
        if rec is None:
            return
        rec["steps"].append({"op": op, "args": _plain(args), "kwargs": _plain(kwargs)})
        self.emit("actions.changed")

    def actions(self):
        if getattr(self, "_actions", None) is None:
            self._actions = {}
            path = self._actions_path()
            if path and os.path.exists(path):
                import json
                try:
                    with open(path) as fh:
                        self._actions = json.load(fh)
                except (OSError, ValueError):
                    self._actions = {}
        return self._actions

    def _actions_path(self):
        return os.path.join(self.state_dir, "actions.json") if self.state_dir else None

    def save_actions(self):
        path = self._actions_path()
        if not path:
            return False
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.actions(), fh, indent=1)
        return True

    def delete_action(self, name):
        if self.actions().pop(name, None) is not None:
            self.save_actions()
            self.emit("actions.changed")
            return True
        return False

    def play_action(self, name_or_steps):
        """Run an action on the current document as ONE undo step."""
        steps = self.actions().get(name_or_steps) if isinstance(name_or_steps, str) else name_or_steps
        if not steps:
            return False
        label = "Action: %s" % (name_or_steps if isinstance(name_or_steps, str) else "steps")
        self.history.begin_group()
        resized = False
        try:
            for step in steps:
                op = step["op"]
                if op not in RECORDABLE:
                    continue          # an action file is data; it may only call these
                before = (self.doc.width, self.doc.height)
                getattr(self, op)(*_tuples(step.get("args", [])), **_tuples(step.get("kwargs", {})))
                resized = resized or before != (self.doc.width, self.doc.height)
        finally:
            self.history.end_group(label)
        if resized:
            self._after_geometry_change()
        self.compositor.invalidate()
        self.invalidate()
        self.emit("history.changed")
        return True

    def batch(self, action, paths, out_dir, fmt="png"):
        """File > Automate > Batch: open each file, play the action, export
        it to out_dir in fmt, close it. Returns [(source, output or error)]."""
        os.makedirs(out_dir, exist_ok=True)
        report = []
        for path in paths:
            try:
                self.open_path(path)
                self.play_action(action)
                base = os.path.splitext(os.path.basename(path))[0]
                target = os.path.join(out_dir, "%s.%s" % (base, fmt))
                if fmt == "ochre":
                    save(self.doc, target)
                else:
                    export_flat(self.doc, target)
                report.append((path, target))
            except Exception as exc:                       # noqa: BLE001
                report.append((path, "error: %s" % exc))
            finally:
                if self.doc is not None and len(self.documents) > 1:
                    self.close_document()
        return report

    def add_fill_layer(self, fill="solid", name=None, **params):
        """A layer that generates its content: a solid colour or a gradient."""
        from ..engine.layer import FILL
        label = "New Fill Layer"
        self.history.begin_group()
        layer = self.add_layer(name or ("Color Fill" if fill == "solid" else "Gradient Fill"))
        layer.kind = FILL
        p = {"type": fill}
        if fill == "solid":
            p["color"] = tuple(params.get("color", self.primary))
        else:
            p["start"] = tuple(params.get("start", self.primary))
            p["end"] = tuple(params.get("end", self.secondary))
            p["angle"] = float(params.get("angle", 90))
        layer.params = p
        if self.selection.mask is not None:
            self.add_layer_mask("selection", layer)
        self.history.end_group(label)
        self.editing_mask = False
        self._after_structure_change()
        return layer

    def set_layer_params(self, layer, params):
        """Re-edit an adjustment or fill layer."""
        merged = dict(layer.params)
        merged.update(params)
        return self.set_layer_property(layer, "params", merged)

    def set_layer_styles(self, layer, styles):
        """Layer > Layer Style. {name: params}; params may carry enabled=False."""
        clean = {k: dict(v) for k, v in (styles or {}).items() if v}
        return self.set_layer_property(layer, "styles", clean)

    def rasterize_layer(self, layer=None):
        """Bake a layer's styles, mask and generated content into plain pixels."""
        from ..engine.commands import MaskDelta, PixelDelta, PropertyDelta
        from ..engine.compositor import _Leaf
        target = layer or self.active_layer
        if target is None or target.is_group or getattr(target, "kind", "raster") == "adjustment":
            return False
        label = "Rasterize Layer"
        self.history.begin_group()
        for frame in self.doc.frames:
            if getattr(target, "kind", "raster") == "raster" and \
                    (target.id, frame.id) not in self.doc.cells:
                continue
            cell = self.doc.cell(target, frame)
            leaf = _Leaf(cell if getattr(target, "kind", "raster") == "raster" else None,
                         self.blends.opcode("Normal"), 255, target)
            out = np.zeros((self.doc.height, self.doc.width, 4), np.uint8)
            self.compositor._blend_leaf(out, leaf, self.doc.bounds, frame)
            delta = PixelDelta.capture(self.doc, target.id, frame.id, self.doc.bounds, label)
            cell.plane("rgba")[...] = out
            cell._touched(self.doc.bounds)
            if delta is not None:
                self.history.push(delta, label, frame_id=frame.id)
        for attr, empty in (("styles", {}), ("kind", "raster"), ("params", {})):
            if getattr(target, attr) != empty:
                self.history.push(PropertyDelta(target.id, attr, getattr(target, attr)), label)
                setattr(target, attr, empty)
        if target.mask is not None:
            self.history.push(MaskDelta.capture(self.doc, target, label), label)
            for key in [k for k in self.doc.cells if k[0] == target.mask.id]:
                del self.doc.cells[key]
            target.mask = None
        self.history.end_group(label)
        self.editing_mask = False
        self._after_structure_change()
        return True

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
        plain = (getattr(upper, "kind", "raster") == "raster" and upper.mask is None
                 and not upper.styles and not upper.clip)
        rects = transform.merge_rects(self.doc, upper, lower)
        if not plain:
            # An adjustment, a fill or a style reaches past the upper layer's
            # own pixels, so the whole canvas is captured for undo.
            rects = {f.id: self.doc.bounds for f in self.doc.frames}
        self.history.begin_group()
        for frame_id, rect in rects.items():
            self.doc.cell(lower, self.doc.frame_by_id(frame_id))
            delta = PixelDelta.capture(self.doc, lower.id, frame_id, rect, "Merge down")
            if delta is not None:
                self.history.push(delta, "Merge down", frame_id=frame_id)
        self.history.push(StructureDelta.removal(self.doc, upper, "Merge down"), "Merge down")
        self.history.end_group("Merge down")
        if plain:
            transform.merge_down(self.doc, upper, lower, self.blends)
        else:
            # Whatever the upper layer LOOKS like -- mask, styles, adjustment,
            # fill, clipping -- is what lands on the lower one.
            from ..engine.compositor import _Leaf
            for frame in self.doc.frames:
                cell = self.doc.cell(lower, frame)
                window = cell.plane("rgba").copy()
                usurf = self.doc.cells.get((upper.id, frame.id))
                leaf = _Leaf(usurf, self.blends.opcode(upper.blend), upper.opacity, upper)
                if usurf is None and getattr(upper, "kind", "raster") == "raster":
                    continue
                self.compositor._blend_leaf(window, leaf, self.doc.bounds, frame)
                if cell.index_locked:
                    transform.composite_onto(cell, window, self.doc.bounds,
                                             self.blends.opcode("Normal"), 255)
                else:
                    cell.plane("rgba")[...] = window
                    cell._touched(self.doc.bounds)
            self.doc.remove_layer(upper)
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

    def content_aware_scale(self, width, height, protect=True):
        """Image > Content-Aware Scale: resize by seam carving. The seams are
        found on the merged image (the selection, if any, protected) and the
        same seams are taken from every layer, mask and frame."""
        from ..engine import seamcarve, transform
        from ..engine.geometry import Rect
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == (self.doc.width, self.doc.height):
            return False
        merged = self.merged().copy()
        prot = self.selection.mask if protect and self.selection.mask is not None else None
        cols, rows_t = seamcarve.maps(merged, width, height, prot)
        sx, sy = width / float(self.doc.width), height / float(self.doc.height)

        def run():
            # Paths, marks and text move proportionally (seams are not
            # uniform, so this is the honest approximation for outlines).
            transform._map_paths(self.doc, lambda p: (p[0] * sx, p[1] * sy))
            transform._map_text(self.doc, lambda p: (p[0] * sx, p[1] * sy), (sx, sy))
            return transform._rebuild(self.doc, width, height,
                                      lambda s, n, p: seamcarve.apply_maps(p, cols, rows_t),
                                      lambda r: Rect(0, 0, width, height))
        return self._geometry("Content-Aware Scale", run)

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
        tool = self.tool
        if tool is not None and tool.active and getattr(tool, "_edit", None) is not None \
                and hasattr(tool, "set_colour") and getattr(self, "_ctx", None) is not None:
            # Picking a colour while editing a text layer recolours the text.
            rect = tool.set_colour(self._ctx, self.primary)
            if rect is not None:
                self.invalidate(rect)
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
            # Mid-edit tools (a text layer being typed) redraw for the change.
            if self.tool.active and hasattr(self.tool, "options_changed") \
                    and getattr(self, "_ctx", None) is not None:
                rect = self.tool.options_changed(self._ctx)
                if rect is not None:
                    self.invalidate(rect)
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

    @property
    def paint_target(self):
        """What the tools write to: the active layer, or its mask."""
        layer = self.active_layer
        if self.editing_mask and layer is not None and getattr(layer, "mask", None) is not None:
            return layer.mask
        return layer

    def _context(self):
        ctx = ToolContext(self.doc, self.paint_target, self.doc.frame,
                          self.selection, self.primary, self.secondary,
                          self.settings, self.history)
        ctx.primary_index = self.primary_index
        ctx.secondary_index = self.secondary_index
        ctx.clip = self.clip
        ctx.history_source = self.history_source
        ctx.pick_radius = self.pick_radius
        ctx.composite = self.merged
        return ctx

    def merged(self):
        """The merged image as displayed, brought up to date."""
        self.flush()
        return self.display

    # ---- snapping -------------------------------------------------------------

    def snap_point(self, x, y):
        """(x, y) snapped to guides / grid / canvas edges, as switched on."""
        from ..engine import annotations
        guides = self.doc.guides if self.snap_guides else ()
        grid = self.grid if self.snap_grid else None
        bounds = (self.doc.width, self.doc.height) if self.snap_bounds else None
        if not guides and grid is None and bounds is None:
            return x, y
        # Twice the pick radius: a few screen pixels, whatever the zoom.
        sx, sy, _a, _b = annotations.snap(x, y, guides, grid, bounds,
                                          tolerance=2.0 * self.pick_radius)
        return sx, sy

    def _snapped(self, x, y):
        tool = self.tool
        if tool is None or not getattr(tool, "snaps", True) or tool.affects != "pixels":
            return x, y
        return self.snap_point(x, y)

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

    def text_layer_at(self, x, y):
        """The topmost visible text layer whose text is under (x, y)."""
        from ..engine import textlayer
        pad = max(4.0, 2.0 * self.pick_radius)
        for layer in reversed(self.doc.layers()):
            if getattr(layer, "kind", "raster") == "text" and layer.effectively_visible() \
                    and textlayer.contains(layer.params, x, y, pad):
                return layer
        return None

    def _prepare_text_layer(self, x, y):
        """Text tool press: edit the text layer clicked, or start a new one
        above the active layer. True when a new one was made."""
        from ..engine.layer import TEXT
        hit = self.text_layer_at(x, y)
        if hit is not None:
            if hit is not self.active_layer:
                self.set_active_layer(hit)
            return False
        active = self.active_layer
        parent, index = None, None
        if active is not None and active.parent is not None:
            parent = active.parent if active.parent is not self.doc.root else None
            index = active.parent.index_of(active) + 1
        # Made outside history: committing the text records the layer and
        # its text as ONE undo step, and an empty click leaves no trace.
        self._text_prev = active
        layer = self.doc.add_layer("Text", parent=parent, index=index)
        layer.kind = TEXT
        layer.params = {}
        self.set_active_layer(layer)
        self.compositor.invalidate()
        self.emit("layers.changed")
        return True

    def _repair_active(self):
        """A tool may remove the layer it was working on (an empty new text
        layer, cancelled or committed blank): go back to the one before."""
        if self.active_layer is None or self.doc.layer(self.active_layer.id) is not None:
            return False
        prev = getattr(self, "_text_prev", None)
        if prev is not None and self.doc.layer(prev.id) is not None:
            self.set_active_layer(prev)
        else:
            leaves = self.doc.layers()
            self.set_active_layer(leaves[-1] if leaves else None)
        self.compositor.invalidate()
        self.invalidate()
        self.emit("layers.changed")
        return True

    def begin_stroke(self, x, y, pressure=1.0, modifiers=0, button=1):
        if self.tool is None or self.active_layer is None:
            return None
        x, y = self._snapped(x, y)
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
        text_new = False
        if (getattr(self.tool, "name", "") == "text" and self.tool.option("layer", True)
                and not getattr(self.tool, "changes_selection", False) and not self.tool.active):
            text_new = self._prepare_text_layer(x, y)
        self._ctx = self._context()
        self._ctx.text_layer_new = text_new
        if self.tool.wants_stroke:
            self.compositor.build_cache(self.active_layer)
        rect = self.tool.begin(self._ctx, ToolEvent(x, y, pressure, modifiers, button))
        self._sync_from_context()
        self.dirty_since_save = True
        return rect

    def motion_stroke(self, x, y, pressure=1.0, modifiers=0, button=1):
        if self.tool is None or not self.tool.active:
            return None
        x, y = self._snapped(x, y)
        rect = self.tool.motion(self._ctx, ToolEvent(x, y, pressure, modifiers, button))
        self._sync_from_context()
        return rect

    def end_stroke(self, x, y, pressure=1.0, modifiers=0, button=1):
        if self.tool is None:
            return None
        x, y = self._snapped(x, y)
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
        if getattr(self.tool, "measurement", None) is not None:
            self.last_measure = self.tool.line
        # A tool that needs the window (the Note tool wants a text box) says
        # so through `request`; it is handed on once and cleared.
        req = getattr(self.tool, "request", None)
        if req is not None:
            self.tool.request = None
            self.emit("tool.request", request=req)
        self.emit("marks.changed")
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
        if self._repair_active():
            self.emit("history.changed")
        if getattr(tool, "name", "") == "text":
            self.compositor.invalidate()
            self.emit("layers.changed")
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
        self._repair_active()
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

    # ---- guides and marks -----------------------------------------------------

    def _marks_change(self, label, fn, attrs=None):
        from ..engine.commands import MARKS, MarksDelta
        before = MarksDelta(self.doc, label, attrs or MARKS)
        result = fn()
        if not before.same_as(self.doc):
            self.history.push(before, label)
            self.dirty_since_save = True
        self.emit("marks.changed")
        self.emit("history.changed")
        return result

    def add_guide(self, orientation, position):
        """A guide across the canvas: 'h' at a y, or 'v' at an x."""
        orientation = "h" if str(orientation).lower().startswith("h") else "v"
        pos = float(position)

        def run():
            self.doc.guides = list(self.doc.guides) + [(orientation, pos)]
            return len(self.doc.guides) - 1
        return self._marks_change("New Guide", run, ("guides",))

    def move_guide(self, index, position):
        if not (0 <= index < len(self.doc.guides)):
            return False
        limit = self.doc.height if self.doc.guides[index][0] == "h" else self.doc.width
        if position < 0 or position > limit:
            return self.remove_guide(index)     # dragged off the canvas

        def run():
            g = list(self.doc.guides)
            g[index] = (g[index][0], float(position))
            self.doc.guides = g
            return True
        return self._marks_change("Move Guide", run, ("guides",))

    def remove_guide(self, index):
        if not (0 <= index < len(self.doc.guides)):
            return False

        def run():
            self.doc.guides = self.doc.guides[:index] + self.doc.guides[index + 1:]
            return True
        return self._marks_change("Remove Guide", run, ("guides",))

    def clear_guides(self):
        def run():
            self.doc.guides = []
            return True
        return self._marks_change("Clear Guides", run, ("guides",))

    def guide_at(self, x, y, radius=None):
        """Index of the guide within `radius` document pixels of a point."""
        r = 2.0 * self.pick_radius if radius is None else radius
        best, idx = r, None
        for i, (o, p) in enumerate(self.doc.guides):
            d = abs((y if o == "h" else x) - p)
            if d <= best:
                best, idx = d, i
        return idx

    def add_note(self, x, y, text):
        def run():
            self.doc.notes = list(self.doc.notes) + [(float(x), float(y), str(text))]
            return len(self.doc.notes) - 1
        return self._marks_change("Note", run, ("notes",))

    def set_note_text(self, index, text):
        if not (0 <= index < len(self.doc.notes)):
            return False

        def run():
            notes = list(self.doc.notes)
            x, y, _t = notes[index]
            notes[index] = (x, y, str(text))
            self.doc.notes = notes
            return True
        return self._marks_change("Edit Note", run, ("notes",))

    def clear_marks(self, attr):
        def run():
            setattr(self.doc, attr, [])
            return True
        return self._marks_change("Clear " + attr.capitalize(), run, (attr,))

    def slices_from_guides(self):
        from ..engine import annotations

        def run():
            self.doc.slices = annotations.slices_from_guides(
                self.doc.guides, self.doc.width, self.doc.height)
            return len(self.doc.slices)
        return self._marks_change("Slices From Guides", run, ("slices",))

    def export_slices(self, directory, fmt="png"):
        """File > Export Slices: each slice of the merged image to its own file."""
        from PIL import Image
        os.makedirs(directory, exist_ok=True)
        image = self.merged()
        out = []
        for name, (x, y, w, h) in self.doc.slices:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(self.doc.width, x + w), min(self.doc.height, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "slice"
            target = os.path.join(directory, "%s.%s" % (safe, fmt))
            Image.fromarray(np.ascontiguousarray(image[y0:y1, x0:x1])).save(target)
            out.append(target)
        return out

    def straighten(self, line=None):
        """Image > Straighten: rotate so the measured line comes out level."""
        from ..engine import annotations
        line = line or self.last_measure
        if line is None:
            return None
        angle = annotations.straighten_angle(*line)
        if abs(angle) < 1e-6:
            return 0.0
        self.rotate_image_by(angle)
        self.last_measure = None
        if getattr(self.tool, "line", None) is not None:
            self.tool.line = None
        return angle

    def sample_at(self, x, y, size=1):
        """The merged colour around a document point, for the Info dock."""
        from ..engine import annotations
        return annotations.sample(self.merged(), x, y, size)

    def histogram(self, source="image", within_selection=True):
        """256-bin counts for R, G, B and luminance, alpha-weighted.

        Over the merged image or the active layer, and over the selection
        when there is one -- what GIMP's Histogram dialog reports.
        """
        if source == "layer" and self.active_layer is not None and not self.active_layer.is_group:
            rgba = self.doc.cell(self.active_layer).plane("rgba")
        else:
            rgba = self.merged()
        weight = rgba[..., 3].astype(np.float64) / 255.0
        if within_selection and self.selection is not None and self.selection.mask is not None:
            weight = weight * (self.selection.mask.astype(np.float64) / 255.0)
        w = weight.ravel()
        out = {}
        for name, ch in (("red", 0), ("green", 1), ("blue", 2)):
            out[name] = np.bincount(rgba[..., ch].ravel(), weights=w, minlength=256)
        lum = (0.2126 * rgba[..., 0] + 0.7152 * rgba[..., 1] + 0.0722 * rgba[..., 2])
        out["value"] = np.bincount(np.clip(np.rint(lum), 0, 255).astype(np.int64).ravel(),
                                   weights=w, minlength=256)
        total = float(w.sum())
        out["pixels"] = total
        if total > 0:
            levels = np.arange(256)
            v = out["value"]
            mean = float((levels * v).sum() / total)
            out["mean"] = mean
            out["std"] = float(np.sqrt(((levels - mean) ** 2 * v).sum() / total))
            c = np.cumsum(v)
            out["median"] = int(np.searchsorted(c, total / 2.0))
        else:
            out["mean"] = out["std"] = 0.0
            out["median"] = 0
        return out

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



# ---- action recording -----------------------------------------------------------
# The commands an Action may record and replay: menu-level operations whose
# arguments are plain data. Tool strokes are not among them (Photoshop
# records very few), and nothing outside this list can be invoked by an
# actions file, which is data.
RECORDABLE = (
    "apply_effect", "resize_image", "resize_canvas", "crop_to_selection", "flip_image",
    "rotate_image", "rotate_image_by", "content_aware_scale", "straighten",
    "fill_selection", "erase_selection", "inpaint_selection",
    "select_all", "deselect", "invert_selection", "grow_selection", "shrink_selection",
    "border_selection", "feather_selection", "round_selection", "remove_selection_holes",
    "add_layer", "duplicate_layer", "merge_down", "flatten", "flip_layer", "rotate_layer",
    "scale_layer", "offset_layer", "align_layer", "set_primary", "set_secondary",
    "add_layer_mask", "delete_layer_mask", "add_guide", "clear_guides", "slices_from_guides",
    "save_selection", "load_selection",
)


def _plain(v):
    """Arguments as JSON-friendly data (tuples become lists)."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _tuples(v):
    """The reverse, for replay: lists of numbers become tuples (colours)."""
    if isinstance(v, dict):
        return {k: _tuples(x) for k, x in v.items()}
    if isinstance(v, list):
        if v and all(isinstance(x, (int, float)) for x in v):
            return tuple(v)
        return [_tuples(x) for x in v]
    return v


def _recorded(fn, name):
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        depth = getattr(self, "_rec_depth", 0)
        if depth == 0 and getattr(self, "recording", None) is not None:
            # Layer and object arguments are not recordable data; an action
            # always works on whatever is active when it plays.
            if any(hasattr(a, "id") for a in args) or any(hasattr(a, "id") for a in kwargs.values()):
                pass
            else:
                self._record(name, args, kwargs)
        self._rec_depth = depth + 1
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._rec_depth = depth
    return wrapper


for _name in RECORDABLE:
    if hasattr(Controller, _name):
        setattr(Controller, _name, _recorded(getattr(Controller, _name), _name))
