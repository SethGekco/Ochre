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
            doc = import_flat(path, settings=self.settings)
            self._adopt(doc, path=None)   # imported, not "the same file"
        return self.doc

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
        self.emit("document.saved", path=target)
        return target

    def _adopt(self, doc, path):
        self.doc = doc
        self.history = HistoryStack(self.settings)
        self.selection = Selection(doc.width, doc.height)
        self.compositor = Compositor(doc)
        self.display = np.zeros((doc.height, doc.width, 4), dtype=np.uint8)
        self.path = path
        self.dirty_since_save = False
        leaves = doc.layers()
        self.active_layer = leaves[-1] if leaves else None
        self._pending = [doc.bounds]
        if self.tool_name:
            self.set_tool(self.tool_name)
        self.emit("document.opened", document=doc, path=path)

    @property
    def title(self):
        name = os.path.basename(self.path) if self.path else "Untitled"
        return ("*" if self.dirty_since_save else "") + name

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

    def add_layer(self, name=None, parent=None, index=None):
        layer = self.doc.add_layer(name or "Layer %d" % (len(self.doc.layers()) + 1),
                                   parent=parent, index=index)
        self.doc.cell(layer)
        self.set_active_layer(layer)
        self._after_structure_change()
        return layer

    def add_group(self, name=None):
        group = self.doc.add_group(name or "Group")
        self._after_structure_change()
        return group

    def remove_layer(self, layer=None):
        target = layer or self.active_layer
        if target is None or len(self.doc.layers()) <= 1:
            return False
        self.doc.remove_layer(target)
        leaves = self.doc.layers()
        self.active_layer = leaves[-1] if leaves else None
        self._after_structure_change()
        return True

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
        parent = parent or self.doc.root
        parent.add(layer, index)
        self._after_structure_change()

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
        frame = self.doc.add_frame(index=self.doc.current + 1,
                                   copy_from=self.doc.frame if copy_current else None)
        self.dirty_since_save = True
        self.emit("frames.changed")
        return frame

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

    def set_tool(self, name, **options):
        tool = self.tools.create(name, **options)
        if tool is None:
            return None
        if self.tool is not None and self.tool.active:
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
        return ctx

    def begin_stroke(self, x, y, pressure=1.0, modifiers=0, button=1):
        if self.tool is None or self.active_layer is None:
            return None
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
        result = self.tool.end(self._ctx, ToolEvent(x, y, pressure, modifiers, button))
        self._sync_from_context()
        self.emit("history.changed")
        return result

    def cancel_stroke(self):
        if self.tool is None or not self.tool.active:
            return None
        rect = self.tool.cancel(self._ctx)
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

    def select_all(self):
        self.selection.select_all()
        self.emit("selection.changed")

    def deselect(self):
        self.selection.select_all()
        self.emit("selection.changed")

    def invert_selection(self):
        self.selection.invert()
        self.emit("selection.changed")

    # ---- history ---------------------------------------------------------

    def undo(self):
        rect = self.history.undo(self.doc)
        if rect is not None:
            self.compositor.invalidate()
            self.invalidate(rect)
            self.dirty_since_save = True
        self.emit("history.changed")
        return rect

    def redo(self):
        rect = self.history.redo(self.doc)
        if rect is not None:
            self.compositor.invalidate()
            self.invalidate(rect)
            self.dirty_since_save = True
        self.emit("history.changed")
        return rect

    def history_labels(self):
        return self.history.labels()

    def history_goto(self, index):
        rect = self.history.goto(index, self.doc)
        self.compositor.invalidate()
        self.invalidate()
        self.emit("history.changed")
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
