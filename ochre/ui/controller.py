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
        self.blends = self.compositor.blends
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
