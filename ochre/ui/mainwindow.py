# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The main window: canvas, docks, menus, status bar."""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QStatusBar)

from ..engine import accel
from ..engine.fileio import supported_extensions
from .canvas import CanvasView
from .docks import (ColorsDock, FramesDock, HistoryDock, LayersDock,
                    PaletteDock, ToolsDock)
from .effectdialog import EffectDialog


class MainWindow(QMainWindow):
    def __init__(self, controller, settings, parent=None):
        super().__init__(parent)
        self.ctl = controller
        self.cfg = settings

        self.setWindowTitle(settings.get("Window", "Title") or "Ochre")
        self.resize(settings.get("Window", "Width") or 1280,
                    settings.get("Window", "Height") or 800)

        self.canvas = CanvasView(controller, settings, self)
        self.setCentralWidget(self.canvas)

        self.tools_dock = ToolsDock(controller, self,
                                    columns=settings.get("Toolbox", "Columns") or 2)
        self.layers_dock = LayersDock(controller, self)
        self.colors_dock = ColorsDock(controller, self)
        self.palette_dock = PaletteDock(controller, self)
        self.history_dock = HistoryDock(controller, self)
        self.frames_dock = FramesDock(controller, self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tools_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.layers_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.colors_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.palette_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.history_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.frames_dock)
        self.tabifyDockWidget(self.palette_dock, self.history_dock)

        # None = decide from the document; True/False = the user has said.
        # Only an explicit toggle sets it, never our own show and hide, so
        # auto behaviour survives until somebody actually disagrees with it.
        self.frames_pinned = None
        self.frames_dock.toggleViewAction().triggered.connect(
            lambda shown: setattr(self, "frames_pinned", bool(shown)))
        self.frames_dock.setVisible(False)

        from .recent import RecentFiles
        state_dir = getattr(controller.addons, "state_dir", None)
        self.recent = RecentFiles(state_dir) if state_dir else None
        self.setAcceptDrops(True)

        from .tooloptions import ToolOptionsBar
        self.options_bar = ToolOptionsBar(controller, self)
        self.addToolBar(Qt.TopToolBarArea, self.options_bar)

        self._build_menus()
        self._build_status()
        self._subscribe()

        controller.set_tool(settings.get("Tools", "DefaultTool") or "brush")
        controller.new_document()
        self.canvas.bind()
        self._sync_all()

    # ---- wiring ----------------------------------------------------------

    def _subscribe(self):
        bus = self.ctl.bus
        if bus is None:
            return
        bus.subscribe("layers.changed", self._on_layers)
        bus.subscribe("layer.active", lambda **k: self.layers_dock.sync())
        bus.subscribe("history.changed", lambda **k: (self.history_dock.sync(),
                                                       self._update_title()))
        bus.subscribe("document.saved", lambda **k: self._update_title())
        bus.subscribe("colour.changed", lambda **k: self.colors_dock.sync())
        bus.subscribe("palette.changed", lambda **k: self.palette_dock.sync())
        bus.subscribe("tool.changed", lambda **k: self._on_tool())
        # Size and other options change the footprint drawn under the cursor.
        bus.subscribe("tool.options", lambda **k: (self.options_bar.sync_values(**k),
                                                    self.canvas.apply_tool_cursor()))
        bus.subscribe("frames.changed", lambda **k: (self.frames_dock.sync(),
                                                     self._update_frames_visible()))
        bus.subscribe("frame.changed", lambda **k: self._on_frame())
        bus.subscribe("document.opened", lambda **k: self._sync_all())
        # A resize, crop or rotate -- or an undo of one -- replaced the
        # display buffer; the canvas must alias the new one.
        bus.subscribe("document.resized", lambda **k: (self.canvas.bind(keep_view=True),
                                                        self.canvas.refresh()))
        bus.subscribe("selection.changed", lambda **k: self.canvas.viewport().update())
        self.canvas.cursor_moved.connect(self._on_cursor)
        self.canvas.zoom_changed.connect(self.zoom_label.setText)

    def _on_tool(self, **_):
        self.tools_dock.sync()
        self.options_bar.rebuild()
        # The pointer belongs to the tool, so it changes with it -- including
        # the size of the footprint, since changing brush size is a tool
        # change too.
        self.canvas.apply_tool_cursor()

    def _on_frame(self, **_):
        self.frames_dock.sync()
        self._update_frames_visible()
        self.canvas.refresh()
        self.canvas.viewport().update()

    def _on_layers(self, **_):
        self.layers_dock.sync()
        self.canvas.refresh()

    def _sync_all(self):
        for dock in (self.tools_dock, self.layers_dock, self.colors_dock,
                     self.palette_dock, self.history_dock, self.frames_dock):
            dock.sync()
        self._update_frames_visible()
        self._update_title()

    def _update_frames_visible(self):
        """Show the frame strip only for documents that HAVE frames.

        A sprite sheet or an animated GIF needs it; a photograph does not,
        and a permanent one-cell strip along the bottom of the window is
        just a band of wasted height telling you there is one frame.

        Deliberately keyed on the frame count rather than on the file type,
        so an addon format with frames gets the strip without the window
        learning anything about that format -- and so a single-frame sprite
        is treated like what it is. `frames_pinned` lets the View menu
        override either way; once a user has an opinion, stop having one for
        them.
        """
        if getattr(self, "frames_pinned", None) is not None:
            self.frames_dock.setVisible(bool(self.frames_pinned))
            return
        doc = self.ctl.doc
        self.frames_dock.setVisible(bool(doc is not None and len(doc.frames) > 1))

    def _update_title(self):
        base = self.cfg.get("Window", "Title") or "Ochre"
        self.setWindowTitle("%s - %s" % (self.ctl.title, base))

    # ---- menus -----------------------------------------------------------

    def _act(self, menu, text, slot, shortcut=None):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menus(self):
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        self._act(file_menu, "&New", self.do_new, "Ctrl+N")
        self._act(file_menu, "&Open...", self.do_open, "Ctrl+O")
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self._rebuild_recent()
        self._act(file_menu, "&Save", self.do_save, "Ctrl+S")
        self._act(file_menu, "Save &As...", self.do_save_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        self._act(file_menu, "E&xit", self.close, "Ctrl+Q")

        edit_menu = bar.addMenu("&Edit")
        self._act(edit_menu, "&Undo", self.do_undo, "Ctrl+Z")
        self._act(edit_menu, "&Redo", self.do_redo, "Ctrl+Y")
        edit_menu.addSeparator()
        self._act(edit_menu, "Cu&t", self.do_cut, "Ctrl+X")
        self._act(edit_menu, "&Copy", self.do_copy, "Ctrl+C")
        self._act(edit_menu, "Copy &Merged", lambda: self.do_copy(merged=True), "Ctrl+Shift+C")
        self._act(edit_menu, "&Paste", self.do_paste, "Ctrl+V")
        self._act(edit_menu, "Paste Into New &Layer",
                  lambda: self.do_paste(new_layer=True), "Ctrl+Shift+V")
        self._act(edit_menu, "Paste Into New &Image", self.do_paste_new_image, "Ctrl+Alt+V")
        edit_menu.addSeparator()
        self._act(edit_menu, "Set &History Brush Source", lambda: self.ctl.set_history_source())
        edit_menu.addSeparator()
        self._act(edit_menu, "&Erase Selection", lambda: self.ctl.erase_selection(), "Del")
        self._act(edit_menu, "&Fill Selection", lambda: self.ctl.fill_selection(), "Backspace")
        edit_menu.addSeparator()

        sel_menu = bar.addMenu("&Select")
        self._act(sel_menu, "&All", self.do_select_all, "Ctrl+A")
        self._act(sel_menu, "&None", self.do_deselect, "Ctrl+D")
        self._act(sel_menu, "&Invert", self.do_invert, "Ctrl+I")
        sel_menu.addSeparator()
        self._act(sel_menu, "&Feather...", lambda: self._ask_radius(
            "Feather Selection", self.ctl.feather_selection, 5))
        self._act(sel_menu, "&Sharpen", lambda: self._sel(self.ctl.sharpen_selection))
        self._act(sel_menu, "S&hrink...", lambda: self._ask_radius(
            "Shrink Selection", self.ctl.shrink_selection, 4))
        self._act(sel_menu, "&Grow...", lambda: self._ask_radius(
            "Grow Selection", self.ctl.grow_selection, 4))
        self._act(sel_menu, "&Border...", lambda: self._ask_radius(
            "Border Selection", self.ctl.border_selection, 4))
        self._act(sel_menu, "Remove &Holes", lambda: self._sel(self.ctl.remove_selection_holes))
        self._act(sel_menu, "Rounded &Corners...", lambda: self._ask_radius(
            "Round Selection Corners", self.ctl.round_selection, 10))
        sel_menu.addSeparator()
        self._act(sel_menu, "Alpha to Se&lection", lambda: self._sel(self.ctl.alpha_to_selection))
        self.quick_mask_action = self._act(sel_menu, "Toggle &Quick Mask",
                                           self.do_quick_mask, "Q")
        self.quick_mask_action.setCheckable(True)

        view_menu = bar.addMenu("&View")
        self._act(view_menu, "Zoom &In", lambda: self.canvas.zoom_step(+1), "Ctrl++")
        self._act(view_menu, "Zoom &Out", lambda: self.canvas.zoom_step(-1), "Ctrl+-")
        self._act(view_menu, "&Actual Size", self.canvas.zoom_actual, "Ctrl+0")
        self._act(view_menu, "&Fit to Window", self.canvas.zoom_fit, "Ctrl+Shift+0")

        # Effects and adjustments, grouped exactly as the registry says --
        # so an addon's effect appears in the right menu without this file
        # knowing it exists.
        from ..engine.effects import EffectRegistry
        self.effects = EffectRegistry(self.ctl.db)
        groups = self.effects.grouped()
        for category in sorted(groups):
            menu = bar.addMenu("&" + category)
            for name in groups[category]:
                self._act(menu, self.effects.label(name),
                          lambda _c=False, n=name: self.run_effect(n))

        image_menu = bar.addMenu("&Image")
        self._act(image_menu, "&Crop to Selection", self.do_crop, "Ctrl+Shift+X")
        self._act(image_menu, "&Resize...", self.do_resize, "Ctrl+R")
        self._act(image_menu, "Canvas &Size...", self.do_canvas_size, "Ctrl+Shift+R")
        image_menu.addSeparator()
        self._act(image_menu, "Flip &Horizontal", lambda: self._geometry(self.ctl.flip_image, True))
        self._act(image_menu, "Flip &Vertical", lambda: self._geometry(self.ctl.flip_image, False))
        self._act(image_menu, "Rotate 90\u00b0 Clock&wise",
                  lambda: self._geometry(self.ctl.rotate_image, 1), "Ctrl+H")
        self._act(image_menu, "Rotate 90\u00b0 Counter-Clockwise",
                  lambda: self._geometry(self.ctl.rotate_image, 3), "Ctrl+G")
        self._act(image_menu, "Rotate &180\u00b0", lambda: self._geometry(self.ctl.rotate_image, 2), "Ctrl+J")
        image_menu.addSeparator()
        self._act(image_menu, "Bind &Grayscale Palette", self.do_bind_palette)
        self._act(image_menu, "Convert Layer to &Indexed...", self.do_convert_indexed)
        image_menu.addSeparator()
        self._act(image_menu, "Add &Frame", lambda: self.ctl.add_frame())
        self._act(image_menu, "&Duplicate Frame", lambda: self.ctl.add_frame(True))
        self._act(image_menu, "Delete Fra&me", lambda: self.ctl.remove_frame())

        tools_menu = bar.addMenu("&Tools")
        from . import icons as _icons
        for group, members in self.ctl.tools.groups():
            for name in members:
                act = self._act(tools_menu, self.ctl.tools.label(name),
                                lambda _c=False, n=name: self.ctl.set_tool(n))
                act.setIcon(_icons.tool_icon(self.ctl.tools.icon(name)))
                key = self.ctl.tools.shortcut(name)
                if key:
                    # Shown, not bound: the toolbox owns the key so that a
                    # repeated press can cycle tools sharing it.
                    act.setText(act.text() + "\t" + key)
            tools_menu.addSeparator()
        self._act(tools_menu, "&Decrease Size", lambda: self.step_size(-1), "[")
        self._act(tools_menu, "&Increase Size", lambda: self.step_size(+1), "]")
        tools_menu.addSeparator()
        self._act(tools_menu, "S&wap Colors", self.swap_colours, "X")
        self._act(tools_menu, "Default &Colors", self.default_colours, "D")

        addon_menu = bar.addMenu("A&ddons")
        self._act(addon_menu, "&Load Addons", self.do_load_addons)
        self._act(addon_menu, "&Manage Addons...", self.do_addon_status)
        self.addon_commands_menu = addon_menu.addMenu("&Commands")
        self.addon_commands_menu.setEnabled(False)

        layer_menu = bar.addMenu("&Layers")
        self._act(layer_menu, "&Add Layer", lambda: self.ctl.add_layer(), "Ctrl+Shift+N")
        self._act(layer_menu, "Add &Group", lambda: self.ctl.add_group())
        self._act(layer_menu, "D&uplicate Layer", lambda: self.ctl.duplicate_layer(), "Ctrl+Shift+D")
        self._act(layer_menu, "&Delete Layer", lambda: self.ctl.remove_layer())
        layer_menu.addSeparator()
        self._act(layer_menu, "Move Layer &Up", lambda: self.ctl.move_layer_by(+1), "Alt+Up")
        self._act(layer_menu, "Move Layer Do&wn", lambda: self.ctl.move_layer_by(-1), "Alt+Down")
        self._act(layer_menu, "&Merge Layer Down", lambda: self.ctl.merge_down(), "Ctrl+M")
        self._act(layer_menu, "&Flatten", lambda: self.ctl.flatten(), "Ctrl+Shift+F")
        layer_menu.addSeparator()
        self._act(layer_menu, "Flip &Horizontal", lambda: self.ctl.flip_layer(True))
        self._act(layer_menu, "Flip &Vertical", lambda: self.ctl.flip_layer(False))
        self._act(layer_menu, "&Rotate 180\u00b0", lambda: self.ctl.rotate_layer_180())
        layer_menu.addSeparator()
        self._act(layer_menu, "&Properties...", self.do_layer_properties, "F4")

    def _build_status(self):
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.pos_label = QLabel("-")
        self.zoom_label = QLabel("100%")
        self.backend_label = QLabel("accel: %s" % accel.BACKEND)
        for widget in (self.pos_label, self.zoom_label, self.backend_label):
            bar.addPermanentWidget(widget)

    def _on_cursor(self, x, y):
        text = "%d, %d" % (int(x), int(y))
        # Under the cursor, report the INDEX when there is one. That is the
        # number a spriter actually works in.
        cell = self.ctl.doc.cell(self.ctl.active_layer) if self.ctl.active_layer else None
        if cell is not None and cell.index_locked and cell.bounds.contains(int(x), int(y)):
            text += "   index %d" % int(cell.plane("index")[int(y), int(x)])
        self.pos_label.setText(text)

    # ---- commands --------------------------------------------------------

    def confirm_discard(self, action="continue"):
        """Ask before throwing away REAL unsaved work. True means go ahead.

        Never asks about a document nobody touched: an ignored untitled
        canvas, or one undone back to its last save, closes silently. This is
        the only prompt of its kind, and every path that replaces the
        document goes through it -- before it existed, opening a file or
        quitting discarded edits without a word.
        """
        if not self.ctl.is_modified():
            return True
        name = os.path.basename(self.ctl.path) if self.ctl.path else "Untitled"
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setText("Save changes to %s before you %s?" % (name, action))
        box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Save)
        choice = box.exec()
        if choice == QMessageBox.Save:
            before = self.ctl.path
            self.do_save()
            # Cancelling the Save As dialog, or a failed save, must not
            # then discard the work it was trying to protect.
            return not self.ctl.is_modified() or (before is None and self.ctl.path is not None)
        return choice == QMessageBox.Discard

    def closeEvent(self, event):
        if self.confirm_discard("closing"):
            self.shutdown()
            event.accept()
        else:
            event.ignore()

    def shutdown(self):
        """Stop everything that can call back into Python.

        Timers firing while the interpreter tears down -- after some of the
        objects their callbacks touch are already gone -- are the classic
        PySide crash at exit: intermittent, load-dependent, and invisible
        except as a non-zero exit code after everything else succeeded.
        """
        for timer_name in ("_timer", "_caret_timer", "_tick_timer"):
            timer = getattr(self.canvas, timer_name, None)
            if timer is not None:
                timer.stop()
        bus = getattr(self.ctl, "bus", None)
        if bus is not None and hasattr(bus, "clear"):
            bus.clear()

    def do_new(self):
        from .dialogs import NewImageDialog
        if not self.confirm_discard("starting a new image"):
            return
        dlg = NewImageDialog(self.ctl, self)
        if not dlg.exec():
            return
        w, h, background = dlg.result()
        self.ctl.new_document(w, h, background=background)
        self.canvas.bind()
        self._sync_all()

    def do_open(self):
        exts = " ".join("*.%s" % e for e in supported_extensions())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "Images (*.ochre %s);;All files (*)" % exts)
        if not path:
            return
        self.open_file(path)

    def do_save(self):
        if self.ctl.path is None:
            return self.do_save_as()
        try:
            self.ctl.save_path()
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(exc))
        self._update_title()

    def do_save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "untitled.ochre",
            "Ochre document (*.ochre);;PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        try:
            self.ctl.save_path(path)
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        if self.recent:
            self.recent.add(path)
            self._rebuild_recent()
        self._update_title()

    def run_effect(self, name):
        """Open a generated dialog for an effect and let it preview live."""
        if self.ctl.active_layer is None:
            return None
        effect = self.effects.create(name)
        if effect is None:
            return None
        dialog = EffectDialog(effect, self.ctl, self)
        dialog.exec()
        self.canvas.refresh()
        self.canvas.viewport().update()
        self._update_title()
        return effect

    def do_load_addons(self):
        """Load discovered addons and surface whatever they contributed."""
        loaded = self.ctl.load_addons()
        self._rebuild_addon_commands()
        crashed = self.ctl.addons.crashed_addon()
        note = ""
        if crashed:
            # A native crash is the one failure Python cannot contain, so at
            # least name the culprit instead of leaving a mystery.
            note = ("\n\nNote: %r was loading when Ochre last exited "
                    "unexpectedly." % crashed.get("id"))
        QMessageBox.information(
            self, "Addons",
            "Loaded %d addon(s).\n%s%s"
            % (len(loaded),
               "\n".join("  %s  %s" % (a.id, a.state) for a in
                          self.ctl.addons.addons.values()) or "  none found",
               note))
        self._sync_all()

    def _rebuild_addon_commands(self):
        self.addon_commands_menu.clear()
        commands = self.ctl.addons.registry.all("commands")
        self.addon_commands_menu.setEnabled(bool(commands))
        for name, provider in sorted(commands.items()):
            self._act(self.addon_commands_menu,
                      getattr(provider, "label", name),
                      lambda _c=False, n=name: self.run_addon_command(n))

    def run_addon_command(self, name):
        """Run an addon command inside the fault guard.

        A command that raises must not break the editor -- it costs the addon
        a fault, and enough faults disable it.
        """
        provider = self.ctl.addons.registry.get("commands", name)
        if provider is None:
            return None
        owner = self.ctl.addons.registry.owner("commands", name)
        with self.ctl.addons.guard(owner, "command %s" % name) as guard:
            result = provider.run(self.ctl.addons_host(owner))
            if result:
                QMessageBox.information(self, getattr(provider, "label", name),
                                        str(result))
            self.canvas.refresh()
            self.canvas.viewport().update()
            self._sync_all()
            return result
        if guard.failed:
            QMessageBox.warning(self, "Addon command failed",
                                "%r raised. See Manage Addons." % name)
        return None

    def do_addon_status(self):
        rows = self.ctl.addons.status()
        if not rows:
            QMessageBox.information(self, "Addons", "No addons found.")
            return
        lines = []
        for r in rows:
            line = "%-18s %-10s v%s" % (r["id"], r["state"], r["version"])
            if r["error"]:
                line += "\n    %s" % r["error"]
            lines.append(line)
        QMessageBox.information(self, "Addons", "\n".join(lines))

    def do_bind_palette(self):
        from ..engine.palette import grayscale
        self.ctl.doc.bind_palette(grayscale())
        self.palette_dock.sync()
        self.canvas.refresh()

    def do_convert_indexed(self):
        """Convert the active layer to indexed, and report what it cost."""
        from ..engine.palette import grayscale
        from ..engine.quantize import ORDERED, convert_layer_to_indexed

        if self.ctl.active_layer is None:
            return
        if self.ctl.doc.palette is None:
            self.ctl.doc.bind_palette(grayscale())
        report = convert_layer_to_indexed(self.ctl.doc, self.ctl.active_layer,
                                          dither=ORDERED)
        worst = max((r["max"] for r in report.values()), default=0)
        mean = max((r["mean"] for r in report.values()), default=0.0)
        self.ctl.compositor.invalidate()
        self.ctl.invalidate()
        self.canvas.refresh()
        self.canvas.viewport().update()
        self._sync_all()
        # A conversion is lossy; say how lossy rather than letting the user
        # discover it later.
        QMessageBox.information(
            self, "Converted to indexed",
            "Converted %d cell(s).\nMean channel error %.1f, worst %d."
            % (len(report), mean, worst))

    # ---- clipboard -------------------------------------------------------

    def _system_clipboard_set(self, clip):
        """Mirror a clip to the system clipboard as an image."""
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        if clip is None:
            return
        h, w = clip.rgba.shape[:2]
        buf = clip.rgba.tobytes()
        image = QImage(buf, w, h, w * 4, QImage.Format_RGBA8888).copy()
        self._clip_key = image.cacheKey()
        QApplication.clipboard().setImage(image)
        # setImage re-wraps the data, so remember the pixels we put there.
        self._clip_bytes = buf

    def _system_clipboard_get(self):
        """A Clip from the system clipboard, if it holds a FOREIGN image.

        Our own copy is recognised by its bytes and left alone, so the
        internal clip -- which may carry indices -- is what gets pasted.
        """
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        from ..engine.clipboard import Clip
        import numpy as np
        image = QApplication.clipboard().image()
        if image.isNull():
            return None
        rgba = image.convertToFormat(QImage.Format_RGBA8888)
        w, h = rgba.width(), rgba.height()
        ptr = rgba.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=rgba.sizeInBytes())
        stride = rgba.bytesPerLine()
        arr = arr.reshape(h, stride)[:, :w * 4].reshape(h, w, 4).copy()
        if getattr(self, "_clip_bytes", None) == arr.tobytes():
            return None
        return Clip.from_rgba(arr)

    def do_copy(self, merged=False):
        clip = self.ctl.copy(merged=merged)
        self._system_clipboard_set(clip)
        return clip

    def do_cut(self):
        clip = self.ctl.cut()
        self._system_clipboard_set(clip)
        self.canvas.refresh()
        return clip

    def _paste_target(self):
        """Where a paste lands: the top-left of what is on screen, so a
        paste is always visible, not off in a corner the user scrolled
        away from."""
        vp = self.canvas.viewport()
        rect = self.canvas.view.visible_doc_rect(vp.width(), vp.height())
        return rect

    def do_paste(self, new_layer=False):
        foreign = self._system_clipboard_get()
        clip = foreign or self.ctl.clip
        if clip is None:
            return None
        at = None
        if foreign is not None:
            vis = self._paste_target()
            at = (max(0, int(vis.x)), max(0, int(vis.y)))
        landed = self.ctl.paste(clip, at=at, new_layer=new_layer)
        self.canvas.refresh()
        self.canvas.viewport().update()
        return landed

    def do_paste_new_image(self):
        if not self.confirm_discard("pasting into a new image"):
            return
        foreign = self._system_clipboard_get()
        clip = foreign or self.ctl.clip
        if clip is None:
            return None
        self.ctl.paste_new_image(clip)
        self.canvas.bind()
        self._sync_all()

    # ---- image geometry --------------------------------------------------

    def _geometry(self, fn, *args):
        done = fn(*args)
        if done:
            self.canvas.bind(keep_view=True)
            self.canvas.refresh()
            self.canvas.viewport().update()
        return done

    def do_crop(self):
        return self._geometry(self.ctl.crop_to_selection)

    def do_resize(self):
        from .dialogs import ResizeDialog
        dlg = ResizeDialog(self.ctl, self)
        if dlg.exec():
            w, h, method = dlg.result()
            self._geometry(self.ctl.resize_image, w, h, method)

    def do_canvas_size(self):
        from .dialogs import CanvasSizeDialog
        dlg = CanvasSizeDialog(self.ctl, self)
        if dlg.exec():
            w, h, anchor = dlg.result()
            self._geometry(self.ctl.resize_canvas, w, h, anchor)

    def do_layer_properties(self):
        from .dialogs import LayerPropertiesDialog
        node = self.ctl.active_layer
        if node is None:
            return
        LayerPropertiesDialog(self.ctl, node, self).exec()

    # ---- recent files ----------------------------------------------------

    def _rebuild_recent(self):
        self.recent_menu.clear()
        items = self.recent.existing() if self.recent else []
        for path in items:
            self._act(self.recent_menu, path.replace("&", "&&"),
                      lambda _c=False, p=path: self.open_file(p))
        if items:
            self.recent_menu.addSeparator()
            self._act(self.recent_menu, "&Clear List", self._clear_recent)
        self.recent_menu.setEnabled(bool(items))

    def _clear_recent(self):
        if self.recent:
            self.recent.items = []
            self.recent.save()
        self._rebuild_recent()

    def open_file(self, path):
        """Open a path with error reporting. Shared by Open, Recent and drop."""
        if not self.confirm_discard("opening another file"):
            return False
        try:
            self.ctl.open_path(path)
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self, "Open failed", str(exc))
            if self.recent:
                self.recent.remove(path)
                self._rebuild_recent()
            return False
        if self.recent:
            self.recent.add(path)
            self._rebuild_recent()
        self.canvas.bind()
        self._sync_all()
        return True

    # ---- drag and drop ---------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(
                u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_file(url.toLocalFile())
                event.acceptProposedAction()
                return

    def _sel(self, fn, *args):
        fn(*args)
        self.canvas.viewport().update()

    def _ask_radius(self, title, fn, default):
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getInt(self, title, "Radius (pixels):", default, 1, 2000)
        if ok:
            self._sel(fn, value)

    def do_quick_mask(self):
        on = self.ctl.toggle_quick_mask()
        self.quick_mask_action.setChecked(on)
        self.statusBar().showMessage(
            "Quick Mask: paint white to select, black to deselect. Q again to finish."
            if on else "Quick Mask applied to the selection.", 6000)
        self.canvas.refresh()
        self.canvas.viewport().update()

    def step_size(self, direction):
        """[ and ]: change the active tool's size the way Photoshop does --
        by a step that grows with the size, so it is fine at 3 px and still
        useful at 300."""
        spec = {o.name: o for o in self.ctl.tools.spec(self.ctl.tool_name)}
        for key in ("size", "stroke"):
            if key in spec:
                cur = int(self.ctl.tool_option_values(self.ctl.tool_name).get(key, spec[key].default))
                step = 1 if cur < 10 else (5 if cur < 50 else (10 if cur < 200 else 25))
                return self.ctl.set_tool_option(key, cur + direction * step)
        return None

    def swap_colours(self):
        p, pi = self.ctl.primary, self.ctl.primary_index
        self.ctl.set_primary(self.ctl.secondary, self.ctl.secondary_index)
        self.ctl.set_secondary(p, pi)

    def default_colours(self):
        self.ctl.set_primary((0, 0, 0, 255))
        self.ctl.set_secondary((255, 255, 255, 255))

    def do_undo(self):
        self.ctl.undo()
        self.canvas.refresh()
        self.canvas.viewport().update()
        self._update_title()

    def do_redo(self):
        self.ctl.redo()
        self.canvas.refresh()
        self.canvas.viewport().update()
        self._update_title()

    def do_select_all(self):
        self.ctl.select_all()
        self.canvas.viewport().update()

    def do_deselect(self):
        self.ctl.deselect()
        self.canvas.viewport().update()

    def do_invert(self):
        self.ctl.invert_selection()
        self.canvas.viewport().update()
