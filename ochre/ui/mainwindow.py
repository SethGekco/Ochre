# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
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

        self.tools_dock = ToolsDock(controller, self)
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
        bus.subscribe("history.changed", lambda **k: self.history_dock.sync())
        bus.subscribe("colour.changed", lambda **k: self.colors_dock.sync())
        bus.subscribe("palette.changed", lambda **k: self.palette_dock.sync())
        bus.subscribe("tool.changed", lambda **k: self.tools_dock.sync())
        bus.subscribe("frames.changed", lambda **k: self.frames_dock.sync())
        bus.subscribe("frame.changed", lambda **k: self._on_frame())
        bus.subscribe("document.opened", lambda **k: self._sync_all())
        self.canvas.cursor_moved.connect(self._on_cursor)
        self.canvas.zoom_changed.connect(self.zoom_label.setText)

    def _on_frame(self, **_):
        self.frames_dock.sync()
        self.canvas.refresh()
        self.canvas.viewport().update()

    def _on_layers(self, **_):
        self.layers_dock.sync()
        self.canvas.refresh()

    def _sync_all(self):
        for dock in (self.tools_dock, self.layers_dock, self.colors_dock,
                     self.palette_dock, self.history_dock, self.frames_dock):
            dock.sync()
        self._update_title()

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
        self._act(file_menu, "&Save", self.do_save, "Ctrl+S")
        self._act(file_menu, "Save &As...", self.do_save_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        self._act(file_menu, "E&xit", self.close, "Ctrl+Q")

        edit_menu = bar.addMenu("&Edit")
        self._act(edit_menu, "&Undo", self.do_undo, "Ctrl+Z")
        self._act(edit_menu, "&Redo", self.do_redo, "Ctrl+Y")
        edit_menu.addSeparator()
        self._act(edit_menu, "Select &All", self.do_select_all, "Ctrl+A")
        self._act(edit_menu, "&Deselect", self.do_deselect, "Ctrl+D")
        self._act(edit_menu, "&Invert Selection", self.do_invert, "Ctrl+I")

        view_menu = bar.addMenu("&View")
        self._act(view_menu, "Zoom &In", self.canvas.view.zoom_in, "Ctrl++")
        self._act(view_menu, "Zoom &Out", self.canvas.view.zoom_out, "Ctrl+-")
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
        self._act(image_menu, "Bind &Grayscale Palette", self.do_bind_palette)
        self._act(image_menu, "Convert Layer to &Indexed...", self.do_convert_indexed)
        image_menu.addSeparator()
        self._act(image_menu, "Add &Frame", lambda: self.ctl.add_frame())
        self._act(image_menu, "&Duplicate Frame", lambda: self.ctl.add_frame(True))

        addon_menu = bar.addMenu("A&ddons")
        self._act(addon_menu, "&Load Addons", self.do_load_addons)
        self._act(addon_menu, "&Manage Addons...", self.do_addon_status)
        self.addon_commands_menu = addon_menu.addMenu("&Commands")
        self.addon_commands_menu.setEnabled(False)

        layer_menu = bar.addMenu("&Layers")
        self._act(layer_menu, "&Add Layer", lambda: self.ctl.add_layer(), "Ctrl+Shift+N")
        self._act(layer_menu, "Add &Group", lambda: self.ctl.add_group())
        self._act(layer_menu, "&Delete Layer", lambda: self.ctl.remove_layer())

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

    def do_new(self):
        self.ctl.new_document()
        self.canvas.bind()
        self._sync_all()

    def do_open(self):
        exts = " ".join("*.%s" % e for e in supported_extensions())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "Images (*.ochre %s);;All files (*)" % exts)
        if not path:
            return
        try:
            self.ctl.open_path(path)
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self, "Open failed", str(exc))
            return
        self.canvas.bind()
        self._sync_all()

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
