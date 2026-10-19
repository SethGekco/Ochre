# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The main window: canvas, docks, menus, status bar."""

import os
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QStatusBar)

from ..engine import accel
from ..engine.fileio import supported_extensions
from .canvas import CanvasView
from .docks import (ColorsDock, FramesDock, HistoryDock, LayersDock,
                    PaletteDock, PathsDock, ToolsDock)
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
        # One tab per open document above the canvas, as Paint.NET and GIMP
        # (single-window mode) show them.
        from PySide6.QtWidgets import QTabBar, QVBoxLayout, QWidget
        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.doc_tabs = QTabBar(central)
        self.doc_tabs.setTabsClosable(True)
        self.doc_tabs.setDocumentMode(True)
        self.doc_tabs.setExpanding(False)
        self.doc_tabs.currentChanged.connect(self._tab_picked)
        self.doc_tabs.tabCloseRequested.connect(self.close_document)
        self._syncing_tabs = False
        lay.addWidget(self.doc_tabs)
        lay.addWidget(self.canvas, 1)
        self.setCentralWidget(central)

        self.tools_dock = ToolsDock(controller, self,
                                    columns=settings.get("Toolbox", "Columns") or 2)
        self.layers_dock = LayersDock(controller, self)
        self.colors_dock = ColorsDock(controller, self)
        self.palette_dock = PaletteDock(controller, self)
        self.history_dock = HistoryDock(controller, self)
        self.paths_dock = PathsDock(controller, self)
        from .panels import (ActionsDock, BrushPresetsDock, ChannelsDock, CompsDock,
                             HistogramDock, InfoDock, NavigatorDock)
        self.navigator_dock = NavigatorDock(controller, self.canvas, self)
        self.info_dock = InfoDock(controller, self)
        self.histogram_dock = HistogramDock(controller, self)
        self.actions_dock = ActionsDock(controller, self)
        self.channels_dock = ChannelsDock(controller, self.canvas, self)
        self.comps_dock = CompsDock(controller, self)
        self.brushes_dock = BrushPresetsDock(controller, self)
        self.frames_dock = FramesDock(controller, self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tools_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.layers_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.colors_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.palette_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.frames_dock)
        # One tab group for the reference panels, every one tabified onto the
        # same anchor in a single pass. Stacking them as separate groups made
        # the window at least 934 px tall, and building the group in steps
        # left Qt drawing a stale tab strip over the dock below.
        for dock in (self.history_dock, self.paths_dock, self.navigator_dock,
                     self.info_dock, self.histogram_dock, self.actions_dock,
                     self.channels_dock, self.comps_dock, self.brushes_dock):
            self.tabifyDockWidget(self.palette_dock, dock)
        self.history_dock.raise_()
        # Give the canvas the room. Left to itself Qt sizes the right-hand
        # column from the docks' preferred widths, which with a dozen panels
        # took 740 px of an 1100 px window. ui.ini [Window] DockWidth sets it.
        dock_w = int(settings.get("Window", "DockWidth") or 300)
        self.resizeDocks([self.layers_dock], [dock_w], Qt.Horizontal)

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
        self._build_window_menu()
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
        bus.subscribe("paths.changed", lambda **k: (self.paths_dock.sync(),
                                                    self.canvas.viewport().update()))
        bus.subscribe("marks.changed", lambda **k: (self.canvas.viewport().update(),
                                                    self.info_dock.schedule()))
        bus.subscribe("history.changed", lambda **k: (self.histogram_dock.schedule(),
                                                      self.navigator_dock.schedule(),
                                                      self.info_dock.schedule(),
                                                      self._sync_tabs()))
        bus.subscribe("documents.changed", lambda **k: self._sync_tabs())
        bus.subscribe("actions.changed", lambda **k: self.actions_dock.sync())
        bus.subscribe("channels.changed", lambda **k: self.channels_dock.sync())
        bus.subscribe("comps.changed", lambda **k: self.comps_dock.sync())
        bus.subscribe("brushes.changed", lambda **k: self.brushes_dock.sync())
        bus.subscribe("tool.request", self._on_tool_request)
        self.canvas.cursor_moved.connect(self.info_dock.pointer)
        self.canvas.cursor_moved.connect(self._on_cursor)
        self.canvas.zoom_changed.connect(self.zoom_label.setText)

    def _on_tool(self, **_):
        self.tools_dock.sync()
        self.brushes_dock.sync()
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
                     self.palette_dock, self.history_dock, self.frames_dock,
                     self.paths_dock, self.info_dock, self.histogram_dock,
                     self.navigator_dock, self.channels_dock, self.comps_dock):
            dock.sync()
        self._sync_tabs()
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

    def _build_window_menu(self):
        menu = self.menuBar().addMenu("&Window")
        for dock in (self.tools_dock, self.layers_dock, self.colors_dock, self.palette_dock,
                     self.history_dock, self.paths_dock, self.navigator_dock, self.info_dock,
                     self.histogram_dock, self.actions_dock, self.channels_dock,
                     self.comps_dock, self.brushes_dock, self.frames_dock):
            menu.addAction(dock.toggleViewAction())
        return menu

    # ---- documents (tabs) -------------------------------------------------

    def _sync_tabs(self):
        if not hasattr(self, "doc_tabs"):
            return
        titles = self.ctl.document_titles()
        self._syncing_tabs = True
        while self.doc_tabs.count() > len(titles):
            self.doc_tabs.removeTab(self.doc_tabs.count() - 1)
        for i, t in enumerate(titles):
            if i < self.doc_tabs.count():
                if self.doc_tabs.tabText(i) != t:
                    self.doc_tabs.setTabText(i, t)
            else:
                self.doc_tabs.addTab(t)
        if self.ctl.current_doc >= 0:
            self.doc_tabs.setCurrentIndex(self.ctl.current_doc)
        self._syncing_tabs = False

    def _tab_picked(self, index):
        if self._syncing_tabs or index < 0 or index == self.ctl.current_doc:
            return
        self.switch_document(index)

    def switch_document(self, index):
        if self.ctl.switch_document(index):
            self.canvas.bind()
            self._sync_all()
            return True
        return False

    def _step_document(self, delta):
        n = len(self.ctl.documents)
        if n > 1:
            self.switch_document((self.ctl.current_doc + delta) % n)

    def close_document(self, index=None):
        """Close a tab, asking first only if it holds unsaved work."""
        index = self.ctl.current_doc if index is None else index
        if index != self.ctl.current_doc:
            self.switch_document(index)
        if not self.confirm_discard("closing it"):
            return False
        self.ctl.close_document()
        self.canvas.bind()
        self._sync_all()
        return True

    # ---- marks, guides, notes ----------------------------------------------

    def _on_tool_request(self, request=None, **_):
        """A tool needs the window -- today, the Note tool wants text."""
        from PySide6.QtWidgets import QInputDialog
        if not request:
            return
        kind, arg = request
        if kind == "new_note":
            text, ok = QInputDialog.getMultiLineText(self, "Note", "Note text:")
            if ok and text.strip():
                self.ctl.add_note(arg[0], arg[1], text)
        elif kind == "edit_note" and 0 <= arg < len(self.ctl.doc.notes):
            text, ok = QInputDialog.getMultiLineText(self, "Note", "Note text:",
                                                     self.ctl.doc.notes[arg][2])
            if ok:
                self.ctl.set_note_text(arg, text)
        self.canvas.viewport().update()

    def do_new_guide(self):
        from PySide6.QtWidgets import QInputDialog
        orient, ok = QInputDialog.getItem(self, "New Guide", "Orientation:",
                                          ["Horizontal", "Vertical"], 0, False)
        if not ok:
            return
        limit = self.ctl.doc.height if orient == "Horizontal" else self.ctl.doc.width
        pos, ok = QInputDialog.getDouble(self, "New Guide", "Position (pixels):",
                                         limit / 2.0, 0, limit, 1)
        if ok:
            self.ctl.add_guide("h" if orient == "Horizontal" else "v", pos)
            self.canvas.viewport().update()

    def do_grid_settings(self):
        from PySide6.QtWidgets import QInputDialog
        size, ok = QInputDialog.getInt(self, "Grid", "Grid spacing (pixels):",
                                       int(self.ctl.grid[0]), 1, 4096)
        if ok:
            self.ctl.grid = (size, size)
            self.canvas.viewport().update()

    def do_straighten(self):
        if self.ctl.last_measure is None:
            self.statusBar().showMessage(
                "Draw a line along something that should be level with the Measure tool first.", 6000)
            return
        self._busy(self.ctl.straighten)
        self.canvas.bind(keep_view=True)

    def do_match_color(self):
        from PySide6.QtWidgets import QInputDialog
        titles = self.ctl.document_titles()
        others = [(i, t) for i, t in enumerate(titles) if i != self.ctl.current_doc]
        if not others:
            self.statusBar().showMessage("Open the image whose colours you want to match first.", 6000)
            return
        label, ok = QInputDialog.getItem(self, "Match Color", "Take colours from:",
                                         ["%d: %s" % (i + 1, t) for i, t in others], 0, False)
        if not ok:
            return
        amount, ok = QInputDialog.getInt(self, "Match Color", "Strength (%):", 100, 0, 100)
        if ok:
            self._busy(self.ctl.match_color, int(label.split(":")[0]) - 1, amount / 100.0)

    def do_batch(self):
        """File > Automate > Batch: an action over every image in a folder."""
        from PySide6.QtWidgets import QInputDialog
        names = sorted(self.ctl.actions())
        if not names:
            self.statusBar().showMessage("Record an action first (Window > Actions).", 6000)
            return
        action, ok = QInputDialog.getItem(self, "Batch", "Action:", names, 0, False)
        if not ok:
            return
        src = QFileDialog.getExistingDirectory(self, "Batch: images from")
        if not src:
            return
        dst = QFileDialog.getExistingDirectory(self, "Batch: save results to")
        if not dst:
            return
        fmt, ok = QInputDialog.getItem(self, "Batch", "Save as:",
                                       ["png", "jpg", "webp", "tif", "ochre"], 0, False)
        if not ok:
            return
        exts = tuple("." + e for e in supported_extensions()) + (".ochre", ".psd", ".ora")
        paths = sorted(os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(exts))
        report = self._busy(self.ctl.batch, action, paths, dst, fmt)
        errors = [r for r in report if str(r[1]).startswith("error")]
        self.canvas.bind()
        self._sync_all()
        QMessageBox.information(self, "Batch", "Processed %d image(s)%s."
                                % (len(report) - len(errors),
                                   (", %d failed" % len(errors)) if errors else ""))

    def do_load_selection(self):
        from PySide6.QtWidgets import QInputDialog
        names = [c.name for c in self.ctl.doc.channels]
        if not names:
            self.statusBar().showMessage("No saved selections: Select > Save Selection first.", 5000)
            return
        name, ok = QInputDialog.getItem(self, "Load Selection", "Channel:", names, 0, False)
        if ok:
            self._sel(self.ctl.load_selection, names.index(name))

    def _slider_dialog(self, title, fields, on_change=None):
        """A small dialog of labelled integer sliders. Returns values or None."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QSlider
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        form = QFormLayout(dlg)
        sliders = {}
        for key, label, lo, hi, val in fields:
            sl = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(val)
            lab = QLabel(str(val))
            sl.valueChanged.connect(lambda v, l=lab: l.setText(str(v)))
            if on_change is not None:
                sl.valueChanged.connect(lambda _v: on_change({k: x.value() for k, x in sliders.items()}))
            row = QHBoxLayout()
            row.addWidget(sl, 1)
            row.addWidget(lab)
            form.addRow(label, row)
            sliders[key] = sl
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if not dlg.exec():
            return None
        return {k: x.value() for k, x in sliders.items()}

    def do_refine_edge(self):
        """Select > Refine Edge, previewed on the marching ants as you drag."""
        if self.ctl.selection.mask is None:
            self.statusBar().showMessage("Make a rough selection first, then refine it.", 5000)
            return
        from ..engine import refine
        original = self.ctl.selection.mask.copy()
        image = self.ctl.merged().copy()

        def preview(v):
            self.ctl.selection.mask = refine.refine_mask(image, original, v["radius"], v["smooth"],
                                                         v["feather"], v["contrast"], v["shift"])
            self.ctl.selection._bbox = None
            self.canvas.viewport().update()
        values = self._slider_dialog("Refine Edge", (
            ("radius", "Edge detection radius", 0, 50, 4), ("smooth", "Smooth", 0, 100, 0),
            ("feather", "Feather", 0, 50, 0), ("contrast", "Contrast", 0, 100, 0),
            ("shift", "Shift edge", -100, 100, 0)), preview)
        self.ctl.selection.mask = original
        self.ctl.selection._bbox = None
        if values is not None:
            self._sel(self.ctl.refine_selection, values["radius"], values["smooth"],
                      values["feather"], values["contrast"], values["shift"])
        self.canvas.viewport().update()

    def do_sky_replacement(self):
        from PySide6.QtWidgets import QInputDialog
        titles = self.ctl.document_titles()
        choices = ["Clouds", "Blue sky", "Sunset", "Overcast"] + [
            "Image: %d: %s" % (i + 1, t) for i, t in enumerate(titles) if i != self.ctl.current_doc]
        pick, ok = QInputDialog.getItem(self, "Sky Replacement", "Sky:", choices, 0, False)
        if not ok:
            return
        shift, ok = QInputDialog.getInt(self, "Sky Replacement",
                                        "Shift edge (more sky +, less -):", 0, -100, 100)
        if not ok:
            return
        presets = {"Clouds": "clouds", "Blue sky": "blue", "Sunset": "sunset", "Overcast": "overcast"}
        source = presets.get(pick)
        if source is None:
            source = int(pick.split(":")[1]) - 1
        layer = self._busy(self.ctl.sky_replace, source, 3.0, shift)
        if layer is None:
            self.statusBar().showMessage("No sky found -- try a positive edge shift.", 6000)
        self.layers_dock.sync()

    def do_merge_hdr(self):
        n = len(self.ctl.documents)
        if n < 2:
            self.statusBar().showMessage("Open the bracketed exposures as images first.", 6000)
            return
        try:
            self._busy(self.ctl.merge_hdr, list(range(n)))
        except ValueError as exc:
            QMessageBox.warning(self, "Merge to HDR", str(exc))
            return
        self.canvas.bind()
        self._sync_all()

    def do_export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "untitled.pdf", "PDF (*.pdf)")
        if path:
            self.ctl.export_pdf(path)

    def do_print(self):
        """Print the merged image, fitted to the page."""
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter
        printer = QPrinter(QPrinter.HighResolution)
        if QPrintDialog(printer, self).exec():
            self.print_to(printer)

    def print_to(self, printer):
        from PySide6.QtGui import QImage, QPainter
        img = np.ascontiguousarray(self.ctl.merged())
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 4,
                      QImage.Format_RGBA8888)
        painter = QPainter(printer)
        page = painter.viewport()
        size = qimg.size()
        size.scale(page.size(), Qt.KeepAspectRatio)
        painter.setViewport(page.x(), page.y(), size.width(), size.height())
        painter.setWindow(qimg.rect())
        painter.drawImage(0, 0, qimg)
        painter.end()

    def do_liquify(self):
        self.ctl.set_tool("warp")
        self.statusBar().showMessage(
            "Liquify: paint with the Warp tool -- Move pushes, Grow bloats, Shrink "
            "puckers, Swirl twirls, Erase restores. Set the mode and size above.", 10000)

    def do_content_aware_scale(self):
        from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
                                       QSpinBox)
        dlg = QDialog(self)
        dlg.setWindowTitle("Content-Aware Scale")
        form = QFormLayout(dlg)
        w = QSpinBox()
        w.setRange(1, 20000)
        w.setValue(self.ctl.doc.width)
        h = QSpinBox()
        h.setRange(1, 20000)
        h.setValue(self.ctl.doc.height)
        protect = QCheckBox("Protect the selection")
        protect.setChecked(self.ctl.selection.mask is not None)
        protect.setEnabled(self.ctl.selection.mask is not None)
        form.addRow("Width", w)
        form.addRow("Height", h)
        form.addRow("", protect)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec():
            self._busy(self._geometry_done, self.ctl.content_aware_scale,
                       w.value(), h.value(), protect.isChecked())

    def _geometry_done(self, fn, *args):
        done = fn(*args)
        if done:
            self.canvas.bind(keep_view=True)
            self.canvas.refresh()
        return done

    def do_export_slices(self):
        if not self.ctl.doc.slices:
            self.statusBar().showMessage(
                "No slices: draw some with the Slice tool, or use Image > Slices From Guides.", 6000)
            return
        folder = QFileDialog.getExistingDirectory(self, "Export Slices To")
        if not folder:
            return
        written = self.ctl.export_slices(folder)
        self.statusBar().showMessage("Exported %d slices to %s" % (len(written), folder), 6000)

    def _busy(self, fn, *args, **kw):
        """Run something slow under a wait cursor, then repaint."""
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return fn(*args, **kw)
        finally:
            QApplication.restoreOverrideCursor()
            self.canvas.refresh()

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
        self._act(file_menu, "Place &Embedded...", self.do_place_embedded)
        self._act(file_menu, "Place &Linked...", self.do_place_linked)
        automate = file_menu.addMenu("Au&tomate")
        automate.addAction("&Photomerge...", self.do_photomerge)
        automate.addAction("Load Files into &Stack...", self.do_load_stack)
        self._act(file_menu, "&Save", self.do_save, "Ctrl+S")
        self._act(file_menu, "Save &As...", self.do_save_as, "Ctrl+Shift+S")
        self._act(file_menu, "&Export As...", self.do_export_as, "Ctrl+Alt+Shift+W")
        self._act(file_menu, "Export S&lices...", self.do_export_slices)
        self._act(file_menu, "Export Art&boards...", self.do_export_artboards)
        automate = file_menu.addMenu("A&utomate")
        self._act(automate, "&Batch...", self.do_batch)
        self._act(automate, "Merge to &HDR...", self.do_merge_hdr)
        self._act(file_menu, "Export P&DF...", self.do_export_pdf)
        self._act(file_menu, "&Print...", self.do_print, "Ctrl+P")
        file_menu.addSeparator()
        self._act(file_menu, "&Close", lambda: self.close_document(), "Ctrl+W")
        self._act(file_menu, "Next Image", lambda: self._step_document(+1), "Ctrl+Tab")
        self._act(file_menu, "Previous Image", lambda: self._step_document(-1), "Ctrl+Shift+Tab")
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
        self._act(edit_menu, "Paste in Pe&rspective", self.do_paste_in_perspective)
        self._act(edit_menu, "Delete Perspective Plane", lambda: (
            self.ctl.clear_planes(), self.canvas.viewport().update()))
        edit_menu.addSeparator()
        self._act(edit_menu, "Set &History Brush Source", lambda: self.ctl.set_history_source())
        edit_menu.addSeparator()
        self._act(edit_menu, "&Erase Selection", lambda: self.ctl.erase_selection(), "Del")
        self._act(edit_menu, "&Fill Selection", lambda: self.ctl.fill_selection(), "Backspace")
        self._act(edit_menu, "Content-&Aware Fill",
                  lambda: self._busy(self.ctl.inpaint_selection, "content_aware"),
                  "Shift+Backspace")
        self._act(edit_menu, "S&mooth Fill (Inpaint)",
                  lambda: self._busy(self.ctl.inpaint_selection, "smooth"))
        self._act(edit_menu, "F&ill...", self.do_fill_dialog, "Shift+F5")
        self._act(edit_menu, "Stro&ke...", self.do_stroke_dialog)
        self._act(edit_menu, "Define &Pattern...", self.do_define_pattern)
        self._act(edit_menu, "Define &Brush Tip", self.do_define_brush_tip)
        edit_menu.addSeparator()
        self._act(edit_menu, "Auto-Align La&yers...", self.do_auto_align)
        self._act(edit_menu, "Auto-Blend Layers...", self.do_auto_blend)
        edit_menu.addSeparator()
        self._act(edit_menu, "Sk&y Replacement...", self.do_sky_replacement)
        edit_menu.addSeparator()
        self._act(edit_menu, "Fill &Path", lambda: self.ctl.fill_path())
        self._act(edit_menu, "Stro&ke Path...", self.do_stroke_path)
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
        self._act(sel_menu, "From &Path", lambda: self._sel(self.ctl.path_to_selection), "Shift+V")
        self._act(sel_menu, "To Pa&th", lambda: self.ctl.selection_to_path())
        sel_menu.addSeparator()
        self._act(sel_menu, "Refine &Edge...", self.do_refine_edge)
        self._act(sel_menu, "S&ky", lambda: self._sel(self.ctl.select_sky))
        self._act(sel_menu, "Color &Range...", self.do_color_range)
        self._act(sel_menu, "&Focus Area...", self.do_focus_area)
        sel_menu.addSeparator()
        self._act(sel_menu, "Sa&ve Selection", lambda: self.ctl.save_selection())
        self._act(sel_menu, "Loa&d Selection...", self.do_load_selection)
        self.quick_mask_action = self._act(sel_menu, "Toggle &Quick Mask",
                                           self.do_quick_mask, "Q")
        self.quick_mask_action.setCheckable(True)

        view_menu = bar.addMenu("&View")
        self._act(view_menu, "Zoom &In", lambda: self.canvas.zoom_step(+1), "Ctrl++")
        self._act(view_menu, "Zoom &Out", lambda: self.canvas.zoom_step(-1), "Ctrl+-")
        self._act(view_menu, "&Actual Size", self.canvas.zoom_actual, "Ctrl+0")
        self._act(view_menu, "&Fit to Window", self.canvas.zoom_fit, "Ctrl+Shift+0")
        view_menu.addSeparator()
        self._act(view_menu, "Rotate View &Left 15\u00b0", lambda: self.canvas.rotate_view(-15))
        self._act(view_menu, "Rotate View &Right 15\u00b0", lambda: self.canvas.rotate_view(15))
        self._act(view_menu, "Reset View &Rotation", lambda: self.canvas.rotate_view(None))
        view_menu.addSeparator()
        self.show_path_action = self._act(view_menu, "Show &Path", self.do_show_path)
        self.show_path_action.setCheckable(True)
        self.show_path_action.setChecked(True)
        view_menu.addSeparator()

        def toggle(text, attr, key, default, after=None):
            act = QAction(text, self)
            act.setShortcut(QKeySequence(key))
            view_menu.addAction(act)
            act.setCheckable(True)
            act.setChecked(default)

            def flip(on, a=attr):
                setattr(self.canvas, a, bool(on))
                if after is not None:
                    after(bool(on))
                self.canvas.viewport().update()
            act.toggled.connect(flip)
            return act
        self.rulers_action = toggle("R&ulers", "show_rulers", "Ctrl+Alt+R", False,
                                    self.canvas.set_rulers)
        self.grid_action = toggle("&Grid", "show_grid", "Ctrl+'", False)
        self.guides_action = toggle("Gui&des", "show_guides", "Ctrl+;", True)
        self.extras_action = toggle("E&xtras (samplers, counts, notes, slices)",
                                    "show_extras", "Ctrl+Alt+H", True)
        snap_menu = view_menu.addMenu("&Snap To")
        self.snap_actions = {}
        for text, attr in (("&Guides", "snap_guides"), ("G&rid", "snap_grid"),
                           ("&Canvas Edges", "snap_bounds")):
            act = snap_menu.addAction(text)
            act.setCheckable(True)
            act.setChecked(bool(getattr(self.ctl, attr)))
            act.toggled.connect(lambda on, a=attr: setattr(self.ctl, a, bool(on)))
            self.snap_actions[attr] = act
        self._act(view_menu, "&New Guide...", self.do_new_guide)
        self._act(view_menu, "C&lear Guides", lambda: (self.ctl.clear_guides(),
                                                       self.canvas.viewport().update()))
        self._act(view_menu, "New Guide &Layout...", self.do_guide_layout)
        self.lock_guides_action = self._act(view_menu, "Lock Gui&des",
                                            lambda: self.ctl.set_guides_locked(
                                                self.lock_guides_action.isChecked()),
                                            "Ctrl+Alt+;")
        self.lock_guides_action.setCheckable(True)
        self._act(view_menu, "Grid &Settings...", self.do_grid_settings)
        view_menu.addSeparator()
        self._act(view_menu, "&Full Screen", self.do_full_screen, "F11")
        self._act(view_menu, "Show / Hide &Panels", self.do_toggle_panels, "Shift+F11")

        # Effects and adjustments, grouped exactly as the registry says --
        # so an addon's effect appears in the right menu without this file
        # knowing it exists.
        from ..engine.effects import EffectRegistry
        self.effects = EffectRegistry(self.ctl.db)
        # Paint.NET's arrangement: Adjustments on its own, everything else
        # under one Effects menu with a submenu per category -- nine
        # top-level menus of filters would push the rest off the bar.
        groups = self.effects.grouped()
        if "Adjustments" in groups:
            menu = bar.addMenu("&Adjustments")
            for name in groups["Adjustments"]:
                self._act(menu, self.effects.label(name),
                          lambda _c=False, n=name: self.run_effect(n))
        effects_menu = bar.addMenu("Effe&cts")
        self._act(effects_menu, "Filter &Gallery...", self.do_filter_gallery)
        self.nde_action = self._act(effects_menu, "Apply Filters &Non-Destructively",
                                    lambda: self.ctl.set_non_destructive(self.nde_action.isChecked()))
        self.nde_action.setCheckable(True)
        self.nde_action.setChecked(bool(getattr(self.ctl, "non_destructive", False)))
        self._act(effects_menu, "Layer &Filters...", self.do_layer_filters)
        effects_menu.addSeparator()
        order = ["Artistic", "Blurs", "Distort", "Edge & Generic", "Noise", "Photo",
                 "Render", "Stylize"]
        for category in sorted((c for c in groups if c != "Adjustments"),
                               key=lambda c: (order.index(c) if c in order else 99, c)):
            sub = effects_menu.addMenu(category.replace("&", "&&"))
            for name in groups[category]:
                if name == "filter_stack":        # the Filter Gallery's own engine
                    continue
                self._act(sub, self.effects.label(name),
                          lambda _c=False, n=name: self.run_effect(n))
            if category == "Distort":
                # Liquify is a brush-driven warp, which Ochre has as a tool.
                sub.addSeparator()
                self._act(sub, "Liquify (Warp tool)...", self.do_liquify)
                self._act(sub, "Puppet Warp (tool)...", lambda: self.ctl.set_tool("puppet_warp"))

        image_menu = bar.addMenu("&Image")
        self._act(image_menu, "&Crop to Selection", self.do_crop, "Ctrl+Shift+X")
        self._act(image_menu, "&Resize...", self.do_resize, "Ctrl+R")
        self._act(image_menu, "Content-Aware Sc&ale...", self.do_content_aware_scale,
                  "Ctrl+Alt+Shift+C")
        self._act(image_menu, "Canvas &Size...", self.do_canvas_size, "Ctrl+Shift+R")
        self._act(image_menu, "&Trim...", self.do_trim)
        self._act(image_menu, "D&uplicate...", self.do_duplicate_image)
        self._act(image_menu, "Appl&y Image...", self.do_apply_image)
        self._act(image_menu, "Calc&ulations...", self.do_calculations)
        image_menu.addSeparator()
        self._act(image_menu, "Flip &Horizontal", lambda: self._geometry(self.ctl.flip_image, True))
        self._act(image_menu, "Flip &Vertical", lambda: self._geometry(self.ctl.flip_image, False))
        self._act(image_menu, "Rotate 90\u00b0 Clock&wise",
                  lambda: self._geometry(self.ctl.rotate_image, 1), "Ctrl+H")
        self._act(image_menu, "Rotate 90\u00b0 Counter-Clockwise",
                  lambda: self._geometry(self.ctl.rotate_image, 3), "Ctrl+G")
        self._act(image_menu, "Rotate &180\u00b0", lambda: self._geometry(self.ctl.rotate_image, 2), "Ctrl+J")
        self._act(image_menu, "Rotate by &Angle...", self.do_rotate_image_by)
        self._act(image_menu, "S&traighten (from Measure)", self.do_straighten)
        self._act(image_menu, "&Match Color...", self.do_match_color)
        image_menu.addSeparator()
        self._act(image_menu, "Slices From &Guides", lambda: (self.ctl.slices_from_guides(),
                                                              self.canvas.viewport().update()))
        self._act(image_menu, "Clear S&lices", lambda: (self.ctl.clear_marks("slices"),
                                                        self.canvas.viewport().update()))
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
        self._act(layer_menu, "Rotate by An&gle...", self.do_rotate_layer)
        self._act(layer_menu, "&Scale Layer...", self.do_scale_layer)
        self._act(layer_menu, "&Offset...", self.do_offset_layer)
        align = layer_menu.addMenu("&Align")
        for how, label in (("left", "&Left"), ("hcenter", "&Center"), ("right", "&Right"),
                           ("top", "&Top"), ("vcenter", "&Middle"), ("bottom", "&Bottom")):
            self._act(align, label, lambda _c=False, h=how: self._refresh_after(
                self.ctl.align_layer, h))
        layer_menu.addSeparator()
        self.build_layer_fx_menu(layer_menu)
        layer_menu.addSeparator()
        self._act(layer_menu, "&Properties...", self.do_layer_properties, "F4")

    def build_layer_fx_menu(self, menu):
        """Masks, clipping, adjustment and fill layers, styles. Shared by the
        Layer menu and the Layers dock's right-click menu."""
        fx = self
        mask = menu.addMenu("Layer &Mask")
        for label, mode in (("&Reveal All", "reveal"), ("&Hide All", "hide"),
                            ("From &Selection", "selection"),
                            ("From &Transparency", "transparency")):
            mask.addAction(label, lambda m=mode: fx._layer_fx(fx.ctl.add_layer_mask, m))
        mask.addSeparator()
        layer = self.ctl.active_layer
        has_mask = layer is not None and getattr(layer, "mask", None) is not None
        edit = mask.addAction("&Edit Mask (paint on it)",
                              lambda: fx._layer_fx(fx.ctl.set_editing_mask,
                                                   not fx.ctl.editing_mask))
        edit.setCheckable(True)
        edit.setChecked(bool(self.ctl.editing_mask))
        edit.setEnabled(has_mask)
        on = mask.addAction("E&nabled", lambda: fx._layer_fx(
            fx.ctl.set_mask_enabled, not getattr(fx.ctl.active_layer, "mask_enabled", True)))
        on.setCheckable(True)
        on.setChecked(bool(getattr(layer, "mask_enabled", True)))
        on.setEnabled(has_mask)
        apply_ = mask.addAction("&Apply Mask", lambda: fx._layer_fx(fx.ctl.delete_layer_mask, True))
        delete = mask.addAction("&Delete Mask", lambda: fx._layer_fx(fx.ctl.delete_layer_mask))
        apply_.setEnabled(has_mask)
        delete.setEnabled(has_mask)

        def refresh():
            # A menu built once at start-up would show stale states; this
            # runs every time the menu opens.
            lay = fx.ctl.active_layer
            hm = lay is not None and getattr(lay, "mask", None) is not None
            edit.setChecked(bool(fx.ctl.editing_mask))
            on.setChecked(bool(getattr(lay, "mask_enabled", True)))
            for a in (edit, on, apply_, delete):
                a.setEnabled(hm)
        menu.aboutToShow.connect(refresh)
        mask.aboutToShow.connect(refresh)
        clip = QAction("Create / Release &Clipping Mask", menu)
        clip.setShortcut(QKeySequence("Ctrl+Alt+G"))
        clip.triggered.connect(lambda: fx._layer_fx(fx.ctl.toggle_clipping))
        menu.addAction(clip)
        adj = menu.addMenu("New Ad&justment Layer")
        for name in self.effects.names():
            effect_cls = self.effects._classes.get(name)
            if getattr(effect_cls, "pointwise", False) and not getattr(effect_cls, "global_stats", False):
                adj.addAction(self.effects.label(name),
                              lambda n=name: fx.do_new_adjustment_layer(n))
        fill = menu.addMenu("New &Fill Layer")
        fill.addAction("&Solid Color...", lambda: fx.do_new_fill_layer("solid"))
        fill.addAction("&Gradient...", lambda: fx.do_new_fill_layer("gradient"))
        fill.addAction("&Pattern...", lambda: fx.do_new_fill_layer("pattern"))
        menu.addAction("Layer St&yle...", self.do_layer_style)
        menu.addAction("Clear Layer Style", lambda: fx._layer_fx(
            fx.ctl.set_layer_styles, fx.ctl.active_layer, {}))
        menu.addAction("Edit Adjustment / Fill...", self.do_edit_layer_content)
        menu.addAction("&Rasterize Layer", lambda: fx._layer_fx(fx.ctl.rasterize_layer))
        vmask = menu.addMenu("&Vector Mask")
        vmask.addAction("From Active &Path", lambda: fx._layer_fx(fx.ctl.add_vector_mask, "path"))
        vmask.addAction("&Reveal All", lambda: fx._layer_fx(fx.ctl.add_vector_mask, "reveal"))
        vmask.addAction("&Hide All", lambda: fx._layer_fx(fx.ctl.add_vector_mask, "hide"))
        vmask.addAction("From &Selection", lambda: fx._layer_fx(fx.ctl.add_vector_mask, "selection"))
        vmask.addSeparator()
        vmask.addAction("&Enabled / Disabled", lambda: fx._layer_fx(
            fx.ctl.set_layer_property, fx.ctl.active_layer, "vector_mask_enabled",
            not getattr(fx.ctl.active_layer, "vector_mask_enabled", True)))
        vmask.addAction("Rasteri&ze into Layer Mask", lambda: fx._layer_fx(
            fx.ctl.delete_vector_mask, None, True))
        vmask.addAction("&Delete", lambda: fx._layer_fx(fx.ctl.delete_vector_mask))
        boards = menu.addMenu("Artboards and &Frames")
        boards.addAction("&Artboard from Layer", lambda: fx._layer_fx(
            fx.ctl.region_from_layer, "artboard"))
        boards.addAction("Convert to &Frame", lambda: fx._layer_fx(fx.ctl.region_from_layer, "frame"))
        boards.addAction("Artboard &Background...", self.do_artboard_background)
        smart = menu.addMenu("Smart O&bject")
        smart.addAction("&Convert to Smart Object", lambda: fx._layer_fx(fx.ctl.convert_to_smart))
        smart.addAction("&Edit Contents", self.do_edit_smart_contents)
        smart.addAction("&Update from This Tab", self.do_update_smart_object)
        smart.addAction("&Replace Contents...", self.do_replace_smart_contents)
        smart.addAction("Clear Smart &Filters", lambda: fx._layer_fx(fx.ctl.set_smart_filters, []))
        smart.addAction("Edit Smart Filters...", self.do_layer_filters)
        smart.addSeparator()
        smart.addAction("Update &Modified Content", self.do_update_linked)
        smart.addAction("Re&link to File...", self.do_relink)
        return menu

    def _active_region(self, kind=None):
        node = self.ctl.active_layer
        while node is not None:
            reg = getattr(node, "region", None)
            if node.is_group and reg and (kind is None or reg["type"] == kind):
                return node
            node = node.parent
        return None

    def do_artboard_background(self):
        group = self._active_region("artboard")
        if group is None:
            QMessageBox.information(self, "Artboard", "Select an artboard, or a layer on one.")
            return
        from PySide6.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(self, "Artboard Background", "Background:",
                                          ["White", "Black", "Transparent", "Other..."], 0, False)
        if not ok:
            return
        colours = {"White": (255, 255, 255, 255), "Black": (0, 0, 0, 255), "Transparent": None}
        if choice == "Other...":
            from PySide6.QtGui import QColor
            from PySide6.QtWidgets import QColorDialog
            c = QColorDialog.getColor(QColor(255, 255, 255), self, "Artboard Background")
            if not c.isValid():
                return
            colour = (c.red(), c.green(), c.blue(), 255)
        else:
            colour = colours[choice]
        self._layer_fx(self.ctl.set_region, group, bg=None if colour is None else list(colour))

    def do_export_artboards(self):
        from ..engine import regions
        if not regions.groups(self.ctl.doc, "artboard"):
            QMessageBox.information(self, "Export Artboards", "This image has no artboards.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Export Artboards to")
        if not directory:
            return
        paths = self.ctl.export_artboards(directory)
        self.statusBar().showMessage("Exported %d artboard(s) to %s" % (len(paths), directory), 6000)

    def do_edit_smart_contents(self):
        if self.ctl.edit_smart_contents() is None:
            QMessageBox.information(self, "Edit Contents", "The active layer is not a smart object.")
            return
        self.canvas.bind()
        self._sync_all()

    def do_update_smart_object(self):
        if getattr(self.ctl, "smart_link", None) is None:
            QMessageBox.information(self, "Update Smart Object",
                                    "This tab is not a smart object's contents.")
            return False
        if not self.ctl.update_smart_link():
            QMessageBox.warning(self, "Update Smart Object",
                                "The document or layer this came from is no longer open.")
            return False
        self._sync_all()
        return True

    def do_replace_smart_contents(self):
        if getattr(self.ctl.active_layer, "kind", "raster") != "smart":
            QMessageBox.information(self, "Replace Contents", "The active layer is not a smart object.")
            return
        exts = " ".join("*.%s" % e for e in supported_extensions())
        path, _ = QFileDialog.getOpenFileName(self, "Replace Contents", "", "Images (%s)" % exts)
        if path:
            self._layer_fx(self.ctl.replace_smart_contents, path)

    def do_place_embedded(self):
        exts = " ".join("*.%s" % e for e in supported_extensions())
        path, _ = QFileDialog.getOpenFileName(self, "Place Embedded", "", "Images (%s)" % exts)
        if not path:
            return
        try:
            self.ctl.place_embedded(path)
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self, "Place failed", str(exc))
            return
        self.layers_dock.sync()
        self.canvas.refresh()
        self.canvas.viewport().update()

    def _layer_fx(self, fn, *args):
        if self.ctl.active_layer is None:
            return None
        result = fn(*args)
        self.layers_dock.sync()
        self.canvas.refresh()
        self.canvas.viewport().update()
        return result

    def do_new_adjustment_layer(self, effect):
        from .layerdialogs import AdjustmentLayerDialog
        layer = self.ctl.add_adjustment_layer(effect)
        self.layers_dock.sync()
        self.canvas.refresh()
        dlg = AdjustmentLayerDialog(self.ctl, layer, self.effects, self)
        dlg.exec()
        self.canvas.refresh()
        return layer

    def do_new_fill_layer(self, kind):
        from .layerdialogs import FillLayerDialog
        layer = self.ctl.add_fill_layer(kind)
        self.layers_dock.sync()
        self.canvas.refresh()
        FillLayerDialog(self.ctl, layer, self).exec()
        self.canvas.refresh()
        return layer

    def do_layer_style(self):
        from .layerdialogs import LayerStyleDialog
        layer = self.ctl.active_layer
        if layer is None or layer.is_group or getattr(layer, "kind", "raster") == "adjustment":
            self.statusBar().showMessage("Layer styles apply to pixel and fill layers.", 5000)
            return
        LayerStyleDialog(self.ctl, layer, self).exec()
        self.layers_dock.sync()
        self.canvas.refresh()

    def do_edit_layer_content(self):
        from .layerdialogs import AdjustmentLayerDialog, FillLayerDialog
        layer = self.ctl.active_layer
        kind = getattr(layer, "kind", "raster")
        if kind == "adjustment":
            AdjustmentLayerDialog(self.ctl, layer, self.effects, self).exec()
        elif kind == "fill":
            FillLayerDialog(self.ctl, layer, self).exec()
        elif kind == "text":
            # Into the text, ready to type, at the end of it.
            from ..engine import textlayer
            box = textlayer.bounds(layer.params)
            self.ctl.set_tool("text")
            if box is not None:
                x, y = box.x1 - 1, box.y1 - 1
                self.ctl.begin_stroke(x, y)
                self.ctl.end_stroke(x, y)
                self.ctl.tool.caret_end()
                self.canvas._start_caret()
                self.canvas.setFocus()
        else:
            return False
        self.canvas.refresh()
        return True

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
        # Every tab with unsaved work gets asked about, brought to the front
        # first so the question is about something you can see.
        for index in self.ctl.modified_documents():
            if index != self.ctl.current_doc:
                self.switch_document(index)
            if not self.confirm_discard("closing"):
                event.ignore()
                return
        self.shutdown()
        event.accept()

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
        # A new image opens in its own tab, so nothing is put at risk.
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
            self, "Open", "",
            "Images (*.ochre *.psd *.psb *.ora %s);;Photoshop (*.psd *.psb);;"
            "OpenRaster (*.ora);;All files (*)" % exts)
        if not path:
            return
        self.open_file(path)

    def do_save(self):
        if self.ctl.path is None and getattr(self.ctl, "smart_link", None) is not None:
            # A smart object's contents tab: Save puts it back, as in Photoshop.
            return self.do_update_smart_object()
        if self.ctl.path is None:
            return self.do_save_as()
        try:
            self.ctl.save_path()
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(exc))
        self._report_save_warnings()
        self._update_title()

    def _report_save_warnings(self):
        """Say what a PSD or ORA could not keep, instead of losing it quietly."""
        warnings = getattr(self.ctl, "last_save_warnings", None)
        if warnings:
            QMessageBox.information(
                self, "Saved, with changes",
                "The file was saved, but this format cannot hold everything:\n\n  "
                + "\n  ".join(warnings[:12])
                + ("\n  ..." if len(warnings) > 12 else "")
                + "\n\nSave as .ochre to keep it all.")

    def do_export_as(self):
        from .exportdialog import ExportDialog
        ExportDialog(self.ctl, self).exec()

    def do_save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "untitled.ochre",
            "Ochre document (*.ochre);;Photoshop (*.psd);;OpenRaster (*.ora);;"
            "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp);;AVIF (*.avif);;TIFF (*.tif)")
        if not path:
            return
        try:
            self.ctl.save_path(path)
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._report_save_warnings()
        if self.recent:
            self.recent.add(path)
            self._rebuild_recent()
        self._update_title()

    # ---- Edit / Select / Image commands with a few settings ---------------------------

    def _blend_choices(self):
        return [(n, n) for n in self.ctl.blends.names() if n != "PassThrough"]

    def _after_pixels(self):
        self.layers_dock.sync()
        self.canvas.refresh()
        self.canvas.viewport().update()
        self._update_title()

    def do_fill_dialog(self):
        from ..engine import patterns
        from .formdialog import ask
        v = ask(self, "Fill", [
            ("contents", "Contents", "choice", "foreground", [
                ("foreground", "Foreground colour"), ("background", "Background colour"),
                ("color", "Colour..."), ("pattern", "Pattern"), ("content_aware", "Content-Aware"),
                ("history", "History"), ("black", "Black"), ("grey", "50% Grey"), ("white", "White")]),
            ("colour", "Colour", "color", tuple(self.ctl.primary), None),
            ("pattern", "Pattern", "choice", "checker",
             [(n, n) for n in patterns.names(self.ctl.state_dir)]),
            ("scale", "Pattern scale (%)", "int", 100, (5, 1000)),
            ("mode", "Mode", "choice", "Normal", self._blend_choices()),
            ("opacity", "Opacity (%)", "int", 100, (0, 100)),
            ("preserve", "Preserve transparency", "bool", False, None)])
        if v is not None:
            self._busy(self.ctl.fill_with, v["contents"], v["colour"], v["pattern"], v["scale"],
                       v["mode"], v["opacity"], v["preserve"])
            self._after_pixels()

    def do_stroke_dialog(self):
        from .formdialog import ask
        v = ask(self, "Stroke", [
            ("width", "Width (px)", "int", 3, (1, 250)),
            ("colour", "Colour", "color", tuple(self.ctl.primary), None),
            ("location", "Location", "choice", "center",
             [("inside", "Inside"), ("center", "Center"), ("outside", "Outside")]),
            ("mode", "Mode", "choice", "Normal", self._blend_choices()),
            ("opacity", "Opacity (%)", "int", 100, (0, 100)),
            ("preserve", "Preserve transparency", "bool", False, None)])
        if v is not None:
            self.ctl.stroke_selection(v["width"], v["colour"], v["location"], v["mode"],
                                      v["opacity"], v["preserve"])
            self._after_pixels()

    def do_define_pattern(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Define Pattern", "Pattern name:", text="Pattern")
        if ok and name.strip():
            saved = self.ctl.define_pattern(name.strip())
            self.statusBar().showMessage("Pattern %r defined." % saved if saved else
                                         "No pattern: select an area first.", 5000)

    def do_color_range(self):
        from ..engine import selectrange
        from .formdialog import ask
        v = ask(self, "Color Range", [
            ("mode", "Select", "choice", "sampled",
             [(m, "Sampled colour (the primary)" if m == "sampled" else m.capitalize())
              for m in selectrange.MODES]),
            ("fuzziness", "Fuzziness", "int", 40, (0, 200)),
            ("invert", "Invert", "bool", False, None)])
        if v is not None:
            self._sel(lambda: self.ctl.select_color_range(v["mode"], None, v["fuzziness"], v["invert"]))

    def do_focus_area(self):
        from .formdialog import ask
        v = ask(self, "Focus Area", [("in_focus", "In-focus range", "int", 50, (0, 100)),
                                     ("noise", "Image noise level", "int", 0, (0, 100))])
        if v is not None:
            self._sel(lambda: self._busy(self.ctl.select_focus_area, v["in_focus"], v["noise"]))

    def do_trim(self):
        from .formdialog import ask
        v = ask(self, "Trim", [
            ("based_on", "Based on", "choice", "transparent",
             [("transparent", "Transparent pixels"), ("top_left", "Top left pixel colour"),
              ("bottom_right", "Bottom right pixel colour")]),
            ("top", "Trim top", "bool", True, None), ("bottom", "Trim bottom", "bool", True, None),
            ("left", "Trim left", "bool", True, None), ("right", "Trim right", "bool", True, None)])
        if v is not None:
            self._geometry(self.ctl.trim, v["based_on"], v["top"], v["bottom"], v["left"], v["right"])

    def do_duplicate_image(self):
        from .formdialog import ask
        v = ask(self, "Duplicate Image", [("flatten", "Merged layers only", "bool", False, None)])
        if v is not None:
            self.ctl.duplicate_document(v["flatten"])
            self.canvas.bind()
            self._sync_all()

    def _source_choices(self):
        return [("merged", "Merged")] + [(l.id, l.name) for l in reversed(self.ctl.doc.layers())]

    def do_apply_image(self):
        from .formdialog import ask
        channels = [("rgb", "RGB"), ("red", "Red"), ("green", "Green"), ("blue", "Blue"),
                    ("alpha", "Transparency")]
        v = ask(self, "Apply Image", [
            ("source", "Layer", "choice", "merged", self._source_choices()),
            ("channel", "Channel", "choice", "rgb", channels),
            ("invert", "Invert", "bool", False, None),
            ("mode", "Blending", "choice", "Multiply", self._blend_choices()),
            ("opacity", "Opacity (%)", "int", 100, (0, 100)),
            ("preserve", "Preserve transparency", "bool", False, None)])
        if v is not None:
            self.ctl.apply_image(v["source"], v["channel"], v["invert"], v["mode"], v["opacity"],
                                 v["preserve"])
            self._after_pixels()

    def do_calculations(self):
        from .formdialog import ask
        channels = [("red", "Red"), ("green", "Green"), ("blue", "Blue"), ("alpha", "Transparency")]
        v = ask(self, "Calculations", [
            ("source1", "Source 1", "choice", "merged", self._source_choices()),
            ("channel1", "Channel", "choice", "red", channels),
            ("invert1", "Invert", "bool", False, None),
            ("source2", "Source 2", "choice", "merged", self._source_choices()),
            ("channel2", "Channel", "choice", "green", channels),
            ("invert2", "Invert", "bool", False, None),
            ("mode", "Blending", "choice", "Multiply", self._blend_choices()),
            ("opacity", "Opacity (%)", "int", 100, (0, 100)),
            ("result", "Result", "choice", "channel",
             [("channel", "New channel"), ("selection", "Selection")])])
        if v is not None:
            self.ctl.calculations(v["source1"], v["channel1"], v["invert1"], v["source2"],
                                  v["channel2"], v["invert2"], v["mode"], v["opacity"], v["result"])
            self._sync_all()

    def _image_files(self, title):
        exts = " ".join("*.%s" % e for e in supported_extensions())
        paths, _ = QFileDialog.getOpenFileNames(self, title, "", "Images (%s)" % exts)
        return paths

    def do_auto_align(self):
        from .formdialog import ask
        v = ask(self, "Auto-Align Layers", [("model", "Projection", "choice", "homography",
                                            [("homography", "Perspective"),
                                             ("similarity", "Reposition (move, rotate, scale)")])])
        if v is None:
            return
        report = self._busy(self.ctl.auto_align_layers, v["model"])
        missed = [n for n, k in (report or {}).items() if not k]
        self.statusBar().showMessage("Aligned %d layer(s)%s" % (len(report or {}) - len(missed),
                                     "; could not match: " + ", ".join(missed) if missed else ""), 8000)
        self._after_pixels()

    def do_auto_blend(self):
        from .formdialog import ask
        v = ask(self, "Auto-Blend Layers", [("method", "Blend method", "choice", "stack",
                                            [("stack", "Stack images (focus)"),
                                             ("panorama", "Panorama")])])
        if v is not None:
            self._busy(self.ctl.auto_blend_layers, v["method"])
            self._after_pixels()

    def do_photomerge(self):
        paths = self._image_files("Photomerge: choose the photos")
        if len(paths) < 2:
            return
        from .formdialog import ask
        v = ask(self, "Photomerge", [("model", "Layout", "choice", "homography",
                                      [("homography", "Perspective"),
                                       ("similarity", "Reposition")]),
                                     ("blend", "Blend images together", "bool", True, None)])
        if v is None:
            return
        try:
            _doc, report = self._busy(self.ctl.photomerge, paths, v["blend"], v["model"])
        except ValueError as exc:
            QMessageBox.warning(self, "Photomerge", str(exc))
            return
        missed = [n for n, k in report.items() if k == 0]
        if missed:
            QMessageBox.information(self, "Photomerge", "Could not place: " + ", ".join(missed))
        self.canvas.bind()
        self._sync_all()

    def do_load_stack(self):
        paths = self._image_files("Load Files into Stack")
        if not paths:
            return
        from .formdialog import ask
        v = ask(self, "Load Files into Stack", [("align", "Attempt to automatically align", "bool",
                                                False, None)])
        if v is None:
            return
        self._busy(self.ctl.load_files_into_stack, paths, v["align"])
        self.canvas.bind()
        self._sync_all()

    def do_guide_layout(self):
        from .formdialog import ask
        v = ask(self, "New Guide Layout", [("columns", "Columns", "int", 4, (0, 100)),
                                           ("rows", "Rows", "int", 0, (0, 100)),
                                           ("gutter", "Gutter (px)", "int", 20, (0, 2000)),
                                           ("margin", "Margin (px)", "int", 0, (0, 2000)),
                                           ("clear", "Clear existing guides", "bool", True, None)])
        if v is not None:
            self.ctl.new_guide_layout(v["columns"], v["rows"], v["gutter"], v["margin"], v["clear"])
            self.canvas.viewport().update()

    def do_full_screen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def do_toggle_panels(self):
        """Photoshop's Tab: hide every panel for the image alone, or bring
        back the ones that were showing."""
        from PySide6.QtWidgets import QDockWidget
        docks = self.findChildren(QDockWidget)
        shown = [d for d in docks if d.isVisible()]
        if shown:
            self._hidden_docks = shown
            for d in shown:
                d.hide()
        else:
            for d in getattr(self, "_hidden_docks", docks):
                d.show()

    def do_define_brush_tip(self):
        ok = self.ctl.define_brush_tip()
        self.statusBar().showMessage(
            "Brush tip defined: pick Tip = Defined in the Paintbrush options." if ok else
            "No brush tip: select something dark on light first.", 6000)

    def do_layer_filters(self):
        from .filterlist import LayerFiltersDialog
        layer = self.ctl.active_layer
        if layer is None or layer.is_group or getattr(layer, "kind", "raster") not in ("raster", "smart"):
            QMessageBox.information(self, "Layer Filters", "Select a pixel layer or a smart object.")
            return
        if not (layer.params or {}).get("filters"):
            QMessageBox.information(self, "Layer Filters",
                                    "This layer has no filters yet. Turn on Effects > Apply Filters "
                                    "Non-Destructively (or use a smart object) and apply one.")
            return
        LayerFiltersDialog(self.ctl, layer, self.effects, self).exec()
        self._after_pixels()

    def do_place_linked(self):
        exts = " ".join("*.%s" % e for e in supported_extensions())
        path, _ = QFileDialog.getOpenFileName(self, "Place Linked", "", "Images (%s)" % exts)
        if path:
            try:
                self.ctl.place_linked(path)
            except Exception as exc:                        # noqa: BLE001
                QMessageBox.warning(self, "Place failed", str(exc))
                return
            self._after_pixels()

    def do_update_linked(self):
        status = self.ctl.linked_status()
        n = self.ctl.update_linked()
        missing = [l.name for l, st in status if st == "missing"]
        msg = "Updated %d linked smart object(s)." % n
        if missing:
            msg += " Missing files for: " + ", ".join(missing) + " (use Relink to File)."
        self.statusBar().showMessage(msg, 8000)
        self._after_pixels()

    def do_relink(self):
        if getattr(self.ctl.active_layer, "kind", "raster") != "smart":
            QMessageBox.information(self, "Relink", "The active layer is not a smart object.")
            return
        exts = " ".join("*.%s" % e for e in supported_extensions())
        path, _ = QFileDialog.getOpenFileName(self, "Relink to File", "", "Images (%s)" % exts)
        if path:
            self.ctl.relink(path)
            self._after_pixels()

    def do_filter_gallery(self):
        from .gallery import FilterGalleryDialog
        if self.ctl.active_layer is None:
            return None
        dlg = FilterGalleryDialog(self.ctl, self.effects, self)
        if dlg.exec() and dlg.stack:
            self.ctl.apply_effect("filter_stack", {"stack": dlg.stack_json()})
            self.canvas.refresh()
            self.canvas.viewport().update()
            self._update_title()
        return dlg

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

    def do_paste_in_perspective(self):
        if not getattr(self.ctl.doc, "planes", None):
            QMessageBox.information(self, "Paste in Perspective",
                                    "Define a plane first with the Vanishing Point tool.")
            return None
        foreign = self._system_clipboard_get()
        clip = foreign or self.ctl.clip
        if clip is None:
            return None
        layer = self.ctl.paste_in_perspective(clip.rgba)
        self.layers_dock.sync()
        self.canvas.refresh()
        self.canvas.viewport().update()
        return layer

    def do_paste_new_image(self):
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
        """Open a path with error reporting. Shared by Open, Recent and drop.

        It opens in a new tab (or replaces an untouched Untitled), so there
        is nothing to ask about.
        """
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
        changed = [l.name for l, st in self.ctl.linked_status() if st != "ok"]
        if changed:
            self.statusBar().showMessage(
                "Linked smart objects changed or missing: %s -- Layer > Smart Object > "
                "Update Modified Content / Relink." % ", ".join(changed), 10000)
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

    def _refresh_after(self, fn, *args):
        result = fn(*args)
        self.canvas.refresh()
        self.canvas.viewport().update()
        return result

    def _ask_number(self, title, label, default, lo, hi, decimals=1):
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getDouble(self, title, label, default, lo, hi, decimals)
        return value if ok else None

    def do_rotate_image_by(self):
        v = self._ask_number("Rotate Image", "Angle (degrees, clockwise):", 15.0, -360, 360)
        if v is not None:
            self._geometry(self.ctl.rotate_image_by, v)

    def do_rotate_layer(self):
        v = self._ask_number("Rotate Layer", "Angle (degrees, clockwise):", 15.0, -360, 360)
        if v is not None:
            self._refresh_after(self.ctl.rotate_layer, v)

    def do_scale_layer(self):
        v = self._ask_number("Scale Layer", "Scale (percent):", 100.0, 1, 10000)
        if v is not None:
            self._refresh_after(self.ctl.scale_layer, v / 100.0)

    def do_offset_layer(self):
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, "Offset Layer",
                                        "Offset x, y in pixels (wraps around the edges):",
                                        text="%d, %d" % (self.ctl.doc.width // 2,
                                                         self.ctl.doc.height // 2))
        if not ok:
            return
        try:
            dx, dy = (int(v) for v in text.replace(" ", "").split(","))
        except ValueError:
            return
        self._refresh_after(self.ctl.offset_layer, dx, dy, True)

    def _sel(self, fn, *args):
        fn(*args)
        self.canvas.viewport().update()

    def _ask_radius(self, title, fn, default):
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getInt(self, title, "Radius (pixels):", default, 1, 2000)
        if ok:
            self._sel(fn, value)

    def do_stroke_path(self):
        from PySide6.QtWidgets import QInputDialog
        if self.ctl.doc.path is None:
            self.statusBar().showMessage("No active path to stroke.", 4000)
            return
        value, ok = QInputDialog.getInt(self, "Stroke Path", "Line width (pixels):",
                                        self.paths_dock.width.value(), 1, 500)
        if ok:
            self.paths_dock.width.setValue(value)
            self.ctl.stroke_path(value)
            self.canvas.refresh()

    def do_show_path(self):
        self.canvas.show_path = self.show_path_action.isChecked()
        self.canvas.viewport().update()

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
