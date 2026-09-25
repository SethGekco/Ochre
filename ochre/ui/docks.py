# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Docks: tools, layers, colours, palette, history.

Every dock reads through the controller and writes through it too; none of
them touches the engine directly. They refresh in response to bus events
rather than polling, which is what keeps panel state from drifting out of
sync with the document.

The palette dock is the one with a rule attached: it renders whatever
annotations it is handed and never interprets them. The core knows that a
palette can carry named index ranges with a role hint; it does not know what
any of them mean. A format addon supplies the meanings.
"""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QColorDialog,
                               QComboBox, QDockWidget, QFormLayout, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QSlider,
                               QToolButton, QVBoxLayout, QWidget)

from ..engine.palette import LENGTH


def _swatch(rgba, size=16):
    pix = QPixmap(size, size)
    pix.fill(QColor(*rgba))
    return pix


class ToolsDock(QDockWidget):
    """One button per registered tool. Populated from the registry, so a
    tool contributed by an addon appears here without this file changing."""

    def __init__(self, controller, parent=None):
        super().__init__("Tools", parent)
        self.setObjectName("Tools")
        self.ctl = controller
        body = QWidget()
        grid = QGridLayout(body)
        grid.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        for i, name in enumerate(controller.tools.names()):
            btn = QToolButton()
            btn.setText(controller.tools.label(name))
            btn.setCheckable(True)
            btn.setToolTip(name)
            btn.clicked.connect(lambda _c=False, n=name: self.ctl.set_tool(n))
            self.group.addButton(btn)
            grid.addWidget(btn, i // 2, i % 2)
            if name == controller.tool_name:
                btn.setChecked(True)
        grid.setRowStretch(grid.rowCount(), 1)
        self.setWidget(body)

    def sync(self, **_):
        for btn in self.group.buttons():
            btn.setChecked(btn.toolTip() == self.ctl.tool_name)


class LayersDock(QDockWidget):
    """Layer list with visibility, opacity and blend mode."""

    def __init__(self, controller, parent=None):
        super().__init__("Layers", parent)
        self.setObjectName("Layers")
        self.ctl = controller
        self._syncing = False

        body = QWidget()
        box = QVBoxLayout(body)

        self.blend = QComboBox()
        from ..engine.blend import BlendRegistry
        self._blends = BlendRegistry(controller.db)
        for mode in self._blends.ordered():
            self.blend.addItem(mode.label, mode.name)
        self.blend.currentIndexChanged.connect(self._blend_changed)

        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 255)
        self.opacity.setValue(255)
        self.opacity.valueChanged.connect(self._opacity_changed)

        form = QFormLayout()
        form.addRow("Blend", self.blend)
        form.addRow("Opacity", self.opacity)
        box.addLayout(form)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self._row_changed)
        self.list.itemChanged.connect(self._item_changed)

        # Drag to restack. InternalMove lets Qt animate the drop, but the
        # model underneath is the layer TREE, not this flat list -- so the
        # view's own reordering is undone on the next sync() and the real
        # move goes through the controller, where it becomes one undo entry.
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.model().rowsMoved.connect(self._rows_moved)

        # Right-click, because reaching for a button to delete something is
        # a preference and not a law. Everything here is also on the buttons
        # or the menu bar; nothing is only reachable this way.
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        box.addWidget(self.list, 1)

        buttons = QHBoxLayout()
        for label, slot in (("Add", self._add), ("Group", self._group),
                            ("Delete", self._delete)):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        box.addLayout(buttons)
        self.setWidget(body)

    # ---- population ------------------------------------------------------

    def sync(self, **_):
        if self.ctl.doc is None:
            return
        self._syncing = True
        self.list.clear()
        self._rows = self.ctl.layer_rows()
        for depth, node in self._rows:
            text = "%s%s" % ("    " * depth, node.name or node.id)
            if node.is_group:
                text += "  [group]"
            elif node.index_locked:
                text += "  [indexed]"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if node.visible else Qt.Unchecked)
            item.setData(Qt.UserRole, node.id)
            self.list.addItem(item)
            if node is self.ctl.active_layer:
                self.list.setCurrentRow(self.list.count() - 1)
        active = self.ctl.active_layer
        if active is not None:
            self.opacity.setValue(active.opacity)
            index = self.blend.findData(active.blend)
            self.blend.setCurrentIndex(index if index >= 0 else 0)
        self._syncing = False

    # ---- handlers --------------------------------------------------------

    def _node_at(self, row):
        if not hasattr(self, "_rows") or not (0 <= row < len(self._rows)):
            return None
        return self._rows[row][1]

    def _row_changed(self, row):
        if self._syncing:
            return
        node = self._node_at(row)
        if node is not None:
            self.ctl.set_active_layer(node)
            self.sync()

    def _item_changed(self, item):
        if self._syncing:
            return
        node = self._node_at(self.list.row(item))
        if node is None:
            return
        wanted = item.checkState() == Qt.Checked
        if node.visible != wanted:
            self.ctl.set_layer_property(node, "visible", wanted)

    def _opacity_changed(self, value):
        if self._syncing or self.ctl.active_layer is None:
            return
        self.ctl.set_layer_property(self.ctl.active_layer, "opacity", int(value))

    def _blend_changed(self, _index):
        if self._syncing or self.ctl.active_layer is None:
            return
        name = self.blend.currentData()
        if name:
            self.ctl.set_layer_property(self.ctl.active_layer, "blend", name)

    def _rows_moved(self, _parent, start, _end, _dest, row):
        """A drag finished. Translate list position into a tree position."""
        if self._syncing:
            return
        node = self._node_at_pre_move(start)
        if node is None:
            self.sync()
            return

        # The list is a FLATTENED tree, so "row 3" is not "index 3". The node
        # now above the drop point decides both the parent and the index:
        # dropping onto a group's contents means joining that group.
        target = row - 1 if row > start else row
        parent, index = self._tree_position(target, node)
        if parent is not None:
            self.ctl.move_layer(node, parent, index)
        self.sync()

    def _node_at_pre_move(self, row):
        """The node that WAS at `row` before Qt reordered its own view."""
        if not hasattr(self, "_rows") or not (0 <= row < len(self._rows)):
            return None
        return self._rows[row][1]

    def _tree_position(self, target_row, moving):
        """(parent, index) for a node dropped at flattened row `target_row`."""
        root = self.ctl.doc.root
        rows = [n for _d, n in getattr(self, "_rows", []) if n is not moving]
        if not rows or target_row <= 0:
            return root, 0
        anchor = rows[min(target_row, len(rows)) - 1]

        # Dropping just under a group header puts the node INSIDE the group,
        # which is what the indentation implies and what every other editor
        # does. Otherwise it becomes the anchor's sibling, just above it.
        if anchor.is_group:
            return anchor, len(anchor.children)
        parent = anchor.parent or root
        if moving.parent is parent and parent.children.index(moving) < \
                parent.children.index(anchor):
            return parent, parent.children.index(anchor)
        return parent, parent.children.index(anchor) + 1

    def _context_menu(self, point):
        item = self.list.itemAt(point)
        node = self._node_at(self.list.row(item)) if item is not None else None
        menu = self.build_context_menu(node)
        menu.exec(self.list.viewport().mapToGlobal(point))

    def build_context_menu(self, node):
        """The right-click menu for `node`, built but not shown.

        Separate from showing it so a test can inspect the entries. Calling
        exec() in a test hangs -- the menu really does open and really does
        wait for a click that is never coming.
        """
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        if node is not None and node is not self.ctl.active_layer:
            self.ctl.set_active_layer(node)
            self.sync()

        menu.addAction("Add Layer", self._add)
        menu.addAction("Add Group", self._group)
        if node is not None:
            menu.addAction("Duplicate", self._duplicate).setEnabled(
                not node.is_group)
            menu.addSeparator()
            visible = menu.addAction("Visible", self._toggle_visible)
            visible.setCheckable(True)
            visible.setChecked(bool(node.visible))
            menu.addSeparator()
            # Delete is last and separated, so a stray click near the edge of
            # the menu does not destroy a layer.
            delete = menu.addAction("Delete", self._delete)
            delete.setEnabled(len(self.ctl.doc.layers()) > 1)
        return menu

    def _toggle_visible(self):
        node = self.ctl.active_layer
        if node is not None:
            self.ctl.set_layer_property(node, "visible", not node.visible)
            self.sync()

    def _duplicate(self):
        self.ctl.duplicate_layer()

    def _add(self):
        self.ctl.add_layer()

    def _group(self):
        self.ctl.add_group()

    def _delete(self):
        self.ctl.remove_layer()


class ColorsDock(QDockWidget):
    """Primary and secondary colour, plus the index readout."""

    def __init__(self, controller, parent=None):
        super().__init__("Colors", parent)
        self.setObjectName("Colors")
        self.ctl = controller
        body = QWidget()
        box = QVBoxLayout(body)

        self.primary_btn = QPushButton("Primary")
        self.primary_btn.clicked.connect(lambda: self._pick("primary"))
        self.secondary_btn = QPushButton("Secondary")
        self.secondary_btn.clicked.connect(lambda: self._pick("secondary"))
        box.addWidget(self.primary_btn)
        box.addWidget(self.secondary_btn)

        self.readout = QLabel("-")
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(self.readout)
        box.addStretch(1)
        self.setWidget(body)
        self.sync()

    def _pick(self, which):
        current = self.ctl.primary if which == "primary" else self.ctl.secondary
        chosen = QColorDialog.getColor(QColor(*current), self,
                                       "Choose %s colour" % which,
                                       QColorDialog.ShowAlphaChannel)
        if not chosen.isValid():
            return
        rgba = (chosen.red(), chosen.green(), chosen.blue(), chosen.alpha())
        if which == "primary":
            self.ctl.set_primary(rgba)
        else:
            self.ctl.set_secondary(rgba)
        self.sync()

    def sync(self, **_):
        self.primary_btn.setIcon(QIcon(_swatch(self.ctl.primary)))
        self.secondary_btn.setIcon(QIcon(_swatch(self.ctl.secondary)))
        # Show the INDEX, not only the hex. A spriter needs to know they are
        # painting index 17, not #A85C20.
        bits = ["#%02X%02X%02X" % self.ctl.primary[:3]]
        if self.ctl.primary_index is not None:
            bits.insert(0, "index %d" % self.ctl.primary_index)
        self.readout.setText("  /  ".join(bits))


class PaletteDock(QDockWidget):
    """256 swatches, with whatever annotations the document's palette carries.

    This widget renders annotated index ranges and never interprets them.
    That separation is what keeps game-specific knowledge out of the core:
    an addon that understands a particular palette supplies the labels.
    """

    CELL = 14
    COLS = 16

    def __init__(self, controller, parent=None):
        super().__init__("Palette", parent)
        self.setObjectName("Palette")
        self.ctl = controller
        body = QWidget()
        box = QVBoxLayout(body)
        self.grid_label = QLabel()
        self.grid_label.setMinimumSize(QSize(self.COLS * self.CELL,
                                             (LENGTH // self.COLS) * self.CELL))
        self.grid_label.mousePressEvent = self._clicked
        self.grid_label.mouseDoubleClickEvent = self._double_clicked
        box.addWidget(self.grid_label)
        self.info = QLabel("No palette bound")
        self.info.setWordWrap(True)
        box.addWidget(self.info)
        box.addStretch(1)
        self.setWidget(body)
        self.sync()

    def _index_at(self, x, y):
        col, row = int(x) // self.CELL, int(y) // self.CELL
        if not (0 <= col < self.COLS):
            return None
        index = row * self.COLS + col
        return index if 0 <= index < LENGTH else None

    def _clicked(self, event):
        pal = self.ctl.doc.palette if self.ctl.doc else None
        if pal is None:
            return
        pos = event.position()
        index = self._index_at(pos.x(), pos.y())
        if index is None:
            return
        if event.button() == Qt.RightButton:
            self.ctl.set_secondary(pal.rgba(index), index)
        else:
            self.ctl.set_primary(pal.rgba(index), index)
        note = pal.annotation_for(index)
        self.info.setText("Index %d%s" % (index, "  -  " + note[2] if note else ""))

    def _double_clicked(self, event):
        """Edit an entry in place. Every pixel using it recolours instantly.

        This is the payoff of keeping the index plane authoritative: the edit
        touches no pixel data at all, so it costs nothing in undo, cannot
        drift however many times it is repeated, and reaches exactly the
        pixels holding that index -- including ones whose current colour
        matches a different entry.
        """
        pal = self.ctl.doc.palette if self.ctl.doc else None
        if pal is None:
            return
        pos = event.position()
        index = self._index_at(pos.x(), pos.y())
        if index is None:
            return
        current = QColor(*pal.rgba(index))
        chosen = QColorDialog.getColor(current, self, "Palette index %d" % index,
                                       QColorDialog.ShowAlphaChannel)
        if not chosen.isValid():
            return
        self.ctl.set_palette_entry(index, (chosen.red(), chosen.green(),
                                           chosen.blue(), chosen.alpha()))
        self.sync()

    def sync(self, **_):
        pal = self.ctl.doc.palette if self.ctl.doc else None
        rows = LENGTH // self.COLS
        pix = QPixmap(self.COLS * self.CELL, rows * self.CELL)
        pix.fill(QColor(40, 40, 44))
        if pal is None:
            self.grid_label.setPixmap(pix)
            self.info.setText("No palette bound")
            return
        painter = QPainter(pix)
        for index in range(LENGTH):
            col, row = index % self.COLS, index // self.COLS
            x, y = col * self.CELL, row * self.CELL
            painter.fillRect(x, y, self.CELL - 1, self.CELL - 1,
                             QColor(*pal.rgba(index)))
            # Annotated and protected ranges are marked, not explained --
            # the core does not know what they mean.
            if index in pal.protected:
                painter.setPen(QColor(255, 255, 255, 180))
                painter.drawRect(x, y, self.CELL - 2, self.CELL - 2)
            if index == pal.transparent:
                painter.setPen(QColor(255, 0, 0, 200))
                painter.drawLine(x, y, x + self.CELL - 2, y + self.CELL - 2)
        painter.end()
        self.grid_label.setPixmap(pix)
        notes = ", ".join("%d-%d %s" % (lo, hi, label)
                          for lo, hi, label, _role in pal.annotations)
        self.info.setText("%s  -  %d protected%s"
                          % (pal.name or "palette", len(pal.protected),
                             ("\n" + notes) if notes else ""))


class HistoryDock(QDockWidget):
    """Clickable history. Selecting an entry travels to that point."""

    def __init__(self, controller, parent=None):
        super().__init__("History", parent)
        self.setObjectName("History")
        self.ctl = controller
        self._syncing = False
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._goto)
        self.setWidget(self.list)

    def sync(self, **_):
        if self.ctl.history is None:
            return
        self._syncing = True
        self.list.clear()
        self.list.addItem("(original)")
        for label in self.ctl.history_labels():
            self.list.addItem(label)
        self.list.setCurrentRow(self.ctl.history.position + 1)
        self._syncing = False

    def _goto(self, row):
        if self._syncing or self.ctl.history is None:
            return
        if row - 1 != self.ctl.history.position:
            self.ctl.history_goto(row - 1)


class FramesDock(QDockWidget):
    """The frame axis, as a strip or a grid.

    One panel serves both layouts because the axis is 1-D with a layout hint
    rather than genuinely two-dimensional -- a tile grid is metadata plus a
    different arrangement here, not a second axis threaded through the whole
    document model.
    """

    def __init__(self, controller, parent=None):
        super().__init__("Frames", parent)
        self.setObjectName("Frames")
        self.ctl = controller
        self._syncing = False

        body = QWidget()
        box = QVBoxLayout(body)

        self.list = QListWidget()
        self.list.setFlow(QListWidget.LeftToRight)
        self.list.setWrapping(True)
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.currentRowChanged.connect(self._row_changed)
        box.addWidget(self.list, 1)

        row = QHBoxLayout()
        for label, slot in (("Add", self._add), ("Duplicate", self._duplicate)):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        box.addLayout(row)

        self.info = QLabel("-")
        box.addWidget(self.info)
        self.setWidget(body)

    def sync(self, **_):
        if self.ctl.doc is None:
            return
        self._syncing = True
        self.list.clear()
        doc = self.ctl.doc
        for i, frame in enumerate(doc.frames):
            text = "%d" % (i + 1)
            if not frame.is_warm:
                text += "*"          # cold or unloaded; costs nothing resident
            item = QListWidgetItem(text)
            item.setToolTip("%s  (%d ms, %s)"
                            % (frame.name or frame.id, frame.duration_ms,
                               frame.residency))
            self.list.addItem(item)
        self.list.setCurrentRow(doc.current)
        warm = sum(1 for f in doc.frames if f.is_warm)
        self.info.setText("%s layout  -  %d frames, %d warm"
                          % (doc.axis_layout, len(doc.frames), warm))
        self._syncing = False

    def _row_changed(self, row):
        if self._syncing or row < 0:
            return
        self.ctl.set_current_frame(row)
        self.sync()

    def _add(self):
        self.ctl.add_frame(copy_current=False)
        self.sync()

    def _duplicate(self):
        self.ctl.add_frame(copy_current=True)
        self.sync()
