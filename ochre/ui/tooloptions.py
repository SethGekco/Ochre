# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The tool options bar, generated from the active tool's options_spec.

Paint.NET puts tool options in a strip under the menu and GIMP in a dock
under the toolbox; the strip wastes less of the canvas, so that is what this
is. Nothing here knows any tool by name: it reads the descriptors, builds a
widget per option, and writes changes back through the controller -- which
remembers them per tool, so switching to the eraser and back does not reset
the brush.

Widgets update from the controller's "tool.options" event as well as feeding
it, so a size changed with [ and ] shows up here immediately. Signals are
blocked during those updates; otherwise every external change would echo
back as a fresh user edit.
"""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFontComboBox, QGridLayout, QLabel, QLineEdit,
                               QMenu, QSpinBox, QToolBar, QToolButton,
                               QVBoxLayout, QWidget, QWidgetAction)

from ..engine.tools import options as O
from . import icons


class ShapePalette(QToolButton):
    """A categorised grid of choices with icons -- the shape chooser."""

    def __init__(self, opt, value, on_pick, parent=None):
        super().__init__(parent)
        self.opt = opt
        self.on_pick = on_pick
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setIconSize(QSize(20, 20))
        self.buttons = {}
        menu = QMenu(self)
        holder = QWidget(menu)
        box = QVBoxLayout(holder)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(2)
        per_row = 9
        for heading, choices in opt.groups:
            title = QLabel(heading)
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            box.addWidget(title)
            grid = QGridLayout()
            grid.setSpacing(2)
            for i, choice in enumerate(choices):
                btn = QToolButton(holder)
                btn.setIcon(icons.shape_icon(choice, 28))
                btn.setIconSize(QSize(28, 28))
                btn.setToolTip(opt.label_for(choice))
                btn.setAutoRaise(True)
                btn.setCheckable(True)
                btn.clicked.connect(lambda _c=False, c=choice: self._pick(c))
                grid.addWidget(btn, i // per_row, i % per_row)
                self.buttons[choice] = btn
            # Without a stretch column a short row (four arrows) spreads its
            # icons across the width of the longest one.
            grid.setColumnStretch(per_row, 1)
            box.addLayout(grid)
        action = QWidgetAction(menu)
        action.setDefaultWidget(holder)
        menu.addAction(action)
        self.setMenu(menu)
        self.set_value(value)

    def _pick(self, choice):
        self.menu().hide()
        self.set_value(choice)
        self.on_pick(choice)

    def set_value(self, choice):
        self.value = choice
        self.setIcon(icons.shape_icon(choice, 20))
        self.setText(self.opt.label_for(choice))
        for c, b in self.buttons.items():
            b.setChecked(c == choice)


class ToolOptionsBar(QToolBar):
    def __init__(self, controller, parent=None):
        super().__init__("Tool Options", parent)
        self.setObjectName("ToolOptions")
        self.setMovable(True)
        self.setIconSize(QSize(20, 20))
        self.ctl = controller
        self.widgets = {}            # option name -> (opt, widget)
        self._syncing = False
        self.rebuild()

    # ---- building ----------------------------------------------------------

    def rebuild(self, **_):
        self.clear()
        self.widgets = {}
        name = self.ctl.tool_name
        if not name:
            return
        tools = self.ctl.tools
        head = QToolButton(self)
        head.setIcon(icons.tool_icon(tools.icon(name)))
        head.setText(tools.label(name))
        head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        head.setAutoRaise(True)
        head.setEnabled(False)
        font = head.font()
        font.setBold(True)
        head.setFont(font)
        self.addWidget(head)
        self.addSeparator()

        values = self.ctl.tool_option_values(name)
        for opt in tools.spec(name):
            value = values.get(opt.name, opt.default)
            widget = self._make(opt, value)
            if widget is None:
                continue
            if opt.kind not in (O.BOOL,) and opt.groups is None:
                label = QLabel(" %s " % opt.label)
                label.setToolTip(opt.tip)
                self.addWidget(label)
            widget.setToolTip(opt.tip or opt.label)
            self.addWidget(widget)
            self.widgets[opt.name] = (opt, widget)
        if not tools.spec(name):
            hint = QLabel("  No options for this tool.")
            hint.setEnabled(False)
            self.addWidget(hint)

    def _make(self, opt, value):
        if opt.kind == O.BOOL:
            w = QCheckBox(opt.label)
            w.setChecked(bool(value))
            w.toggled.connect(lambda v, n=opt.name: self._set(n, bool(v)))
            return w
        if opt.kind == O.INT:
            w = QSpinBox()
            w.setRange(int(opt.minimum if opt.minimum is not None else -10 ** 6),
                       int(opt.maximum if opt.maximum is not None else 10 ** 6))
            if opt.suffix:
                w.setSuffix(opt.suffix)
            w.setValue(int(value))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, n=opt.name: self._set(n, int(v)))
            return w
        if opt.kind == O.FLOAT and opt.percent:
            w = QSpinBox()
            w.setRange(int(round((opt.minimum or 0) * 100)), int(round((opt.maximum or 1) * 100)))
            w.setSuffix(" %")
            w.setValue(int(round(float(value) * 100)))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, n=opt.name: self._set(n, v / 100.0))
            return w
        if opt.kind == O.FLOAT:
            w = QDoubleSpinBox()
            w.setDecimals(opt.decimals)
            w.setRange(float(opt.minimum if opt.minimum is not None else -1e6),
                       float(opt.maximum if opt.maximum is not None else 1e6))
            w.setSingleStep(float(opt.step or 0.1))
            w.setValue(float(value))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, n=opt.name: self._set(n, float(v)))
            return w
        if opt.kind == O.CHOICE and opt.groups:
            return ShapePalette(opt, value, lambda c, n=opt.name: self._set(n, c), self)
        if opt.kind == O.CHOICE:
            w = QComboBox()
            for c in opt.choices or ():
                w.addItem(opt.label_for(c), c)
            idx = w.findData(value)
            w.setCurrentIndex(max(0, idx))
            w.currentIndexChanged.connect(
                lambda _i, n=opt.name, cb=w: self._set(n, cb.currentData()))
            return w
        if opt.kind == O.FONT:
            w = QFontComboBox()
            w.setCurrentFont(QFont(str(value)))
            w.setMaximumWidth(200)
            w.currentFontChanged.connect(lambda f, n=opt.name: self._set(n, f.family()))
            return w
        if opt.kind == O.TEXT:
            w = QLineEdit(str(value or ""))
            w.setMaximumWidth(200)
            w.editingFinished.connect(lambda n=opt.name, le=w: self._set(n, le.text()))
            return w
        return None

    # ---- data flow ---------------------------------------------------------

    def _set(self, name, value):
        if self._syncing:
            return
        self.ctl.set_tool_option(name, value)

    def sync_values(self, name=None, key=None, value=None, **_):
        """Reflect an option changed elsewhere ([ / ] keys, a script)."""
        if name is not None and name != self.ctl.tool_name:
            return
        self._syncing = True
        try:
            values = self.ctl.tool_option_values(self.ctl.tool_name)
            for opt_name, (opt, w) in self.widgets.items():
                if key is not None and opt_name != key:
                    continue
                v = values.get(opt_name, opt.default)
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, ShapePalette):
                    w.set_value(v)
                elif isinstance(w, QComboBox) and not isinstance(w, QFontComboBox):
                    w.setCurrentIndex(max(0, w.findData(v)))
                elif isinstance(w, QFontComboBox):
                    w.setCurrentFont(QFont(str(v)))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(v))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(round(float(v) * 100)) if opt.percent else int(v))
                elif isinstance(w, QLineEdit):
                    w.setText(str(v or ""))
        finally:
            self._syncing = False
