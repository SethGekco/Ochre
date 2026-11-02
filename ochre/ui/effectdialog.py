# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Effect dialogs, generated from an effect's declared properties.

No effect lays out a dialog. It declares typed properties with constraints
and rules, and the widgets are built from that description -- which is why
every effect looks and behaves consistently, and why an addon's effect gets
a proper dialog without shipping any UI code.

Two behaviours are worth calling out because they are what make a preview
feel responsive rather than merely correct:

Preview renders only the VISIBLE region, inflated by the effect's halo, and
never the whole document. A blur of a 4000x4000 canvas is seconds of work;
the part you are looking at is milliseconds.

A slider moving cancels the in-flight render rather than queueing another.
Without that, dragging a slider builds a backlog and the preview lags further
behind the further you drag.
"""

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QTabWidget,
                               QVBoxLayout, QWidget)

from ..engine.effects import CancelToken, Cancelled


class CurveEditor(QWidget):
    """A tone curve you drag: click to add a point, drag to move it, drag it
    out sideways past its neighbours (or right-click) to remove it. The ends
    stay at x = 0 and 255. Drawn through the same monotone interpolation the
    adjustment uses, so what you see is the curve you get."""

    changed = Signal()

    def __init__(self, points, parent=None):
        super().__init__(parent)
        self.points = [tuple(p) for p in points]
        self._grab = None
        self.setMinimumSize(170, 170)
        self.setMouseTracking(False)

    def set_points(self, points):
        pts = [tuple(p) for p in points]
        if pts != self.points:
            self.points = pts
            self.update()

    def _to_screen(self, x, y):
        w, h = self.width() - 1, self.height() - 1
        return x / 255.0 * w, h - y / 255.0 * h

    def _to_value(self, sx, sy):
        w, h = max(1, self.width() - 1), max(1, self.height() - 1)
        return (max(0, min(255, int(round(sx / w * 255)))),
                max(0, min(255, int(round((h - sy) / h * 255)))))

    def paintEvent(self, event):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPainter, QPainterPath, QPen
        from ..engine.colors import monotone_curve
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(250, 250, 250))
        p.setPen(QPen(QColor(220, 220, 220), 1))
        for i in range(1, 4):
            x, _ = self._to_screen(i * 64, 0)
            _, y = self._to_screen(0, i * 64)
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.drawLine(QPointF(0, y), QPointF(self.width(), y))
        p.setPen(QPen(QColor(200, 200, 200), 1, Qt.DashLine))
        p.drawLine(QPointF(*self._to_screen(0, 0)), QPointF(*self._to_screen(255, 255)))
        lut = monotone_curve(self.points)
        path = QPainterPath(QPointF(*self._to_screen(0, int(lut[0]))))
        for x in range(1, 256):
            path.lineTo(QPointF(*self._to_screen(x, int(lut[x]))))
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(30, 30, 30), 1.5))
        p.drawPath(path)
        p.setBrush(QColor(255, 255, 255))
        for x, y in self.points:
            sx, sy = self._to_screen(x, y)
            p.drawRect(int(sx) - 3, int(sy) - 3, 6, 6)
        p.end()

    def _nearest(self, sx, sy):
        best, idx = 64.0, None
        for i, (x, y) in enumerate(self.points):
            px, py = self._to_screen(x, y)
            d = (px - sx) ** 2 + (py - sy) ** 2
            if d <= best:
                best, idx = d, i
        return idx

    def mousePressEvent(self, event):
        pos = event.position()
        idx = self._nearest(pos.x(), pos.y())
        if event.button() == Qt.RightButton:
            if idx is not None and 0 < idx < len(self.points) - 1:
                del self.points[idx]
                self.update()
                self.changed.emit()
            return
        if idx is None:
            x, y = self._to_value(pos.x(), pos.y())
            if any(px == x for px, _ in self.points):
                return
            self.points.append((x, y))
            self.points.sort()
            idx = self.points.index((x, y))
            self.changed.emit()
        self._grab = idx
        self.update()

    def mouseMoveEvent(self, event):
        if self._grab is None:
            return
        i = self._grab
        x, y = self._to_value(event.position().x(), event.position().y())
        if i == 0:
            x = 0
        elif i == len(self.points) - 1:
            x = 255
        else:
            lo, hi = self.points[i - 1][0], self.points[i + 1][0]
            if x <= lo - 8 or x >= hi + 8:
                # Dragged out past a neighbour: remove it, as GIMP does.
                del self.points[i]
                self._grab = None
                self.update()
                self.changed.emit()
                return
            x = max(lo + 1, min(hi - 1, x))
        self.points[i] = (x, y)
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, event):
        self._grab = None


class _Row(QWidget):
    """One property, as a labelled control plus its live value."""

    changed = Signal()

    def __init__(self, prop, parent=None):
        super().__init__(parent)
        self.prop = prop
        self.widget = None
        box = QHBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self._build(box)
        self.refresh()

    def _build(self, box):
        prop = self.prop
        kind = prop.kind

        if kind in ("int", "float", "angle"):
            is_int = kind == "int"
            self.slider = QSlider(Qt.Horizontal)
            steps = 1 if is_int else 100
            self._steps = steps
            self.slider.setRange(int(prop.minimum * steps), int(prop.maximum * steps))
            self.spin = QSpinBox() if is_int else QDoubleSpinBox()
            self.spin.setRange(prop.minimum, prop.maximum)
            if not is_int:
                self.spin.setDecimals(int(prop.ui.get("decimals", 2)))
                self.spin.setSingleStep(0.05)
            self.slider.valueChanged.connect(self._from_slider)
            self.spin.valueChanged.connect(self._from_spin)
            box.addWidget(self.slider, 1)
            box.addWidget(self.spin)
            self.widget = self.slider

        elif kind == "bool":
            self.check = QCheckBox(prop.ui.get("description") or "")
            self.check.toggled.connect(self._from_check)
            box.addWidget(self.check, 1)
            self.widget = self.check

        elif kind == "curve":
            self.curve = CurveEditor(prop.value)
            self.curve.changed.connect(self._from_curve)
            box.addWidget(self.curve, 1)
            self.widget = self.curve

        elif kind == "choice":
            self.combo = QComboBox()
            labels = prop.ui.get("labels", {})
            for option in prop.choices:
                self.combo.addItem(str(labels.get(option, option)), option)
            self.combo.currentIndexChanged.connect(self._from_combo)
            box.addWidget(self.combo, 1)
            self.widget = self.combo

        elif kind == "color":
            self.button = QPushButton("Choose...")
            self.button.clicked.connect(self._pick_colour)
            box.addWidget(self.button, 1)
            self.widget = self.button

        elif kind == "text":
            # Free text; a property named "file" gets a Browse button.
            self.line = QLineEdit(str(prop.value))
            self.line.editingFinished.connect(self._from_line)
            box.addWidget(self.line, 1)
            if prop.name == "file":
                browse = QPushButton("...")
                browse.setFixedWidth(30)
                browse.clicked.connect(self._browse)
                box.addWidget(browse)
            self.widget = self.line

        else:
            self.label = QLabel(str(prop.value))
            box.addWidget(self.label, 1)
            self.widget = self.label

        reset = QPushButton("Reset")
        reset.setFixedWidth(56)
        reset.clicked.connect(self._reset)
        box.addWidget(reset)

    # ---- widget -> model -------------------------------------------------

    def _from_slider(self, raw):
        if self._guard():
            return
        self.prop.set(raw / self._steps if self.prop.kind != "int" else raw)
        self.refresh()
        self.changed.emit()

    def _from_spin(self, raw):
        if self._guard():
            return
        self.prop.set(raw)
        self.refresh()
        self.changed.emit()

    def _from_check(self, flag):
        if self._guard():
            return
        self.prop.set(flag)
        self.changed.emit()

    def _from_combo(self, _index):
        if self._guard():
            return
        self.prop.set(self.combo.currentData())
        self.changed.emit()

    def _from_curve(self):
        if self._guard():
            return
        self.prop.set(self.curve.points)
        self.changed.emit()

    def _from_line(self):
        if self._guard() or self.line.text() == self.prop.value:
            return
        self.prop.set(self.line.text())
        self.changed.emit()

    def _browse(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, self.prop.ui.get("label", "File"))
        if path:
            self.line.setText(path)
            self._from_line()

    def _pick_colour(self):
        chosen = QColorDialog.getColor(QColor(*self.prop.value), self,
                                       self.prop.ui.get("label", "Colour"),
                                       QColorDialog.ShowAlphaChannel)
        if chosen.isValid():
            self.prop.set((chosen.red(), chosen.green(), chosen.blue(),
                           chosen.alpha()))
            self.refresh()
            self.changed.emit()

    def _reset(self):
        self.prop.reset()
        self.refresh()
        self.changed.emit()

    _updating = False

    def _guard(self):
        return self._updating

    # ---- model -> widget -------------------------------------------------

    def refresh(self):
        """Pull the model's current state into the widgets.

        Rules can move properties the user did not touch -- linked sliders,
        a min/max pair pushing its partner -- so every row refreshes after
        any change, not just the one that was edited.
        """
        self._updating = True
        try:
            prop = self.prop
            if prop.kind in ("int", "float", "angle"):
                self.slider.setValue(int(round(prop.value * self._steps)))
                self.spin.setValue(prop.value)
            elif prop.kind == "bool":
                self.check.setChecked(bool(prop.value))
            elif prop.kind == "choice":
                index = self.combo.findData(prop.value)
                if index >= 0:
                    self.combo.setCurrentIndex(index)
            elif prop.kind == "curve":
                self.curve.set_points(prop.value)
            elif prop.kind == "color":
                colour = QColor(*prop.value)
                self.button.setStyleSheet("background-color: %s;" % colour.name())
            elif prop.kind == "text":
                if self.line.text() != str(prop.value):
                    self.line.setText(str(prop.value))
            # Read-only is a fact about the MODEL; the widget only reflects it.
            if self.widget is not None:
                self.setEnabled(not prop.readonly)
        finally:
            self._updating = False


class EffectDialog(QDialog):
    """Generated configuration dialog with live preview."""

    def __init__(self, effect, controller, parent=None, debounce_ms=80):
        super().__init__(parent)
        self.effect = effect
        self.ctl = controller
        self.setWindowTitle(effect.label)
        self._cancel = None
        self._committed = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.rows = []
        # An effect may bring its own panel for settings a plain row cannot
        # show (Selective Color's nine ranges); those props are then hidden.
        # A panel with a tab_title gets a tab of its own beside the rows'
        # "Custom" tab, as Lens Correction's Profile tab does.
        panel = PANELS.get(getattr(effect, "ui_panel", None))
        hidden = set(getattr(effect, "ui_hidden", ())) if panel else set()
        if panel is not None:
            self.panel = panel(effect, self._on_changed, controller)
        # Props tagged with a section (Camera Raw's panels) go on tabs, the
        # untagged ones on the first; otherwise one plain form.
        shown = [p for p in effect.props if p.name not in hidden]
        sections = {}
        section_tabs = None
        if any(p.ui.get("section") for p in shown):
            section_tabs = QTabWidget()
            for prop in shown:
                name = prop.ui.get("section") or "Basic"
                if name not in sections:
                    page = QWidget()
                    sections[name] = QFormLayout(page)
                    scroll = QScrollArea()
                    scroll.setWidgetResizable(True)
                    scroll.setWidget(page)
                    section_tabs.addTab(scroll, name)
        for prop in shown:
            row = _Row(prop)
            row.changed.connect(self._on_changed)
            target = sections.get(prop.ui.get("section") or "Basic", form)
            target.addRow(prop.ui.get("label", prop.name), row)
            self.rows.append(row)
        self.tabs = None
        if panel is not None and getattr(self.panel, "tab_title", None):
            self.tabs = QTabWidget()
            self.tabs.addTab(self.panel, self.panel.tab_title)
            custom = QWidget()
            custom.setLayout(form)
            self.tabs.addTab(custom, "Custom")
            layout.addWidget(self.tabs)
        else:
            if panel is not None:
                layout.addWidget(self.panel)
            if section_tabs is not None:
                layout.addWidget(section_tabs)
            layout.addLayout(form)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # The original pixels, kept so Cancel is an exact restore and each
        # preview starts from a clean source rather than from the last one.
        # The paint target: the active layer, or its mask while editing one.
        target = getattr(controller, "paint_target", controller.active_layer)
        self._target = target
        cell = controller.doc.cell(target) if target is not None else None
        self._cell = cell
        # The selection fences the effect, softly, as it fences every tool.
        sel = getattr(controller, "selection", None)
        self._sel = None if sel is None or sel.mask is None else sel.mask
        self._original = None if cell is None else cell.pixels.copy()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._render_preview)
        self._debounce = debounce_ms
        self._render_preview()

    # ---- preview ---------------------------------------------------------

    def _on_changed(self):
        for row in self.rows:
            row.refresh()          # rules may have moved other properties
        # Restart rather than queue: dragging a slider must not build a
        # backlog of renders the user no longer cares about.
        if self._cancel is not None:
            self._cancel.cancel()
        self._timer.start(self._debounce)

    def _preview_rect(self):
        """Only what is on screen, inflated by the effect's halo."""
        canvas = getattr(self.parent(), "canvas", None)
        if canvas is None:
            return self.ctl.doc.bounds
        vp = canvas.viewport()
        # A canvas that is not on screen, or whose viewport is still being
        # laid out, has no visible region to restrict to -- preview it all
        # rather than a sliver that happens to match a placeholder size.
        if not canvas.isVisible() or vp.width() < 64 or vp.height() < 64:
            return self.ctl.doc.bounds
        visible = canvas.view.visible_doc_rect(vp.width(), vp.height())
        if visible is None:
            return self.ctl.doc.bounds
        return visible.inflated(self.effect.halo).clipped_to(
            self.ctl.doc.width, self.ctl.doc.height) or self.ctl.doc.bounds

    def _render_preview(self):
        if self._cell is None or self._original is None:
            return
        rect = self._preview_rect()
        self._cancel = CancelToken()
        try:
            result = self.effect.apply(self._original, [rect],
                                       cancel=self._cancel)
        except Cancelled:
            return
        self._cell.plane("rgba")[rect.slice()] = self._fenced(result, rect)
        self._cell.dirty.add(rect)
        self.ctl.invalidate(rect)
        canvas = getattr(self.parent(), "canvas", None)
        if canvas is not None:
            canvas.refresh()
            canvas.viewport().update()

    # ---- commit / abandon ------------------------------------------------

    def accept(self):
        """Apply to the whole layer and record one undo entry.

        The previewed region is already correct, but the rest of the layer is
        not, so the commit renders everything. What it must NOT do is render
        from the previewed pixels -- it renders from the pristine original,
        or an effect would compound with itself.
        """
        if self._cell is None or self._original is None:
            return super().accept()
        if self._cancel is not None:
            self._cancel.cancel()

        # Restore first, so the undo entry captures the true "before", then
        # apply through the controller: one undo step, the selection and the
        # paint target honoured -- and recordable in an Action.
        self._cell.plane("rgba")[...] = self._original
        if hasattr(self.ctl, "apply_effect"):
            self.ctl.apply_effect(self.effect.name, dict(self.effect.props.token()))
        else:                                              # a bare controller (tests)
            from ..engine.commands import PixelDelta
            bounds = self.ctl.doc.bounds
            before = PixelDelta.capture(self.ctl.doc, self._target.id,
                                        self.ctl.doc.frame.id, bounds, self.effect.label)
            self._cell.lerp_masked(bounds, self.effect.apply(self._original), self._sel)
            self._cell._touched(bounds)
            self.ctl.history.push(before, self.effect.label, frame_id=self.ctl.doc.frame.id)
        self.ctl.invalidate()
        self.ctl.dirty_since_save = True
        self.ctl.emit("history.changed")
        self._committed = True
        super().accept()

    def _fenced(self, result, rect):
        """result over rect, limited to the selection (for the preview)."""
        win = result[rect.slice()]
        if self._sel is None:
            return win
        orig = self._original[rect.slice()].astype(np.float32)
        k = self._sel[rect.slice()].astype(np.float32)[..., None] / 255.0
        return np.clip(np.rint(orig + (win.astype(np.float32) - orig) * k), 0, 255).astype(np.uint8)

    def reject(self):
        if self._cancel is not None:
            self._cancel.cancel()
        if self._cell is not None and self._original is not None and not self._committed:
            self._cell.plane("rgba")[...] = self._original
            self._cell.dirty.add(self.ctl.doc.bounds)
            self.ctl.invalidate()
            canvas = getattr(self.parent(), "canvas", None)
            if canvas is not None:
                canvas.refresh()
                canvas.viewport().update()
        super().reject()


# ---- custom panels ------------------------------------------------------------------

class SelectiveColorPanel(QWidget):
    """Photoshop's Selective Color face: pick a colour range, set its four
    inks; every range is kept, and all apply together."""

    def __init__(self, effect, on_change, controller=None, parent=None):
        import json
        from ..engine.morefilters import RANGES
        super().__init__(parent)
        self.effect, self.on_change = effect, on_change
        self.values = {r: [0, 0, 0, 0] for r in RANGES}
        try:
            for r, v in json.loads(effect.props.get("ranges") or "{}").items():
                if r in self.values:
                    self.values[r] = [int(x) for x in v][:4]
        except (ValueError, AttributeError):
            pass
        form = QFormLayout(self)
        self.range = QComboBox()
        for r in RANGES:
            self.range.addItem(r.capitalize(), r)
        self.range.currentIndexChanged.connect(self._show)
        form.addRow("Colors", self.range)
        self.spins = []
        for label in ("Cyan", "Magenta", "Yellow", "Black"):
            spin = QSpinBox()
            spin.setRange(-100, 100)
            spin.setSuffix(" %")
            spin.valueChanged.connect(self._edit)
            form.addRow(label, spin)
            self.spins.append(spin)
        self._loading = False
        self._show()

    def _show(self, *_):
        self._loading = True
        for spin, v in zip(self.spins, self.values[self.range.currentData()]):
            spin.setValue(v)
        self._loading = False

    def _edit(self, *_):
        if self._loading:
            return
        import json
        self.values[self.range.currentData()] = [s.value() for s in self.spins]
        used = {r: v for r, v in self.values.items() if any(v)}
        self.effect.props.set("ranges", json.dumps(used))
        self.on_change()


class LensProfilePanel(QWidget):
    """Lens Correction's Profile tab (CS6's Auto Correction): a lens from
    lensfun's database, matched from the photo's EXIF where it can be,
    at the shot's focal length, aperture and focus distance, and which
    of its corrections to apply. With no database installed it says so
    and changes nothing; the Custom tab still works."""

    tab_title = "Profile"

    def __init__(self, effect, on_change, controller=None, parent=None):
        from ..engine import lensprofile, metadata
        super().__init__(parent)
        self.effect, self.on_change = effect, on_change
        self.db = lensprofile.installed()
        layout = QVBoxLayout(self)
        self.status = QLabel()
        self.status.setWordWrap(True)
        if not self.db:
            self.status.setText(
                "No lens database found. Lens profiles come from lensfun's database "
                "(the lensfun-data package, or lensfun-update-data), which Ochre reads "
                "where it is installed. Looked in: %s. The Custom tab works without it."
                % ", ".join(lensprofile.search_roots()))
            layout.addWidget(self.status)
            layout.addStretch(1)
            return
        meta = getattr(getattr(controller, "doc", None), "meta", None)
        self.exif = metadata.lens_exif(meta)
        form = QFormLayout()
        self.camera = QComboBox()
        self.camera.addItem("(Any camera)", None)
        for i, cam in sorted(enumerate(self.db.cameras), key=lambda p: p[1].name.lower()):
            self.camera.addItem("%s (crop %g)" % (cam.name, cam.crop), i)
        self.lens = QComboBox()
        self.lens.addItem("(No profile)", "")
        for name in sorted({l.name for l in self.db.lenses}, key=str.lower):
            self.lens.addItem(name, name)
        self.focal = self._spin(1.0, 2000.0, " mm", 1)
        self.aperture = self._spin(0.5, 64.0, "", 1, prefix="f/")
        self.distance = self._spin(0.01, 1000.0, " m", 2)
        form.addRow("Camera", self.camera)
        form.addRow("Lens", self.lens)
        form.addRow("Focal length", self.focal)
        form.addRow("Aperture", self.aperture)
        form.addRow("Focus distance", self.distance)
        self.checks = {}
        for name in ("profile_distortion", "profile_ca", "profile_vignette", "auto_scale"):
            box = QCheckBox(effect.props[name].ui.get("label", name))
            box.setChecked(bool(effect.props.get(name)))
            box.toggled.connect(lambda on, n=name: self._toggle(n, on))
            form.addRow("" if name != "profile_distortion" else "Correction", box)
            self.checks[name] = box
        layout.addLayout(form)
        layout.addWidget(self.status)
        self._loading = True
        self._prefill()
        self._loading = False
        for w in (self.camera, self.lens):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.focal, self.aperture, self.distance):
            w.valueChanged.connect(self._changed)
        # Set the profile now; the dialog renders its first preview itself.
        self._changed(notify=False)

    @staticmethod
    def _spin(lo, hi, suffix, decimals, prefix=""):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setPrefix(prefix)
        return spin

    def _prefill(self):
        """The camera, lens and shot settings the photo's EXIF names -- or,
        for a lens profile already chosen (a smart filter reopened), that."""
        from ..engine.lensprofile import AUTO_MATCH
        ex = self.exif
        cam = self.db.find_camera(ex["make"], ex["model"])
        if cam is not None:
            self.camera.setCurrentIndex(self.camera.findData(self.db.cameras.index(cam)))
        chosen = self.effect.props.get("profile") or ""
        if not chosen:
            matches = self.db.match_lenses(ex["lens"], ex["lens_make"] or ex["make"], cam)
            if matches and matches[0][0] >= AUTO_MATCH:
                chosen = matches[0][1].name
                self.matched = True
        index = self.lens.findData(chosen)
        self.lens.setCurrentIndex(max(0, index))
        lens = self.db.lens_by_name(chosen) if chosen else None
        self.focal.setValue(ex["focal"] or (lens.min_focal if lens else 50.0) or 50.0)
        self.aperture.setValue(ex["aperture"] or 8.0)
        self.distance.setValue(ex["distance"] or 1000.0)

    matched = False

    def _crop(self):
        """The camera's crop factor: the chosen body's, else the EXIF's
        35 mm equivalent over the real focal length, else unknown."""
        index = self.camera.currentData()
        if index is not None:
            return self.db.cameras[index].crop
        ex = self.exif
        if ex["focal"] and ex["focal35"]:
            return ex["focal35"] / ex["focal"]
        return None

    def _toggle(self, name, on):
        self.effect.props.set(name, on)
        self.on_change()

    def _changed(self, *_, notify=True):
        if self._loading:
            return
        from ..engine import lensprofile
        lens = self.db.lens_by_name(self.lens.currentData() or "")
        if lens is None:
            self.effect.props.set("profile", "")
            self.effect.props.set("profile_model", "")
            self.status.setText("No profile: only the Custom tab's corrections apply." if not
                                self.exif["lens"] else
                                "The lens in this photo's EXIF (%s) is not in the lens "
                                "database; pick one, or use the Custom tab." % self.exif["lens"])
        else:
            model = lensprofile.lens_model(lens, self.focal.value(), self.aperture.value(),
                                           self.distance.value(), self._crop())
            self.effect.props.set("profile", lens.name)
            self.effect.props.set("profile_model", lensprofile.model_json(model))
            has = [n for n, on in zip(("distortion", "chromatic aberration", "vignetting"),
                                      lens.corrects()) if on]
            self.status.setText("%s%s: lensfun calibrates %s." % (
                lens.name, " (matched from the photo's EXIF)" if self.matched else "",
                ", ".join(has)))
        if notify:
            self.on_change()


PANELS = {"selective_color": SelectiveColorPanel, "lens_correction": LensProfilePanel}
