# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Help > Search (Shift+Space): find any command by name and run it.

With some four hundred menu commands, a hundred tools and a dozen panels,
"where is it?" is the real question. This answers it the way Kooka
Pictura's search bar does: type part of a name, see every match with the
place it lives ("Image > Canvas Size"), press Enter to run it.

What is searched:

- every command in the menu bar, with its menu path and shortcut;
- every tool, from the tool registry (addon tools included);
- every panel: the docks and the addons' toolbars and docks;
- every Preferences page, findable by the names of its settings too;
- every addon command, from the addon registry.

The index is rebuilt from the LIVE window each time the popup opens, so
there is nothing to keep in sync: an addon's command is found while that
addon is loaded and not after it is unloaded, and a command that cannot
run right now (Edit Mask with no mask) is listed, greyed, and refuses
Enter rather than vanishing -- you learn it exists and that it is not for
now.

Ranking (Kooka's tiers, then two additions): every query word must occur
in the path or the keywords; then an exact label beats a label prefix,
which beats every word starting a label word, then words inside the
label, then a match only in the path or keywords; after those, a fuzzy
match ("gblur" finds Gaussian Blur). Inside a tier, what you ran recently
comes first. Recent picks are kept in <state dir>/search_recent.txt, and
an empty query lists them.
"""

import os
import re

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeySequence, QPalette
from PySide6.QtWidgets import (QApplication, QDockWidget, QFrame, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QStyle, QStyledItemDelegate,
                               QVBoxLayout)

RUNNABLE_ROLE = Qt.UserRole + 1
SHORTCUT_ROLE = Qt.UserRole + 2
MAX_VISIBLE_ROWS = 12
MAX_RESULTS = 60

# Cheap synonyms: a label containing the key is also found by these words.
# Spelling variants first, so "colour" and "grey" find the American labels.
SYNONYMS = {
    "color": "colour", "gray": "grey", "center": "centre",
    "grayscale": "greyscale black white monochrome desaturate",
    "resize": "scale image size dimensions",
    "canvas size": "expand border extend",
    "crop": "trim cut",
    "undo": "revert back",
    "redo": "again",
    "preferences": "settings options",
    "flatten": "merge",
    "merge": "combine join",
    "export as": "save for web",
    "hue": "tint",
    "brightness": "light dark exposure",
    "blur": "soften smooth",
    "sharpen": "crisp detail",
    "invert": "negative",
    "eraser": "rubber",
    "eyedropper": "picker sample",
    "type": "text font",
    "pen": "bezier path",
    "marquee": "rectangle select",
    "lasso": "freehand select",
    "magic wand": "select similar",
    "clone": "stamp copy",
    "liquify": "warp push",
    "indexed": "palette quantize",
    "shortcuts": "keys keyboard hotkeys",
}

_SEPARATORS = re.compile(r"[^\w]+")


def clean_label(text):
    """A menu text as a person reads it: no "&" mnemonics (a doubled "&&"
    is a real ampersand), no tab-separated shortcut column, no trailing
    ellipsis."""
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "&" and i + 1 < len(text) and not text[i + 1].isspace():
            if text[i + 1] == "&":
                out.append("&")
                i += 1
            i += 1
            continue
        if c == "\t":
            break
        out.append(c)
        i += 1
    label = "".join(out).strip()
    while True:
        if label.endswith("…"):
            label = label[:-1].rstrip()
        elif label.endswith("..."):
            label = label[:-3].rstrip()
        else:
            return label


class Entry:
    """One searchable thing. `key` is stable across runs (for recents);
    `run` is what Enter does."""

    __slots__ = ("key", "kind", "path", "keywords", "shortcut", "runnable", "run")

    def __init__(self, key, kind, path, run, keywords=(), shortcut="", runnable=True):
        self.key = key
        self.kind = kind
        self.path = [clean_label(p) for p in path]
        self.keywords = [k.lower() for k in keywords if k]
        self.shortcut = shortcut or ""
        self.runnable = bool(runnable)
        self.run = run
        label = " ".join(w for w in _SEPARATORS.split(self.label.lower()) if w)
        for word, extra in SYNONYMS.items():
            # Whole words (plural allowed): "pen" must not tag "Open".
            if re.search(r"\b%ss?\b" % re.escape(word), label):
                self.keywords.extend(extra.split())

    @property
    def label(self):
        return self.path[-1] if self.path else ""

    @property
    def where(self):
        return " > ".join(self.path)

    def __repr__(self):
        return "<Entry %s %r>" % (self.kind, self.where)


def _fuzzy_span(label, query):
    """Length of the shortest greedy subsequence match of `query` in
    `label` that starts at a word start, or None. Spaces are ignored on
    both sides, so "gblur" finds "Gaussian Blur" (span 12) and
    "lvls" finds "Levels"."""
    if len(query) < 2:
        return None
    best = None
    for start in range(len(label)):
        if label[start] != query[0] or (start and label[start - 1].isalnum()):
            continue
        j = start
        for ch in query:
            j = label.find(ch, j)
            if j < 0:
                break
            j += 1
        else:
            span = j - start
            if best is None or span < best:
                best = span
    return best


def match_tier(entry, words, phrase):
    """(tier, tie-break) for an entry, lower is better; None if it does
    not match. Tiers: 0 exact label, 1 label prefix, 2 every word starts a
    label word, 3 every word inside the label, 4 words found only in the
    path or keywords, 5 fuzzy."""
    label = entry.label.lower()
    hay = [p.lower() for p in entry.path] + entry.keywords
    if all(any(w in h for h in hay) for w in words):
        if label == phrase:
            return (0, 0)
        if label.startswith(phrase):
            return (1, 0)
        label_words = [w for w in _SEPARATORS.split(label) if w]
        if all(any(lw.startswith(w) for lw in label_words) for w in words):
            return (2, 0)
        if all(w in label for w in words):
            return (3, 0)
        return (4, 0)
    span = _fuzzy_span(label.replace(" ", ""), phrase.replace(" ", ""))
    return None if span is None else (5, span)


def search(entries, query, recent=(), limit=MAX_RESULTS):
    """Entries matching `query`, best first. An empty query lists the
    recently run entries, newest first."""
    rank = {key: i for i, key in enumerate(recent)}
    words = [w for w in query.lower().split() if w]
    if not words:
        hits = [e for e in entries if e.key in rank]
        hits.sort(key=lambda e: rank[e.key])
        return hits[:limit]
    phrase = " ".join(words)
    scored = []
    for i, entry in enumerate(entries):
        tier = match_tier(entry, words, phrase)
        if tier is not None:
            scored.append((tier, rank.get(entry.key, len(rank)), not entry.runnable,
                           len(entry.path), i, entry))
    scored.sort(key=lambda t: t[:5])
    return [t[-1] for t in scored[:limit]]


class RecentCommands:
    """Keys of recently run entries, newest first; one per line in
    <state dir>/search_recent.txt. In memory only without a state dir."""

    def __init__(self, state_dir, limit=30):
        self.path = os.path.join(state_dir, "search_recent.txt") if state_dir else None
        self.limit = int(limit)
        self.items = []
        if self.path and os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.items = [l.rstrip("\n") for l in f if l.strip()][:self.limit]
            except OSError:
                pass

    def add(self, key):
        self.items = ([key] + [k for k in self.items if k != key])[:self.limit]
        if self.path:
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write("\n".join(self.items) + "\n")
            except OSError:
                pass


# ---- the index ----------------------------------------------------------

def _shortcut_text(action):
    seq = action.shortcut()
    if not seq.isEmpty():
        return seq.toString(QKeySequence.NativeText)
    # A shortcut shown but not bound (the Tools menu's) rides after a tab.
    text = action.text()
    return text.split("\t", 1)[1] if "\t" in text else ""


def _menu_entries(win):
    out = []
    skip_menus = {m for m in (getattr(win, "recent_menu", None),
                              getattr(win, "addon_commands_menu", None)) if m is not None}
    skip_actions = {getattr(win, "search_action", None)}
    skip_actions.update(d.toggleViewAction() for d in getattr(win, "panel_docks", ()))

    def walk(menu, path, enabled):
        # Menus refresh their enabled states as they open (the Layer
        # menu's mask items do); give each the same chance before reading.
        menu.aboutToShow.emit()
        for act in menu.actions():
            if act.isSeparator() or not act.isVisible() or act in skip_actions:
                continue
            text = clean_label(act.text())
            if not text:
                continue
            sub = act.menu()
            if sub is not None:
                if sub not in skip_menus:
                    walk(sub, path + [text], enabled and act.isEnabled())
                continue
            data = act.data()
            if isinstance(data, str) and data.startswith("tool:"):
                continue                    # indexed from the tool registry instead
            out.append(Entry("menu:" + "/".join(path + [text]), "command", path + [text],
                             act.trigger, shortcut=_shortcut_text(act),
                             runnable=enabled and act.isEnabled()))

    for act in win.menuBar().actions():
        if act.menu() is not None and act.isVisible() and act.menu() not in skip_menus:
            walk(act.menu(), [clean_label(act.text())], act.isEnabled())
    return out


def _tool_entries(win):
    from .mainwindow import GROUP_LABELS
    tools = win.ctl.tools
    out = []
    for group, members in tools.groups():
        where = ["Tools"]
        if len(members) > 1:
            where.append(GROUP_LABELS.get(group, group.replace("_", " ").title()))
        for name in members:
            out.append(Entry("tool:" + name, "tool", where + [tools.label(name)],
                             lambda n=name: win.ctl.set_tool(n),
                             keywords=("tool", name.replace("_", " "), group),
                             shortcut=tools.shortcut(name) or ""))
    return out


def show_panel(holder):
    """Show a dock or toolbar and bring it to the front of its tab group.
    Triggering its toggle action is what the Window menu does, so the
    window learns the user asked for it (the frame strip's pin)."""
    toggle = holder.toggleViewAction()
    if not toggle.isChecked():
        toggle.trigger()
    holder.setVisible(True)
    holder.raise_()


def _panel_entries(win):
    out = []
    for dock in getattr(win, "panel_docks", ()):
        title = clean_label(dock.windowTitle())
        out.append(Entry("panel:" + (dock.objectName() or title), "panel", ["Window", title],
                         lambda d=dock: show_panel(d), keywords=("panel",)))
    for name, holder in getattr(win, "_addon_holders", {}).items():
        widget = win.addon_panels.get(name)
        title = clean_label(holder.windowTitle() or getattr(widget, "label", "") or name)
        kind = "panel" if isinstance(holder, QDockWidget) else "toolbar"
        out.append(Entry("panel:addon:" + name, "panel", ["Window", title],
                         lambda h=holder: show_panel(h), keywords=(kind, "panel", "addon")))
    return out


def _preference_entries(win):
    from ..engine.settings import DEFAULTS
    out = []
    for section, keys in DEFAULTS.items():
        # The setting names as keywords: "DefaultTool" finds its page.
        words = ["settings", "preferences"] + [k.lower() for k in keys]
        out.append(Entry("prefs:" + section, "preferences", ["Edit", "Preferences", section],
                         lambda s=section: win.do_preferences(page=s), keywords=words))
    return out


def _addon_entries(win):
    reg = win.ctl.addons.registry
    out = []
    for name, provider in sorted(reg.all("commands").items()):
        owner = reg.owner("commands", name)
        addon = win.ctl.addons.addons.get(owner)
        out.append(Entry("addon:" + name, "addon", ["Addons", "Commands",
                                                     getattr(provider, "label", name)],
                         lambda n=name: win.run_addon_command(n),
                         keywords=("addon", name, getattr(provider, "category", ""),
                                   addon.name if addon is not None else owner or "")))
    return out


def build_index(win):
    """Every searchable entry in the window, as it is right now."""
    return (_menu_entries(win) + _tool_entries(win) + _panel_entries(win)
            + _preference_entries(win) + _addon_entries(win))


# ---- the popup ----------------------------------------------------------

class _ResultDelegate(QStyledItemDelegate):
    """Greys a row that cannot run, and draws its shortcut at the right."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if not index.data(RUNNABLE_ROLE):
            option.state &= ~QStyle.State_Enabled

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        shortcut = index.data(SHORTCUT_ROLE)
        if not shortcut:
            return
        enabled = bool(index.data(RUNNABLE_ROLE))
        selected = bool(option.state & QStyle.State_Selected)
        group = QPalette.Normal if enabled else QPalette.Disabled
        role = QPalette.HighlightedText if selected else QPalette.Text
        painter.save()
        painter.setPen(option.palette.color(group, role))
        painter.drawText(option.rect.adjusted(0, 0, -8, 0), Qt.AlignRight | Qt.AlignVCenter,
                         shortcut)
        painter.restore()


_BAR_KEYS = (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape, Qt.Key_Up, Qt.Key_Down,
             Qt.Key_PageUp, Qt.Key_PageDown)


class CommandSearch(QFrame):
    """The popup: a field and a list, floating over the top of the window.

    A child frame rather than a separate window, so it needs no window
    manager and behaves the same offscreen. While it is open an event
    filter on the application routes Enter, Esc and the arrows from the
    field to the list, and a click anywhere else closes it.
    """

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setObjectName("commandSearch")
        self.setFrameShape(QFrame.StyledPanel)
        self.setAutoFillBackground(True)
        self.hide()
        self.entries = []
        self.rows = []
        self.recent = RecentCommands(getattr(win.ctl, "state_dir", None))
        self._previous_focus = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(6)
        self.field = QLineEdit(self)
        self.field.setPlaceholderText("Search commands, tools, panels and settings")
        self.field.setClearButtonEnabled(True)
        lay.addWidget(self.field)
        self.results = QListWidget(self)
        # The field keeps the focus; a click on a row runs it.
        self.results.setFocusPolicy(Qt.NoFocus)
        self.results.setUniformItemSizes(True)
        self.results.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.results.setItemDelegate(_ResultDelegate(self.results))
        lay.addWidget(self.results)
        self.empty = QLabel("No matches", self)
        lay.addWidget(self.empty)
        footer = QLabel("Enter run  ·  Up/Down choose  ·  Esc close", self)
        footer.setEnabled(False)
        lay.addWidget(footer)
        self.field.textChanged.connect(self.update_results)
        self.results.itemClicked.connect(lambda item: self.activate(self.results.row(item)))

    # ---- opening and closing ---------------------------------------------

    def open(self):
        self.entries = build_index(self.win)
        self._previous_focus = QApplication.focusWidget()
        self.field.clear()
        self.update_results()
        QApplication.instance().installEventFilter(self)
        self.show()
        self.raise_()
        self.field.setFocus(Qt.ShortcutFocusReason)

    def dismiss(self):
        if not self.isVisible():
            return
        QApplication.instance().removeEventFilter(self)
        # Hand the focus back before hiding: a hidden field that kept it
        # would take the next Shift+Space as a typed space.
        if self.field.hasFocus():
            prev = self._previous_focus
            if prev is not None and prev is not self.field and prev.isVisible():
                prev.setFocus(Qt.OtherFocusReason)
            else:
                self.win.setFocus(Qt.OtherFocusReason)
        self.hide()

    # ---- results ----------------------------------------------------------

    def update_results(self, *_):
        self.results.clear()
        self.rows = search(self.entries, self.field.text(), self.recent.items)
        for entry in self.rows:
            item = QListWidgetItem(entry.where, self.results)
            item.setData(RUNNABLE_ROLE, entry.runnable)
            item.setData(SHORTCUT_ROLE, entry.shortcut)
            if not entry.runnable:
                item.setToolTip("Not available right now")
        if self.rows:
            self.results.setCurrentRow(0)
        self._relayout()

    def _relayout(self):
        typed = bool(self.field.text().strip())
        n = min(len(self.rows), MAX_VISIBLE_ROWS)
        self.results.setVisible(n > 0)
        self.empty.setVisible(typed and n == 0)
        if n:
            self.results.setFixedHeight(n * max(1, self.results.sizeHintForRow(0))
                                        + 2 * self.results.frameWidth())
        host = self.win.centralWidget().geometry() if self.win.centralWidget() else self.win.rect()
        self.setFixedWidth(max(240, min(640, host.width() * 3 // 5)))
        self.adjustSize()
        # Centred over the canvas, in its upper part.
        self.move(host.center().x() - self.width() // 2, host.top() + host.height() // 8)

    def current_entry(self):
        row = self.results.currentRow()
        return self.rows[row] if 0 <= row < len(self.rows) else None

    def _step(self, delta):
        if not self.rows:
            return
        row = max(0, min(len(self.rows) - 1, self.results.currentRow() + delta))
        self.results.setCurrentRow(row)
        self.results.scrollToItem(self.results.item(row))

    def activate(self, row=None):
        """Run the chosen row. One that cannot run now says so and stays."""
        row = self.results.currentRow() if row is None else row
        if not 0 <= row < len(self.rows):
            return False
        entry = self.rows[row]
        if not entry.runnable:
            self.win.statusBar().showMessage(
                "%s is not available right now." % entry.where, 4000)
            return False
        self.recent.add(entry.key)
        # Close first: what runs may open a modal dialog.
        self.dismiss()
        entry.run()
        return True

    # ---- keys and clicks --------------------------------------------------

    def _key(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.dismiss()
        elif key == Qt.Key_Up:
            self._step(-1)
        elif key == Qt.Key_Down:
            self._step(1)
        elif key == Qt.Key_PageUp:
            self._step(-MAX_VISIBLE_ROWS)
        elif key == Qt.Key_PageDown:
            self._step(MAX_VISIBLE_ROWS)
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self.activate()
        else:
            return False
        return True

    def eventFilter(self, watched, event):
        if not self.isVisible():
            return False
        kind = event.type()
        if watched is self.field and kind == QEvent.ShortcutOverride \
                and event.key() in _BAR_KEYS:
            # The field's keys must not reach a window shortcut.
            event.accept()
            return True
        if watched is self.field and kind == QEvent.KeyPress and self._key(event):
            return True
        if kind == QEvent.MouseButtonPress and hasattr(event, "globalPosition"):
            inside = self.rect().contains(self.mapFromGlobal(event.globalPosition().toPoint()))
            if not inside:
                self.dismiss()       # without consuming: the click still lands
        return False
