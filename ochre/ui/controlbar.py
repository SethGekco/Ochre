# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The control server inside the window: the person it can ask.

    python -m ochre.ui --control --control-tier edit --control-root ~/agent

control.py decides; this supplies what only a window has. The approval bar
puts a question across the top of the window -- an agent wants to go beyond
its tier, or touch a file outside its folders -- with Allow and Deny. Allow
arms after half a second, so a click already on its way when the bar
appears cannot approve something nobody read; no answer in a minute is a
no. A request that would change the document waits its turn while the
person is mid-action (a dialog or menu open, a mouse button held, a stroke
in progress) and is answered `human_busy` instead. Kooka Pictura's
control_approval_bar.cpp and control_server_gate.cpp are the models.

The server is driven from the event loop by socket notifiers; nothing runs
on another thread.
"""

import sys

from PySide6.QtCore import QSocketNotifier, Qt, QTimer
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QLabel,
                               QPushButton)

from . import control

ARM_DELAY_MS = 500
TIMEOUT_MS = 60000

FLAGS = {"--control-tier": "tier", "--control-root": "roots", "--control-socket": "socket"}


class ApprovalBar(QFrame):
    """One question at a time, across the top of the window."""

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("controlApprovalBar")
        self.setFrameShape(QFrame.StyledPanel)
        self.setAutoFillBackground(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.label = QLabel(self)
        # The text comes from an agent (paths, command lines): it must not be
        # able to style itself, or to look like something else.
        self.label.setTextFormat(Qt.PlainText)
        self.label.setWordWrap(True)
        self.allow = QPushButton("Allow", self)
        self.deny = QPushButton("Deny", self)
        for button in (self.allow, self.deny):
            button.setFocusPolicy(Qt.NoFocus)
        lay = QHBoxLayout(self)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.allow)
        lay.addWidget(self.deny)
        self._answer = None
        self._arm = QTimer(self)
        self._arm.setSingleShot(True)
        self._arm.timeout.connect(lambda: self.allow.setEnabled(True))
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(lambda: self._finish("timeout"))
        self.allow.clicked.connect(lambda: self._finish("allow"))
        self.deny.clicked.connect(lambda: self._finish("deny"))
        self.hide()

    def ask(self, question, answer):
        """Show the question; answer("allow" | "deny" | "timeout") later."""
        if self._answer is not None:          # the server asks one at a time
            self._finish("deny")
        self._answer = answer
        self.label.setText(question)
        self.allow.setEnabled(False)
        self._arm.start(ARM_DELAY_MS)
        self._timeout.start(TIMEOUT_MS)
        window = self.parentWidget()
        if window is not None:
            self.adjustSize()
            width = min(window.width(), max(self.sizeHint().width(), 480))
            self.setGeometry((window.width() - width) // 2, 0, width, self.sizeHint().height())
            QApplication.alert(window)
        self.show()
        self.raise_()

    def question(self):
        return self.label.text() if self.isVisible() else None

    def _finish(self, verdict):
        answer, self._answer = self._answer, None
        self._arm.stop()
        self._timeout.stop()
        self.hide()
        if answer is not None:
            answer(verdict)


class WindowHost:
    """What the gate asks of the window."""

    def __init__(self, window):
        self.window = window
        self.bar = ApprovalBar(window)

    def busy_reason(self):
        if QApplication.activeModalWidget() is not None or QApplication.activePopupWidget():
            return "a dialog or menu is open"
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QDialog) and widget.isVisible():
                return "a dialog is open"
        if QApplication.mouseButtons() != Qt.NoButton:
            return "a mouse button is held"
        tool = self.window.ctl.tool
        if tool is not None and getattr(tool, "active", False):
            return "a tool is in use"
        return None

    def ask(self, question, answer):
        self.bar.ask(question, answer)

    def refresh(self, method):
        w = self.window
        w.canvas.bind(keep_view=method not in ("open", "new"))
        w.canvas.refresh()
        w.canvas.viewport().update()
        w._sync_all()


class WindowControl:
    """A ControlServer pumped by the window's event loop."""

    def __init__(self, server):
        self.server = server
        self._notifiers = {}
        self._sync()

    def _activated(self, *_):
        self.server.poll(0)
        self._sync()

    def _sync(self):
        live = set(self.server.fds())
        for fd in [fd for fd in self._notifiers if fd not in live]:
            notifier = self._notifiers.pop(fd)
            notifier.setEnabled(False)
            notifier.deleteLater()
        for fd in live - set(self._notifiers):
            notifier = QSocketNotifier(fd, QSocketNotifier.Read)
            notifier.activated.connect(self._activated)
            self._notifiers[fd] = notifier

    def close(self):
        for notifier in self._notifiers.values():
            notifier.setEnabled(False)
        self._notifiers.clear()
        self.server.close()


def take_flags(argv):
    """Pull --control and its settings out of argv. Returns (settings or
    None, the rest), so a folder given to --control-root is never mistaken
    for a file to open."""
    rest, opts, on = [], {"tier": "read", "roots": [], "socket": None}, False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--control":
            on = True
        elif arg in FLAGS and i + 1 < len(argv):
            key = FLAGS[arg]
            i += 1
            if key == "roots":
                opts["roots"].append(argv[i])
            else:
                opts[key] = argv[i]
        else:
            rest.append(arg)
        i += 1
    return (opts if on else None), rest


def start(window, controller, opts, state_dir):
    """Listen for the window. Returns a WindowControl, or None after saying
    why on stderr -- a control socket that cannot start must not stop the
    editor from opening."""
    try:
        policy = control.Policy(opts["tier"], opts["roots"], denied=[state_dir], live=True)
        gate = control.Gate(controller, policy, WindowHost(window))
        server = control.ControlServer(gate, opts["socket"]).listen()
    except (ValueError, OSError) as exc:
        print("ochre control: not started: %s" % (getattr(exc, "strerror", None) or exc),
              file=sys.stderr)
        return None
    print("ochre control: %s" % server.socket_path, file=sys.stderr)
    print("ochre control token: %s" % server.token_path, file=sys.stderr)
    sys.stderr.flush()
    return WindowControl(server)
