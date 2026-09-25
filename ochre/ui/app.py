# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Application entry point."""

import os
import sys

from PySide6.QtWidgets import QApplication

from ..engine.settings import Settings
from .bus import EventBus
from .config import UiSettings
from .controller import Controller
from .mainwindow import MainWindow


def _root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _data_dir():
    return os.path.join(_root(), "data")


def _addon_dirs():
    """Where addons live: beside the app, and in the user's config dir."""
    user = os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
        "ochre", "addons")
    return [os.path.join(_root(), "addons"), user]


def _state_dir():
    return os.path.join(
        os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")),
        "ochre")


def _ask_trust(addon, digest):
    """Consent prompt for never-before-seen addon code.

    Deliberately blunt about what is being agreed to. An addon is arbitrary
    Python running with the user's privileges -- no dialog can make that
    safe, so the dialog says so rather than implying a sandbox exists.
    """
    # A modal dialog with nobody to answer it is a hang, not a prompt, and
    # "no" is the safe answer when there is no one to ask.
    if os.environ.get("OCHRE_SELFTEST") or os.environ.get("OCHRE_NO_PROMPT"):
        return False

    from PySide6.QtWidgets import QMessageBox
    box = QMessageBox()
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Load addon?")
    box.setText("Load the addon %r?" % (addon.manifest.name or addon.id))
    box.setInformativeText(
        "%s\n\nVersion %s by %s\nProvides: %s\n%s\n\n"
        "An addon is ordinary Python and runs with your privileges. Ochre "
        "cannot sandbox it. Load it only if you trust where it came from.\n\n"
        "You will be asked again if its contents change."
        % (addon.manifest.description or "",
           addon.manifest.version, addon.manifest.author or "unknown",
           ", ".join(addon.manifest.provides) or "unstated",
           addon.directory))
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    box.setDefaultButton(QMessageBox.No)
    return box.exec() == QMessageBox.Yes


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName("Ochre")
    app.setOrganizationName("Ochre")

    data = _data_dir()
    engine_settings = Settings(data)
    ui_settings = UiSettings(data)
    bus = EventBus()
    controller = Controller(engine_settings, data, bus,
                            addon_dirs=_addon_dirs(),
                            state_dir=_state_dir(),
                            trust_callback=_ask_trust)
    window = MainWindow(controller, ui_settings)

    # Load addons BEFORE anything opens a file. There is a menu item for this
    # too, and relying on it meant a format addon could not claim a file
    # passed on the command line: it fell through to Pillow and died on a
    # file the editor supports. Nothing caught it because every test calls
    # load_addons() explicitly.
    #
    # After the window exists, so the trust prompt has a parent to sit on.
    controller.load_addons()

    # Opening happens BEFORE the self-test branch, so running the self-test
    # with a file argument exercises the real open path. It used to come
    # after, which is why the self-test never noticed that the command line
    # could not open an addon format at all.
    for arg in argv[1:]:
        if os.path.exists(arg):
            controller.open_path(arg)
            window.canvas.bind()
            break

    # A self-test that only constructs widgets verifies almost nothing about
    # a canvas. This one forces a real synchronous paintEvent over a live
    # document, which exercises the numpy-to-QImage aliasing, the checker,
    # the nearest-neighbour path and drawImage -- precisely the code that
    # would segfault if the buffer-lifetime rules were broken.
    if os.environ.get("OCHRE_SELFTEST"):
        return _selftest(window, controller)

    window.show()
    return app.exec()


def _selftest(window, controller):
    from PySide6.QtWidgets import QDockWidget

    from ..engine import accel

    # findChildren with a concrete subclass finds only that subclass; the
    # base class is what enumerates every dock.
    docks = {d.objectName() for d in window.findChildren(QDockWidget)}
    assert {"Tools", "Layers", "Colors", "Palette", "History"} <= docks, docks
    assert controller.doc is not None, "no document"
    assert controller.tool is not None, "no tool"

    window.resize(900, 600)
    window.canvas.resize(800, 520)

    # Draw something through the real tool path, then force a paint.
    controller.set_tool("brush", size=24)
    controller.begin_stroke(40, 40)
    controller.motion_stroke(120, 90)
    controller.end_stroke(160, 120)
    rects = controller.flush()
    assert rects, "painting produced no dirty rects"

    window.canvas.repaint()
    assert window.canvas.surface.is_valid(), "canvas surface invalid after paint"
    assert window.canvas.surface.image.width() == controller.doc.width

    # The aliasing must be real: a numpy write must show through the QImage.
    controller.display[0, 0] = (1, 2, 3, 255)
    px = window.canvas.surface.image.pixelColor(0, 0)
    assert (px.red(), px.green(), px.blue()) == (1, 2, 3), "QImage is not aliasing"

    controller.undo()
    assert controller.history.position == -1, "undo did not rewind"

    status = controller.status()
    loaded = sorted(a.id for a in controller.addons.loaded())
    print("selftest OK: %d docks, %dx%d doc, %d layers, %d frames, tool=%s, "
          "accel=%s, %d dirty rects, aliasing verified, addons=%s, format=%s"
          % (len(docks), status["size"][0], status["size"][1], status["layers"],
             len(controller.doc.frames), status["tool"], accel.BACKEND,
             len(rects), ",".join(loaded) or "none",
             controller.doc.meta.get("format", "-")))
    return 0
