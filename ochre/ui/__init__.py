# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The UI layer. Nothing in ochre/engine/ may import from here.

Not everything in this package needs Qt, and that is deliberate rather than
incidental: `viewstate`, `controller`, `bus` and `config` are Qt-free so the
whole application can be driven headlessly, scripted, or tested without a
display. Only `app`, `canvas`, `docks` and `mainwindow` touch PySide6.

`main` is therefore resolved LAZILY. A plain `from .app import main` here
would import Qt the moment anyone touched any module in this package, which
would quietly destroy that property -- and the failure would look like an
unrelated ImportError on a machine without Qt installed.
"""

__all__ = ["main"]


def __getattr__(name):
    # PEP 562. Keeps `from ochre.ui import main` working while leaving
    # `import ochre.ui.viewstate` free of Qt.
    if name == "main":
        from .app import main as _main
        return _main
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(list(globals()) + __all__)
