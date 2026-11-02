# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Choose between the window and the windowless modes before importing Qt.

`--headless` must work on a machine with no display and no
PySide6, so the decision is made here, Qt-free, and only the window path
imports app.py.
"""

import sys


def main(argv=None):
    argv = sys.argv if argv is None else argv
    flags = argv[1:]
    if "--headless" in flags:
        from .headless import main as run
    else:
        from .app import main as run
    return run(argv)
