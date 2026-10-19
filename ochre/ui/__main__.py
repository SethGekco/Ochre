# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Run the GUI with:  python -m ochre.ui"""

import os
import sys

from .app import main

code = main()
# Leave without running interpreter finalisation. Shiboken and Qt then
# destroy their C++ objects in an arbitrary order, and a known PySide hazard
# makes that segfault intermittently -- "no Python frame", after everything
# else has succeeded -- which a desktop shows as a crash report on quit.
# Every piece of state (documents, session.ini, the trust store) is written
# synchronously before this point, so finalisation has nothing left to do
# except crash.
sys.stdout.flush()
sys.stderr.flush()
os._exit(int(code or 0))
