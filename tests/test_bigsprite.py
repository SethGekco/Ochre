#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The real stress sprite still opens under the open budget.

~/Desktop/sppnuke.shp is a 203 MB game asset: 136 frames of 2052 x 2816,
786 million pixels in all -- more than one frame's cap (536.9 million), so a
budget that was global rather than per frame would refuse it. It is not
redistributable and never enters the repository; this test skips when it is
absent. OCHRE_STRESS_SHP names another file (or "off" to skip). It runs in
the default pass only: the numpy-fallback pass would add another half minute
and gigabyte to the suite and decode it identically.

Run: python3 tests/test_bigsprite.py
"""

import os
import resource
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ochre.engine import budget
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ADDONS = os.path.join(ROOT, "addons")
DEFAULT = os.path.expanduser("~/Desktop/sppnuke.shp")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    path = os.environ.get("OCHRE_STRESS_SHP", DEFAULT)
    if path == "off" or not os.path.isfile(path):
        print("ok: skipped (stress sprite %s not present)" % path)
        return
    if os.environ.get("OCHRE_ACCEL") == "0":
        print("ok: skipped (stress sprite runs in the default pass only)")
        return
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[ADDONS],
                     state_dir=tempfile.mkdtemp(prefix="ochre-fileio-bigsprite-"))
    for ident in list(ctl.addons.addons):
        ctl.addons.trust(ident)
    ctl.load_addons(all_addons=True)
    if "cnc.shp" not in [n for _l, n, _e, _i in ctl.addon_formats()]:
        print("ok: skipped (C&C addon not present)")
        return
    t0 = time.time()
    ctl.open_path(path)
    dt = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    doc = ctl.doc
    total = doc.width * doc.height * len(doc.frames)
    check("%s opens: %d frames of %d x %d in %.1f s, peak RSS %.0f MB"
          % (os.path.basename(path), len(doc.frames), doc.width, doc.height, dt, peak),
          len(doc.frames) > 1)
    print("    %d pixels in all against a per-frame cap of %d" % (total, budget.frame_pixels()))
    if total > budget.frame_pixels():
        check("...although its frames together exceed one frame's cap (the cap is per frame)", True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
