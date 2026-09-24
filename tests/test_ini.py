#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""IniDB: [Type:Name] parsing, typed accessors, override order.

Run: python3 tests/test_ini.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ochre.engine.ini import IniDB


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


SAMPLE = """
; a comment
# another comment

[Display]
CheckerSize = 16
Enabled = yes
Ratio = 0.75
Light = 200,200,200
Opaque = 10,20,30,40
Hex = 0x1F
Names = alpha, beta , gamma
Ranges = 16-31, 4, 240-254
Junk = not-a-number

[BlendMode:Multiply]
Label = Multiply
Order = 2

[BlendMode:Screen]
Label = Screen
Order = 3

[Tool:Pencil]
Label = Pencil
"""


def main():
    db = IniDB()
    db.load_string(SAMPLE)

    # ---- structure ----
    check("sections by prefix",
          db.sections("BlendMode") == ["Multiply", "Screen"],
          "-- got %r" % (db.sections("BlendMode"),))
    check("prefix does not leak across types", db.sections("Tool") == ["Pencil"])
    check("unknown prefix yields empty", db.sections("Nope") == [])
    check("all sections includes singletons", "Display" in db.sections())
    check("has section", db.has("Display"))
    check("has key", db.has("Display", "CheckerSize"))
    check("missing key", not db.has("Display", "Absent"))
    check("keys lists options", "Enabled" in db.keys("Display"))
    check("keys of missing section is empty", db.keys("Nothing") == [])

    # ---- typed accessors ----
    check("getint", db.getint("Display", "CheckerSize") == 16)
    check("getint accepts hex", db.getint("Display", "Hex") == 31)
    check("getbool yes", db.getbool("Display", "Enabled") is True)
    check("getfloat", abs(db.getfloat("Display", "Ratio") - 0.75) < 1e-9)
    check("getlist splits and strips",
          db.getlist("Display", "Names") == ["alpha", "beta", "gamma"])
    check("getcolor rgb pads alpha",
          db.getcolor("Display", "Light") == (200, 200, 200, 255))
    check("getcolor rgba", db.getcolor("Display", "Opaque") == (10, 20, 30, 40))
    check("getranges", db.getranges("Display", "Ranges") == [(16, 31), (4, 4), (240, 254)])

    # A malformed value must yield the default, never raise -- one bad line in
    # a data file must not stop the editor from starting.
    check("bad int yields default", db.getint("Display", "Junk", 7) == 7)
    check("bad float yields default", db.getfloat("Display", "Junk", 1.5) == 1.5)
    check("missing int yields default", db.getint("Display", "Gone", 42) == 42)
    check("missing section yields default", db.getint("Gone", "Gone", 9) == 9)
    check("missing list yields default",
          db.getlist("Display", "Gone", ["x"]) == ["x"])
    check("missing color yields default",
          db.getcolor("Display", "Gone", (1, 2, 3, 4)) == (1, 2, 3, 4))
    check("bad color yields default",
          db.getcolor("Display", "Names", (9, 9, 9, 9)) == (9, 9, 9, 9))
    check("getbool default when absent",
          db.getbool("Display", "Gone", True) is True)

    # ---- override order across files ----
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "10_base.ini"), "w") as f:
            f.write("[Display]\nCheckerSize = 8\nOnlyInBase = 1\n")
        with open(os.path.join(tmp, "20_over.ini"), "w") as f:
            f.write("[Display]\nCheckerSize = 32\n[New:Thing]\nLabel = T\n")
        addons = os.path.join(tmp, "addons", "demo")
        os.makedirs(addons)
        with open(os.path.join(addons, "extra.ini"), "w") as f:
            f.write("[Display]\nCheckerSize = 64\n[New:FromAddon]\nLabel = A\n")

        d2 = IniDB(tmp)
        check("later file overrides earlier", d2.getint("Display", "CheckerSize") == 64,
              "-- got %d" % d2.getint("Display", "CheckerSize"))
        check("earlier keys survive", d2.getint("Display", "OnlyInBase") == 1)
        check("later file adds sections", "Thing" in d2.sections("New"))
        check("addon dir loaded after core", "FromAddon" in d2.sections("New"))
        check("load order recorded", len(d2.loaded) == 3,
              "-- loaded %r" % (d2.loaded,))

    # ---- robustness ----
    d3 = IniDB()
    check("missing file returns False", d3.load_file("/nonexistent/nope.ini") is False)
    check("failed load is not recorded", d3.loaded == [])
    d4 = IniDB("/nonexistent/directory")
    check("missing dir is survivable", d4.sections() == [])

    # Case sensitivity: keys must not be lowercased, or [Type:Name] content
    # tables would collide with each other.
    d5 = IniDB()
    d5.load_string("[S]\nCamelCase = 1\n")
    check("keys are case-sensitive", d5.get("S", "CamelCase") == "1")
    check("wrong case misses", d5.get("S", "camelcase") is None)

    print("\nall ini checks passed")


if __name__ == "__main__":
    main()
