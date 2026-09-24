#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Settings: DEFAULTS is the schema, and precedence is DEFAULTS < ini < env.

Run: python3 tests/test_settings.py
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ochre.engine.ini import IniDB
from ochre.engine.settings import Settings, DEFAULTS, _coerce


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    # ---- defaults alone ----
    s = Settings(db=IniDB(), env={})
    check("default int", s.get("Display", "CheckerSize") == 16)
    check("default str", s.get("IndexLocked", "EffectPolicy") == "refuse")
    check("default tuple", s.get("Display", "CheckerLight") == (200, 200, 200, 255))
    check("unknown key is None, not an error", s.get("Display", "Nope") is None)
    check("unknown section is None", s.get("Nope", "Nope") is None)
    check("known() reports membership", s.known("Display", "CheckerSize"))
    check("known() rejects strangers", not s.known("Display", "Nope"))

    # ---- ini overrides defaults ----
    db = IniDB()
    db.load_string("[Display]\nCheckerSize = 32\nCheckerLight = 1,2,3\n")
    s = Settings(db=db, env={})
    check("ini overrides default int", s.get("Display", "CheckerSize") == 32)
    check("ini rgb pads to rgba", s.get("Display", "CheckerLight") == (1, 2, 3, 255))

    # ---- env beats ini ----
    s = Settings(db=db, env={"OCHRE_CHECKERSIZE": "64"})
    check("env overrides ini", s.get("Display", "CheckerSize") == 64)

    # ...and env beats a default even with no ini at all.
    s = Settings(db=IniDB(), env={"OCHRE_CHECKERSIZE": "128"})
    check("env overrides default", s.get("Display", "CheckerSize") == 128)

    # ---- unknown keys are ignored, which is the compat story ----
    db2 = IniDB()
    db2.load_string("[Display]\nCheckerSize = 24\nFutureSetting = 99\n")
    s = Settings(db=db2, env={})
    check("unknown ini key ignored", s.get("Display", "FutureSetting") is None)
    check("known keys still resolve alongside it",
          s.get("Display", "CheckerSize") == 24)

    # ---- bad values fall back rather than crash ----
    db3 = IniDB()
    db3.load_string("[Display]\nCheckerSize = banana\nCheckerLight = 1,2\n")
    s = Settings(db=db3, env={})
    check("bad int falls back to default", s.get("Display", "CheckerSize") == 16)
    check("wrong-length tuple falls back",
          s.get("Display", "CheckerLight") == (200, 200, 200, 255))
    s = Settings(db=IniDB(), env={"OCHRE_CHECKERSIZE": "banana"})
    check("bad env falls back to default", s.get("Display", "CheckerSize") == 16)

    # ---- coercion is driven by the default's type ----
    check("coerce bool on", _coerce(False, "on") is True)
    check("coerce bool off", _coerce(True, "off") is False)
    # bool is checked before int -- otherwise True would coerce as an integer.
    check("bool default beats int path", isinstance(_coerce(False, "1"), bool))
    check("coerce int hex", _coerce(0, "0x10") == 16)
    check("coerce float", _coerce(0.0, "2.5") == 2.5)
    check("coerce str passthrough", _coerce("x", " hello ") == "hello")
    check("coerce tuple", _coerce((0, 0, 0, 0), "1,2,3,4") == (1, 2, 3, 4))
    check("coerce None raw yields default", _coerce(5, None) == 5)

    # ---- real data dir loads and every default survives a round trip ----
    data_dir = os.path.join(ROOT, "data")
    if os.path.isdir(data_dir):
        s = Settings(data_dir, env={})
        check("data/ dir loads", s.get("Display", "CheckerSize") is not None)
        for section, keys in DEFAULTS.items():
            for key, default in keys.items():
                got = s.get(section, key)
                check("typed %s/%s" % (section, key),
                      isinstance(got, type(default)),
                      "-- got %r (%s), want %s"
                      % (got, type(got).__name__, type(default).__name__))

    # ---- views ----
    s = Settings(db=IniDB(), env={})
    check("section view is a dict", isinstance(s["Display"], dict))
    check("section view has the keys", "CheckerSize" in s["Display"])
    check("as_dict round-trips sections",
          set(s.as_dict()) == set(DEFAULTS))
    check("db is reachable for content tables", s.db is not None)

    print("\nall settings checks passed")


if __name__ == "__main__":
    main()
