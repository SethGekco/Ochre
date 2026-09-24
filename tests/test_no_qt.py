#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""The engine must never import Qt, and must be plain ASCII.

Both halves of this exist because the alternative is a convention nobody
enforces. The ASCII check earns its place too: during design a stray CJK
character landed inside an identifier, and that should fail a test rather
than reach a reviewer.

Run: python3 tests/test_no_qt.py
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "ochre", "engine")

BANNED = ("PySide6", "PyQt5", "PyQt6", "shiboken6", "shiboken2")


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def engine_sources():
    for dirpath, dirnames, filenames in os.walk(ENGINE):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def imported_names(tree):
    """Every module name reachable from an import statement in this file."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def main():
    sources = list(engine_sources())
    check("engine has sources to check", len(sources) > 0,
          "(nothing found under %s)" % ENGINE)

    for path in sources:
        rel = os.path.relpath(path, ROOT)
        raw = open(path, "rb").read()

        # ASCII: decode strictly, and report the exact offending character.
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError as exc:
            bad = raw[exc.start:exc.end]
            line = raw[:exc.start].count(b"\n") + 1
            check("ascii %s" % rel, False,
                  "-- non-ASCII %r at line %d" % (bad, line))
            return

        tree = ast.parse(text, filename=path)
        offenders = sorted(m for m in imported_names(tree)
                           if m.split(".")[0] in BANNED)
        check("clean %s" % rel, not offenders,
              "-- imports %s" % ", ".join(offenders))

    check("engine is Qt-free and ASCII (%d files)" % len(sources), True)


if __name__ == "__main__":
    main()
