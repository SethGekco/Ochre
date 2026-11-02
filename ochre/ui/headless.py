# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""`ochre --headless`: open, run commands, save -- no window, no Qt.

    ochre --headless in.png --run "image.resize width=100 height=80" --export out.png
    ochre --headless --new 64x64 --run "filter.noise" --save noise.ochre
    ochre --headless --list

Options run in the order given, so `--run A --export a.png --run B
--export b.png` writes both stages. The first failure stops everything,
prints one line saying why, and exits 1, so a shell script can trust both
the exit status and that nothing was written after the error. A usage
mistake exits 2.

The commands are the controller's RECORDABLE list (see scripting.py): the
same whitelist that makes an actions file safe to replay. Nothing here
imports Qt, so this runs on a server with no display and no PySide6.
"""

import os
import sys

USAGE = """\
usage: ochre --headless [INPUT] [options]

Run Ochre without a window. Options run in the order given.

  INPUT            an image or document to open
  --new WxH        start from a blank WxH canvas (white) instead
  --run COMMAND    run a command, e.g. "image.resize width=100 height=80"
                   or "filter.gaussian_blur radius=2" (repeatable)
  --save PATH      save the document; the extension picks the format
                   (.ochre, .psd, .ora, addon formats, or a flat image)
  --export PATH    write the merged image as a flat file (.png, .jpg, ...)
  --list           list the commands and effects, then exit
  -q, --quiet      print nothing on success
  -h, --help       this text

Exit status: 0 done, 1 a step failed (nothing after it ran), 2 usage.
"""


class Usage(Exception):
    pass


def _size(text):
    w, sep, h = text.lower().partition("x")
    try:
        w, h = int(w), int(h)
    except ValueError:
        raise Usage("--new takes WxH, e.g. 640x480, not %r" % text) from None
    if not sep or w < 1 or h < 1:
        raise Usage("--new takes WxH, e.g. 640x480, not %r" % text)
    return w, h


def parse_args(argv):
    """[(step, value)] in order, plus flags. Raises Usage."""
    steps, flags = [], {"quiet": False, "list": False, "help": False}
    takes = {"--new": "new", "--run": "run", "--save": "save", "--export": "export"}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--headless":
            pass
        elif arg in ("-h", "--help"):
            flags["help"] = True
        elif arg in ("-q", "--quiet"):
            flags["quiet"] = True
        elif arg == "--list":
            flags["list"] = True
        elif arg in takes:
            if i + 1 >= len(argv):
                raise Usage("%s needs a value" % arg)
            i += 1
            value = argv[i]
            steps.append((takes[arg], _size(value) if arg == "--new" else value))
        elif arg.startswith("-") and arg != "-":
            raise Usage("unknown option %s" % arg)
        else:
            steps.append(("open", arg))
        i += 1
    return steps, flags


def _say(flags, text):
    if not flags["quiet"]:
        print(text)


def list_commands(ctl, out=sys.stdout):
    from . import scripting
    out.write("Commands (name, then arguments; name=value also works):\n")
    for c in scripting.describe(ctl):
        out.write("  %-28s %s\n" % (c["name"], c["params"]))
    out.write("\nEffects (filter.NAME param=value ...):\n")
    for e in scripting.effect_names(ctl):
        out.write("  %-28s %s\n" % ("filter." + e["name"], " ".join(e["params"])))


def run_steps(ctl, steps, flags):
    """Run every step; raises scripting.CommandError on the first failure."""
    from ..engine.fileio import export_flat
    from . import scripting
    for kind, value in steps:
        if kind == "open":
            if not os.path.isfile(value):
                raise scripting.CommandError("no such file: %s" % value)
            try:
                ctl.open_path(value)
            except Exception as exc:                       # noqa: BLE001
                raise scripting.CommandError(str(exc)) from exc
            _say(flags, "opened %s (%dx%d)" % (value, ctl.doc.width, ctl.doc.height))
        elif kind == "new":
            ctl.new_document(*value)
            _say(flags, "new %dx%d" % value)
        elif kind == "run":
            if ctl.doc is None:
                raise scripting.CommandError("--run %r: no document; give INPUT or --new first"
                                             % value)
            name, args, kwargs = scripting.parse(value)
            result = scripting.run(ctl, name, args, kwargs)
            _say(flags, "ran %s%s" % (value, "" if result["changed"] else " (no change)"))
        elif kind in ("save", "export"):
            if ctl.doc is None:
                raise scripting.CommandError("--%s: no document to write" % kind)
            if os.path.isdir(value):
                raise scripting.CommandError("--%s %s: is a folder" % (kind, value))
            try:
                if kind == "save":
                    ctl.save_path(value)
                    for warning in getattr(ctl, "last_save_warnings", None) or ():
                        print("ochre: warning: %s" % warning, file=sys.stderr)
                else:
                    export_flat(ctl.doc, value)
            except Exception as exc:                       # noqa: BLE001
                raise scripting.CommandError("cannot %s %s: %s" % (kind, value, exc)) from exc
            _say(flags, "%s %s" % ("saved" if kind == "save" else "exported", value))


def main(argv=None):
    argv = sys.argv if argv is None else argv
    try:
        steps, flags = parse_args(argv[1:])
    except Usage as exc:
        print("ochre: %s\n" % exc, file=sys.stderr)
        sys.stderr.write(USAGE)
        return 2
    if flags["help"]:
        sys.stdout.write(USAGE)
        return 0
    if not steps and not flags["list"]:
        sys.stderr.write(USAGE)
        return 2

    from .scripting import CommandError, headless_controller
    ctl = headless_controller()
    if flags["list"]:
        list_commands(ctl)
        return 0
    try:
        run_steps(ctl, steps, flags)
    except CommandError as exc:
        print("ochre: error: %s" % exc, file=sys.stderr)
        return 1
    return 0
