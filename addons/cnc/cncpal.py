# Ochre C&C addon.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later.
"""Westwood .pal palettes, and the annotations that give them meaning.

A .pal is 768 bytes: 256 entries of three bytes, holding SIX-BIT VGA values
(0-63) rather than the full 0-255 range.

The expansion matters and is easy to get subtly wrong. The correct form is

    out = (v << 2) | (v >> 4)

so 63 maps to 255. The obvious shortcut `v << 2` gives 252, quietly clipping
the top of every channel -- that is the bug in WorldAlteringEditor's RGBColor,
and it makes a palette that can never reach pure white. Writing back is
`v >> 2`, which round-trips the expansion exactly.

Everything this module knows about what particular indices MEAN lives in
data/palettes.ini, not in code, and is handed to Ochre as generic
annotations. Ochre renders them without understanding any of it.
"""

import os

import numpy as np

from ochre.engine.palette import LENGTH, Palette

PAL_BYTES = LENGTH * 3


def expand6(values):
    """6-bit VGA -> 8-bit. (v << 2) | (v >> 4), so 63 becomes 255."""
    v = np.asarray(values, dtype=np.uint8) & 0x3F
    return ((v << 2) | (v >> 4)).astype(np.uint8)


def compress8(values):
    """8-bit -> 6-bit VGA. Exact inverse of expand6 for expanded values."""
    return (np.asarray(values, dtype=np.uint8) >> 2).astype(np.uint8)


def read_pal(data, name="palette"):
    """Parse 768 bytes into a Palette, expanding from 6-bit."""
    if len(data) < PAL_BYTES:
        raise ValueError("a .pal is %d bytes; got %d" % (PAL_BYTES, len(data)))
    raw = np.frombuffer(data[:PAL_BYTES], dtype=np.uint8).reshape(LENGTH, 3)
    entries = np.empty((LENGTH, 4), dtype=np.uint8)
    entries[:, :3] = expand6(raw)
    entries[:, 3] = 255
    return Palette(entries, name=name)


def write_pal(palette):
    """Serialise back to 768 bytes of 6-bit VGA."""
    return compress8(palette.entries[:, :3]).tobytes()


def load_pal_file(path):
    with open(path, "rb") as f:
        return read_pal(f.read(), os.path.splitext(os.path.basename(path))[0])


def grayscale_fallback(name="cnc-fallback"):
    """A stand-in when no real palette is available.

    A SHP carries NO palette reference -- which one applies is pure external
    convention -- so an editor must be able to open one and show something
    rather than refusing. Indices are preserved exactly regardless; only the
    colours shown are a guess, and rebinding a real palette fixes the view
    without touching a pixel.
    """
    entries = np.zeros((LENGTH, 4), dtype=np.uint8)
    ramp = np.arange(LENGTH, dtype=np.int32)
    entries[:, 0] = ramp.astype(np.uint8)
    entries[:, 1] = ramp.astype(np.uint8)
    entries[:, 2] = ramp.astype(np.uint8)
    entries[:, 3] = 255
    # Make the conventional special indices visible at a glance.
    entries[0] = (255, 0, 255, 255)        # transparent, shown as magenta
    entries[1] = (40, 40, 40, 255)         # shadow
    for i in range(16, 32):                # the remap ramp, as red
        shade = 60 + (i - 16) * 12
        entries[i] = (min(255, shade), 30, 30, 255)
    return Palette(entries, name=name)


def annotate(palette, db, section="Palette:Default"):
    """Apply protection, transparency and annotations from addon INI.

    This is the seam. Ochre knows a palette may carry named index ranges with
    a role hint. It does not know that 16-31 is a house-colour remap ramp in
    Red Alert 2 -- that fact lives in this addon's data file and arrives here
    as an opaque label.
    """
    for lo, hi in db.getranges(section, "Protected"):
        palette.protect((lo, hi))
    if db.has(section, "Transparent"):
        palette.set_transparent(db.getint(section, "Transparent", 0))
    for key in sorted(db.keys(section)):
        if not key.startswith("Annotation."):
            continue
        parts = (db.get(section, key) or "").split(",")
        if len(parts) < 2:
            continue
        span = parts[0].split("-")
        try:
            lo = int(span[0])
            hi = int(span[1]) if len(span) > 1 else lo
        except ValueError:
            continue
        palette.annotate(lo, hi, parts[1].strip(),
                         parts[2].strip() if len(parts) > 2 else "")
    return palette


class CncPalettes:
    """PaletteProvider: the theatre palettes, discovered from disk or synthesised."""

    name = "cnc.palettes"

    def __init__(self, host):
        self._host = host
        self._db = host.read_ini("data", "palettes.ini")
        self._search = []
        raw = self._db.get("Search", "Directories", "") or ""
        for part in raw.split(","):
            part = os.path.expanduser(part.strip())
            if part and os.path.isdir(part):
                self._search.append(part)

    def names(self):
        found = []
        for directory in self._search:
            for entry in sorted(os.listdir(directory)):
                if entry.lower().endswith(".pal"):
                    found.append(os.path.splitext(entry)[0])
        return found + [n for n in self._db.sections("Palette") if n != "Default"]

    def load(self, name, host):
        for directory in self._search:
            candidate = os.path.join(directory, name + ".pal")
            if os.path.isfile(candidate):
                palette = load_pal_file(candidate)
                return annotate(palette, self._db, self._section_for(name))
        palette = grayscale_fallback(name)
        return annotate(palette, self._db, self._section_for(name))

    def _section_for(self, name):
        specific = "Palette:" + name
        return specific if self._db.has(specific) else "Palette:Default"
