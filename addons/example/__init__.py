# Example Ochre addon.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later.
"""A worked example of the addon API.

Read this alongside ochre/addons/api.py. It demonstrates the three things a
format addon actually does -- claim a file extension, supply palettes, and
add a command -- and it does all of it without Ochre containing a single line
of knowledge about this format.

Note what is NOT here: no import-time side effects, no Qt, and no reaching
into Ochre's internals. Everything arrives through the `host` handed to
register().
"""

import struct

import numpy as np

# Only the documented surface. Reaching past this is how an addon breaks on
# somebody else's refactor.
from ochre.engine.document import Document
from ochre.engine.geometry import Rect
from ochre.engine.palette import LENGTH, Palette

MAGIC = b"OCHREX1\0"


class ExampleFormat:
    """A deliberately tiny indexed format: header, palette, index plane.

    Real formats are messier, but the SHAPE is the same -- and notice it
    returns a Document with an index-locked layer, so indices are preserved
    exactly rather than round-tripped through RGBA.
    """

    name = "example.x1"
    extensions = ("x1",)
    can_read = True
    can_write = True

    def sniff(self, head, path):
        # Extensions lie; content does not.
        return head.startswith(MAGIC)

    def load(self, path, host):
        with open(path, "rb") as f:
            data = f.read()
        if not data.startswith(MAGIC):
            raise ValueError("%s is not an example.x1 file" % path)
        w, h = struct.unpack_from("<HH", data, len(MAGIC))
        off = len(MAGIC) + 4
        entries = np.frombuffer(data, np.uint8, LENGTH * 3, off).reshape(LENGTH, 3)
        off += LENGTH * 3
        indices = np.frombuffer(data, np.uint8, w * h, off).reshape(h, w)

        palette = Palette(entries, name="example.x1")
        _annotate(palette, host)

        doc = Document(w, h, palette=palette)
        layer = doc.add_layer("Image", planes=("index", "rgba"),
                              authoritative=("index",))
        cell = doc.cell(layer)
        cell.plane("index")[...] = indices
        cell.content_bbox = doc.bounds
        cell.refresh_derived()
        doc.meta["format"] = "example.x1"
        return doc

    def save(self, document, path, host, options):
        layer = next((l for l in document.layers() if l.index_locked), None)
        if layer is None or document.palette is None:
            raise ValueError("example.x1 needs an index-locked layer "
                             "and a bound palette")
        cell = document.cell(layer)
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<HH", document.width, document.height))
            f.write(document.palette.entries[:, :3].tobytes())
            f.write(cell.plane("index").tobytes())
        return path


class ExamplePalettes:
    """Palettes read from the addon's OWN data directory."""

    name = "example.palettes"

    def __init__(self, host):
        self._db = host.read_ini("data", "palettes.ini")

    def names(self):
        return self._db.sections("Palette")

    def load(self, name, host):
        section = "Palette:" + name
        # A visible ramp so the annotations have something to describe.
        entries = np.zeros((LENGTH, 4), np.uint8)
        # Compute in a wider dtype and cast down. numpy 2 raises rather than
        # silently wrapping when a uint8 multiply overflows, so the obvious
        # `(ramp * 3) % 256` on a uint8 array is an error, not a ramp.
        ramp = np.arange(LENGTH, dtype=np.int32)
        entries[:, 0] = ramp.astype(np.uint8)
        entries[:, 1] = ((ramp * 3) % 256).astype(np.uint8)
        entries[:, 2] = ((ramp * 7) % 256).astype(np.uint8)
        entries[:, 3] = 255
        palette = Palette(entries, name=name)
        _apply_ini(palette, self._db, section)
        return palette


class ExampleCommand:
    """A command: something the core has no concept of."""

    name = "example.describe"
    label = "Describe Document"
    category = "Example"

    def run(self, host, **kwargs):
        doc = getattr(host.controller, "doc", None)
        if doc is None:
            return "no document"
        return ("%dx%d, %d layer(s), %d frame(s), palette=%s"
                % (doc.width, doc.height, len(doc.layers()), len(doc.frames),
                   doc.palette.name if doc.palette else "none"))


def _apply_ini(palette, db, section):
    """Read protection, transparency and annotations out of addon data."""
    for lo, hi in db.getranges(section, "Protected"):
        palette.protect((lo, hi))
    if db.has(section, "Transparent"):
        palette.set_transparent(db.getint(section, "Transparent", 0))
    for key in db.keys(section):
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


def _annotate(palette, host):
    db = host.read_ini("data", "palettes.ini")
    names = db.sections("Palette")
    if names:
        _apply_ini(palette, db, "Palette:" + names[0])


def register(host):
    """The single entry point. Nothing happens until this is called."""
    host.register_format(ExampleFormat())
    host.register_palette(ExamplePalettes(host))
    host.register_command(ExampleCommand())
    host.log("example addon registered 1 format, 1 palette set, 1 command")
