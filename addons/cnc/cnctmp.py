# Ochre C&C addon.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later.
"""TMP (Tiberian Sun / Red Alert 2) isometric terrain templates.

  File header, 16 bytes, four int32
      cblocks_x, cblocks_y    template size, in tiles
      cx, cy                  tile pixel size (48x24, or 60x30)

  Offset table at byte 16: cblocks_x * cblocks_y int32 file offsets.
  AN OFFSET OF ZERO MEANS AN EMPTY SLOT. Templates are stored as rectangular
  grids but are actually diamond-shaped, so most slots are unused -- assuming
  every slot has a tile is the quickest way to get garbage.

  Tile record, 52-byte header then data
      i32 x, y                    position in template pixel space
      i32 extra_off, z_off, extra_z_off     0 when absent
      i32 extra_x, extra_y, extra_w, extra_h
      u32 flags               bit0 has-extra, bit1 has-z, bit2 randomised
      u8  height              elevation
      u8  land_type           terrain class
      u8  ramp_type           slope shape
      u8[3] radar_low, u8[3] radar_high

The base tile is NOT a rectangle. It is cx*cy/2 bytes holding only the
rhombus: row y of the top half is 4*(y+1) pixels wide starting at
cx/2 - 2*(y+1), and the bottom half mirrors it. For 48x24 that is widths
4,8,...,48 then 44,...,0, totalling 576. Off by one here shears every tile
diagonally.

Two traps worth stating:

  * The extra and z offsets are relative to the TILE RECORD, not the file.
    Treating them as absolute is the number one way to get garbage.
  * Z-data is 0-31 valid with 255 meaning no-data; 32-254 are invalid. It
    drives the software depth buffer that decides whether a unit draws in
    front of or behind terrain, so wrong z-data makes units walk through
    cliffs.

Mapping onto Ochre: each tile becomes a document FRAME, laid out as a grid,
carrying an index-locked colour layer plus a named "height" plane for the
z-data -- which is exactly why the core grew arbitrary non-colour planes.
Per-tile land and ramp types live in frame metadata and are preserved
verbatim, because several distinct numeric values share a human label and
normalising them would silently change tile behaviour.
"""

import struct
import weakref

import numpy as np

from ochre.engine.document import Document
from ochre.engine.frame import Frame
from ochre.engine.geometry import Rect

# {document: {frame_id: tile record}} as read from the file, carrying the two
# things the document model cannot hold: the original 52 header bytes (see
# build_tmp on why the uninitialised ones matter) and the EXTRA block -- the
# plain rectangle holding art that overflows the diamond, which is what
# cliffs, bridges and tall rocks are made of.
#
# The extra lives beside the document rather than in it because it does not
# fit the plane model, and that was measured rather than assumed: across the
# shipped theaters, 767 of 3147 tiles have one, the largest is 60x84 against a 60x30
# tile, and they sit up to 72 pixels ABOVE the tile origin. A cell's planes
# are canvas-aligned by design -- one dirty rect, one history delta, one
# bbox -- so a block nearly three times the canvas height cannot be one.
#
# The consequence, and it is a real limitation rather than a detail: extras
# round-trip and are never corrupted, but they cannot yet be PAINTED, and
# they do not survive a trip through .ochre. Making them editable means
# giving the document a canvas large enough for tile plus overhang and
# recording where the diamond sits inside it. See docs/DESIGN.md.
_SOURCE_TILES = weakref.WeakKeyDictionary()

HEADER = struct.Struct("<iiii")
TILE = struct.Struct("<iiiiiiiiiIBBB3s3s")   # 52 bytes to the padding
TILE_HEADER_SIZE = 52

FLAG_EXTRA = 0x01
FLAG_Z = 0x02
FLAG_RANDOM = 0x04

Z_MAX = 31
Z_NONE = 255


def diamond_rows(cx, cy):
    """[(row, x_start, width)] for the rhombus. Sums to exactly cx*cy/2."""
    out = []
    x = cx // 2
    width = 0
    for y in range(cy // 2):
        width += 4
        x -= 2
        out.append((y, x, width))
    for y in range(cy // 2, cy):
        width -= 4
        x += 2
        out.append((y, x, width))
    return out


def decode_diamond(data, cx, cy, fill=0):
    """Diamond-packed bytes to a full cx-by-cy rectangle."""
    rect = np.full((cy, cx), fill, dtype=np.uint8)
    i = 0
    for y, x, width in diamond_rows(cx, cy):
        if width <= 0:
            continue
        chunk = data[i:i + width]
        if len(chunk) < width:
            break
        rect[y, x:x + width] = np.frombuffer(chunk, np.uint8)
        i += width
    return rect


def encode_diamond(rect, cx, cy):
    """The exact inverse: a rectangle back to diamond-packed bytes."""
    out = bytearray()
    for y, x, width in diamond_rows(cx, cy):
        if width <= 0:
            continue
        out += rect[y, x:x + width].tobytes()
    return bytes(out)


def is_tmp(data):
    """Sniff. TMP has no magic, so this is a plausibility check."""
    if len(data) < HEADER.size + 4:
        return False
    bx, by, cx, cy = HEADER.unpack_from(data, 0)
    if not (0 < bx <= 256 and 0 < by <= 256):
        return False
    if cx not in (48, 60) or cy not in (24, 30):
        return False
    if cx * cy % 2:
        return False
    return len(data) >= HEADER.size + 4 * bx * by


def read_tmp(data):
    """Parse into (header, [tile dicts]). Absent slots yield None."""
    bx, by, cx, cy = HEADER.unpack_from(data, 0)
    count = bx * by
    offsets = struct.unpack_from("<%di" % count, data, HEADER.size)

    tiles = []
    for slot, offset in enumerate(offsets):
        if offset <= 0 or offset + TILE_HEADER_SIZE > len(data):
            tiles.append(None)         # empty slot: the common case
            continue
        (x, y, extra_off, z_off, extra_z_off, ex, ey, ew, eh,
         flags, height, land, ramp, radar_lo, radar_hi) = TILE.unpack_from(data, offset)

        cb = cx * cy // 2
        image = decode_diamond(data[offset + TILE_HEADER_SIZE:
                                    offset + TILE_HEADER_SIZE + cb], cx, cy)

        zdata = None
        if (flags & FLAG_Z) and z_off:
            # RELATIVE to the tile record, not the file.
            start = offset + z_off
            zdata = decode_diamond(data[start:start + cb], cx, cy, fill=Z_NONE)

        extra = extra_z = None
        if (flags & FLAG_EXTRA) and extra_off and ew > 0 and eh > 0:
            start = offset + extra_off
            need = ew * eh
            raw = data[start:start + need]
            if len(raw) == need:
                # The extra block is a PLAIN RECTANGLE, not diamond-packed.
                extra = np.frombuffer(raw, np.uint8).reshape(eh, ew)
            if (flags & FLAG_Z) and extra_z_off:
                start = offset + extra_z_off
                raw = data[start:start + need]
                if len(raw) == need:
                    extra_z = np.frombuffer(raw, np.uint8).reshape(eh, ew)

        tiles.append({"slot": slot, "x": x, "y": y, "flags": flags,
                      "height": height, "land": land, "ramp": ramp,
                      "radar_low": radar_lo, "radar_high": radar_hi,
                      "image": image, "z": zdata,
                      "extra": extra, "extra_z": extra_z,
                      "extra_x": ex, "extra_y": ey,
                      # The original 52 bytes, kept verbatim. See build_tmp:
                      # the vanilla files carry uninitialised memory in the
                      # fields the flags say to ignore, and reproducing a
                      # file we did not change means reproducing that too.
                      "raw": bytes(data[offset:offset + TILE_HEADER_SIZE])})
    return {"bx": bx, "by": by, "cx": cx, "cy": cy}, tiles


def build_tmp(header, tiles, preserve=True):
    """Serialise back to TMP bytes.

    `preserve` keeps each tile's original 52 header bytes and overwrites only
    the fields this writer manages. That sounds like pedantry and is not: the
    vanilla templates were written by a debug-build MSVC tool that left
    uninitialised heap in every field the flags mark absent -- `0xCD` fill in
    the extra offsets and extents, and in the three bytes of padding past the
    49-byte struct. Measured across the shipped RA2 theaters, that garbage is
    the ONLY thing that differs on a read-write cycle; every pixel, z-plane
    and extra block already matched. Normalising it to zero is harmless to
    the game and fatal to the one test that can prove this codec correct, so
    a file we did not edit is reproduced exactly as found.
    """
    bx, by = header["bx"], header["by"]
    cx, cy = header["cx"], header["cy"]
    cb = cx * cy // 2
    count = bx * by

    body = bytearray()
    offsets = [0] * count
    base = HEADER.size + 4 * count

    for slot in range(count):
        tile = tiles[slot] if slot < len(tiles) else None
        if tile is None:
            continue                    # leave the offset at 0: empty slot
        offsets[slot] = base + len(body)

        has_z = tile.get("z") is not None
        has_extra = tile.get("extra") is not None
        flags = 0
        if has_extra:
            flags |= FLAG_EXTRA
        if has_z:
            flags |= FLAG_Z
        flags |= (tile.get("flags", 0) & FLAG_RANDOM)

        extra = tile.get("extra")
        ew = int(extra.shape[1]) if extra is not None else 0
        eh = int(extra.shape[0]) if extra is not None else 0

        # Offsets are relative to this record, and the layout is fixed:
        # header, base diamond, then whichever optional blocks exist.
        cursor = TILE_HEADER_SIZE + cb
        z_off = 0
        extra_off = 0
        extra_z_off = 0
        if has_z:
            z_off = cursor
            cursor += cb
        if has_extra:
            extra_off = cursor
            cursor += ew * eh
            if has_z and tile.get("extra_z") is not None:
                extra_z_off = cursor
                cursor += ew * eh

        raw = tile.get("raw") if preserve else None
        header = bytearray(raw if raw and len(raw) == TILE_HEADER_SIZE
                           else b"\0" * TILE_HEADER_SIZE)
        if raw:
            # Keep whatever the original tool left behind in the fields this
            # tile does not use, and merge our three meaningful flag bits
            # into the rest of its word rather than replacing it.
            flags = (struct.unpack_from("<I", header, 36)[0]
                     & ~(FLAG_EXTRA | FLAG_Z)) | (flags & (FLAG_EXTRA | FLAG_Z))

        struct.pack_into("<ii", header, 0,
                         int(tile.get("x", 0)), int(tile.get("y", 0)))
        struct.pack_into("<I", header, 36, flags)
        struct.pack_into("<BBB3s3s", header, 40,
                         int(tile.get("height", 0)) & 0xFF,
                         int(tile.get("land", 0)) & 0xFF,
                         int(tile.get("ramp", 0)) & 0xFF,
                         _rgb(tile.get("radar_low")), _rgb(tile.get("radar_high")))
        # Only write an offset or extent the flags actually point at. Writing
        # a zero into a field the game will never read would still change the
        # bytes, and a stale value there is the original file's, not ours.
        if has_z or not raw:
            struct.pack_into("<i", header, 12, z_off)
        if has_extra or not raw:
            struct.pack_into("<i", header, 8, extra_off)
            struct.pack_into("<iiii", header, 20,
                             int(tile.get("extra_x", 0)),
                             int(tile.get("extra_y", 0)), ew, eh)
            if (has_extra and has_z) or not raw:
                struct.pack_into("<i", header, 16, extra_z_off)
        body += bytes(header)
        body += encode_diamond(tile["image"], cx, cy)
        if has_z:
            body += encode_diamond(tile["z"], cx, cy)
        if has_extra:
            body += extra.tobytes()
            if has_z and tile.get("extra_z") is not None:
                body += tile["extra_z"].tobytes()

    out = bytearray(HEADER.pack(bx, by, cx, cy))
    out += struct.pack("<%di" % count, *offsets)
    out += body
    return bytes(out)


def _rgb(value):
    if not value:
        return b"\0\0\0"
    if isinstance(value, bytes):
        return (value + b"\0\0\0")[:3]
    return bytes((int(value[0]), int(value[1]), int(value[2])))


class TmpFormat:
    """FormatProvider for TMP terrain templates.

    A tile grid is not a layer stack and not a second axis. It is the
    document's one frame axis with a GRID layout hint, which is exactly the
    arrangement the core was designed to express.
    """

    name = "cnc.tmp"
    extensions = ("tem", "sno", "urb", "ubn", "des", "lun", "tmp")
    can_read = True
    can_write = True

    def __init__(self, palettes=None, db=None):
        self._palettes = palettes
        self._db = db

    def sniff(self, head, path):
        return is_tmp(head)

    def load(self, path, host):
        with open(path, "rb") as f:
            data = f.read()
        header, tiles = read_tmp(data)
        cx, cy = header["cx"], header["cy"]

        palette = self._pick_palette(host)
        doc = Document(cx, cy, palette=palette)
        doc.meta["format"] = "cnc.tmp"
        doc.meta["cnc.blocks_x"] = str(header["bx"])
        doc.meta["cnc.blocks_y"] = str(header["by"])
        doc.axis_layout = "grid"
        doc.axis_columns = header["bx"]

        # Colour and z travel together on one layer: same dirty rect, same
        # undo entry, same file entry. A parallel layer would desynchronise
        # the first time someone moved or deleted one of the pair.
        layer = doc.add_layer("Tile", planes=("index", "rgba", "height"),
                              authoritative=("index", "height"))

        doc.frames = []
        for slot, tile in enumerate(tiles):
            frame = Frame(name="Tile %d" % slot)
            frame.meta["cnc.slot"] = str(slot)
            if tile is None:
                # An absent slot stays absent -- no cell at all. This is what
                # sparse cells are for: a diamond template is mostly holes.
                frame.meta["cnc.empty"] = "1"
                doc.frames.append(frame)
                continue

            for key in ("x", "y", "height", "land", "ramp", "flags",
                        "extra_x", "extra_y"):
                frame.meta["cnc." + key] = str(tile[key])
            frame.meta["cnc.radar_low"] = _hex(tile["radar_low"])
            frame.meta["cnc.radar_high"] = _hex(tile["radar_high"])
            doc.frames.append(frame)

            _SOURCE_TILES.setdefault(doc, {})[frame.id] = tile

            cell = doc.cell(layer, frame)
            cell.plane("index")[...] = tile["image"]
            if tile["z"] is not None:
                cell.plane("height")[...] = tile["z"]
            if tile["extra"] is not None:
                frame.meta["cnc.extra_w"] = str(tile["extra"].shape[1])
                frame.meta["cnc.extra_h"] = str(tile["extra"].shape[0])

            cell.content_bbox = doc.bounds
            cell.refresh_derived()

        doc.current = 0
        return doc

    def save(self, document, path, host, options):
        layer = next((l for l in document.layers() if l.index_locked), None)
        if layer is None:
            raise ValueError("TMP needs an index-locked layer")

        bx = int(document.meta.get("cnc.blocks_x", document.axis_columns or 1))
        by = int(document.meta.get("cnc.blocks_y",
                                   max(1, -(-len(document.frames) // max(1, bx)))))
        header = {"bx": bx, "by": by, "cx": document.width, "cy": document.height}

        was_tiles = _SOURCE_TILES.get(document, {})
        tiles = []
        for frame in document.frames:
            document.ensure_warm(frame)
            cell = document.cells.get((layer.id, frame.id))
            if cell is None or frame.meta.get("cnc.empty") == "1":
                tiles.append(None)
                continue
            was = was_tiles.get(frame.id, {})
            entry = {
                "x": _int(frame.meta.get("cnc.x")),
                "y": _int(frame.meta.get("cnc.y")),
                "height": _int(frame.meta.get("cnc.height")),
                # Land and ramp are written back VERBATIM. Several distinct
                # values share one human label (0, 1 and 13 all read as
                # "Clear"), so normalising them would silently change how the
                # tile behaves in game.
                "land": _int(frame.meta.get("cnc.land")),
                "ramp": _int(frame.meta.get("cnc.ramp")),
                "flags": _int(frame.meta.get("cnc.flags")),
                "extra_x": _int(frame.meta.get("cnc.extra_x")),
                "extra_y": _int(frame.meta.get("cnc.extra_y")),
                "radar_low": _unhex(frame.meta.get("cnc.radar_low")),
                "radar_high": _unhex(frame.meta.get("cnc.radar_high")),
                "image": cell.plane("index"),
                "z": cell.plane("height") if cell.has("height") else None,
                "extra": was.get("extra"),
                "extra_z": was.get("extra_z"),
                "raw": was.get("raw"),
            }
            tiles.append(entry)

        with open(path, "wb") as f:
            f.write(build_tmp(header, tiles))
        return path

    def _pick_palette(self, host):
        if self._palettes is not None:
            names = [n for n in self._palettes.names() if n.lower().startswith("iso")]
            if names:
                return self._palettes.load(names[0], host)
        from . import cncpal
        palette = cncpal.grayscale_fallback("cnc-iso-fallback")
        if self._db is not None:
            cncpal.annotate(palette, self._db, "Palette:Default")
        return palette


def _int(text, default=0):
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _hex(value):
    if isinstance(value, bytes):
        return value.hex()
    return ""


def _unhex(text):
    try:
        return bytes.fromhex(text or "")
    except ValueError:
        return b"\0\0\0"
