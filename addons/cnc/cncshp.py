# Ochre C&C addon.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later.
"""SHP (Tiberian Sun / Red Alert 2) sprites.

Format, all little-endian:

  File header, 8 bytes
      u16 zero        always 0 -- the only magic, used for sniffing
      u16 width       canvas, shared by every frame
      u16 height
      u16 frames

  Frame header, 24 bytes x frames
      u16 x, y        where the cropped rect sits in the canvas
      u16 w, h        the frame's own cropped size (may be 0)
      u32 flags       compression; see below
      u8[4] radar     average colour, cosmetic
      u32 reserved
      u32 offset      absolute file offset; 0 means an EMPTY frame

  Compression: 0/1 raw, 2 raw with per-line length, 3 RLE-Zero with per-line
  length. We read all four and write 0 or 3, choosing 3 only when it is
  actually smaller -- which is what XCC does.

RLE-Zero, per scanline, never crossing a row:

      u16 length      INCLUDES its own two bytes
      then: a non-zero byte is a literal palette index;
            a zero byte means the NEXT byte is a count of transparent pixels.

Three details are the difference between a file the games load and one they
do not, and all three are easy to miss:

  * Transparent runs CLAMP AT 255. A 600-wide empty row is 00 FF 00 FF 00 5A,
    not one run.
  * The line length counts itself. Off by two is the classic SHP bug.
  * A literal index 0 cannot be stored at all -- index 0 IS the run marker.
    That is a property of the format, not a choice.

Mapping onto Ochre: each SHP frame becomes a document FRAME (not a layer),
holding one index-locked layer. Indices are authoritative, so a file that is
opened and re-saved without editing comes back byte-identical -- including
the house-colour remap ramp at 16-31, whose entries are near-duplicates in
RGB space and would be scrambled by anything that round-tripped through
colour.
"""

import struct

import numpy as np

from ochre.engine.document import Document
from ochre.engine.geometry import Rect

HEADER = struct.Struct("<HHHH")
FRAME = struct.Struct("<HHHHI4sII")
FLAG_TRANSPARENT = 0x01
FLAG_RLE = 0x02


# ---- RLE-Zero -----------------------------------------------------------

def decode_rle_line(buf, offset, width):
    """One scanline. Returns (pixels, next_offset).

    Deliberately lenient: real SHPs from old tools contain truncated lines
    and over-long runs, and refusing to open them would be useless
    pedantry. Clamp and carry on.
    """
    if offset + 2 > len(buf):
        return np.zeros(width, dtype=np.uint8), len(buf)
    length = buf[offset] | (buf[offset + 1] << 8)
    if length < 2:
        return np.zeros(width, dtype=np.uint8), offset + 2
    end = min(offset + length, len(buf))
    row = np.zeros(width, dtype=np.uint8)
    i, x = offset + 2, 0
    while i < end and x < width:
        value = buf[i]
        if value:
            row[x] = value
            x += 1
            i += 1
        else:
            if i + 1 >= end:
                break
            x = min(width, x + buf[i + 1])
            i += 2
    return row, offset + length


def encode_rle_line(row):
    """One scanline to RLE-Zero, with the length prefix counting itself."""
    out = bytearray()
    x, w = 0, len(row)
    while x < w:
        if row[x] == 0:
            run = 0
            while x < w and row[x] == 0 and run < 255:
                run += 1
                x += 1
            out += bytes((0, run))
        else:
            out.append(int(row[x]))
            x += 1
    return struct.pack("<H", len(out) + 2) + bytes(out)


def encode_rle(plane):
    return b"".join(encode_rle_line(plane[y]) for y in range(plane.shape[0]))


def decode_rle(buf, offset, width, height):
    out = np.zeros((height, width), dtype=np.uint8)
    pos = offset
    for y in range(height):
        out[y], pos = decode_rle_line(buf, pos, width)
    return out


# ---- reading ------------------------------------------------------------

def is_shp(data):
    """Sniff. The leading u16 is always zero, which is all the magic there is."""
    if len(data) < HEADER.size:
        return False
    zero, width, height, frames = HEADER.unpack_from(data, 0)
    if zero != 0 or frames == 0 or width == 0 or height == 0:
        return False
    if width > 4096 or height > 4096 or frames > 4096:
        return False
    return len(data) >= HEADER.size + FRAME.size * frames


def read_frames(data):
    """Parse into (width, height, [frame dicts]) with decoded index planes."""
    zero, width, height, count = HEADER.unpack_from(data, 0)
    if zero != 0:
        raise ValueError("not a TS/RA2 SHP (leading word is %d, not 0)" % zero)

    frames = []
    for i in range(count):
        off = HEADER.size + FRAME.size * i
        x, y, w, h, flags, radar, _reserved, data_off = FRAME.unpack_from(data, off)
        entry = {"x": x, "y": y, "w": w, "h": h, "flags": flags,
                 "radar": radar, "offset": data_off, "plane": None}
        # A zero offset, or a zero-sized rect, is the canonical EMPTY frame.
        # Very common -- blank shadow frames especially -- so it must not be
        # treated as corruption.
        if data_off and w and h and data_off < len(data):
            # Only the low bits are meaningful: some writers pack a size into
            # the upper half of this dword, so reading it whole yields values
            # like 0x02A40002 rather than 2.
            if flags & FLAG_RLE:
                entry["plane"] = decode_rle(data, data_off, w, h)
            elif (flags & 0xFF) == 2:
                entry["plane"] = _decode_prefixed(data, data_off, w, h)
            else:
                need = w * h
                raw = data[data_off:data_off + need]
                if len(raw) == need:
                    entry["plane"] = np.frombuffer(raw, np.uint8).reshape(h, w)
        frames.append(entry)
    return width, height, frames


def _decode_prefixed(data, offset, width, height):
    """Compression 2: raw scanlines, each with a u16 length prefix."""
    out = np.zeros((height, width), dtype=np.uint8)
    pos = offset
    for y in range(height):
        if pos + 2 > len(data):
            break
        length = data[pos] | (data[pos + 1] << 8)
        payload = data[pos + 2:pos + max(2, length)]
        take = min(width, len(payload))
        if take:
            out[y, :take] = np.frombuffer(payload[:take], np.uint8)
        pos += max(2, length)
    return out


# ---- writing ------------------------------------------------------------

def crop_bounds(plane, transparent=0):
    """Tight bounds of non-transparent pixels, or None if the frame is empty."""
    mask = plane != transparent
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return Rect.from_points(int(xs.min()), int(ys.min()),
                            int(xs.max()), int(ys.max()))


def build_shp(width, height, planes, transparent=0, compress=True,
              radar_colours=None, bounds=None):
    """Serialise index planes to SHP bytes.

    Each frame is auto-cropped to its own content, which is what the format
    expects and what keeps files small. `bounds` can override that per frame
    to preserve a file's original rectangles on a round-trip -- re-saving an
    untouched sprite should reproduce it, not re-optimise it.
    """
    count = len(planes)
    headers = bytearray()
    body = bytearray()
    base = HEADER.size + FRAME.size * count

    for i, plane in enumerate(planes):
        box = (bounds[i] if bounds and bounds[i] is not None
               else crop_bounds(plane, transparent))
        if box is None:
            # Empty frame: zero size, zero offset, no data block.
            headers += FRAME.pack(0, 0, 0, 0, 0, b"\0\0\0\0", 0, 0)
            continue

        sub = plane[box.y:box.y1, box.x:box.x1]
        rle = encode_rle(sub)
        raw = sub.tobytes()
        # Only use RLE when it actually wins, as XCC does.
        use_rle = compress and len(rle) < len(raw)
        payload = rle if use_rle else raw
        flags = (FLAG_RLE | FLAG_TRANSPARENT) if use_rle else 0

        # Frame data blocks are 8-byte aligned.
        pad = (-(base + len(body))) % 8
        body += b"\0" * pad
        offset = base + len(body)
        body += payload

        radar = b"\0\0\0\0"
        if radar_colours and radar_colours[i] is not None:
            value = radar_colours[i]
            if isinstance(value, (bytes, bytearray)):
                radar = (bytes(value) + b"\0\0\0\0")[:4]
            else:
                r, g, b = value[:3]
                radar = bytes((int(r), int(g), int(b), 0))
        headers += FRAME.pack(box.x, box.y, box.w, box.h, flags, radar, 0, offset)

    return HEADER.pack(0, width, height, count) + bytes(headers) + bytes(body)


# ---- the provider -------------------------------------------------------

class ShpFormat:
    """FormatProvider for SHP (TS/RA2).

    Frames become document frames, each holding one index-locked layer, and
    the per-frame rectangle and flags are kept in frame metadata so a
    round-trip can reproduce them rather than re-deriving them.
    """

    name = "cnc.shp"
    extensions = ("shp",)
    can_read = True
    can_write = True

    def __init__(self, palettes=None, db=None):
        self._palettes = palettes
        self._db = db

    def sniff(self, head, path):
        return is_shp(head)

    def load(self, path, host):
        with open(path, "rb") as f:
            data = f.read()
        width, height, frames = read_frames(data)

        palette = self._pick_palette(path, host)
        doc = Document(width, height, palette=palette)
        doc.meta["format"] = "cnc.shp"
        doc.meta["cnc.frames"] = str(len(frames))

        # Unit sprites conventionally store art in the first half of the
        # frame list and 1-bit shadows in the second. Recording the guess as
        # metadata lets a UI group them without the core knowing anything.
        if len(frames) >= 2 and len(frames) % 2 == 0:
            doc.meta["cnc.shadow_split"] = str(len(frames) // 2)

        layer = doc.add_layer("Sprite", planes=("index", "rgba"),
                              authoritative=("index",))
        doc.frames = []
        from ochre.engine.frame import Frame
        for i, entry in enumerate(frames):
            frame = Frame(name="Frame %d" % (i + 1))
            frame.meta["cnc.x"] = str(entry["x"])
            frame.meta["cnc.y"] = str(entry["y"])
            frame.meta["cnc.w"] = str(entry["w"])
            frame.meta["cnc.h"] = str(entry["h"])
            frame.meta["cnc.flags"] = str(entry["flags"])
            # Cosmetic, but it is data the file carried. Recomputing it on
            # save would mean re-saving an untouched sprite changed bytes,
            # which is exactly what a round-trip guarantee rules out.
            frame.meta["cnc.radar"] = entry["radar"].hex()
            doc.frames.append(frame)
            cell = doc.cell(layer, frame)
            if entry["plane"] is not None:
                target = cell.plane("index")
                box = Rect(entry["x"], entry["y"], entry["w"], entry["h"])
                clipped = box.clipped_to(width, height)
                if clipped is not None:
                    src = entry["plane"][:clipped.h, :clipped.w]
                    target[clipped.slice()] = src
                cell.content_bbox = clipped or doc.bounds
            else:
                cell.content_bbox = None
            cell.refresh_derived()
        doc.current = 0
        return doc

    def save(self, document, path, host, options):
        layer = next((l for l in document.layers() if l.index_locked), None)
        if layer is None:
            raise ValueError("SHP needs an index-locked layer; convert the "
                             "document to indexed first")

        planes, bounds, radar = [], [], []
        palette = document.palette
        preserve = str((options or {}).get("preserve_bounds", "1")) not in ("0", "off")
        # Recompute radar colours only when asked. The default preserves what
        # the file said, so an untouched sprite re-saves byte-identically.
        recompute = str((options or {}).get("recompute_radar", "0")) in ("1", "on", "true")
        for frame in document.frames:
            document.ensure_warm(frame)
            cell = document.cells.get((layer.id, frame.id))
            if cell is None or cell.planes.get("index") is None:
                planes.append(np.zeros((document.height, document.width), np.uint8))
                bounds.append(None)
                radar.append(None)
                continue
            plane = cell.plane("index")
            planes.append(plane)
            # Preserve the file's original crop rect ONLY while the content
            # still fits inside it. Honouring it unconditionally would give
            # byte-exact round-trips at the cost of silently discarding any
            # edit made outside the original bounds -- which is a far worse
            # bug than a re-cropped frame.
            stored_rect = _meta_rect(frame) if preserve else None
            if stored_rect is not None:
                actual = crop_bounds(plane, 0)
                if actual is not None and not stored_rect.contains_rect(actual):
                    stored_rect = None      # content grew; re-crop
            bounds.append(stored_rect)
            stored = frame.meta.get("cnc.radar")
            if stored is not None and not recompute:
                radar.append(_unhex_radar(stored))
            else:
                radar.append(_average_colour(plane, palette))

        data = build_shp(document.width, document.height, planes,
                         transparent=0, compress=True,
                         radar_colours=radar, bounds=bounds)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _pick_palette(self, path, host):
        """A SHP references no palette, so one has to be chosen for it.

        Which palette applies is external convention -- unittem.pal for a
        temperate unit, isotem.pal for terrain, and so on. Indices are
        preserved exactly whichever is picked; only the colours shown are a
        guess, and rebinding later costs nothing.
        """
        if self._palettes is not None:
            names = self._palettes.names()
            if names:
                import os
                stem = os.path.basename(path).lower()
                for candidate in names:
                    if candidate.lower() in stem:
                        return self._palettes.load(candidate, host)
                return self._palettes.load(names[0], host)
        from . import cncpal
        palette = cncpal.grayscale_fallback("cnc-fallback")
        if self._db is not None:
            cncpal.annotate(palette, self._db, "Palette:Default")
        return palette


def _meta_rect(frame):
    try:
        w = int(frame.meta.get("cnc.w", "0"))
        h = int(frame.meta.get("cnc.h", "0"))
        if w <= 0 or h <= 0:
            return None
        return Rect(int(frame.meta.get("cnc.x", "0")),
                    int(frame.meta.get("cnc.y", "0")), w, h)
    except (TypeError, ValueError):
        return None


def _unhex_radar(text):
    try:
        return (bytes.fromhex(text) + b"\0\0\0\0")[:4]
    except (TypeError, ValueError):
        return None


def _average_colour(plane, palette):
    """The radar colour: mean of non-transparent pixels. Cosmetic."""
    if palette is None:
        return None
    mask = plane != 0
    if not mask.any():
        return None
    rgb = palette.entries[plane[mask], :3].astype(np.int32)
    return tuple(int(v) for v in rgb.mean(axis=0))
