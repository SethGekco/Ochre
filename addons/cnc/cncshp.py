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
import weakref

import numpy as np

from ochre.engine.document import Document
from ochre.engine.geometry import Rect

# Documents loaded from a SHP, mapped to the frame records they came from,
# so save() can hand an untouched frame back its original bytes.
#
# Deliberately NOT frame metadata. Metadata is persisted strings, and putting
# payloads there would write the whole original sprite into every .ochre file
# as hex. This is a cache: it makes open-edit-save exact within a session and
# is simply absent afterwards, at which point re-encoding is correct anyway.
# Weak keys so closing a document frees it.
_SOURCES = weakref.WeakKeyDictionary()

# How often to apply the frame budget while loading. Small enough that peak
# memory stays near the budget, large enough that the zlib cost of freezing is
# not paid on every single frame.
RESIDENCY_STRIDE = 16

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


def rle_span(buf, offset, height):
    """How many bytes an RLE block occupies. The format never records it.

    Needed to keep an untouched frame's ORIGINAL bytes -- see build_shp, and
    the leniency note above: the shipped art is full of rows whose trailing
    zero run is declared one longer than the row, so a re-encode is correct
    but not identical.
    """
    pos = offset
    for _ in range(height):
        if pos + 2 > len(buf):
            break
        pos += max(2, buf[pos] | (buf[pos + 1] << 8))
    return min(pos, len(buf)) - offset


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
        x, y, w, h, flags, radar, reserved, data_off = FRAME.unpack_from(data, off)
        entry = {"x": x, "y": y, "w": w, "h": h, "flags": flags,
                 "radar": radar, "offset": data_off, "plane": None,
                 "payload": b"", "pad": b"",
                 # Nominally reserved, and in the shipped art frequently a
                 # leaked 32-bit stack address (0x0012fxxx). Meaningless to
                 # the game, and still part of the file.
                 "reserved": reserved}
        # A zero offset, or a zero-sized rect, is the canonical EMPTY frame.
        # Very common -- blank shadow frames especially -- so it must not be
        # treated as corruption.
        if data_off and w and h and data_off < len(data):
            # Only the low bits are meaningful: some writers pack a size into
            # the upper half of this dword, so reading it whole yields values
            # like 0x02A40002 rather than 2.
            if flags & FLAG_RLE:
                entry["plane"] = decode_rle(data, data_off, w, h)
                span = rle_span(data, data_off, h)
            elif (flags & 0xFF) == 2:
                entry["plane"] = _decode_prefixed(data, data_off, w, h)
                span = _prefixed_span(data, data_off, h)
            else:
                need = w * h
                raw = data[data_off:data_off + need]
                span = len(raw)
                if len(raw) == need:
                    entry["plane"] = np.frombuffer(raw, np.uint8).reshape(h, w)
            # The bytes exactly as found. An untouched frame is copied rather
            # than re-encoded, which is the only way to reproduce quirks no
            # rule explains -- see build_shp.
            entry["payload"] = bytes(data[data_off:data_off + span])
        frames.append(entry)

    # Second pass for the gaps BETWEEN data blocks. They are not always zero
    # padding: real files leave fragments of whatever the writer's buffer
    # previously held there (`1e 0f 0f 0f ...` and the like). Writing zeros
    # instead is invisible to the game and still changes the file, so the gap
    # is carried along with the frame that follows it. File order, not header
    # order -- the two need not agree.
    ordered = sorted((f for f in frames if f["offset"]), key=lambda f: f["offset"])
    cursor = HEADER.size + FRAME.size * count
    for entry in ordered:
        if entry["offset"] > cursor:
            entry["pad"] = bytes(data[cursor:entry["offset"]])
        cursor = entry["offset"] + len(entry["payload"])
    return width, height, frames


def _prefixed_span(data, offset, height):
    pos = offset
    for _ in range(height):
        if pos + 2 > len(data):
            break
        pos += max(2, data[pos] | (data[pos + 1] << 8))
    return min(pos, len(data)) - offset


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
              radar_colours=None, bounds=None, source=None, align=8,
              tail=b""):
    """Serialise index planes to SHP bytes.

    Each frame is auto-cropped to its own content, which is what the format
    expects and what keeps files small. `bounds` can override that per frame
    to preserve a file's original rectangles on a round-trip -- re-saving an
    untouched sprite should reproduce it, not re-optimise it.

    `source`, the frame list from read_frames(), extends that principle to
    the three choices a writer is otherwise free to make, all of which were
    measured against the shipped RA2 art and all of which vary in the wild:

      * WHETHER TO COMPRESS. Picking RLE whenever it is smaller is the sane
        default and is what XCC does, but the original tool left 1327 frames
        uncompressed where RLE would have won. Re-compressing them is an
        improvement nobody asked for on a file the user only opened.
      * WHICH RAW FLAG. Compression 0 and 1 both mean raw scanlines, and real
        files use both. There is no way to choose correctly; there is only
        remembering.
      * ALIGNMENT. 52 of 73 sampled files are NOT 8-byte aligned -- most pack
        frames flat. Padding is therefore reproduced from the original
        offsets rather than imposed, falling back to `align` for any frame
        whose payload an edit has resized.

    `tail` is whatever followed the last frame's data in the original file.
    Some writers round the file length up to 8 and some do not, and the two
    groups cannot be told apart from the frame offsets -- every rule worth
    trying has files on both sides of it. So it is remembered, not inferred.
    """
    count = len(planes)
    headers = bytearray()
    body = bytearray()
    base = HEADER.size + FRAME.size * count

    for i, plane in enumerate(planes):
        was = source[i] if source and i < len(source) else None
        box = (bounds[i] if bounds and bounds[i] is not None
               else crop_bounds(plane, transparent))
        if box is None:
            # Empty frame: zero size, no data block. "Zero offset" is the
            # canonical spelling and not the only one -- plenty of real files
            # point their empty frames at the end of the data instead, and
            # they still carry a flag word and a radar colour. Nothing reads
            # any of it, since w and h are zero; it is still what the file
            # said, so an untouched frame keeps it.
            headers += FRAME.pack(was["x"] if was else 0,
                                  was["y"] if was else 0, 0, 0,
                                  was["flags"] if was else 0,
                                  _radar_bytes(radar_colours, i,
                                               was["radar"] if was else None),
                                  was["reserved"] if was else 0,
                                  was["offset"] if was else 0)
            continue

        sub = plane[box.y:box.y1, box.x:box.x1]
        raw = sub.tobytes()
        if was is not None:
            # Mirror what this frame was: the low bits carry the compression,
            # and some writers pack a size into the upper half, so the flag
            # word goes back verbatim rather than being rebuilt.
            flags = was["flags"]
            if _unchanged(sub, box, was):
                payload = was["payload"]
            else:
                payload = (encode_rle(sub) if was["flags"] & FLAG_RLE else raw)
        else:
            rle = encode_rle(sub)
            # Only use RLE when it actually wins, as XCC does.
            use_rle = compress and len(rle) < len(raw)
            payload = rle if use_rle else raw
            flags = (FLAG_RLE | FLAG_TRANSPARENT) if use_rle else 0

        here = base + len(body)
        pad = b"\0" * ((-here) % max(1, align))
        if was is not None and was["offset"] >= here:
            # Reproduce the original gap exactly -- its bytes, not zeros. When
            # the payload is unchanged this lands on the original offset; when
            # an edit has moved things along, the align rule stands instead.
            want = was["offset"] - here
            original = was.get("pad") or b""
            pad = original if len(original) == want else b"\0" * want
        body += pad
        offset = base + len(body)
        body += payload

        radar = _radar_bytes(radar_colours, i, was["radar"] if was else None)
        headers += FRAME.pack(box.x, box.y, box.w, box.h, flags, radar,
                              was["reserved"] if was else 0, offset)

    return (HEADER.pack(0, width, height, count) + bytes(headers)
            + bytes(body) + bytes(tail or b""))


def _unchanged(sub, box, was):
    """Do these pixels still match the ones this frame was read with?

    If they do, the frame's original bytes go back untouched. That matters
    because a re-encode is only guaranteed to be CORRECT, not identical: the
    shipped RA2 art is littered with rows whose trailing zero run is declared
    one longer than the row is wide, and the same frame mixes padded and
    exact rows, so no rule recovers it. Copying does.
    """
    prior = was.get("plane")
    if prior is None or not was.get("payload"):
        return False
    if (box.x, box.y, box.w, box.h) != (was["x"], was["y"], was["w"], was["h"]):
        return False
    return prior.shape == sub.shape and np.array_equal(prior, sub)


def trailing_bytes(data, frames):
    """Whatever sits past the last frame's payload. Usually nothing.

    Derived by re-encoding the last frame rather than stored, because the
    format records where a frame STARTS and never how long it is.
    """
    last = max((f for f in frames if f["offset"] and f["plane"] is not None),
               key=lambda f: f["offset"], default=None)
    if last is None:
        return b""
    end = last["offset"] + len(last["payload"])
    return data[end:] if 0 < end <= len(data) else b""


def _unhex_tail(text):
    if not text:
        return b""
    try:
        return bytes.fromhex(text)
    except ValueError:
        return b""


def _source_from_meta(document):
    """The per-frame writer choices this document was loaded with.

    Metadata carries the scalars, which survive a trip through .ochre; the
    cache adds the original pixels and bytes, which do not. Frame count is
    checked because inserting or deleting a frame invalidates the pairing
    entirely, and a misaligned payload would be far worse than re-encoding.
    """
    cached = _SOURCES.get(document)
    if cached is not None and len(cached) != len(document.frames):
        cached = None

    out = []
    for i, frame in enumerate(document.frames):
        entry = {"flags": _int_meta(frame, "cnc.flags"),
                 "offset": _int_meta(frame, "cnc.offset"),
                 "reserved": _int_meta(frame, "cnc.reserved"),
                 "x": _int_meta(frame, "cnc.x"), "y": _int_meta(frame, "cnc.y"),
                 "w": _int_meta(frame, "cnc.w"), "h": _int_meta(frame, "cnc.h"),
                 "radar": None, "plane": None, "payload": b"", "pad": b""}
        if cached is not None:
            was = cached[i]
            entry.update({"plane": was["plane"], "payload": was["payload"],
                          "pad": was["pad"],
                          "x": was["x"], "y": was["y"],
                          "w": was["w"], "h": was["h"]})
        out.append(entry)
    return out


def _int_meta(frame, key):
    try:
        return int(frame.meta.get(key, "0"))
    except (TypeError, ValueError):
        return 0


def _radar_bytes(radar_colours, i, fallback=None):
    """The 4-byte radar field: an explicit colour, else whatever was there."""
    value = radar_colours[i] if radar_colours and i < len(radar_colours) else None
    if value is None:
        value = fallback
    if value is None:
        return b"\0\0\0\0"
    if isinstance(value, (bytes, bytearray)):
        return (bytes(value) + b"\0\0\0\0")[:4]
    r, g, b = value[:3]
    return bytes((int(r), int(g), int(b), 0))


# ---- the provider -------------------------------------------------------

class ShpFormat:
    """FormatProvider for SHP (TS/RA2).

    Frames become document frames, each holding one index-locked layer, and
    the per-frame rectangle and flags are kept in frame metadata so a
    round-trip can reproduce them rather than re-deriving them.
    """

    name = "cnc.shp"
    # The theater suffixes are here as well as on the TMP provider, and that
    # is not a mistake. In TS/RA2 an extension names the THEATER, not the
    # format: terrain templates and the theater-specific sprites that sit on
    # them -- trees, smudges, bridges, overlays -- all end in .tem or .sno.
    # Measured on the shipped art, 1392 of 2108 theater-suffixed files are
    # SHPs rather than templates. Both providers therefore claim them, and
    # the host sniffs to decide, which is exactly the case its two-candidate
    # path exists for.
    extensions = ("shp", "tem", "sno", "urb", "ubn", "des", "lun")
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
        tail = trailing_bytes(data, frames)
        if tail:
            doc.meta["cnc.tail"] = tail.hex()

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
            frame.meta["cnc.offset"] = str(entry["offset"])
            frame.meta["cnc.reserved"] = str(entry["reserved"])
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

            # NOT refresh_derived() here. The rgba cache is four times the
            # size of the index plane it is derived from, and Surface.plane()
            # builds it coherently on first access -- so materialising it for
            # a frame nobody is looking at costs 80% of the document's memory
            # to cache something that will be thrown away by the residency
            # policy before it is read. Measured on a real 136-frame sprite:
            # 5.08 GB resident, against 203 MB of file.
            #
            # Keeping frames warm as they are built is a second copy of the
            # same mistake, so the budget is applied as we go rather than
            # after the damage. enforce_residency keeps the frames nearest
            # the current one and compresses the rest.
            if len(doc.frames) % RESIDENCY_STRIDE == 0:
                doc.enforce_residency()
        doc.current = 0
        doc.enforce_residency()
        _SOURCES[doc] = frames
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
                # An empty frame still carried a radar colour, and blank
                # shadow frames routinely do. Dropping it here would undo the
                # preservation the loaded metadata exists for.
                stored = frame.meta.get("cnc.radar")
                radar.append(_unhex_radar(stored) if stored else None)
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

        source = _source_from_meta(document) if preserve else None
        data = build_shp(document.width, document.height, planes,
                         transparent=0, compress=True,
                         radar_colours=radar, bounds=bounds, source=source,
                         tail=_unhex_tail(document.meta.get("cnc.tail")))
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
