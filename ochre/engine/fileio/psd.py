# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop documents (.psd, and .psb for large ones), read and written.

Written from Adobe's published "Photoshop File Formats Specification".
Pillow can read a PSD's layers, but not their opacity, blend mode,
visibility, masks, clipping or group structure -- which is most of what makes
a PSD worth opening as one -- so this reads the layer records itself.

READ: 8-, 16- and 32-bit RGB and greyscale. Ochre works in 8 bits, so deeper
files are reduced on opening -- rounded, not truncated (see _narrow) -- and
the document carries a note saying so, which the window, the headless CLI
and the control server pass on. Layers with names (Unicode when present),
position, opacity, blend mode, visibility, clipping, layer masks, and groups
(nested), whether in the layer info or, as Photoshop stores 16- and 32-bit
layers, in the document's Lr16 / Lr32 block. Raw, RLE (PackBits) and ZIP
(with or without prediction) channel data. Anything else -- CMYK, Lab, adjustment and smart-object
layers, layer effects -- falls back to the flattened composite the file
carries, so the picture still opens.

WRITE: 8-bit RGB with transparency. Raster and fill layers (as pixels),
groups, opacity, blend modes, visibility, clipping, layer masks, Unicode
names, and the flattened composite (which is what non-layer readers show).
Adjustment layers and live layer styles have no portable form here, so the
saver reports them as warnings rather than silently dropping them.

KEPT: what an opened PSD holds that Ochre does not model -- unknown image
resources, a layer's tagged blocks (type, smart objects, Photoshop's own
effects, adjustments, IDs, ...), document-level tagged blocks and the global
layer mask info -- is written back as it was read, so open-then-save loses
nothing nobody touched. The few kinds an edit can make false follow the
rules beside CONTENT, below.
"""

import struct
import zlib

import numpy as np

from .. import accel, budget
from ..atomic import write_bytes

# Photoshop blend keys <-> Ochre blend modes. "pass" is a group's Pass
# Through; "diss" (dissolve) has no Ochre equivalent and reads as Normal.
BLEND_KEYS = {
    b"norm": "Normal", b"dark": "Darken", b"mul ": "Multiply", b"idiv": "ColorBurn",
    b"lbrn": "LinearBurn", b"dkCl": "DarkerColor", b"lite": "Lighten", b"scrn": "Screen",
    b"div ": "ColorDodge", b"lddg": "Additive", b"lgCl": "LighterColor", b"over": "Overlay",
    b"sLit": "SoftLight", b"hLit": "HardLight", b"vLit": "VividLight", b"lLit": "LinearLight",
    b"pLit": "PinLight", b"hMix": "HardMix", b"diff": "Difference", b"smud": "Exclusion",
    b"fsub": "Subtract", b"fdiv": "Divide", b"hue ": "Hue", b"sat ": "Saturation",
    b"colr": "Color", b"lum ": "Luminosity", b"pass": "PassThrough",
}
KEY_FOR = {v: k for k, v in BLEND_KEYS.items()}

# The specification's limit on channels, for the file and for one layer.
MAX_CHANNELS = 56


# ---- PackBits ------------------------------------------------------------------------
#
# The RLE most PSDs use. Decoding and encoding run in the C accelerator when
# it is built (whole channels at a time, no Python per packet); the numpy
# backend's versions are the specification (accel/fallback.py).

def packbits_decode(data, expected):
    """Decode one PackBits-compressed row to `expected` bytes."""
    return accel.packbits_decode(data, [len(data)], expected).tobytes()


def packbits_encode(row):
    """Encode one row (bytes) with PackBits; runs of 3+ become repeat packets."""
    row = np.frombuffer(bytes(row), np.uint8)
    return accel.packbits_encode(row.reshape(1, -1))[1]


def _rle_rows(r, rows, rowbytes, psb):
    """`rows` PackBits rows read from `r`: the byte-count table, then the
    data. Returns (rows, rowbytes) uint8."""
    width = 4 if psb else 2
    counts = np.frombuffer(r.take(rows * width), ">u4" if psb else ">u2")
    return accel.packbits_decode(r.take(int(counts.sum(dtype=np.int64))), counts, rowbytes)


# ---- reading ---------------------------------------------------------------------------

class _Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def take(self, n):
        out = self.d[self.p:self.p + n]
        if len(out) < n:
            raise ValueError("PSD truncated")
        self.p += n
        return out

    def u8(self):
        return self.take(1)[0]

    def u16(self):
        return struct.unpack(">H", self.take(2))[0]

    def i16(self):
        return struct.unpack(">h", self.take(2))[0]

    def u32(self):
        return struct.unpack(">I", self.take(4))[0]

    def i32(self):
        return struct.unpack(">i", self.take(4))[0]

    def u64(self):
        return struct.unpack(">Q", self.take(8))[0]


def _narrow(raw, h, w, depth):
    """Samples at `depth` bits -> (h, w) uint8, ROUNDED to Ochre's 8 bits.

    16-bit is round(v / 257), half up (65535 / 257 = 255 exactly, and no
    16-bit value lies halfway, 257 being odd). Truncating with v >> 8
    instead would put every value whose low byte is 128 or more one step
    too dark. 32-bit is floating point: clipped to 0..1 (NaN reads as 0)
    and rounded to v * 255, taken as stored -- no transfer curve is
    applied to it."""
    n = h * w
    if depth == 8:
        return np.frombuffer(raw, np.uint8, n).reshape(h, w)
    if depth == 16:
        v = np.frombuffer(raw, ">u2", n).astype(np.uint32)
        return ((v + 128) // 257).astype(np.uint8).reshape(h, w)
    f = np.nan_to_num(np.frombuffer(raw, ">f4", n).astype(np.float32), nan=0.0)
    return np.floor(np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(h, w)


def _unpredict(raw, h, w, depth):
    """Undo ZIP-with-prediction: a running sum along each row, of bytes at
    8 bits and of 16-bit samples at 16. A 32-bit row stores its floats'
    four byte planes one after another, summed bytewise; summing, then
    interleaving the planes back, gives big-endian floats."""
    if depth == 8:
        a = np.frombuffer(raw, np.uint8, h * w).reshape(h, w)
        return np.cumsum(a, axis=1, dtype=np.uint8).tobytes()
    if depth == 16:
        a = np.frombuffer(raw, ">u2", h * w).reshape(h, w).astype(np.uint16)
        return np.cumsum(a, axis=1, dtype=np.uint16).astype(">u2").tobytes()
    a = np.frombuffer(raw, np.uint8, h * w * 4).reshape(h, 4 * w)
    a = np.cumsum(a, axis=1, dtype=np.uint8).reshape(h, 4, w)
    return np.ascontiguousarray(a.transpose(0, 2, 1)).tobytes()


def _channel(r, w, h, depth, psb, length):
    """One channel's pixels (h, w) uint8, from compressed channel data."""
    start = r.p
    comp = r.u16()
    rowbytes = w * (depth // 8)
    if w == 0 or h == 0:
        r.p = start + length
        return np.zeros((h, w), np.uint8)
    if comp == 0:
        raw = r.take(rowbytes * h)
    elif comp == 1:
        raw = _rle_rows(r, h, rowbytes, psb)
    elif comp in (2, 3):
        # Bounded by what the channel can hold: a zip bomb in a channel
        # stops at rowbytes * h instead of at the end of memory.
        raw = budget.inflate(r.take(length - 2), rowbytes * h, "a layer channel")
        if len(raw) < rowbytes * h:
            raise ValueError("a PSD layer channel inflates to less than its size")
        if comp == 3:
            raw = _unpredict(raw, h, w, depth)
    else:
        raise ValueError("unknown PSD compression %d" % comp)
    r.p = start + length
    return _narrow(raw, h, w, depth)


def _pascal(r, pad):
    n = r.u8()
    name = r.take(n)
    total = 1 + n
    if total % pad:
        r.take(pad - total % pad)
    return name.decode("latin-1", "replace")


# Tagged-block keys whose length is 8 bytes in a PSB (psd-tools' set, which
# Kooka's codec uses as well); every other key's length is 4 bytes.
BIG_KEYS = frozenset((b"Alph", b"FELS", b"FEid", b"FMsk", b"FXid", b"LMsk", b"Layr", b"Lr16",
                      b"Lr32", b"Mt16", b"Mt32", b"Mtrn", b"PxSD", b"artd", b"cinf", b"extd",
                      b"extn", b"lnk2", b"lnk3", b"lnkE", b"pths"))
SIGNATURES = (b"8BIM", b"8B64")


def _tagged(r, end, psb, pad):
    """Tagged blocks from r.p up to `end`: [(key, data, raw, at)], raw being
    the block exactly as stored (signature to padding) and `at` where its
    data starts. Data is padded to `pad` bytes (2 in a layer record, 4 at
    document level) -- or less, when the next signature starts sooner, as
    some writers pad less. A block that does not parse ends the walk."""
    out = []
    d = r.d
    while r.p + 12 <= end:
        start = r.p
        if d[start:start + 4] not in SIGNATURES:
            break
        r.p += 4
        key = r.take(4)
        n = r.u64() if psb and key in BIG_KEYS else r.u32()
        if r.p + n > end:
            r.p = start
            break
        at = r.p
        data = r.take(n)
        skip = (-n) % pad
        for candidate in (skip, n % 2, 0):
            if r.p + candidate >= end or d[r.p + candidate:r.p + candidate + 4] in SIGNATURES:
                skip = candidate
                break
        r.p = min(end, r.p + skip)
        out.append((key, data, d[start:r.p], at))
    return out


def _resources(section):
    """The image resources section as [(id, raw block)] in stored order, and
    the bytes after the last block that parses (kept as they are)."""
    out = []
    r = _Reader(section)
    end = len(section)
    try:
        while r.p + 12 <= end and section[r.p:r.p + 4] in (b"8BIM", b"MeSa", b"8B64", b"AgHg",
                                                            b"PHUT", b"DCSR"):
            start = r.p
            r.take(4)
            rid = r.u16()
            n = r.u8()
            r.take(n + ((n + 1) % 2))         # Pascal name, padded to even
            size = r.u32()
            r.take(size + size % 2)
            out.append((rid, section[start:r.p]))
    except ValueError:
        r.p = start
    return out, section[r.p:]


class PSDLayer:
    def __init__(self):
        self.name = ""
        self.rect = (0, 0, 0, 0)          # top, left, bottom, right
        self.channels = []                # (id, length)
        self.blend = "Normal"
        self.blend_key = b"norm"          # as stored, known to Ochre or not
        self.opacity = 255
        self.clipping = False
        self.hidden = False
        self.flags = 0                    # the record's flag byte, as stored
        self.mask_rect = None
        self.mask_default = 255
        self.mask_disabled = False
        self.section = 0                  # lsct: 0 layer, 1/2 group open/closed, 3 group end
        self.blend_if = None
        self.ranges = b""                 # the blending ranges as stored
        self.fill = 255                   # iOpa: Fill opacity
        self.knockout = False             # knko (PSD stores one flag: shallow)
        self.blocks = []                  # tagged blocks Ochre does not model: (key, data, raw)
        self.index = 0                    # the record's place in the file, bottom-first
        self.pixels = None                # (h, w, 4)
        self.mask = None                  # (mh, mw)


# Tagged blocks the reader turns into Ochre's own layer state. The writer
# makes them from that state; they are never copied.
MODELED = frozenset((b"luni", b"lsct", b"lsdk", b"iOpa", b"knko"))


class PSDFile:
    """Everything parse() reads from one file."""

    def __init__(self):
        self.width = self.height = 0
        self.depth = 8
        self.mode = 3
        self.nchan = 3
        self.psb = False
        self.layers = []                  # bottom-first
        self.composite = None
        self.resources = []               # image resources: (id, raw block)
        self.resources_tail = b""         # what follows the last block that parses
        self.global_mask = b""            # global layer mask info, as stored
        self.global_blocks = []           # document-level tagged blocks: (key, data, raw)


def read(path):
    """Parse a PSD/PSB. Returns (width, height, layers bottom-first, composite)."""
    f = parse(path)
    return f.width, f.height, f.layers, f.composite


def parse(path):
    """Parse a PSD/PSB into a PSDFile."""
    with open(path, "rb") as fh:
        data = fh.read()
    f = PSDFile()
    r = _Reader(data)
    if r.take(4) != b"8BPS":
        raise ValueError("not a Photoshop file")
    version = r.u16()
    f.psb = psb = version == 2
    r.take(6)
    f.nchan = nchan = r.u16()
    f.height, f.width = height, width = r.u32(), r.u32()
    f.depth, f.mode = depth, mode = r.u16(), r.u16()
    if depth not in (8, 16, 32):
        raise ValueError("PSD bit depth %d is not supported (8, 16 and 32 are)" % depth)
    # Sizes the header declares, checked before anything that size exists.
    budget.check_frame(width, height, "the image")
    if not 1 <= nchan <= MAX_CHANNELS:
        raise ValueError("PSD declares %d channels (the format allows 1 to %d)"
                         % (nchan, MAX_CHANNELS))
    r.take(r.u32())                       # colour mode data
    f.resources, f.resources_tail = _resources(r.take(r.u32()))
    lm_len = r.u64() if psb else r.u32()
    lm_end = r.p + lm_len
    if lm_len:
        li_len = r.u64() if psb else r.u32()
        li_end = r.p + li_len
        if li_len:
            f.layers = _layer_info(r, li_end, depth, mode, psb)
        r.p = li_end
        if r.p + 4 <= lm_end:
            f.global_mask = r.take(r.u32())
            blocks = _tagged(r, lm_end, psb, 4)
            # 16- and 32-bit files keep their layers in a document-level
            # block instead, and leave the layer info above empty.
            wanted = {16: b"Lr16", 32: b"Lr32"}.get(depth)
            for key, block, _raw, at in blocks:
                if key == wanted and not f.layers:
                    r.p = at
                    f.layers = _layer_info(r, at + len(block), depth, mode, psb)
            f.global_blocks = [b[:3] for b in blocks]
    r.p = lm_end
    f.composite = _read_composite(r, width, height, nchan, depth, mode, psb)
    return f


def _layer_info(r, end, depth, mode, psb):
    """The layer records and their channel data, from r.p to `end`."""
    layers = []
    count = abs(r.i16())
    # Per frame: each layer (and each layer mask, and the composite) may be
    # as large as one frame, however many layers there are.
    allowance = budget.Budget(2 * count + 1, "the file")
    for index in range(count):
        L = PSDLayer()
        L.index = index
        t, l, b, rt = r.i32(), r.i32(), r.i32(), r.i32()
        L.rect = (t, l, b, rt)
        nlc = r.u16()
        if nlc > MAX_CHANNELS:
            raise ValueError("a PSD layer declares %d channels (at most %d)"
                             % (nlc, MAX_CHANNELS))
        for _c in range(nlc):
            cid = r.i16()
            clen = r.u64() if psb else r.u32()
            L.channels.append((cid, clen))
        if r.take(4) not in SIGNATURES:
            raise ValueError("bad layer record signature")
        L.blend_key = r.take(4)
        L.blend = BLEND_KEYS.get(L.blend_key, "Normal")
        L.opacity = r.u8()
        L.clipping = r.u8() == 1
        L.flags = flags = r.u8()
        L.hidden = bool(flags & 2)
        r.u8()
        extra = r.u32()
        extra_end = r.p + extra
        mlen = r.u32()
        if mlen >= 20:
            mt, ml, mb, mr = r.i32(), r.i32(), r.i32(), r.i32()
            L.mask_default = r.u8()
            mflags = r.u8()
            L.mask_disabled = bool(mflags & 2)
            L.mask_rect = (mt, ml, mb, mr)
            r.take(mlen - 18)
        else:
            r.take(mlen)
        br = r.u32()                      # blending ranges: Photoshop's Blend If
        L.ranges = r.take(br)
        if br >= 8:
            vals = tuple(L.ranges[:8])
            # Gray source then destination, each black lo/hi, white lo/hi.
            if vals != (0, 0, 255, 255, 0, 0, 255, 255):
                L.blend_if = vals
        L.name = _pascal(r, 4)
        for key, data, raw, _at in _tagged(r, extra_end, psb, 2):
            if key == b"luni" and len(data) >= 4:
                n = struct.unpack(">I", data[:4])[0]
                L.name = data[4:4 + n * 2].decode("utf-16-be", "replace")
            elif key in (b"lsct", b"lsdk") and len(data) >= 4:
                L.section = struct.unpack(">I", data[:4])[0]
                # A group's blend mode lives here, after "8BIM"; the record's
                # own key is often just "norm".
                if len(data) >= 12 and data[4:8] == b"8BIM" and data[8:12] in BLEND_KEYS:
                    L.blend_key = data[8:12]
                    L.blend = BLEND_KEYS[L.blend_key]
            elif key == b"iOpa" and len(data) >= 1:
                L.fill = data[0]
            elif key == b"knko" and len(data) >= 1:
                L.knockout = data[0] != 0
            if key not in MODELED:
                L.blocks.append((key, data, raw))
        r.p = extra_end
        layers.append(L)
    for L in layers:
        t, l, b, rt = L.rect
        w, h = max(0, rt - l), max(0, b - t)
        allowance.charge(w, h, 4, "layer %r" % L.name)
        if L.mask_rect is not None:
            mt, ml, mb, mr = L.mask_rect
            allowance.charge(max(0, mr - ml), max(0, mb - mt), 1,
                             "the mask of layer %r" % L.name)
        planes = {}
        for cid, clen in L.channels:
            if cid in (-2, -3) and L.mask_rect is not None:
                mt, ml, mb, mr = L.mask_rect
                planes[cid] = _channel(r, max(0, mr - ml), max(0, mb - mt), depth, psb, clen)
            else:
                planes[cid] = _channel(r, w, h, depth, psb, clen)
        px = np.zeros((h, w, 4), np.uint8)
        if mode == 1:                       # greyscale
            g = planes.get(0, np.zeros((h, w), np.uint8))
            px[..., 0] = px[..., 1] = px[..., 2] = g
        else:
            for c in range(3):
                if c in planes:
                    px[..., c] = planes[c]
        px[..., 3] = planes.get(-1, np.full((h, w), 255, np.uint8))
        L.pixels = px
        if -2 in planes and L.mask_rect is not None:
            L.mask = planes[-2]
    if r.p > end:
        raise ValueError("PSD layer data runs past its section")
    return layers


def _read_composite(r, w, h, nchan, depth, mode, psb):
    try:
        comp = r.u16()
    except ValueError:
        return None
    rowbytes = w * (depth // 8)
    # Only the colour channels and the merged alpha are kept; any further
    # (spot or saved selection) channels are passed over.
    keep = min(nchan, 2 if mode == 1 else 4)
    if comp == 1:
        rows = _rle_rows(r, h * nchan, rowbytes, psb)
        planes = [_narrow(rows[c * h:(c + 1) * h], h, w, depth) for c in range(keep)]
    elif comp == 0:
        planes = [_narrow(r.take(rowbytes * h), h, w, depth) for _c in range(keep)]
    else:
        return None
    out = np.zeros((h, w, 4), np.uint8)
    if mode == 1:
        out[..., 0] = out[..., 1] = out[..., 2] = planes[0]
        out[..., 3] = planes[1] if nchan > 1 else 255
    elif mode == 3 and nchan >= 3:
        for c in range(3):
            out[..., c] = planes[c]
        out[..., 3] = planes[3] if nchan > 3 else 255
    else:
        return None
    if nchan > (1 if mode == 1 else 3):
        out = _unmatte(out)
    return out


def _matte(rgba):
    """Photoshop stores the merged image's colour MATTED against white where
    it is partly transparent (so a reader ignoring alpha still shows it
    right). 44 at 75% alpha is stored as 96. Measured on real files."""
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    out = rgba.copy()
    out[..., :3] = np.clip(np.rint(rgba[..., :3] * a + 255.0 * (1 - a)), 0, 255)
    return out


def _unmatte(rgba):
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    safe = np.where(a > 0, a, 1.0)
    out = rgba.copy()
    rgb = (rgba[..., :3] - 255.0 * (1 - a)) / safe
    rgb = np.where(a > 0, rgb, 0)
    out[..., :3] = np.clip(np.rint(rgb), 0, 255)
    return out


def read_icc(path):
    """The embedded ICC profile (image resource 1039), or None."""
    with open(path, "rb") as fh:
        data = fh.read()
    r = _Reader(data)
    if r.take(4) != b"8BPS":
        return None
    r.take(2 + 6 + 2 + 4 + 4 + 2 + 2)
    r.take(r.u32())                       # colour mode data
    end = r.u32()
    end += r.p
    while r.p + 12 <= end:
        if r.take(4) not in (b"8BIM", b"MeSa"):
            return None
        rid = r.u16()
        n = r.u8()
        r.take(n + ((n + 1) % 2))         # Pascal name, padded to even
        size = r.u32()
        block = r.take(size)
        if size % 2:
            r.take(1)
        if rid == 1039:
            return bytes(block)
    return None


def _icc_resource(doc):
    """Image resource 1039 for the document's profile, or b''."""
    data = (getattr(doc, "meta", {}) or {}).get("icc_data")
    if not data:
        return b""
    import base64
    icc = base64.b64decode(data)
    block = b"8BIM" + struct.pack(">H", 1039) + b"\x00\x00" + struct.pack(">I", len(icc)) + icc
    if len(icc) % 2:
        block += b"\x00"
    return block


def load(path, settings=None):
    """A PSD as an Ochre Document, with its colour profile."""
    doc = _load(path, settings)
    try:
        icc = read_icc(path)
    except (OSError, ValueError, IndexError):
        icc = None
    if icc:
        from .. import colormanage
        try:
            prof = colormanage.open_profile(icc)
            if not colormanage.is_srgb(prof):
                colormanage.set_doc_profile(doc, prof)
            else:
                # sRGB is Ochre's default and is not stored on the document,
                # so "no profile" at save time still means this one.
                doc.psd_source.icc_is_default = True
        except Exception:                                   # noqa: BLE001
            pass
    return doc


def _load(path, settings=None):
    """A PSD as an Ochre Document: layers, groups, masks and all."""
    from ..document import Document
    from ..layer import LayerGroup, LayerMask
    from ..geometry import Rect
    f = parse(path)
    width, height, layers, composite = f.width, f.height, f.layers, f.composite
    doc = Document(width, height, settings=settings)
    # Said when the file opens, not discovered later: Ochre works in 8 bits.
    doc.open_notes = []
    if f.depth == 16:
        doc.open_notes.append(
            "This is a 16-bit PSD; Ochre edits in 8 bits per channel, so it was "
            "reduced to 8 bits (rounded) on opening, and saving writes an 8-bit PSD.")
    elif f.depth == 32:
        doc.open_notes.append(
            "This is a 32-bit (floating-point) PSD; Ochre edits in 8 bits per channel, "
            "so it was reduced to 8 bits on opening (values clipped to 0..1, then "
            "rounded), and saving writes an 8-bit PSD.")
    doc.psd_source = PSDSource(f)
    if not layers:
        layer = doc.add_layer("Background")
        if composite is not None:
            doc.cell(layer).plane("rgba")[...] = composite
            doc.cell(layer)._touched(doc.bounds)
        return doc
    stack = [doc.root]
    for L in layers:                       # bottom-first in the file
        if L.section == 3:                 # the hidden marker that OPENS a group, bottom-up
            group = LayerGroup(name="Group")
            group.psd_source = {"divider": L.index, "divider_blocks": L.blocks}
            stack[-1].add(group)
            stack.append(group)
            continue
        if L.section in (1, 2):            # the group's own record CLOSES it
            group = stack.pop() if len(stack) > 1 else LayerGroup()
            group.name = L.name
            group.opacity = L.opacity
            group.visible = not L.hidden
            group.blend = L.blend
            group.psd_source = dict(getattr(group, "psd_source", None) or {}, **_kept(L))
            continue
        node = doc.add_layer(L.name or "Layer", parent=stack[-1])
        node.psd_source = _kept(L)
        node.opacity = L.opacity
        node.visible = not L.hidden
        node.blend = L.blend
        node.clip = L.clipping
        node.blend_if = L.blend_if
        node.fill_opacity = L.fill
        node.knockout = "shallow" if L.knockout else "none"
        t, l, b, rt = L.rect
        cell = doc.cell(node)
        dst = Rect(l, t, rt - l, b - t).clipped_to(width, height)
        if dst is not None and L.pixels is not None and L.pixels.size:
            cell.plane("rgba")[dst.slice()] = L.pixels[dst.y - t:dst.y - t + dst.h,
                                                       dst.x - l:dst.x - l + dst.w]
            cell._touched(dst)
        if L.mask is not None:
            node.mask = LayerMask(node)
            node.mask_enabled = not L.mask_disabled
            mcell = doc.cell(node.mask)
            mpx = mcell.plane("rgba")
            mpx[..., :3] = L.mask_default
            mpx[..., 3] = 255
            mt, ml, mb, mr = L.mask_rect
            mrect = Rect(ml, mt, mr - ml, mb - mt).clipped_to(width, height)
            if mrect is not None:
                sub = L.mask[mrect.y - mt:mrect.y - mt + mrect.h, mrect.x - ml:mrect.x - ml + mrect.w]
                for c in range(3):
                    mpx[mrect.slice() + (c,)] = sub
            mcell._touched(doc.bounds)
    # What every layer looks like as opened, to tell at save time whether it
    # (and the whole document) was changed since.
    for node in doc.root.walk():
        src = getattr(node, "psd_source", None)
        if src is not None and node is not doc.root:
            src["state"] = _state(doc, node, doc.frame)
    return doc

# ---- what Ochre keeps without understanding it ----------------------------------------
#
# Kooka Pictura's rule, "preserve, don't model": what a PSD holds that Ochre
# has no model for -- image resources, a layer's tagged blocks, document-
# level tagged blocks, the global layer mask info -- is kept as it was read
# and written back as it was, so open-then-save loses nothing nobody
# touched. Blocks Ochre DOES model are always written from Ochre's state
# instead (MODELED). A few kinds describe something an edit can make
# false, and those follow the rules below; everything else is copied
# verbatim, in its original order, whatever was edited. The bytes live on
# the document and its layers as `psd_source` attributes, not in `meta`:
# meta goes into an .ochre file's one-line INI manifest, and an embedded
# smart object's data (megabytes) does not belong there. So they last as
# long as the opened document; Save As .ochre and reopening drops them.

# Layer blocks Photoshop RENDERS the layer from: type, placed and smart
# objects, fill layers, shape content and stroke, adjustments. Kept only
# while the layer's pixels are exactly as opened; after a pixel edit
# Photoshop would redraw the layer from them and silently undo the edit,
# so they are dropped and the layer becomes the plain pixels Ochre saved.
CONTENT = frozenset((b"TySh", b"tySh", b"SoLd", b"SoLE", b"PlLd", b"plLd", b"SoCo", b"GdFl",
                     b"PtFl", b"vscg", b"vstk", b"CgEd", b"levl", b"curv", b"brit", b"blnc",
                     b"hue2", b"hue ", b"selc", b"mixr", b"grdm", b"phfl", b"expA", b"vibA",
                     b"nvrt", b"thrs", b"post", b"clrL", b"blwh"))
# Photoshop's own layer effects: kept unless the layer has Ochre layer
# styles, which the saver bakes into the pixels -- both would be drawn.
EFFECTS = frozenset((b"lfx2", b"lrFX", b"lmfx"))
# A vector mask: kept unless the layer has an Ochre vector mask, which the
# saver applies to the pixels (and which may differ from the old one).
VECTOR_MASK = frozenset((b"vmsk", b"vsms"))
# Document-level blocks that ARE the source depth's layers or merged
# transparency. The layers are rewritten as 8-bit layer info, so these are
# never copied.
DEPTH_BLOCKS = frozenset((b"Lr16", b"Lr32", b"Layr", b"Mt16", b"Mt32"))
# Image resources with rules of their own (all others are copied):
ICC_PROFILE = 1039                  # written from the document's profile
THUMBNAILS = (1033, 1036)           # a picture of the image: only if nothing changed
LAYER_INDEXED = (1024, 1026, 1069, 1072)    # per-record arrays: only if the
#                                     records are the same ones in the same order
CHANNEL_RESOURCES = (1006, 1007, 1045, 1053, 1077)  # alpha-channel names and display:
#                                     only if no extra channel was dropped
CHANNEL_BLOCKS = frozenset((b"Alph",))


class PSDSource:
    """The parts of an opened PSD that Ochre keeps without modeling."""

    def __init__(self, f):
        self.width, self.height = f.width, f.height
        self.psb = f.psb
        self.records = len(f.layers)
        self.resources = list(f.resources)
        self.resources_tail = f.resources_tail
        self.global_mask = f.global_mask
        self.global_blocks = [b for b in f.global_blocks if b[0] not in DEPTH_BLOCKS]
        # Spot or saved-selection channels beyond colour and one alpha are
        # not read, so what describes them goes stale when saved without them.
        self.extra_channels = f.nchan > (1 if f.mode == 1 else 3) + 1
        self.icc_is_default = False


def _kept(L):
    """A layer record's unmodeled parts, for its node's psd_source."""
    return {"index": L.index, "blocks": L.blocks, "ranges": L.ranges, "blend_if": L.blend_if,
            "blend_key": L.blend_key, "flags": L.flags}


def _plane_crc(doc, node, frame):
    cell = doc.cells.get((node.id, frame.id)) if node is not None else None
    arr = cell.planes.get("rgba") if cell is not None else None
    if arr is None or not arr.any():
        return 0
    return zlib.crc32(np.ascontiguousarray(arr))


def _state(doc, node, frame):
    """(everything that decides how the layer looks, its pixels' CRC)."""
    props = (node.name, node.opacity, node.visible, node.blend, node.is_group)
    if node.is_group:
        return props, 0
    mask = getattr(node, "mask", None)
    props += (getattr(node, "clip", False), getattr(node, "fill_opacity", 255),
              getattr(node, "knockout", "none"), getattr(node, "blend_if", None),
              getattr(node, "kind", "raster"), repr(getattr(node, "params", None)),
              repr(getattr(node, "styles", None)), getattr(node, "vector_mask", None) is None,
              getattr(node, "mask_enabled", True), _plane_crc(doc, mask, frame))
    return props, _plane_crc(doc, node, frame)


def _reframe(key, data, raw, psb, pad):
    """A kept tagged block as this writer stores it. A PSD's block is its
    original bytes; a PSB's is re-framed with a 4-byte length."""
    if not psb:
        return raw
    return b"8BIM" + key + struct.pack(">I", len(data)) + data + bytes((-len(data)) % pad)


def _kept_blocks(src, psb, node=None, pixels_same=True):
    """The node's kept tagged blocks, after the rules above."""
    out = b""
    for key, data, raw in src or ():
        if key in CONTENT and not pixels_same:
            continue
        if key in EFFECTS and node is not None and getattr(node, "styles", None):
            continue
        if key in VECTOR_MASK and node is not None and getattr(node, "vector_mask", None) is not None:
            continue
        out += _reframe(key, data, raw, psb, 2)
    return out


def _resource_payload(raw):
    """The data of one raw image-resource block."""
    n = raw[6]
    at = 7 + n + ((n + 1) % 2)
    size = struct.unpack(">I", raw[at:at + 4])[0]
    return raw[at + 4:at + 4 + size]


def _resources_out(doc, src, untouched, same_records):
    """The image resources section to write."""
    icc = _icc_resource(doc)
    if src is None:
        return icc
    out = b""
    wrote_icc = False
    for rid, raw in src.resources:
        if rid == ICC_PROFILE:
            if wrote_icc:
                continue
            wrote_icc = True
            if not icc:
                # No profile on the document: the original stays only when it
                # was the sRGB that "no profile" means.
                out += raw if src.icc_is_default else b""
            elif _resource_payload(icc) == _resource_payload(raw):
                out += raw
            else:
                out += icc                  # replaced where it was
            continue
        if rid in THUMBNAILS and not untouched:
            continue
        if rid in LAYER_INDEXED and not same_records:
            continue
        if rid in CHANNEL_RESOURCES and src.extra_channels:
            continue
        out += raw
    if icc and not wrote_icc:
        out += icc
    return out + src.resources_tail


# ---- writing ---------------------------------------------------------------------------

def _rle_channel(plane):
    """(compression=1) + row byte counts + PackBits rows, for one channel."""
    counts, data = accel.packbits_encode(plane)
    return struct.pack(">H", 1) + counts.astype(">u2").tobytes() + data


def _pascal_bytes(name, pad=4):
    raw = name.encode("latin-1", "replace")[:255]
    out = bytes((len(raw),)) + raw
    if len(out) % pad:
        out += bytes(pad - len(out) % pad)
    return out


def _block(key, payload):
    if len(payload) % 2:
        payload += b"\x00"
    return b"8BIM" + key + struct.pack(">I", len(payload)) + payload


def save(doc, path, compositor=None):
    """Write a PSD. Returns a list of warnings (what could not be kept)."""
    from ..compositor import Compositor, _Leaf, mask_value
    width, height = doc.width, doc.height
    if width > 30000 or height > 30000:
        raise ValueError("PSD is limited to 30000 px a side")
    comp = compositor or Compositor(doc)
    frame = doc.frame
    warnings = []
    records = []      # (record bytes, channel data bytes), bottom-first
    source = getattr(doc, "psd_source", None)
    psb = source is not None and source.psb
    order = []        # each record's index in the opened file (None: new), as written
    untouched = [source is not None and (doc.width, doc.height) == (source.width, source.height)]

    def kept(node, index_key="index"):
        """The node's psd_source, or None when it has none or another node
        already used it (a copy carries no claim to the original's blocks)."""
        src = getattr(node, "psd_source", None)
        index = src.get(index_key) if src else None
        if index is None or index in order:
            order.append(None)
            untouched[0] = False
            return None
        order.append(index)
        if index_key == "index" and src.get("state") != _state(doc, node, frame):
            untouched[0] = False
        return src

    def layer_record(name, rect, chans, blend, opacity, clip, hidden, section=None, mask=None,
                     blend_if=None, fill=255, knockout=False, src=None, blocks=b"",
                     pixels_same=True):
        t, l, b, r = rect
        rec = struct.pack(">iiii", t, l, b, r)
        rec += struct.pack(">H", len(chans))
        data = b""
        lengths = []
        for cid, plane in chans:
            enc = _rle_channel(plane) if plane.size else struct.pack(">H", 0)
            data += enc
            lengths.append((cid, len(enc)))
        for cid, n in lengths:
            rec += struct.pack(">hI", cid, n)
        key = KEY_FOR.get(blend, b"norm")
        if src and src["blend_key"] not in BLEND_KEYS and blend == "Normal":
            # A mode Ochre does not know (Dissolve, say) read as Normal; while
            # it still is Normal, the original key is the truth.
            key = src["blend_key"]
        rec += b"8BIM" + key
        # Flag bits other than visibility (the transparency lock, "pixel
        # data irrelevant") are kept -- the last one not once the pixels were
        # edited, when they are all there is.
        flags = (src["flags"] & 0x19) if src else 0
        if not pixels_same:
            flags &= ~0x10
        rec += struct.pack(">BBBB", opacity, 1 if clip else 0, flags | (2 if hidden else 0) | 8, 0)
        extra = b""
        if mask is not None:
            mt, ml, mb, mr, default, disabled = mask
            extra += struct.pack(">I", 20) + struct.pack(">iiiiBB", mt, ml, mb, mr, default,
                                                          2 if disabled else 0) + b"\x00\x00"
        else:
            extra += struct.pack(">I", 0)
        if src and src["blend_if"] == (tuple(blend_if) if blend_if else None):
            # Unchanged Blend If: the ranges as stored, per-channel ones too.
            extra += struct.pack(">I", len(src["ranges"])) + src["ranges"]
        elif blend_if:
            extra += struct.pack(">I", 8) + bytes(int(v) & 255 for v in blend_if[:8])
        else:
            extra += struct.pack(">I", 0)
        extra += _pascal_bytes(name)
        uni = name.encode("utf-16-be")
        extra += _block(b"luni", struct.pack(">I", len(name)) + uni)
        if section is not None:
            # A group's blend mode belongs here too: Photoshop reads it from
            # the section divider, not from the record.
            extra += _block(b"lsct", struct.pack(">I", section)
                            + (b"8BIM" + KEY_FOR.get(blend, b"norm") if section != 3 else b""))
        if fill != 255:
            extra += _block(b"iOpa", struct.pack(">B3x", fill))
        if knockout:
            extra += _block(b"knko", struct.pack(">B3x", 1))
        extra += blocks
        rec += struct.pack(">I", len(extra)) + extra
        return rec, data

    def emit(group):
        for node in group.children:
            if node.is_group and getattr(node, "region", None):
                warnings.append("%s %r is saved as a plain group (its edges are not kept)"
                                % (node.region.get("type", "region"), node.name))
            if node.is_group:
                empty = np.zeros((0, 0), np.uint8)
                chans = [(-1, empty), (0, empty), (1, empty), (2, empty)]
                src = kept(node, "divider")
                records.append(layer_record("</Layer group>", (0, 0, 0, 0), chans, "Normal", 255,
                                            False, False, section=3,
                                            blocks=_kept_blocks(src and src["divider_blocks"], psb)))
                emit(node)
                src = kept(node)
                records.append(layer_record(node.name or "Group", (0, 0, 0, 0), chans,
                                            node.blend, node.opacity, False, not node.visible,
                                            section=1, src=src,
                                            blocks=_kept_blocks(src and src["blocks"], psb, node)))
                continue
            kind = getattr(node, "kind", "raster")
            if kind == "smart":
                warnings.append("smart object %r is saved as pixels (it stays smart in .ochre)"
                                % node.name)
            if kind == "text":
                warnings.append("text layer %r is saved as pixels (it stays editable in .ochre)"
                                % node.name)
            if kind == "shape":
                warnings.append("shape layer %r is saved as pixels (it stays a shape in .ochre)"
                                % node.name)
            if kind == "adjustment":
                warnings.append("adjustment layer %r is not kept in PSD" % node.name)
                continue
            src = kept(node)
            if getattr(node, "styles", None):
                warnings.append("layer styles on %r are flattened into its pixels" % node.name)
            if getattr(node, "vector_mask", None) is not None:
                warnings.append("the vector mask on %r is applied to its pixels" % node.name)
            if kind == "raster" and (getattr(node, "params", None) or {}).get("filters"):
                warnings.append("non-destructive filters on %r are merged into its pixels" % node.name)
            cell = doc.cells.get((node.id, frame.id))
            if kind == "raster" and (cell is None or cell.is_empty()):
                px = np.zeros((height, width, 4), np.uint8)
            else:
                px = np.zeros((height, width, 4), np.uint8)
                leaf = _Leaf(cell, comp.blends.opcode("Normal"), 255, node)
                # The mask and Blend If travel as themselves, not baked in.
                saved_mask, node.mask = node.mask, None
                saved_bi, node.blend_if = getattr(node, "blend_if", None), None
                # Fill opacity too -- unless styles are baked into the pixels,
                # when only the baked result is right and iOpa stays 255.
                saved_fill = getattr(node, "fill_opacity", 255)
                if not getattr(node, "styles", None):
                    node.fill_opacity = 255
                try:
                    comp._blend_leaf(px, leaf, doc.bounds, frame)
                finally:
                    node.mask = saved_mask
                    node.blend_if = saved_bi
                    node.fill_opacity = saved_fill
            ys, xs = np.nonzero(px[..., 3])
            if ys.size:
                t, l, b, r = int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
            else:
                t = l = b = r = 0
            sub = px[t:b, l:r]
            chans = [(-1, np.ascontiguousarray(sub[..., 3])), (0, np.ascontiguousarray(sub[..., 0])),
                     (1, np.ascontiguousarray(sub[..., 1])), (2, np.ascontiguousarray(sub[..., 2]))]
            mask = None
            if getattr(node, "mask", None) is not None:
                mcell = doc.cells.get((node.mask.id, frame.id))
                if mcell is not None and not mcell.is_empty():
                    chans.append((-2, mask_value(mcell.plane("rgba"))))
                    mask = (0, 0, height, width, 255, not node.mask_enabled)
            same = src is not None and src["state"][1] == _plane_crc(doc, node, frame)
            records.append(layer_record(node.name or "Layer", (t, l, b, r), chans, node.blend,
                                        node.opacity, getattr(node, "clip", False),
                                        not node.visible, mask=mask,
                                        blend_if=getattr(node, "blend_if", None),
                                        fill=255 if getattr(node, "styles", None)
                                        else getattr(node, "fill_opacity", 255),
                                        knockout=getattr(node, "knockout", "none") != "none",
                                        src=src, blocks=_kept_blocks(src and src["blocks"], psb,
                                                                     node, same),
                                        pixels_same=same))
            if getattr(node, "knockout", "none") == "deep":
                warnings.append("deep knockout on %r is saved as shallow (PSD keeps one flag)"
                                % node.name)
    emit(doc.root)

    same_records = source is not None and order == list(range(source.records))
    untouched = untouched[0] and same_records
    head = b"8BPS" + struct.pack(">H6xHIIHH", 1, 4, height, width, 8, 3)
    res = _resources_out(doc, source, untouched, same_records)
    body = struct.pack(">I", 0) + struct.pack(">I", len(res)) + res
    info = struct.pack(">h", -len(records))                   # negative: merged alpha present
    info += b"".join(rec for rec, _ in records) + b"".join(dat for _, dat in records)
    if len(info) % 2:
        info += b"\x00"
    layer_info = struct.pack(">I", len(info)) + info
    if source is None:
        lm = layer_info + struct.pack(">I", 0)                # no global mask info
    else:
        lm = layer_info + struct.pack(">I", len(source.global_mask)) + source.global_mask
        lm += b"".join(_reframe(k, d, raw, psb, 4) for k, d, raw in source.global_blocks
                       if not (k in CHANNEL_BLOCKS and source.extra_channels))
    body += struct.pack(">I", len(lm)) + lm
    merged = _matte(comp.composite(doc.bounds, frame))
    # All four channels' rows in one table, then all their data.
    counts, data = accel.packbits_encode(np.moveaxis(merged, 2, 0).reshape(4 * height, width))
    body += struct.pack(">H", 1) + counts.astype(">u2").tobytes() + data
    write_bytes(path, head + body)
    return warnings


def is_psd(path):
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"8BPS"
    except OSError:
        return False
