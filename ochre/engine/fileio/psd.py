# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Photoshop documents (.psd, and .psb for large ones), read and written.

Written from Adobe's published "Photoshop File Formats Specification".
Pillow can read a PSD's layers, but not their opacity, blend mode,
visibility, masks, clipping or group structure -- which is most of what makes
a PSD worth opening as one -- so this reads the layer records itself.

READ: 8- and 16-bit RGB and greyscale (16-bit is reduced to 8). Layers with
names (Unicode when present), position, opacity, blend mode, visibility,
clipping, layer masks, and groups (nested). Raw, RLE (PackBits) and ZIP
channel data. Anything else -- CMYK, Lab, adjustment and smart-object
layers, layer effects -- falls back to the flattened composite the file
carries, so the picture still opens.

WRITE: 8-bit RGB with transparency. Raster and fill layers (as pixels),
groups, opacity, blend modes, visibility, clipping, layer masks, Unicode
names, and the flattened composite (which is what non-layer readers show).
Adjustment layers and live layer styles have no portable form here, so the
saver reports them as warnings rather than silently dropping them.
"""

import struct

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


def _channel(r, w, h, depth, psb, length):
    """One channel's pixels (h, w) uint8, from compressed channel data."""
    start = r.p
    comp = r.u16()
    bpp = depth // 8
    rowbytes = w * bpp
    if w == 0 or h == 0:
        r.p = start + length
        return np.zeros((h, w), np.uint8)
    if comp == 0:
        raw = r.take(rowbytes * h)
    elif comp == 1:
        raw = _rle_rows(r, h, rowbytes, psb).tobytes()
    elif comp in (2, 3):
        # Bounded by what the channel can hold: a zip bomb in a channel
        # stops at rowbytes * h instead of at the end of memory.
        raw = budget.inflate(r.take(length - 2), rowbytes * h, "a layer channel")
        if comp == 3:
            a = np.frombuffer(raw, np.uint8 if bpp == 1 else ">u2").reshape(h, w).copy()
            a = np.cumsum(a, axis=1, dtype=a.dtype)
            raw = a.astype(">u2").tobytes() if bpp == 2 else a.astype(np.uint8).tobytes()
    else:
        raise ValueError("unknown PSD compression %d" % comp)
    r.p = start + length
    if bpp == 2:
        a = np.frombuffer(raw[:rowbytes * h], ">u2").reshape(h, w)
        return (a >> 8).astype(np.uint8)
    return np.frombuffer(raw[:rowbytes * h], np.uint8).reshape(h, w)


def _pascal(r, pad):
    n = r.u8()
    name = r.take(n)
    total = 1 + n
    if total % pad:
        r.take(pad - total % pad)
    return name.decode("latin-1", "replace")


class PSDLayer:
    def __init__(self):
        self.name = ""
        self.rect = (0, 0, 0, 0)          # top, left, bottom, right
        self.channels = []                # (id, length)
        self.blend = "Normal"
        self.opacity = 255
        self.clipping = False
        self.hidden = False
        self.mask_rect = None
        self.mask_default = 255
        self.mask_disabled = False
        self.section = 0                  # lsct: 0 layer, 1/2 group open/closed, 3 group end
        self.blend_if = None
        self.fill = 255                   # iOpa: Fill opacity
        self.knockout = False             # knko (PSD stores one flag: shallow)
        self.pixels = None                # (h, w, 4)
        self.mask = None                  # (mh, mw)


def read(path):
    """Parse a PSD/PSB. Returns (width, height, layers bottom-first, composite)."""
    with open(path, "rb") as fh:
        data = fh.read()
    r = _Reader(data)
    if r.take(4) != b"8BPS":
        raise ValueError("not a Photoshop file")
    version = r.u16()
    psb = version == 2
    r.take(6)
    nchan = r.u16()
    height, width = r.u32(), r.u32()
    depth, mode = r.u16(), r.u16()
    if depth not in (8, 16):
        raise ValueError("PSD bit depth %d is not supported (8 and 16 are)" % depth)
    # Sizes the header declares, checked before anything that size exists.
    budget.check_frame(width, height, "the image")
    if not 1 <= nchan <= MAX_CHANNELS:
        raise ValueError("PSD declares %d channels (the format allows 1 to %d)"
                         % (nchan, MAX_CHANNELS))
    r.take(r.u32())                       # colour mode data
    r.take(r.u32())                       # image resources
    lm_len = r.u64() if psb else r.u32()
    lm_end = r.p + lm_len
    layers = []
    if lm_len:
        li_len = r.u64() if psb else r.u32()
        li_end = r.p + li_len
        if li_len:
            count = abs(r.i16())
            # Per frame: each layer (and each layer mask, and the composite)
            # may be as large as one frame, however many layers there are.
            allowance = budget.Budget(2 * count + 1, "the file")
            for _ in range(count):
                L = PSDLayer()
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
                if r.take(4) not in (b"8BIM", b"8B64"):
                    raise ValueError("bad layer record signature")
                L.blend = BLEND_KEYS.get(r.take(4), "Normal")
                L.opacity = r.u8()
                L.clipping = r.u8() == 1
                flags = r.u8()
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
                br = r.u32()              # blending ranges: Photoshop's Blend If
                ranges = r.take(br)
                if br >= 8:
                    vals = tuple(ranges[:8])
                    # Gray source then destination, each black lo/hi, white lo/hi.
                    if vals != (0, 0, 255, 255, 0, 0, 255, 255):
                        L.blend_if = vals
                L.name = _pascal(r, 4)
                while r.p + 12 <= extra_end:
                    sig = r.take(4)
                    if sig not in (b"8BIM", b"8B64"):
                        break
                    key = r.take(4)
                    big = psb and key in (b"LMsk", b"Lr16", b"Lr32", b"Layr", b"Mt16", b"Mt32",
                                          b"Mtrn", b"Alph", b"FMsk", b"lnk2", b"FEid", b"FXid",
                                          b"PxSD")
                    klen = r.u64() if big else r.u32()
                    kend = r.p + klen
                    if key == b"luni" and klen >= 4:
                        n = r.u32()
                        L.name = r.take(n * 2).decode("utf-16-be", "replace")
                    elif key in (b"lsct", b"lsdk") and klen >= 4:
                        L.section = r.u32()
                    elif key == b"iOpa" and klen >= 1:
                        L.fill = r.u8()
                    elif key == b"knko" and klen >= 1:
                        L.knockout = r.u8() != 0
                    r.p = kend
                    if klen % 2 and r.p < extra_end and data[r.p:r.p + 4] not in (b"8BIM", b"8B64"):
                        r.p += 1
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
            r.p = li_end
    r.p = lm_end
    composite = _read_composite(r, width, height, nchan, depth, mode, psb)
    return width, height, layers, composite


def _read_composite(r, w, h, nchan, depth, mode, psb):
    try:
        comp = r.u16()
    except ValueError:
        return None
    bpp = depth // 8
    rowbytes = w * bpp
    chans = []
    if comp == 1:
        rows = _rle_rows(r, h * nchan, rowbytes, psb)
        for c in range(nchan):
            chans.append(rows[c * h:(c + 1) * h].tobytes())
    elif comp == 0:
        for c in range(nchan):
            chans.append(r.take(rowbytes * h))
    else:
        return None
    planes = [np.frombuffer(c, np.uint8 if bpp == 1 else ">u2").reshape(h, w) for c in chans]
    if bpp == 2:
        planes = [(p >> 8).astype(np.uint8) for p in planes]
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
        except Exception:                                   # noqa: BLE001
            pass
    return doc


def _load(path, settings=None):
    """A PSD as an Ochre Document: layers, groups, masks and all."""
    from ..document import Document
    from ..layer import LayerGroup, LayerMask
    from ..geometry import Rect
    width, height, layers, composite = read(path)
    doc = Document(width, height, settings=settings)
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
            stack[-1].add(group)
            stack.append(group)
            continue
        if L.section in (1, 2):            # the group's own record CLOSES it
            group = stack.pop() if len(stack) > 1 else LayerGroup()
            group.name = L.name
            group.opacity = L.opacity
            group.visible = not L.hidden
            group.blend = L.blend
            continue
        node = doc.add_layer(L.name or "Layer", parent=stack[-1])
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
    return doc


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

    def layer_record(name, rect, chans, blend, opacity, clip, hidden, section=None, mask=None,
                     blend_if=None, fill=255, knockout=False):
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
        rec += b"8BIM" + KEY_FOR.get(blend, b"norm")
        rec += struct.pack(">BBBB", opacity, 1 if clip else 0, (2 if hidden else 0) | 8, 0)
        extra = b""
        if mask is not None:
            mt, ml, mb, mr, default, disabled = mask
            extra += struct.pack(">I", 20) + struct.pack(">iiiiBB", mt, ml, mb, mr, default,
                                                          2 if disabled else 0) + b"\x00\x00"
        else:
            extra += struct.pack(">I", 0)
        if blend_if:
            extra += struct.pack(">I", 8) + bytes(int(v) & 255 for v in blend_if[:8])
        else:
            extra += struct.pack(">I", 0)
        extra += _pascal_bytes(name)
        uni = name.encode("utf-16-be")
        extra += _block(b"luni", struct.pack(">I", len(name)) + uni)
        if section is not None:
            extra += _block(b"lsct", struct.pack(">I", section))
        if fill != 255:
            extra += _block(b"iOpa", struct.pack(">B3x", fill))
        if knockout:
            extra += _block(b"knko", struct.pack(">B3x", 1))
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
                records.append(layer_record("</Layer group>", (0, 0, 0, 0), chans, "Normal", 255,
                                            False, False, section=3))
                emit(node)
                records.append(layer_record(node.name or "Group", (0, 0, 0, 0), chans,
                                            node.blend, node.opacity, False, not node.visible,
                                            section=1))
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
            records.append(layer_record(node.name or "Layer", (t, l, b, r), chans, node.blend,
                                        node.opacity, getattr(node, "clip", False),
                                        not node.visible, mask=mask,
                                        blend_if=getattr(node, "blend_if", None),
                                        fill=255 if getattr(node, "styles", None)
                                        else getattr(node, "fill_opacity", 255),
                                        knockout=getattr(node, "knockout", "none") != "none"))
            if getattr(node, "knockout", "none") == "deep":
                warnings.append("deep knockout on %r is saved as shallow (PSD keeps one flag)"
                                % node.name)
    emit(doc.root)

    head = b"8BPS" + struct.pack(">H6xHIIHH", 1, 4, height, width, 8, 3)
    res = _icc_resource(doc)                                  # the colour profile, if any
    body = struct.pack(">I", 0) + struct.pack(">I", len(res)) + res
    info = struct.pack(">h", -len(records))                   # negative: merged alpha present
    info += b"".join(rec for rec, _ in records) + b"".join(dat for _, dat in records)
    if len(info) % 2:
        info += b"\x00"
    layer_info = struct.pack(">I", len(info)) + info
    lm = layer_info + struct.pack(">I", 0)                    # no global mask info
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
