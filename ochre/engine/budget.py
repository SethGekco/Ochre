# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The budget an opened file decodes under.

A file's header, its layer rectangles and its compressed streams are chosen
by whoever made the file, so every reader checks the sizes a file DECLARES
here, before allocating anything for them. (The idea is Kooka Pictura's
budget.rs.) A refusal is a FileTooLarge: a ValueError whose message is
written for the person who picked the file, never a MemoryError, a frozen
machine or the out-of-memory killer.

THE CAP IS PER FRAME. A file with N frames, layers or cells -- an animation,
a layered document, a tile set -- may hold N frames each up to the cap, and
a Budget for it allows N times the per-frame cap in all. There is
deliberately no single global cap: a long animation must never be refused
merely for having many frames. What IS refused is one frame larger than the
cap, a side longer than MAX_SIDE, and a compressed stream that inflates past
what the frame it fills can hold.

The defaults, and why:

  frame_pixels  2**29 = 536,870,912 pixels in one frame: 2 GiB as RGBA, a
                23,170 x 23,170 square. A 20,000 x 20,000 photo (400 Mpx,
                1.6 GB as RGBA) fits with room to spare; a 2,052 x 2,816
                sprite frame (5.8 Mpx) fits 92 times over. 2 GiB is also
                Kooka's per-allocation cap. Ochre stores pixels as RGBA, so
                this is the largest frame worth even trying to hold.
  max_side      300,000 pixels on either side: the large-document Photoshop
                format's own limit, so no real file is refused for it. It is
                there to stop absurd values (a 2**31-wide strip) before any
                arithmetic is done with them.
  text_bytes    64 MiB for a non-pixel part: a manifest, a layer-stack
                description, a palette file. Real ones are kilobytes.

All three are [Open] settings in data/engine.ini (configure() applies them).
"""

import contextlib
import os
import warnings
import zlib

FRAME_PIXELS = 1 << 29
MAX_SIDE = 300000
TEXT_BYTES = 64 << 20

_limits = {"frame_pixels": FRAME_PIXELS, "max_side": MAX_SIDE, "text_bytes": TEXT_BYTES}


class FileTooLarge(ValueError):
    """A file declares more than the open budget allows. The message says
    what, in words meant for the user."""


# ---- the limits ---------------------------------------------------------

def frame_pixels():
    return _limits["frame_pixels"]


def frame_bytes(channels=4):
    """The bytes one frame may decode to: the per-frame pixel cap at
    `channels` bytes a pixel (RGBA by default)."""
    return _limits["frame_pixels"] * channels


def encoded_frame_bytes():
    """The most an ENCODED frame (a PNG inside a container, say) can
    plausibly take: incompressible pixels plus a generous allowance for
    the format's own overhead."""
    raw = frame_bytes()
    return raw + raw // 64 + (1 << 20)


def text_bytes():
    return _limits["text_bytes"]


def max_side():
    return _limits["max_side"]


def configure(settings):
    """Apply the [Open] settings. Called once by the controller."""
    if settings is None:
        return
    for key, name in (("frame_pixels", "MaxFramePixels"), ("max_side", "MaxSide"),
                      ("text_bytes", "MaxTextBytes")):
        value = settings.get("Open", name)
        if isinstance(value, int) and value > 0:
            _limits[key] = value
    _pillow()


@contextlib.contextmanager
def limits(**overrides):
    """Temporarily different limits (frame_pixels=, max_side=, text_bytes=)."""
    saved = dict(_limits)
    _limits.update({k: int(v) for k, v in overrides.items() if k in _limits})
    _pillow()
    try:
        yield
    finally:
        _limits.clear()
        _limits.update(saved)
        _pillow()


# ---- checks -------------------------------------------------------------

def _mpx(pixels):
    return "%.2f megapixels" % (pixels / 1e6)


def check_frame(width, height, what="the image"):
    """Refuse a frame larger than the per-frame cap. Returns its pixel count.
    Call it with the DECLARED size, before allocating anything."""
    width, height = int(width), int(height)
    if width < 0 or height < 0:
        raise ValueError("%s declares an impossible size, %d x %d" % (what, width, height))
    side = _limits["max_side"]
    if width > side or height > side:
        raise FileTooLarge("%s is %d x %d pixels; Ochre opens images up to %d pixels on "
                           "a side" % (what, width, height, side))
    pixels = width * height
    if pixels > _limits["frame_pixels"]:
        raise FileTooLarge("%s is %d x %d pixels (%s); Ochre opens up to %s in one frame "
                           "or layer" % (what, width, height, _mpx(pixels),
                                         _mpx(_limits["frame_pixels"])))
    return pixels


class Budget:
    """One file's running allowance: N frames get N times the per-frame cap.

        budget = Budget(len(frames), "sprite.gif")
        for f in frames:
            budget.charge(f.w, f.h, 4, "frame %d" % i)    # before allocating

    Each charge must itself fit the per-frame cap; the running total only
    trips when a reader allocates more than the frames it declared could
    hold -- the sign of a file whose parts point at each other.
    """

    def __init__(self, frames, what="the file"):
        self.frames = max(1, int(frames))
        self.what = what
        self.remaining = self.frames * frame_bytes()

    def charge(self, width, height, channels=4, what="a frame"):
        pixels = check_frame(width, height, what)
        need = pixels * int(channels)
        if need > self.remaining:
            raise FileTooLarge("%s asks for more pixels than its %d frame%s can hold"
                               % (self.what, self.frames, "" if self.frames == 1 else "s"))
        self.remaining -= need
        return pixels


def inflate(data, limit, what="compressed data"):
    """zlib-decompress `data` to at most `limit` bytes; more is refused
    without ever being produced -- a zip bomb stops at the limit, not at
    the end of memory."""
    d = zlib.decompressobj()
    out = d.decompress(data, limit)
    # At the limit and not at the end of the stream: one more byte decides.
    if not d.eof and d.decompress(d.unconsumed_tail, 1):
        raise FileTooLarge("%s inflates to more than the %d bytes it can hold"
                           % (what, limit))
    return out


def read_file(path, limit, what=None):
    """A file's bytes, refusing one larger than `limit` before reading it."""
    what = what or os.path.basename(path)
    size = os.path.getsize(path)
    if size > limit:
        raise FileTooLarge("%s is %d bytes; Ochre reads at most %d for this kind of file"
                           % (what, size, limit))
    with open(path, "rb") as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise FileTooLarge("%s grew past %d bytes while being read" % (what, limit))
    return data


def read_member(zf, name, limit, what=None):
    """One zip member's bytes, never more than `limit`.

    The declared size is checked first, and then the member is inflated a
    megabyte at a time. Both matter: zipfile's own read() inflates a whole
    compressed chunk in one call before trimming it to the declared size,
    so a member that lies about its size would otherwise be fully inflated
    in memory before the CRC check rejected it.
    """
    what = what or name
    info = zf.getinfo(name)
    if info.file_size > limit:
        raise FileTooLarge("%s is %d bytes uncompressed; at most %d is expected"
                           % (what, info.file_size, limit))
    out = bytearray()
    with zf.open(info) as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            out += chunk
            if len(out) > limit:
                raise FileTooLarge("%s inflates to more than %d bytes" % (what, limit))
    return out


# ---- Pillow -------------------------------------------------------------

def _pillow():
    """Pillow's own decompression-bomb check, set to the same per-frame cap
    (it warns above MAX_IMAGE_PIXELS and refuses above twice that)."""
    try:
        from PIL import Image
    except ImportError:                                     # pragma: no cover
        return
    Image.MAX_IMAGE_PIXELS = _limits["frame_pixels"]


def open_image(source, what=None):
    """Image.open with the per-frame cap applied to the declared size, before
    any pixel is decoded. `source` is a path or a file object. For a
    multi-frame image, check_image() each frame after seeking to it."""
    from PIL import Image
    _pillow()
    what = what or "the image"
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            img = Image.open(source)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            img = None
    if img is None:
        # Pillow refused on the header alone. Read just the header again with
        # its check off -- open() decodes no pixels -- to say how large.
        if hasattr(source, "seek"):
            source.seek(0)
        saved, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, None
        try:
            with Image.open(source) as probe:
                size = probe.size
        finally:
            Image.MAX_IMAGE_PIXELS = saved
        check_frame(size[0], size[1], what)
        raise FileTooLarge("%s is larger than %s in one frame, the most Ochre opens"
                           % (what, _mpx(_limits["frame_pixels"])))
    check_image(img, what)
    return img


def check_image(img, what="the image"):
    """check_frame for an opened (not yet loaded) Pillow image."""
    return check_frame(img.size[0], img.size[1], what)


_pillow()
