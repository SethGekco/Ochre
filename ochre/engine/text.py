# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Text rasterisation, in the engine.

The design document assumed this would have to live in the UI, because glyph
rasterisation needs font machinery and the engine may not import Qt. That
turned out to be the wrong conclusion: Pillow is ALREADY an engine dependency
and ships FreeType, so text rasterises here alongside everything else. The
engine stays Qt-free, a batch script can render text with no display, and the
text tool is testable headlessly like every other tool.

What comes out is a COVERAGE PLANE -- the same (H, W) uint8 currency a brush
dab and a selection mask trade in. So text composites, clips to a selection
and enters undo through exactly the same path as everything else, with no
text-specific case anywhere downstream.

Two things are deliberately not done here. Subpixel (LCD) antialiasing is
avoided: it produces colour fringes that are correct against a known opaque
background and wrong on a transparent layer, which is most of what an image
editor draws on. And font FALLBACK is left to fontconfig rather than
reimplemented -- if a face lacks a glyph, that is fontconfig's problem to
solve and it is better at it.
"""

import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import Rect

LEFT, CENTER, RIGHT = "left", "center", "right"
ALIGNMENTS = (LEFT, CENTER, RIGHT)

DEFAULT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "~/.local/share/fonts",
    "~/.fonts",
)


class FontBook:
    """Finds font faces and caches loaded ones.

    Resolution order: an explicit path, then fontconfig, then a scan of the
    usual directories. fontconfig is tried before scanning because it
    understands family names, styles and aliases like "sans-serif", and
    reimplementing that badly is a classic waste.
    """

    def __init__(self, directories=None, default_family="DejaVu Sans"):
        self.directories = [os.path.expanduser(d)
                            for d in (directories or DEFAULT_DIRS)]
        self.default_family = default_family
        self._faces = None
        self._cache = {}

    # ---- discovery -------------------------------------------------------

    def faces(self):
        """Every .ttf/.otf found, as {display name: path}. Scanned once."""
        if self._faces is not None:
            return self._faces
        found = {}
        for directory in self.directories:
            if not os.path.isdir(directory):
                continue
            for root, _dirs, files in os.walk(directory):
                for fn in sorted(files):
                    if fn.lower().endswith((".ttf", ".otf")):
                        found.setdefault(os.path.splitext(fn)[0],
                                         os.path.join(root, fn))
        self._faces = found
        return found

    def resolve(self, family):
        """A font family name to a file path, or None."""
        if not family:
            family = self.default_family
        if os.path.isfile(family):
            return family

        path = self._via_fontconfig(family)
        if path:
            return path

        faces = self.faces()
        if family in faces:
            return faces[family]
        lowered = family.lower().replace(" ", "")
        for name, candidate in faces.items():
            if name.lower().replace(" ", "") == lowered:
                return candidate
        for name, candidate in faces.items():
            if lowered in name.lower().replace(" ", ""):
                return candidate
        return None

    @staticmethod
    def _via_fontconfig(family):
        try:
            out = subprocess.run(["fc-match", "-f", "%{file}", family],
                                 capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        path = (out.stdout or "").strip()
        return path if path and os.path.isfile(path) else None

    def load(self, family, size):
        """An ImageFont at a size, cached. Falls back rather than raising."""
        key = (family or self.default_family, int(size))
        if key in self._cache:
            return self._cache[key]

        path = self.resolve(family)
        font = None
        if path:
            try:
                font = ImageFont.truetype(path, max(1, int(size)))
            except OSError:
                font = None
        if font is None:
            path = self.resolve(self.default_family)
            if path:
                try:
                    font = ImageFont.truetype(path, max(1, int(size)))
                except OSError:
                    font = None
        if font is None:
            # Pillow's built-in bitmap font: tiny and size-fixed, but it
            # means text NEVER fails outright on a machine with no fonts.
            font = ImageFont.load_default()
        self._cache[key] = font
        return font

    def names(self):
        return sorted(self.faces())


DEFAULT_BOOK = FontBook()


# ---- rasterising ---------------------------------------------------------

def measure(text, font, line_spacing=1.0):
    """(width, height, [line heights]) for a possibly multi-line string."""
    lines = (text or "").split("\n")
    widths, heights = [], []
    for line in lines:
        box = font.getbbox(line or " ")
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (0, 0)
    step = int(round((ascent + descent) * float(line_spacing))) or max(heights + [1])
    return max(widths + [0]), step * len(lines), step


def render_coverage(text, family="DejaVu Sans", size=24, align=LEFT,
                    line_spacing=1.0, book=None, antialias=True):
    """Rasterise `text` to a tight (h, w) uint8 coverage plane.

    Returns (coverage, width, height). An empty string yields a 0x0 plane
    rather than a special case for callers to remember.
    """
    book = book or DEFAULT_BOOK
    font = book.load(family, size)
    if not text:
        return np.zeros((0, 0), dtype=np.uint8), 0, 0

    width, height, step = measure(text, font, line_spacing)
    # Pad generously: glyphs overhang their advance width, and italic or
    # script faces overhang a lot. Cropping to content afterwards costs
    # nothing and is safer than guessing the exact extent.
    pad = max(4, int(size))
    canvas = Image.new("L", (max(1, width + pad * 2), max(1, height + pad * 2)), 0)
    draw = ImageDraw.Draw(canvas)
    # "L" mode gives greyscale antialiasing from FreeType. Never "1", which
    # would be hard-edged, and never an RGB mode, which on some stacks
    # enables subpixel rendering and its colour fringes.
    draw.multiline_text((pad, pad), text, fill=255, font=font,
                        align=align, spacing=max(0, step - _line_height(font)))

    arr = np.asarray(canvas, dtype=np.uint8)
    if not antialias:
        arr = ((arr >= 128).astype(np.uint8)) * 255

    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        return np.zeros((0, 0), dtype=np.uint8), 0, 0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = np.ascontiguousarray(arr[y0:y1, x0:x1])
    return cropped, cropped.shape[1], cropped.shape[0]


class Layout:
    """A rasterised block plus the metrics a caret needs.

    render_coverage() crops its output tight, which is right for compositing
    and useless for a caret: the caret has to sit at a position computed in
    UNCROPPED layout space and then be shifted by however much the crop
    removed. So the crop delta is returned rather than discarded.
    """

    __slots__ = ("coverage", "width", "height", "crop_x", "crop_y",
                 "line_step", "lines", "font")

    def __init__(self, coverage, width, height, crop_x, crop_y, line_step,
                 lines, font):
        self.coverage = coverage
        self.width = width
        self.height = height
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.line_step = line_step
        self.lines = lines
        self.font = font

    def caret_at(self, index):
        """(x, y, height) of the caret before character `index`.

        Coordinates are relative to the cropped block's top-left, which is
        what a caller already has a position for.
        """
        index = max(0, min(int(index), sum(len(l) for l in self.lines)
                           + max(0, len(self.lines) - 1)))
        row, col, seen = 0, index, 0
        for i, line in enumerate(self.lines):
            if index <= seen + len(line):
                row, col = i, index - seen
                break
            seen += len(line) + 1          # +1 for the newline
            row = i + 1
            col = 0

        prefix = self.lines[row][:col] if row < len(self.lines) else ""
        try:
            advance = self.font.getlength(prefix)
        except AttributeError:
            advance = self.font.getbbox(prefix)[2] if prefix else 0

        return (int(round(advance)) - self.crop_x,
                row * self.line_step - self.crop_y,
                self.line_step)

    def index_at(self, x, y):
        """The character index nearest a point -- for click-to-position."""
        row = max(0, min(int((y + self.crop_y) // max(1, self.line_step)),
                         len(self.lines) - 1))
        line = self.lines[row]
        target = x + self.crop_x
        best, best_dist = 0, None
        for col in range(len(line) + 1):
            try:
                advance = self.font.getlength(line[:col])
            except AttributeError:
                advance = self.font.getbbox(line[:col])[2] if col else 0
            dist = abs(advance - target)
            if best_dist is None or dist < best_dist:
                best, best_dist = col, dist
        return sum(len(l) + 1 for l in self.lines[:row]) + best


def layout(text, family="DejaVu Sans", size=24, align=LEFT, line_spacing=1.0,
           book=None, antialias=True):
    """Rasterise AND report the metrics a caret needs."""
    book = book or DEFAULT_BOOK
    font = book.load(family, size)
    lines = (text or "").split("\n")

    width, height, step = measure(text or " ", font, line_spacing)
    pad = max(4, int(size))
    canvas = Image.new("L", (max(1, width + pad * 2), max(1, height + pad * 2)), 0)
    if text:
        draw = ImageDraw.Draw(canvas)
        draw.multiline_text((pad, pad), text, fill=255, font=font, align=align,
                            spacing=max(0, step - _line_height(font)))

    arr = np.asarray(canvas, dtype=np.uint8)
    if not antialias:
        arr = ((arr >= 128).astype(np.uint8)) * 255

    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        # An empty or whitespace-only string still needs a caret position, so
        # the layout reports the padded origin rather than collapsing.
        return Layout(np.zeros((0, 0), dtype=np.uint8), 0, 0, pad, pad,
                      step, lines, font)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = np.ascontiguousarray(arr[y0:y1, x0:x1])
    return Layout(cropped, cropped.shape[1], cropped.shape[0],
                  x0 - pad, y0 - pad, step, lines, font)


def _line_height(font):
    try:
        ascent, descent = font.getmetrics()
        return ascent + descent
    except AttributeError:
        return font.getbbox("Ag")[3]


def place(coverage, x, y, align=LEFT, baseline=False):
    """Where a coverage block lands, given an anchor and an alignment."""
    h, w = coverage.shape[:2] if coverage.size else (0, 0)
    if align == CENTER:
        x -= w / 2.0
    elif align == RIGHT:
        x -= w
    if baseline:
        y -= h
    return Rect(int(round(x)), int(round(y)), w, h)
