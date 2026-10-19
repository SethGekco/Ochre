# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Surface: a named container of uint8 planes that travel together.

Three features in the design turn out to be one requirement wearing three
hats -- index-locked colour, a frame axis, and non-colour per-pixel data all
need "a layer carries more than one plane, and those planes must move together
through dirty rects, history and the file format". So that is what is built,
once, here.

    normal raster        planes {rgba}                  authoritative (rgba,)
    index-locked         planes {index, rgba}           authoritative (index,)
    with height data     planes {rgba, height}          authoritative both
    index-locked+height  planes {index, rgba, height}   authoritative (index, height)

Three invariants fall out, and they are what make the rest cheap:

1. One content_bbox, one dirty region, one history delta per Surface,
   regardless of plane count. Planes cannot desync.
2. The compositor reads planes["rgba"] and nothing else. Extra planes cost
   literally zero in the composite path -- there is no branch to skip.
3. Only authoritative planes are saved and undone. Derived planes are caches,
   reconstructible at any time, and never enter history.

Storage is flat numpy, not tiled. Measured: a strided sub-rect view into a big
flat array costs ~0.02 ms at 256x256, while gathering the same region from 16
tiles costs ~0.029 ms because of roughly 2 microseconds of Python overhead per
tile. Tiles impose an interpreter tax exactly where numpy is weakest -- many
small operations -- to buy a locality win that dirty rects already provide.
Lazy allocation plus content_bbox recovers the one thing tiles genuinely win,
which is memory on sparse layers.

Every pixel access goes through this class rather than through .planes
directly, so a tiled backend could be substituted behind this interface
without touching the compositor, the tools or the history.
"""

import numpy as np

from .arith import apply_coverage, blend_over, expand_palette, mul255
from .geometry import DirtyRegion, Rect

# Channels per plane. Anything not listed is a scalar (single-channel) plane.
PLANE_CHANNELS = {"rgba": 4}

RGBA = "rgba"
INDEX = "index"


def plane_channels(name):
    return PLANE_CHANNELS.get(name, 1)


class Surface:
    """Pixel storage for one cell. Not thread-safe for writes."""

    __slots__ = ("width", "height", "planes", "authoritative", "palette",
                 "content_bbox", "dirty", "revision")

    def __init__(self, width, height, planes=(RGBA,), authoritative=None,
                 palette=None, max_dirty_rects=8):
        if width <= 0 or height <= 0:
            raise ValueError("surface must be at least 1x1, got %dx%d" % (width, height))
        self.width = int(width)
        self.height = int(height)
        self.planes = {name: None for name in planes}
        self.authoritative = tuple(authoritative if authoritative is not None
                                   else (planes[0],))
        for name in self.authoritative:
            if name not in self.planes:
                raise ValueError("authoritative plane %r is not present" % name)
        self.palette = palette
        self.content_bbox = None
        self.dirty = DirtyRegion(max_dirty_rects)
        # Bumped on every recorded write, so derived results (a layer's
        # styles) can tell they are stale without comparing pixels.
        self.revision = 0

    # ---- introspection ---------------------------------------------------

    @property
    def bounds(self):
        return Rect(0, 0, self.width, self.height)

    @property
    def pixels(self):
        """The rgba plane, allocating it if needed. The compositor's entry point."""
        return self.plane(RGBA)

    def has(self, name):
        return name in self.planes

    def is_allocated(self, name):
        return self.planes.get(name) is not None

    def is_derived(self, name):
        """A present plane that is a cache of an authoritative one."""
        return name in self.planes and name not in self.authoritative

    @property
    def index_locked(self):
        return INDEX in self.authoritative

    def is_empty(self):
        """True if nothing has ever been written to any authoritative plane."""
        return self.content_bbox is None

    def nbytes(self):
        return sum(p.nbytes for p in self.planes.values() if p is not None)

    # ---- allocation ------------------------------------------------------

    def plane(self, name):
        """The named plane, allocated zero-filled on first access.

        Lazy allocation is why a fresh 20-layer document costs no pixel
        memory at all.
        """
        if name not in self.planes:
            raise KeyError("surface has no plane %r (has %s)"
                           % (name, ", ".join(sorted(self.planes))))
        arr = self.planes[name]
        if arr is None:
            ch = plane_channels(name)
            shape = (self.height, self.width) if ch == 1 else (self.height, self.width, ch)
            arr = np.zeros(shape, dtype=np.uint8)
            self.planes[name] = arr
            # A derived cache must be coherent from the instant it exists.
            # An all-zero index plane means "every pixel is palette entry 0",
            # not "every pixel is transparent black" -- leaving the cache
            # zero-filled would make it disagree with its own source.
            if name == RGBA and self.is_derived(RGBA) and self.palette is not None:
                self._expand(self.bounds)
        return arr

    def drop_derived(self):
        """Free derived plane caches. They rebuild on next access.

        The rgba cache of a 4000x4000 index-locked layer is 64 MB and
        reconstructs in about 86 ms, so it is worth releasing under pressure.
        """
        freed = 0
        for name in list(self.planes):
            if self.is_derived(name) and self.planes[name] is not None:
                freed += self.planes[name].nbytes
                self.planes[name] = None
        return freed

    # ---- reading ---------------------------------------------------------

    def view(self, rect, name=RGBA):
        """Zero-copy strided view of a sub-rectangle. None if fully clipped."""
        r = rect.clipped_to(self.width, self.height)
        if r is None:
            return None
        return self.plane(name)[r.slice()]

    def read(self, rect, names=None):
        """Copies of the given planes over rect, as {name: array}.

        Defaults to the authoritative planes, which is exactly what a history
        delta needs -- derived caches are never recorded.
        """
        r = rect.clipped_to(self.width, self.height)
        if r is None:
            return {}
        names = self.authoritative if names is None else names
        return {n: self.plane(n)[r.slice()].copy() for n in names}

    # ---- writing ---------------------------------------------------------

    def _touched(self, rect):
        """Record a write: extend content_bbox and mark dirty. One call site."""
        self.content_bbox = rect if self.content_bbox is None \
            else self.content_bbox.union(rect)
        self.dirty.add(rect)
        self.revision += 1

    def write(self, rect, src, name=RGBA):
        """Overwrite a rectangle from src. src must match the clipped shape."""
        r = rect.clipped_to(self.width, self.height)
        if r is None:
            return None
        dst = self.plane(name)[r.slice()]
        src = np.asarray(src, dtype=np.uint8)
        if src.shape != dst.shape:
            raise ValueError("write shape %s does not match target %s"
                             % (src.shape, dst.shape))
        dst[...] = src
        self._after_write(r, name)
        return r

    def fill(self, rect, value, name=RGBA):
        """Fill a rectangle with a constant (a scalar, or an RGBA tuple)."""
        r = rect.clipped_to(self.width, self.height)
        if r is None:
            return None
        self.plane(name)[r.slice()] = value
        self._after_write(r, name)
        return r

    def apply_masked(self, rect, src, mask=None, name=RGBA):
        """The single chokepoint through which every tool writes pixels.

        For the rgba plane this is a straight-alpha source-over; for scalar
        planes it is a coverage lerp. mask is a (h,w) uint8 coverage plane, or
        None for full coverage.

        Routing every tool through one call is what makes selection clipping
        free and impossible to forget: the stroke path fuses the selection
        into the brush's own falloff with a single multiply, so antialiased
        selection edges give antialiased clipping, and effects and adjustments
        clip identically because they use this same call.
        """
        r = rect.clipped_to(self.width, self.height)
        if r is None:
            return None

        # On an index-locked surface the rgba plane is a CACHE. Writing
        # colour into it shows the right thing on screen and is discarded on
        # save, because only authoritative planes are written -- so a tool
        # that paints here has to paint indices instead. Doing that at this
        # one call is what gives every tool indexed support at once; doing it
        # per tool is how six of them came to be silently broken.
        if name == RGBA and self.index_locked:
            return self._apply_indexed(r, src, mask)

        plane = self.plane(name)
        dst = plane[r.slice()]
        src = np.asarray(src, dtype=np.uint8)

        if mask is not None:
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.shape[:2] != dst.shape[:2]:
                raise ValueError("mask shape %s does not match target %s"
                                 % (mask.shape[:2], dst.shape[:2]))

        if plane_channels(name) == 4:
            if mask is None and src.shape == dst.shape and np.all(src[..., 3] == 255):
                dst[...] = src            # fully opaque: skip the blend entirely
            else:
                dst[...] = blend_over(dst, src, mask)
        elif name == INDEX and mask is not None:
            # Indices are labels, not quantities. apply_coverage would lerp
            # them, and halfway between index 5 and index 200 is not a
            # blend of two colours -- it is a third, unrelated one. Scalar
            # planes that ARE quantities (height, z-data) still lerp below.
            dst[np.asarray(mask, dtype=np.uint8) >= 128] = \
                np.broadcast_to(src, dst.shape)[np.asarray(mask, dtype=np.uint8) >= 128]
        else:
            dst[...] = src if mask is None else apply_coverage(dst, src, mask)

        self._after_write(r, name)
        return r

    def _apply_indexed(self, r, src, mask):
        """Paint colour onto an index plane.

        COVERAGE BECOMES A THRESHOLD, and that is deliberate rather than a
        shortcut. Indices cannot be interpolated -- half of index 5 and half
        of index 200 is not index 102, it is a different colour entirely, and
        on a C&C palette it is frequently the shadow or a remap entry. So a
        pixel is either painted or it is not, which is also exactly what a
        pixel artist wants: an antialiased edge on a 60x60 sprite is a defect.

        The consequence worth stating: a soft brush behaves as a hard one
        here. That is the honest translation of a soft brush into a medium
        with no partial coverage.
        """
        from .quantize import index_for

        src = np.asarray(src, dtype=np.uint8)
        if src.ndim == 1:
            src = np.broadcast_to(src, (r.h, r.w, src.shape[-1]))
        dst = self.plane(INDEX)[r.slice()]

        alpha = src[..., 3]
        cover = alpha if mask is None else mul255(alpha, mask)
        write = cover >= 128
        if not write.any():
            return None

        idx = index_for(src[..., :3], self.palette)
        dst[write] = np.asarray(idx, dtype=np.uint8)[write]
        self._after_write(r, INDEX)
        return r

    def lerp_masked(self, rect, src, mask=None):
        """Move pixels TOWARD src by coverage. The write a filter brush needs.

        apply_masked composites src OVER dst, which can never make a pixel
        more transparent or take away colour that is there. A blur, a dodge,
        a smudge or a background eraser has to be able to move a pixel to
        any value at all, so this is a straight per-channel lerp instead.

        On an index-locked surface the result lands as indices: the colour is
        matched into the palette (exactly where it can be), transparent
        results take the palette's transparent entry, and coverage is a
        threshold for the same reason as everywhere else -- an index cannot
        be half one value and half another.
        """
        r = rect.clipped_to(self.width, self.height)
        if r is None:
            return None
        src = np.asarray(src, dtype=np.uint8)
        if src.ndim == 1:
            src = np.broadcast_to(src, (r.h, r.w, src.shape[-1]))
        if self.index_locked:
            from .quantize import index_for
            dst = self.plane(INDEX)[r.slice()]
            idx = np.asarray(index_for(src[..., :3], self.palette), dtype=np.uint8)
            clear = 0 if self.palette is None or self.palette.transparent is None \
                else int(self.palette.transparent)
            idx = np.where(src[..., 3] >= 128, idx, clear).astype(np.uint8)
            write = np.ones(dst.shape, dtype=bool) if mask is None \
                else (np.asarray(mask, dtype=np.uint8) >= 128)
            if not write.any():
                return None
            dst[write] = idx[write]
            self._after_write(r, INDEX)
            return r
        dst = self.plane(RGBA)[r.slice()]
        dst[...] = src if mask is None else apply_coverage(dst, src, mask)
        self._after_write(r, RGBA)
        return r

    def erase_masked(self, rect, mask=None):
        """Remove coverage. The counterpart to apply_masked, and its equal.

        The eraser used to reach past apply_masked and multiply the rgba
        alpha itself, which worked on a normal layer and did nothing
        whatsoever on an index-locked one -- the cache it edited was
        regenerated from indices it never touched. Erasing to an index plane
        means writing the palette's TRANSPARENT entry, which is a different
        operation, not a variation on the same one, so it gets its own call
        rather than a flag on the other.
        """
        r = rect.clipped_to(self.width, self.height)
        if r is None:
            return None

        if self.index_locked:
            clear = 0 if self.palette is None or self.palette.transparent is None \
                else int(self.palette.transparent)
            dst = self.plane(INDEX)[r.slice()]
            if mask is None:
                dst[...] = clear
            else:
                dst[np.asarray(mask, dtype=np.uint8) >= 128] = clear
            self._after_write(r, INDEX)
            return r

        dst = self.plane(RGBA)[r.slice()]
        if mask is None:
            dst[..., 3] = 0
        else:
            keep = (255 - np.asarray(mask, dtype=np.uint8)).astype(np.uint8)
            dst[..., 3] = mul255(dst[..., 3], keep)
        self._after_write(r, RGBA)
        return r

    def _after_write(self, r, name):
        """Keep derived planes coherent, then record the write.

        Writing an authoritative index plane refreshes the rgba cache over
        exactly the same rectangle -- measured at 0.058 ms per 256x256 region,
        which is noise against a ~1.9 ms composite. That refresh is why an
        index-locked layer needs no special case anywhere in the compositor:
        by the time it is composited it is ordinary RGBA.
        """
        if name == INDEX and self.is_derived(RGBA):
            self.refresh_derived(r)
        self._touched(r)

    # ---- derived plane maintenance ---------------------------------------

    def _expand(self, r):
        """Raw index -> rgba over r. Bypasses plane() to avoid re-entry."""
        idx = self.planes[INDEX]
        if idx is None:
            ch = plane_channels(INDEX)
            idx = np.zeros((self.height, self.width), dtype=np.uint8)
            self.planes[INDEX] = idx
        expand_palette(self.palette.render_entries, idx[r.slice()],
                       out=self.planes[RGBA][r.slice()])

    def refresh_derived(self, rect=None):
        """Rebuild rgba from index over rect (or the whole surface).

        Every path that rewrites planes wholesale -- undo, transforms, offset
        -- ends here, so it also counts as a write for the revision that
        caches (filters, styles, smart objects) key on."""
        self.revision += 1
        if not self.is_derived(RGBA) or self.palette is None:
            return None
        r = self.bounds if rect is None else rect.clipped_to(self.width, self.height)
        if r is None:
            return None
        self.plane(RGBA)                      # ensure allocated (and coherent)
        self._expand(r)
        return r

    def recolor_index(self, index):
        """Refresh only the pixels using a given palette index.

        Editing a palette entry must instantly recolour every pixel using it.
        Measured at 4000x4000: a masked update is ~17.3 ms against ~88 ms for
        a full re-expand, so this is the path a colour picker scrubs on.

        It touches exactly the pixels whose index is N -- including ones whose
        current RGB happens to match a different entry. No RGB-based approach
        can do that at all, and that capability is the whole reason the index
        plane is authoritative rather than derived.
        """
        if not self.is_derived(RGBA) or self.palette is None:
            return None
        if self.planes.get(INDEX) is None:
            return None
        idx = self.plane(INDEX)
        where = idx == index
        if not where.any():
            return None
        self.plane(RGBA)[where] = self.palette.render_entries[index]
        ys, xs = np.nonzero(where)
        r = Rect.from_points(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        self.dirty.add(r)
        return r

    # ---- whole-surface operations ----------------------------------------

    def tight_bbox(self):
        """The bounds of what is actually painted, measured, without storing it.

        content_bbox only ever grows -- it is a cheap upper bound kept for
        history and compositing -- so after a move or an erase it can enclose
        a lot of nothing. Anything that SHOWS bounds to a user (transform
        handles, a layer's position) wants these instead.
        """
        box = self.content_bbox
        if box is None:
            # Never recorded -- but an addon (or a script) that wrote a plane
            # directly leaves real pixels with no bounds, and a transform tool
            # that then silently did nothing would be baffling. Scan instead.
            if not any(self.planes.get(n) is not None for n in self.authoritative):
                return None
            box = self.bounds
        box = box.clipped_to(self.width, self.height)
        if box is None:
            return None
        if INDEX in self.authoritative and self.planes.get(INDEX) is not None:
            clear = 0 if self.palette is None or self.palette.transparent is None \
                else int(self.palette.transparent)
            here = self.planes[INDEX][box.slice()] != clear
        elif self.planes.get(RGBA) is not None:
            here = self.planes[RGBA][box.slice()][..., 3] > 0
        else:
            return None
        if not here.any():
            return None
        ys, xs = np.nonzero(here)
        return Rect.from_points(box.x + int(xs.min()), box.y + int(ys.min()),
                                box.x + int(xs.max()), box.y + int(ys.max()))

    def recompute_content_bbox(self):
        """Measure content_bbox from the data instead of trusting a guess.

        Used after whole-plane rewrites -- resample, rotate -- where the
        footprint cannot be predicted exactly: a bilinear or Lanczos filter
        spreads a pixel or two past the scaled rectangle, and every history
        snapshot is cropped to content_bbox, so a bbox that is one pixel too
        tight silently drops the edge on undo. Opaque alpha counts for rgba;
        for an index plane anything that is not the transparent entry; for
        any other plane anything non-zero.
        """
        mask = None
        for name in self.authoritative:
            arr = self.planes.get(name)
            if arr is None:
                continue
            if name == RGBA:
                here = arr[..., 3] > 0
            elif name == INDEX:
                clear = 0 if self.palette is None or self.palette.transparent is None \
                    else int(self.palette.transparent)
                here = arr != clear
            else:
                here = arr != 0
            mask = here if mask is None else (mask | here)
        if mask is None or not mask.any():
            self.content_bbox = None
            return None
        ys, xs = np.nonzero(mask)
        self.content_bbox = Rect.from_points(int(xs.min()), int(ys.min()),
                                             int(xs.max()), int(ys.max()))
        return self.content_bbox

    def clear(self):
        """Drop every plane and reset tracking. Cheaper than zero-filling."""
        for name in self.planes:
            self.planes[name] = None
        self.content_bbox = None
        self.dirty.clear()

    def copy(self):
        out = Surface(self.width, self.height, tuple(self.planes),
                      self.authoritative, self.palette, self.dirty.max_rects)
        for name, arr in self.planes.items():
            out.planes[name] = None if arr is None else arr.copy()
        out.content_bbox = self.content_bbox
        return out

    def readonly(self, name=RGBA):
        """A non-writeable view, for handing to effect or addon code.

        One line that turns a whole class of "the effect scribbled on its own
        input" bug into an immediate exception instead of silent corruption.
        """
        arr = self.plane(name).view()
        arr.flags.writeable = False
        return arr
