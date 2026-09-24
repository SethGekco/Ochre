# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Integer pixel arithmetic. This module is normative.

Every operation Ochre may one day accelerate in C is specified here in
*integer* arithmetic, and this specification is the contract. Floating point
is deliberately excluded from the canonical path: C compilers contract a*b+c
into FMA, auto-vectorise with different association orders, and numpy's own
reductions are pairwise. Bit-exact agreement between two independent floating
point implementations is not achievable without crippling both. Integer
operations are exactly defined in C and in numpy alike, so agreement is by
construction rather than by luck -- and the parity test is therefore
meaningful instead of flaky.

When the C extension and this module disagree, the C is wrong.

The core primitive is exact division by 255:

    t = a * b + 128
    mul255 = (t + (t >> 8)) >> 8

which equals round(a * b / 255) exactly for every a, b in 0..255. That identity
is checked over all 65536 operand pairs by tests/test_arith.py -- it is not
taken on faith.

This is also the arithmetic Paint.NET uses (INT_SCALE in UserBlendOps), which
is why its blend formulas drop into Ochre without translation error.
"""

import numpy as np


def mul255(a, b):
    """round(a * b / 255), exactly, for uint8-ranged inputs.

    Accepts scalars or numpy arrays. Computes in uint32 so the intermediate
    a*b+128 (max 65153) and the shifted sum cannot overflow.
    """
    t = np.asarray(a, dtype=np.uint32) * np.asarray(b, dtype=np.uint32) + 128
    return ((t + (t >> 8)) >> 8).astype(np.uint8)


def div255(t):
    """round(t / 255) for an already-computed product. Same identity as mul255.

    Use when the product was accumulated separately, e.g. after summing
    weighted samples.
    """
    t = np.asarray(t, dtype=np.uint32) + 128
    return ((t + (t >> 8)) >> 8).astype(np.uint8)


def lerp255(a, b, t):
    """a + (b - a) * t/255, exact, no overflow, no signed intermediates.

    Written as two complementary scaled terms rather than a signed difference
    so it behaves identically in C with unsigned arithmetic.
    """
    a32 = np.asarray(a, dtype=np.uint32)
    b32 = np.asarray(b, dtype=np.uint32)
    t32 = np.asarray(t, dtype=np.uint32)
    lo = a32 * (255 - t32)
    hi = b32 * t32
    s = lo + hi + 128
    return ((s + (s >> 8)) >> 8).astype(np.uint8)


def blend_over(dst, src, mask=None):
    """Straight-alpha source-over: src composited onto dst.

    Both arrays are (..., 4) uint8 straight (non-premultiplied) RGBA. Returns
    a new array; neither input is modified.

    mask, if given, is a (...) uint8 coverage plane scaling the source alpha --
    this is how a selection and a brush falloff fuse into one multiply.

    Straight alpha is deliberate. Premultiplying at 8 bits is lossy in a way
    that breaks tools: a pixel at alpha=1 retains roughly one bit of colour,
    which wrecks the eyedropper, magic-wand comparison, bucket-fill tolerance
    and lossless PNG round-trip. Premultiplication belongs in a compositor's
    wide working buffers, not at rest.

    The colour terms are weighted by alpha and divided by the resulting alpha
    sum. Averaging straight RGB across differing alpha is the single most
    common defect in hand-rolled editors -- it produces dark fringes around
    every transparent edge.
    """
    dst = np.asarray(dst, dtype=np.uint8)
    src = np.asarray(src, dtype=np.uint8)

    sa = src[..., 3].astype(np.uint32)
    if mask is not None:
        sa = mul255(sa, np.asarray(mask, dtype=np.uint8)).astype(np.uint32)
    da = dst[..., 3].astype(np.uint32)

    # out_a = sa + da*(1-sa)
    da_keep = div255(da * (255 - sa)).astype(np.uint32)
    out_a = sa + da_keep

    out = np.empty(np.broadcast(dst, src).shape, dtype=np.uint8)
    out[..., 3] = out_a.astype(np.uint8)

    # Colour weighted by each contribution's alpha, divided by the total.
    # Where out_a == 0 there is no colour information at all; emit zero
    # rather than dividing, which also keeps fully-transparent pixels from
    # carrying stale colour into a later composite.
    nonzero = out_a > 0
    safe = np.where(nonzero, out_a, 1)
    for c in range(3):
        num = src[..., c].astype(np.uint32) * sa + dst[..., c].astype(np.uint32) * da_keep
        out[..., c] = np.where(nonzero, (num + safe // 2) // safe, 0).astype(np.uint8)
    return out


def apply_coverage(dst, src, mask):
    """dst*(255-mask)/255 + src*mask/255, per channel, exact.

    A straight coverage lerp with no alpha semantics -- used for scalar planes
    (height, index-as-scalar) where the values are data rather than colour.
    """
    dst = np.asarray(dst, dtype=np.uint8)
    src = np.asarray(src, dtype=np.uint8)
    m = np.asarray(mask, dtype=np.uint8)
    if dst.ndim > m.ndim:
        m = m[..., None]
    return lerp255(dst, src, m)


def expand_palette(entries, index, out=None):
    """palette[index] -> RGBA, via np.take.

    Always use np.take with out=, never entries[index]. The fancy-index form
    is roughly 8x slower (0.453 ms vs 0.058 ms on a 256x256 region) and
    allocates a fresh array every call, which matters because this runs on
    every write to an index-locked layer.
    """
    return np.take(entries, index, axis=0, out=out)
