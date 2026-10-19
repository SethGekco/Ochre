# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The pure-numpy backend. This file is the normative specification.

Every operation Ochre may accelerate in C is defined here first, in integer
arithmetic, and the C extension's only job is to produce bit-identical output.
When the two disagree, the C is wrong. That is not a courtesy to this file --
it is what makes the parity test meaningful, because a floating-point
specification could not be matched bit-for-bit by an independent
implementation at all.

This backend is not a degraded mode. It is fast enough to edit with: the
below/above compositor cache keeps a 512x512 dirty rect on a ten-layer
document inside a 60 Hz frame budget with no compiled code present. The C
extension is an optimisation, not a requirement, and the test suite runs
twice -- once with each backend -- so this path can never quietly rot.

Blend formulas are Paint.NET's, taken from OpenPDN's UserBlendOps
(MIT-licensed). See docs/MINING.md. A is the LOWER layer, B the UPPER.
Three of them are Paint.NET-specific and appear in no public blend spec:
Reflect, Glow (which is Reflect with its arguments swapped, not a formula of
its own) and Negation.
"""

import numpy as np

BACKEND = "numpy"

# Opcode ids. These are part of the C ABI once the extension exists, so they
# are append-only: never renumber, never reuse a retired id.
(NORMAL, MULTIPLY, ADDITIVE, COLORBURN, COLORDODGE, REFLECT, GLOW, OVERLAY,
 DIFFERENCE, NEGATION, LIGHTEN, DARKEN, SCREEN, XOR) = range(14)

# Photoshop's and GIMP's modes beyond Paint.NET's set, appended (the ids are
# ABI). The C extension implements ids below NATIVE_OPCODES; the selector in
# __init__ routes the rest here, so the extension can catch up one mode at a
# time without a flag day. Formulas: the W3C Compositing spec (separable and
# non-separable), GIMP's for grain extract/merge, Pegtop's soft light (GIMP's,
# continuous where Photoshop's original has a kink) -- all in integers.
(SOFTLIGHT, HARDLIGHT, VIVIDLIGHT, LINEARLIGHT, PINLIGHT, HARDMIX, LINEARBURN,
 SUBTRACT, DIVIDE, EXCLUSION, GRAINEXTRACT, GRAINMERGE, DARKERCOLOR,
 LIGHTERCOLOR, HUE, SATURATION, COLOR, LUMINOSITY) = range(14, 32)
NATIVE_OPCODES = 14
# Modes that need all three channels at once (they work on hue, saturation
# and luminosity, or compare whole colours) rather than channel by channel.
NONSEPARABLE = frozenset((DARKERCOLOR, LIGHTERCOLOR, HUE, SATURATION, COLOR, LUMINOSITY))

OPCODES = {
    "Normal": NORMAL, "Multiply": MULTIPLY, "Additive": ADDITIVE,
    "ColorBurn": COLORBURN, "ColorDodge": COLORDODGE, "Reflect": REFLECT,
    "Glow": GLOW, "Overlay": OVERLAY, "Difference": DIFFERENCE,
    "Negation": NEGATION, "Lighten": LIGHTEN, "Darken": DARKEN,
    "Screen": SCREEN, "Xor": XOR,
    "SoftLight": SOFTLIGHT, "HardLight": HARDLIGHT, "VividLight": VIVIDLIGHT,
    "LinearLight": LINEARLIGHT, "PinLight": PINLIGHT, "HardMix": HARDMIX,
    "LinearBurn": LINEARBURN, "Subtract": SUBTRACT, "Divide": DIVIDE,
    "Exclusion": EXCLUSION, "GrainExtract": GRAINEXTRACT, "GrainMerge": GRAINMERGE,
    "DarkerColor": DARKERCOLOR, "LighterColor": LIGHTERCOLOR, "Hue": HUE,
    "Saturation": SATURATION, "Color": COLOR, "Luminosity": LUMINOSITY,
}


def _mul(a, b):
    """round(a*b/255) on uint32 arrays. The primitive everything rests on."""
    t = a * b + 128
    return (t + (t >> 8)) >> 8


def _safe_div(num, den):
    """num // den with den == 0 yielding 0. Branchless, no warnings."""
    safe = np.where(den == 0, 1, den)
    return np.where(den == 0, 0, num // safe)


def blend_channel(mode, a, b):
    """B(Cb, Cs) for one channel. a and b are uint32 arrays in 0..255.

    Returns uint32 in 0..255. Both operands are treated as opaque here;
    alpha is applied by the caller.
    """
    if mode == NORMAL:
        return b
    if mode == MULTIPLY:
        return _mul(a, b)
    if mode == ADDITIVE:
        return np.minimum(255, a + b)
    if mode == SCREEN:
        return a + b - _mul(a, b)
    if mode == LIGHTEN:
        return np.maximum(a, b)
    if mode == DARKEN:
        return np.minimum(a, b)
    if mode == DIFFERENCE:
        # Unsigned operands: compute as max-min rather than abs(b-a).
        return np.maximum(a, b) - np.minimum(a, b)
    if mode == NEGATION:
        # 255 - |255 - a - b| simplifies, for s = a + b in 0..510, to
        #     s        when s <= 255
        #     510 - s  when s >  255
        # which is the same function with no absolute value and, crucially,
        # no subtraction that could underflow. np.where evaluates BOTH
        # branches, so a form that merely happens to pick the safe one still
        # computes an unsigned underflow in the other.
        s = a + b
        return np.where(s > 255, 510 - s, s)
    if mode == XOR:
        # Genuinely bitwise, not an arithmetic difference.
        return (a ^ b) & 0xFF
    if mode == COLORBURN:
        return np.where(b == 0, 0,
                        255 - np.minimum(255, _safe_div((255 - a) * 255, b)))
    if mode == COLORDODGE:
        return np.where(b == 255, 255,
                        np.minimum(255, _safe_div(a * 255, 255 - b)))
    if mode == REFLECT:
        return np.where(b == 255, 255,
                        np.minimum(255, _safe_div(a * a, 255 - b)))
    if mode == GLOW:
        # Reflect with the arguments swapped. Not its own formula.
        return np.where(a == 255, 255,
                        np.minimum(255, _safe_div(b * b, 255 - a)))
    if mode == OVERLAY:
        # Keys on A, the LOWER layer. Several editors key on the upper one
        # and produce different output.
        lo = _mul(2 * a, b)
        hi = 255 - _mul(2 * (255 - a), 255 - b)
        return np.where(a < 128, lo, hi)
    if mode == HARDLIGHT:
        # Overlay with the layers' roles swapped: keys on the UPPER layer.
        lo = _mul(2 * b, a)
        hi = 255 - _mul(2 * (255 - b), 255 - a)
        return np.where(b < 128, lo, hi)
    if mode == SOFTLIGHT:
        # Pegtop: a^2 + 2b(a - a^2). a - a^2/255 is never negative.
        a2 = _mul(a, a)
        return np.minimum(255, a2 + _mul(2 * b, a - a2))
    if mode == LINEARBURN:
        s = a + b
        return np.where(s > 255, s - 255, 0)
    if mode == SUBTRACT:
        return np.where(a > b, a - b, 0)
    if mode == DIVIDE:
        return np.where(b == 0, 255, np.minimum(255, _safe_div(a * 255, b)))
    if mode == EXCLUSION:
        return a + b - 2 * _mul(a, b)
    if mode == VIVIDLIGHT:
        # Colour burn by 2b below the middle, colour dodge by 2(b-128) above.
        b2 = 2 * b
        burn = np.where(b2 == 0, 0, 255 - np.minimum(255, _safe_div((255 - a) * 255, b2)))
        # 2b - 1 in unit terms is 2b - 255 here (not 256): 255 is "one".
        d = np.where(b >= 128, 2 * b - 255, 0)
        dodge = np.where(d >= 255, 255, np.minimum(255, _safe_div(a * 255, 255 - d)))
        return np.where(b < 128, burn, dodge)
    if mode == LINEARLIGHT:
        s = a + 2 * b
        return np.where(s > 255, np.minimum(255, s - 255), 0)
    if mode == PINLIGHT:
        return np.where(b < 128, np.minimum(a, 2 * b),
                        np.maximum(a, np.where(b >= 128, 2 * b - 255, 0)))
    if mode == HARDMIX:
        return np.where(a + b >= 255, 255, 0)
    if mode == GRAINEXTRACT:
        s = a + 128
        return np.where(s > b, np.minimum(255, s - b), 0)
    if mode == GRAINMERGE:
        s = a + b
        return np.where(s > 128, np.minimum(255, s - 128), 0)
    raise ValueError("unknown blend mode id %r" % (mode,))


# ---- non-separable modes, after the W3C spec, in integers -------------------

def _lum(r, g, b):
    # 0.30 / 0.59 / 0.11, scaled by 256: the spec's luma weights.
    return (77 * r + 151 * g + 28 * b + 128) >> 8


def _clip_colour(r, g, b):
    lum = _lum(r, g, b)
    n = np.minimum(np.minimum(r, g), b)
    x = np.maximum(np.maximum(r, g), b)
    out = []
    for c in (r, g, b):
        lo_den = lum - n
        c1 = np.where((n < 0) & (lo_den > 0),
                      lum + _sdiv((c - lum) * lum, lo_den), c)
        hi_den = x - lum
        c2 = np.where((x > 255) & (hi_den > 0),
                      lum + _sdiv((c1 - lum) * (255 - lum), hi_den), c1)
        out.append(np.clip(c2, 0, 255))
    return out


def _sdiv(num, den):
    """Signed integer division rounding toward zero; den > 0 where used."""
    safe = np.where(den == 0, 1, den)
    q = np.abs(num) // np.abs(safe)
    return np.where((num < 0) != (safe < 0), -q, q)


def _set_lum(r, g, b, lum):
    d = lum - _lum(r, g, b)
    return _clip_colour(r + d, g + d, b + d)


def _sat(r, g, b):
    return np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)


def _set_sat(r, g, b, sat):
    # (c - min) * sat / (max - min): the max channel becomes sat, the min 0,
    # the middle scales between -- ties included, with no sorting.
    mn = np.minimum(np.minimum(r, g), b)
    span = np.maximum(np.maximum(r, g), b) - mn
    return [np.where(span > 0, _sdiv((c - mn) * sat, span), 0) for c in (r, g, b)]


def blend_colour(mode, a, b):
    """B(Cb, Cs) for the non-separable modes. a, b: (r, g, b) uint32 tuples.

    Returns three uint32 arrays in 0..255.
    """
    ar, ag, ab_ = (x.astype(np.int64) for x in a)
    br, bg, bb = (x.astype(np.int64) for x in b)
    if mode in (DARKERCOLOR, LIGHTERCOLOR):
        la, lb = _lum(ar, ag, ab_), _lum(br, bg, bb)
        take_b = lb < la if mode == DARKERCOLOR else lb > la
        out = [np.where(take_b, y, x) for x, y in ((ar, br), (ag, bg), (ab_, bb))]
    elif mode == HUE:
        out = _set_lum(*_set_sat(br, bg, bb, _sat(ar, ag, ab_)), _lum(ar, ag, ab_))
    elif mode == SATURATION:
        out = _set_lum(*_set_sat(ar, ag, ab_, _sat(br, bg, bb)), _lum(ar, ag, ab_))
    elif mode == COLOR:
        out = _set_lum(br, bg, bb, _lum(ar, ag, ab_))
    elif mode == LUMINOSITY:
        out = _set_lum(ar, ag, ab_, _lum(br, bg, bb))
    else:
        raise ValueError("not a non-separable mode: %r" % (mode,))
    return [np.clip(c, 0, 255).astype(np.uint32) for c in out]


def blend_rect(dst, src, mode=NORMAL, opacity=255, mask=None, out=None):
    """Composite src over dst with a blend mode. Straight-alpha RGBA uint8.

    Implements the standard separable blend composite:

        Cs' = (1 - ab)*Cs + ab*B(Cb, Cs)        blended source colour
        ao  = as + ab*(1 - as)                  Porter-Duff over
        Co  = (as*Cs' + ab*Cb*(1 - as)) / ao    alpha-weighted, then divided

    The final division by ao is the part that matters and the part most
    hand-rolled editors omit. Averaging straight RGB across differing alpha
    produces dark fringes at every transparent edge; weighting by alpha and
    dividing by the resulting alpha does not.

    opacity scales the source alpha (0..255), as does mask, a (h,w) coverage
    plane. Fusing both into the source alpha is what makes selection clipping
    free -- there is no separate clipping path to keep in sync.
    """
    dst = np.asarray(dst, dtype=np.uint8)
    src = np.asarray(src, dtype=np.uint8)
    if dst.shape != src.shape:
        raise ValueError("blend_rect shape mismatch: %s vs %s"
                         % (dst.shape, src.shape))

    ab = dst[..., 3].astype(np.uint32)
    a_s = src[..., 3].astype(np.uint32)
    if opacity != 255:
        a_s = _mul(a_s, np.uint32(opacity))
    if mask is not None:
        a_s = _mul(a_s, np.asarray(mask, dtype=np.uint8).astype(np.uint32))

    if out is None:
        out = np.empty(dst.shape, dtype=np.uint8)

    # Fast path: an opaque source in Normal mode simply replaces the
    # backdrop. This is by far the most common composite in an editor -- a
    # fully-painted layer at full opacity -- and the general path would spend
    # three channel blends and a division arriving at exactly `src`.
    if mode == NORMAL and opacity == 255 and mask is None and a_s.min() == 255:
        out[...] = src
        return out

    inv_as = 255 - a_s
    ab_keep = _mul(ab, inv_as)
    ao = a_s + ab_keep

    out[..., 3] = ao.astype(np.uint8)

    nonzero = ao > 0
    safe = np.where(nonzero, ao, 1)

    whole = None
    if mode in NONSEPARABLE:
        whole = blend_colour(mode, tuple(dst[..., c].astype(np.uint32) for c in range(3)),
                             tuple(src[..., c].astype(np.uint32) for c in range(3)))
    for c in range(3):
        cb = dst[..., c].astype(np.uint32)
        cs = src[..., c].astype(np.uint32)
        if whole is not None:
            cs_eff = _mul(255 - ab, cs) + _mul(ab, whole[c])
        elif mode == NORMAL:
            # Cs' = (1-ab)*Cs + ab*B(Cb,Cs), and B is the identity here, so
            # Cs' == Cs exactly. Computing the two scaled halves and adding
            # them back would be slower AND would introduce rounding error
            # that the closed form does not have.
            cs_eff = cs
        else:
            blended = blend_channel(mode, cb, cs)
            # Where the backdrop is transparent the blend has nothing to act
            # on, so the source shows through unmodified.
            cs_eff = _mul(255 - ab, cs) + _mul(ab, blended)
        num = a_s * cs_eff + ab_keep * cb
        out[..., c] = np.where(nonzero, (num + safe // 2) // safe, 0).astype(np.uint8)
    return out


def composite_stack(layers, shape, out=None):
    """Flatten an ordered list of (rgba, mode, opacity, mask) bottom-first.

    Returns straight-alpha RGBA. An empty list yields fully transparent,
    which is the correct identity: compositing nothing onto nothing is
    nothing, not black.
    """
    h, w = shape[0], shape[1]
    if out is None:
        out = np.zeros((h, w, 4), dtype=np.uint8)
    else:
        out[...] = 0
    for rgba, mode, opacity, mask in layers:
        blend_rect(out, rgba, mode, opacity, mask, out=out)
    return out


def flood_fill(plane, x, y, tolerance, connectivity=4, mask=None):
    """Scanline flood fill returning a bool stencil.

    Queue holds contiguous runs, not pixels -- a per-pixel BFS on a 16 MP
    region is unusable in Python, and the run-based form does the same work
    in a fraction of the iterations.

    The tolerance metric follows Paint.NET: the RGB terms are scaled by the
    reference pixel's alpha, so transparent regions match loosely and opaque
    ones strictly, while the alpha difference is unweighted and therefore
    dominates. That is not Euclidean RGB distance, and the difference matters
    on anything with soft edges.

    mask, if given, is a bool array of pixels the fill may not enter. Passing
    the complement of a selection turns a per-pixel containment test into a
    free early-out, because those pixels simply look already-visited.
    """
    arr = np.asarray(plane)
    h, w = arr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), dtype=bool)

    if arr.ndim == 2:
        ref = arr[y, x].astype(np.int32)
        diff = arr.astype(np.int32) - ref
        similar = (diff * diff) <= (tolerance * tolerance)
    else:
        ref = arr[y, x].astype(np.int32)
        a_ref = ref[3] if arr.shape[2] == 4 else 255
        total = np.zeros((h, w), dtype=np.int64)
        for c in range(3):
            d = arr[..., c].astype(np.int32) - ref[c]
            total += (1 + d * d) * a_ref // 256
        if arr.shape[2] == 4:
            d = arr[..., 3].astype(np.int32) - a_ref
            total += d * d
        similar = total <= (tolerance * tolerance * 4)

    if mask is not None:
        similar = similar & ~np.asarray(mask, dtype=bool)

    out = np.zeros((h, w), dtype=bool)
    stack = [(x, x, y)]
    while stack:
        x0, x1, row = stack.pop()
        if row < 0 or row >= h:
            continue
        line = similar[row]
        done = out[row]
        # Walk left and right from the seed span.
        left = x0
        while left > 0 and line[left - 1] and not done[left - 1]:
            left -= 1
        right = x1
        while right < w - 1 and line[right + 1] and not done[right + 1]:
            right += 1
        if not line[x0] and left == x0 and right == x1:
            continue
        out[row, left:right + 1] = line[left:right + 1]
        # Seed the rows above and below, one entry per contiguous run.
        for nrow in (row - 1, row + 1):
            if nrow < 0 or nrow >= h:
                continue
            nline = similar[nrow] & ~out[nrow]
            run = None
            for cx in range(left, right + 1):
                if nline[cx]:
                    if run is None:
                        run = cx
                elif run is not None:
                    stack.append((run, cx - 1, nrow))
                    run = None
            if run is not None:
                stack.append((run, right, nrow))
    return out
