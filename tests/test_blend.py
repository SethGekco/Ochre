#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Blend modes, checked against hand-computed values from the formulas.

Every mode is verified against an independent reference implementation over
an exhaustive operand sweep, not spot-checked. The three Paint.NET-specific
modes get extra attention because there is no public spec to fall back on if
they are wrong: Reflect, Glow (Reflect with swapped arguments) and Negation.

Run: python3 tests/test_blend.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import accel
from ochre.engine.blend import BlendRegistry
from ochre.engine.ini import IniDB


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def r255(x):
    return int(round(x))


# Independent references, written straight from the formula table in
# docs/MINING.md rather than from the implementation.
def ref(mode, a, b):
    if mode == "Normal":     return b
    if mode == "Multiply":   return r255(a * b / 255.0)
    if mode == "Additive":   return min(255, a + b)
    if mode == "Screen":     return a + b - r255(a * b / 255.0)
    if mode == "Lighten":    return max(a, b)
    if mode == "Darken":     return min(a, b)
    if mode == "Difference": return abs(b - a)
    if mode == "Negation":   return 255 - abs(255 - a - b)
    if mode == "Xor":        return a ^ b
    if mode == "ColorBurn":
        return 0 if b == 0 else max(0, 255 - min(255, ((255 - a) * 255) // b))
    if mode == "ColorDodge":
        return 255 if b == 255 else min(255, (a * 255) // (255 - b))
    if mode == "Reflect":
        return 255 if b == 255 else min(255, (a * a) // (255 - b))
    if mode == "Glow":
        return 255 if a == 255 else min(255, (b * b) // (255 - a))
    if mode == "Overlay":
        if a < 128:
            return r255(2 * a * b / 255.0)
        return 255 - r255(2 * (255 - a) * (255 - b) / 255.0)
    raise KeyError(mode)


def main():
    names = sorted(accel.OPCODES)
    check("14 blend modes registered", len(names) == 14, "-- got %d" % len(names))

    # ---- exhaustive per-mode sweep --------------------------------------
    a = np.arange(256, dtype=np.uint32)
    for name in names:
        opcode = accel.OPCODES[name]
        worst = None
        for b_val in range(0, 256, 1):
            b = np.full(256, b_val, dtype=np.uint32)
            got = np.asarray(accel.blend_channel(opcode, a, b)).astype(int)
            want = np.array([ref(name, int(av), b_val) for av in range(256)])
            bad = np.nonzero(got != want)[0]
            if bad.size:
                i = int(bad[0])
                worst = (i, b_val, int(got[i]), int(want[i]))
                break
        check("%s exact over all 65536 operand pairs" % name, worst is None,
              "-- a=%s b=%s got=%s want=%s" % worst if worst else "")

    # ---- the traps -------------------------------------------------------
    A, B = np.uint32(200), np.uint32(60)
    glow = int(accel.blend_channel(accel.GLOW, A, B))
    reflect_swapped = int(accel.blend_channel(accel.REFLECT, B, A))
    check("Glow is Reflect with swapped arguments", glow == reflect_swapped,
          "-- glow(200,60)=%d reflect(60,200)=%d" % (glow, reflect_swapped))

    x = int(accel.blend_channel(accel.XOR, np.uint32(0b1010), np.uint32(0b0110)))
    check("Xor is bitwise, not arithmetic", x == 0b1100, "-- got %d" % x)

    # Overlay keys on the LOWER layer; swapping operands must change it.
    lo = int(accel.blend_channel(accel.OVERLAY, np.uint32(100), np.uint32(200)))
    hi = int(accel.blend_channel(accel.OVERLAY, np.uint32(200), np.uint32(100)))
    check("Overlay keys on the lower layer", lo != hi,
          "-- overlay(100,200)=%d overlay(200,100)=%d" % (lo, hi))

    # Division-by-zero guards.
    check("ColorBurn handles b=0",
          int(accel.blend_channel(accel.COLORBURN, np.uint32(128), np.uint32(0))) == 0)
    check("ColorDodge handles b=255",
          int(accel.blend_channel(accel.COLORDODGE, np.uint32(128), np.uint32(255))) == 255)
    check("Reflect handles b=255",
          int(accel.blend_channel(accel.REFLECT, np.uint32(128), np.uint32(255))) == 255)
    check("Glow handles a=255",
          int(accel.blend_channel(accel.GLOW, np.uint32(255), np.uint32(128))) == 255)

    # Negation is symmetric; Difference is too.
    for mode in (accel.NEGATION, accel.DIFFERENCE):
        sym = all(int(accel.blend_channel(mode, np.uint32(p), np.uint32(q)))
                  == int(accel.blend_channel(mode, np.uint32(q), np.uint32(p)))
                  for p in range(0, 256, 17) for q in range(0, 256, 17))
        check("mode %d is symmetric" % mode, sym)

    # Unknown opcode is an error, not silent nonsense.
    try:
        accel.blend_channel(999, a, a)
        check("unknown opcode rejected", False, "-- no error")
    except ValueError:
        check("unknown opcode rejected", True)

    # ---- blend_rect alpha behaviour --------------------------------------
    def px(*v):
        return np.array([[list(v)]], dtype=np.uint8)

    red, blue = px(255, 0, 0, 255), px(0, 0, 255, 255)
    out = accel.blend_rect(red.copy(), blue, accel.NORMAL)
    check("normal over opaque replaces", tuple(out[0, 0]) == (0, 0, 255, 255))

    out = accel.blend_rect(red.copy(), px(0, 0, 255, 0), accel.NORMAL)
    check("transparent source leaves backdrop", tuple(out[0, 0]) == (255, 0, 0, 255))

    out = accel.blend_rect(px(0, 0, 0, 0), blue, accel.MULTIPLY)
    check("blending onto nothing shows the source",
          tuple(out[0, 0]) == (0, 0, 255, 255),
          "-- got %r" % (tuple(out[0, 0]),))

    out = accel.blend_rect(red.copy(), blue, accel.NORMAL, opacity=0)
    check("opacity 0 is a no-op", tuple(out[0, 0]) == (255, 0, 0, 255))
    out = accel.blend_rect(red.copy(), blue, accel.NORMAL, opacity=255)
    check("opacity 255 is full", tuple(out[0, 0]) == (0, 0, 255, 255))
    out = accel.blend_rect(red.copy(), blue, accel.NORMAL, opacity=128)
    check("opacity 128 lands between", 120 <= int(out[0, 0, 2]) <= 135)

    zero = np.zeros((1, 1), dtype=np.uint8)
    out = accel.blend_rect(red.copy(), blue, accel.NORMAL, mask=zero)
    check("zero mask blocks the source", tuple(out[0, 0]) == (255, 0, 0, 255))

    # Multiply with white is the identity; with black, black.
    white, black = px(255, 255, 255, 255), px(0, 0, 0, 255)
    out = accel.blend_rect(px(90, 120, 200, 255), white, accel.MULTIPLY)
    check("multiply by white is identity", tuple(out[0, 0]) == (90, 120, 200, 255))
    out = accel.blend_rect(px(90, 120, 200, 255), black, accel.MULTIPLY)
    check("multiply by black is black", tuple(out[0, 0])[:3] == (0, 0, 0))

    # The fringe test again, at the blend_rect level.
    ghost = px(0, 0, 0, 0)
    out = accel.blend_rect(white.copy(), ghost, accel.NORMAL)
    check("invisible black does not darken white",
          tuple(out[0, 0]) == (255, 255, 255, 255))

    try:
        accel.blend_rect(np.zeros((2, 2, 4), np.uint8), np.zeros((3, 3, 4), np.uint8))
        check("blend_rect rejects shape mismatch", False, "-- no error")
    except ValueError:
        check("blend_rect rejects shape mismatch", True)

    # ---- composite_stack -------------------------------------------------
    empty = accel.composite_stack([], (4, 4))
    check("empty stack is transparent, not black", int(empty.sum()) == 0)

    layers = [(np.broadcast_to(red, (4, 4, 4)).copy(), accel.NORMAL, 255, None),
              (np.broadcast_to(blue, (4, 4, 4)).copy(), accel.NORMAL, 255, None)]
    out = accel.composite_stack(layers, (4, 4))
    check("stack composites bottom-first", tuple(out[0, 0]) == (0, 0, 255, 255))

    # ---- registry --------------------------------------------------------
    reg = BlendRegistry()
    check("registry knows every opcode", len(reg) == 14)
    check("lookup by name", reg.opcode("Multiply") == accel.MULTIPLY)
    check("unknown name falls back to Normal", reg.opcode("Nonsense") == accel.NORMAL)
    check("membership", "Overlay" in reg and "Nope" not in reg)

    db = IniDB()
    db.load_string("[BlendMode:Multiply]\nLabel = Times\nOrder = 1\nGroup = G\n"
                   "[BlendMode:Xor]\nEnabled = off\n"
                   "[BlendMode:Invented]\nLabel = Nope\n")
    reg = BlendRegistry(db)
    check("INI relabels", reg.get("Multiply").label == "Times")
    check("INI reorders", reg.get("Multiply").order == 1)
    check("INI can hide a mode", "Xor" not in reg and len(reg) == 13)
    check("INI cannot invent a mode", reg.get("Invented") is None)
    check("hidden mode still has no opcode of its own",
          reg.opcode("Xor") == accel.NORMAL)

    real = IniDB(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data"))
    reg = BlendRegistry(real)
    check("shipped data/blendmodes.ini parses", len(reg) >= 13)
    groups = reg.groups()
    check("modes are grouped for a menu", len(groups) >= 3,
          "-- groups %r" % ([g for g, _ in groups],))
    check("Normal sorts first", reg.names()[0] == "Normal")

    print("\nall blend checks passed")


if __name__ == "__main__":
    main()
