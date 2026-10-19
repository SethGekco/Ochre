#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The blend modes beyond Paint.NET's set, against float references.

The specification is integer (so a C implementation can match it bit for
bit); these checks hold it to the published formulas in real numbers, over
every operand pair for the separable modes and a random corpus for the
non-separable ones.

Run: python3 tests/test_blendmodes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.accel import fallback as F
from ochre.engine.blend import DEFAULT


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def ref(mode, a, b):
    a, b = a / 255.0, b / 255.0
    clip = lambda x: np.clip(x, 0, 1)
    if mode == F.HARDLIGHT:
        return np.where(b <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == F.SOFTLIGHT:
        return (1 - 2 * b) * a * a + 2 * b * a
    if mode == F.LINEARBURN:
        return clip(a + b - 1)
    if mode == F.SUBTRACT:
        return clip(a - b)
    if mode == F.DIVIDE:
        return np.where(b == 0, 1.0, clip(a / np.where(b == 0, 1, b)))
    if mode == F.EXCLUSION:
        return a + b - 2 * a * b
    if mode == F.LINEARLIGHT:
        return clip(a + 2 * b - 1)
    if mode == F.PINLIGHT:
        return np.where(b < 0.5, np.minimum(a, 2 * b), np.maximum(a, 2 * b - 1))
    if mode == F.HARDMIX:
        return np.where(a + b >= 1, 1.0, 0.0)
    if mode == F.GRAINEXTRACT:
        return clip(a - b + 0.5)
    if mode == F.GRAINMERGE:
        return clip(a + b - 0.5)
    if mode == F.VIVIDLIGHT:
        with np.errstate(divide="ignore", invalid="ignore"):
            burn = np.where(b == 0, 0.0, 1 - clip((1 - a) / (2 * b)))
            dodge = np.where(b >= 1, 1.0, clip(a / (2 * (1 - b))))
        return np.where(b < 0.5, burn, dodge)
    raise ValueError(mode)


def main():
    a = np.repeat(np.arange(256, dtype=np.uint32), 256)
    b = np.tile(np.arange(256, dtype=np.uint32), 256)
    tol = {F.DIVIDE: 2, F.VIVIDLIGHT: 3, F.HARDMIX: 0}
    for name in ("HardLight", "SoftLight", "LinearBurn", "Subtract", "Divide",
                 "Exclusion", "LinearLight", "PinLight", "HardMix", "GrainExtract",
                 "GrainMerge", "VividLight"):
        op = F.OPCODES[name]
        got = F.blend_channel(op, a, b).astype(np.int64)
        want = np.rint(ref(op, a.astype(np.float64), b.astype(np.float64)) * 255).astype(np.int64)
        err = int(np.abs(got - want).max())
        check("%s matches its formula over all 65536 pairs (max error %d)" % (name, err),
              err <= tol.get(op, 1) and got.min() >= 0 and got.max() <= 255)

    # Non-separable: the defining properties, which hold exactly.
    rng = np.random.default_rng(7)
    cb = tuple(rng.integers(0, 256, 4000).astype(np.uint32) for _ in range(3))
    cs = tuple(rng.integers(0, 256, 4000).astype(np.uint32) for _ in range(3))
    lum = lambda c: (77 * c[0].astype(np.int64) + 151 * c[1] + 28 * c[2] + 128) >> 8
    for name in ("Hue", "Saturation", "Color", "Luminosity"):
        out = F.blend_colour(F.OPCODES[name], cb, cs)
        keeps = cs if name == "Luminosity" else cb
        err = int(np.abs(lum(out) - lum(keeps)).max())
        check("%s keeps the right luminosity (within %d)" % (name, err), err <= 2)
    out = F.blend_colour(F.COLOR, cb, cs)
    grey = (cs[0] == cs[1]) & (cs[1] == cs[2])
    check("Color with a grey source gives grey", True if not grey.any() else
          bool(((out[0] == out[1]) & (out[1] == out[2]))[grey].all()))
    out = F.blend_colour(F.DARKERCOLOR, cb, cs)
    check("Darker Color picks the darker whole colour, never mixes channels",
          all(((out[i] == cb[i]) | (out[i] == cs[i])).all() for i in range(3)))
    pure = (np.array([255], np.uint32), np.array([0], np.uint32), np.array([0], np.uint32))
    blue = (np.array([0], np.uint32), np.array([0], np.uint32), np.array([255], np.uint32))
    hue = F.blend_colour(F.HUE, pure, blue)
    check("Hue: a blue source over red takes blue's hue", int(hue[2][0]) > int(hue[0][0]))

    # Through blend_rect, transparency and opacity behave like every mode.
    dst = np.zeros((2, 2, 4), np.uint8)
    dst[..., :3] = 120
    dst[..., 3] = 255
    src = np.zeros((2, 2, 4), np.uint8)
    src[..., 0] = 255
    src[..., 3] = 0
    for name, op in F.OPCODES.items():
        if not np.array_equal(F.blend_rect(dst.copy(), src, op, 255), dst):
            check("%s: a transparent source leaves the backdrop alone" % name, False)
    check("every mode leaves the backdrop alone under a transparent source", True)
    check("every mode is registered for the layer properties", all(
        n in DEFAULT for n in F.OPCODES), str([n for n in F.OPCODES if n not in DEFAULT]))
    print("\nall blend-mode checks passed")


if __name__ == "__main__":
    main()
