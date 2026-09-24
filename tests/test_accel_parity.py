#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""The C accelerator must be BIT-IDENTICAL to the numpy specification.

This is the test that makes the accelerator safe to exist. It compares with
array_equal, never allclose -- "close" is not the contract. If the two ever
disagree by one in the last place on one pixel, the C is wrong and this
should say so loudly.

It skips cleanly when the extension is not built, so the suite stays green on
a machine with no compiler.

Run: python3 tests/test_accel_parity.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine import accel
from ochre.engine.accel import fallback as spec


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def main():
    if not accel.HAVE_NATIVE:
        reason = ("OCHRE_ACCEL=0" if os.environ.get("OCHRE_ACCEL", "").strip()
                  in ("0", "off", "false", "no") else "extension not built")
        print("ok: skipped (%s)" % reason)
        return

    native = accel.native
    print("    backend: %s  (%s)" % (accel.BACKEND, accel.describe()))

    # ---- blend_channel, exhaustively ------------------------------------
    a = np.arange(256, dtype=np.uint32)
    worst = None
    for name, opcode in sorted(spec.OPCODES.items()):
        for b_val in range(256):
            b = np.full(256, b_val, dtype=np.uint32)
            want = np.asarray(spec.blend_channel(opcode, a, b)).astype(np.int64)
            got = np.asarray(native.blend_channel(opcode, a, b)).astype(np.int64)
            if not np.array_equal(got, want):
                bad = int(np.nonzero(got != want)[0][0])
                worst = (name, bad, b_val, int(got[bad]), int(want[bad]))
                break
        if worst:
            break
    check("blend_channel matches over ALL 14 modes x 65536 operand pairs",
          worst is None,
          "-- %s(a=%d, b=%d): C gave %d, spec says %d" % worst if worst else "")

    try:
        native.blend_channel(999, a, a)
        check("C rejects an unknown opcode", False, "-- no error")
    except ValueError:
        check("C rejects an unknown opcode", True)

    # ---- blend_rect: random corpus across every mode and opacity --------
    rng = np.random.default_rng(20260923)
    mismatches = []
    for opcode in sorted(spec.OPCODES.values()):
        for trial in range(6):
            h, w = int(rng.integers(1, 9)), int(rng.integers(1, 9))
            dst = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
            src = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
            opacity = int(rng.integers(0, 256))
            mask = (rng.integers(0, 256, (h, w), dtype=np.uint8)
                    if trial % 2 else None)
            want = spec.blend_rect(dst.copy(), src, opcode, opacity, mask)
            got = native.blend_rect(dst.copy(), src, opcode, opacity, mask)
            if not np.array_equal(got, want):
                diff = int(np.abs(got.astype(int) - want.astype(int)).max())
                mismatches.append((opcode, trial, diff))
    check("blend_rect matches across every mode, opacity and mask",
          not mismatches, "-- %r" % mismatches[:4])

    # ---- the edge cases that break naive implementations ----------------
    def px(*v):
        return np.array([[list(v)]], dtype=np.uint8)

    cases = {
        "fully transparent over fully transparent": (px(0, 0, 0, 0), px(0, 0, 0, 0)),
        "opaque over transparent": (px(0, 0, 0, 0), px(9, 8, 7, 255)),
        "transparent over opaque": (px(9, 8, 7, 255), px(1, 2, 3, 0)),
        "alpha 1 (the lossy-premultiply case)": (px(255, 255, 255, 1), px(0, 0, 0, 1)),
        "alpha 254": (px(1, 2, 3, 254), px(254, 253, 252, 254)),
        "both fully opaque": (px(10, 20, 30, 255), px(200, 100, 50, 255)),
        "channel extremes": (px(0, 255, 0, 255), px(255, 0, 255, 255)),
    }
    bad = []
    for label, (d, s) in cases.items():
        for opcode in sorted(spec.OPCODES.values()):
            for opacity in (0, 1, 127, 128, 254, 255):
                want = spec.blend_rect(d.copy(), s, opcode, opacity)
                got = native.blend_rect(d.copy(), s, opcode, opacity)
                if not np.array_equal(got, want):
                    bad.append((label, opcode, opacity))
    check("blend_rect matches on every alpha and channel edge case",
          not bad, "-- %r" % bad[:4])

    # ---- out= writes in place, exactly like the spec --------------------
    dst = rng.integers(0, 256, (6, 5, 4), dtype=np.uint8)
    src = rng.integers(0, 256, (6, 5, 4), dtype=np.uint8)
    want_buf = np.zeros_like(dst)
    got_buf = np.zeros_like(dst)
    want = spec.blend_rect(dst, src, spec.MULTIPLY, 200, None, out=want_buf)
    got = native.blend_rect(dst, src, spec.MULTIPLY, 200, None, out=got_buf)
    check("out= returns the same buffer it was given",
          want is want_buf and got is got_buf)
    check("out= produces identical contents", np.array_equal(got_buf, want_buf))

    # Aliasing dst and out is how the compositor actually calls this.
    a1 = dst.copy()
    a2 = dst.copy()
    spec.blend_rect(a1, src, spec.SCREEN, 255, None, out=a1)
    native.blend_rect(a2, src, spec.SCREEN, 255, None, out=a2)
    check("in-place blending (out is dst) matches", np.array_equal(a1, a2),
          "-- this is how the compositor calls it")

    # ---- non-contiguous input -------------------------------------------
    big = rng.integers(0, 256, (12, 12, 4), dtype=np.uint8)
    view_d = big[2:10, 3:11]
    view_s = big[0:8, 0:8]
    check("non-contiguous views are handled",
          np.array_equal(spec.blend_rect(view_d.copy(), view_s, spec.OVERLAY),
                         native.blend_rect(view_d.copy(), view_s, spec.OVERLAY)))

    # ---- odd sizes, which defeat SIMD tail handling ---------------------
    odd = []
    for h, w in ((1, 1), (1, 7), (7, 1), (3, 5), (17, 13), (1, 255)):
        d = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
        s = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
        if not np.array_equal(spec.blend_rect(d.copy(), s, spec.OVERLAY, 137),
                              native.blend_rect(d.copy(), s, spec.OVERLAY, 137)):
            odd.append((h, w))
    check("odd and prime dimensions match", not odd, "-- %r" % odd)

    # ---- input validation parity ----------------------------------------
    try:
        native.blend_rect(np.zeros((2, 2, 4), np.uint8), np.zeros((3, 3, 4), np.uint8))
        check("C rejects mismatched shapes", False, "-- no error")
    except ValueError:
        check("C rejects mismatched shapes", True)
    try:
        native.blend_rect(np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8))
        check("C rejects non-RGBA input", False, "-- no error")
    except ValueError:
        check("C rejects non-RGBA input", True)
    try:
        native.blend_rect(np.zeros((2, 2, 4), np.uint8),
                          np.zeros((2, 2, 4), np.uint8),
                          spec.NORMAL, 255, np.zeros((5, 5), np.uint8))
        check("C rejects a mask of the wrong size", False, "-- no error")
    except ValueError:
        check("C rejects a mask of the wrong size", True)

    # ---- the selector reports honestly ----------------------------------
    check("selector routes blend_rect to C", accel.BACKENDS["blend_rect"] == "c")
    check("selector admits what is still numpy",
          accel.BACKENDS["flood_fill"] == "numpy",
          "-- an unimplemented op must fall back, not silently vanish")
    check("describe() names the accelerated ops",
          "blend_rect" in accel.describe())

    # ---- and it should actually be faster -------------------------------
    d = rng.integers(0, 256, (512, 512, 4), dtype=np.uint8)
    s = rng.integers(0, 256, (512, 512, 4), dtype=np.uint8)

    def timeit(fn, n=3):
        best = 1e9
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best * 1000

    t_spec = timeit(lambda: spec.blend_rect(d.copy(), s, spec.OVERLAY, 200))
    t_c = timeit(lambda: native.blend_rect(d.copy(), s, spec.OVERLAY, 200))
    check("the accelerator is actually faster", t_c < t_spec,
          "-- C %.1f ms vs numpy %.1f ms" % (t_c, t_spec))
    print("    (512x512 Overlay: numpy %.1f ms -> C %.1f ms, %.1fx)"
          % (t_spec, t_c, t_spec / max(t_c, 1e-9)))

    t_spec = timeit(lambda: spec.blend_rect(d.copy(), s, spec.NORMAL, 255))
    t_c = timeit(lambda: native.blend_rect(d.copy(), s, spec.NORMAL, 255))
    print("    (512x512 Normal:  numpy %.1f ms -> C %.1f ms, %.1fx)"
          % (t_spec, t_c, t_spec / max(t_c, 1e-9)))

    print("\nall accelerator parity checks passed")


if __name__ == "__main__":
    main()
