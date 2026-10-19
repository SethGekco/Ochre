#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Filters: each held to a property that defines it, plus the invariants
every one must keep (shape, determinism, no dark fringes).

Run: python3 tests/test_filters.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ochre.engine.effects import EffectRegistry
from ochre.engine.filters import FILTERS
from ochre.engine.ini import IniDB

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REG = EffectRegistry(IniDB(DATA))


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def run(name, img, **params):
    return REG.create(name, **params).apply(img)


def canvas(h=64, w=64, rgb=(128, 128, 128)):
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = 255
    return out


def main():
    rng = np.random.default_rng(1)
    noisy = rng.integers(0, 256, (48, 48, 4), dtype=np.uint8)
    noisy[..., 3] = 255

    # ---- every filter -----------------------------------------------------------
    bad = []
    for cls in FILTERS:
        a = cls().apply(noisy)
        b = cls().apply(noisy)
        if a.shape != noisy.shape or a.dtype != np.uint8 or not np.array_equal(a, b):
            bad.append(cls.name)
    check("all %d filters keep the shape and are deterministic" % len(FILTERS), not bad, str(bad))
    check("every filter is registered", all(c.name in REG for c in FILTERS))

    # ---- blurs ----------------------------------------------------------------------
    edges = canvas()
    edges[:, 32:, :3] = 255          # a vertical edge
    edges[32:, :, :3] = 0            # and a horizontal one below
    mb = run("motion_blur", edges, angle=0.0, distance=12)
    check("a horizontal motion blur softens a vertical edge...",
          0 < int(mb[10, 32, 0]) < 255 and 128 < int(mb[10, 30, 0]) < 255)
    check("...but not a horizontal one", mb[30, 10, 0] == 128 and mb[34, 10, 0] == 0)
    soft = canvas(rgb=(255, 0, 0))
    soft[:, :32, 3] = 0
    for name in ("box_blur", "gaussian_blur", "lens_blur", "motion_blur", "zoom_blur"):
        out = run(name, soft)
        edge = out[20, 26:38]
        dark = (edge[:, 3] > 0) & (edge[:, 0] < 250)
        if dark.any():
            check("%s keeps colour at a transparent edge (no dark fringe)" % name, False, str(edge))
    check("blurs keep colour at transparent edges (premultiplied work)", True)
    dot = np.zeros((41, 41, 4), np.uint8)
    dot[..., 3] = 255
    dot[20, 20, :3] = 255
    bokeh = run("lens_blur", dot, radius=8)
    ring = bokeh[20, 20 - 6:20 + 7, 0].astype(int)
    check("lens blur spreads a point into a flat disc (bokeh), not a peak",
          ring.max() - ring.min() <= 2 and ring.min() > 0)
    px = run("pixelate", noisy, cell=8)
    check("pixelate makes uniform cells", len(np.unique(px[0:8, 0:8, 0])) == 1
          and not np.array_equal(px[0:8, 0:8], px[0:8, 8:16]))
    st = run("surface_blur", edges, radius=4, threshold=10)
    check("surface blur keeps strong edges sharp", st[10, 31, 0] == 128 and st[10, 32, 0] == 255)

    # ---- photo -------------------------------------------------------------------------
    us = run("unsharp_mask", edges, radius=2.0, amount=100)
    check("unsharp mask overshoots at an edge (more local contrast)",
          us[10, 30, 0] < 128 and us[10, 33, 0] == 255)
    flat_noise = canvas()
    flat_noise[..., :3] += rng.integers(0, 4, (64, 64, 1), dtype=np.uint8)
    us2 = run("unsharp_mask", flat_noise, radius=2.0, amount=200, threshold=10)
    check("...and its threshold leaves fine noise alone", np.array_equal(us2, flat_noise))
    check("high pass of a flat image is mid-grey", np.all(run("high_pass", canvas(rgb=(50, 90, 200)))[..., :3] == 128))
    vg = run("vignette", canvas(rgb=(200, 200, 200)))
    check("vignette darkens corners, not the centre", vg[0, 0, 0] < 120 and vg[32, 32, 0] == 200)
    sh = run("shadows_highlights", canvas(rgb=(40, 40, 40)), shadows=80)
    check("shadows/highlights lifts a dark image", sh[5, 5, 0] > 55)

    # ---- noise ----------------------------------------------------------------------------
    salt = canvas()
    salt[10, 10, :3] = 255
    salt[30, 40, :3] = 0
    check("median removes salt and pepper", np.array_equal(run("median", salt, radius=1)[..., :3],
                                                         canvas()[..., :3]))
    ds = run("dust_scratches", np.where(np.arange(64)[None, :, None] < 32, salt, edges), radius=1,
             threshold=30)
    check("dust & scratches removes a speck", ds[10, 10, 0] == 128)
    sp = run("spread", noisy, amount=3)
    check("spread only moves pixels (same colours, rearranged)",
          set(map(tuple, sp.reshape(-1, 4)[:200])) <= set(map(tuple, noisy.reshape(-1, 4))))

    # ---- distort --------------------------------------------------------------------------
    for name, params in (("twirl", {"angle": 0.0}), ("bulge", {"amount": 0}),
                         ("ripple", {"amplitude": 0}), ("waves", {"amplitude": 0}),
                         ("lens_distortion", {"main": 0, "edge": 0, "zoom": 0})):
        out = run(name, noisy, **params)
        check("%s at zero strength leaves the image unchanged" % name,
              np.abs(out.astype(int) - noisy.astype(int)).max() <= 1)
    tw = run("twirl", noisy, angle=180.0)
    check("twirl leaves the corners (outside its radius) alone",
          np.array_equal(tw[0, 0], noisy[0, 0]) and not np.array_equal(tw[24, 20], noisy[24, 20]))
    # On a smooth image (a ramp, whose ends do not match) -- on noise,
    # neighbouring columns are unrelated and nothing could make them meet.
    ramp_img = canvas(48, 48)
    ramp_img[..., :3] = np.linspace(0, 255, 48).astype(np.uint8)[None, :, None]
    ts = run("tile_seamless", ramp_img)
    jump_before = np.abs(ramp_img[:, 0, :3].astype(int) - ramp_img[:, -1, :3].astype(int)).mean()
    jump_after = np.abs(ts[:, 0, :3].astype(int) - ts[:, -1, :3].astype(int)).mean()
    check("tile seamless makes the left and right edges meet", jump_after < jump_before / 2,
          "(%.1f -> %.1f)" % (jump_before, jump_after))

    # ---- stylize ----------------------------------------------------------------------------
    check("emboss of a flat image is flat grey", np.all(run("emboss", canvas(rgb=(10, 200, 60)))[..., :3] == 128))
    fe = run("edge_detect", edges)
    check("find edges: white where flat, dark on the edge", fe[10, 10, 0] == 255 and fe[10, 32, 0] < 200)
    so = run("solarize", canvas(rgb=(200, 50, 50)), threshold=128)
    check("solarize inverts only what is above its threshold", tuple(so[0, 0, :3]) == (55, 50, 50))
    check("solarize is pointwise, so it can be an adjustment layer", REG._classes["solarize"].pointwise)
    mx = run("maximum_minimum", salt, mode="maximum", radius=2)
    check("maximum grows a light speck into a square", np.all(mx[8:13, 8:13, 0] == 255))

    # ---- artistic ------------------------------------------------------------------------------
    step = canvas()
    step[:, 32:, :3] = 220
    step[..., :3] = np.clip(step[..., :3].astype(int) + rng.integers(-20, 20, (64, 64, 1)), 0, 255)
    oil = run("oil_painting", step, brush=4)
    check("oil painting flattens texture...",
          oil[10:20, 5:20, 0].std() < step[10:20, 5:20, 0].std() / 2)
    check("...but keeps the edge between areas",
          oil[15, 29, 0] < 160 and oil[15, 35, 0] > 180)
    cr = run("crystallize", noisy, cell=8)
    check("crystallize makes far fewer distinct colours (one per cell)",
          len(np.unique(cr.reshape(-1, 4), axis=0)) < 80)
    ps = run("pencil_sketch", canvas(rgb=(90, 90, 90)))
    check("pencil sketch of a flat image is near white paper", ps[10, 10, 0] > 240)

    # ---- render --------------------------------------------------------------------------------
    c1 = run("clouds", canvas(), seed=1)
    c2 = run("clouds", canvas(), seed=2)
    check("clouds differ by seed and span the colours", not np.array_equal(c1, c2)
          and c1[..., 0].min() < 30 and c1[..., 0].max() > 225)
    mand = run("mandelbrot", canvas(128, 128), center_re=-0.5, center_im=0.0, zoom=1.0)
    check("the Mandelbrot set is black inside", tuple(mand[64, 64, :3]) == (0, 0, 0))
    cb = run("checkerboard", canvas(), size=8)
    check("checkerboard alternates every 8 px", cb[0, 0, 0] == 0 and cb[0, 8, 0] == 255 and cb[8, 8, 0] == 0)

    # ---- generic ---------------------------------------------------------------------------------
    check("the identity convolution changes nothing", np.array_equal(run("convolution", noisy), noisy))
    shift = run("convolution", noisy, k11=0, k12=1)
    check("a custom kernel does what it says (shift left)", np.array_equal(shift[:, :-1, :3], noisy[:, 1:, :3]))
    shape = np.zeros((40, 40, 4), np.uint8)
    shape[10:20, 10:20] = (255, 0, 0, 255)
    dsf = run("drop_shadow_filter", shape, offset_x=6, offset_y=6, blur=0, opacity=100)
    check("the drop shadow filter adds a shadow outside the shape",
          dsf[24, 24, 3] > 200 and tuple(dsf[24, 24, :3]) == (0, 0, 0) and tuple(dsf[15, 15, :3]) == (255, 0, 0))

    print("\nall filter checks passed")


if __name__ == "__main__":
    main()
