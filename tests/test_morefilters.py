#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The rest of the Filter and Adjustments menus: each does what it says.

Run: python3 tests/test_morefilters.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image

from ochre.engine import morefilters as M
from ochre.engine.effects import EffectRegistry

REG = EffectRegistry()


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def run(name, img, **params):
    return REG.create(name, **params).apply(img)


def solid(h, w, rgb, a=255):
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = a
    return out


def blocks(seed=1, n=8, cell=8):
    rng = np.random.default_rng(seed)
    out = np.zeros((n * cell, n * cell, 4), np.uint8)
    out[..., :3] = np.kron(rng.integers(20, 235, (n, n, 3)), np.ones((cell, cell, 1)))
    out[..., 3] = 255
    return out


def write_cube(path, fn, n=9):
    g = np.linspace(0, 1, n)
    with open(path, "w") as fh:
        fh.write("TITLE \"test\"\nLUT_3D_SIZE %d\n" % n)
        for b in g:
            for gg in g:
                for r in g:
                    fh.write("%.6f %.6f %.6f\n" % fn(r, gg, b))


def main():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (48, 64, 4), dtype=np.uint8)
    img[..., 3] = rng.integers(100, 256, (48, 64))

    # ---- all of them ---------------------------------------------------------------
    # Colour-only effects must leave alpha exactly alone; blurs, distortions
    # and re-tiling legitimately move it (as Gaussian Blur does).
    keeps_alpha = {"selective_color", "color_lookup", "hdr_toning", "smart_sharpen",
                   "shake_reduction", "despeckle", "difference_clouds", "lighting_effects",
                   "extrude", "trace_contour", "mezzotint", "hsb_hsl"}
    for cls in M.MORE:
        out = run(cls.name, img)
        check("%s runs and keeps the image's shape%s" % (cls.name, ", and its alpha"
                                                          if cls.name in keeps_alpha else ""),
              out.shape == img.shape and out.dtype == np.uint8
              and (cls.name not in keeps_alpha or np.array_equal(out[..., 3], img[..., 3])))
    from PySide6.QtWidgets import QApplication
    from ochre.ui.effectdialog import _Row
    app = QApplication.instance() or QApplication([])
    for cls in M.MORE:
        fx = REG.create(cls.name)
        rows = [_Row(p) for p in fx.props]
        check("%s: every setting gets a real control in the dialog" % cls.name,
              all(not hasattr(r, "label") for r in rows))

    # ---- Selective Color -------------------------------------------------------------
    pair = np.concatenate([solid(4, 4, (220, 40, 40)), solid(4, 4, (40, 40, 220))], axis=1)
    sc = run("selective_color", pair, colors="reds", cyan=100)
    check("Selective Color: cyan into the reds takes red out of a red pixel...",
          sc[0, 0, 0] < 200 and abs(int(sc[0, 0, 1]) - 40) <= 1)
    check("...and leaves a blue pixel alone", np.array_equal(sc[0, 6], pair[0, 6]))
    wk = run("selective_color", np.concatenate([solid(4, 4, (250, 250, 250)), solid(4, 4, (128, 128, 128))], 1),
             colors="whites", black=60)
    check("...black in the whites darkens white, not mid grey",
          wk[0, 0, 0] < 230 and abs(int(wk[0, 6, 0]) - 128) <= 1)

    # ---- Color Lookup ------------------------------------------------------------------
    d = tempfile.mkdtemp()
    ident = os.path.join(d, "identity.cube")
    inv = os.path.join(d, "invert.cube")
    write_cube(ident, lambda r, g, b: (r, g, b))
    write_cube(inv, lambda r, g, b: (1 - r, 1 - g, 1 - b))
    a = blocks()
    out = run("color_lookup", a, file=ident)
    check("Color Lookup: an identity .cube changes nothing",
          np.abs(out.astype(int) - a.astype(int)).max() <= 1)
    out = run("color_lookup", a, file=inv)
    check("...an inverting .cube inverts", np.abs(out[..., :3].astype(int) - (255 - a[..., :3])).max() <= 1)
    check("...amount 0 is no change", np.array_equal(run("color_lookup", a, file=inv, amount=0), a))
    for look in M.LOOKS:
        check("...the built-in look %r changes the image" % look,
              not np.array_equal(run("color_lookup", a, look=look), a))
    bad = os.path.join(d, "bad.cube")
    with open(bad, "w") as fh:
        fh.write("LUT_3D_SIZE 4\n0 0 0\n")
    try:
        M.parse_cube(open(bad).read())
        refused = False
    except ValueError:
        refused = True
    check("...a malformed .cube is refused, and the look is used instead",
          refused and np.array_equal(run("color_lookup", a, file=bad, look="warm"),
                                     run("color_lookup", a, look="warm")))

    # ---- HDR Toning ----------------------------------------------------------------------
    half = np.concatenate([solid(48, 48, (20, 20, 20)), solid(48, 48, (235, 235, 235))], axis=1)
    noise = rng.integers(-6, 7, half.shape[:2] + (1,))
    half[..., :3] = np.clip(half[..., :3].astype(int) + noise, 0, 255)
    hd = run("hdr_toning", half, strength=80, detail=50)
    gap_before = int(half[:, 60:, 0].mean() - half[:, :36, 0].mean())
    gap_after = int(hd[:, 60:, 0].mean() - hd[:, :36, 0].mean())
    check("HDR Toning pulls shadows and highlights together (gap %d -> %d)" % (gap_before, gap_after),
          gap_after < gap_before * 0.8)
    check("...while keeping local detail", hd[:, :36, 0].std() >= half[:, :36, 0].std() * 0.8)

    # ---- sharpening and deconvolution ----------------------------------------------------------
    edge = solid(32, 64, (40, 40, 40))
    edge[:, 32:, :3] = 210
    soft = run("gaussian_blur", edge, radius=3) if "gaussian_blur" in REG else edge

    def slope(x):
        return int(np.abs(np.diff(x[16, :, 0].astype(int))).max())
    resharp = run("smart_sharpen", soft, amount=200, radius=3.0)
    check("Smart Sharpen, told the blur's radius, steepens the edge (%d -> %d)"
          % (slope(soft), slope(resharp)), slope(resharp) > slope(soft) * 1.3)
    sharp = blocks(3)
    k = M._kernel("motion", 4, 0)
    shaken = sharp.copy()
    shaken[..., :3] = np.clip(np.rint(M._convolve(sharp[..., :3].astype(float), k)), 0, 255)
    fixed = run("shake_reduction", shaken, angle=0, length=8, iterations=30, smoothing=0)
    e0 = np.abs(shaken[..., :3].astype(int) - sharp[..., :3]).mean()
    e1 = np.abs(fixed[..., :3].astype(int) - sharp[..., :3]).mean()
    check("Shake Reduction undoes a known straight shake (error %.1f -> %.1f)" % (e0, e1), e1 < e0 * 0.7)

    # ---- Blur Gallery ---------------------------------------------------------------------------
    pic = blocks(5, 12, 8)
    iris = run("iris_blur", pic, radius_x=25, radius_y=25, feather=20, blur=10)
    check("Iris Blur keeps the centre sharp and blurs the corners",
          np.array_equal(iris[44:52, 44:52], pic[44:52, 44:52])
          and np.abs(iris[:8, :8].astype(int) - pic[:8, :8]).mean() > 3)
    fb = run("field_blur", pic, pin1_x=50, pin1_y=0, blur1=0, pin2_x=50, pin2_y=100, blur2=12)
    top_change = np.abs(fb[:6].astype(int) - pic[:6]).mean()
    bottom_change = np.abs(fb[-6:].astype(int) - pic[-6:]).mean()
    check("Field Blur: sharp at the 0 px pin, blurred at the 12 px pin (%.1f vs %.1f)"
          % (top_change, bottom_change), top_change < 1 and bottom_change > 3)

    # ---- Despeckle ----------------------------------------------------------------------------
    flat = solid(40, 80, (100, 100, 100))
    flat[:, 40:, :3] = 200
    speck = flat.copy()
    speck[..., :3] = np.clip(flat[..., :3].astype(int) + rng.normal(0, 12, (40, 80, 1)), 0, 255)
    ds = run("despeckle", speck)
    check("Despeckle quietens flat areas", ds[:, 5:30, 0].std() < speck[:, 5:30, 0].std() * 0.8)
    check("...and keeps the edge", int(ds[:, 44:, 0].mean()) - int(ds[:, :36, 0].mean()) > 85)

    # ---- Render ---------------------------------------------------------------------------------
    black = solid(40, 60, (0, 0, 0))
    once = run("difference_clouds", black, seed=4)
    twice = run("difference_clouds", once, seed=4)
    check("Difference Clouds: clouds on black, and the same clouds again cancel to black",
          once[..., :3].std() > 10 and twice[..., :3].max() == 0)
    fib = run("fibers", black).astype(float)[..., 0]
    vert = np.abs(np.diff(fib, axis=0)).mean()
    horiz = np.abs(np.diff(fib, axis=1)).mean()
    check("Fibers run vertically (row-to-row change %.1f < column-to-column %.1f)" % (vert, horiz),
          vert < horiz)
    grey = solid(60, 60, (160, 160, 160))
    lit = run("lighting_effects", grey, light="spot", light_x=20, light_y=20)
    check("Lighting Effects: brighter under the spot than far from it",
          int(lit[12, 12, 0]) > int(lit[55, 55, 0]) + 30)

    # ---- Stylize ----------------------------------------------------------------------------------
    pyr = run("extrude", solid(32, 32, (128, 128, 128)), type="pyramids", size=32)
    faces = {int(pyr[2, 16, 0]), int(pyr[16, 2, 0]), int(pyr[29, 16, 0]), int(pyr[16, 29, 0])}
    check("Extrude pyramids: four faces, four shades", len(faces) == 4, str(faces))
    tl = run("tiles", pic, tiles=6, offset=20, fill="background", color=(255, 0, 255, 255))
    check("Tiles: tiles shift and the gaps show the background colour",
          ((tl[..., 0] == 255) & (tl[..., 1] == 0) & (tl[..., 2] == 255)).sum() > 10)
    step = solid(20, 40, (50, 50, 50))
    step[:, 20:, :3] = 200
    tc = run("trace_contour", step, level=128)
    check("Trace Contour draws a line where the level is crossed, and nothing elsewhere",
          (tc[:, 18:21, 0] == 0).all(axis=0).any() and (tc[:, :15, 0] == 255).all()
          and (tc[:, 25:, 0] == 255).all())

    # ---- Distort ----------------------------------------------------------------------------------
    check("ZigZag at amount 0 changes nothing",
          np.abs(run("zigzag", pic, amount=0).astype(int) - pic.astype(int)).max() <= 1)
    mid = np.roll(pic, 4, axis=(0, 1))          # the centre now falls inside a flat block
    zz = run("zigzag", mid, amount=60)
    check("...and keeps the centre while rippling the rest",
          np.abs(zz[47:49, 47:49].astype(int) - mid[47:49, 47:49]).max() <= 1
          and not np.array_equal(zz, mid))
    line = solid(40, 80, (255, 255, 255))
    line[:, 40, :3] = 0
    sh = run("shear", line, top=0, middle=40, bottom=0)

    def where_dark(a, y):
        xs = np.nonzero(a[y, :, 0] < 128)[0]
        return float(xs.mean()) if xs.size else None
    check("Shear moves the middle rows and leaves the top and bottom",
          abs(where_dark(sh, 0) - 40) <= 1.5 and abs(where_dark(sh, 39) - 40) <= 2.5
          and abs(where_dark(sh, 20) - 40) > 8)
    mp = os.path.join(d, "map.png")
    Image.fromarray(np.full((10, 10, 3), (228, 128, 128), np.uint8)).save(mp)
    dp = run("displace", line, horizontal=10, vertical=10, file=mp)
    moved = where_dark(dp, 20)
    check("Displace by a map file: red above mid grey moves pixels sideways (line at %.1f)" % moved,
          moved is not None and 6 <= abs(moved - 40) <= 14)

    # ---- Pixelate / Other -----------------------------------------------------------------------
    photo = rng.integers(0, 256, (40, 40, 4), dtype=np.uint8)
    photo[..., :3] = np.clip(np.kron(rng.integers(30, 220, (5, 5, 3)), np.ones((8, 8, 1)))
                             + rng.normal(0, 8, (40, 40, 3)), 0, 255)
    photo[..., 3] = 255
    fc = run("facet", photo)

    def colours(a):
        return len({tuple(p) for p in a[..., :3].reshape(-1, 3)})
    check("Facet clumps colours (%d -> %d distinct)" % (colours(photo), colours(fc)),
          colours(fc) < colours(photo) * 0.8)
    mz = run("mezzotint", photo)
    check("Mezzotint leaves every channel fully on or fully off",
          set(np.unique(mz[..., :3]).tolist()) <= {0, 255})
    for there, back in (("rgb_to_hsb", "hsb_to_rgb"), ("rgb_to_hsl", "hsl_to_rgb")):
        rt = run("hsb_hsl", run("hsb_hsl", photo, mode=there), mode=back)
        err = int(np.abs(rt[..., :3].astype(int) - photo[..., :3]).max())
        check("HSB/HSL: %s then %s comes back (max error %d)" % (there, back, err), err <= 6)

    # ---- the gallery can stack them --------------------------------------------------------------
    import json
    st = run("filter_stack", pic, stack=json.dumps([{"name": "despeckle", "params": {}},
                                                     {"name": "color_lookup", "params": {"look": "cool"}}]))
    check("the Filter Gallery stacks these too",
          np.array_equal(st, run("color_lookup", run("despeckle", pic), look="cool")))
    del app
    print("\nall more-filter checks passed")


if __name__ == "__main__":
    main()
