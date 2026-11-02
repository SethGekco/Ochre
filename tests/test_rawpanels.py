#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The Camera Raw Filter's panels beyond Basic (ported from Kooka
Pictura): Tone Curve, Detail, HSL / Grayscale, Split Toning, Effects and
Camera Calibration. Each slider does what its name says on a picture made
to show it, every slider end renders cleanly, the whole filter still runs
through the controller as the menu does, and its dialog puts each panel
on its own tab.

Run: python3 tests/test_rawpanels.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine import rawpanels
from ochre.engine.effects import EffectRegistry
from ochre.engine.props import BoolProperty, ChoiceProperty, FloatProperty, IntProperty
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REG = EffectRegistry()


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def raw(img, **params):
    return REG.create("camera_raw", **params).apply(img)


def solid(rgb, h=8, w=8):
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = 255
    return out


def ramp(h=8, w=64):
    """A gray ramp, black to white, left to right."""
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = (np.arange(w) * 255 // (w - 1))[None, :, None]
    out[..., 3] = 255
    return out


def patches():
    """Red, orange, yellow, green, aqua, blue, purple and magenta patches,
    then a dark and a light gray; a soft alpha corner."""
    colors = [(220, 40, 40), (230, 140, 40), (220, 220, 40), (40, 200, 40), (40, 200, 200),
              (40, 60, 220), (130, 50, 200), (210, 40, 200), (40, 40, 40), (215, 215, 215)]
    out = np.zeros((8, 8 * len(colors), 4), np.uint8)
    for i, c in enumerate(colors):
        out[:, 8 * i:8 * i + 8, :3] = c
    out[..., 3] = 255
    out[:2, :2, 3] = 90
    return out


def patch(img, i):
    return img[4, 8 * i + 4, :3].astype(int)


def main():
    pic = patches()
    gray = ramp()
    rng = np.random.default_rng(4)
    noisy = solid((128, 128, 128), 32, 32)
    noisy[..., :3] = np.clip(128 + rng.normal(0, 18, (32, 32, 3)), 0, 255)

    check("every new panel at its defaults changes nothing",
          np.array_equal(raw(pic), pic) and np.array_equal(raw(gray), gray))

    # ---- Tone Curve ---------------------------------------------------------------
    up = raw(gray, region_shadows=100)[0, :, 0].astype(int)
    base = gray[0, :, 0].astype(int)
    check("Tone Curve: Shadows lifts the shadows and leaves the highlights",
          up[10] > base[10] and up[-1] == base[-1])
    down = raw(gray, region_highlights=-100)[0, :, 0].astype(int)
    check("...Highlights darkens the highlights and leaves the shadows",
          down[56] < base[56] and down[2] == base[2])
    moved = raw(gray, region_shadows=100, shadow_split=60)[0, :, 0].astype(int)
    check("...and the Shadow Split moves where Shadows reaches", moved[20] > up[20])
    bright = raw(gray, curve_rgb=((0, 0), (128, 180), (255, 255)))[0, :, 0].astype(int)
    check("the point curve bends every channel, and stays monotone",
          bright[32] > base[32] and np.all(np.diff(bright) >= 0))
    red = raw(gray, curve_red="0,0;128,190;255,255")[0, 32].astype(int)
    check("...the red curve only the red", red[0] > red[1] and red[1] == red[2])

    # ---- Detail ------------------------------------------------------------------
    edge = np.zeros((8, 32, 4), np.uint8)
    edge[..., :3] = 100
    edge[:, 16:, :3] = 160
    edge[..., 3] = 255
    sharp = raw(edge, sharpen_amount=150)[4, :, 0].astype(int)
    check("Detail: sharpening steepens an edge", sharp[15] < 100 and sharp[16] > 160)
    masked = raw(edge, sharpen_amount=150, sharpen_masking=100)
    check("...Masking keeps it off the flats", np.array_equal(masked[4, :8], edge[4, :8]))
    smooth = raw(noisy, noise_luminance=100)
    check("...Luminance noise reduction calms brightness noise",
          smooth[..., :3].std() < noisy[..., :3].std())
    calm = raw(noisy, noise_color=100).astype(int)
    chroma = lambda a: (a[..., :3].max(-1) - a[..., :3].min(-1)).mean()
    check("...Color noise reduction calms color noise", chroma(calm) < chroma(noisy.astype(int)))

    # ---- HSL / Grayscale ---------------------------------------------------------
    g = raw(pic, grayscale=True)
    check("Grayscale: every pixel gray, alpha kept",
          np.all(g[..., 0] == g[..., 1]) and np.all(g[..., 1] == g[..., 2])
          and np.array_equal(g[..., 3], pic[..., 3]))
    mixed = raw(pic, grayscale=True, gray_blues=100)
    check("...the Blues mix lightens only the blues",
          patch(mixed, 5)[0] > patch(g, 5)[0] and np.array_equal(patch(mixed, 0), patch(g, 0)))
    desat = raw(pic, sat_blues=-100)
    b = patch(desat, 5)
    check("HSL: Blues saturation -100 grays the blue and leaves the red",
          b.max() - b.min() < 20 and np.array_equal(patch(desat, 0), patch(pic, 0)))
    turned = raw(pic, hue_reds=100)
    check("...Reds hue turns red toward orange", patch(turned, 0)[1] > patch(pic, 0)[1] + 20)
    lit = raw(pic, lum_greens=100)
    check("...Greens luminance lightens the green, not the grays",
          patch(lit, 3).sum() > patch(pic, 3).sum()
          and np.array_equal(patch(lit, 8), patch(pic, 8)))

    # ---- Split Toning --------------------------------------------------------------
    toned = raw(pic, shadow_hue=240, shadow_saturation=100, highlight_hue=40,
                highlight_saturation=100)
    dark, light = patch(toned, 8), patch(toned, 9)
    check("Split Toning: blue shadows, warm highlights",
          dark[2] > dark[0] and light[0] > light[2])
    check("...nothing without saturation", np.array_equal(raw(pic, shadow_hue=240), pic))

    # ---- Effects -------------------------------------------------------------------
    mid = solid((150, 150, 150), 48, 48)
    v = raw(mid, vignette=-80)
    v_far = raw(mid, vignette=-80, vignette_midpoint=100)
    check("Effects: a later Midpoint darkens less of the frame",
          int(v_far[..., 0].sum()) > int(v[..., 0].sum()))
    soft = raw(mid, vignette=-80, vignette_feather=0)
    check("...less Feather, a gentler fall", int(soft[..., 0].sum()) > int(v[..., 0].sum()))
    square = raw(mid, vignette=-80, vignette_roundness=-100)
    check("...Roundness reshapes it", not np.array_equal(square, v))
    white = solid((245, 245, 245), 48, 48)
    kept = raw(white, vignette=-80, vignette_highlights=100)
    check("...Highlights spares the lights under Highlight Priority",
          kept[0, 0, 0] > raw(white, vignette=-80)[0, 0, 0]
          and np.array_equal(raw(white, vignette=-80, vignette_highlights=100,
                                 vignette_style="color_priority"), raw(white, vignette=-80)))
    grain = raw(mid, grain=60)
    check("...grain Size and Roughness change the grain",
          not np.array_equal(raw(mid, grain=60, grain_size=90), grain)
          and raw(mid, grain=60, grain_roughness=100)[..., 0].std() > grain[..., 0].std())

    # ---- Camera Calibration ---------------------------------------------------------
    tinted = patch(raw(pic, shadows_tint=100), 8)
    check("Calibration: a magenta Shadows Tint", tinted[0] > tinted[1] and tinted[2] > tinted[1])
    primary = raw(pic, blue_saturation=-100)
    pb = patch(primary, 5)
    check("...Blue Primary saturation grays the blue", pb.max() - pb.min() < 20)

    # ---- every slider end ----------------------------------------------------------
    fx = REG.create("camera_raw")
    bad = []
    for prop in fx.props:
        if isinstance(prop, (IntProperty, FloatProperty)):
            ends = (prop.minimum, prop.maximum)
        elif isinstance(prop, ChoiceProperty):
            ends = tuple(prop.choices)
        elif isinstance(prop, BoolProperty):
            ends = (True,)
        else:
            ends = ("0,255;255,0",)                        # an inverted curve
        for value in ends:
            out = raw(pic, **{prop.name: value})
            if out.shape != pic.shape or not np.array_equal(out[..., 3], pic[..., 3]):
                bad.append((prop.name, value))
    check("every slider end renders, alpha kept", not bad, str(bad))

    # ---- through the controller ----------------------------------------------------
    params = {"region_lights": 40, "sat_reds": -60, "shadow_saturation": 50,
              "sharpen_amount": 80, "red_hue": 30, "vignette": -40, "vignette_midpoint": 30}
    direct = raw(pic, **params)
    ctl = Controller(Settings(DATA), DATA, addon_dirs=[],
                     state_dir=tempfile.mkdtemp(prefix="ochre-rawpanels-"))
    ctl.new_document(pic.shape[1], pic.shape[0], background=None)
    cell = ctl.doc.cell(ctl.active_layer)
    cell.plane("rgba")[...] = pic
    cell._touched(ctl.doc.bounds)
    ctl.start_recording("probe")
    ctl.apply_effect("camera_raw", params)
    rec = ctl.stop_recording()
    got = cell.plane("rgba").copy()
    check("applied as the menu does, it is the direct result", np.array_equal(got, direct))
    ctl.undo()
    check("...one undo step", np.array_equal(cell.plane("rgba"), pic))
    ctl.play_action(rec["steps"])
    check("...and the recorded action replays it", np.array_equal(cell.plane("rgba"), got))
    stacked = REG.create("filter_stack", stack=json.dumps(
        [{"name": "camera_raw", "params": params, "enabled": True}])).apply(pic)
    check("...as does a Filter Gallery stack", np.array_equal(stacked, direct))

    # ---- the dialog ----------------------------------------------------------------
    from PySide6.QtWidgets import QApplication, QTabWidget
    from ochre.ui.effectdialog import EffectDialog
    app = QApplication.instance() or QApplication([])
    dlg = EffectDialog(REG.create("camera_raw"), ctl)
    tabs = dlg.findChild(QTabWidget)
    names = [tabs.tabText(i) for i in range(tabs.count())] if tabs else []
    check("the dialog has a tab per panel",
          names == ["Basic", "Tone Curve", "Detail", "HSL / Grayscale", "Split Toning",
                    "Effects", "Camera Calibration"], str(names))
    plain = EffectDialog(REG.create("gaussian_blur"), ctl)
    check("...and a filter without panels keeps its plain form",
          plain.findChild(QTabWidget) is None)
    dlg.reject()
    plain.reject()
    del app
    check("rawpanels lists only real, used sliders",
          len(rawpanels.props()) + len(rawpanels.calibration_props()) == 62)


if __name__ == "__main__":
    main()
