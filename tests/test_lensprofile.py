#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Lens Correction with lens profiles read from lensfun's database.

lensfun's data is CC BY-SA and not installed here, so the database is a
synthetic one this test writes into a temporary folder and points
OCHRE_LENSFUN_DIR at: an invented camera and lens with invented
coefficients, in lensfun's XML format. The user's path is then driven for
real: a photo with EXIF is opened, Filter > Lens Correction's dialog
matches the lens on its Profile tab, and OK applies it.

Run: python3 tests/test_lensprofile.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ochre.engine import lensprofile
from ochre.engine.effects import EffectRegistry
from ochre.engine.settings import Settings
from ochre.ui.controller import Controller

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TMP = tempfile.mkdtemp(prefix="ochre-lensprofile-")
REG = EffectRegistry()

# Invented: no real camera or lens, no lensfun data.
DATABASE = """<lensdatabase version="1">
  <mount><name>Acme X</name></mount>
  <camera>
    <maker>Acme</maker><model>Acme One</model><mount>Acme X</mount>
    <cropfactor>1.5</cropfactor>
  </camera>
  <lens>
    <maker>Acme</maker><model>Acme 18-55mm f/3.5-5.6</model><mount>Acme X</mount>
    <cropfactor>1.5</cropfactor>
    <calibration>
      <distortion model="ptlens" focal="18" a="0.01" b="-0.04" c="0"/>
      <distortion model="poly5" focal="55" k1="0.02" k2="-0.001"/>
      <tca model="linear" focal="18" kr="1.0004" kb="0.9996"/>
      <tca model="poly3" focal="55" vr="1.0001" vb="0.9999" cr="0.0002" br="0.0001"/>
      <vignetting model="pa" focal="18" aperture="3.5" distance="10" k1="-0.5" k2="0.1" k3="0"/>
      <vignetting model="pa" focal="18" aperture="8" distance="10" k1="-0.2" k2="0" k3="0"/>
      <vignetting model="pa" focal="55" aperture="5.6" distance="10" k1="-0.3" k2="0" k3="0"/>
    </calibration>
  </lens>
  <lens>
    <maker>Acme</maker><model>Acme Fish 8mm</model><type>fisheye</type><cropfactor>1.5</cropfactor>
    <calibration><distortion model="poly3" focal="8" k1="0.1"/></calibration>
  </lens>
  <lens>
    <maker>Acme</maker><model>Acme Broken 35mm</model><cropfactor>1.5</cropfactor>
    <calibration><distortion model="poly3" focal="35" k1="oops"/></calibration>
  </lens>
</lensdatabase>
"""


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(np.ravel(a), np.ravel(b)))


def database_dir():
    root = os.path.join(TMP, "lensfun")
    os.makedirs(os.path.join(root, "version_1"))
    with open(os.path.join(root, "version_1", "acme.xml"), "w") as fh:
        fh.write(DATABASE)
    return root


def picture(path, lens="18-55mm f/3.5-5.6", focal=18.0, aperture=3.5):
    """A grey photo with one dark vertical line near the left edge, saved
    with the EXIF a camera would write."""
    from PIL import Image
    px = np.full((120, 160, 3), 100, np.uint8)
    px[:, 20:22] = 0
    ex = Image.Exif()
    ex[0x010F] = "ACME"
    ex[0x0110] = "Acme One"
    ex[0x8769] = {0x920A: focal, 0x829D: aperture, 0xA434: lens}
    Image.fromarray(px, "RGB").save(path, "PNG", exif=ex.tobytes())
    return path


def main():
    root = database_dir()
    os.environ[lensprofile.ENV_DIR] = root

    # ---- reading the database ----------------------------------------------------------
    db = lensprofile.installed()
    check("the lensfun XML is read: one camera, one usable lens (a fisheye and a lens "
          "with a bad number are skipped)",
          [c.name for c in db.cameras] == ["Acme One"]
          and [l.name for l in db.lenses] == ["Acme 18-55mm f/3.5-5.6"], str(db.errors))
    lens = db.lenses[0]
    cal = lens.calibrations[0]
    s = math.hypot(1.5, 1.0)          # lensfun's default 3:2 sensor, in its units
    d = 1 - 0.01 + 0.04
    check("ptlens is converted to the half-diagonal with the centre scale divided out",
          close(cal.distortion[0][1], [0, -0.04 * s * s / d ** 3, 0.01 * s ** 3 / d ** 4, 0, 0, 0]))
    check("poly5 is converted to the half-diagonal",
          close(cal.distortion[1][1], [0, 0.02 * s * s, 0, -0.001 * s ** 4, 0, 0]))
    check("poly3 TCA scales c by s and b by s^2",
          close(cal.tca[1][1], [[1.0001, 0.0002 * s, 0.0001 * s * s], [1, 0, 0], [0.9999, 0, 0]]))

    # ---- one shot's model ----------------------------------------------------------------
    at = lambda f, a=5.6, dist=10.0, crop=1.5: lensprofile.lens_model(lens, f, a, dist, crop)
    mid = at(36.5)
    check("distortion and TCA are interpolated by focal length (halfway: the mean)",
          close(mid["distortion"], np.add(at(18)["distortion"], at(55)["distortion"]) / 2)
          and close(mid["ca"], np.add(at(18)["ca"], at(55)["ca"]) / 2))
    check("...and held at the ends outside the calibrated range",
          at(10)["distortion"] == at(18)["distortion"] and at(80)["ca"] == at(55)["ca"])
    check("vignetting at a calibrated focal length, aperture and distance is that calibration",
          close(at(18, 3.5)["vignette"], [-0.5, 0.1, 0.0]) and close(at(18, 8)["vignette"], [-0.2, 0, 0]))
    v = at(18, 5.0)["vignette"][0]
    check("...and between apertures lies between them (%.3f)" % v, -0.5 < v < -0.2)
    full = at(18, crop=1.0)
    check("a full-frame body scales each r^i term by (1.5 / 1)^i",
          close(full["distortion"], [c * 1.5 ** (i + 1) for i, c in enumerate(at(18)["distortion"])])
          and close(full["vignette"], [k * 1.5 ** (2 * i + 2) for i, k in enumerate(at(18)["vignette"])]))
    check("a photo's lens is found from its EXIF name, numbers and all",
          db.match_lenses("18-55mm f/3.5-5.6", "ACME")[0][1] is lens
          and db.match_lenses("18-55mm f/4", "ACME") == []
          and db.find_camera("ACME", "Acme One") is db.cameras[0])

    # ---- the user's path: open a photo, Lens Correction, Profile tab, OK ------------------
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("ok: skipped the dialog (PySide6 unavailable)")
        return
    from ochre.ui.bus import EventBus
    from ochre.ui.config import UiSettings
    from ochre.ui.effectdialog import EffectDialog
    from ochre.ui.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[],
                     state_dir=tempfile.mkdtemp(prefix="ochre-lensprofile-state-"))
    win = MainWindow(ctl, UiSettings(DATA))
    ctl.open_path(picture(os.path.join(TMP, "shot.png")))
    layer = ctl.active_layer
    original = ctl.doc.cell(layer).plane("rgba").copy()

    effect = win.effects.create("lens_correction")
    dlg = EffectDialog(effect, ctl, win, debounce_ms=0)
    panel = dlg.panel
    check("Lens Correction's dialog has a Profile tab beside Custom",
          dlg.tabs is not None and [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())]
          == ["Profile", "Custom"])
    check("...which matched the camera and lens from the photo's EXIF",
          panel.lens.currentData() == lens.name and "Acme One" in panel.camera.currentText()
          and "matched" in panel.status.text())
    check("...at the shot's focal length and aperture",
          panel.focal.value() == 18.0 and panel.aperture.value() == 3.5)

    # Vignetting alone first: its effect is exactly predictable.
    panel.checks["profile_distortion"].setChecked(False)
    panel.checks["profile_ca"].setChecked(False)
    n0 = len(ctl.history_labels())
    dlg.accept()
    out = ctl.doc.cell(layer).plane("rgba")
    r = math.hypot(79.5, 59.5) / math.hypot(80.0, 60.0)
    falloff = 1 - 0.5 * r ** 2 + 0.1 * r ** 4
    lin = ((100 / 255.0 + 0.055) / 1.055) ** 2.4 / falloff
    want = 255 * (1.055 * lin ** (1 / 2.4) - 0.055)
    check("OK divides the lens's falloff out: corner %d -> %d (expected %.1f), centre kept"
          % (original[0, 159, 0], out[0, 159, 0], want),
          abs(int(out[0, 159, 0]) - want) <= 1.0 and abs(int(out[60, 80, 0]) - 100) <= 1
          and len(ctl.history_labels()) == n0 + 1)
    ctl.undo()

    effect = win.effects.create("lens_correction")
    dlg = EffectDialog(effect, ctl, win, debounce_ms=0)
    dlg.accept()
    bent = ctl.doc.cell(layer).plane("rgba").copy()

    def line_x(a, y):
        """Where the dark line crosses row y, to a fraction of a pixel."""
        row = a[y, :60, 0].astype(float)
        dark = np.clip(np.median(row) - row, 0, None)
        return float((dark * np.arange(60)).sum() / max(dark.sum(), 1e-6))
    bow = (line_x(bent, 60) - line_x(bent, 3), line_x(original, 60) - line_x(original, 3))
    check("with every correction on, the profile's distortion bends the edge line "
          "(bow %.2f px, was %.2f)" % bow, abs(bow[0]) > 1.0 and abs(bow[1]) < 0.01)
    check("...and Auto scale leaves no blank border", int(bent[..., 3].min()) == 255)

    # The profile travels as numbers, so an Action replays it with no database.
    params = dict(effect.props.token())
    os.environ[lensprofile.ENV_DIR] = os.path.join(TMP, "empty")
    os.makedirs(os.environ[lensprofile.ENV_DIR])
    replay = REG.create("lens_correction", **params).apply(original)
    check("the recorded settings replay the same correction without the database",
          not lensprofile.installed() and np.array_equal(replay, bent))

    # ---- no database: the tab says so, and changes nothing ------------------------------
    ctl.undo()
    dlg = EffectDialog(win.effects.create("lens_correction"), ctl, win, debounce_ms=0)
    check("with no database the Profile tab says \"No lens database found\"",
          "No lens database found" in dlg.panel.status.text()
          and dlg.tabs.tabText(0) == "Profile")
    dlg.accept()
    check("...and the dialog's defaults still change nothing",
          np.abs(ctl.doc.cell(layer).plane("rgba").astype(int) - original.astype(int)).max() <= 1)

    # A lens the database does not have is not guessed at.
    os.environ[lensprofile.ENV_DIR] = root
    ctl.open_path(picture(os.path.join(TMP, "other.png"), lens="50mm f/1.4"))
    dlg = EffectDialog(win.effects.create("lens_correction"), ctl, win, debounce_ms=0)
    check("a lens missing from the database picks no profile, and says so",
          dlg.panel.lens.currentData() == "" and "not in the lens database" in dlg.panel.status.text()
          and dlg.effect.props.get("profile_model") == "")
    dlg.reject()
    app.processEvents()
    print("\nall lens profile checks passed")


if __name__ == "__main__":
    main()
