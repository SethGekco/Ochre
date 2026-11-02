#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Autosave and crash recovery, through the real window.

The crash is real: a child Ochre paints, takes its autosave through the
window's own timer callback, and is killed with SIGKILL -- no cleanup runs,
exactly as after a segfault or the OOM killer. The next start is a real
MainWindow on the same state dir, whose recovery offer is answered by
pressing its Recover button. The recovered documents must hold the pixels
the dead one had.

Everything runs in a temporary state dir; XDG_STATE_HOME is pointed there
too, so nothing can reach the user's own ~/.local/state/ochre.

Run: QT_QPA_PLATFORM=offscreen python3 tests/test_recovery.py
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
TMP = tempfile.mkdtemp(prefix="ochre-recovery-test-")
os.environ["XDG_STATE_HOME"] = os.path.join(TMP, "xdg-state")
os.environ.pop("OCHRE_NO_PROMPT", None)
os.environ.pop("OCHRE_SELFTEST", None)


def check(name, cond, detail=""):
    if not cond:
        print("FAIL: %s %s" % (name, detail))
        sys.exit(1)
    print("ok: " + name)


try:
    import numpy as np
    from PySide6.QtWidgets import QApplication, QMessageBox
except ImportError as exc:
    print("ok: skipped (PySide6 unavailable: %s)" % exc)
    sys.exit(0)

from ochre.engine.compositor import Compositor
from ochre.engine.settings import Settings
from ochre.ui.bus import EventBus
from ochre.ui.config import UiSettings
from ochre.ui.controller import Controller
from ochre.ui.mainwindow import MainWindow
from ochre.ui.recovery import RecoveryStore, recovery_root

DATA = os.path.join(ROOT, "data")

# The instance that crashes. It opens a saved file, paints on it, opens a
# second (untitled) tab and paints there, lets the autosave timer fire, waits
# for the snapshots to land, records what it had, and dies by SIGKILL.
CRASHER = r'''
import os, signal, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from PySide6.QtWidgets import QApplication
from ochre.engine.compositor import Compositor
from ochre.engine.settings import Settings
from ochre.ui.bus import EventBus
from ochre.ui.config import UiSettings
from ochre.ui.controller import Controller
from ochre.ui.mainwindow import MainWindow
DATA = os.path.join(sys.argv[1], "data")
state, original, expected = sys.argv[2], sys.argv[3], sys.argv[4]
app = QApplication([])
ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=state)
win = MainWindow(ctl, UiSettings(DATA))
win.open_file(original)

def stroke(layer, pts, colour):
    ctl.set_active_layer(layer)
    ctl.set_primary(colour)
    ctl.begin_stroke(*pts[0])
    for p in pts[1:-1]:
        ctl.motion_stroke(*p)
    ctl.end_stroke(*pts[-1])

ctl.set_tool("brush", size=9)
layers = ctl.doc.layers()
stroke(layers[0], [(3, 3), (40, 30), (90, 70)], (250, 10, 10, 255))
stroke(layers[1], [(90, 5), (10, 70)], (10, 250, 10, 255))
ctl.selection.select_rect((5, 5, 30, 20))
ctl.new_document(64, 48)
stroke(ctl.active_layer, [(0, 0), (63, 47)], (0, 0, 200, 255))
win._autosave_timer.timeout.emit()
ctl.recovery.wait()
out = {}
for i, state_ in enumerate(ctl.documents):
    st = ctl._doc_state(i)
    doc = st["doc"]
    for (lid, fid), s in doc.cells.items():
        out["%d|%s|%s" % (i, lid, fid)] = s.plane("rgba")
    out["%d|composite" % i] = Compositor(doc).composite(doc.bounds)
out["sel"] = ctl._doc_state(0)["selection"].mask
np.savez(expected, **out)
print(ctl.recovery.directory, flush=True)
os.kill(os.getpid(), signal.SIGKILL)
'''

# An instance that is alive the whole time the test looks at its files.
LIVE = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
from PySide6.QtWidgets import QApplication, QMessageBox
from ochre.engine.settings import Settings
from ochre.ui.bus import EventBus
from ochre.ui.config import UiSettings
from ochre.ui.controller import Controller
from ochre.ui.mainwindow import MainWindow
DATA = os.path.join(sys.argv[1], "data")
app = QApplication([])
ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=sys.argv[2])
win = MainWindow(ctl, UiSettings(DATA))
ctl.set_tool("brush", size=5)
ctl.begin_stroke(2, 2)
ctl.end_stroke(30, 30)
win._autosave_timer.timeout.emit()
ctl.recovery.wait()
print(ctl.recovery.directory, flush=True)
sys.stdin.read()                       # alive until the test closes our stdin
QMessageBox.exec = lambda self: QMessageBox.Discard
win.close()                            # a clean exit, through closeEvent
'''


def child_env():
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def snapshots(directory):
    return sorted(n for n in os.listdir(directory) if n.endswith(".ochre")) \
        if directory and os.path.isdir(directory) else []


def sessions(state):
    root = recovery_root(state)
    return sorted(n for n in os.listdir(root) if not n.startswith(".") and n != "unrecoverable") \
        if os.path.isdir(root) else []


class Answer:
    """Patch QMessageBox.exec to press one button by its text, and note
    every question asked."""

    def __init__(self, text):
        self.text, self.asked = text, []

    def __enter__(self):
        self._real = QMessageBox.exec
        answer = self

        def exec_(box):
            answer.asked.append(box.text())
            for b in box.buttons():
                if b.text().replace("&", "") == answer.text:
                    b.click()
                    # exec() returns the standard button pressed, if it was one.
                    standard = box.standardButton(b)
                    return standard if standard != QMessageBox.NoButton else 0
            return QMessageBox.Cancel
        QMessageBox.exec = exec_
        return self

    def __exit__(self, *exc):
        QMessageBox.exec = self._real


def start(state):
    """A fresh Ochre on a state dir, as app.main builds it, with the event
    loop given a turn so its start-up recovery offer runs."""
    ctl = Controller(Settings(DATA), DATA, EventBus(), addon_dirs=[], state_dir=state)
    win = MainWindow(ctl, UiSettings(DATA))
    QApplication.processEvents()
    return ctl, win


def stroke(ctl, pts, colour, layer=None):
    if layer is not None:
        ctl.set_active_layer(layer)
    ctl.set_primary(colour)
    ctl.begin_stroke(*pts[0])
    for p in pts[1:-1]:
        ctl.motion_stroke(*p)
    ctl.end_stroke(*pts[-1])


def main():
    try:
        run()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def run():
    app = QApplication.instance() or QApplication([])
    state = os.path.join(TMP, "state")
    check("(the test's state dir is a temporary one)",
          state.startswith(tempfile.gettempdir()) and recovery_root(state).startswith(TMP))

    # ---- the original file, saved through a normal window ------------------
    original = os.path.join(TMP, "art.ochre")
    ctl, win = start(state)
    check("a window with a state dir opens a locked recovery session",
          ctl.recovery is not None and os.path.isfile(os.path.join(ctl.recovery.directory, "lock")))
    check("...and starts the autosave timer at the Preferences interval (5 min)",
          win._autosave_timer.isActive() and win._autosave_timer.interval() == 5 * 60 * 1000)
    ctl.new_document(96, 72)
    ctl.add_layer("Ink")
    ctl.set_tool("brush", size=7)
    stroke(ctl, [(10, 60), (80, 60)], (120, 80, 40, 255))
    ctl.save_path(original)
    with Answer("Discard"):
        win.close()
    check("a clean exit deletes its session folder", sessions(state) == [], sessions(state))
    before = sha(original)

    # ---- a crash -------------------------------------------------------------
    expected = os.path.join(TMP, "expected.npz")
    proc = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", CRASHER,
                           ROOT, state, original, expected],
                          env=child_env(), capture_output=True, text=True, timeout=300)
    check("(the crashing instance died by SIGKILL after autosaving)",
          proc.returncode == -9 and proc.stdout.strip(), (proc.returncode, proc.stderr[-2000:]))
    dead = proc.stdout.strip().splitlines()[-1]
    check("a killed instance leaves its session behind: lock and one snapshot per document",
          os.path.isfile(os.path.join(dead, "lock")) and len(snapshots(dead)) == 2,
          os.listdir(dead))
    want = np.load(expected)

    # ---- the next start offers them back -------------------------------------
    with Answer("Recover") as answer:
        ctl, win = start(state)
    check("the next start asks about the documents the dead instance left",
          len(answer.asked) == 1 and "did not shut down normally" in answer.asked[0], answer.asked)
    titles = ctl.document_titles()
    check("Recover opens each as an unsaved '-Recovered' tab",
          titles == ["*art-Recovered", "*Untitled-Recovered"], titles)
    check("...untitled, so Save cannot overwrite the original",
          all(ctl._doc_state(i)["path"] is None for i in range(2)))
    check("...and modified, so closing them asks first",
          ctl.modified_documents() == [0, 1], ctl.modified_documents())
    same = True
    for key in want.files:
        parts = key.split("|")
        if key == "sel":
            got = ctl._doc_state(0)["selection"].mask
        elif parts[1] == "composite":
            doc = ctl._doc_state(int(parts[0]))["doc"]
            got = Compositor(doc).composite(doc.bounds)
        else:
            cell = ctl._doc_state(int(parts[0]))["doc"].cells.get((parts[1], parts[2]))
            got = None if cell is None else cell.plane("rgba")
        if got is None or not np.array_equal(got, want[key]):
            same = False
            print("    differs: %s" % key)
    check("the recovered pixels (every layer, the composite, the selection) equal the "
          "dead instance's", same)
    check("the original file was not touched", sha(original) == before)
    check("the dead session is deleted once its documents are recovered", not os.path.exists(dead))
    check("...only after they were snapshotted into this instance's session",
          len(snapshots(ctl.recovery.directory)) == 2)
    win.switch_document(0)
    check("it remembers the file it came from, for Save As to start beside it",
          ctl.document_name == "art-Recovered" and ctl.recovered_from == original)

    # ---- saving or closing drops a document's snapshot -------------------------
    ctl.save_path(os.path.join(TMP, "art-Recovered.ochre"))
    check("saving a document deletes its recovery snapshot",
          len(snapshots(ctl.recovery.directory)) == 1)
    with Answer("Discard"):
        win.close_document(1)
    check("closing a document (Discard) deletes its snapshot",
          snapshots(ctl.recovery.directory) == [])

    # ---- the autosave tick, as the timer runs it -------------------------------
    stroke(ctl, [(5, 5), (50, 5)], (0, 0, 0, 255))
    ctl.begin_stroke(5, 40)                        # mid-stroke...
    ctl.motion_stroke(30, 40)
    win._autosave_timer.timeout.emit()
    ctl.recovery.wait()
    check("a tick mid-stroke takes no snapshot", snapshots(ctl.recovery.directory) == [])
    ctl.end_stroke(60, 40)
    win._autosave_timer.timeout.emit()
    ctl.recovery.wait()
    check("...the next tick does", len(snapshots(ctl.recovery.directory)) == 1)
    token = ctl.recovery_token
    path = ctl.recovery.snapshot_path(token)
    mtime = os.stat(path).st_mtime_ns
    win._autosave_timer.timeout.emit()
    ctl.recovery.wait()
    check("a tick with nothing new since the last snapshot writes nothing",
          os.stat(path).st_mtime_ns == mtime)
    ctl.undo()
    ctl.undo()
    win._autosave_timer.timeout.emit()
    ctl.recovery.wait()
    check("undoing back to the saved state drops the snapshot", snapshots(ctl.recovery.directory) == [])

    # ---- a big document: the tick does not hold up the UI ----------------------
    ctl.new_document(1800, 1800)
    rng = np.random.default_rng(3)
    big = [ctl.active_layer] + [ctl.add_layer("B%d" % i) for i in range(3)]
    for i, lay in enumerate(big):
        cell = ctl.doc.cell(lay)
        img = np.full((1800, 1800, 4), 255, np.uint8)
        img[:600, :, :3] = rng.integers(0, 256, (600, 1800, 3), dtype=np.uint8)
        img[..., 1] = 60 * i
        ctl.set_active_layer(lay)
        stroke(ctl, [(0, 0), (1, 1)], (1, 1, 1, 255))  # a real edit, so it is modified
        cell.write(ctl.doc.bounds, img)
    t0 = time.perf_counter()
    win.autosave_tick()
    tick = time.perf_counter() - t0
    ctl.recovery.wait()
    total = time.perf_counter() - t0
    check("on a 4-layer 1800x1800 document the timer callback returns in a fraction of "
          "the write (%.0f ms of %.0f ms)" % (tick * 1e3, total * 1e3), tick < total / 4)
    misses = ctl.save_cache.misses
    stroke(ctl, [(100, 100), (900, 900)], (255, 0, 0, 255), layer=big[2])
    win.autosave_tick()
    ctl.recovery.wait()
    check("the next tick re-encodes only the layer that changed (the save cache)",
          ctl.save_cache.misses == misses + 1, ctl.save_cache.misses - misses)
    with Answer("Discard"):
        win.close_document()

    # ---- Preferences turn it off and on, and set the interval -------------------
    from ochre.ui.prefsdialog import PreferencesDialog
    stroke(ctl, [(1, 1), (20, 20)], (9, 9, 9, 255))
    win.autosave_tick()
    ctl.recovery.wait()
    check("(setup: a snapshot exists)", len(snapshots(ctl.recovery.directory)) == 1)

    def prefs(on, minutes):
        real = PreferencesDialog.exec

        def exec_(dlg):
            dlg.widgets[("Recovery", "Autosave")][1].setChecked(on)
            dlg.widgets[("Recovery", "AutosaveMinutes")][1].setValue(minutes)
            dlg.accept()
            return 1
        PreferencesDialog.exec = exec_
        try:
            win.do_preferences()
        finally:
            PreferencesDialog.exec = real
    prefs(False, 5)
    check("turning Autosave off in Preferences stops the timer and deletes the snapshots",
          not win._autosave_timer.isActive() and snapshots(ctl.recovery.directory) == [])
    win.autosave_tick()
    ctl.recovery.wait()
    check("...and a tick then does nothing", snapshots(ctl.recovery.directory) == [])
    prefs(True, 2)
    check("turning it on with 2 minutes restarts the timer at 2 minutes",
          win._autosave_timer.isActive() and win._autosave_timer.interval() == 2 * 60 * 1000)
    with Answer("Discard"):
        win.close()

    # ---- another instance's files are never taken --------------------------------
    live = subprocess.Popen([sys.executable, "-c", LIVE, ROOT, state], env=child_env(),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        live_dir = live.stdout.readline().strip()
        check("(a second, live instance has a snapshot)", len(snapshots(live_dir)) == 1)
        with Answer("Recover") as answer:
            ctl, win = start(state)
        check("a start while another instance runs offers nothing of its", answer.asked == [])
        check("...and leaves its files alone", len(snapshots(live_dir)) == 1
              and os.path.isfile(os.path.join(live_dir, "lock")))
        with Answer("Discard"):
            win.close()
    finally:
        live.stdin.close()
        live.wait(timeout=120)
    check("the live instance's own clean exit deletes its session",
          live.returncode == 0 and not os.path.exists(live_dir), live.returncode)

    # ---- two instances finding the same orphan: one gets it ----------------------
    proc = subprocess.run([sys.executable, "-c", CRASHER, ROOT, state, original, expected],
                          env=child_env(), capture_output=True, text=True, timeout=300)
    dead = proc.stdout.strip().splitlines()[-1]
    first, second = RecoveryStore(state), RecoveryStore(state)
    first.open()
    second.open()
    mine = first.orphans()
    check("of two instances scanning at once, the first claims the orphan",
          [o.directory for o in mine] == [dead])
    check("...and the second sees nothing while the first is deciding", second.orphans() == [])
    for o in mine:
        o.release()
    with Answer("Later") as answer:
        ctl, win = start(state)
    check("Later leaves the documents for the next start",
          len(answer.asked) == 1 and os.path.isdir(dead) and len(snapshots(dead)) == 2)
    with Answer("Discard"):
        win.close()
    first.close()
    second.close()
    with Answer("Discard") as answer:
        ctl, win = start(state)
    check("Discard deletes them", len(answer.asked) == 1 and not os.path.exists(dead))
    with Answer("Discard"):
        win.close()

    # ---- a snapshot that will not open is set aside, not offered forever ---------
    proc = subprocess.run([sys.executable, "-c", CRASHER, ROOT, state, original, expected],
                          env=child_env(), capture_output=True, text=True, timeout=300)
    dead = proc.stdout.strip().splitlines()[-1]
    bad = os.path.join(dead, snapshots(dead)[0])
    with open(bad, "wb") as fh:
        fh.write(b"not a zip")
    real_warning = QMessageBox.warning
    warned = []
    QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a[2]))
    try:
        with Answer("Recover"):
            ctl, win = start(state)
    finally:
        QMessageBox.warning = real_warning
    check("a corrupt snapshot is reported, the good one still recovered",
          len(warned) == 1 and len(ctl.documents) == 1 and ctl.is_modified(), warned)
    check("...the corrupt one moved under recovery/unrecoverable/, the session deleted",
          not os.path.exists(dead) and os.listdir(os.path.join(recovery_root(state), "unrecoverable",
                                                               os.path.basename(dead))))
    with Answer("Discard"):
        win.close()
    check("nothing is left behind after clean exits", sessions(state) == [], sessions(state))
    app.processEvents()


if __name__ == "__main__":
    main()
