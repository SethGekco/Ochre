# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Autosave and crash recovery: the recovery folder and its locks.

    <state dir>/recovery/
    |-- 41873-1793512345678-a1b2c3/      one folder per running Ochre
    |   |-- lock                         held (flock) while that Ochre runs
    |   |-- 3f9c0e12aa41.ochre           the latest snapshot of one document
    |   `-- 3f9c0e12aa41.ini             its name, its file, when it was taken
    `-- unrecoverable/                   snapshots that would not open

Each running instance owns one session folder and holds an exclusive lock on
its lock file for as long as it runs. The operating system drops that lock
when the process dies, however it dies, so "can I take this lock?" is the
whole of crash detection: a folder whose lock can be taken belongs to an
Ochre that is gone. A PID in a file can be reused by an unrelated process;
a lock cannot outlive its holder. And a live instance's lock cannot be
taken, so one Ochre never reads, offers or deletes another's snapshots.

An instance that finds an orphan keeps holding the orphan's lock until the
user decides, so two Ochres started together cannot both recover it.

A clean exit deletes its session folder; saving or closing a document
deletes that document's snapshot. Whatever is left was work in progress
when something went wrong. (Kooka Pictura's frame_recovery.cpp is where the
shape of this comes from: the per-instance session, the "-Recovered" tabs,
and re-snapshotting recovered documents before their orphans are deleted.)

Snapshots are written by one worker thread, atomically, through the shared
helper (engine/atomic.py): to a temporary name in the same folder, fsynced,
then os.replace, so a crash mid-write leaves the previous snapshot intact.

Qt-free, like the controller, so all of it is testable headlessly.
"""

import concurrent.futures
import configparser
import io
import os
import shutil
import socket
import threading
import time
import uuid

try:
    import fcntl
except ImportError:                     # no flock: recovery stays off
    fcntl = None

from ..engine.atomic import atomic_write

LOCK = "lock"
UNRECOVERABLE = "unrecoverable"


def recovery_root(state_dir):
    return os.path.join(state_dir, "recovery") if state_dir else None


def recovered_name(name):
    """The tab name of a recovered document. Photoshop appends -Recovered;
    the extension goes, because a recovered document is an Ochre document
    whatever it was opened from."""
    base = os.path.splitext(name)[0] if name else ""
    return "%s-Recovered" % (base or "Untitled")


class _Incomplete(Exception):
    """A writer reported a torn result: abandon the temp file, keep the old."""


def _replace_atomically(path, write):
    """write(file object) through the shared atomic-write helper. True once
    path holds the new bytes; False -- with the previous file untouched --
    when write reports a torn result by returning False."""
    try:
        with atomic_write(path) as fh:
            if write(fh) is False:
                raise _Incomplete()
    except _Incomplete:
        return False
    return True


def _try_lock(path):
    """An open fd holding an exclusive lock on path, or None if someone else
    holds it (or it cannot be opened)."""
    if fcntl is None:
        return None
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _unlock(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    os.close(fd)


class RecoverableDocument:
    """One snapshot a session left behind."""

    def __init__(self, snapshot, name, path, saved, order=0):
        self.snapshot = snapshot        # the .ochre file
        self.name = name                # its tab name when it was written
        self.path = path                # the file it came from; "" if untitled
        self.saved = saved              # time.time() of the snapshot
        self.order = order              # its tab position, so tabs come back in order

    def __repr__(self):
        return "RecoverableDocument(%r, %r)" % (self.name, self.snapshot)


class OrphanSession:
    """The session folder of an Ochre that is no longer running, LOCKED by
    whoever found it until release() or discard()."""

    def __init__(self, directory, fd):
        self.directory = directory
        self._fd = fd
        self.documents = _read_documents(directory)

    def release(self):
        """Leave it for another time (and for another instance)."""
        _unlock(self._fd)
        self._fd = None

    def discard(self):
        """Delete the folder: its documents were recovered or refused."""
        shutil.rmtree(self.directory, ignore_errors=True)
        self.release()

    def quarantine(self, doc):
        """Move a snapshot that would not open out of the way, under
        recovery/unrecoverable/, where no scan offers it again. True when
        both files moved (or were already gone)."""
        root = os.path.dirname(self.directory)
        dest = os.path.join(root, UNRECOVERABLE, os.path.basename(self.directory))
        try:
            os.makedirs(dest, exist_ok=True)
            stem = os.path.splitext(doc.snapshot)[0]
            for src in (stem + ".ini", doc.snapshot):
                if os.path.exists(src):
                    os.replace(src, os.path.join(dest, os.path.basename(src)))
            return True
        except OSError:
            return False


def _read_documents(directory):
    out = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".ini"):
            continue
        stem = os.path.join(directory, name[:-4])
        snapshot = stem + ".ochre"
        if not os.path.isfile(snapshot):
            continue
        cp = configparser.ConfigParser(interpolation=None)
        try:
            cp.read(stem + ".ini", encoding="utf-8")
            info = dict(cp.items("Recovered")) if cp.has_section("Recovered") else {}
        except (OSError, configparser.Error):
            info = {}
        try:
            saved = float(info.get("saved", "")) if info.get("saved") else os.path.getmtime(snapshot)
        except (OSError, ValueError):
            saved = 0.0
        try:
            order = int(info.get("tab", "0"))
        except ValueError:
            order = 0
        out.append(RecoverableDocument(snapshot, info.get("name") or os.path.basename(snapshot),
                                       info.get("path", ""), saved, order))
    out.sort(key=lambda d: (d.order, d.saved))
    return out


class RecoveryStore:
    """This instance's session folder, and the worker that writes into it."""

    def __init__(self, state_dir):
        self.root = recovery_root(state_dir)
        self.directory = None
        self._fd = None
        self._pool = None
        self._pending = {}          # token -> Future of its latest write
        self._written = {}          # token -> what the latest COMPLETE write was of
        self._torn = set()          # tokens whose latest write saw pixels move
        self._lock = threading.Lock()

    @property
    def is_open(self):
        return self._fd is not None

    def open(self):
        """Create and lock this instance's session folder. False (and the
        store stays closed) when the state dir is not writable or the
        platform has no flock."""
        if self.is_open:
            return True
        if not self.root or fcntl is None:
            return False
        name = "%d-%d-%s" % (os.getpid(), int(time.time() * 1000), uuid.uuid4().hex[:6])
        # Built under a dot-name and renamed into place already locked: a
        # scan by another instance must never see a session folder whose
        # lock does not exist yet, or it would take it for a dead one.
        staging = os.path.join(self.root, "." + name)
        final = os.path.join(self.root, name)
        try:
            os.makedirs(staging)
            fd = _try_lock(os.path.join(staging, LOCK))
            if fd is None:
                shutil.rmtree(staging, ignore_errors=True)
                return False
            os.write(fd, ("pid = %d\nhost = %s\nstarted = %d\n"
                          % (os.getpid(), socket.gethostname(), int(time.time()))).encode("ascii", "replace"))
            os.rename(staging, final)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            return False
        self._fd = fd
        self.directory = final
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="ochre-autosave")
        return True

    def close(self):
        """Clean shutdown: delete the folder and drop the lock. A write still
        queued is cancelled; one already running is let finish first, so it
        cannot land in a folder that is being deleted."""
        if not self.is_open:
            return
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._pool = None
        shutil.rmtree(self.directory, ignore_errors=True)
        _unlock(self._fd)
        self._fd = None
        self._pending.clear()
        self._written.clear()

    # ---- this instance's snapshots ------------------------------------------

    def snapshot_path(self, token):
        return os.path.join(self.directory, token + ".ochre")

    def has(self, token):
        return token in self._pending or token in self._written

    def busy(self, token):
        """True while a write of token is queued or running."""
        with self._lock:
            return token in self._pending

    def written(self, token):
        """What the latest complete write of token was taken from (whatever
        the caller passed as `what`), or None."""
        return self._written.get(token)

    def was_torn(self, token):
        return token in self._torn

    def write(self, token, snapshot, name, path, what=None, tab=0):
        """Queue a snapshot (fileio.ochre_format.Snapshot, already taken) to
        be written as token's recovery copy. Returns the Future; its result
        is True once the copy is on disk and consistent."""
        if not self.is_open:
            return None
        directory = self.directory
        target = self.snapshot_path(token)
        info = os.path.join(directory, token + ".ini")

        def job():
            ok = _replace_atomically(target, snapshot.write)
            if ok:
                def write_info(fh):
                    cp = configparser.ConfigParser(interpolation=None)
                    cp["Recovered"] = {"Name": _one_line(name), "Path": _one_line(path or ""),
                                       "Saved": "%.3f" % time.time(), "Tab": str(int(tab))}
                    text = io.StringIO()
                    cp.write(text)
                    fh.write(text.getvalue().encode("utf-8"))
                ok = _replace_atomically(info, write_info)
            with self._lock:
                if ok:
                    self._written[token] = what
                    self._torn.discard(token)
                elif snapshot.torn:
                    self._torn.add(token)
            return ok

        with self._lock:
            future = self._pool.submit(job)
            self._pending[token] = future
        future.add_done_callback(lambda f, t=token: self._settled(t, f))
        return future

    def _settled(self, token, future):
        with self._lock:
            if self._pending.get(token) is future:
                del self._pending[token]

    def wait(self, token=None, timeout=None):
        """Block until token's write (or every write) has landed. True if
        nothing is left in flight."""
        with self._lock:
            futures = list(self._pending.values()) if token is None \
                else [f for t, f in self._pending.items() if t == token]
        if not futures:
            return True
        done, pending = concurrent.futures.wait(futures, timeout)
        return not pending

    def remove(self, token):
        """Delete token's recovery copy. Waits for a write in flight first:
        otherwise it would land after the delete and outlive the document."""
        if not self.is_open or token is None:
            return
        self.wait(token)
        with self._lock:
            self._written.pop(token, None)
            self._torn.discard(token)
        for ext in (".ochre", ".ini"):
            try:
                os.remove(os.path.join(self.directory, token + ext))
            except OSError:
                pass

    # ---- other instances' leftovers -----------------------------------------

    def orphans(self):
        """Sessions under the recovery folder whose owner is gone, each now
        locked by this instance. Empty ones are deleted on the spot."""
        if not self.root or fcntl is None or not os.path.isdir(self.root):
            return []
        out = []
        for name in sorted(os.listdir(self.root)):
            directory = os.path.join(self.root, name)
            if name == UNRECOVERABLE or directory == self.directory \
                    or not os.path.isdir(directory):
                continue
            if name.startswith("."):
                _sweep_staging(directory)
                continue
            fd = _try_lock(os.path.join(directory, LOCK))
            if fd is None:
                continue                # a live instance's, or unreadable: not ours
            orphan = OrphanSession(directory, fd)
            if not orphan.documents:
                orphan.discard()
                continue
            out.append(orphan)
        return out


def _sweep_staging(directory, age=3600):
    """A session folder still under its staging name belongs to an instance
    that died inside open(). Old and unlocked, it goes; young, it may be one
    being born this instant, and is left alone."""
    try:
        if time.time() - os.path.getmtime(directory) < age:
            return
    except OSError:
        return
    fd = _try_lock(os.path.join(directory, LOCK))
    if fd is not None:
        shutil.rmtree(directory, ignore_errors=True)
        _unlock(fd)


def _one_line(text):
    return str(text).replace("\r", " ").replace("\n", " ")
