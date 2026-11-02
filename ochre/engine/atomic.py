# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Atomic file writes: a save lands whole or not at all.

Opening the target for writing truncates it first, so a crash, a kill, a
full disk or an encoder that raises halfway through leaves neither the old
file nor the new one. Every writer in Ochre therefore goes through here:

    with atomic_write(path) as fh:
        fh.write(data)              # or img.save(fh, ...), ZipFile(fh, "w")

The bytes go to a fresh temporary file in the SAME directory (so the final
rename never crosses a filesystem), which is flushed and fsynced and only
then renamed over the target with os.replace -- atomic on POSIX and on
Windows. Until that rename the previous good file is untouched; after it,
the new one is complete on disk. If the block raises, the temporary file is
removed, the target is not touched, and the exception propagates.

Details that matter:

  * The temporary name is random and created exclusively (mkstemp, which is
    O_CREAT|O_EXCL), so a symlink planted at a guessable name cannot
    redirect the write.
  * The existing file's permission bits carry over. A new file gets what
    open() would have given it (0666 less the umask); mkstemp's own 0600
    would otherwise quietly make every saved file private.
  * A symlinked target is followed: the file it points to is replaced and
    the link survives. A HARD link does not -- the target becomes a new
    inode. That is the price of never writing in place, and every atomic
    saver pays it.
  * Ownership is not copied (an ordinary user cannot chown to someone else
    anyway); a file saved over is owned by whoever saved it.
"""

import contextlib
import os
import stat
import tempfile

# Temporary files are hidden and say what they are, so one orphaned by a
# power cut is recognisable for what it is: ".<name>.<random>.ochre-tmp".
TEMP_SUFFIX = ".ochre-tmp"


@contextlib.contextmanager
def atomic_write(path, mode="wb", encoding=None, newline=None):
    """Context manager yielding a file object whose contents replace `path`
    atomically when the block exits cleanly. `mode` is "wb" or "w"."""
    if mode not in ("wb", "w"):
        raise ValueError("atomic_write writes whole files: mode 'wb' or 'w', not %r" % mode)
    if mode == "w" and encoding is None:
        encoding = "utf-8"
    target = os.path.realpath(os.fspath(path))
    folder = os.path.dirname(target) or os.curdir
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(target) + ".",
                               suffix=TEMP_SUFFIX, dir=folder)
    try:
        with os.fdopen(fd, mode, encoding=encoding, newline=newline) as fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        _set_mode(tmp, target)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _sync_folder(folder)


def write_bytes(path, data):
    """Replace `path` with `data` atomically."""
    with atomic_write(path, "wb") as fh:
        fh.write(data)
    return path


def write_text(path, text, encoding="utf-8"):
    """Replace `path` with `text` atomically."""
    with atomic_write(path, "w", encoding=encoding) as fh:
        fh.write(text)
    return path


def save_image(img, path, format=None, **params):
    """Pillow's img.save(path, ...), atomically. The format is chosen from
    the extension the way Pillow would choose it, unless one is named."""
    if format is None:
        from PIL import Image
        ext = os.path.splitext(os.fspath(path))[1].lower()
        format = Image.registered_extensions().get(ext)
        if format is None:
            raise ValueError("unknown file extension: %r" % (ext or path))
    with atomic_write(path) as fh:
        img.save(fh, format=format, **params)
    return path


def _set_mode(tmp, target):
    try:
        mode = stat.S_IMODE(os.stat(target).st_mode)
    except OSError:
        mode = 0o666 & ~_umask()
    # Some filesystems (FAT, many network mounts) refuse chmod; the save
    # itself matters more than the mode bits, so that is not a failure.
    with contextlib.suppress(OSError):
        os.chmod(tmp, mode)


def _umask():
    """The process umask, read without changing it where the OS allows."""
    try:
        with open("/proc/self/status", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    # Elsewhere the only way to read it is to set it; put it straight back.
    old = os.umask(0o022)
    os.umask(old)
    return old


def _sync_folder(folder):
    """fsync the directory so the rename itself survives a power cut.
    Best effort: not every platform can open a directory."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(folder, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)

