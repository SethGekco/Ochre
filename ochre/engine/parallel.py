# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Row bands on a thread pool: compositing a big rect on several cores.

The compositor's per-pixel work reads and writes only its own pixel -- every
blend mode, masks, clipping, Blend If, knockout, pointwise adjustments -- so a
rect split into horizontal bands and composited band by band is the same
image, byte for byte, as the rect composited whole. The heavy lifting runs
in C (accel.blend_rect) or in large numpy operations, both of which release
the GIL, so the bands genuinely run at once. Kooka Pictura does the same with
rayon over rows (crates/pictura-render/src/composite_rows.rs).

Three rules keep it exact and safe; the compositor applies them:

  * The FIRST band runs alone on the calling thread, before the others. It
    is short, and it fills every cache the bands share -- a layer style's
    images, a shape's raster, a smart object's render -- so the bands then
    only read them.
  * Anything that reads beyond its own pixel (an adjustment layer that is not
    pointwise) is not banded at all.
  * A band never bands again: work submitted from a pool thread runs on that
    thread, so the pool cannot deadlock waiting on itself.

Effects use the same pool for their tiles when they declare
`parallel_tiles` (render_tile only reads what prepare() stored), which is
the promise effects.py was already written to.

Below MIN_PIXELS a rect is composited on the calling thread: a brush dab's
dirty rect is a few milliseconds of work, less than handing it out costs.
The thread count is the Performance/Threads setting, 0 meaning one per core
up to AUTO_CAP -- compositing is memory-bound, and past that more threads
mostly queue for the same memory while taking cores from everything else.

Measured on the development machine (24 cores, other work running), full
composite of 8000x6000 with four layers (Normal, Multiply, Screen, Overlay):

    threads        1        2       4       8      12      16
    C          1477 ms   728 ms  392 ms  206 ms  232 ms  194 ms
    numpy*     1896 ms     --    935 ms  682 ms    --      --
    (* 4000x3000: the numpy fallback is slower per pixel and gains less,
       its per-operation Python overhead holding the GIL)
"""

import concurrent.futures
import os
import threading

from .geometry import Rect

#: A rect smaller than this is composited on the calling thread.
MIN_PIXELS = 1 << 18
#: No band is smaller than this (bar the last), so dispatch stays negligible.
MIN_BAND_PIXELS = 1 << 16
#: Rows in the first band, the one that warms the shared caches.
WARM_ROWS = 8
#: The most threads "auto" uses.
AUTO_CAP = 8

_threads = 0                # resolved count; 0 = not configured yet
_settings = None            # engine settings, read on use so Preferences apply at once
_pool = None
_pool_size = 0
_lock = threading.Lock()
_local = threading.local()

#: How often work was actually handed to the pool -- so a test can prove the
#: banded path ran rather than quietly falling back to one thread.
stats = {"banded": 0}


def auto_threads():
    """One per core, at most AUTO_CAP."""
    return max(1, min(os.cpu_count() or 1, AUTO_CAP))


def _resolve(count):
    count = int(count or 0)
    return auto_threads() if count <= 0 else max(1, count)


def set_threads(count):
    """Use `count` threads (0 = auto); 1 turns banding off. Overrides the
    settings until configure() is called again."""
    global _threads, _settings
    _settings = None
    _threads = _resolve(count)
    return _threads


def configure(settings):
    """Follow Performance/Threads in engine settings from now on."""
    global _settings
    _settings = settings
    return threads()


def threads():
    """The thread count in force."""
    global _threads
    if _settings is not None:
        try:
            return _resolve(_settings.get("Performance", "Threads"))
        except (TypeError, ValueError):
            return auto_threads()
    if not _threads:
        _threads = _resolve(0)
    return _threads


def in_worker():
    """True on a pool thread, where nothing is handed out again."""
    return getattr(_local, "worker", False)


def bands(rect, min_pixels=None, align=1):
    """rect split into row bands for the pool, first a short warm-up band;
    None when it should simply run on the calling thread.

    align makes every band boundary a multiple of that many rows (from
    rect.y), for work done in blocks -- a box reduction.
    """
    n = threads()
    floor = MIN_PIXELS if min_pixels is None else min_pixels
    if n <= 1 or in_worker() or rect.w <= 0 or rect.area < floor:
        return None
    align = max(1, int(align))
    warm = min(rect.h, -(-WARM_ROWS // align) * align)
    rest = rect.h - warm
    if rest <= 0:
        return None
    rows = max(-(-rest // n), -(-MIN_BAND_PIXELS // rect.w))
    rows = -(-rows // align) * align
    out = [Rect(rect.x, rect.y, rect.w, warm)]
    y = rect.y + warm
    while y < rect.y1:
        h = min(rows, rect.y1 - y)
        out.append(Rect(rect.x, y, rect.w, h))
        y += h
    return out if len(out) > 2 else None


def _pool_for(n):
    global _pool, _pool_size
    with _lock:
        if _pool is None or _pool_size != n:
            old = _pool
            _pool = concurrent.futures.ThreadPoolExecutor(
                n, thread_name_prefix="ochre-band", initializer=_mark_worker)
            _pool_size = n
            if old is not None:
                old.shutdown(wait=False)
        return _pool


def _mark_worker():
    _local.worker = True


def run(work, jobs):
    """work(*job) for every job, the last on the calling thread and the rest
    on the pool; returns when all are done. The first exception, in job
    order, is raised."""
    jobs = list(jobs)
    if not jobs:
        return
    if len(jobs) == 1 or threads() <= 1 or in_worker():
        for job in jobs:
            work(*job)
        return
    pool = _pool_for(threads() - 1)
    stats["banded"] += 1
    futures = [pool.submit(work, *job) for job in jobs[:-1]]
    error = None
    # The calling thread is a band too while it works: anything it would
    # hand out now would queue behind the bands that are waiting for it.
    _local.worker = True
    try:
        work(*jobs[-1])
    except BaseException as exc:            # noqa: BLE001 -- re-raised below, in order
        error = exc
    finally:
        _local.worker = False
    for future in futures:
        exc = future.exception()
        if exc is not None:
            raise exc
    if error is not None:
        raise error


def banded(window, rect, work, min_pixels=None, align=1):
    """work(window_band, band_rect) over rect, where window covers rect:
    the warm-up band first, alone, then the rest at once."""
    split = bands(rect, min_pixels, align)
    if split is None:
        work(window, rect)
        return
    first = split[0]
    work(window[:first.h], first)
    run(work, [(window[b.y - rect.y:b.y1 - rect.y], b) for b in split[1:]])
