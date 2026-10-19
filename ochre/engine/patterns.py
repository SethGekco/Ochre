# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Patterns: the built-in two-colour ones and the user's own (Edit > Define
Pattern), for pattern fill layers, Edit > Fill, Pattern Overlay and the
Pattern Stamp.

A user pattern is a PNG in <state dir>/patterns/. Wherever a pattern is
USED in a document (a fill layer, a layer style) its tile is embedded as
base64 PNG, so the document looks the same on a machine that never had
the pattern defined.
"""

import base64
import hashlib
import io
import os

import numpy as np

BUILTIN = ("checker", "stripes", "diagonal", "dots", "bricks", "grid", "noise")
_CACHE = {}


def user_dir(state_dir):
    return os.path.join(state_dir, "patterns") if state_dir else None


def user_names(state_dir):
    d = user_dir(state_dir)
    if not d or not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(".png"))


def names(state_dir=None):
    return list(BUILTIN) + [n for n in user_names(state_dir) if n not in BUILTIN]


def builtin_tile(name, size=8, fg=(0, 0, 0, 255), bg=(255, 255, 255, 255)):
    from .tools.retouch import pattern_tile
    return pattern_tile(name if name in BUILTIN else "checker", size, fg, bg)


def load_user(state_dir, name):
    from PIL import Image
    d = user_dir(state_dir)
    path = os.path.join(d, name + ".png") if d else None
    if not path or not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("RGBA")).copy()


def tile(name, state_dir=None, size=8, fg=(0, 0, 0, 255), bg=(255, 255, 255, 255)):
    """A pattern's tile, user patterns first."""
    if name not in BUILTIN:
        t = load_user(state_dir, name)
        if t is not None:
            return t
    return builtin_tile(name, size, fg, bg)


def save_user(state_dir, name, rgba):
    from PIL import Image
    d = user_dir(state_dir)
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip() or "pattern"
    path = os.path.join(d, safe + ".png")
    Image.fromarray(np.ascontiguousarray(rgba, np.uint8), "RGBA").save(path)
    return safe


def encode_tile(rgba):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba, np.uint8), "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_tile(text):
    if not text:
        return None
    key = hashlib.sha1(text.encode("ascii", "replace")).hexdigest()
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    from PIL import Image
    try:
        arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(text))).convert("RGBA")).copy()
    except (ValueError, OSError):
        return None
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = arr
    return arr


def scaled(t, scale):
    """The tile resized by scale (1.0 = as is)."""
    if abs(scale - 1.0) < 1e-6:
        return t
    from PIL import Image
    h, w = t.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return np.asarray(Image.fromarray(t, "RGBA").resize((nw, nh), Image.BILINEAR))


def fill(t, r, scale=1.0, origin=(0, 0)):
    """(r.h, r.w, 4) of the tile repeated, anchored at origin (document
    coordinates), so neighbouring fills join seamlessly."""
    t = scaled(t, scale)
    th, tw = t.shape[:2]
    ys = (np.arange(r.y, r.y + r.h) - origin[1]) % th
    xs = (np.arange(r.x, r.x + r.w) - origin[0]) % tw
    return t[ys[:, None], xs[None, :]]
