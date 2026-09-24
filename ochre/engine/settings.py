# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Settings: the DEFAULTS dict below IS the schema.

Precedence, lowest to highest:

    DEFAULTS  <  data/*.ini  <  OCHRE_<KEY> environment variable

The type of the default drives coercion, so adding a setting is a one-line
change that gets typing, INI override and env override for free. A key that
is not in DEFAULTS is silently ignored wherever it appears -- that is the
forward-compatibility story: a newer Ochre's config file opens cleanly in an
older one, minus the settings it does not know about.

Nothing here is C&C-specific, and nothing here is UI-specific.
"""

import os

from .ini import IniDB


# section -> key -> default value (its type is the coercion rule)
DEFAULTS = {
    "Canvas": {
        "DefaultWidth": 800,
        "DefaultHeight": 600,
        "MaxDimension": 32768,
    },
    "Display": {
        "MaxDirtyRects": 8,        # above this, rects are merged pairwise
        "CheckerSize": 16,
        "CheckerLight": (200, 200, 200, 255),
        "CheckerDark": (160, 160, 160, 255),
    },
    "History": {
        "MaxBytes": 1073741824,    # 1 GiB of compressed deltas
        "MaxEntries": 200,
        "MinEntries": 10,          # never evict below this, whatever MaxBytes says
        "BlockSize": 256,          # stroke checkpoint block, px
        "CompressLevel": 1,        # zlib; runs on mouse-up where latency shows
        "MaxStrokeBlocks": 512,    # ~32 MB of live checkpoints, then auto-commit
    },
    "Frames": {
        "MaxWarmFrames": 8,
        "MaxWarmBytes": 1073741824,
        "PreviewMaxDim": 512,
        "ThawAheadCount": 2,
    },
    "Palette": {
        "SnapBits": 5,             # 5-5-5 snap LUT; 6-6-6 costs 1.5 s to build
    },
    "IndexLocked": {
        "EffectPolicy": "refuse",  # refuse | snap | unlock
    },
    "Effects": {
        "TileSize": 256,           # small tiles are GIL-bound and scale negatively
        "MaxThreads": 0,           # 0 = os.cpu_count()
    },
    "Planes": {
        # Plane names the core recognises. Addons append to this list, so a
        # typo'd plane name is an error rather than a silently orphaned array.
        "Known": "rgba,index,height",
    },
}


def _coerce(default, raw):
    """Coerce raw (a string) to the type of default. Returns default on failure."""
    if raw is None:
        return default
    if isinstance(raw, type(default)) and not isinstance(raw, str):
        return raw
    text = str(raw).strip()
    try:
        if isinstance(default, bool):
            return text.lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            return int(text, 0)
        if isinstance(default, float):
            return float(text)
        if isinstance(default, tuple):
            parts = [int(p.strip(), 0) for p in text.split(",") if p.strip()]
            if len(parts) == 3 and len(default) == 4:
                parts.append(255)
            if len(parts) != len(default):
                return default
            return tuple(parts)
        return text
    except ValueError:
        return default


class Settings:
    """Resolved configuration. Read-only after construction."""

    ENV_PREFIX = "OCHRE_"

    def __init__(self, data_dir=None, defaults=None, env=None, db=None):
        self._defaults = defaults if defaults is not None else DEFAULTS
        self._env = os.environ if env is None else env
        self._db = db if db is not None else IniDB(data_dir)
        self._cache = {}
        self._resolve()

    def _resolve(self):
        for section, keys in self._defaults.items():
            for key, default in keys.items():
                value = default
                raw = self._db.get(section, key)
                if raw is not None:
                    value = _coerce(default, raw)
                env_raw = self._env.get(self.ENV_PREFIX + key.upper())
                if env_raw is not None:
                    value = _coerce(default, env_raw)
                self._cache[(section, key)] = value

    # ---- access ----------------------------------------------------------

    def get(self, section, key):
        """Resolved value. Unknown section/key returns None, never raises."""
        return self._cache.get((section, key))

    def __getitem__(self, section):
        return {k: v for (s, k), v in self._cache.items() if s == section}

    def known(self, section, key):
        return (section, key) in self._cache

    def sections(self):
        return list(self._defaults)

    @property
    def db(self):
        """The underlying IniDB, for content tables the schema does not cover."""
        return self._db

    def as_dict(self):
        out = {}
        for (section, key), value in self._cache.items():
            out.setdefault(section, {})[key] = value
        return out
