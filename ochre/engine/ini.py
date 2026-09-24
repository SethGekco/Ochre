# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""INI database: every tunable and every piece of content lives in data/*.ini.

Load order: data/*.ini sorted by filename, then data/addons/**/*.ini sorted.
Later files override earlier keys and may add new sections, so an addon can
extend a core table without editing it.

Section naming convention:
    [Type:Name]   content definitions   -- [BlendMode:Multiply], [Tool:Pencil]
    [Section]     singleton config      -- [Display], [History]

Nothing in this module knows what any section means. Callers ask for what
they want and supply the default, which is what makes the default the schema.
"""

import configparser
import os
import glob


class IniDB:
    """A merged view over a directory of INI files."""

    def __init__(self, data_dir=None):
        self.cp = configparser.ConfigParser(
            interpolation=None,
            strict=False,                 # duplicate sections merge, last wins
            delimiters=("=",),
            comment_prefixes=(";", "#"),
            inline_comment_prefixes=(";",),
        )
        self.cp.optionxform = str         # keys are case-sensitive
        self.loaded = []
        if data_dir:
            self.load_dir(data_dir)

    # ---- loading ---------------------------------------------------------

    def load_dir(self, data_dir):
        """Load data_dir/*.ini then data_dir/addons/**/*.ini, each sorted."""
        for path in sorted(glob.glob(os.path.join(data_dir, "*.ini"))):
            self.load_file(path)
        addons = os.path.join(data_dir, "addons")
        if os.path.isdir(addons):
            for path in sorted(glob.glob(os.path.join(addons, "**", "*.ini"),
                                         recursive=True)):
                self.load_file(path)

    def load_file(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                self.cp.read_file(f, source=path)
        except (OSError, configparser.Error):
            return False
        self.loaded.append(path)
        return True

    def load_string(self, text, source="<string>"):
        """Load INI text directly. Used by tests and by addon manifests."""
        self.cp.read_string(text, source=source)
        self.loaded.append(source)

    # ---- structure -------------------------------------------------------

    def sections(self, prefix=None):
        """Names of [prefix:Name] sections, or all section names if None.

        Returns just the Name part, in file order.
        """
        if prefix is None:
            return list(self.cp.sections())
        p = prefix + ":"
        return [s[len(p):] for s in self.cp.sections() if s.startswith(p)]

    def has(self, section, key=None):
        if key is None:
            return self.cp.has_section(section)
        return self.cp.has_option(section, key)

    def keys(self, section):
        return list(self.cp[section]) if self.cp.has_section(section) else []

    # ---- typed accessors -------------------------------------------------
    # All follow get*(section, key, default). An absent section, an absent
    # key, or a value that will not coerce all yield the default -- a bad
    # line in a data file must never crash the editor.

    def get(self, section, key, default=None):
        try:
            return self.cp.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def getint(self, section, key, default=0):
        raw = self.get(section, key)
        if raw is None:
            return default
        try:
            return int(raw.strip(), 0)     # base 0 accepts 0x1F and 0b101
        except ValueError:
            return default

    def getfloat(self, section, key, default=0.0):
        raw = self.get(section, key)
        if raw is None:
            return default
        try:
            return float(raw.strip())
        except ValueError:
            return default

    def getbool(self, section, key, default=False):
        raw = self.get(section, key)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    def getlist(self, section, key, default=None):
        raw = self.get(section, key)
        if raw is None:
            return list(default) if default else []
        return [p.strip() for p in raw.split(",") if p.strip()]

    def getints(self, section, key, default=None):
        out = []
        for part in self.getlist(section, key):
            try:
                out.append(int(part, 0))
            except ValueError:
                return list(default) if default else []
        return out or (list(default) if default else [])

    def getcolor(self, section, key, default=(0, 0, 0, 255)):
        """R,G,B or R,G,B,A -- each 0..255. Anything else yields the default."""
        vals = self.getints(section, key)
        if len(vals) == 3:
            vals = vals + [255]
        if len(vals) != 4 or any(not 0 <= v <= 255 for v in vals):
            return tuple(default)
        return tuple(vals)

    def getranges(self, section, key, default=None):
        """Parse "16-31,4,240-254" into [(16,31), (4,4), (240,254)].

        This is how a palette declares annotated index ranges. The core knows
        the syntax; it never learns what a range means.
        """
        raw = self.get(section, key)
        if raw is None:
            return list(default) if default else []
        out = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                if "-" in part[1:]:
                    lo, hi = part.split("-", 1)
                    lo, hi = int(lo, 0), int(hi, 0)
                else:
                    lo = hi = int(part, 0)
            except ValueError:
                return list(default) if default else []
            if lo > hi:
                lo, hi = hi, lo
            out.append((lo, hi))
        return out
