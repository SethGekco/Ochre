# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""addon.ini -- what an addon declares about itself before any code runs.

Same house rule as everywhere else: the DEFAULTS dict is the schema, the type
of each default drives coercion, and unknown keys are ignored. That last part
is the compatibility story -- a manifest written for a newer Ochre still
loads here, minus the fields this version has never heard of.

The manifest DECLARES and the code REGISTERS, and the host cross-checks the
two. That split is what makes it possible to show a user what an addon will
contribute before deciding whether to run it.
"""

import os

from ..engine.ini import IniDB

DEFAULTS = {
    "Id": "",
    "Name": "",
    "Version": "0.0.0",
    "Author": "",
    "License": "",
    "Description": "",
    "Entry": "__init__.py",
    "Api": "1.0",
    "Provides": "",
    "Requires": "",
    "Enabled": True,
    "Order": 100,
    # Reserved. Out-of-process isolation is the only thing that would
    # genuinely contain a native crash, and reserving the key now costs one
    # line while leaving the door open.
    "Isolation": "inprocess",
}


class Manifest:
    __slots__ = ("id", "name", "version", "author", "license", "description",
                 "entry", "api", "provides", "requires", "enabled", "order",
                 "isolation", "path")

    def __init__(self, **kw):
        for key in self.__slots__:
            setattr(self, key, kw.get(key))

    @staticmethod
    def read(path):
        db = IniDB()
        if not db.load_file(path):
            return None
        if not db.has("Addon"):
            return None

        def get(key):
            return db.get("Addon", key, DEFAULTS[key])

        ident = (get("Id") or "").strip()
        if not ident:
            # Fall back to the directory name, so a minimal manifest works.
            ident = os.path.basename(os.path.dirname(os.path.abspath(path)))
        return Manifest(
            id=ident,
            name=get("Name") or ident,
            version=get("Version"),
            author=get("Author"),
            license=get("License"),
            description=get("Description"),
            entry=get("Entry"),
            api=_version_tuple(get("Api")),
            provides=tuple(db.getlist("Addon", "Provides")),
            requires=tuple(db.getlist("Addon", "Requires")),
            enabled=db.getbool("Addon", "Enabled", True),
            order=db.getint("Addon", "Order", 100),
            isolation=get("Isolation"),
            path=path,
        )

    def __repr__(self):
        return "Manifest(%r, v%s, api=%r)" % (self.id, self.version, self.api)


def _version_tuple(text):
    parts = []
    for chunk in str(text or "").split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            break
    return tuple(parts) if parts else (1, 0)
