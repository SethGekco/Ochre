# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""The user's keyboard shortcuts: overrides on top of tools.ini and the
built-in menu keys, kept in <state dir>/shortcuts.ini. Qt-free, so the
controller can apply tool keys before any widget exists.

    [Tools]   tool name = key           (empty value = no key)
    [Menu]    Menu/Submenu/Action = key
"""

import configparser
import os


def path_for(state_dir):
    return os.path.join(state_dir, "shortcuts.ini") if state_dir else None


def load(state_dir):
    out = {"Tools": {}, "Menu": {}}
    p = path_for(state_dir)
    if not p or not os.path.isfile(p):
        return out
    cp = configparser.ConfigParser(interpolation=None, delimiters=("=",))
    cp.optionxform = str
    try:
        cp.read(p, encoding="utf-8")
    except (OSError, configparser.Error):
        return out
    for section in out:
        if cp.has_section(section):
            out[section] = dict(cp.items(section))
    return out


def save(state_dir, data):
    p = path_for(state_dir)
    if not p:
        return None
    cp = configparser.ConfigParser(interpolation=None, delimiters=("=",))
    cp.optionxform = str
    for section in ("Tools", "Menu"):
        cp.add_section(section)
        for k, v in sorted((data.get(section) or {}).items()):
            cp.set(section, k, v or "")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        cp.write(fh)
    return p


def apply_tools(registry, data):
    """Override tool keys in a ToolRegistry. An empty value removes the key."""
    shortcuts = getattr(registry, "_shortcuts", None)
    if shortcuts is None:
        return 0
    n = 0
    for name, key in (data.get("Tools") or {}).items():
        if name in registry:
            if key:
                shortcuts[name] = key
            else:
                shortcuts.pop(name, None)
            n += 1
    return n
