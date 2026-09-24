# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""The blend-mode registry.

A deliberate split, because the INI-everything rule has a boundary here:
names, display labels, menu ordering and which modes are offered are
configuration and live in data/blendmodes.ini; the arithmetic is code and
lives in accel/fallback.py. Pretending a formula is a tunable would mean
either shipping an expression evaluator or lying about what is editable --
and the formulas must match the C extension bit-for-bit, which a
user-editable string cannot promise.

So: the INI may rename Multiply, reorder it, group it, or hide it. It may not
redefine what multiplying is.
"""

from . import accel


class BlendMode:
    __slots__ = ("name", "opcode", "label", "order", "group")

    def __init__(self, name, opcode, label=None, order=0, group=""):
        self.name = name
        self.opcode = opcode
        self.label = label or name
        self.order = order
        self.group = group

    def __repr__(self):
        return "BlendMode(%r, opcode=%d)" % (self.name, self.opcode)


class BlendRegistry:
    """Known blend modes, ordered for display."""

    def __init__(self, db=None):
        self._by_name = {}
        for name, opcode in accel.OPCODES.items():
            self._by_name[name] = BlendMode(name, opcode, order=opcode)
        if db is not None:
            self.load(db)

    def load(self, db):
        """Apply [BlendMode:Name] sections: labels, ordering, grouping.

        A section naming an unknown mode is ignored rather than invented --
        the arithmetic has to exist in code, so an INI cannot conjure a new
        blend mode into being. Unknown *keys* are ignored per the house rule.
        """
        for name in db.sections("BlendMode"):
            mode = self._by_name.get(name)
            if mode is None:
                continue
            section = "BlendMode:" + name
            mode.label = db.get(section, "Label", mode.label)
            mode.order = db.getint(section, "Order", mode.order)
            mode.group = db.get(section, "Group", mode.group)
            if not db.getbool(section, "Enabled", True):
                del self._by_name[name]
        return self

    # ---- lookup ----------------------------------------------------------

    def get(self, name):
        return self._by_name.get(name)

    def opcode(self, name):
        """Opcode for a mode name, falling back to Normal.

        Falling back rather than raising is deliberate: an unknown blend mode
        in a document file -- from a newer Ochre, or a corrupted manifest --
        should render the layer plainly, not refuse to open the file.
        """
        mode = self._by_name.get(name)
        return accel.NORMAL if mode is None else mode.opcode

    def names(self):
        return [m.name for m in self.ordered()]

    def ordered(self):
        return sorted(self._by_name.values(), key=lambda m: (m.order, m.name))

    def groups(self):
        """[(group, [modes])] in display order, for building a menu."""
        out = []
        for mode in self.ordered():
            if not out or out[-1][0] != mode.group:
                out.append((mode.group, []))
            out[-1][1].append(mode)
        return out

    def __contains__(self, name):
        return name in self._by_name

    def __len__(self):
        return len(self._by_name)

    def __iter__(self):
        return iter(self.ordered())


DEFAULT = BlendRegistry()
