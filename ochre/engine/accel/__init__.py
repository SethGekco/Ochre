# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Backend selection: the optional C extension, or the numpy specification.

Resolution happens once, at import, and the chosen functions are bound at
module level so the hot path pays no dispatch cost.

Selection is PER FUNCTION, not all-or-nothing. The C extension implements
whichever operations are worth accelerating and the numpy specification
supplies the rest, so the extension can grow one function at a time without
a flag day, and a function can be reverted by simply not exporting it.

Setting OCHRE_ACCEL=0 forces the numpy path. That is not a debug escape
hatch: tests/run_all.sh runs the whole suite twice, once with each backend,
so the fallback is a first-class tested path rather than an untested
contingency for machines without a compiler. It is also how the parity test
proves the two agree.

A wrong-ABI or wrong-architecture .so simply fails to import and the fallback
takes over, so a stale build degrades in speed rather than breaking anything.
"""

import os

from . import fallback as _fallback

# Everything the backend may provide. Anything absent from the extension
# falls back silently.
OPS = ("blend_channel", "blend_rect", "composite_stack", "flood_fill")

_forced = os.environ.get("OCHRE_ACCEL", "").strip().lower()
_disabled = _forced in ("0", "off", "false", "no")

_native = None
if not _disabled:
    try:
        from . import _ochre_accel as _native   # built by setup.py, optional
    except ImportError:
        _native = None

HAVE_NATIVE = _native is not None

_bound = {}
for _op in OPS:
    _fn = getattr(_native, _op, None) if HAVE_NATIVE else None
    _bound[_op] = (_fn, "c") if _fn is not None else (getattr(_fallback, _op), "numpy")

blend_channel = _bound["blend_channel"][0]
blend_rect = _bound["blend_rect"][0]
composite_stack = _bound["composite_stack"][0]
flood_fill = _bound["flood_fill"][0]

#: Which backend actually serves each operation.
BACKENDS = {name: where for name, (_fn, where) in _bound.items()}
#: Overall label: "c" only when every operation is native.
BACKEND = "c" if HAVE_NATIVE and all(v == "c" for v in BACKENDS.values()) else (
    "mixed" if HAVE_NATIVE else "numpy")

# Opcodes are ABI, not implementation -- they always come from the spec.
from .fallback import (OPCODES, NORMAL, MULTIPLY, ADDITIVE, COLORBURN,      # noqa: E402
                       COLORDODGE, REFLECT, GLOW, OVERLAY, DIFFERENCE,
                       NEGATION, LIGHTEN, DARKEN, SCREEN, XOR)

#: The specification is always importable by name, so the parity test can
#: compare the two directly rather than through this selector.
spec = _fallback
native = _native


def describe():
    """One line for a status bar or a bug report."""
    if not HAVE_NATIVE:
        return "numpy (no C extension built)"
    accelerated = sorted(n for n, w in BACKENDS.items() if w == "c")
    return "c: %s" % (", ".join(accelerated) or "none")


__all__ = ["BACKEND", "BACKENDS", "HAVE_NATIVE", "spec", "native", "OPCODES",
           "OPS", "describe", "blend_channel", "blend_rect", "composite_stack",
           "flood_fill"]
