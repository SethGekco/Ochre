# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Reading and writing documents."""

from .flat import (DEFAULT_TABLE, FormatTable, export_flat, export_frames,
                   flatten_onto, import_flat, import_indexed,
                   supported_extensions)
from .ochre_format import FORMAT_VERSION, is_ochre, load, save

__all__ = [
    "save", "load", "is_ochre", "FORMAT_VERSION",
    "import_flat", "import_indexed", "export_flat", "export_frames",
    "flatten_onto", "FormatTable", "DEFAULT_TABLE", "supported_extensions",
]
