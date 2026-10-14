#!/usr/bin/env python3
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
"""Draw the application icon, using Ochre.

Deliberately built with the editor's own engine rather than in an image
program: three offset layers in pigment tones, composited by the real
compositor and written by the real exporter. So regenerating the icon is
also a small end-to-end check that layers, alpha and PNG export behave --
and it means the icon cannot drift out of sync with a repository that has
no binary assets to speak of.

    python3 packaging/make_icon.py            # writes assets/icon.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ochre.engine.document import Document
from ochre.engine.fileio.flat import export_flat
from ochre.engine.geometry import Rect

SIZE = 256

# Ochre the pigment: iron oxide, from pale yellow through to burnt red.
TONES = (
    (0xE8, 0xA8, 0x3A, 235),      # yellow ochre
    (0xC8, 0x6A, 0x28, 235),      # gold ochre
    (0x8E, 0x3A, 0x22, 240),      # burnt sienna
)


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "assets", "icon.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    doc = Document(SIZE, SIZE)
    side = int(SIZE * 0.47)
    step = int(SIZE * 0.13)
    # Stagger along the diagonal so the stack reads as layers even at 32
    # pixels, where internal detail would just be mud. The span is centred
    # with a real margin -- an icon drawn edge to edge looks cropped once a
    # desktop rounds its corners.
    span = side + step * (len(TONES) - 1)
    margin = (SIZE - span) // 2
    for i, colour in enumerate(TONES):
        layer = doc.add_layer("sheet %d" % (i + 1))
        x = margin + step * i
        y = margin + step * (len(TONES) - 1 - i)
        doc.cell(layer).fill(Rect(x, y, side, side), colour)


    export_flat(doc, out)
    print("wrote %s (%d bytes)" % (out, os.path.getsize(out)))


if __name__ == "__main__":
    main()
