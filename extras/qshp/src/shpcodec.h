// qshp - Westwood SHP (Tiberian Sun / Red Alert 2) support for Qt.
// Copyright (C) 2026 Ochre contributors.
// Licensed under the GNU General Public License v3 or later.
//
// The codec, with no Qt in it: parse, decode, encode, palettes, sheets.
// A port of Ochre's addons/cnc/cncshp.py and cncpal.py.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace shp {

// One frame as stored: a cropped rectangle of palette indices inside the
// shared canvas. An empty frame has w == 0 or h == 0 and no pixels.
struct Frame {
    int x = 0, y = 0, w = 0, h = 0;
    uint32_t flags = 0;
    std::vector<uint8_t> pixels;    // w * h indices, row-major
};

struct Sprite {
    int width = 0, height = 0;
    std::vector<Frame> frames;
};

// Is this a TS/RA2 SHP? The only magic is a leading zero word, so the whole
// frame table is validated: every rectangle inside the canvas, every data
// offset past the table and inside the file, a known compression. `size` is
// the full file size, or 0 when it is unknown (a sequential stream); `data`
// must hold at least the header and frame table.
bool sniff(const uint8_t* data, size_t avail, size_t size);

// Bytes needed to sniff: the header, then the header plus the frame table.
constexpr size_t kHeader = 8;
constexpr size_t kFrameHeader = 24;
size_t tableBytes(const uint8_t* header8);

// Decode a whole file. Lenient about truncated scanlines, as real files
// need; false only when it is not an SHP at all.
bool parse(const uint8_t* data, size_t size, Sprite& out, std::string* error = nullptr);

// Frame `i` placed on the full canvas (width * height indices, 0 elsewhere).
std::vector<uint8_t> canvas(const Sprite& sprite, size_t i);

// Serialise full-canvas index planes: each frame cropped to its content and
// stored RLE-Zero (compression 3) when that is smaller, raw (0) otherwise.
std::vector<uint8_t> build(int width, int height, const std::vector<std::vector<uint8_t>>& canvases);

// ---- palettes -------------------------------------------------------------

// 256 colours as 0xAARRGGBB. Index 0 is transparent (alpha 0).
struct Palette {
    uint32_t argb[256];
    std::string name;
};

// A 768-byte 6-bit VGA .pal or a JASC-PAL text file.
bool readPalette(const std::vector<uint8_t>& bytes, Palette& out);
bool loadPalette(const std::string& path, Palette& out);

// Used when no palette is configured: a grey ramp with the conventional
// special indices made visible (0 transparent, 16-31 red). The indices are
// what matter; the colours only make the sprite readable.
Palette fallbackPalette();

// The palette with every index given a colour of its own. Game palettes
// repeat colours (RA2's unittem.pal has one colour on 36 indices, and 200-202
// copy entries of the 16-31 remap ramp), so a full-colour editor cannot tell
// those indices apart once it has the image. Each repeat is nudged by the
// smallest step that makes it unique (a level or two, spread over the three
// channels, too little to see), so exporting full colour and converting back
// restores every index. Index 0 keeps its colour; it is transparent anyway.
Palette uniqueColours(const Palette& palette);

// Nearest palette index for a colour, 1..255 (index 0 is reserved for
// transparency, which the caller decides by alpha). Exact matches win; on a
// tie, an index outside the 16-31 remap ramp wins, so ordinary pixels never
// turn into house colour by accident.
uint8_t nearest(const Palette& palette, uint32_t argb);

// ---- sprite sheets ----------------------------------------------------------

// Columns for a sheet of n frames: 8 when n is a multiple of 8 (the facings
// of a unit), otherwise the smallest c with c * c >= n.
int autoColumns(size_t n);

} // namespace shp
