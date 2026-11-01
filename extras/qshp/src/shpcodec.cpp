// qshp - Westwood SHP (Tiberian Sun / Red Alert 2) support for Qt.
// Copyright (C) 2026 Ochre contributors.
// Licensed under the GNU General Public License v3 or later.
//
// File layout, all little-endian (see Ochre's addons/cnc/cncshp.py for the
// long version and the real-file quirks behind each rule):
//
//   header   u16 zero, u16 width, u16 height, u16 frames
//   frame    u16 x, y, w, h; u32 flags; u8[4] radar; u32 reserved; u32 offset
//
// The low byte of flags is the compression: 0 and 1 raw, 2 raw scanlines
// each with a u16 length prefix, 3 RLE-Zero (per scanline: a u16 length that
// counts itself, then a non-zero byte is a literal index and 0 n is a run of
// n transparent pixels, n <= 255). Some writers put a size in the upper bits
// of flags, so only the low byte is read.

#include "shpcodec.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

namespace shp {

namespace {

constexpr int kMaxSide = 4096;
constexpr int kMaxFrames = 4096;

uint16_t u16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t u32(const uint8_t* p) { return uint32_t(p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t(p[3]) << 24)); }

void put16(std::vector<uint8_t>& out, unsigned v)
{
    out.push_back(uint8_t(v & 0xFF));
    out.push_back(uint8_t((v >> 8) & 0xFF));
}

void put32(std::vector<uint8_t>& out, uint32_t v)
{
    for (int i = 0; i < 4; ++i)
        out.push_back(uint8_t((v >> (8 * i)) & 0xFF));
}

// One RLE-Zero scanline into `row` (already zeroed). Returns the next offset.
size_t decodeRleLine(const uint8_t* data, size_t size, size_t pos, uint8_t* row, int width)
{
    if (pos + 2 > size)
        return size;
    const size_t length = u16(data + pos);
    if (length < 2)
        return pos + 2;
    const size_t end = std::min(pos + length, size);
    size_t i = pos + 2;
    int x = 0;
    while (i < end && x < width) {
        const uint8_t v = data[i];
        if (v) {
            row[x++] = v;
            ++i;
        } else {
            if (i + 1 >= end)
                break;
            x = std::min(width, x + data[i + 1]);
            i += 2;
        }
    }
    return pos + length;
}

// One length-prefixed raw scanline (compression 2).
size_t decodePrefixedLine(const uint8_t* data, size_t size, size_t pos, uint8_t* row, int width)
{
    if (pos + 2 > size)
        return size;
    const size_t length = std::max<size_t>(2, u16(data + pos));
    const size_t avail = std::min(pos + length, size) - (pos + 2);
    std::memcpy(row, data + pos + 2, std::min<size_t>(width, avail));
    return pos + length;
}

void encodeRleLine(const uint8_t* row, int w, std::vector<uint8_t>& out)
{
    const size_t start = out.size();
    put16(out, 0);
    int x = 0;
    while (x < w) {
        if (row[x] == 0) {
            int run = 0;
            while (x < w && row[x] == 0 && run < 255) {
                ++run;
                ++x;
            }
            out.push_back(0);
            out.push_back(uint8_t(run));
        } else {
            out.push_back(row[x++]);
        }
    }
    const size_t length = out.size() - start;
    out[start] = uint8_t(length & 0xFF);
    out[start + 1] = uint8_t(length >> 8);
}

} // namespace

size_t tableBytes(const uint8_t* header8)
{
    return kHeader + kFrameHeader * size_t(u16(header8 + 6));
}

bool sniff(const uint8_t* data, size_t avail, size_t size)
{
    if (avail < kHeader)
        return false;
    const int width = u16(data + 2), height = u16(data + 4), count = u16(data + 6);
    if (u16(data) != 0 || width < 1 || height < 1 || count < 1 || width > kMaxSide
        || height > kMaxSide || count > kMaxFrames)
        return false;
    const size_t table = tableBytes(data);
    if (avail < table || (size && size < table))
        return false;
    int drawn = 0;
    bool noData = true;
    for (int i = 0; i < count; ++i) {
        const uint8_t* f = data + kHeader + kFrameHeader * size_t(i);
        const int x = u16(f), y = u16(f + 2), w = u16(f + 4), h = u16(f + 6);
        const unsigned compression = u32(f + 8) & 0xFF;
        const uint32_t offset = u32(f + 20);
        noData = noData && offset == 0;
        if (w == 0 || h == 0)
            continue;                       // an empty frame; nothing else is read
        if (x + w > width || y + h > height || compression > 3 || offset < table)
            return false;
        if (size && (offset >= size || (compression <= 1 && offset + size_t(w) * h > size)))
            return false;
        ++drawn;
    }
    // Only files with two leading zero bytes get this far (ICO, TGA, WBMP and
    // the like), so an all-empty table -- which a stray header can fake -- is
    // claimed only in the exact shape real blank sprites have: nothing after
    // the table and every data offset 0.
    return drawn > 0 || (size == table && noData);
}

bool parse(const uint8_t* data, size_t size, Sprite& out, std::string* error)
{
    if (!sniff(data, size, size)) {
        if (error)
            *error = "not a Tiberian Sun / Red Alert 2 SHP";
        return false;
    }
    out.width = u16(data + 2);
    out.height = u16(data + 4);
    const int count = u16(data + 6);
    out.frames.assign(size_t(count), Frame{});
    for (int i = 0; i < count; ++i) {
        const uint8_t* f = data + kHeader + kFrameHeader * size_t(i);
        Frame& frame = out.frames[size_t(i)];
        frame.x = u16(f);
        frame.y = u16(f + 2);
        frame.w = u16(f + 4);
        frame.h = u16(f + 6);
        frame.flags = u32(f + 8);
        const uint32_t offset = u32(f + 20);
        if (frame.w == 0 || frame.h == 0 || offset == 0) {
            frame.w = frame.h = 0;
            continue;
        }
        frame.pixels.assign(size_t(frame.w) * frame.h, 0);
        const unsigned compression = frame.flags & 0xFF;
        size_t pos = offset;
        for (int y = 0; y < frame.h; ++y) {
            uint8_t* row = frame.pixels.data() + size_t(y) * frame.w;
            if (compression == 3)
                pos = decodeRleLine(data, size, pos, row, frame.w);
            else if (compression == 2)
                pos = decodePrefixedLine(data, size, pos, row, frame.w);
            else {
                const size_t take = pos < size ? std::min<size_t>(frame.w, size - pos) : 0;
                std::memcpy(row, data + pos, take);
                pos += size_t(frame.w);
            }
        }
    }
    return true;
}

std::vector<uint8_t> canvas(const Sprite& sprite, size_t i)
{
    std::vector<uint8_t> out(size_t(sprite.width) * sprite.height, 0);
    if (i >= sprite.frames.size())
        return out;
    const Frame& f = sprite.frames[i];
    for (int y = 0; y < f.h; ++y)
        std::memcpy(out.data() + size_t(f.y + y) * sprite.width + f.x,
                    f.pixels.data() + size_t(y) * f.w, size_t(f.w));
    return out;
}

std::vector<uint8_t> build(int width, int height, const std::vector<std::vector<uint8_t>>& canvases)
{
    std::vector<uint8_t> headers, body;
    const size_t count = canvases.size();
    const size_t base = kHeader + kFrameHeader * count;
    for (const auto& plane : canvases) {
        // Tight bounds of the non-transparent pixels.
        int x0 = width, y0 = height, x1 = -1, y1 = -1;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                if (plane[size_t(y) * width + x]) {
                    x0 = std::min(x0, x);
                    x1 = std::max(x1, x);
                    y0 = std::min(y0, y);
                    y1 = std::max(y1, y);
                }
        if (x1 < 0) {                               // empty: no data block
            for (int k = 0; k < 4; ++k)
                put16(headers, 0);
            put32(headers, 0);
            put32(headers, 0);
            put32(headers, 0);
            put32(headers, 0);
            continue;
        }
        const int w = x1 - x0 + 1, h = y1 - y0 + 1;
        std::vector<uint8_t> raw, rle;
        raw.reserve(size_t(w) * h);
        for (int y = y0; y <= y1; ++y) {
            const uint8_t* row = plane.data() + size_t(y) * width + x0;
            raw.insert(raw.end(), row, row + w);
            encodeRleLine(row, w, rle);
        }
        const bool useRle = rle.size() < raw.size();    // only when it wins, as XCC does
        const std::vector<uint8_t>& payload = useRle ? rle : raw;
        while ((base + body.size()) % 8)
            body.push_back(0);
        const uint32_t offset = uint32_t(base + body.size());
        body.insert(body.end(), payload.begin(), payload.end());

        // Radar colour: cosmetic; left black like most tools.
        put16(headers, unsigned(x0));
        put16(headers, unsigned(y0));
        put16(headers, unsigned(w));
        put16(headers, unsigned(h));
        put32(headers, useRle ? 3u : 0u);
        put32(headers, 0);
        put32(headers, 0);
        put32(headers, offset);
    }
    std::vector<uint8_t> out;
    out.reserve(base + body.size());
    put16(out, 0);
    put16(out, unsigned(width));
    put16(out, unsigned(height));
    put16(out, unsigned(count));
    out.insert(out.end(), headers.begin(), headers.end());
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

// ---- palettes -------------------------------------------------------------

bool readPalette(const std::vector<uint8_t>& bytes, Palette& out)
{
    static const char kJasc[] = "JASC-PAL";
    for (auto& c : out.argb)
        c = 0xFF000000u;
    if (bytes.size() >= 8 && std::memcmp(bytes.data(), kJasc, 8) == 0) {
        // Text: magic, version, count, then "r g b" per line, 8-bit values.
        std::istringstream text(std::string(bytes.begin(), bytes.end()));
        std::string line;
        int lineNo = 0, i = 0;
        while (std::getline(text, line) && i < 256) {
            if (lineNo++ < 3)
                continue;
            std::istringstream fields(line);
            int r, g, b;
            if (fields >> r >> g >> b)
                out.argb[i++] = 0xFF000000u | uint32_t((r & 0xFF) << 16) | uint32_t((g & 0xFF) << 8) | uint32_t(b & 0xFF);
        }
    } else {
        if (bytes.size() < 768)
            return false;
        // 6-bit VGA values: (v << 2) | (v >> 4), so 63 becomes 255, not 252.
        for (int i = 0; i < 256; ++i) {
            uint32_t c = 0xFF000000u;
            for (int k = 0; k < 3; ++k) {
                const unsigned v = bytes[size_t(i) * 3 + k] & 0x3F;
                c |= uint32_t((v << 2) | (v >> 4)) << (16 - 8 * k);
            }
            out.argb[i] = c;
        }
    }
    out.argb[0] &= 0x00FFFFFFu;                 // index 0 is transparent
    return true;
}

bool loadPalette(const std::string& path, Palette& out)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return false;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (!readPalette(bytes, out))
        return false;
    const size_t slash = path.find_last_of("/\\");
    out.name = slash == std::string::npos ? path : path.substr(slash + 1);
    return true;
}

Palette fallbackPalette()
{
    Palette p;
    p.name = "grey (no palette set)";
    for (int i = 0; i < 256; ++i)
        p.argb[i] = 0xFF000000u | uint32_t(i << 16) | uint32_t(i << 8) | uint32_t(i);
    p.argb[0] = 0x00FF00FFu;                    // transparent
    p.argb[1] = 0xFF282828u;                    // shadow
    for (int i = 16; i < 32; ++i)               // the remap ramp, as red
        p.argb[i] = 0xFF000000u | uint32_t((60 + (i - 16) * 12) << 16);
    return p;
}

Palette uniqueColours(const Palette& palette)
{
    Palette out = palette;
    std::vector<uint32_t> used;
    used.reserve(256);
    auto taken = [&](uint32_t rgb) { return std::find(used.begin(), used.end(), rgb) != used.end(); };
    for (int i = 0; i < 256; ++i) {
        const uint32_t rgb = out.argb[i] & 0xFFFFFFu;
        uint32_t pick = rgb;
        if (i == 0) {                   // transparent: keeps its colour
            used.push_back(pick);
            continue;
        }
        // The nearest free colour: every offset with all three channels
        // within r, for r = 1, 2, ..., fewest total levels first.
        const int r0 = int((rgb >> 16) & 0xFF), g0 = int((rgb >> 8) & 0xFF), b0 = int(rgb & 0xFF);
        for (int r = 1; taken(pick) && r < 256; ++r)
            for (int total = 1; taken(pick) && total <= 3 * r; ++total)
                for (int dr = -r; dr <= r && taken(pick); ++dr)
                    for (int dg = -r; dg <= r && taken(pick); ++dg)
                        for (int db = -r; db <= r && taken(pick); ++db) {
                            if (std::abs(dr) + std::abs(dg) + std::abs(db) != total
                                || std::max({std::abs(dr), std::abs(dg), std::abs(db)}) != r)
                                continue;
                            const int R = r0 + dr, G = g0 + dg, B = b0 + db;
                            if (R < 0 || R > 255 || G < 0 || G > 255 || B < 0 || B > 255)
                                continue;
                            const uint32_t c = uint32_t(R << 16 | G << 8 | B);
                            if (!taken(c))
                                pick = c;
                        }
        used.push_back(pick);
        out.argb[i] = (out.argb[i] & 0xFF000000u) | pick;
    }
    return out;
}

uint8_t nearest(const Palette& palette, uint32_t argb)
{
    const int r = (argb >> 16) & 0xFF, g = (argb >> 8) & 0xFF, b = argb & 0xFF;
    int best = 1;
    long bestD = -1;
    // The remap ramp (16-31) is searched last, so it only wins outright.
    for (int k = 1; k < 256; ++k) {
        const int i = k < 16 ? k : k < 240 ? k + 16 : k - 224;
        const uint32_t c = palette.argb[i];
        const long dr = long((c >> 16) & 0xFF) - r, dg = long((c >> 8) & 0xFF) - g, db = long(c & 0xFF) - b;
        const long d = dr * dr * 2 + dg * dg * 4 + db * db * 3;
        if (bestD < 0 || d < bestD) {
            bestD = d;
            best = i;
            if (d == 0)
                break;
        }
    }
    return uint8_t(best);
}

int autoColumns(size_t n)
{
    if (n >= 8 && n % 8 == 0)
        return 8;
    int c = 1;
    while (size_t(c) * size_t(c) < n)
        ++c;
    return c;
}

} // namespace shp
