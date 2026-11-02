# qshp -- SHP sprites in any Qt program

A Qt image-format plugin that lets programs built on Qt open Westwood **SHP**
sprites (Tiberian Sun / Red Alert 2), plus a small command-line converter for
saving them. It is one file you drop into a program's `imageformats` folder.
The program needs no changes and does not have to know SHP exists.

It was written for [Kooka Pictura](https://github.com/KookaStudio/KookaPictura),
which opens every non-PSD, non-PDF image through Qt's image reader, and it
works the same way in other Qt image viewers and editors. It is a C++ port of
the SHP codec in [Ochre](https://github.com/SethGekco/Ochre)'s C&C addon and
is licensed GPL-3.0-or-later, like both.

## What you get

- **Open SHP** in the host program. A multi-frame SHP opens as a **sprite
  sheet**: frames left to right, then top to bottom, 8 per row when the frame
  count is a multiple of 8 (one row per facing set), otherwise a near-square
  grid. Index 0 is transparent.
- **Save SHP** with `qshp-convert`, because most editors only save the formats
  they list themselves. Export your edited sheet as PNG, then:

      qshp-convert unit.png unit.shp --palette unittem.pal --frames 24

  To go the other way, without opening an editor:

      qshp-convert unit.shp unit.png --palette unittem.pal

  Frames are cropped to their content, and each frame is stored RLE-Zero when
  that is smaller and raw otherwise, the same choice XCC makes.

## Installing

Copy `libqshp.so` (`qshp.dll` on Windows) into the program's Qt
`imageformats` folder:

| Program / package | Folder |
| --- | --- |
| Kooka Pictura `.deb` / `.rpm` | `/opt/kooka-pictura/plugins/imageformats/` (needs root; checked against the 0.3.0 `.deb`) |
| Programs using the system Qt 6 (Debian/Ubuntu) | `/usr/lib/x86_64-linux-gnu/qt6/plugins/imageformats/` |
| Read-only bundles (AppImage, Flatpak) | any folder `X/imageformats/`, then start the program with `QT_PLUGIN_PATH=X` |

In Kooka Pictura, choose **All files (\*)** in the Open dialog's type list
to see `.shp` files (its "All Formats" list names only the formats it knows),
or drag the file onto the window.

A plugin built against Qt 6.N loads in any Qt 6.M where M >= N, so the
release build uses an old Qt 6 on purpose.

## Choosing the palette

An SHP does not say which palette it uses. The plugin looks in this order:

1. the `QSHP_PALETTE` environment variable (a path to a `.pal`);
2. `Palette=` in `qshp.ini`, first next to the plugin, then in
   `~/.config/qshp/qshp.ini`;
3. `qshp.pal` next to the plugin, or the only `.pal` file there.

So the simplest setup is to copy your palette (for example `unittem.pal`)
into the same `imageformats` folder as the plugin. With no palette it shows
greys, so the sprite is still readable: index 0 transparent, the 16-31
remap ramp in red. Both 768-byte 6-bit VGA palettes and JASC-PAL text files
work.

`qshp.ini`:

    [SHP]
    Palette=unittem.pal     ; relative to this file's folder
    Frames=sheet            ; sheet | first | frames (one image per frame)
    Columns=0               ; sheet width in frames, 0 = automatic

`QSHP_FRAMES` and `QSHP_COLUMNS` override the last two. Images are delivered
as 8-bit indexed images whose colour table is the palette, so a program that
keeps indexed images keeps the exact palette indices.

Game palettes repeat colours. RA2's `unittem.pal` puts one colour on 36
indices, and gives 200-202 the same colours as entries in the 16-31 remap
ramp. Once a full-colour editor has the image, those indices can no longer be
told apart, and a careless conversion would turn ordinary pixels into house
colour. So both the plugin and `qshp-convert` give every index a colour of its
own, nudging the repeats by at most a few levels (3 of 255 for `unittem.pal`).
A sheet exported as full-colour PNG therefore converts back to exactly the
indices it was opened with. When you paint a genuinely new colour, it is
matched to the nearest palette entry, and on a tie an index outside the remap
ramp wins.

## Building

Needs CMake, a C++17 compiler and the Qt development package
(`qt6-base-dev` on Debian/Ubuntu; it falls back to Qt 5).

    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    # build/imageformats/libqshp.so and build/qshp-convert

## How it was checked

- **Decoding:** every TS/RA2 SHP in a 4,819-file collection of game and mod
  art, 4,810 valid files, decodes to exactly the same indices as Ochre's
  Python codec, frame by frame. The other 9 are older Tiberian Dawn / Red
  Alert 1 SHPs or damaged files, and both codecs refuse them.
- **Detection:** SHP has no real magic number (only a leading zero word), so
  the plugin validates the entire frame table before it claims a file. In
  testing it claimed no non-SHP file: 5,400 generated TGA, ICO, CUR, WBMP,
  BMP, GIF, PCX, TIFF, SGI, PPM, XBM and PNG files, plus 4,000 PNG and SVG
  files from a desktop install. Every one decoded exactly as it did without
  the plugin.
- **Round trip:** SHP to indexed PNG sheet to SHP, with RA2's `unittem.pal`,
  gives back identical frames for 199 of 199 sampled files. The same
  holds through a full-colour RGBA PNG, which is what an editor exports, for
  120 of 120 sampled files.

- **In Kooka Pictura 0.3.0:** with the plugin and `unittem.pal` dropped into
  the unpacked `.deb`'s `imageformats` folder, Kooka opened a 32-frame turret
  SHP as a 544x256 sheet whose pixels match `qshp-convert`'s exactly. Kooka's
  own PNG export of that document, converted back with `qshp-convert`, gave
  all 32 frames back unchanged.
- **Qt versions:** built against Qt 6.4 (Ubuntu's `qt6-base-dev`), the plugin
  loads in a separately bundled Qt 6.11.2, the same minor version Kooka
  Pictura ships. It decodes the same as the Qt 5 build there.

`tests/readcheck.cpp` is the tool used for these checks. It reads files the
way a program that knows nothing about the format does, and prints the
format Qt detected and an MD5 of every frame.
