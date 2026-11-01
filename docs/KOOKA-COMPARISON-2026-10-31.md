# Ochre vs Kooka Pictura -- feature and implementation comparison

Date: 2026-10-31.

Compared:

- **Kooka Pictura**, https://github.com/KookaStudio/KookaPictura, shallow
  clone at commit `1666806` (2026-10-31). Rust engine (13 crates, about
  330k lines of Rust) under a Qt 6 C++ shell (about 235k lines of C++).
  GPL-3.0-or-later. Kooka paths below are relative to that repo's root.
- **Ochre**, this repository at `36d226e`. Python + numpy engine with an
  optional C accelerator, PySide6 UI, about 43k lines. GPL-3.0-or-later.
  Ochre paths below are relative to this repo's root.

Method: Kooka's `README.md`, `ROADMAP.md` (shipped `[x]` vs open `[ ]` vs
"Not planned") and `public_docs/` gave the inventory; every claim below was
then spot-checked in code on both sides with grep. An item that Kooka's
ROADMAP lists as open is treated as NOT shipped, even when some code for it
exists. Ochre's README was not trusted on its own either: each Ochre feature
named here was found in code (file and, where useful, line).

Ratings: **value** high / medium / low (to Ochre's stated goal of the
Paint.NET + GIMP + Photoshop feature sets), **effort** small / medium / large
(for Ochre, in Python, given its architecture).

Scope difference worth keeping in mind: Kooka is a photo editor whose north
star is faithful PSD/PSB and Photoshop CS6 behavior. Ochre is a general
layered raster editor whose distinctive strengths are indexed / palette-
locked art, frames, and an addon system. Many gaps in both directions follow
directly from that.

---

## 1. Kooka has, Ochre lacks

### 1.1 Faithful PSD/PSB round-trip -- value HIGH, effort LARGE

- Kooka: reads and writes Gray, RGB, CMYK, Lab, Indexed and Bitmap at
  8/16/32-bit; raw, RLE and ZIP(-prediction) channels; layer effects
  (`lfx2`), type layers (`TySh`), vector masks (`vmsk`), smart objects and
  smart filters, adjustment layers; and it **keeps every block it does not
  model** (color mode data, image resources, global mask info and unknown
  tagged blocks are stored verbatim and written back).
  `crates/pictura-codec/src/read.rs` (comments at lines 199, 204, 642),
  `crates/pictura-codec/src/write.rs`, `write_layers.rs` (line 709: an
  adjustment payload is written back as read), `read/decode.rs`.
- Ochre (partial): `ochre/engine/fileio/psd.py`. Reads 8/16-bit RGB and
  grey only (16-bit reduced to 8 with `>> 8`, a truncation rather than a
  rounded rescale); raises on 32-bit; CMYK/Lab, adjustment, smart-object and
  effect layers fall back to the flattened composite. Writes 8-bit RGB
  only; adjustment layers and live styles become warnings. Unknown blocks
  are dropped, so open-then-save loses data the user never touched.
- A cheaper first step (value high, effort medium): stash the image
  resources section and every unrecognised per-layer tagged block in
  `layer.meta` / `doc.meta` as bytes and write them back unchanged. That is
  exactly Kooka's "preserve, don't model" strategy, and it is what makes
  their round-trip safe before they model a block.

### 1.2 High bit depth (16-bit and 32-bit float, HDR) -- value MEDIUM, effort LARGE

- Kooka: document modes at 8/16/32 bpc, a native-depth compositing branch
  (`crates/pictura-render/src/composite_native.rs`), HDR Toning with all
  four methods, mode conversions in `crates/pictura-render/src/document_ops/mode/`.
- Ochre: none. Pixels are uint8 RGBA by design (`docs/DESIGN.md`, "Pixel
  storage", and the integer-arithmetic accelerator contract). Adding depth
  breaks that contract, which is why this is large. Medium value: it
  matters for photo work (banding after stacked adjustments), not for
  sprite / palette work.

### 1.3 CMYK, Lab, Duotone and Bitmap document modes -- value MEDIUM, effort LARGE

- Kooka: `crates/pictura-render/src/document_ops/mode/`, CMYK through the
  working ICC profile (`crates/pictura-color/src/`). Duotone and
  Multichannel conversion are still open (#553, #554).
- Ochre (partial): soft proof, gamut warning and CMYK *export* through a
  profile (`ochre/engine/colormanage.py`, `ochre/ui/exportdialog.py`,
  CMYK JPEG/TIFF only). There is no CMYK or Lab working document.

### 1.4 Metadata: EXIF / IPTC / XMP, File Info, templates -- value HIGH, effort SMALL-MEDIUM

- Kooka: `crates/pictura-codec/src/exif.rs`, `iptc.rs`, `xmp.rs`,
  `crates/pictura-codec/src/raster_metadata/` (re-embeds EXIF, IPTC-IIM and
  XMP on JPEG and TIFF Save As), `crates/pictura-codec/src/metadata/template.rs`
  (XMP templates); six editable IPTC fields in File Info.
- Ochre: none. `ochre/engine/fileio/flat.py` keeps only the ICC profile;
  EXIF, XMP and IPTC are dropped on import and never written.
- **Related Ochre bug, independent of Kooka:** `flat.py` does not apply the
  EXIF Orientation tag on import (no `ImageOps.exif_transpose`), so most
  phone photos open rotated or mirrored. That fix is about five lines.
  (Kooka's own handling of orientation was not verified; Qt's image reader
  may do it for them.)
- Effort note: Pillow already reads `img.getexif()` and `img.info["xmp"]`
  and can write `exif=` on JPEG, PNG, WebP and TIFF. Keep the bytes in
  `doc.meta`, offer "strip GPS / strip all" on export (privacy), and a small
  File Info dialog. That is a small-to-medium job.

### 1.5 Command search (Shift+Space) -- value HIGH, effort SMALL

- Kooka: `crates/pictura-app/cpp/search_bar.cpp` (popup),
  `search_index.cpp` (224 lines: index and ranking), `frame_search.cpp`
  (what gets indexed). Finds any command, tool, panel, Preferences page or
  key panel control, shows its menu path ("Edit > Transform > Warp"), runs
  it on Enter, reveals it on Alt+Enter, and greys out entries that cannot run
  now.
- Ochre: none. Ochre has about 185 static menu actions
  (`ochre/ui/mainwindow.py`), about 110 generated effect entries and 85
  tools (`data/tools.ini`), so discoverability is a real problem.
- Port: walk `menuBar()` recursively collecting `QAction`s with their path,
  add tools from the registry, rank with Kooka's five tiers (see 2.12).
  One file, no engine change.

### 1.6 Headless mode, control server, MCP / agent control -- value HIGH, effort MEDIUM

- Kooka: `--headless` (`crates/pictura-app/cpp/main.cpp`), a Unix-socket
  JSON control server (`control_server.cpp`, 805 lines), an MCP stdio
  bridge with 27 tools (`crates/pictura-mcp/src/tools.rs`: status, document,
  get_pixel, screenshot, layer tree, filters, adjustments, run_command,
  selection, layer ops, tool pointer input, history, export_png),
  a private headless instance via `--mcp --spawn --tier --root`
  (`crates/pictura-mcp/src/spawn.rs`), and Preferences > Agent Control with
  permission tiers, allowed folders and approval prompts
  (`agent_control.cpp`, `preferences_agent_page.cpp`, `client_registration.cpp`).
  User doc: `public_docs/agent-control.md`.
- Ochre: no CLI beyond opening files (`ochre/ui/app.py`). But Ochre is
  unusually well placed: `ochre/ui/controller.py` is Qt-free (about 275
  methods) and already has a `RECORDABLE` whitelist (`controller.py:4598`)
  used to make action files safe to replay. That whitelist is most of an
  MCP tool surface already. See 2.11 for the permission model to copy.

### 1.7 Autosave and crash recovery -- value HIGH, effort SMALL-MEDIUM

- Kooka: `crates/pictura-app/cpp/frame_recovery.cpp` (248 lines) and
  `recovery_store.cpp`. A timer (5/10/15/30/60 min, `session.h`) snapshots
  every dirty document into a per-instance session directory guarded by a
  `QLockFile`; save or close drops the snapshot; a clean exit deletes the
  session; a launch that finds a session whose owner died offers the
  documents back as unsaved "-Recovered" tabs. Careful edge cases: skip a
  mid-stroke tick, wait for an in-flight write before deleting, keep the
  whole session if any snapshot could not be moved, re-snapshot recovered
  documents before deleting the orphan copies.
- Ochre: none. `ochre/ui/session.py` remembers only the last canvas size.
- Port: the `.ochre` writer already exists; add a timer, a lock file with
  the PID, and a startup scan. Pair it with 2.3 so a tick does not
  re-encode untouched layers.

### 1.8 Save for Web indexed export (GIF / PNG-8) -- value MEDIUM, effort SMALL

- Kooka: `crates/pictura-app/cpp/save_for_web_dialog.cpp`,
  `crates/pictura-codec/src/web/quantize.rs` (median cut, web-safe,
  black-and-white and grey tables, diffusion / pattern / noise dither,
  transparency with a matte), `web/gif.rs`. 2-Up/4-Up and the colour-table
  editor are still open (#629).
- Ochre (partial): `ochre/ui/exportdialog.py` already encodes live and shows
  the real file size and a preview, which is better than most. What is
  missing is indexed output: colour count, palette method, dither, matte.
  The engine already has the pieces (`ochre/engine/palettefiles.py`
  `median_cut`, `ochre/engine/quantize.py` Bayer and Floyd-Steinberg with
  protected indices), so this is wiring, not algorithms.

### 1.9 Pictura Raw panels beyond Basic -- value MEDIUM, effort MEDIUM

- Kooka: `crates/pictura-adjust/src/pictura_raw/` -- `basic.rs`,
  `tone_curve.rs`, `detail.rs`, `hsl.rs` (HSL / Grayscale), `split_toning.rs`,
  `lens.rs`, `effects.rs`, `calibration.rs`; dialog in
  `crates/pictura-app/cpp/pictura_raw_dialog.cpp`. Presets tab is a stub.
- Ochre (partial): `ochre/engine/photo.py` `CameraRaw` (line 131) is the
  Basic panel plus vignette and grain.
- Neither editor opens camera raw files (Kooka #400: no container reader,
  no demosaic). So this is the "Camera Raw Filter" on decoded images only.

### 1.10 Lens Correction with real lens profiles -- value MEDIUM, effort MEDIUM

- Kooka: `crates/pictura-filters/src/distort/lens_correction/lensfun.rs`
  reads the system lensfun database at runtime (none bundled), plus
  `profile.rs`, `catalog.rs`, a camera list and EXIF focal-length prefill;
  sub-profiles by focal length, f-stop and focus distance.
- Ochre (partial): `ochre/engine/photo.py` `LensCorrection` (line 224), the
  Custom tab only (manual sliders).
- Depends on 1.4 (needs EXIF to pick the lens). Note that the lensfun data
  is not installed on this machine (`/usr/share/lensfun` is absent), so it
  would be an optional dependency. Kooka's camera list is separately
  MIT-licensed (`LICENSES/CameraDatabase-MIT.txt`).

### 1.11 The CS6 Filter Gallery families -- value MEDIUM, effort MEDIUM (each filter small)

- Kooka (`crates/pictura-filters/src/catalogue/rows/*.rs`, engines under
  `crates/pictura-filters/src/`), filters Ochre has no equivalent for:
  - Sketch (14): Bas Relief, Chalk & Charcoal, Charcoal, Chrome, Conte
    Crayon, Graphic Pen, Halftone Pattern, Note Paper, Photocopy, Plaster,
    Reticulation, Stamp, Torn Edges, Water Paper.
  - Brush Strokes (8): Accented Edges, Angled Strokes, Crosshatch, Dark
    Strokes, Ink Outlines, Spatter, Sprayed Strokes, Sumi-e.
  - Texture (5 of 6; Ochre has Canvas Texture): Craquelure, Grain, Mosaic
    Tiles, Patchwork, Stained Glass (Texturizer overlaps Canvas Texture).
  - Artistic (14 of 15; Ochre has Neon): Colored Pencil, Cutout, Dry Brush, Film
    Grain, Fresco, Paint Daubs, Palette Knife, Plastic Wrap, Poster Edges,
    Rough Pastels, Smudge Stick, Sponge, Underpainting, Watercolor.
  - Others: Diffuse Glow, Glass, Ocean Ripple, Glowing Edges, Smart Blur,
    Shape Blur, Average, Path Blur, Spin Blur, Sharpen Edges / Sharpen More,
    Picture Frame, NTSC Colors, Embed / Read Watermark.
- Ochre has about 75 filter classes plus about 35 adjustments
  (`ochre/engine/filters.py`, `morefilters.py`, `photo.py`, `effects.py`,
  `colors.py`), with its own extras Kooka lacks (Julia / Mandelbrot,
  Voronoi, Kaleidoscope, Tile Reflection, Cartoon, Polar Inversion,
  Shake Reduction, Soften Portrait, Color to Alpha, ...).
- Each Kooka filter is a self-contained kernel plus a declarative
  `FilterSpec` row, which maps cleanly onto Ochre's property-declared
  effects. Port a family at a time; Sketch and Brush Strokes are the most
  visible gaps in Ochre's Filter Gallery (`ochre/ui/gallery.py`).

### 1.12 Typography -- value MEDIUM, effort LARGE

- Kooka: rustybuzz shaping, swash rasterising, unicode-bidi; per-run
  character styling, Character / Paragraph / Glyphs / Character Styles /
  Paragraph Styles panels, type on path and area type (#683), warp text
  (#684), missing-font handling and missing-glyph protection.
  `crates/pictura-render/src/type_layer.rs`, `crates/pictura-core/src/type_tool/`,
  `crates/pictura-core/src/text_render.rs`.
- Ochre (partial): `ochre/engine/textlayer.py` and `text.py`. One font,
  size and colour per layer, Pillow `ImageFont`, paragraph wrap by box
  width, eight warp styles, vertical text, type masks
  (`tests/test_typecurve.py`). No per-character runs, tracking/kerning
  controls, styles panels or type on a path.

### 1.13 Actions: `.atn` read/write and a full Actions engine -- value LOW-MEDIUM, effort MEDIUM

- Kooka: exact version-16 `.atn` read/write
  (`crates/pictura-codec/src/action_file.rs`, `action_descriptor.rs`), a
  command catalogue that drives the menus (`crates/pictura-script/src/command_catalogue.rs`),
  playback with conditional steps, Button mode, tool-stroke recording,
  atomic panel persistence (`crates/pictura-app/cpp/panels/actions_panel.cpp`).
- Ochre (partial): `controller.py` `start_recording` / `play_action` /
  `batch` (lines 1720-1800) store steps as JSON in `actions.json`; replay is
  one undo step and limited to `RECORDABLE`. That is a sound design. `.atn`
  interop has limited payoff because real-world actions reference
  Photoshop descriptors that only map where Ochre has the same command.

### 1.14 Smaller Kooka-only items -- each value LOW, effort SMALL unless noted

- PDF / EPS import through Ghostscript: `crates/pictura-codec/src/ghostscript.rs`,
  `eps.rs`, `pdf/`. Ochre exports PDF but cannot open one.
- SVG export (raster in SVG): `crates/pictura-codec/src/svg.rs`.
- Print dialog with placement and scale, printer-managed colour and Save as
  PDF: `crates/pictura-app/cpp/print_dialog.cpp` (423 lines). Ochre's
  `do_print` (`ochre/ui/mainwindow.py:541`) fits the merged image to the page.
- Refine Edge "Decontaminate Colors" (value MEDIUM):
  `crates/pictura-select/src/refine.rs` `decontaminate`. Ochre's
  `ochre/engine/refine.py` has a guided-filter refine but no colour
  decontamination of soft edges.
- Spot Healing "Create Texture": a Laplace fill plus noise matched to the
  roughness of the surrounding ring, fixed seed
  (`crates/pictura-paint/src/healing/mod.rs`). Ochre's `harmonic_fill`
  (`ochre/engine/inpaint.py`) is the same Laplace fill without the matched
  grain, which looks plastic on noisy photos. A few lines to add.
- Live UI scale 75-200 % without restart (`crates/pictura-app/cpp/theme.cpp`),
  document tab tear-out, named workspaces, Help > System Info / Submit
  Issue (`system_info.cpp`), opt-in performance metrics
  (`crates/pictura-render/src/metrics.rs`).
- Panels Ochre lacks: Character, Paragraph, Glyphs, Notes list, Clone
  Source, Properties, Adjustments icon grid. (Ochre has Layers, Channels,
  Paths, History, Colors, Palette, Info, Histogram, Navigator, Actions,
  Layer Comps, Brush Presets, Frames: `ochre/ui/docks.py`,
  `ochre/ui/panels.py`.)

---

## 2. Both have it; Kooka does it better or faster (or not -- said honestly)

### 2.1 Allocation budget for untrusted files -- Kooka better. Port: value HIGH, effort SMALL

- Kooka: `crates/pictura-codec/src/budget.rs` and `probe.rs`. Every plane
  is charged before it is allocated: one allocation may not exceed 2 GiB,
  width/height may not exceed 30,000, and one document may not claim more
  than three quarters of `MemAvailable` from `/proc/meminfo`. Decompression
  goes through `read_bounded` (read at most limit+1 bytes, refuse if more).
  A refusal is a typed error, never an allocator abort. Header probing
  happens before decode.
- Ochre: `ochre/engine/fileio/psd.py:153` calls `zlib.decompress` with no
  output bound (a zip bomb inside a PSD channel is unbounded), and
  `np.zeros((h, w, 4))` is sized straight from header fields. The `.ochre`
  manifest `Width` / `Height` (`ochre_format.py`, load path) and the C&C
  SHP headers are also uncapped. Pillow's `MAX_IMAGE_PIXELS` is left at its
  default.
- Port: one `budget.py` with `check_dims(w, h)` and
  `bounded_decompress(data, limit)` (`zlib.decompressobj().decompress(data, limit)`
  then check `unconsumed_tail`), called from psd.py, ora.py, ochre_format.py
  and the addon codecs; set `Image.MAX_IMAGE_PIXELS` from the same budget.

### 2.2 Atomic file writes -- Kooka better. Port: value HIGH, effort SMALL

- Kooka: `QSaveFile` (write to a temp file, commit by rename) in
  `crates/pictura-app/cpp/recovery_store.cpp`, and atomic persistence of
  the Actions palette.
- Ochre: `zipfile.ZipFile(path, "w")` truncates the existing file before
  writing (`ochre/engine/fileio/ochre_format.py:256`, `ora.py:166`), and
  `actions.json` / `brushes.json` are rewritten in place
  (`ochre/ui/controller.py` around lines 1690 and 1760). A crash, kill or
  full disk mid-save destroys the previous good copy. (This machine's root
  disk has filled up before and produced 0-byte writes.) No `os.replace`
  appears anywhere in `ochre/` or `addons/`.
- Port: a shared `atomic_write(path)` context manager: write
  `path + ".tmp-<pid>"` in the same directory, flush, `os.fsync`,
  `os.replace`. Use it in every saver, including addon savers via the host
  API.

### 2.3 Encoded-stream cache on save -- Kooka better. Port: value MEDIUM, effort SMALL

- Kooka: `crates/pictura-codec/src/write_cache.rs` caches each encoded
  layer-channel stream keyed by the plane's content identity plus every
  input that changes the bytes; a reused stream is byte-identical to a fresh
  encode; the cache is pruned to the keys the latest save used.
- Ochre: every save re-encodes every plane PNG (`ochre_format.py` builds
  `blobs` from scratch; `docs/DESIGN.md` measures 178 ms per 4000x4000
  layer). Ochre already has the key: `Surface.revision`
  (`ochre/engine/surface.py:78`, bumped on every write). Cache
  `(surface id, plane, revision) -> png bytes`. Makes Ctrl+S on a 20-layer
  document near-instant after a one-layer edit, and makes autosave (1.7)
  cheap.

### 2.4 Reduced-resolution previews when zoomed out -- Kooka better. Port: value MEDIUM-HIGH, effort MEDIUM

- Kooka: `crates/pictura-render/src/render_session.rs` renders interactive
  previews at a power-of-two "presented level" of the visible region,
  reusing a cached backdrop of the stack below the edited layer keyed by a
  prefix digest, under a byte budget. `crates/pictura-render/src/filter_proxy.rs`
  box-averages the visible region down to about the pane's device pixels
  for filter-gallery previews ("a preview costs what the pane shows rather
  than what the document holds").
- Ochre: `ochre/ui/effectdialog.py` already previews only the visible
  region inflated by the effect halo, on a thread, and cancels in-flight
  renders when a slider moves -- that is as good as Kooka when zoomed in.
  But there is no reduced level: at Fit to Window on a large canvas the
  visible region is the whole canvas at full resolution. The same applies
  to whole-layer property scrubbing (opacity, blend mode).
- Caveat Kooka itself records: radius-based filters need their pixel
  parameters scaled with the level, and things not reduced (live type,
  vector masks, effect sizes) make a proxy preview approximate.

### 2.5 Row-band parallel CPU kernels -- Kooka better. Port: value MEDIUM, effort MEDIUM

- Kooka: `crates/pictura-render/src/composite_rows.rs` runs per-pixel
  compositing rows on rayon; effect kernels and the knockout cover canvas
  stay sequential, so output is byte-identical to the serial oracle (#1011,
  also adjustments, effect mattes, clipping and the display transform).
- Ochre: the C accelerator releases the GIL (`ochre/engine/accel/_ochre_accel.c`,
  two `Py_BEGIN_ALLOW_THREADS` sites) and large numpy ops release it too,
  but compositing runs on one thread. Split a dirty rect into horizontal
  bands over a `concurrent.futures.ThreadPoolExecutor`; keep knockout and
  effects serial like Kooka; add a serial-vs-banded byte-equality check
  next to the existing accel parity tests. Ochre's integer contract makes
  the equality exact rather than approximate.
- Note: the accelerator today only implements `blend_channel` and
  `blend_rect` (369 lines). `docs/DESIGN.md` also lists `composite_stack`,
  `brush_dab`, `flood_fill`, `box_blur_pass`, `rasterize_polygon` for C;
  those are not built.

### 2.6 PSD decode speed -- Kooka better. Port: value MEDIUM, effort SMALL

- Kooka: `crates/pictura-codec/src/read/decode.rs` decodes layer channels,
  merged RLE rows and ZIP-prediction rows in parallel, byte-identical to
  the serial decode with the same first error (#1019).
- Ochre: `packbits_decode` (`ochre/engine/fileio/psd.py:46`) is a
  pure-Python loop per packet, per row, per channel -- the slowest possible
  path for the most common PSD compression. Move it into the C accelerator
  (with the numpy version as the spec, per Ochre's own rule), or use
  Pillow's built-in PackBits decoder, and decode channels on a thread pool.

### 2.7 GPU compute path -- Kooka has it; not worth porting now. Value LOW, effort LARGE

- Kooka: `crates/pictura-render/src/gpu/` (wgpu / Vulkan) for compositing,
  the heavy filters and the live paint stroke. Gated by a one-time
  comparison of a representative scene against the CPU compositor
  (`gpu/mod.rs`, +-1 LSB), with per-mode and per-adjustment fallback to
  CPU, and `crates/pictura-render/src/composite_route.rs` classifying each
  layer's content path with an exhaustive `match` so a new CPU path cannot
  ship without a GPU decision.
- Ochre: none. Python would need wgpu-py or moderngl, and Ochre's
  bit-exact integer contract would have to relax to Kooka's +-1 LSB. The
  ideas worth remembering if Ochre ever does it: the runtime parity gate,
  and the exhaustive route enum.

### 2.8 Dirty-region accounting -- roughly equal

- Kooka: `crates/pictura-render/src/canvas_damage.rs` ("any change nobody
  described is a change to the whole canvas": forgetting to describe a
  change costs speed, never a stale picture) and `damage.rs` (a layer's
  reach = content box + effect reach + mask extent + the clipped run above
  it; `None` means unprovable, so full recomposite).
- Ochre: writes go through `Surface`, which marks dirty itself
  (`ochre/engine/surface.py`), and the compositor inflates by
  `layerstyles.reach(styles)` (`ochre/engine/compositor.py:444`). For pixel
  writes Ochre is as safe. The one idea to take: property-only edits
  (opacity, visibility, blend mode, clipping) depend on an explicit
  invalidate; a debug-mode check at `flush()` that compares layer property
  hashes against the last flush would catch a forgotten one.

### 2.9 Undo storage -- Ochre as good or better; Kooka's generic diff is the idea to note

- Kooka: `crates/pictura-app/src/history.rs` and `history/planes.rs`.
  Each state is diffed against the previous one after the fact, in 64x64
  tiles compared on every core; planes carry stamps so an unchanged plane is
  skipped without reading bytes; each changed byte is kept once (the tile
  holds whichever side the walk needs and is swapped on traversal); the
  anchor at the cursor is "hollow" (reconstructed from the live document).
  Limits are fixed: 20 states, 10 snapshots. Tiles are **uncompressed**
  (their own `ponytail` note says RLE them if needed).
- Ochre: `ochre/engine/history.py`, `commands.py`. One direction per entry
  (redo derived from the live canvas), zlib level 1 (41 % at 2.8 ms per
  256x256 block measured), block checkpointing with run coalescing, a
  byte budget (`data/history.ini`: 1 GiB, up to 200 entries), and only
  authoritative planes recorded (index-locked layers cost a quarter).
  Ochre retains far more history in less memory.
- What Kooka's design buys: no operation can forget to record its region,
  because the diff is computed, not declared. Worth having as a debug
  assertion in Ochre (diff before/after for an op and compare with the
  recorded delta) rather than as the storage model.

### 2.10 Agent permission model (pairs with 1.6) -- Kooka has it, Ochre has nothing yet

The parts of `public_docs/agent-control.md` and
`crates/pictura-app/cpp/agent_control.cpp` worth copying verbatim as
policy: off by default; tiers read < edit < save; allowed folders;
an approval bar whose Allow button arms after 0.5 s and which denies after
60 s of no answer; "auto-approve files the agent made"; an agent can never
close or revert a document with unsaved changes; edits wait while a dialog
or menu is open, a mouse button is held, or a transform is pending; an
agent naming the wrong document is stopped; the MCP bridge enforces
nothing itself (`crates/pictura-mcp/src/lib.rs`), the server does, so an
agent cannot talk around it; a private headless instance refuses rather
than prompts.

### 2.11 Search ranking (pairs with 1.5)

`crates/pictura-app/cpp/search_index.cpp` `matchTier`: every query word
must occur in the path or keywords; then tier 0 exact label, 1 label
prefix, 2 every word is a prefix of a label word, 3 every word is a
substring of the label, 4 keyword-only match. Labels are cleaned of `&`
mnemonics (keeping `&&`), tab-shortcut suffixes and trailing ellipses.
Simple, predictable, and about 60 lines to port.

### 2.12 Things where Ochre's implementation is as good or better

- **Palette generation.** Kooka `crates/pictura-codec/src/web/quantize.rs`
  is a plain population median cut. Ochre `ochre/engine/palettefiles.py`
  `median_cut(rgba, n, refine=4)` adds k-means refinement, and
  `ochre/engine/quantize.py` structurally excludes protected indices from
  the snap LUT. Ochre is better here.
- **Content-aware fill.** Kooka `crates/pictura-paint/src/healing/synthesis.rs`:
  onion-peel initial fill from the boundary inward, then PatchMatch
  search-and-vote, single-threaded. Ochre `ochre/engine/inpaint.py`:
  Wexler-Shechtman-Irani completion with PatchMatch, premultiplied so
  transparent edges do not fringe, plus harmonic fill and Poisson blend.
  Comparable; if anything is borrowed, compare both on the same holes first
  (Kooka's onion-peel initialisation is the candidate).
- **Refine Edge.** Kooka `crates/pictura-select/src/refine.rs` is a
  self-described approximation; Ochre `ochre/engine/refine.py` uses a
  guided filter (He et al.). Comparable or better; only decontamination
  (1.14) is missing.
- **Filter catalogue as data.** Kooka's `FilterSpec` rows (apron, alpha,
  GPU and dialog policies) and Ochre's property-declared effects with
  generated dialogs (`ochre/engine/props.py`, `ochre/ui/effectdialog.py`)
  are the same idea. Ochre's also gives addon effects a dialog for free.
- **Selections.** Ochre's selection is an anti-aliased coverage mask
  everywhere (`ochre/engine/selection.py`); Kooka still lists engine-wide
  anti-aliased selection rasterisation as open (#614).
- **Below/above cache.** Ochre's compositor caches the stack below the
  active layer and flattens the stack above only when legal
  (`docs/DESIGN.md`, "Compositing"); Kooka's render session does the
  equivalent with a prefix digest. Equal, except for the reduced level
  (2.4).

---

## 3. Ochre has, Kooka lacks -- what could be offered back

Checked against Kooka's ROADMAP. "Open #N" means Kooka wants it and has an
issue for it; those are the best offers.

| Ochre feature | Ochre code | Kooka status |
|---|---|---|
| Index-locked layers: an authoritative index plane, palette edits that recolour exactly the pixels of index N with zero history cost, protected index ranges, a re-snap LUT | `ochre/engine/surface.py`, `palette.py`, `quantize.py`; `docs/DESIGN.md` "Surface is a named-plane container" | Color Table, Indexed transparency and Forced palette open (#555) |
| Palette files: GIMP `.gpl`, `.act`, JASC `.pal`, Paint.NET `.txt`, hex lists (read and write) | `ochre/engine/palettefiles.py` | Swatch libraries open (#605, which names `.aco` / `.ase`) |
| Median cut with k-means refinement | `ochre/engine/palettefiles.py` `median_cut` | Their median cut is unrefined (`web/quantize.rs`) |
| Layer comps | `ochre/ui/panels.py` `CompsDock`, `ochre/engine/commands.py` | Open (#265) |
| Auto-Align Layers (Harris corners, ratio test, RANSAC homography) | `ochre/engine/align.py`, `controller.py:3720` | Open (#582) |
| Auto-Blend Layers (focus stack and panorama via Laplacian pyramids) | `ochre/engine/align.py`, `controller.py:3751` | Open (#583) |
| Photomerge | `controller.py:3804` | Open (#619) |
| Content-Aware Scale (seam carving, one seam map across all layers, masks and frames) | `ochre/engine/seamcarve.py`, `controller.py:2155` | Open (#528) |
| Puppet Warp | `ochre/engine/tools/transform.py` | Open (#527) |
| Quick Mask | `ochre/ui/controller.py`, `mainwindow.py` | Open (#172) |
| Blend If editor UI | `ochre/ui/layerdialogs.py:215`, `compositor.py` | Engine ships; UI open (#264) |
| Apply Image, Calculations | `controller.py:3650` and following | Open (#609, #610) |
| Batch | `controller.py` `batch` | Open (#618) |
| Merge to HDR | `controller.py:1603`, `ochre/engine/refine.py` | Open (#558) |
| Define Pattern | `controller.py` `define_pattern`, `ochre/engine/patterns.py` | Open (#599) |
| Single Row / Single Column marquee | `data/tools.ini` `select_row`, `select_col` | Open (#525) |
| Compressed, budgeted undo (zlib-1, one direction per entry) | `ochre/engine/history.py`, `commands.py` | Their tiles are uncompressed, with a note to compress later (`history.rs`) |
| OpenRaster (.ora) read/write; AVIF, QOI, TGA, ICO, JPEG 2000, PCX, SGI | `ochre/engine/fileio/ora.py`, `flat.py` | Not present; TGA dialog open (#625) |

How to offer: Kooka is Rust, so the useful contribution is the algorithm,
its parameters, and its tests as a spec, not the Python. Kooka is
spec-first (`docs/` corpus and `openspec/`), so an issue that names the
method (paper, parameters, measured behaviour) and links Ochre's GPL code
and tests fits how they work.

**Design input, not code:** Kooka plans its own plug-in ABI and a Rust
C-ABI loader (#623, `docs/01-architecture/plugin-and-scripting-abi.md`).
Ochre's addon contract (`ochre/addons/`, `addons/README.md`: one manifest,
one `register(host)`, format providers, tool target planes, the test "could
a third party ship this with no commit access?") is a worked example they
could learn from.

**Ask first, do not assume:** Kooka's stated target is CS6 parity, and its
ROADMAP rejects Artboards specifically as "a post-CS6 feature". Their
owner has made exceptions (Help > Search, live UI scale). So Ochre's
post-CS6 classical features -- Select Subject / Sky, Focus Area, Sky
Replacement, Shake Reduction, Curvature Pen, Cage / Handle transforms --
should be raised as a question, not a pull request.

**Do not pitch (explicitly out of scope for Kooka):**

- Artboards -- "Not planned", post-CS6.
- Frames, sprite animation, onion skin -- "Video, the timeline, and video
  layers" are not planned, and CS6 frame animation lives in the Timeline.
- The C&C SHP / TMP addon -- game-specific, and Kooka is a photo editor.
- Anything shaped like Adobe plug-in or ExtendScript compatibility.

---

## 4. Licensing

- Both projects are **GPL-3.0-or-later**, so code can move in either
  direction. When porting from Kooka into Ochre: keep the GPL, credit Kooka
  Pictura with the upstream URL and the source file path in the module
  docstring, and add a row to `docs/MINING.md` (the place Ochre already
  records what came from where). Kooka files carry no per-file copyright
  or SPDX header; the project-level `LICENSE` and `NOTICE.md` are the
  attribution anchors.
- Watch for third-party code inside Kooka that carries its own notice:
  Liquify (`crates/pictura-filters/src/distort/liquify.rs` and submodules)
  is ported from PhotoCraft under MIT (`LICENSES/photocraft.txt`), and the
  lens-correction camera database is MIT (`LICENSES/CameraDatabase-MIT.txt`).
  Both are GPL-compatible but the MIT notice must travel with the code.
- **Contributing back** binds Ochre-derived submissions to Kooka's
  `CONTRIBUTING.md` provenance attestation: nothing derived from Adobe source
  code, decompiled Adobe binaries, NDA material, Adobe developer-program
  knowledge, or non-public Adobe-employee knowledge; behaviour not sourced
  from public documentation or lawful black-box observation must be marked
  as an open question. No Adobe assets (profiles, brushes, patterns,
  gradients, swatches, actions, icons, sample files), and test fixtures must
  come from their generators (`scripts/generate-fixtures.py`), not from
  Photoshop-authored files. Also: new dependencies need a reason and must
  pass `cargo deny`; `docs/` changes need a `TASK-ALLOWS-DOCS` marker.
  Ochre's sources are compatible with this as far as checked: its mined
  material is OpenPDN (MIT, assets excluded), GIMP, XCC Utilities and
  WorldAlteringEditor (`docs/MINING.md`), and its PSD code is written from
  Adobe's published file-format specification, which is public
  documentation. Check each offered item against that rule before
  submitting.
- Kooka's AI policy (`README.md`, "A note on AI") accepts AI-assisted
  contributions with mandatory human review and disclosure, which matches
  how Ochre is built.

---

## 5. Top 10 recommendations for Ochre, ranked

Ranked by value for effort, safety first.

1. **Atomic saves** (2.2). Small. Every saver writes to a temp file in the
   same directory and `os.replace`s it. Today a failed save destroys the
   previous good file.
2. **Allocation and decompression budget for opened files** (2.1). Small.
   One `budget.py` modelled on Kooka's `budget.rs`; bound `zlib` output in
   `psd.py`, cap header dimensions in every reader including addons.
3. **Autosave and crash recovery** (1.7), together with the
   **revision-keyed save cache** (2.3) that makes it cheap. Small-medium.
   Copy Kooka's lock-file and "-Recovered" flow from `frame_recovery.cpp`.
4. **EXIF orientation on import, and metadata pass-through on export**
   (1.4). Orientation is a few lines and fixes sideways phone photos;
   pass-through plus a strip-GPS option is a small job on Pillow.
5. **Command search, Shift+Space** (1.5, 2.11). Small, one UI file, and the
   most noticeable usability gain for an editor with about 400 commands.
6. **Headless CLI and an MCP server over `controller.py`** (1.6), with
   Kooka's permission model (2.10). Medium. Ochre's Qt-free controller and
   `RECORDABLE` whitelist make this cheaper than it was for Kooka.
7. **Indexed export in Export As** (1.8). Small. Wire the existing
   `median_cut` and dithering into `exportdialog.py` for GIF and PNG-8.
8. **Reduced-resolution previews when zoomed out** (2.4). Medium. Turns
   filter previews and opacity scrubbing on large canvases from seconds
   into milliseconds at Fit to Window.
9. **PSD fidelity, step one** (1.1, 2.6). Medium. Preserve unknown blocks
   and image resources verbatim, round 16-bit to 8 properly, and move
   PackBits decoding out of pure Python. Full effects / type / smart-object
   modelling can follow later.
10. **Port Kooka's missing CS6 filter families** (1.11), starting with Sketch
    and Brush Strokes, then the extra Camera Raw panels (1.9). Medium, and
    each filter is independently shippable.

Honourable mentions: banded multi-threaded compositing (2.5), Refine Edge
colour decontamination and Spot Healing's matched grain (1.14), lensfun
profiles once metadata exists (1.10).
