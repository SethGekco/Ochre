# Ochre — design

This is the design document the editor was built from, written before any
code existed and kept here because the *reasoning* is the part that is hard
to recover from source. Every decision below has a "why", and several of them
look arbitrary without it.

It is not a description of the finished code — the code is authoritative for
that, and `README.md` states the invariants. Where implementation taught us
something the plan got wrong, the plan has been corrected in place and the
correction is marked **[corrected during implementation]** so the original
reasoning and its revision are both visible.

All eleven phases are complete. Appendices A and B are the SHP and TMP byte
layouts, and remain the reference for the C&C addon.

## Context

Rex asked whether we could make "our own spin" on Paint.NET, targeting Linux natively, with C&C SHP sprite support arriving later as an addon — and, added mid-design, TMP terrain-tile support as a second, separate addon.

Research established these facts, which shape the whole plan:

1. **Paint.NET cannot be forked.** The `paintdotnet` GitHub org holds release downloads, issue trackers, plugin samples, and a few extracted utility libraries — not the application. The app itself is closed-source and built on WPF + Direct2D, so it is Windows-bound even in principle. "Our own spin" therefore means a clean-room build with Paint.NET's *UX and plugin model* as the north star.

2. **OS SHP Builder is a format reference, not a dependency.** It is Delphi/Pascal (`.pas`/`.dfm`/`.dpr`), SVN rev 113, still maintained as of April 2025 by banshee, with no visible LICENSE. We mine it for format behaviour and reimplement — the same rule already set for the FinalAlert2 source.

3. **The existing C# SHP reader is GPLv3 — which the licence decision below turns into an asset.** `WorldAlteringEditor/src/MapEditorLibrary/CCEngine/ShpFile.cs` and `Palette.cs` carry no separate attribution in `LICENSE.txt`, so they fall under the project's GPLv3 (`WorldAlteringEditor/COPYING`, verified). Since Ochre ships GPL-3.0, they're directly portable rather than read-only reference. Note one bug not to carry over: its `RGBColor` expands 6-bit palette values with `v << 2`, which maps 63 to 252 and clips the top of the range. The correct expansion is `(v << 2) | (v >> 4)`.

4. **FS-21's two editors are the best references.** [Advanced-SHP-Editor](https://github.com/FS-21/Advanced-SHP-Editor) and [Advanced-TMP-Editor](https://github.com/FS-21/Advanced-TMP-Editor) are JavaScript + Tauri, both **LGPL-3.0** (GPL-3.0 compatible). They matter more than WAE because, unlike a map editor, they *write* both formats — and the TMP editor ships `ramp_types/` and `game_zdata/`, the height and ramp reference data that is the genuinely hard part. Appendices A and B below are largely reverse-engineered from their source. Two cautions: their READMEs describe the TMP editor as an AI-assisted build, so treat their choices as evidence rather than authority where they diverge from XCC; and **do not redistribute** the `.pal` files or ramp/z-data PNGs from these repos — those are game assets, a separate question from the code licence.

5. **No Python SHP or palette code exists anywhere in the collection — but the TMP seed is already ours.** `RA2MapGen/tmp.py:48` is Rex-authored, unencumbered, and already has the TMP header (`<iiii` = cblocks_x, cblocks_y, cx, cy; then `cbx*cby` int32 offsets at byte 16) plus the per-tile header bytes he verified himself against WAE's `PlaceTile` and the theater slope tiles: **`+40` = height/z, `+41` = terrain type, `+42` = ramp type**. It stops short of pixel decode. So the TMP addon starts from solid owned ground; the SHP codec is genuinely new work.

**Intended outcome:** a general-purpose Linux image editor that stands on its own, which later gains C&C sprite and terrain support through the same addon mechanism any third party could use.

### Decisions taken by Rex

| Decision | Choice |
|---|---|
| Stack | **Python 3.12 + PySide6** |
| Build order | **Full layered editor first**, addon system after it stands alone |
| Extensions | **Two tiers** — compiled accelerator + plain-Python script tier |
| Colour model | **RGBA core + index-locked overlay mode** |
| Working scale | **Both** — 4000×4000 × 10 layers *and* 60×60 sprites |
| Licence | **GPL-3.0** |
| Name | Ochre |

Two of those need their consequences spelled out.

**The licence choice pays for itself.** GPL-3.0 matches the C&C tooling world — XCC, OpenRA, and WAE are all GPL-3.0 — so their SHP, PAL, MIX, and LCW/Format80 code becomes directly reusable instead of read-only reference, and FS-21's LGPL-3.0 editors are compatible too. That removes a large slice of codec work. The accepted cost is that no third party can ship a proprietary addon. Note this does **not** relax the architectural requirement below: the addon API must still be good enough that a third party could ship the C&C addon without touching the core. That's now a design-quality standard rather than a legal one.

**"Index-locked" needs a specific mechanism to actually be lossless.** The naive reading — snap each written pixel to the nearest palette entry in RGB space — is wrong, because a C&C palette's house-colour remap ramp (indices 16–31) and shadow index are frequently near-duplicates of other entries in RGB space. Nearest-colour snapping scrambles exactly the pixels that matter and the file stops round-tripping. Instead, an index-locked layer carries a **parallel authoritative `uint8` index plane** beside its RGBA plane: tools write an *index*, RGBA is derived as `palette[index]` for display/compositing/effects, and the index plane is the source of truth on save. It also makes palette editing trivial — recolouring entry N instantly recolours every pixel using N.

(FS-21's editor solves the same problem with a `uint16` buffer using 65535 as an "unpainted" sentinel, because a `uint8` index plane can't distinguish *unpainted* from *painted with index 0*. Ochre doesn't need that hack — the RGBA alpha channel already carries coverage, so a `uint8` index plane suffices.)

### The hard constraint on scope

SHP/TMP editing is **not a feature of the editor**. It is a separately-downloaded addon: users install the editor, then install the C&C addon. That means **zero C&C code in the core** — no SHP, no TMP, no `.pal` loader, no remap-index constants, no game palettes shipped in the core's `data/`. Anything the addon needs must exist in the core as a *generic* primitive. The core may know "a palette can carry annotated index ranges"; it must not know that 16–31 means house-colour remap in Red Alert 2. The addon supplies that.

The test applied throughout: **could a third party ship this exact addon with no commit access to the editor?** Anywhere the answer is no, the addon API is defective and gets fixed — the core does not get special-cased.

---

## Environment (verified on this machine)

| | |
|---|---|
| Python | 3.12.3, `venv` + `pip 24.0` + `setuptools 83.0.0` |
| numpy | 2.5.1 (system) |
| Pillow | 10.2.0 (system) |
| PySide6 | 6.11.1 — present only in `YRBalanceTool/.venv`; Ochre gets its own `.venv` |
| Cython | **not installed** — Phase 7 dependency |
| Toolchain | gcc/g++/make, Python dev headers at `/usr/include/python3.12` |
| Machine | 24 cores, 30 GB RAM, X11 |

---

## Architecture

Three layers, with an absolute rule: **`ochre/engine/` never imports Qt.** Mechanised by a test that AST-walks every engine module and fails on a `PySide6`/`shiboken`/`PyQt` import. `ochre/ui/` is the only Qt code; `ochre/ui/controller.py` is Qt-free glue between them. This mirrors the YRBalanceTool split, including Qt being a packaging *extra* so the engine installs and tests without it.

### Pixel storage — flat numpy, not tiled

Layers are flat `uint8` arrays, shape `(H, W, 4)`, C-contiguous, RGBA, **straight (non-premultiplied) alpha**, with lazy allocation (`pixels = None` until first write) and a `content_bbox` tracking everything ever written.

Straight alpha because premultiplying at 8 bits is lossy in a way that breaks tools — a pixel at `alpha=1` keeps about one bit of colour, which wrecks the eyedropper, magic-wand comparison, bucket-fill tolerance, and lossless PNG round-trip. Premultiplication happens inside the compositor's working buffers, where it fuses for free into the widen-to-working-precision step.

Flat rather than 64×64 tiles, and this was measured rather than assumed. A strided sub-rect view into a big flat array is essentially free (0.02 ms for 256×256), while gathering the same region from 16 tiles is *slower* (0.029 ms vs 0.020 ms) because of ~2 µs of Python overhead per tile. Tiles impose an interpreter tax exactly where numpy is weakest — many small operations — to buy a locality win that dirty rects already provide. Lazy allocation plus `content_bbox` recovers the one thing tiles genuinely win, which is memory on sparse layers.

> **Known ceiling, accepted and contained.** Flat storage fails at 8000×8000 × 20 dense layers (5.1 GB). The target is 4000×4000 × 10–15, where flat is decisively simpler and measurably faster. Every pixel access in the engine goes through the `Surface` API, never `layer.pixels` directly, so a tiled backend can be swapped in behind that interface without touching the compositor, tools, or history. Revisit only if print-resolution canvases become a requirement.

### Compositing — dirty rects are the architecture, not an optimisation

Measured on this machine: a full-canvas composite of 10 layers at 4000×4000 takes **1077 ms**. That's 1 fps, it's memory-bandwidth bound (~2.6 GB touched), and no amount of C fixes it. Everything is therefore dirty-rect driven:

> **[corrected during implementation]** The real figure is **18.5 s**, not
> 1077 ms — an order of magnitude worse. The estimate assumed a simple
> source-over; the built compositor does the full separable blend with
> per-channel alpha-weighted division, which is far more work. The conclusion
> does not change, it hardens: nothing may composite the full canvas on an
> input event. The UI passes the visible viewport and never more, and
> full-canvas work belongs to export and flatten, which are progress-bar
> operations. That viewport-clipping discipline is load-bearing rather than
> merely tidy.

| Dirty rect | 10-layer composite |
|---|---|
| 64×64 | 0.15 ms |
| 128×128 | 0.76 ms |
| 256×256 | 3.15 ms |
| 512×512 | 13.75 ms |

A **below/above cache** removes the layer-count dependency entirely: during a stroke the stack is static except the active layer, so cache the flattened composite below it and above it, and every stroke recomposite is three blends regardless of stack depth. On the slowest path (integer fallback, no C at all) that turns a 512×512 × 10-layer composite from 28.98 ms into 7.89 ms — comfortably inside a 16.7 ms frame budget with no compiled code.

> **[corrected during implementation]** This is only half true, and the false
> half is a correctness bug rather than a performance one. Caching the layers
> BELOW the active one is always exact. Caching those ABOVE is exact only when
> every one of them uses Normal, because source-over is associative while
> blend modes in general are not: a Multiply layer above the active layer must
> multiply against everything beneath it, the active layer included, and
> pre-flattening it against transparency computes a different image. The test
> caught this immediately — all 2304 pixels differed. The built compositor
> flattens the upper half only when it legally can and otherwise replays those
> layers per composite, which is still cheap because the common case is
> painting at or near the top of the stack.
>
> Measured costs, cached, on the built code:
>
> | dirty rect | on top (flattenable) | at bottom, blends above | with C |
> |---|---|---|---|
> | 128×128 | 0.98 ms | 3.09 ms | 0.82 ms |
> | 256×256 | 2.09 ms | 13.51 ms | 3.03 ms |
> | 512×512 | 6.97 ms | 77.25 ms — over budget | 12.35 ms |
>
> The over-budget cell is exactly what the optional C extension exists for,
> and closing it is what Phase 9 did.

Two rules make the pipeline work:
1. **The engine never composites on its own schedule.** Tools only mark dirty; compositing happens when the UI calls `flush()`, at display-frame rate. A 200 Hz tablet does not cause 200 composites per second.
2. **`Document.display` is one stable, never-reallocated `(H,W,4)` uint8 array.** The UI wraps it once in a `QImage(Format_RGBA8888)` and every repaint is a pure blit from a buffer Qt already points at.

**The zero-copy path is verified, not assumed.** Tested on this machine against PySide6 6.11.1 + numpy 2.5.1: `QImage(buf.data, w, h, w*4, Format_RGBA8888)` aliases the numpy memory directly — a numpy write is immediately visible through `QImage.pixelColor`, `bytesPerLine()` equals `w*4`, and sub-rect copies work. The hazard is lifetime, not correctness: `QImage` does not own the buffer, so the array must outlive it. The stable `Document.display` contract is exactly what guarantees that, and it's the reason the buffer must never be reallocated on resize — resize replaces contents, not identity.

### The accelerator boundary — integer arithmetic makes parity real

The requirement "compiled and pure-Python agree bit-for-bit" **rules out floating point** as the canonical format: compilers contract `a*b+c` into FMA, auto-vectorise with different association orders, and numpy's reductions are pairwise. Bit-exact float agreement between two independent implementations isn't achievable without crippling both.

So every accelerated op is specified in **integer fixed-point**, and that spec is normative. The core primitive, verified exact against `round(x/255)` across all 65536 operand pairs:

```
t = a * b + 128
result = (t + (t >> 8)) >> 8        # == round(a*b/255), exactly
```

Integer ops are exactly defined in both C and numpy, so agreement is **by construction** and the parity test is meaningful rather than flaky. The cost is ~2× versus float on the fallback path, which the below/above cache absorbs.

To C: `blend_rect`, `composite_stack`, `brush_dab`, `flood_fill`, `box_blur_pass`, `rasterize_polygon`. Stays numpy: affine/resample (rare, not interactive), all adjustments (they're 256-entry LUT lookups — `lut[arr]` is already C-speed), tree operations (no pixels move). The principle: **C where numpy needs many temporaries, where the algorithm is sequential, or where per-call overhead dominates.**

Built as a **raw CPython C API extension**, one file, via `setup.py` with an `optional_build_ext` that catches compiler errors and warns rather than failing the install. Cython is rejected — not installed, would become a hard build dependency, and generates code that still needs compiling, so it adds a dependency without removing one. `-fno-fast-math -fwrapv` are load-bearing: they forbid the reassociation that would break bit-exactness. `OCHRE_ACCEL=0` forces the fallback, which the test suite uses to run everything twice so the numpy path is a first-class tested path rather than an untested contingency.

`accel/fallback.py` is **the executable specification**. When the two disagree, the C is wrong.

### Selections — a coverage mask, with `None` as the fast path

`Selection.mask` is a full-canvas `(H,W)` uint8 coverage array, or `None` meaning select-all. Mask rather than path because boolean combination of a lasso with a magic wand with an ellipse is trivial on masks and genuinely hard on paths — and the magic wand has no path form at all. Antialiasing *is* coverage: a 0–255 value is the antialiased edge, natively.

`mask is None` is the critical optimisation — "no selection" is overwhelmingly the common state, and an identity check keeps it free in the hot brush path.

Tools never write to a `Surface` directly. Everything funnels through `Surface.apply_masked(rect, src, mask)`, and the stroke path fuses the selection into the brush's own falloff with one multiply. Consequences that come free: antialiased selection edges give antialiased brush clipping, effects and adjustments clip identically, and there is no separate clipping path to keep in sync.

### History — linear, one direction per entry, block checkpointing

Linear stack, no branching tree. Three command kinds: `PixelDelta` (zlib level-1, measured 41% ratio in 2.8 ms at 256² — level 1 deliberately, because this runs on mouse-up where latency is felt), `PropertyDelta` (opacity/visibility/blend/reorder — tens of bytes, never a pixel copy), and `StructuralDelta` (add/delete/merge, cropped to `content_bbox`).

**Each entry stores only one direction.** Undoing does not read a stored "after" image — it snapshots the current pixels into a freshly-built redo entry *first*, then restores. This is Paint.NET's inverse-generating memento model, and it's strictly better than storing both: it halves undo memory, and it makes undo and redo structurally incapable of disagreeing, because redo is always derived from what was actually on the canvas rather than from what something predicted would be.

A multi-dab drag coalesces into **one** entry via lazy copy-on-first-touch block checkpointing: blocks are snapshotted uncompressed as the stroke first touches them, then assembled and compressed once on mouse-up. Two refinements taken from Paint.NET's `Tool.SaveRegion`:

- **Coalesce runs of adjacent unsaved blocks within a row into a single copy.** A wide stroke then does a few large copies instead of hundreds of small ones. Tracking is a bit-per-block boolean array, so the hot path is an array lookup rather than a region union.
- **The scratch buffer doubles as the live-preview buffer.** `restore_saved_region()` lets shape, line, and gradient tools re-render every mouse-move by restoring from scratch and redrawing. The undo snapshot and the rubber-band preview are the same memory, so interactive shape preview needs no separate preview layer at all.

`MaxStrokeBlocks` forces an auto-commit so a pathological drag can't exhaust memory. All bounds live in `data/history.ini`.

### Native format — zip + PNG layers + INI manifest

`.ochre` is a zip containing `manifest.ini`, a thumbnail, an optional selection mask, any palettes, and one PNG per plane per cell under `frames/NNNN/layers/`. PNG round-trips both RGBA and 8-bit greyscale bit-exactly (verified) at 178 ms per 4000×4000 layer at compress level 1. Single-frame documents still write `frames/0000/`, so there is one layout and a reader needs no special case.

The manifest uses the house `[Type:Name]` convention with `[Layer:0001]` sections and a bare `[Document]` singleton; groups are expressed by flat `Parent`/`Index` keys rather than nesting; extra planes appear as `Plane:<name>=` keys. An index-locked layer writes **both** its authoritative `IndexFile` and its derived RGBA PNG — the latter purely so external tools can read the file; Ochre ignores it on load and regenerates from the index.

> **Ordering trap:** the format must be designed *after* the document model is settled, or it gets versioned twice. That's why it's Phase 5, not Phase 2.

The house rule that unknown keys are silently ignored **is** the forward-compatibility story — a v2 Ochre writing a new key produces a file v1 opens correctly, minus that feature, with no version-negotiation code. `unzip -l` inspects it, `cat manifest.ini` explains it, corruption is localised to one entry, and there is no serialisation code to write or version.

### `Surface` is a named-plane container — one change, three features

The index-locked mode, the frame axis, and TMP's per-pixel z-data are the same requirement wearing three hats: **a layer needs more than one uint8 plane, and those planes must travel together through dirty rects, history, and the file format.** So that is what gets built, once.

```python
class Surface:
    planes: dict[str, np.ndarray | None]   # name → (H,W) or (H,W,4) uint8
    authoritative: tuple[str, ...]         # which planes are truth; the rest are caches
    content_bbox: Rect | None              # ONE bbox for the whole surface
```

| Layer kind | `planes` | `authoritative` |
|---|---|---|
| Normal raster | `rgba` | `rgba` |
| Index-locked | `index`, `rgba` | `index` — rgba is a derived cache |
| With height data | `rgba`, `height` | both |
| Index-locked + height (TMP) | `index`, `rgba`, `height` | `index`, `height` |

`surface.pixels` stays as a property aliasing `planes["rgba"]`, so the compositor, the tools, and the controller API are **unchanged**. Three invariants fall out, and they're what make everything else cheap:

1. **One `content_bbox`, one dirty rect, one history delta per surface**, regardless of plane count. Planes cannot desync.
2. **The compositor reads `planes["rgba"]` and nothing else.** Extra planes cost literally zero in the composite path — not "cheaply", but with no branch to skip and no flag to test.
3. **Only authoritative planes are saved and undone.** Derived caches never persist and never enter history.

**Why the index plane must be authoritative, measured rather than argued.** In a real C&C palette there is **1 exact-duplicate RGB entry**, and the minimum inter-entry squared distance within the remap ramp at 16–31 is **3** — below the rounding error of a single blend operation. Nearest-RGB cannot recover the original index for exactly the pixels that motivate the feature, and an exact duplicate has no correct answer at all. It's also 1250× too slow (brute-force nearest over 256 entries is ~6000 ms/MP; a 256×256 rect would take 401 ms). Keeping the index authoritative makes both problems vanish — the index is never inferred, so it is never wrong and never searched.

The costs and payoffs, all measured:

- **Memory:** +16 MB per 4000×4000 layer (1.25× a normal layer). The RGBA cache is reconstructible in 86 ms, so it can be dropped under pressure.
- **History: four times *cheaper*.** Only the index plane is recorded — 16 MB/plane instead of 64 MB before compression, and 8-bit indexed art compresses far better than RGBA. The feature that adds 25% to resident memory removes 75% from undo memory.
- **Compositing: no special case.** The RGBA cache is refreshed at the one chokepoint where the surface is already being written, via `np.take(palette, index, axis=0, out=...)` — 0.058 ms per 256×256 rect, noise against a 1.94 ms composite. Use `np.take(..., out=)`, never `pal[idx]`: the fancy-index form is 8× slower and allocates.
- **Palette editing: zero history cost, zero drift, zero ambiguity.** Recolouring entry N is a masked cache refresh (17.3 ms at 4000×4000, so a colour picker scrubs at 60 Hz on a 16 MP document) and pushes a 4-byte `PropertyDelta` — not one authoritative byte changes. It's idempotent no matter how many times it's repeated, and it touches *exactly* the pixels whose index is N, including ones whose current RGB matches a different entry. No RGB-based approach can do that at all. **That last property is the feature**, not an optimisation.

**Re-snap, for effects that emit off-palette colour.** A cached 5-5-5 LUT built over the **non-protected** entries only (194 ms build, 32 KB, then 0.31 ms per 256×256 snap). Protection is structural rather than heuristic — protected indices are excluded from the candidate set, so the remap ramp and shadow index **cannot** be produced by a snap. Policy is INI-driven with `refuse` as the default: a Gaussian blur on a 16-colour sprite is almost never what the user wants, and silently snapping it destroys the very semantics the feature exists to protect. `snap` and `unlock` are the other two options.

### Document model — cells, so the frame axis and the layer tree coexist

The one structural decision that cannot be retrofitted. A `Document` has a layer tree *and* a frame axis; pixels live in a **sparse `cells[(layer_id, frame_index)]` map**. With a single frame this degenerates exactly to "one surface per layer", so the common case costs nothing — but a multi-frame sprite stops being a pile of layers pretending to be an animation.

Deliberate refusals, each one load-bearing:
- **The frame axis is 1-D, with a `layout` hint** (`strip` or `grid`). TMP's tile grid is *metadata plus one panel behaviour*, not a second axis. A 2-D axis would permanently complicate cell keys, undo, selection, playback, and the file format for one format family.
- **No per-cell offsets.** Cells are canvas-sized. Per-cell bounding boxes are recomputed at export. This saves complicating selection, effects, blit, and undo to reclaim a few hundred KB.

`Document`, `Layer`, and `Frame` each carry a `meta: dict[str,str]` round-tripped by the native format. Without it, every addon keeps a side table keyed by frame index, which desyncs the instant a frame is inserted or an undo fires — a silent, infuriating class of data loss.

Canvas size, palette, and selection are **document-level**; the layer list is document-level too, so a layer keeps its identity across frames. That matches how sprite editing actually works — you select a region once and apply it across frames, and a palette edit must hit every frame at once. A cell that doesn't exist is simply absent, so per-frame-independent layer structures fall out of sparseness for free.

**Extra planes are addressable by tools.** `Document.active_channel` plus `Tool.target_plane` (declared in `data/tools.ini`, so an addon registers a height brush without touching core code) means the TMP addon gets brush, line, fill, rectangle, and selection on z-data for free rather than reimplementing them worse. A scalar brush is the same code with `channels=1`; selection clipping comes free because `apply_masked` is plane-shaped, not colour-shaped. Scope is deliberately capped at **uint8, single-channel, canvas-aligned** — float or multi-channel planes would force a special case into the compositor, the file format, or the accelerator's integer contract, and nothing needs them.

**Frame memory is the one place the naive approach fails outright.** 100 frames at 4000×4000 with one RGBA layer each is 6.4 GB resident — impossible, and still 1.6 GB even index-locked. (The realistic addon case, a 200×200 sprite with 100 frames, is 20 MB and trivially fine.) So frames have a three-state residency policy under an LRU governed by `data/frames.ini`: **warm** (live arrays), **cold** (zlib-1 per authoritative plane; freeze 115 ms, thaw 80 ms), **unloaded** (still in the zip, never read). Documents open with everything unloaded but frame 0. The 100-frame stress case becomes 8 warm plus 92 cold — resident and workable.

> The measured 64 MB → 2.92 MB cold ratio was taken on a synthetic gradient and is optimistic; real photographic content lands nearer 30–50%, consistent with the history measurements. `MaxWarmBytes` is the actual safety bound — treat the cold figure as a best case and budget cold frames at ~30% of warm.

**Frame switching and playback are different problems and get different code.** The below/above cache belongs to the current frame and is invalidated on switch, so a switch costs one full recomposite (~100 ms typical, up to 1 s worst case). That's fine for clicking a frame and far too slow for scrubbing, so playback is served by a separate reduced-resolution preview cache. History is one linear stack with entries tagged by frame; on undo, if the entry targets another frame the document **switches to it first** — anything else produces invisible undos, the single most confusing thing a multi-frame editor can do. "Apply to all frames" is one entry holding N deltas, so a 100-frame batch undoes with one Ctrl+Z.

### The addon system

**One discovery API, two implementation tiers.** An addon is a directory under `addons/` with an `addon.ini` manifest and an entry module exposing `register(host)`. A compiled `.so` and a plain `.py` are discovered by the *same* `importlib` machinery, resolved by its own suffix ordering — so a wrong-ABI `.so` simply isn't found and the `.py` fallback happens for free.

Load-bearing rules:
- **Protocols, not ABCs** — better Python, and a materially stronger arm's-length boundary.
- **Explicit `register(host)`, never import side effects.** This is the hinge that lets a `.so` and a `.py` be the same kind of thing.
- **Manifest declares, code registers, host cross-checks.** What makes lazy loading and a meaningful trust prompt possible.
- **Only addon *panels* may import Qt.** The core's layer split propagates into the addon tier, which buys headless batch mode and headless addon tests.
- **The host opens an undo transaction around every addon call and rolls it back on exception.** This is the isolation mechanism, not a convenience — it's the only reason a crashing effect leaves the document bit-identical.
- **Honest security.** Addons are arbitrary Python; Python cannot sandbox that, so no fake sandbox is built. Instead: nothing runs at scan time, nothing runs until used, content-hashed trust on first use, a real kill switch, and a crash breadcrumb — because a segfault in native code is the one failure Python genuinely cannot contain.

**The rule that settles every boundary dispute: if the second addon would have to duplicate it, it belongs in the core.** That is what puts indexed mode, the palette editor, the frame-strip panel, and the fixed-palette quantizer in the core, and what keeps isometric geometry, diamond masks, terrain enums, house-colour remap, facings, and the RLE codecs entirely in the addon. Palette *matching* (into a given palette) is core because SHP and TMP both need it; palette *generation* (median cut, octree, k-means) is an addon, because it's contested and neither format uses it.

**Accelerator toolchain, unified.** The core accelerator is a raw CPython C API extension (§ above). Addons may use whatever they like — Cython, raw C, `ctypes`, or nothing — because the addon contract is only "a Python-importable module", so build tooling is each addon's own business. The core ships one documented raw-C example as the reference. This keeps gcc plus Python headers the sole build dependency for the core, and sidesteps the question of committing generated `.c` entirely.

### The canvas

`QAbstractScrollArea` with a custom `paintEvent` — not `QGraphicsView` (unnecessary scene-graph machinery for a single image) and not `QOpenGLWidget` (a GL context to manage for a blit that's already fast enough).

`CanvasSurface` owns a numpy buffer aliased by a `QImage`, and the **UI owns that cache, not the engine** — the copy out of the engine is also where premultiplication happens, so it isn't waste. Zoom below 100% uses smooth interpolation; at or above 100% it switches to strict nearest-neighbour, with a pixel grid appearing above a configurable threshold. `ViewState` holds all zoom/pan math as **pure Qt-free functions**, so the round-trip `doc→widget→doc` is unit-testable at every zoom without a `QApplication`.

Effects tile at **256×256** across a `QThreadPool`. The tile size is not arbitrary: numpy releases the GIL inside ufunc and copy calls on arrays above a small threshold, so N tiles genuinely run on N cores — but per-tile Python overhead *holds* the GIL, so tiles must be big enough that C work dominates. 32×32 would be dominated by GIL-held dispatch and would scale negatively. A corollary that belongs in the effect base class docstring: **an effect written as a pure-Python per-pixel loop will not parallelize at all** and serializes the whole pool. A test times a trivial effect across 1 vs N threads and asserts speedup, which catches accidental de-vectorization in review.

Thread safety is by **partitioning, not locking**: each worker gets a disjoint destination slice, the source is read-only for the duration, and the controller cancels and joins before the engine is allowed to mutate anything.

---

## Phases

**All eleven are complete.** Kept because the ORDER carries the argument:
each phase exists where it does because building it earlier would have meant
guessing, and building it later would have meant retrofitting. Phase 1 is
flagged un-retrofittable and was; Phase 9 deliberately comes last because
optimising against a working reference is safe and writing C first is not.

Built strictly bottom-up. The engine is complete and fully tested headlessly before a single widget exists; the editor stands alone before the addon system; the addon system is proven before any C&C code is written.

**Phase 0 — Spine.** Repo, `.venv`, `pyproject.toml` (Qt as an extra, not a core dep), GPL-3.0 headers, `ini.py`, `settings.py`, `geometry.py`, `tests/run_all.sh`. Ships with the `test_no_qt.py` guard live from day one.

**Phase 1 — Document model.** `Document`/`Layer`/`Frame`/`Cell` with the sparse cell map, `ColorMode`, `Palette`, aux planes, the three `meta` dicts. `Surface` with lazy allocation and `content_bbox`. Undo transactions over cells and aux planes. **This is the phase that cannot be retrofitted** — the structural items all land here and everything else is built on them.

**Phase 2 — Compositing.** Layer tree, the blend-mode set, `Compositor` with dirty rects and the below/above cache, `accel/fallback.py` as the normative integer spec. Headless and fully testable; no tools yet.

**Phase 3 — Mutation.** History stack, the three delta kinds, stroke sessions with block checkpointing. First real editing, still headless.

**Phase 4 — Selections and tools.** Coverage-mask selection, the four selection tools, then the paint tools against `Surface.apply_masked`.

**Phase 5 — File I/O.** `.ochre` zip format written against the *settled* document model (doing this earlier versions the format twice), plus flat import/export via Pillow.

**Phase 6 — UI.** `controller.py`, then `ViewState` (pure math, tested first), then `CanvasSurface`, then the canvas widget, then docks — Layers first, since it's the one that proves the model/delegate/bus sync. `OCHRE_SELFTEST` green from the first day of this phase.

**Phase 7 — Adjustments and effects.** LUT-based adjustments, the effect base class, auto-generated property dialogs, then tiled live preview with cancel.

**Phase 8 — The indexed workflow.** The *structure* landed in Phase 1; this is the user-facing half — the palette dock with annotated index ranges, index shown alongside RGB in the readout and status bar, the snap LUT, the fixed-palette quantizer with dither, and the frame-strip panel in both strip and grid layouts. Generic throughout: the core knows a palette may carry annotated ranges; it never learns what any of them mean.

**Phase 9 — C accelerator.** Written *last*, against a green parity test. Optimising against a working reference is safe; writing C first is not.

**Phase 10 — Addon host.** Manifest, loader, the Protocol set, fault isolation, the trust prompt. Proven with a trivial fixture addon before anything real depends on it.

**Phase 11 — The C&C addon suite.** One distributable containing two independent, separately-disableable providers (SHP and TMP) over a shared C&C support layer (palette loading, MIX access, the palette-range annotations). Shipped and versioned independently of the editor.

---

## Verification

Every phase is verifiable headlessly, in the house idiom: standalone scripts in `tests/` printing `ok: <name>`, exiting non-zero on first failure, driven by `tests/run_all.sh` reporting `passed=N failed=N`. No pytest.

**The whole suite runs twice** — once normally, once with `OCHRE_ACCEL=0` — so the numpy fallback is a first-class tested path rather than an untested contingency.

Load-bearing tests, beyond the per-module ones:

- **`test_no_qt.py`** — AST-walks every `ochre/engine/**/*.py` and fails on any `PySide6`/`shiboken`/`PyQt` import. The house rule, mechanised. Also asserts every engine source decodes as ASCII — during design, one agent's own output contained a stray CJK character inside an identifier, and that class of glitch should fail a test rather than reach a reviewer.
- **`test_indexlock.py`** — round-trip preserves indices **exactly** through save, load, and undo, *including duplicate-RGB entries and the remap ramp*; a palette edit changes zero authoritative bytes; an index-locked layer composites byte-identically to a normal layer holding the expanded RGBA. This is the test that proves the whole hybrid colour model works.
- **`test_planes.py`** — a height plane shares one dirty rect and one delta with its colour plane; composite output is byte-identical with and without the extra plane present (proving the compositor really does ignore it for free).
- **`test_frames.py`** — per-frame dirty isolation; undo switches frames; cold→warm→cold round-trips bit-exactly; `MaxWarmBytes` is respected under a 100-frame batch operation.
- **`test_accel_parity.py`** — compiled vs numpy over a seeded corpus plus edge cases (alpha 0 and 255, 1-pixel rects, odd widths that defeat SIMD tails, non-contiguous views, every blend mode, every opacity). Asserts `array_equal`, not `allclose`. Skips cleanly when the extension isn't built.
- **`test_composite.py`** — the below/above cached result must be *identical* to an uncached full composite. This is what makes the cache trustworthy.
- **`test_addon_isolation.py`** — an addon raising in `register` leaves the registry byte-identical; an effect raising mid-apply leaves the layer bit-identical; writing outside the given ROI is caught by a canary-filled destination.
- **`test_ui_smoke.py`** — under `QT_QPA_PLATFORM=offscreen`, builds the window and forces a real synchronous `repaint()` on a live document. Constructing widgets verifies almost nothing about a canvas; forcing a `paintEvent` exercises the buffer aliasing, the premultiply, and `drawImage` — precisely the code that segfaults if the lifetime rules are broken.
- **`shp_roundtrip.py` / `tmp_roundtrip.py`** — over a corpus directory, skipped with a printed `ok: skipped` when unset. Read→write→read must be **byte-identical** for untouched files. This is the only honest proof that a reimplemented format codec is correct, and it's the regression net that later makes the compiled tier safe to optimise aggressively. XCC Mixer's output is the external oracle.

**Beyond tests**, three things get driven manually because no test substitutes for them: open a large photo and zoom to fit (checks downscale quality, which has no mip chain in v1); paint a long stroke on a 4000×4000 document with ten layers and confirm it feels immediate; and load a written SHP in XCC Mixer and in the actual game before declaring the codec correct.

### Already verified during planning

- **Zero-copy `QImage` over numpy** — PySide6 6.11.1 + numpy 2.5.1, aliasing confirmed, correct stride, sub-rect copies fine.
- **RLE-Zero round-trip** — all-transparent, no-transparency, trailing runs, alternating, odd dimensions, and the 255-clamp; a 600-wide empty row produces the predicted `08 00 00 ff 00 ff 00 5a`.
- **TMP diamond packing** — row widths `4*(y+1)` from `x = cx/2 - 2*(y+1)`, verified to sum to exactly `cx*cy/2` for both 48×24 (576 bytes) and 60×30 (900 bytes), every row in bounds and centred, rectangle→diamond→rectangle exact.
- **Compositing costs** — measured on this machine, not estimated; they are what forced the dirty-rect architecture.

### Invariants mined from Paint.NET

Things a from-scratch implementation gets wrong, established by surveying the OpenPDN source. Each is cheap to honour and expensive to retrofit.

**Every average of RGBA pixels weights colour by alpha and divides by the alpha sum.** Blur, resample, supersample, gradient LUT — all of them. Averaging straight RGB across pixels with differing alpha produces grey or dark fringes around transparent edges, and it is the single most common defect in hand-rolled editors. Paint.NET ships two nearly identical helpers, `Lerp` and `Blend`, where `Lerp`'s own documentation warns it is wrong for this; the gradient renderer correctly uses `Blend`. Ochre gets one function, alpha-weighted, and no trap to fall into.

**Dirty rects are inflated by one pixel before invalidating.** Antialiased edges bleed outside their geometric region; without the inflation you get one-pixel trails behind every moving overlay. Ochre's `DirtyRegion` already merges toward a cap, which is the same instinct as Paint.NET's `SimplifyAndInflateRegion` — a lasso selection can produce thousands of one-pixel-tall scanline rects, and repainting them individually is far slower than a little overdraw. The inflation is the missing half.

**A repaint-coalescing timer between "tile finished" and "repaint" is mandatory.** Tiles completing at high frequency will drown the UI thread in invalidations, making a multithreaded effect render *slower* than a single-threaded one. Batch completed tiles into a dirty region and repaint on a timer.

**Preview work is not thrown away on commit.** When the user accepts an effect, render only `region − union(already_previewed)`. Most implementations re-render from scratch.

**Render the first tile synchronously, and make it one scanline tall.** Paint.NET deliberately slices tile 0 thin and renders it on the calling thread before spawning workers, so an expensive effect shows a strip of result almost instantly. Pure perceived-performance engineering, nearly free.

**The effect source buffer is made read-only for the duration of the render.** In numpy that is `arr.flags.writeable = False` — one line that turns a whole class of addon bug into an immediate exception instead of silent corruption.

**Zoom is an exact rational, not a float.** `fractions.Fraction`, so 100% is exactly 100% and zoom-in/zoom-out cycles don't accumulate error.

**Constraint modifiers are re-evaluated on every motion event, not sampled at mouse-down.** That is what lets pressing or releasing Shift mid-drag update the preview live.

**Enable/disable belongs to the property, not the widget.** Paint.NET models it as `ReadOnly` on the property itself, which is what keeps the whole effect-property system headless and scriptable. Ochre follows.

---

## What was built, and what was not

The plan describes Paint.NET's full tool set. What exists is the subset the
phases actually reached:

**Built** — 15 tools: pencil, paintbrush, eraser, paint bucket, colour
picker; four selection tools (rectangle, ellipse, lasso, magic wand); four
shape tools (line, rectangle, ellipse, freeform); a gradient with five
types; and text. Adjustments: brightness/contrast, levels, hue/saturation, invert,
posterize, black & white, sepia. Effects: gaussian blur, sharpen, add noise.

**Not built** — clone stamp, move-selection and move-pixels, zoom and pan
tools. None is blocked by the architecture; each is a `Tool` subclass
registered in `data/tools.ini`.

The text tool's IN-CANVAS EDITOR is also still missing, and that distinction
matters. The engine side is complete: text rasterises, places, clips, undoes
and commits, and a script can drive it. What does not exist is the caret and
keyboard handling that let you type directly on the canvas. The seam is
already there — the text session deliberately stays open after mouse-up, and
`set_text()` redraws live — so the UI work is wiring keystrokes to a method
that already does the right thing.

The shape tools did confirm the prediction this document makes in the history
section: because they redraw from scratch on every motion event via
`StrokeSession.restore()`, and that buffer is the same one that becomes the
undo entry, the rubber-band preview came for free and cannot disagree with
what is committed. Their outlines are also composed from the *selection*
rasterisers — `coverage(grown)` minus `coverage(shrunk)` — so they inherited
the supersampling symmetry fix rather than needing their own.

## Open questions

None of these blocked starting, and none blocks using the editor. Each is
contained, and each names what would settle it.

| # | Question | What resolves it |
|---|---|---|
| 1 | **Flat storage has a hard ceiling** — 8000×8000 × 20 dense layers is 5.1 GB and the design fails. | Decide whether print-resolution canvases are ever in scope. Contained either way: every pixel access goes through the `Surface` API, so a tiled backend swaps in without touching the compositor, tools, or history. |
| 2 | ~~Paint.NET's exact formulas for Reflect, Glow, and Negation~~ | **RESOLVED.** Extracted verbatim from OpenPDN `Data/UserBlendOps.Generated.H.cs`. All integer, so they drop straight into the parity contract. `Glow(A,B) = Reflect(B,A)` — arguments swapped — and `Xor` is genuinely bitwise; both would have been wrong if guessed. |
| 3 | **Downscale quality without a mip chain.** Bilinear at 12% on a 6000px image will alias. | Build v1 without it, open a large photo, zoom to fit, look. The insertion point is already designed in. |
| 4 | **Onion-skinning** would need 2–3 frames composited live, which the current per-frame below/above cache doesn't serve. | Decide before building the timeline UI. Contained to `compositor.py` (the cache becomes per-frame and LRU'd). |
| 5 | ~~Text rendering is the one place the UI generates pixels~~ | **RESOLVED, and the premise was wrong.** The question assumed glyph rasterisation must happen in the UI because it needs font machinery, and proposed rendering through `QPainter` into a `QImage` and handing the engine a blit. That is unnecessary: Pillow is ALREADY an engine dependency and ships FreeType (with raqm, so complex scripts shape correctly). Text rasterises in `ochre/engine/text.py` to an ordinary coverage plane, the engine stays Qt-free, and text is scriptable and headlessly testable like every other tool. Font family names resolve through fontconfig, so aliases such as `sans-serif` work and fallback is delegated rather than reimplemented. One deliberate exclusion: subpixel (LCD) antialiasing is avoided, because its colour fringes are correct against a known opaque background and wrong on a transparent layer. |
| 6 | **Tablet pressure on X11** needs XInput2 and varies by device. | Test with the actual tablet; degrades cleanly to `pressure=1.0`, which is already the default. |
| 7 | **How large do TMP "extra" extents get in practice?** Drives whether canvas-sized cells stay free. | Cheap to settle empirically — scan the vanilla theater tiles under the `RA2_TILES` root that `tmp.py` already walks, and histogram the extra dimensions. Worth doing before the canvas decision is frozen. |
| 8 | ~~Does SHP writing need byte-exact reproduction of original compression choices?~~ | **RESOLVED.** The addon preserves each frame's original crop rect and radar colour, and chooses RLE only when it is actually smaller — so an untouched sprite re-saves byte-identically, asserted by `tests/test_cnc.py`. One refinement the plan did not anticipate: the stored crop rect is honoured only while the content still fits inside it, because honouring it unconditionally silently discards any edit made outside the original bounds. Still unverified in-game. |

---

## Appendix A — SHP (TS/RA2) format reference

Established from [ModdingWiki SHP (TS)](https://moddingwiki.shikadi.net/wiki/Westwood_SHP_Format_(TS)), [RLE-Zero](https://moddingwiki.shikadi.net/wiki/Westwood_RLE-Zero), [ModEnc PAL](https://modenc.renegadeprojects.com/PAL), cross-checked against XCC's and OpenRA's behaviour. All little-endian.

**File header — 8 bytes.** `Zero` u16 (always 0 — the only magic, used for sniffing), `FullWidth` u16, `FullHeight` u16, `NumFrames` u16.

**Frame header — 24 bytes × NumFrames**, immediately after. `FrameX` u16, `FrameY` u16, `FrameWidth` u16, `FrameHeight` u16, `Flags` u32, `FrameColor` byte[4] (radar/minimap average colour — cosmetic, safe to zero), `Reserved` u32, `DataOffset` u32 (**0 = empty frame, no data block**).

**Compression values.** `0`/`1` = raw scanlines, no prefix. `2` = raw but each scanline length-prefixed. `3` = RLE-Zero, length-prefixed. **Read 0–3; write only 0 or 3**, choosing 3 only when the encoded result is actually smaller than `w*h` (this is what XCC does).

**RLE-Zero, per scanline** (never crosses a row boundary):
```
u16 lineLength          # INCLUDES its own 2 bytes; payload = lineLength - 2
loop until payload consumed:
    b = next byte
    if b != 0:  emit b                      # literal palette index
    else:       c = next byte; emit c zeros # transparent run
```

**Encoder details that are easy to get wrong** (all verified against XCC's `encode3`):
- **Transparent runs clamp to 255.** A 512-wide empty row emits `00 FF 00 FF 00 02`, not one run. Get this wrong and the games misparse.

> **Already validated.** A Python encode/decode pair following exactly this spec was round-tripped on this machine against all-transparent rows, no-transparency rows, trailing runs, alternating patterns, odd frame dimensions, and the 255-clamp case. All exact, all bytes consumed. A 600-wide empty row encodes to `08 00 00 ff 00 ff 00 5a` — three clamped runs, matching the predicted output. The codec is roughly 25 lines and is the lowest-risk part of the addon.
- Auto-crop every frame to the bounding box of non-zero pixels; that box becomes `FrameX/Y/Width/Height`.
- A fully transparent frame is written as `Width=Height=0`, `DataOffset=0`, no data block — and readers must tolerate it.
- 8-byte-align each frame's data block.

**Palette (`.pal`) — exactly 768 bytes**, 256 × RGB, **6-bit VGA values (0–63)**.
- Expand for display: `(v << 2) | (v >> 4)` → 63 maps to 255. The common shortcut `v << 2` gives 252 and clips the top of the range (this is the bug in WAE's `RGBColor`).
- Write back: `v >> 2`. Lossless round-trip.
- The engine further squashes to 16-bit 5:6:5, so authored palettes want R/B divisible by 8 and G by 4.

**Index conventions** — these are the pixels that matter most and that any RGBA round-trip would destroy:

| Index | Meaning |
|---|---|
| `0` | Transparent. Never drawn; RLE-Zero compresses exactly this value. Must never be paintable as a colour. |
| `1` | Conventional shadow colour (shadow frames are filled with it). |
| `16–31` | **House-colour remap ramp.** Substituted with the owning player's colours at draw time. Needs a "preview as house colour X" mode. |
| `240–254` | Unaffected by map lighting — muzzle flashes, glows. |

Palettes are **external and theatre-specific** (`unittem/unitsno/unitubn/unitlun/unitdes.pal` for units and buildings; `iso*.pal` for terrain; `anim.pal`, `cameo.pal`). A SHP carries no palette reference at all — which palette applies is pure convention based on what the sprite is. Unit sprites conventionally put **art frames first, shadow frames second**, each half the frame count.

**Sibling formats.** TD/RA1 SHP is a genuinely different format (14-byte header, LCW compression plus XOR inter-frame deltas, full-canvas frames). Dune II SHP is a third. Sniff on the first u16: `0` ⇒ TS/RA2, non-zero ⇒ frame count of a TD file. Out of scope for v1 of the addon, but the codec API should leave room.

**Reference implementations.** With Ochre on GPL-3.0, everything here except OS SHP Builder is legally reusable; the ranking below is by *usefulness*, with attribution obligations noted.

| Project | Licence | Use |
|---|---|---|
| [XCC](https://github.com/OlafvdSpek/xcc) (C++) | GPL-3.0 | **The de-facto ground truth.** `shp_ts_file_write.cpp`, `encode3()`. Reusable with attribution. XCC Mixer's binary is also the community's "does my file load?" oracle — use it for golden-file round-trip tests. |
| [EngieFileConverter](https://github.com/Nyerguds/EngieFileConverter) (C#) | WTFPL | Reads *and writes* all SHP types including Dune II. No attribution burden at all. |
| [OpenRA](https://github.com/OpenRA/OpenRA) (C#) | GPL-3.0 | Best format-sniffing heuristic. Note it has **no** TS SHP writer. |
| [cnc-formats](https://github.com/iron-curtain-engine/cnc-formats) (Rust) | MIT/Apache-2.0 | Explicitly clean-room; immature, but a useful independent second opinion on the spec. |
| [Advanced-SHP-Editor](https://github.com/FS-21/Advanced-SHP-Editor) (JS) | LGPL-3.0 | Compatible. Strongest for *editor UX* ideas and for the traps listed above. |
| [OS SHP Builder](https://www.ppmsite.com/shpbuilderinfo/) (Delphi) | none stated | **Read-only reference** — no licence means no copying, regardless of ours. |

**Additional SHP traps** (reverse-engineered from FS-21's encoder, which auto-crops every frame unconditionally):

- **Shadows are two completely different mechanisms.** TS/RA2: the file holds `2N` frames — `0..N-1` are the sprite, `N..2N-1` are 1-bit shadow masks using only index 0 and index 1. TD/RA1: the shadow is **index 4 inline within the frame**. An editor that treats frame count as animation length will play every unit animation twice, the second pass as black silhouettes. The frame model needs an explicit shadow-pairing concept.
- **Alpha-image SHPs use index 127 as transparent**, not 0 (RA2's `alpha_image.pal` glow overlays are intensity maps where mid-grey means "no effect"). Transparency index must be a per-document setting, never a constant.
- **Frame header offset 0x08 is ambiguous in the wild.** ModEnc/XCC document a u32 compression field; FS-21 writes `u16 flags` + `u16 size`. They only agree when size is 0. Read by masking the low bits; write 0 at 0x0A. Their `size` field also truncates above 64 KiB, so prefer writing 0 — it's redundant with `DataOffset` deltas anyway.
- **A literal index-0 pixel cannot be stored in an RLE-Zero frame.** Index 0 *is* the run marker. This is a format constraint, not an implementation choice.
- **Indices 16–31 must be protected from every resampler, quantizer, and nearest-colour match.** Otherwise scaling a sprite smears the remap ramp and the house colours break. Nearest-index search must run over the non-protected palette subset only.
- Decoders should be **lenient** — truncated lines and over-long runs are common in files from old tools. Clamp and continue rather than raising.

---

## Appendix B — TMP (TS/RA2 terrain tile) format reference

Reverse-engineered from FS-21's `tmp_format.js`, and **independently confirming** the offsets already established in `RA2MapGen/tmp.py:48`. Origin of the logic is XCC's `tmp_ts_file.h`.

**File header — 16 bytes**, 4 × int32: `cblocks_x`, `cblocks_y` (template size in tiles), `cx`, `cy` (tile pixel size — 48×24, or 60×30).

**Offset table** at byte 16: `cblocks_x * cblocks_y` int32 absolute file offsets. **An offset of 0 means an empty slot** — templates are rectangular grids but real templates are diamond-shaped, so most slots are unused. Never assume a slot has a tile.

**Per-tile record — 52-byte header, then data:**

| Off | Type | Field |
|---|---|---|
| 0, 4 | i32 | `X`, `Y` — tile position in template pixel space |
| 8, 12, 16 | i32 | `ExtraOffset`, `ZDataOffset`, `ExtraZOffset` — 0 if absent |
| 20, 24, 28, 32 | i32 | `ExtraX`, `ExtraY`, `ExtraWidth`, `ExtraHeight` |
| 36 | u32 | `Flags` — bit0 HasExtra, bit1 HasZ, bit2 IsRandomized |
| **40** | u8 | **`Height`** — elevation level |
| **41** | u8 | **`LandType`** — terrain type |
| **42** | u8 | **`RampType`** — slope shape |
| 43–45 | u8×3 | `RadarLowColor` RGB |
| 46–48 | u8×3 | `RadarHighColor` RGB |
| 52 | | diamond pixel data begins |

**The diamond layout.** The base tile is *not* a rectangle — it is `cx*cy/2` bytes holding only the rhombus. Row `y` in the top half has width `4*(y+1)` starting at `x = cx/2 - 2*(y+1)`; the bottom half mirrors it. For 48×24 that's widths 4,8,…,48 then 44,40,…,0, totalling 576 = 48×24/2. Off-by-one here shears every tile diagonally.

**Four independent data blocks per tile**, each with its own offset and each optionally present: base diamond (always), base z-diamond (bit1), extra rectangle (bit0), extra z-rectangle (bit0 & bit1). The **extra block is a plain rectangle, not diamond-packed** — it exists for art taller than the diamond (cliffs, bridges, tall rocks), drawn at `ExtraX/Y` minus the tile's `X/Y`.

**Traps:**
- **Extra/Z offsets are relative to the tile record start, not the file start.** Treating them as absolute is the number-one way to get garbage tiles.
- **Z-data is 0–31 valid, 255 = no-data sentinel, 32–254 invalid.** It's a 5-bit depth in a byte. It drives the software depth buffer that decides whether a unit draws in front of or behind terrain — wrong z-data makes units walk through cliffs. It also drives hit-testing, not just rendering.
- **`LandType` is many-to-one**: 0/1/13 all mean Clear, 2/3/4 Ice, 7/8 Rock, 11/12 Road. These are distinct engine values with identical labels, so an editor must **preserve the original numeric value** on round-trip rather than normalising it, or tile behaviour changes silently.
- **`RampType` has 21 values** (0 = flat, 1–4 single edges, 5–8 outer corners, 9–12 inner corners, 13–16 steep, 17–20 double diagonals). Not derivable from pixels — it must be editable.
- **Two radar colours per tile** (Low and High, interpolated by height/lighting for the minimap). Not derivable from a single pixel average, unlike SHP's `AverageColor`. Tools that write only one produce a broken minimap.

**Do not redistribute** the `.pal` files or ramp/z-data reference PNGs from these repos — those are game assets, a separate question from the code licence.
