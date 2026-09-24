# Mined references

Ochre is GPL-3.0, which makes the established free-software image editors
licence-compatible sources rather than things we can only look at. This file
records what came from where, so attribution is traceable and so nobody
re-derives something we already have.

| Source | Licence | Compatible? |
|---|---|---|
| [Paint.NET 3.36.7 / OpenPDN](https://github.com/rivy/OpenPDN) | MIT (3 exceptions) | Yes |
| [GIMP](https://github.com/GNOME/gimp) | GPL-3.0 | Yes |
| [XCC Utilities](https://github.com/OlafvdSpek/xcc) | GPL-3.0 | Yes — addon only |
| [WorldAlteringEditor](https://github.com/CnCNet/WorldAlteringEditor) | GPL-3.0 | Yes — addon only |
| [FS-21 SHP/TMP editors](https://github.com/FS-21/Advanced-SHP-Editor) | LGPL-3.0 | Yes — addon only |
| [EngieFileConverter](https://github.com/Nyerguds/EngieFileConverter) | WTFPL | Yes — addon only |
| OS SHP Builder | none stated | **No.** Read-only reference. |

## Two things from OpenPDN we deliberately do not use

**Artwork and resource assets** — every `.png`, `.resx` and `.resources` file
is CC BY-NC-ND, not MIT. Ochre ships none of them and draws its own icons.

**GPC**, the polygon clipper under `SystemLayer/GpcWrapper` and
`ShellExtension`, requires a commercial licence from the University of
Manchester. Ochre never touches it, and the reason is architectural rather
than legal: selections are coverage masks, not polygon paths, so the
dependency never arises.

---

## Blend mode formulas

Taken verbatim from `Data/UserBlendOps.Generated.H.cs`. `A` is the **lower**
layer, `B` the **upper**, both `0..255` per channel. These are already integer
arithmetic in the original, which is exactly what Ochre's accelerator parity
contract requires — the C extension and the numpy fallback can agree
bit-for-bit by construction.

`INT_SCALE(a, b)` is `round(a * b / 255)`, computed exactly as:

    t = a * b + 128
    INT_SCALE = (t + (t >> 8)) >> 8

(Verified equal to `round(a*b/255)` for all 65536 operand pairs.)

| Mode | Formula |
|---|---|
| Normal | `B` |
| Multiply | `INT_SCALE(A, B)` |
| Additive | `min(255, A + B)` |
| ColorBurn | `B == 0 ? 0 : max(0, 255 - (255-A)*255 / B)` |
| ColorDodge | `B == 255 ? 255 : min(255, A*255 / (255-B))` |
| Reflect | `B == 255 ? 255 : min(255, A*A / (255-B))` |
| Glow | `Reflect(B, A)` |
| Overlay | `A < 128 ? INT_SCALE(2A, B) : 255 - INT_SCALE(2*(255-A), 255-B)` |
| Difference | `abs(B - A)` |
| Negation | `255 - abs(255 - A - B)` |
| Lighten | `max(A, B)` |
| Darken | `min(A, B)` |
| Screen | `A + B - INT_SCALE(A, B)` |
| Xor | `A ^ B` |

Three traps in that table:

- **`Glow` is `Reflect` with the arguments swapped.** It is not a formula of
  its own.
- **`Xor` is genuinely bitwise**, not arithmetic difference.
- **`Overlay` keys on `A`, the lower layer**, not the upper one. Several
  other editors key on the upper layer and produce different output.

`Reflect`, `Glow` and `Negation` are Paint.NET-specific and appear in no
public blend-mode specification, so there was no way to get them right by
reasoning — they had to come from the source.

---

## Invariants worth honouring

Each of these is something a from-scratch implementation typically gets wrong.
They are recorded in the plan's architecture section too; this is the
provenance.

1. **Alpha-weighted averaging, everywhere.** Blur, resample, supersample and
   the gradient LUT all weight colour by alpha and divide by the alpha sum.
   Averaging straight RGB across differing alpha causes dark fringes at
   transparent edges. Note that Paint.NET ships both `Lerp` and `Blend` and
   its own docs warn that `Lerp` is wrong for this — Ochre has one function
   and therefore no trap.
2. **Undo entries generate their redo.** `OnUndo()` captures current pixels
   into a new redo entry before restoring. Halves memory; makes undo/redo
   incapable of disagreeing.
3. **Tile-granular stroke checkpointing**, with runs of adjacent unsaved
   blocks in a row coalesced into one copy. Cost is O(area touched) paid once,
   not O(dabs). The scratch buffer doubles as the rubber-band preview buffer,
   so shape and gradient tools need no preview layer.
4. **Inflate dirty rects by 1px before invalidating** — antialiased edges
   bleed outside their geometric bounds.
5. **Coalesce repaints on a timer.** Without it, fast multithreaded effects
   drown the UI thread and run slower than single-threaded.
6. **Reuse preview work on commit**: render only
   `region - union(already_previewed)`.
7. **Render tile 0 synchronously and make it one scanline tall**, so an
   expensive effect shows a result strip immediately.
8. **Mark the effect source buffer read-only** during render:
   `arr.flags.writeable = False`.
9. **Zoom is `fractions.Fraction`**, not a float, so 100% is exactly 100%.
10. **Re-evaluate constraint modifiers every motion event**, so Shift pressed
    mid-drag updates the preview live.
11. **Enable/disable is a property of the property, not the widget** — it is
    what keeps the effect system headless and scriptable.

## Flood fill / magic wand

The tolerance metric is not Euclidean RGB distance. From
`tools/FloodToolBase.cs`:

    sum = 0
    for ch in (R, G, B):
        d = a[ch] - b[ch];  sum += (1 + d*d) * a.A // 256
    d = a.A - b.A;          sum += d*d
    match = sum <= tolerance * tolerance * 4

The RGB terms are scaled by the reference pixel's **alpha**, so transparent
regions match loosely and opaque ones strictly; alpha difference is unweighted
and therefore dominates. The UI slider value is squared before it gets here.

Fill is 4-connected and **scanline-based**: the queue holds contiguous runs,
not pixels. Selection clipping is done by pre-seeding the stencil with the
complement of the selection, so out-of-selection pixels look already-visited
and the fill simply cannot leave — turning a per-pixel containment test into a
free early-out.

## Not worth porting

`OctreeQuantizer` (pointer-chasing, the worst case for Python — use Pillow or
k-means and port only the two-pass structure), `ConvolutionFilterEffect`
(`scipy.ndimage.convolve` supersedes it), the GDI+ shape rasterisers (use
`QPainter` with antialiasing), and the entire WinForms application shell.
