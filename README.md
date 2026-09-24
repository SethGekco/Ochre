# Ochre

A Linux-native, layered raster image editor. Paint.NET's approachability, built
for Linux, with an addon system capable enough that format support ships
separately from the editor.

Status: **working editor with C&C support.** Layers, groups, blend modes,
selections, tools, undo, adjustments and effects, a native document format,
an optional C accelerator, an addon system, and a Command & Conquer addon
that reads and writes SHP sprites and TMP terrain templates.

    ./tests/run_all.sh      # 43 checks, both backends
    python -m ochre.ui      # the editor

## The rules

These are invariants, not aspirations. Each one is enforced by a test.

- **`ochre/engine/` never imports Qt.** The engine is I/O-free and
  UI-free; `ochre/ui/` is the only Qt code, and `ochre/ui/controller.py` is
  Qt-free glue between them. Checked by `tests/test_no_qt.py`, which walks the
  AST of every engine module.
- **Everything tunable lives in `data/*.ini`.** The `DEFAULTS` dict in
  `ochre/engine/settings.py` *is* the schema — the type of each default is the
  coercion rule. Precedence is `DEFAULTS < data/*.ini < OCHRE_<KEY>`. Unknown
  keys are silently ignored, and that is the forward-compatibility story: a
  config written by a newer Ochre opens correctly in an older one, minus the
  settings it does not know about.
- **Dirty rects are the architecture, not an optimisation.** A full-canvas
  composite of 10 layers at 4000x4000 measures 1077 ms on the development
  machine. Nothing may composite the whole canvas on an input event.
- **Zero game-specific code in the core.** Command & Conquer sprite and terrain
  support is a separately-downloaded addon. The core may know "a palette can
  carry annotated index ranges"; it must never know what any of them mean. The
  test applied throughout: *could a third party ship that addon with no commit
  access to this repository?*
- **Every accelerated operation is specified in integer arithmetic**, so the
  optional C extension and the pure-numpy fallback agree bit-for-bit by
  construction rather than by luck. `accel/fallback.py` is the normative
  specification: when the two disagree, the C is wrong.

## Layout

```
ochre/engine/     I/O-free, Qt-free core
  accel/          numpy specification + optional C extension
  tools/          pencil, brush, eraser, bucket, picker, 4 selection tools
  fileio/         the .ochre container, and flat import/export
ochre/ui/         PySide6. viewstate/controller/bus/config are Qt-free.
ochre/addons/     the addon contract and host
data/*.ini        every tunable
addons/cnc/       Command & Conquer: SHP sprites, TMP terrain
addons/example/   a worked example of the addon API
tests/            standalone scripts, no pytest
```

Build the optional accelerator (never required):

    python setup.py build_ext --inplace

## Running the tests

```sh
./tests/run_all.sh        # verbose
./tests/run_all.sh -q     # one line per test
```

No display required; the runner forces `QT_QPA_PLATFORM=offscreen`. The suite
runs **twice** — once normally and once with `OCHRE_ACCEL=0` — so the numpy
fallback is a tested path rather than an untested contingency.

Each test is a standalone script that prints `ok: <name>` per check and exits
non-zero on the first failure, so any of them can be run directly:

```sh
python3 tests/test_geometry.py
```

## Licence

GPL-3.0-or-later. See [COPYING](COPYING).

Ochre draws on several existing projects, all licence-compatible:

| Project | Licence | Used for |
|---|---|---|
| [Paint.NET 3.36.7 / OpenPDN](https://github.com/rivy/OpenPDN) | MIT (with exceptions) | Blend-mode formulas, surface and history models, the effect property system |
| [GIMP](https://github.com/GNOME/gimp) | GPL-3.0 | Selection and flood-fill behaviour |
| [XCC Utilities](https://github.com/OlafvdSpek/xcc) | GPL-3.0 | C&C file formats (addon only) |
| [WorldAlteringEditor](https://github.com/CnCNet/WorldAlteringEditor) | GPL-3.0 | C&C file formats (addon only) |

Two deliberate exclusions from OpenPDN: its **artwork and resource assets**
(`.png`, `.resx`, `.resources`) are CC BY-NC-ND and are not used, and **GPC**
(the polygon clipper in `SystemLayer/GpcWrapper`) is non-commercial-encumbered
and is not used — Ochre represents selections as coverage masks rather than
polygon paths, so the dependency never arises.
