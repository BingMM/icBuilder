# Repository guidance

## Purpose and project tracks

`icBuilder` processes IMAGE WIC, SI12, and SI13 observations into
height-integrated Hall and Pedersen conductance estimates with propagated
uncertainties.

Keep these parts of the repository distinct:

- `icbuilder/` contains the reusable image, binning, conductance, and spline
  classes.
- `scripts/make_orbit_h5_files.py`, `scripts/make_orbit_nc_files.py`, and
  `scripts/make_conductance_orbit_files.py` form the primary orbit-processing
  workflow.
- plotting, resolution, spline, and neural-network scripts are secondary or
  exploratory workflows and are not all described accurately by the README.

Do not infer current project status from old dated notes. Check Git and the
live code, then read the current-state and handoff notes.

## Project memory

This repository preserves its established Obsidian vault at `log/icBuilder/`.
Before substantive work, read:

1. `log/icBuilder/01_Project/Current State.md`
2. `log/icBuilder/05_Handoff/Handoff - Latest.md`
3. `log/icBuilder/01_Project/Project Brief.md`

Read `log/icBuilder/02_Algorithm/Processing Pipeline.md` when changing the
processing workflow. Treat the older dated notes and images at the vault root
as historical evidence, not current instructions.

Use this source-of-truth order:

1. live code, Git state, tests, and regenerated results;
2. `Current State.md` and `Handoff - Latest.md`;
3. decision and algorithm notes;
4. dated historical notes.

If memory conflicts with live evidence, follow the repository and update the
live notes when the task changes project understanding.

## Working rules

- Preserve unrelated or pre-existing worktree changes.
- Use repository-relative paths in documentation.
- Do not run processing scripts against tracked `example_data/` unless
  overwriting versioned artifacts is explicitly intended.
- Use a copied or scratch `--base` directory for trial runs, and verify that
  all expected output directories exist before starting.
- Begin with a small representative orbit and serial execution before
  increasing worker counts. Some workflows are memory-intensive and have high
  multiprocessing defaults.
- Do not run the full external IMAGE corpus unless the input location,
  environment, runtime, and output destination are explicitly in scope.
- Do not commit the external multi-terabyte corpus, credentials, caches, or
  regenerated products solely because they are near the project vault.
- Treat tracked example data, figures, and historical parameter claims as
  evidence requiring provenance, not automatically current validation.
- Use `icReader` rather than this package when the task is only to read
  generated conductance products.

## Verification

No automated test suite or CI workflow was present at the 2026-07-26 review.
A safe syntax-only check that does not write bytecode is:

```bash
python -c "import ast, pathlib; files=list(pathlib.Path('icbuilder').rglob('*.py'))+list(pathlib.Path('scripts').rglob('*.py')); [ast.parse(p.read_text(), filename=str(p)) for p in files]; print(f'parsed {len(files)} files')"
```

For documentation-only changes, also run:

```bash
git diff --check
```

Functional verification requires external dependencies and can overwrite HDF5,
NetCDF, NumPy, spline-factor, or figure outputs. Before running a pipeline,
confirm the environment, use an isolated data copy, and inspect the selected
script's path and worker defaults.

## Memory closeout

After a meaningful project session:

1. create `log/icBuilder/04_Sessions/YYYY-MM-DD.md` only when historical
   detail is worth preserving;
2. rewrite `Current State.md` when verified project state changed;
3. append only durable choices to the decision log;
4. replace obsolete content in `Handoff - Latest.md`;
5. update algorithm notes only when scientific interpretation changed.

Never append new work to an older dated note.
