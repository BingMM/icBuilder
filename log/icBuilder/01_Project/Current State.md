# Current State

Last reviewed: 2026-07-26
Code snapshot: `main` at `3fc2fc6`
Upstream at review: aligned with `origin/main` before the vault migration

## Current position

**Confirmed by the user:** the project is complete for now. It should be
revisited later, but no revisit date, trigger, or scope has been defined.

The 2026-07-26 vault migration is documentation-only. It does not establish new
scientific validation, regenerate the conductance dataset, or change the
implementation state represented by the code snapshot above.

## Confirmed

- The repository processes IMAGE WIC, SI12, and SI13 data into Hall and
  Pedersen conductance estimates with propagated uncertainties.
- The README links the published conductance dataset and describes the
  processed 2000–2001 IMAGE corpus as about 2.5 TB.
- The primary scripts for orbit indexing, background-removed NetCDF creation,
  and conductance-orbit generation are present.
- Conductance, spline, representative raw/intermediate data, and figures are
  tracked under `example_data/`.
- The established `log/icBuilder/` vault contains dated technical history and
  figures from 2025.
- No automated test suite or CI workflow was found.
- On 2026-07-26, all 25 Python files under `icbuilder/` and `scripts/` passed a
  read-only AST syntax parse.

## Deferred questions for a future revisit

- What event or downstream need should reopen the project?
- Can the processing environment be reconstructed reproducibly from declared
  dependencies rather than live imports?
- Which small representative orbit should become a safe functional
  verification case?
- Do the published data, tracked examples, and current scripts share
  adequately recorded configuration and provenance?
- Which historical spline or neural-network conclusions remain relevant?

## Known maintenance and verification gaps

- `pyproject.toml` declares no runtime dependencies even though the workflows
  require a substantial scientific stack.
- The README marks conductance figures as unimplemented even though
  `scripts/make_conductance_figures.py` exists, and it names an absent
  `make_spline_model.py` rather than the live
  `scripts/make_spline_model_files.py`.
- `scripts/make_orbit_h5_files.py` and several exploratory scripts contain
  machine-specific paths.
- Some workflows can overwrite tracked HDF5, NetCDF, NumPy, spline-factor, and
  figure outputs and can use large worker counts.
- `icbuilder/conductanceimage.py` currently sets energy/flux covariance to zero
  as an implementation placeholder; this review makes no claim about its
  scientific impact.
- The full production pipeline and published dataset were not regenerated or
  independently validated during this documentation review.

Older dated task lists remain historical evidence. The user's current
completion statement supersedes them as project status.
