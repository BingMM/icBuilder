# Handoff - Latest

Last updated: 2026-07-26
Verified code snapshot: `main` at `3fc2fc6`

## Project state

`icBuilder` processes IMAGE WIC, SI12, and SI13 observations into Hall and
Pedersen conductance products with propagated uncertainties. The user considers
the project complete for now and wants it revisited later. No revisit date or
trigger is confirmed.

The vault modernization is documentation-only. It does not change the code,
regenerate products, or establish new scientific validation.

## Live repository snapshot

At the 2026-07-26 review:

- `main` was clean and aligned with `origin/main` at `3fc2fc6` before the vault
  migration;
- the primary orbit-index, background-removal, conductance, and spline scripts
  were present;
- all 25 Python files under `icbuilder/` and `scripts/` passed a read-only AST
  syntax parse;
- no automated tests or CI workflow were found;
- `pyproject.toml` declared no dependencies despite many runtime imports;
- the README's final workflow steps did not fully match the live script names
  and files;
- the full external-data pipeline and published dataset were not rerun.

Inspect Git and live code again when the project is reopened. This note is
orientation, not authority.

## Next action when revisited

1. Confirm why the project is being reopened and what output must change.
2. Reconstruct and document the required environment from live imports.
3. Select a small representative orbit and copy its inputs to an isolated
   scratch base.
4. Reconcile the README workflow, dependency metadata, and live entry points.
5. Define functional and scientific checks before rerunning broader data.

Do not start by regenerating the tracked `example_data/` tree or the full
external corpus.

## Risks and verification limits

- Production inputs are large and external to the repository.
- Several scripts write or overwrite HDF5, NetCDF, NumPy, factor, and figure
  outputs.
- Some scripts contain machine-specific paths or high multiprocessing
  defaults.
- Tracked examples and historical notes do not by themselves prove current
  provenance or scientific validity.
- Historical spline and neural-network findings have not been independently
  reproduced in the current review.

## Entry points

- `README.md`
- `scripts/make_orbit_h5_files.py`
- `scripts/make_orbit_nc_files.py`
- `scripts/make_conductance_orbit_files.py`
- `icbuilder/conductanceimage.py`
- `scripts/make_spline_model_files.py`
- `02_Algorithm/Processing Pipeline.md`

## Historical evidence

The dated notes and attachments at the `log/icBuilder/` root are preserved as
legacy session history. Read them only when investigating a specific earlier
decision, configuration, or result.
