# Project Brief

Last reviewed: 2026-07-26

## Purpose

`icBuilder` documents and implements the processing of IMAGE WIC, SI12, and
SI13 observations into height-integrated Hall and Pedersen ionospheric
conductance estimates with propagated uncertainties.

The repository is a builder, not the preferred reader for generated products.
Use the companion `icReader` project when the task is to consume conductance
files.

## Scope and outputs

The primary workflow:

1. maps raw sensor files to IMAGE orbits;
2. removes image backgrounds and writes per-sensor orbit NetCDF files;
3. time-aligns WIC, SI12, and SI13 observations;
4. projects and bins the measurements;
5. estimates characteristic energy, energy flux, Hall conductance, Pedersen
   conductance, and associated uncertainties;
6. writes per-orbit conductance NetCDF products.

The README links a published Zenodo dataset containing estimated conductances
and uncertainties. It describes the processed 2000–2001 IMAGE corpus as about
2.5 TB and available on request.

A secondary spline workflow writes smoothed conductance and factor files.
Plotting, grid-resolution, hyperparameter, and neural-network experiments also
exist, but they are not all part of the primary production sequence.

## Code map

- `icbuilder/preimage.py`: reads and prepares sensor images.
- `icbuilder/binnedimage.py`: bins observations and handles measurement
  dispersion.
- `icbuilder/conductanceimage.py`: constructs and serializes conductance
  products.
- `icbuilder/robinson.py` and
  `icbuilder/imagesat_e0_eflux_estimates.py`: physical conversion helpers.
- `icbuilder/splineimage.py`: spatial-temporal spline representation and
  uncertainty factors.
- `scripts/make_orbit_h5_files.py`: raw-file-to-orbit indices.
- `scripts/make_orbit_nc_files.py`: background removal and sensor orbit files.
- `scripts/make_conductance_orbit_files.py`: combined conductance products.
- `scripts/make_spline_model_files.py`: spline and factor products.
- `example_data/`: tracked representative inputs, intermediates, products, and
  figures; preserve unless regeneration is deliberate.
- `vault/`: technical project memory and historical research notes.

## External dependencies

The workflows rely on scientific packages beyond the empty dependency list in
`pyproject.toml`, including NumPy, SciPy, pandas/PyTables, `fuvpy`, `secsy`,
NetCDF4, ApexPy, and tqdm. Plotting, spline, and experimental workflows require
additional packages such as matplotlib, polplot, `icReader`, CHOLMOD-backed
tools, and machine-learning libraries.

Reconstruct the environment from live imports and a representative workflow
before attempting a full rerun.
