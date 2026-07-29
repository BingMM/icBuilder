# Processing Pipeline

Last reviewed: 2026-07-26

This note records the durable high-level workflow visible in the README and
live code. It is orientation, not a claim that the full production dataset was
reproduced during the review.

## Primary orbit workflow

1. **Orbit indexing**
   `scripts/make_orbit_h5_files.py` assigns raw WIC, SI12, and SI13 filenames
   to orbit intervals and writes sensor-specific HDF5 indices.

2. **Sensor preprocessing**
   `scripts/make_orbit_nc_files.py` reads the raw images, selects northern
   observations, applies the `fuvpy` background models, and writes compressed
   per-sensor orbit NetCDF files plus availability arrays.

3. **Optional inspection and grid choice**
   Background-removal plotting and grid-resolution scripts support inspection.
   The README records 225 km for WIC and 450 km for SI12/SI13 as suggested
   resolutions; treat those as documented configuration, not newly validated
   values.

4. **Conductance construction**
   `scripts/make_conductance_orbit_files.py` time-aligns the three instruments,
   removes empty or low-coverage frames, converts coordinates, grids the
   observations, and builds a `ConductanceImage`.

5. **Serialization**
   `icbuilder/conductanceimage.py` writes sensor statistics, characteristic
   energy, energy flux, Hall and Pedersen conductance, propagated
   uncertainties, weights, times, and grid metadata to per-orbit NetCDF.

## Secondary spline workflow

`scripts/make_spline_model_files.py` reads conductance products through
`icReader`, constructs `icbuilder.splineimage.SplineImage`, and writes spline
and factor NetCDF products. Historical parameter exploration and resolution
discussion remain in the dated vault notes.

Before trusting a historical parameter or resolution claim, locate its input,
configuration, commit, and generated result. The 2026-07-26 vault migration did
not reproduce those analyses.

## Execution boundary

The processing scripts write or overwrite data products. Use an isolated
copied base directory, ensure output directories exist, start with serial
execution on a representative orbit, and verify dependencies before scaling
up. Do not use the tracked `example_data/` tree as scratch space.
