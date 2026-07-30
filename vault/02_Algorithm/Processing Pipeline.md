# Processing Pipeline

Last reviewed: 2026-07-29

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

   The current uncommitted `BinnedImage` path can apply the provisional
   pixel-level `cos(DZA)` normalization before calculating image statistics.
   Each binned sensor image retains median SZA, median DZA, median
   `cos(DZA)`, and whether the correction was selected. The geometry uses the
   same valid source pixels as the brightness statistic and follows SI/SI13
   target-grid interpolation and later frame selection. Median
   `cos(DZA)` is diagnostic: because correction precedes the image median, it
   is not generally an exact multiplier between corrected and uncorrected
   binned brightness.

5. **Serialization**
   `icbuilder/conductanceimage.py` writes sensor statistics, characteristic
   energy, energy flux, Hall and Pedersen conductance, propagated
   uncertainties, weights, times, and grid metadata to per-orbit NetCDF.
   It also writes per-sensor `sza`, `dza`, and `los_factor` arrays with units.
   Root attributes record each sensor's source-image correction (`SH`, `DG`,
   or raw) and whether the pixel-level LOS correction was applied.

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

## Exploratory Zhang–Paxton latitude collapse

`scripts/ZhangPaxton2008_collapse.py` reduces the external Zhang–Paxton model
from `ZP(Kp, MLT, MLAT)` to one electron mean energy for each `(Kp, MLT)`.
This is not yet part of the production orbit workflow.

For each fixed Kp and MLT, it evaluates E0 and Q at the centres of 0.01-degree
cells covering 50–90 degrees northern MLAT. It identifies the principal Q
maximum and retains only the contiguous cells around that maximum where Q is
above the selected threshold. The area-weighted mean and median E0 are then
calculated conditionally over those cells with exact spherical-area factors
`sin(latitude_upper) - sin(latitude_lower)`. Consequently, the large
near-zero background outside the selected oval cannot dominate the result,
and poleward cells do not receive the same weight as larger equatorward
cells.

The provisional default is `Q > 0.05 mW m-2`, matching the Figure 8
global-mean inclusion criterion in Zhang and Paxton (2008). The only
sensitivity definition retained is the absolute `Q > 0.25 mW m-2` criterion
used as an auroral-boundary contour in the paper. The user rejected the
relative 10%-of-peak definition because it changes with the strength of each
slice. Empty masks produce NaN rather than a fabricated fallback. Contact
with the 50-degree equatorward limit is flagged as possible truncation;
extension to 90 degrees is separately reported as contact with the physical
pole.

The function also returns weighted spread, a Q-weighted energy mean, latitude
bounds, selected area, and selection diagnostics. The area-weighted mean
remains the primary representative; the area-weighted median describes a
typical unit area and is reported alongside it. The Q-weighted mean emphasizes
the precipitation power and remains a sensitivity result.

The model equations are continuous in MLT because their MLT dependence is a
six-harmonic Fourier series. The collapse therefore evaluates the exact MLT
provided by the caller rather than averaging within a finite MLT sector.
Diagnostic figures use 0.05-hour sampling. Zhang and Paxton fitted the Fourier
parameters from 48 empirical sectors with 0.5-hour width; evaluating the
analytic fit more densely resolves its smooth variation but does not increase
the underlying empirical resolution.

The Epstein latitude profiles are likewise continuous. The 0.01-degree MLAT
grid is deliberately finer than the earlier 0.25-degree diagnostic grid
because a hard threshold on coarse cells made the collapsed curves visibly
jitter as boundary cells entered and left the selection. This is numerical
oversampling, not added physical resolution.

### Integration boundary

Do not pass this result directly as `ConductanceImage.Ep`: that parameter is
proton characteristic energy. Electron E0 is currently inferred from the
corrected WIC/SI13 ratio, including hard-coded low-signal fallbacks.
**Confirmed by the user:** comparison with DMSP did not reproduce the
empirical WIC/SI13-to-E0 relation attributed to Frey et al. (2003). The
integration target is therefore a complete replacement of IMAGE-derived E0
and dE0, not a fallback or blend. Before implementation, confirm whether
Zhang–Paxton mean energy is compatible with the energy quantity required by
the WIC response and conductance equations.

If integration is approved, do not recompute the 0.01-degree latitude collapse
for each IMAGE frame. The IMAGE products use a fixed 36-by-36 Cubed-Sphere
grid with fixed MLT and MLAT at each cell. Precompute the selected
representative energy for every supported Kp state and grid cell, store the
scientific configuration with the table, and reduce frame-time evaluation to
Kp lookup.

Because the present collapse deliberately removes MLAT dependence, each table
cell would contain `E0(Kp, MLT_cell)`; cells sharing an MLT receive the same
value. The cell MLAT would become relevant only if a future decision returned
to the uncollapsed `E0(Kp, MLT, MLAT)` model.

The user does not consider an additional automatic IMAGE precipitation mask
robustly achievable, so do not make one an integration requirement. Evaluate
the collapsed E0 on the fixed grid and let corrected WIC brightness carry the
observed spatial structure into `Fe = Wprime / Wm(E0)`. Existing IMAGE
coverage, background correction, and validity handling still apply, but they
should not be reinterpreted as a newly inferred auroral-oval boundary. This
means the product must be described honestly: its E0 is a Kp/MLT
representative, while IMAGE determines where appreciable radiance-derived
energy flux occurs.

The published Zhang–Paxton coefficients do not contain uncertainty or
fit-covariance information. The existing collapse spread and threshold
sensitivity do not recover that missing model-fit uncertainty. **Confirmed by
the user:** do not introduce DMSP calibration. Define dE0 from the
Zhang–Paxton latitude profile that is collapsed to form the mean. The existing
`weighted_spread` return value is the direct candidate:

`dE0 = sqrt(sum(w * (E0_lat - E0_mean)^2) / sum(w))`,

where `w` is the same exact spherical latitude-cell area used for the mean and
the sum covers the same contiguous Q-selected interval. Do not divide this
spread by the square root of the number of latitude samples: the 0.01-degree
cells are numerical samples of one profile, not independent observations.
This dE0 represents unresolved latitude variability created by collapsing
away MLAT. It must not be described as Zhang–Paxton predictive uncertainty or
coefficient error.

The replacement calculation should conceptually be:

1. obtain `E0 = collapsed_ZP(Kp, MLT)` from the precomputed lookup;
2. obtain `dE0` from the area-weighted spread of the same selected
   Zhang–Paxton latitude profile;
3. retain corrected IMAGE WIC brightness `Wprime` and calculate
   `Fe = Wprime / Wm(E0)`;
4. propagate WIC and E0 uncertainty through Fe and the Robinson conductance
   equations.

Although Zhang–Paxton E0 is independent of the IMAGE count noise,
`Fe = Wprime / Wm(E0)` makes the resulting E0 and Fe statistically dependent.
The current conductance implementation sets their covariance to zero as a
placeholder. A first-order implementation must include the derivative-induced
covariance; an ensemble or Monte Carlo propagation through the response and
conductance equations would represent the nonlinearity more faithfully.

Once E0 no longer comes from WIC/SI13, SI13 is not intrinsically required for
electron-energy inference. Decide separately whether it remains in the
pipeline for quality control, validity screening, or validation. This
decision could change frame-availability requirements and product provenance;
WIC remains the electron-brightness input and SI12 remains part of proton
correction.

### Proposed SI13 role after E0 replacement

Do not use SI13 to invert for E0. Its cleanest remaining use is an independent
radiometric consistency diagnostic for frames where simultaneous SI13 exists:

1. calculate corrected WIC electron counts `Wprime` and corrected SI13
   electron counts `Sprime` with the existing dayglow and SI12-based proton
   corrections;
2. take E0 and dE0 from the collapsed Zhang–Paxton lookup;
3. calculate the primary flux estimate
   `Fe_WIC = Wprime / WIC_response(E0)`;
4. predict the SI13 electron signal
   `Sprime_predicted = SI13_response(E0) * Fe_WIC`;
5. retain `Sprime - Sprime_predicted`, and preferably its normalized residual,
   as a validation variable.

Equivalently, calculate
`Fe_SI13 = Sprime / SI13_response(E0)` and compare it with `Fe_WIC`. This does
not recreate the rejected ratio-to-E0 inversion: E0 remains fixed by
Zhang–Paxton, while the two cameras test whether they recover a consistent
energy flux.

The initial recommendation is to use SI13 for validation only and not require
it for product generation. If the two independently derived flux estimates
prove consistent after proton/background correction, a later joint Fe
estimator could be considered. Such a combination must account for their
shared E0 and SI12-proton uncertainties and should not use naive
inverse-variance weights as if the channels were independent.

For arbitrary Kp, evaluate the published interpolation before performing the
latitude collapse. The six empirical bins are represented at Kp centres 0.75,
2.25, 3.75, 5.25, 7, and 9. Mean energy is interpolated linearly in Kp;
energy flux is interpolated in the paper's nonlinear hemispheric-power
coordinate. Selecting the nearest empirical bin would introduce artificial
discontinuities.
