# Current State

Last reviewed: 2026-08-05
Repository snapshot: `main` at `ff957b5`
Upstream state: uncommitted verified Kp-range and orbit-restart changes

## Current position

The first-stage Zhang–Paxton (2008) integration is implemented. The orbit
pipeline now assigns definitive GFZ Kp to each final IMAGE frame and replaces
the WIC/SI13-derived E0 and dE0 with the fixed-grid lookup. The stage keeps the
existing three-camera common-frame population and SI13 ratio diagnostics so
old and new products can be compared without simultaneously changing frame
support.

The first full Halley run exposed that the bundled Kp series stopped at the
end of 2001 although the IMAGE inputs continue through June 2003. The local
series now extends through July 2003. Orbit processing also resumes by default
from structurally valid output files and publishes new files atomically.

The current implementation and generated figures are uncommitted. A full
read-only audit examined the live pipeline and five tracked example
conductance products. The user has since corrected the SI-grid construction
and regenerated the two tracked example orbit products and their figures.

**Audit verdict:** the repository is a useful research prototype, but the
current pipeline and products are not publication-ready. The Frey response
table is correctly transcribed; the observed WIC/SI13 problem is not explained
by a simple coefficient or ratio-orientation bug. Current E0 products are
dominated by hard fallbacks and saturation, while confirmed grid, resolution,
clipping, proton-correction, and uncertainty defects amplify the problem.
Calibration provenance and response validity remain the leading unresolved
causal boundary. See [[Audit - 2026-07-29]].

## Confirmed

- The repository processes IMAGE WIC, SI12, and SI13 data into Hall and
  Pedersen conductance estimates with propagated uncertainties.
- `icbuilder/zhang_paxton_collapse.py` provides the reusable documented
  latitude collapse as direct NumPy calculations returning ordinary
  dictionaries. Diagnostic plotting and the command-line entry point are
  separate in `scripts/ZhangPaxton2008_collapse.py`.
- The collapse refactor reduced the reusable scientific module from 1,065 to
  306 lines and removed three dataclasses, array type machinery, plotting,
  and command-line code from it. The four existing diagnostic figure sets
  remain reproducible from the separate script.
- Old and refactored collapse outputs agree exactly for scalar, broadcast,
  broad Kp/MLT, both threshold, empty-selection, non-finite, and sampling-edge
  cases. The bundled lookup file is byte-for-byte unchanged.
- The collapse selects the contiguous Q-above-threshold component containing
  the principal Q maximum, then uses exact spherical latitude-cell weights
  proportional to `sin(latitude_upper) - sin(latitude_lower)`.
- It returns the conditional area-weighted E0 mean and median, weighted spread,
  Q-weighted sensitivity mean, selected bounds and area, empty-mask flags,
  possible equatorward truncation flags, and physical-pole contact.
- **Confirmed by the user:** retain the area-weighted mean as the primary
  representative and calculate the area-weighted median alongside it.
- **Confirmed by the user:** remove the relative 10%-of-peak threshold. The
  remaining absolute definitions are `Q > 0.05 mW m-2` (default) and
  `Q > 0.25 mW m-2` (sensitivity).
- The collapse accepts arbitrary MLT values and performs no within-MLT-bin
  averaging. The diagnostic figures now sample the continuous Fourier model
  every 0.05 MLT hour. The published fit used 48 empirical sectors of width
  0.5 MLT hour, but the fitted equations themselves are continuous.
- The default MLAT grid is now 0.01 degrees. This is numerical oversampling of
  the continuous Epstein profiles to suppress hard-threshold discretization
  jitter; it is not a claim of 0.01-degree empirical accuracy.
- **Confirmed by the user:** production use should not repeat the latitude
  collapse for every IMAGE frame. IMAGE frames share a fixed 36-by-36
  Cubed-Sphere grid with fixed MLT/MLAT coordinates, so the representative
  energy should be precomputed once on that grid for each Kp state and used as
  a lookup table.
- The lookup is implemented and bundled as an 8.4-MB NetCDF table with
  dimensions `(901 Kp, 36 eta, 36 xi)`. It contains direct collapsed layers
  for Kp 0.00--9.00 at 0.01 spacing. The loader rounds input Kp to the nearest
  hundredth, performs no interpolation, and directly checks the table's
  two-dimensional coordinates against the active grid.
- The lookup code was simplified for a small scientific codebase. Generation
  is a visible Kp loop with optional `process_map`; the loader is one function
  returning an ordinary dictionary. The NetCDF stores only Kp, grid
  coordinates, E0, dE0, median E0, units, and scientific provenance.
- The simplified table was made by copying the three previously verified
  scientific arrays into the smaller schema, not by rerunning all 901 costly
  latitude collapses. Every stored value and coordinate is exactly equal to
  the previous table. Fresh direct collapses at Kp 0.00, 1.52, and 9.00 agree
  at the expected float32 precision (maximum tested difference
  `2.4e-7 keV`).
- MLT is correctly retained as a two-dimensional grid coordinate:
  `(grid.lon / 15) % 24`. A native `(xi, eta)` diagnostic shows the MLT,
  E0, and dE0 fields together.
- All 45 focused tests pass. They cover the collapse, lookup and grid,
  definitive-Kp integrity and boundary matching, paired E0 override, SI13
  invariance, induced covariance, zero-flux Robinson propagation, geometry,
  and NetCDF provenance. Selected Kp=1.52 lookup cells agree with direct
  collapse values to float32 storage precision.
- A serial orbit-0085 run completed in an isolated `/tmp` output in 21
  seconds. It retained the old 20-frame shape, assigned original Kp 2.667 and
  4.333 to lookup layers 2.67 and 4.33, and produced 18,958 finite E0 pixels.
  Every finite E0 exactly equalled its selected lookup value; all corresponding
  covariance values were finite, with zeros only where WIC-derived Fe was
  zero. Tracked example products were not modified.
- The lookup contains no empty selections. Under its provisional
  `Q > 0.05 mW m-2`, 50--90-degree configuration, 231,123 of 1,167,696 table
  cells touch the 50-degree equatorward sampling limit. The table therefore
  works technically but does not settle the production latitude-domain
  decision.
- The bundled GFZ JSON response contains 10,464 definitive three-hour Kp
  intervals from 2000-01-01 00:00 through 2003-07-31 21:00 UTC. It records
  source, DOI `10.5880/Kp.0001`, CC BY 4.0 licence, query, acquisition date,
  and the SHA-256 of the file actually loaded. Orbit processing uses only this
  local copy. Structural validation replaces the former hard-coded checksum
  gate and record-count/date checks.
- IMAGE frame times are matched to half-open Kp intervals
  `[start, start + 3 h)`. Exact three-hour and midnight boundaries select the
  new interval. Gaps and out-of-range frames fail instead of being
  interpolated, filled, or clipped. The timezone-free IMAGE times are
  explicitly interpreted as UTC.
- Orbit processing treats a structurally valid `or_XXXX.nc` as its completion
  record. By default it skips valid products and reruns missing or invalid
  ones; `--overwrite` requests a full recomputation. Each worker writes and
  validates `or_XXXX.nc.partial` before an atomic same-directory rename, so a
  crash cannot expose a partial product under the final name.
- `ConductanceImage` loads all requested lookup layers once per orbit,
  validates the `(time, 36, 36)` shape and grid, and preserves both original
  thirds-valued GFZ Kp and nearest-hundredth lookup Kp. It also verifies that
  every frame time lies within its serialized half-open three-hour interval.
- The count conversion accepts E0/dE0 only as a pair. Under the override it
  retains SI12 proton correction, WIC proton subtraction, Fe and dFe
  propagation, and SI13 R/dR diagnostics, while skipping the ratio-to-E0
  inversion. Finite SI13 changes therefore do not change E0 or Fe.
- The first-order covariance induced by `Fe = Wprime / Wm(E0)` is now
  calculated as
  `-Wprime * Wm'(E0) / Wm(E0)^2 * dE0^2`, passed to both Robinson uncertainty
  functions, and serialized. It does not add model-coefficient or Kp
  uncertainty.
- At zero Fe, the Robinson flux derivative is singular. dP and dH are now
  defined as one-sided conductance excursions from `Fe=0` to `Fe=dFe`. This
  remains finite below, at, and above the former 4-keV Pedersen singularity.
- All 41 Python files under `icbuilder/`, `scripts/`, and `tests/` pass AST
  parsing. The four collapse figure sets and the native-grid lookup diagnostic
  were visually inspected.
- On the diagnostic grid Kp 0–9 by 0.05-hour MLT (4,800 slices), using
  0.01-degree MLAT cells over 50–90 degrees, the default rule produced no
  empty slices. It reached the 50-degree equatorward sampling limit in 1,001
  slices and the physical 90-degree pole in 1,378 slices. Pole contact is not
  sampling truncation.
- A comparison against 0.01-hour MLT sampling found that linear interpolation
  of the 0.05-hour area-mean grid has a 0.021 keV 99th-percentile error and a
  0.037 keV maximum error. The selected 0.05-hour diagnostic spacing is
  therefore numerically adequate for the mean-energy product.
- The visible jitter in the Kp=2 and Kp=5 threshold-sensitivity curves was a
  0.25-degree MLAT selection artifact. Refining to 0.01 degrees reduced the
  RMS second difference from 0.016–0.023 keV to 0.00095–0.00124 keV across
  those four plotted curves.
- The Frey response nodes and WIC/SI13 orientation in
  `imagesat_e0_eflux_estimates.py` are correct.
- Across the five tracked conductance products, only about 1.7–11.2% of valid
  pixels are interior response-table retrievals. Most are exact 0.2-, 1-, or
  25-keV fallback/saturation values.
- For orbit 0085, the median uncorrected binned WIC/SI13 ratio with SI13 above
  three counts is about 208, already above the response-table maximum near
  136.5. Fixed-2-keV proton correction raises the comparable median to about
  244.
- The former SI-grid construction error is fixed in the current uncommitted
  work. The resulting 18-by-18 SI grid has exactly every second edge of the
  36-by-36 WIC grid and pairwise-aligned centres. Separately, 450-km SI13 is
  still interpolated to the 225-km WIC grid without matching effective
  resolution.
- The example sensor times differ by only 0–1 seconds; time matching does not
  explain the observed example failure.
- The upstream Laundal/Østgaard FUVIEW3 product uses corrected counts,
  including detector flat-field and mission-time/temperature correction.
  This reduces the concern that icBuilder is receiving uncorrected raw
  detector counts, but the generated conductance files do not preserve the
  exact upstream recipe.
- Østgaard et al. (2018) used solar- and satellite-zenith-angle-dependent
  dayglow subtraction but did not apply a quantitative correction of auroral
  intensities from oblique views to nadir. Their analysis avoided absolute
  intensity comparisons. icBuilder cannot do so because WIC brightness
  determines Fe. The correct long-term treatment is an angle-dependent
  instrument/atmosphere forward response; multiplication by `cos(DZA)` is
  only a first-order plane-parallel diagnostic to validate over moderate DZA.
- The current uncommitted code loads SZA/DZA and can multiply corrected pixels
  by `cos(DZA)` before binning. The arithmetic and ordering are correct for the
  provisional plane-parallel diagnostic. `BinnedImage` now preserves median
  SZA, median DZA, median `cos(DZA)`, and the correction flag through binning,
  target-grid interpolation, and frame selection; `PreImage.discard` now also
  retains raw-image alignment. `ConductanceImage` now preserves and serializes
  all three geometry fields separately for WIC, SI12, and SI13, together with
  each channel's image-correction mode and LOS-applied flag. The scientific
  approximation remains provisional: it defaults on and is enabled for all
  three channels including SI12.
- `BinnedImage` now groups populated source pixels once by a flattened bin
  number, caches Student-t and chi-square multipliers by sample count, and
  shares SI-to-WIC triangulations only among fields with identical non-NaN
  source masks. A complete orbit-0085 scratch product remained byte-for-byte
  identical while elapsed time fell from 25.83 to 18.94 seconds locally.
- `ConductanceImage` now computes the combined weight and orbit-invariant
  proton responses once, reuses persistent camera-response interpolators, and
  applies the Zhang--Paxton production equations as masked float64 arrays.
  The `image_ratio` comparison remains scalar. On the same tracked orbit-0085
  inputs, elapsed time fell from 18.84 to 7.52 seconds and the complete
  32-variable NetCDF remained byte-for-byte identical. Profiled conductance
  calculation time fell from 20.41 to 0.004 seconds; the before profile
  included 208,538 repeated SciPy interpolator constructions. Maximum RSS was
  effectively unchanged at 349.2 versus 349.5 MB. Full-orbit vector/scalar
  comparison gave exact E0, dE0, Fe, R, covariance, P, H, weights, and NaN
  patterns. Maximum uncertainty differences were `4.4e-16` (dFe), `2.9e-11`
  (dR), and `1.8e-15` (dP/dH), all attributable to vector floating-point
  evaluation order.

## Immediate scientific questions

- Is `Q > 0.05 mW m-2` too inclusive for a per-MLT auroral-oval
  representative, given the equatorward-limit and polar-cap contacts?
- **Confirmed by the user:** do not introduce a new IMAGE-derived
  precipitation-support mask; no sufficiently robust automatic definition is
  available. The collapsed E0 may be evaluated over the grid, while corrected
  WIC brightness supplies the observed spatial amplitude through the Fe
  calculation. Existing IMAGE coverage and background handling remain, but
  are not promoted into a new scientific oval mask.
- Is Zhang–Paxton electron **mean** energy compatible with the electron
  characteristic-energy quantity expected by the IMAGE conversion?
- **Confirmed by the user:** attempts to reproduce the Frey et al. (2003)
  WIC/SI13-to-E0 relation with DMSP did not establish the claimed empirical
  connection. The IMAGE-derived E0 and dE0 are therefore to be replaced
  entirely by a Zhang–Paxton-based estimate rather than retained, blended, or
  used as calibration truth.
- `ConductanceImage.Ep` is proton characteristic energy and must not receive
  this electron quantity.
- **Confirmed by the user:** do not introduce DMSP into the replacement
  estimator. Use the spherical-area-weighted spread of E0 within the same
  selected Zhang–Paxton latitude profile as dE0. This quantifies the latitude
  variability discarded by the collapse; it is not a published
  Zhang–Paxton coefficient or predictive-error estimate.
- Replacing the ratio-derived E0 removes SI13 from that inference path.
  **Proposed, not decided:** retain simultaneous SI13 as an optional,
  independent radiometric validation channel. Given Zhang–Paxton E0 and the
  WIC-derived Fe, predict corrected SI13 electron counts and compare them with
  observed corrected SI13. Do not invert that comparison to modify E0 or make
  SI13 availability a prerequisite for every conductance frame. WIC remains
  necessary for electron brightness and SI12 for proton correction.
- Stage 2 remains separate: decide whether SI13 should become optional and
  whether it can provide a useful validation product before changing common
  frame selection, pixel support, or the combined weight.

## Other maintenance and verification gaps

- The existing IMAGE conductance dataset requires correction and validation
  before publication use. The source product's basic flat-field and temporal
  calibration are now documented, while atmosphere and oblique-view response
  remain unresolved.
- `pyproject.toml` declares no runtime dependencies even though the workflows
  require a substantial scientific stack.
- No general automated test suite or CI workflow exists beyond the new focused
  collapse tests.
- Several processing scripts contain machine-specific paths or high
  multiprocessing defaults.
- Generated products do not preserve enough source, calibration,
  configuration, unit, sample-count, and retrieval-status provenance for a
  forensic reconstruction.
- `SplineImage.solverP` passes Hall uncertainty to the Pedersen solver. The
  uncertainty chain still omits shared-channel covariance beyond the newly
  implemented E0–Fe term.
- The full production pipeline and published dataset were not regenerated
  during this audit.

Older dated task lists remain historical evidence. The user's current
reopening statement supersedes the earlier "complete for now" status for this
bounded work.
