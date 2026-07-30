# Current State

Last reviewed: 2026-07-29
Repository snapshot: `main` at `8e509ea`
Upstream state: aligned with `origin/main` before the uncommitted
Zhang–Paxton-collapse work

## Current position

**Confirmed by the user:** the project has been reopened for a bounded
Zhang–Paxton (2008) electron-energy reduction. The immediate purpose is to
collapse `ZP(Kp, MLT, MLAT)` to one representative electron mean energy per
`(Kp, MLT)` pair. Integration into the conductance pipeline is deliberately
deferred.

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
- `scripts/ZhangPaxton2008_collapse.py` now provides a reusable, documented
  latitude collapse and a headless figure-generation CLI.
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
- Ten focused tests pass in `tConductance`: seven Zhang–Paxton-collapse tests,
  two binned viewing-geometry tests, and one ConductanceImage NetCDF
  round-trip test. All 29 Python files under
  `icbuilder/`, `scripts/`, and `tests/` pass AST parsing.
- Four explanatory figure sets were generated as PNG and PDF under
  `figures/` and visually inspected. The fourth is a direct companion to the
  collapsed-mean result and maps the profile-derived dE0 (`weighted_spread`)
  over Kp/MLT with the same threshold-sensitivity layout. The polar-map
  regression test requires 90 degrees at the centre and 50 degrees at the
  outer edge. SVG is not generated.
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
- The current conductance path propagates scalar `dE0` and `dFe` through the
  Robinson formulas but sets `var(E0, Fe)` to zero as a placeholder. Because
  `Fe = Wprime / Wm(E0)`, replacing E0 does not make E0 and Fe independent.
  Any Zhang–Paxton integration must preserve uncertainty and explicitly
  calculate or sample that covariance.

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
  uncertainty chain also omits shared-channel covariance and hard-codes
  E0–Fe covariance to zero.
- The full production pipeline and published dataset were not regenerated
  during this audit.

Older dated task lists remain historical evidence. The user's current
reopening statement supersedes the earlier "complete for now" status for this
bounded work.
