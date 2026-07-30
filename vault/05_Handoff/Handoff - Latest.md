# Handoff - Latest

Last updated: 2026-07-29
Repository snapshot: `main` at `8e509ea`
Worktree state: uncommitted Zhang–Paxton collapse, tests, figures, memory,
SI-grid fix, and regenerated example products

## Project state

`icBuilder` has been reopened for a bounded Zhang–Paxton (2008) investigation.
`scripts/ZhangPaxton2008_collapse.py` now implements a readable, reusable
reduction from `ZP(Kp, MLT, MLAT)` to an oval-conditional representative
electron mean energy for each `(Kp, MLT)`. It also generates three explanatory
process/mean figures plus a fourth companion figure mapping the
profile-derived dE0 as PNG and PDF. It is not connected to the production IMAGE
pipeline.

The Zhang–Paxton implementation did not read or modify IMAGE data or tracked
`example_data/` products. The user subsequently fixed the independent
SI-grid construction defect and regenerated orbit 0085/0086 products and
figures.

A subsequent full read-only audit did read the five tracked example
conductance products and the representative raw/intermediate metadata. It did
not regenerate or modify them. The audit establishes that the current
pipeline and products are not publication-ready. The Frey table transcription
is correct, but most stored E0 values are hard fallbacks or saturation rather
than interior ratio retrievals. See
`vault/02_Algorithm/Audit - 2026-07-29.md`.

## Implemented definition

For each latitude slice:

1. evaluate Zhang–Paxton E0 (keV) and Q (mW m-2) in cell centres;
2. find the principal Q maximum;
3. retain the contiguous cells around that maximum where Q exceeds the chosen
   threshold;
4. calculate the area-weighted mean and median E0 over those cells using exact
   spherical weights `sin(latitude_upper) - sin(latitude_lower)`.

The reusable API supports broadcastable Kp/MLT inputs. It returns the
area-weighted mean and median, weighted spread, Q-weighted sensitivity mean,
selected latitude bounds and area, threshold cutoff, peak diagnostics, empty
masks, possible sampling-limit contact, and physical-pole contact.

The provisional default is `Q > 0.05 mW m-2`, based on the Zhang–Paxton Figure
8 mean-energy criterion. The implementation and figures also compare the
absolute `Q > 0.25 mW m-2` boundary criterion. **Confirmed by the user:** the
relative 10%-of-peak threshold is removed, the area mean remains primary, and
the area median is calculated and displayed alongside it.

The collapse evaluates the exact MLT supplied by the caller and does not
average within an MLT bin. Diagnostic maps use 0.05-hour sampling. The
published model is continuous in MLT through its Fourier representation,
although it was fitted from empirical 0.5-hour sectors.

MLAT cells now use 0.01-degree spacing over 50–90 degrees. This fine grid
numerically resolves the continuous Epstein profiles and prevents the hard
Q-threshold boundary from producing the visible jitter present at
0.25-degree spacing. It does not imply 0.01-degree empirical accuracy.

## Verification

- `tConductance`: 7 focused collapse tests passed, including regressions that
  verify the 0.05-hour diagnostic grid and area-weighted median and one that
  fixes the pole at map centre and 50 degrees at the outer radius.
- Two additional `BinnedImage` regression tests pass for retained SZA, DZA,
  LOS factor, correction state, and corrected-versus-uncorrected brightness.
- A ConductanceImage NetCDF round-trip test confirms that all nine
  sensor-specific geometry arrays, units, correction modes, and LOS-applied
  flags survive serialization.
- AST: all 29 Python files under `icbuilder/`, `scripts/`, and `tests/` parsed.
- Headless generation produced four PNG and four PDF files under `figures/`.
- Every PNG was visually inspected; the final polar map is complete, uses
  noon-up/dawn-right orientation, and contains no seam. The dE0 result is
  smooth and uses the same Kp/MLT and threshold-sensitivity layout as the mean.
- `git diff --check` passed.
- No SVG files were generated.

For Kp integers 0–9 and 480 MLT samples at 0.05-hour spacing (4,800 slices),
using 0.01-degree MLAT cells over 50–90 degrees and the default threshold:

- 0 selections were empty;
- 1,001 reached the 50-degree equatorward sampling limit, indicating possible
  domain truncation;
- 0 reached a non-polar upper sampling limit;
- 1,378 reached the physical 90-degree pole, which is not sampling truncation.

Against a 0.01-hour reference grid, interpolation of the 0.05-hour area-mean
product had a 0.021 keV 99th-percentile error and a 0.037 keV maximum error.
For the four Kp=2/Kp=5 threshold curves in the result figure, refining MLAT
from 0.25 to 0.01 degrees reduced RMS second differences from
0.016–0.023 keV to 0.00095–0.00124 keV and removed the visible jitter.

## Scientific integration boundary

Preserve these issues for the later integration decision:

- `ConductanceImage.Ep` is proton characteristic energy, despite stale
  electron wording in parts of its documentation.
- Electron E0 is currently inferred from the corrected WIC/SI13 ratio, with
  hard-coded low-signal fallbacks.
- **Confirmed by the user:** DMSP tests did not reproduce the Frey et al.
  (2003) WIC/SI13-to-E0 relationship. Replace IMAGE-derived E0 and dE0
  entirely with Zhang–Paxton rather than using a fallback or blend.
- Zhang–Paxton electron mean energy versus icBuilder electron
  characteristic-energy compatibility requires scientific confirmation.
- **Confirmed by the user:** use the 0.01-degree collapse to generate a lookup
  once for each supported Kp state on the fixed 36-by-36 IMAGE Cubed-Sphere
  grid. Production frames should index this table rather than recompute
  latitude profiles.
- Since the collapsed model is `E0(Kp, MLT)`, grid-cell MLAT does not enter the
  table value; it would matter only for a future return to the uncollapsed
  Zhang–Paxton model.
- **Confirmed by the user:** do not add a new precipitation-support mask; no
  robust automatic construction is available. Corrected WIC brightness
  supplies the spatial amplitude through Fe, while existing IMAGE coverage
  and background handling remain in place.
- Do not choose a single empirical Kp bin. The published method and current
  package interpolate E0 linearly in Kp and Q in hemispheric-power space
  between fitted-bin centres before the latitude collapse.
- Zhang–Paxton publishes no coefficient covariance or predictive-error model.
  **Confirmed by the user:** do not introduce DMSP calibration. Use the
  spherical-area-weighted E0 spread within the same selected latitude profile
  as dE0. This is uncertainty from discarding MLAT in the collapse, not formal
  Zhang–Paxton predictive or coefficient uncertainty.
- The existing IMAGE code propagates dE0 and dFe through conductance but sets
  E0–Fe covariance to zero as a placeholder. Since Fe is calculated from E0,
  preserve the uncertainty chain and calculate or sample that covariance
  during integration.
- Removing the WIC/SI13 ratio from E0 inference may remove SI13 as a required
  electron-energy input. **Proposed, not decided:** retain simultaneous SI13
  as an optional validation channel. At fixed Zhang–Paxton E0, compare
  WIC-derived and SI13-derived Fe, or predict corrected SI13 counts from
  WIC-derived Fe. Do not initially require SI13 for product generation or
  combine it into Fe.

## Audit findings that constrain integration

- Orbit 0085 has a median uncorrected binned WIC/SI13 ratio near 208 for SI13
  above three counts, already beyond the table maximum near 136.5. Fixed
  proton correction raises the comparable median to about 244.
- Across five tracked conductance products, only about 1.7–11.2% of valid
  pixels are interior-table E0 retrievals. Most are exact 0.2-, 1-, or
  25-keV fallback/saturation values.
- The former coarse-SI-grid defect is fixed in the current uncommitted work.
  Numerical inspection confirms exact 18-by-18 nesting: every SI boundary is
  every second 36-by-36 WIC edge, and corresponding SI centres are the
  pairwise WIC-centre midpoints.
- Even after that bug is fixed, 450-km SI13 is upsampled to 225-km WIC without
  matching effective spatial resolution.
- Fixed-2-keV proton correction, independent clipping, a three-count SI13
  threshold, and hard fallbacks amplify the failure. They are not sufficient
  explanations for the already-high raw/uncorrected ratio.
- The user is investigating the SI-to-WIC mapping. Published IMAGE-FUV work
  did map SI into WIC image space, so interpolation itself is not an error.
  The confirmed nested-grid construction defect appears modest in aggregate
  and is not the leading E0 explanation.
- The upstream Laundal/Østgaard FUVIEW3 product uses corrected counts,
  including detector flat-field and mission-time/temperature correction.
  The generated products still do not retain enough calibration,
  detector-state, source, unit, processing, sample-count, or retrieval-status
  metadata to prove the exact recipe or compatibility with the Frey response.
- A fixed standard-atmosphere/nadir response is applied to frames admitted out
  to 75 degrees detector zenith angle. Published work identifies atmospheric
  oxygen variability as a direct cause of high WIC/SI13 and unreasonable
  inferred energy.
- Follow-up inspection confirms that fuvpy uses DZA in its dayglow model but
  does not normalize the remaining auroral counts to a nadir-equivalent
  response. In an exploratory same-225-km-grid orbit-0085 test, the median
  uncorrected strong-pixel ratio increased from about 56 for DZA below 30
  degrees to 93, 130, and 156 for cutoffs of 45, 60, and 75 degrees. Changing
  geographic coverage prevents a causal interpretation, but viewing angle is
  now a concrete diagnostic rather than only a literature concern.
- The uncertainty chain is not publication-grade: saturation can return
  dE0=0, corrected channels share omitted uncertainty, E0–Fe covariance is
  forced to zero, and binned-median uncertainty is not estimated coherently.
  The user asked that uncertainty redesign be recorded and deferred.
- `SplineImage.solverP` passes Hall uncertainty into the Pedersen solver, a
  confirmed secondary-product bug. The user asked that it be retained for
  later correction.
- With Zhang–Paxton replacing ratio-derived E0, the weak-SI13 fallback problem
  no longer affects the primary E0 product. Revisit it only if SI13 remains a
  validation channel.
- Østgaard et al. (2018) used solar- and satellite-zenith-angle-dependent
  dayglow subtraction but deliberately avoided absolute auroral intensity
  comparisons; it warns about oblique views and supplies no auroral LOS
  normalization. The correct quantitative treatment is an angle-dependent
  atmosphere/instrument forward response. Multiplying each
  dayglow-subtracted pixel by `cos(DZA)` is only the plane-parallel,
  optically-thin first-order diagnostic and must be validated over a chosen
  moderate-DZA range. LOS remains relevant after Zhang–Paxton because WIC
  intensity still determines Fe.
- The user has added an uncommitted SZA/DZA and cosine-correction path.
  Inspection confirms that all three example input channels contain matching
  degree-valued geometry arrays and that the operation correctly multiplies
  the background-subtracted pixels by `cos(DZA)` before binning.
  `BinnedImage` now stores median SZA, median DZA, median `cos(DZA)`, and the
  correction flag, and carries them through interpolation and discard.
  `PreImage.discard` now also keeps raw `img` aligned. `ConductanceImage`
  copies and serializes the geometry separately for WIC, SI12, and SI13 and
  records each channel's correction mode and LOS-applied flag. It is not yet
  production-ready: the correction is on by default and explicitly applied
  to WIC, SI12, and SI13, while SI12 needs channel-specific viewing-response
  consideration. A one-frame orbit-0085 WIC test gave
  corrected/uncorrected binned ratios of 0.296, 0.623, and 0.932 at the 5th,
  50th, and 95th percentiles.

## Next action

Before pipeline integration, document the exact Laundal/FUVIEW3 corrected
counts recipe and ask whether it includes any auroral slant-to-nadir
normalization beyond detector/temporal corrections and dayglow modeling.
Then reproduce one documented strong-signal, near-nadir WIC/SI13 event on
common support and matched effective resolution. Compare uncorrected and
pixel-level cosine-corrected WIC-derived Fe over controlled DZA strata. Make
LOS treatment an explicit channel-specific option that defaults off, preserve
binned geometry and the selected treatment in output provenance, and do not
apply the WIC diagnostic automatically to SI12. Fix `PreImage.discard` to
subset raw `img`. Do not adopt the approximation without validation. Add
retrieval-status and processing-provenance variables in the isolated test
path.

In parallel at the decision level, review the two absolute Zhang–Paxton
thresholds and boundary-contact diagnostics. The representative statistic,
lookup architecture, and replacement role remain settled: keep the area mean
as primary, report the area median, precompute per supported Kp state, and
replace IMAGE-derived E0 and dE0. Integration must also repair uncertainty and
covariance, confirm energy-definition compatibility, and postpone SI13
validation until SI13 calibration and resolution issues are resolved.

## Portfolio impact

- Central update needed: No
- Changes: The central card already reflects the audit and changed next
  action. The verified grid fix and clarified upstream calibration/LOS
  boundary refine the technical investigation without changing portfolio
  status, priority, deadline, or publication significance.
- Sync summary: the SI grid now nests exactly. The source product uses
  FUVIEW3 corrected counts, while the 2018 paper's zenith-angle treatment is
  dayglow modeling rather than nadir-equivalent auroral correction. The paper
  avoids absolute intensity comparison, so icBuilder still needs a validated
  quantitative LOS treatment for WIC-derived Fe.

## Entry points

- `scripts/ZhangPaxton2008_collapse.py`
- `tests/test_zhang_paxton_collapse.py`
- `figures/README.md`
- `figures/zhang_paxton_collapse_process_map.png`
- `figures/zhang_paxton_collapse_latitude_slice.png`
- `figures/zhang_paxton_collapse_result.png`
- `figures/zhang_paxton_collapse_dE0_result.png`
- `vault/02_Algorithm/Processing Pipeline.md`
- `vault/02_Algorithm/Audit - 2026-07-29.md`
