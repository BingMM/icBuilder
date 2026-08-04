# Handoff - Latest

Last updated: 2026-07-31
Repository snapshot: `main` at `d854016`
Worktree state: uncommitted Zhang–Paxton collapse and fixed-grid lookup,
definitive Kp integration, tests, figures, memory, SI-grid fix, and regenerated
example products

## Project state

`icBuilder` now contains an importable Zhang–Paxton latitude collapse and a
complete fixed-grid lookup. The 8.4-MB bundled NetCDF has dimensions
`(901 Kp, 36 eta, 36 xi)` and stores direct layers from Kp 0.00 through 9.00
at 0.01 spacing. The loader rounds Kp to the nearest hundredth and directly
indexes E0, profile-derived dE0, and the area-weighted median without runtime
interpolation. It directly compares the stored two-dimensional coordinates
with the active Cubed-Sphere grid.

The lookup implementation has been reduced from production-style
infrastructure to transparent research code. The loader is a 79-line module
with one public function returning a dictionary. The 150-line generation
script shows the serial Kp loop, optional `process_map`, collapse, and NetCDF
writing in execution order. The table stores only the three scientific
fields, coordinates, units, collapse settings, and ZhangPaxton2008 package
version.

The latitude-collapse implementation is now separated in the same style. The
reusable module is 306 lines of direct NumPy calculation (down from a
1,065-line calculation/plotting/CLI module), returns ordinary dictionaries,
and contains no dataclasses or array type machinery. The 520-line diagnostic
script retains the necessarily verbose construction of four explanatory
figures and the CLI. This is deliberate plotting content rather than
scientific-framework complexity.

The orbit pipeline now loads the bundled definitive GFZ Kp series once before
multiprocessing and matches each final retained frame to the enclosing
half-open three-hour interval. `ConductanceImage` requests all lookup layers
once per orbit, checks their shape and grid, verifies that every frame remains
inside its serialized interval, and supplies E0/dE0 to the count conversion.
Missing, gapped, or out-of-range Kp fails explicitly. The local GFZ response
must also match its documented SHA-256 before it is parsed.

The paired E0/dE0 override bypasses the WIC/SI13 energy inversion while keeping
SI12 proton correction, WIC-derived Fe/dFe, and R/dR diagnostics. The induced
first-order E0--Fe covariance is passed to both Robinson uncertainty functions
and stored. Original GFZ thirds, rounded lookup Kp, interval starts, source
metadata, lookup/collapse provenance, and dE0 interpretation are serialized.
At `Fe=0`, dP and dH are the one-sided conductance excursions to `Fe=dFe`;
this replaces an invalid minimum-uncertainty calculation that failed at and
above 4 keV.

Stage 1 intentionally keeps the existing three-camera frame/support population
and combined weight. SI13 changes do not affect E0 or Fe under the override,
but making SI13 optional remains a separate stage.

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

- The conductance-orbit CLI now accepts independent WIC, SI12, SI13, and
  output folder names beneath `--base`, while retaining the original defaults.
  AST parsing, `--help`, and direct assertions for Chapman's `=wic`, `=s12`,
  `=s13`, and `=conductance` paths pass without running the data pipeline.
- The lookup contains 901 direct Kp layers and 1,167,696 grid values per
  field. It has no empty selections. E0 spans 0.211--4.383 keV and dE0 spans
  0.212--2.233 keV.
- The simplified NetCDF was populated from the previous verified table rather
  than recomputing all 901 layers. The Kp, xi, eta, MLT, E0, dE0, and median
  E0 arrays are exactly equal before and after the schema rewrite.
- One 36-by-36 Kp layer took 10.67 seconds and 259 MB RSS. A full run was
  estimated at 2.7 hours serial or roughly 40--60 minutes with four workers,
  which did not justify recomputing scientifically unchanged values.
- Fresh direct collapses checked nine cells spanning Kp 0.00, 1.52, and 9.00.
  The maximum difference across E0, dE0, and median E0 was `2.4e-7 keV`, as
  expected from float32 table storage.
- All 36 focused tests pass. They cover canonical grid shape/nesting,
  two-dimensional MLT, nearest-hundredth Kp quantization, lookup shape,
  scalar/vector access, direct-collapse agreement, definitive-Kp integrity and
  boundary matching and checksum, paired E0 override, SI13 invariance, induced
  covariance, zero-flux Robinson uncertainty, binned geometry, and NetCDF
  provenance.
- The refactored collapse is numerically identical to the previous
  implementation for synthetic disconnected and empty intervals, strict
  threshold boundaries, non-finite inputs, equatorward/non-polar/polar domain
  flags, scalar inputs, broad broadcast Kp/MLT grids, and both absolute
  thresholds. The comparison used exact tolerances (`rtol=0`, `atol=0`) with
  aligned NaNs.
- All four collapse diagnostics regenerate as valid one-page PNG/PDF pairs
  and were visually inspected after the refactor. Their 4,800-slice summary
  is unchanged: 0 empty, 1,001 equatorward-limit contacts, 0 non-polar upper
  contacts, and 1,378 physical-pole contacts.
- `icbuilder/data/zhang_paxton_e0_lookup.nc` remains byte-for-byte unchanged
  (SHA-256 `53c06e2c2ebb5ac8c605185aab7bba65575630be5339bb0d7e1c789262d07e65`);
  no lookup regeneration is required.
- A serial scratch run of orbit 0085 completed in 21 seconds without touching
  tracked examples. The new and old products both contain 20 frames. Original
  Kp 2.667/4.333 map to lookup Kp 2.67/4.33; all 18,958 finite E0 pixels
  exactly match the selected lookup, and their covariance values are finite.
- `figures/zhang_paxton_lookup_kp1_52.png` and PDF show the MLT coordinate,
  E0, and dE0 in native `(xi, eta)` coordinates and were visually inspected.
- The table uses `Q > 0.05 mW m-2` over 50--90 degrees with 0.01-degree
  numerical MLAT cells. 231,123 table cells touch the 50-degree equatorward
  sampling limit; changing this provisional domain requires regeneration.

- All Python files under `icbuilder/`, `scripts/`, and `tests/` parse, and
  `git diff --check` passes.
- The earlier four collapse figure sets remain valid; their polar orientation,
  threshold curves, and dE0 companion were visually checked previously.

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

## Remaining scientific boundary

Preserve these issues while testing the implemented stage-1 path:

- `ConductanceImage.Ep` is proton characteristic energy, despite stale
  electron wording in parts of its documentation.
- **Confirmed by the user:** DMSP tests did not reproduce the Frey et al.
  (2003) WIC/SI13-to-E0 relationship. Stage 1 now replaces IMAGE-derived E0
  and dE0 entirely with Zhang–Paxton rather than using a fallback or blend.
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
- The stage-1 path now propagates the first-order E0–Fe covariance. This does
  not resolve shared-channel covariance or missing model-prediction
  uncertainty.
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

Run one representative orbit to a scratch output and compare the new E0, dE0,
Fe, covariance, and conductance fields with the old product without changing
frame support. Inspect high/low Kp and MLT behavior and confirm that finite
SI13 changes cannot alter E0 or Fe.

Before treating the result as publication-ready, resolve whether
Zhang--Paxton electron mean energy is compatible with the energy quantity
assumed by the WIC response and Robinson relations. In parallel, continue the
channel-specific LOS investigation because WIC brightness still determines
Fe.

Only after the controlled comparison should stage 2 decide whether SI13
becomes optional and which validation diagnostic, if any, is scientifically
defensible. Do not combine that support change with the initial E0 comparison.

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
- `icbuilder/zhang_paxton_collapse.py`
- `icbuilder/zhang_paxton_lookup.py`
- `icbuilder/kp.py`
- `icbuilder/grids.py`
- `icbuilder/conductanceimage.py`
- `icbuilder/imagesat_e0_eflux_estimates.py`
- `scripts/download_gfz_kp.py`
- `scripts/make_conductance_orbit_files.py`
- `scripts/make_zhang_paxton_lookup.py`
- `scripts/plot_zhang_paxton_lookup.py`
- `icbuilder/data/zhang_paxton_e0_lookup.nc`
- `icbuilder/data/gfz_kp_2000_2001.json`
- `icbuilder/data/README.md`
- `tests/test_kp.py`
- `tests/test_e0_override.py`
- `tests/test_conductanceimage_zhang_paxton.py`
- `tests/test_zhang_paxton_lookup.py`
- `tests/test_grids.py`
- `tests/test_zhang_paxton_collapse.py`
- `figures/README.md`
- `figures/zhang_paxton_collapse_process_map.png`
- `figures/zhang_paxton_collapse_latitude_slice.png`
- `figures/zhang_paxton_collapse_result.png`
- `figures/zhang_paxton_collapse_dE0_result.png`
- `figures/zhang_paxton_lookup_kp1_52.png`
- `vault/02_Algorithm/Processing Pipeline.md`
- `vault/02_Algorithm/Audit - 2026-07-29.md`
