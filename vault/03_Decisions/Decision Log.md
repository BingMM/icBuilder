# Decision Log

Last reviewed: 2026-07-29

## 2026-07-29 — Replace IMAGE-derived electron E0 and dE0

**Decision:** Replace the WIC/SI13-ratio-derived electron E0 and dE0 entirely
with a Zhang–Paxton-based estimate. Do not retain the IMAGE estimate as a
fallback, blend component, or calibration target.

**Rationale:** The purpose of introducing Zhang–Paxton is not merely to fill
low-signal gaps. The user reports that comparison with DMSP did not reproduce
the empirical relationship between the WIC/SI13 ratio and E0 proposed by Frey
et al. (2003). Without validation of that relationship, the current IMAGE E0
and its propagated dE0 do not have an adequate basis.

This decision sets the scientific direction but does not authorize production
integration yet. The replacement still requires:

- confirmation that Zhang–Paxton mean energy is compatible with the energy
  quantity used by the WIC response and Robinson relations;
- use of the spherical-area-weighted E0 spread within the selected
  Zhang–Paxton latitude profile as dE0, explicitly interpreted as uncertainty
  from collapsing away MLAT rather than model-fit error;
- nonzero E0–Fe covariance or ensemble propagation, because Fe is calculated
  using E0; and
- a decision on whether SI13 remains for quality control, validity checks, or
  validation after it leaves the E0 inference.

**Clarification later on 2026-07-29:** Do not add a new precipitation-support
mask; the user does not consider one robustly automatable. Corrected WIC
brightness supplies the spatial amplitude. Do not introduce DMSP into the new
estimator or uncertainty calculation. The DMSP result is the reason to reject
the Frey relation, not an input to its replacement.

## 2026-07-29 — Precompute the collapse on the fixed IMAGE grid

**Decision:** If the Zhang–Paxton collapse is integrated, calculate it once
for each supported Kp state on the fixed 36-by-36 Cubed-Sphere grid and use a
lookup during IMAGE processing.

**Rationale:** Every IMAGE frame uses the same grid-cell MLT and MLAT
coordinates. Repeating the expensive 0.01-degree latitude calculation per
frame would reproduce identical results. A small Kp-indexed table turns the
production calculation into array indexing while preserving the fine-grid
offline result.

**Superseded in part on 2026-07-29:** The lookup architecture did not initially
settle the integration role. The user subsequently chose full replacement of
IMAGE-derived E0 and dE0. Compatibility with the downstream energy definition
remains unresolved.

## 2026-07-29 — Use absolute oval thresholds and report area mean and median

**Decision:** Implement a provisional `ZP(Kp, MLT, MLAT)` collapse by selecting
the contiguous Q-above-threshold interval around the principal Q maximum and
computing the exact spherical-area-weighted mean electron energy inside it.
Retain that mean as the primary representative and report the area-weighted
median alongside it.

**Rationale:** Averaging the full latitude slice would let the extensive
near-zero region outside the auroral oval dominate. Exact latitude-cell areas
also avoid overrepresenting equal-width cells near the pole. The median gives
a robust typical-area diagnostic without replacing the spatial mean.

The default `Q > 0.05 mW m-2` follows the Zhang–Paxton Figure 8 mean-energy
criterion. `Q > 0.25 mW m-2` remains an absolute sensitivity check. The user
rejected and removed the relative 10%-of-peak threshold because its meaning
changes with every slice. This decision defines the exploratory reduction
only; it does not approve conductance-pipeline integration.

Diagnostic MLT sampling is 0.05 hour. This is numerical sampling of a
continuous Fourier model, not a claim of 0.05-hour empirical resolution; the
published fit used 0.5-hour observational sectors.

Default MLAT sampling is 0.01 degree. This numerical oversampling of the
continuous Epstein profiles removes the threshold-selection jitter seen at
0.25-degree spacing; it is not a claim of 0.01-degree empirical resolution.

## 2026-07-26 — Treat the project as complete for now

**Decision:** Record `icBuilder` as completed for the current scope while
preserving an explicit need to revisit it later.

**Rationale:** The user directly confirmed both points. No revisit date or
trigger was supplied, so none is inferred.

## 2026-07-26 — Preserve and modernize the established vault

**Decision:** Keep `log/icBuilder/` as the project vault, add the shared live
note architecture within it, and retain the older dated notes and attachments
as historical evidence.

**Rationale:** The repository already used this Obsidian-backed location. A
second renamed vault would split project memory and break continuity.

The architecture migration is documentation-only and does not imply
scientific, dataset, or implementation progress.

**Superseded 2026-07-26:** The user subsequently standardized all project
memory at the repository-root `vault/` path. The complete existing vault was
moved intact; only its location and operational references changed.
