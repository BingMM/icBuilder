# Zhang–Paxton latitude-collapse figures

Last reviewed: 2026-07-29

These figures document the exploratory reduction implemented in
`scripts/ZhangPaxton2008_collapse.py`. They use only the Zhang–Paxton model and
do not read or alter IMAGE data.

The reduction evaluates electron mean energy, `E0` (keV), and energy flux, `Q`
(mW m-2), in 0.01-degree cells from 50 to 90 degrees northern magnetic
latitude. At each `(Kp, MLT)` pair it:

1. finds the principal Q maximum;
2. keeps the contiguous interval around that maximum for which Q exceeds the
   chosen threshold;
3. calculates the area-weighted mean and area-weighted median E0 over that
   interval using exact latitude-cell weights
   `sin(latitude_upper) - sin(latitude_lower)`;
4. calculates the area-weighted standard deviation over the same interval as
   the profile-derived dE0.

The default threshold is `Q > 0.05 mW m-2`, following the inclusion criterion
used for the global mean energy in Zhang and Paxton (2008), Figure 8. The
figures also show `Q > 0.25 mW m-2`, used as an auroral-boundary contour in the
paper. Relative thresholds are intentionally excluded because they would
change the oval definition with the peak strength of each slice.

The Zhang--Paxton equations use a Fourier series and can be evaluated at any
MLT; the collapse function therefore evaluates the exact MLT supplied by the
caller and does not average within an MLT bin. The diagnostic maps sample this
continuous function every 0.05 MLT hour (3 minutes). The empirical model was
fitted from 48 sectors of width 0.5 MLT hour, so the denser diagnostic grid
smoothly resolves the fitted function but does not add independent
observational information.

The model's Epstein latitude profiles are also continuous. The 0.01-degree
MLAT grid is numerical oversampling used to prevent the selected threshold
boundary from moving in visible 0.25-degree steps; it does not imply
kilometre-scale empirical accuracy.

The latitude-slice figure includes an area-times-Q-weighted diagnostic,
`sum(area * Q * E0) / sum(area * Q)`. It emphasizes locations carrying the
largest modeled energy flux. It is not the primary spatial mean and is not a
particle-number-weighted mean energy.

Outputs:

- `zhang_paxton_collapse_process_map`: model fields and the selected interval;
- `zhang_paxton_collapse_latitude_slice`: one annotated Kp=5, MLT=00 slice;
- `zhang_paxton_collapse_result`: the collapsed Kp/MLT map and threshold
  sensitivity for representative E0;
- `zhang_paxton_collapse_dE0_result`: the corresponding profile-derived dE0
  map and threshold sensitivity.

The dE0 figure uses the `weighted_spread` returned by the collapse:
`sqrt(sum(w * (E0 - E0_mean)^2) / sum(w))`, with the same spherical-area
weights and selected latitude interval as the mean. It represents the
latitude variability discarded by reducing the profile to `(Kp, MLT)`. It is
not divided by the number of 0.01-degree samples and is not a formal
uncertainty estimate for the Zhang--Paxton fit coefficients.

Each output is saved as PNG for convenient viewing and PDF as the
publication-oriented high-resolution format. SVG is intentionally not
generated.

Regenerate from the repository root in the `tConductance` environment:

```bash
python scripts/ZhangPaxton2008_collapse.py
```

The script reports empty selections, contact with the 50-degree equatorward
sampling limit (possible domain truncation), and extension to the physical
90-degree pole. Reaching the physical pole is not itself sampling truncation.

This collapse is not yet connected to `ConductanceImage`. In particular,
`ConductanceImage.Ep` is proton characteristic energy; it must not receive the
electron mean energy calculated here without a separate scientific decision.
